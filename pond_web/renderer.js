import init, {
    WasmWorld,
    state_header_len,
    state_agent_stride,
    state_tile_stride,
    trait_bounds,
} from '../pond_core/pkg/pond_core.js';
import { decodeAgent } from './decode.js';
import { createChain, updateChain } from './chain.js';
import { deriveMorphology } from './morphology.js';
import { drawBody } from './body.js';
import { rgbToOklch, oklchToRgb, smoothOklch } from './color.js';
import { initLegend, initGenomePanel } from './panels.js';
import { initInspector } from './inspector.js';

// ── Sim config ────────────────────────────────────────────────────────────────
const GRID = 12;
const POPULATION = 100;
const SEED = 42n;

// ── Bioluminescent palette per genome cluster (6 clusters) ───────────────────
const CLUSTER_RGB = [
    [0,   255, 200],  // teal
    [100,  60, 255],  // violet
    [255,  60, 140],  // magenta
    [40,  210, 255],  // cyan
    [255, 185,   0],  // amber
    [80,  255,  60],  // lime
];

// Cluster palette in Oklch, precomputed once for the color smoother
const CLUSTER_OKLCH = CLUSTER_RGB.map(rgbToOklch);
const COLOR_TAU_MS = 600;   // color crossfade time constant (~1s visible fade)

// ── Runtime state ─────────────────────────────────────────────────────────────
let canvas, ctx, world;
let HEADER_LEN, AGENT_STRIDE, TILE_STRIDE;

let chains = new Map();   // agent id → kinematic chain (stable across swap_remove reshuffles)
let morph_cache = new Map(); // agent id → derived MorphSpec (traits are immutable per life)
let color_state = new Map(); // agent id → current displayed color [L,C,h] (Oklch)
let prev_ts = 0;
let paused = false;
let speed_mult = 1;       // applied to delta_ms before world.update()
let frame_delta = 16.67;  // last frame's raw delta, for color smoothing

let stir_active = false;
let mouse_down = null;    // {x, y} canvas px at mousedown; null when button up
let mouse_world = { x: 0, y: 0 };
const DRAG_THRESHOLD_PX = 4;   // below = click (select), above = drag (stir)

// Selection / inspector
let selected_id = null;
let inspector;            // initInspector() handle
let insp_first = false;   // next inspector update should (re)fill trait rows
let last_agents = [];     // {id, x, y, cluster} decoded this frame, for click hit-test
let selected_pos = null;  // interpolated world pos of selected agent this frame

// Panels
let update_legend_counts, update_genome_panel;
let last_panel_step = -1;
let last_genome_step = -1;
const GENOME_SAMPLE_EVERY = 10;   // sim steps between average-genome samples

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
    await init();

    world = new WasmWorld(GRID, POPULATION, SEED);
    HEADER_LEN  = state_header_len();
    AGENT_STRIDE = state_agent_stride();
    TILE_STRIDE  = state_tile_stride();

    canvas = document.getElementById('c');
    ctx = canvas.getContext('2d');

    update_legend_counts = initLegend(CLUSTER_RGB);
    update_genome_panel = initGenomePanel(trait_bounds());
    inspector = initInspector();

    resize();
    window.addEventListener('resize', resize);

    canvas.addEventListener('mousedown',  on_mousedown);
    canvas.addEventListener('mousemove',  on_mousemove);
    canvas.addEventListener('mouseup',    on_mouseup);
    canvas.addEventListener('mouseleave', on_mouseup);
    canvas.addEventListener('dblclick',   on_dblclick);
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    window.addEventListener('keydown', on_key);

    requestAnimationFrame(frame);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
}

/** Convert canvas pixel → world coordinates (float). */
function screen_to_world(sx, sy) {
    const { tile_w, tile_h } = layout();
    return { x: sx / tile_w, y: sy / tile_h };
}

/** Return rendering layout params from current canvas size. The pond fills
 *  the whole window: tiles stretch per-axis (no letterbox margins), while
 *  scale_px (the smaller axis) sizes body geometry so agents don't distort. */
function layout() {
    const W = canvas.width, H = canvas.height;
    const tile_w = W / GRID;
    const tile_h = H / GRID;
    return { W, H, tile_w, tile_h, scale_px: Math.min(tile_w, tile_h) };
}

/** Toroidal lerp between prev and cur (world units, wraps at GRID). */
function lerp_wrap(prev, cur, a) {
    let d = cur - prev;
    if (d >  GRID * 0.5) d -= GRID;
    if (d < -GRID * 0.5) d += GRID;
    const v = prev + d * a;
    return ((v % GRID) + GRID) % GRID;
}

// ── Events ────────────────────────────────────────────────────────────────────

function on_mousedown(e) {
    if (e.button !== 0) return;
    // Don't stir yet — a short press-and-release is a select click; stirring
    // starts only once the cursor drags past DRAG_THRESHOLD_PX.
    mouse_down = { x: e.clientX, y: e.clientY };
    mouse_world = screen_to_world(e.clientX, e.clientY);
}

function on_mousemove(e) {
    mouse_world = screen_to_world(e.clientX, e.clientY);
    if (mouse_down && !stir_active) {
        const dx = e.clientX - mouse_down.x;
        const dy = e.clientY - mouse_down.y;
        if (dx * dx + dy * dy > DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) stir_active = true;
    }
    if (stir_active) {
        world.stir(mouse_world.x, mouse_world.y, 1.8, 0.45);
    }
}

function on_mouseup(e) {
    if (mouse_down && !stir_active && e && e.type === 'mouseup') {
        select_agent_at(screen_to_world(e.clientX, e.clientY));
    }
    mouse_down = null;
    stir_active = false;
}

/** Toroidal-aware nearest agent within pick radius; empty water deselects. */
function select_agent_at(w) {
    const PICK_RADIUS = 0.6;   // world units
    let best = null, best_d2 = PICK_RADIUS * PICK_RADIUS;
    for (const a of last_agents) {
        let dx = a.x - w.x, dy = a.y - w.y;
        if (dx >  GRID * 0.5) dx -= GRID;
        if (dx < -GRID * 0.5) dx += GRID;
        if (dy >  GRID * 0.5) dy -= GRID;
        if (dy < -GRID * 0.5) dy += GRID;
        const d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best = a; }
    }
    if (best) {
        selected_id = best.id;
        insp_first = true;
        const rgb = CLUSTER_RGB[best.cluster % CLUSTER_RGB.length];
        inspector.show(best.id, rgb);
        refresh_inspector();   // immediate feedback, even while paused
    } else {
        deselect();
    }
}

function deselect() {
    selected_id = null;
    selected_pos = null;
    inspector.hide();
}

/** Pull a fresh inspect snapshot for the selected agent; handles death. */
function refresh_inspector() {
    if (selected_id === null) return;
    const buf = world.inspect_agent(selected_id);
    if (buf.length === 0) {
        inspector.showDead();
        selected_pos = null;
        selected_id = null;
        return;
    }
    inspector.update(buf, insp_first);
    insp_first = false;
}

function on_dblclick(e) {
    const w = screen_to_world(e.clientX, e.clientY);
    world.pour_agents(w.x, w.y, 12);
}

function on_key(e) {
    if (e.key === ' ') {
        e.preventDefault();
        paused = !paused;
        document.getElementById('paused-banner').style.display = paused ? 'block' : 'none';
    }
    if (e.key === '+' || e.key === '=') speed_mult = Math.min(speed_mult * 2, 16);
    if (e.key === '-')                  speed_mult = Math.max(speed_mult / 2, 0.25);
    if (e.key === 'l' || e.key === 'L') {
        const panel = document.getElementById('side-right');
        panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
    }
    if (e.key === 'Escape') deselect();
    document.getElementById('h-speed').textContent = `speed ×${speed_mult}`;
}

// ── Main loop ─────────────────────────────────────────────────────────────────

function frame(ts) {
    const raw_delta = prev_ts ? ts - prev_ts : 16.67;
    prev_ts = ts;
    frame_delta = Math.min(raw_delta, 200);

    if (!paused) {
        // Cap delta to 200ms to avoid spiral of death after tab-switch
        const delta = frame_delta * speed_mult;
        world.update(delta);
    }

    const buf = world.get_state();
    render(buf, ts / 1000);
    update_panels(buf[2] | 0);

    requestAnimationFrame(frame);
}

/** Sim-step-gated panel refreshes — activations and means change per tick
 *  (20 Hz max), not per frame, so skip when the step hasn't advanced. */
function update_panels(step) {
    if (step === last_panel_step) return;
    last_panel_step = step;

    refresh_inspector();

    // Legend counts: tally decoded this frame in draw_agents
    const counts = new Array(CLUSTER_RGB.length).fill(0);
    for (const a of last_agents) counts[a.cluster % CLUSTER_RGB.length]++;
    update_legend_counts(counts);

    if (step - last_genome_step >= GENOME_SAMPLE_EVERY) {
        last_genome_step = step;
        update_genome_panel(world.trait_means());
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function render(buf, time_sec) {
    const L = layout();
    const n     = buf[0] | 0;   // agent count
    const step  = buf[2] | 0;
    const food  = buf[3] | 0;
    const avgE  = buf[4].toFixed(1);
    const alpha = buf[5];        // interpolation factor

    // Background
    ctx.fillStyle = '#040810';
    ctx.fillRect(0, 0, L.W, L.H);

    draw_tiles(buf, n, L);
    draw_agents(buf, n, alpha, L, time_sec);

    if (stir_active) draw_stir(L);

    // HUD
    document.getElementById('h-step').textContent   = `step   ${step}`;
    document.getElementById('h-agents').textContent = `agents ${n}`;
    document.getElementById('h-energy').textContent = `energy ${avgE}`;
    document.getElementById('h-food').textContent   = `food   ${food}`;
}

// ── Tile layer ────────────────────────────────────────────────────────────────

function draw_tiles(buf, n, { tile_w, tile_h }) {
    const tile_base = HEADER_LEN + n * AGENT_STRIDE;

    for (let ty = 0; ty < GRID; ty++) {
        for (let tx = 0; tx < GRID; tx++) {
            const ti  = ty * GRID + tx;
            const off = tile_base + ti * TILE_STRIDE;
            const food      = buf[off];       // 0–3
            const fertility = buf[off + 1];   // 0–1ish

            const sx = tx * tile_w;
            const sy = ty * tile_h;

            // Base tile: deep ocean tint, subtly brighter where fertile
            const b_base = Math.floor(fertility * 18);
            const g_base = Math.floor(fertility * 12);
            ctx.fillStyle = `rgb(${b_base},${g_base + 8},${b_base + 20})`;
            ctx.fillRect(sx, sy, tile_w + 0.5, tile_h + 0.5);

            // Food glow (bioluminescent green-cyan)
            if (food > 0) {
                const cx_s = sx + tile_w * 0.5;
                const cy_s = sy + tile_h * 0.5;
                const glowR = Math.min(tile_w, tile_h) * (0.35 + food * 0.18);
                const grd = ctx.createRadialGradient(cx_s, cy_s, 0, cx_s, cy_s, glowR);
                const a = 0.09 + food * 0.07;
                grd.addColorStop(0, `rgba(80,255,160,${a * 2})`);
                grd.addColorStop(0.5, `rgba(40,200,120,${a})`);
                grd.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = grd;
                ctx.fillRect(sx, sy, tile_w, tile_h);
            }
        }
    }
}

// ── Agent layer ───────────────────────────────────────────────────────────────

/** Drop chain/morph-spec/color cache entries for agents no longer alive. */
function reap_stale(chains_map, morph_map, color_map, live_ids) {
    for (const id of chains_map.keys()) {
        if (!live_ids.has(id)) {
            chains_map.delete(id);
            morph_map.delete(id);
            color_map.delete(id);
        }
    }
}

function draw_agents(buf, n, alpha, L, time_sec) {
    const { tile_w, tile_h, scale_px } = L;
    const live_ids = new Set();

    // Clip to the pond (= the whole window now); seam copies from wrapping
    // bodies would otherwise paint past the canvas edges.
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, GRID * tile_w, GRID * tile_h);
    ctx.clip();

    last_agents = [];
    selected_pos = null;

    for (let i = 0; i < n; i++) {
        const a = decodeAgent(buf, i, HEADER_LEN, AGENT_STRIDE);
        live_ids.add(a.id);

        // Traits (and thus morphology) are immutable for an agent's life —
        // derive the shape spec once per id and cache it, keyed by stable id
        // rather than array slot (slots get reshuffled by Rust's swap_remove).
        let spec = morph_cache.get(a.id);
        if (!spec) {
            spec = deriveMorphology(a.morph);
            morph_cache.set(a.id, spec);
        }

        let chain = chains.get(a.id);
        if (!chain) {
            chain = createChain(a.x, a.y, spec.segCount);
            chains.set(a.id, chain);
        }

        const hx_w = lerp_wrap(a.prevX, a.x, alpha);
        const hy_w = lerp_wrap(a.prevY, a.y, alpha);
        updateChain(chain, hx_w, hy_w, { segCount: spec.segCount, segDist: spec.segDist, gridSize: GRID });

        last_agents.push({ id: a.id, x: hx_w, y: hy_w, cluster: a.genomeCluster });
        if (a.id === selected_id) selected_pos = { x: hx_w, y: hy_w };

        // Smoothed color: crossfade toward the cluster color in Oklch so a
        // cluster reassignment fades over ~1s instead of flashing.
        const target = CLUSTER_OKLCH[a.genomeCluster % CLUSTER_OKLCH.length];
        let lch = color_state.get(a.id);
        if (!lch) {
            lch = [...target];   // newborns start at their cluster color
            color_state.set(a.id, lch);
        }
        smoothOklch(lch, target, frame_delta, COLOR_TAU_MS);
        const palette = oklchToRgb(lch);
        const base_r = scale_px * (0.07 + a.energyNorm * 0.05 + a.morph.pointiness * 0.04);

        drawBody(
            ctx, chain, spec, palette,
            { tile_w, tile_h, scale_px, off_x: 0, off_y: 0, gridSize: GRID },
            { baseR: base_r, energyNorm: a.energyNorm, velX: a.velX, velY: a.velY, timeSec: time_sec },
        );
    }

    if (selected_pos) draw_selection_ring(L, time_sec);

    ctx.restore();
    reap_stale(chains, morph_cache, color_state, live_ids);
}

/** Pulsing ring around the selected agent (drawn inside the grid clip). */
function draw_selection_ring({ tile_w, tile_h, scale_px }, time_sec) {
    const sx = selected_pos.x * tile_w;
    const sy = selected_pos.y * tile_h;
    const r = scale_px * (0.30 + 0.03 * Math.sin(time_sec * 4));
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.stroke();
}

// ── Stir indicator ────────────────────────────────────────────────────────────

function draw_stir({ tile_w, tile_h, scale_px }) {
    const sx = mouse_world.x * tile_w;
    const sy = mouse_world.y * tile_h;
    const r  = 1.8 * scale_px;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
    grd.addColorStop(0,   'rgba(200,240,255,0.08)');
    grd.addColorStop(0.5, 'rgba(80,160,255,0.04)');
    grd.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();

    // Ripple ring
    ctx.strokeStyle = 'rgba(100,200,255,0.20)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, r * 0.65, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}

// ── Launch ────────────────────────────────────────────────────────────────────
boot();
