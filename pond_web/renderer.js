import init, {
    WasmWorld,
    state_header_len,
    state_agent_stride,
    state_tile_stride,
} from '../pond_core/pkg/pond_core.js';

// ── Sim config ────────────────────────────────────────────────────────────────
const GRID = 12;
const POPULATION = 100;
const SEED = 42n;

// ── Kinematic chain config ────────────────────────────────────────────────────
const NUM_SEGS = 7;     // segments per agent (head is 0)
const SEG_DIST = 0.28;  // world units between consecutive segments

// Size envelope: head(0) → body peak → tail(6) relative to base_r
const SEG_ENV = [0.85, 1.05, 1.25, 1.15, 0.95, 0.65, 0.35];

// ── Bioluminescent palette per genome cluster (6 clusters) ───────────────────
const CLUSTER_RGB = [
    [0,   255, 200],  // teal
    [100,  60, 255],  // violet
    [255,  60, 140],  // magenta
    [40,  210, 255],  // cyan
    [255, 185,   0],  // amber
    [80,  255,  60],  // lime
];

// ── Runtime state ─────────────────────────────────────────────────────────────
let canvas, ctx, world;
let HEADER_LEN, AGENT_STRIDE, TILE_STRIDE;

let chains = [];          // kinematic chain state, one per agent slot
let prev_ts = 0;
let paused = false;
let speed_mult = 1;       // applied to delta_ms before world.update()

let stir_active = false;
let mouse_world = { x: 0, y: 0 };

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
    await init();

    world = new WasmWorld(GRID, POPULATION, SEED);
    HEADER_LEN  = state_header_len();
    AGENT_STRIDE = state_agent_stride();
    TILE_STRIDE  = state_tile_stride();

    canvas = document.getElementById('c');
    ctx = canvas.getContext('2d');

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
    const { tile_px, off_x, off_y } = layout();
    return {
        x: (sx - off_x) / tile_px,
        y: (sy - off_y) / tile_px,
    };
}

/** Return rendering layout params from current canvas size. */
function layout() {
    const W = canvas.width, H = canvas.height;
    const tile_px = Math.min(W, H) / GRID;
    const off_x = (W - tile_px * GRID) * 0.5;
    const off_y = (H - tile_px * GRID) * 0.5;
    return { W, H, tile_px, off_x, off_y };
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
    stir_active = true;
    mouse_world = screen_to_world(e.clientX, e.clientY);
}

function on_mousemove(e) {
    mouse_world = screen_to_world(e.clientX, e.clientY);
    if (stir_active) {
        world.stir(mouse_world.x, mouse_world.y, 1.8, 0.45);
    }
}

function on_mouseup() { stir_active = false; }

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
    document.getElementById('h-speed').textContent = `speed ×${speed_mult}`;
}

// ── Main loop ─────────────────────────────────────────────────────────────────

function frame(ts) {
    const raw_delta = prev_ts ? ts - prev_ts : 16.67;
    prev_ts = ts;

    if (!paused) {
        // Cap delta to 200ms to avoid spiral of death after tab-switch
        const delta = Math.min(raw_delta, 200) * speed_mult;
        world.update(delta);
    }

    const buf = world.get_state();
    render(buf);

    requestAnimationFrame(frame);
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function render(buf) {
    const { W, H, tile_px, off_x, off_y } = layout();
    const n     = buf[0] | 0;   // agent count
    const step  = buf[2] | 0;
    const food  = buf[3] | 0;
    const avgE  = buf[4].toFixed(1);
    const alpha = buf[5];        // interpolation factor

    // Background
    ctx.fillStyle = '#040810';
    ctx.fillRect(0, 0, W, H);

    draw_tiles(buf, n, tile_px, off_x, off_y);
    draw_agents(buf, n, alpha, tile_px, off_x, off_y);

    if (stir_active) draw_stir(tile_px, off_x, off_y);

    // HUD
    document.getElementById('h-step').textContent   = `step   ${step}`;
    document.getElementById('h-agents').textContent = `agents ${n}`;
    document.getElementById('h-energy').textContent = `energy ${avgE}`;
    document.getElementById('h-food').textContent   = `food   ${food}`;
}

// ── Tile layer ────────────────────────────────────────────────────────────────

function draw_tiles(buf, n, tile_px, off_x, off_y) {
    const tile_base = HEADER_LEN + n * AGENT_STRIDE;

    for (let ty = 0; ty < GRID; ty++) {
        for (let tx = 0; tx < GRID; tx++) {
            const ti  = ty * GRID + tx;
            const off = tile_base + ti * TILE_STRIDE;
            const food      = buf[off];       // 0–3
            const fertility = buf[off + 1];   // 0–1ish

            const sx = off_x + tx * tile_px;
            const sy = off_y + ty * tile_px;

            // Base tile: deep ocean tint, subtly brighter where fertile
            const b_base = Math.floor(fertility * 18);
            const g_base = Math.floor(fertility * 12);
            ctx.fillStyle = `rgb(${b_base},${g_base + 8},${b_base + 20})`;
            ctx.fillRect(sx, sy, tile_px + 0.5, tile_px + 0.5);

            // Food glow (bioluminescent green-cyan)
            if (food > 0) {
                const cx_s = sx + tile_px * 0.5;
                const cy_s = sy + tile_px * 0.5;
                const glowR = tile_px * (0.35 + food * 0.18);
                const grd = ctx.createRadialGradient(cx_s, cy_s, 0, cx_s, cy_s, glowR);
                const a = 0.09 + food * 0.07;
                grd.addColorStop(0, `rgba(80,255,160,${a * 2})`);
                grd.addColorStop(0.5, `rgba(40,200,120,${a})`);
                grd.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = grd;
                ctx.fillRect(sx, sy, tile_px, tile_px);
            }
        }
    }

    // Faint grid lines
    ctx.strokeStyle = 'rgba(40,60,80,0.35)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID; i++) {
        ctx.beginPath();
        ctx.moveTo(off_x + i * tile_px, off_y);
        ctx.lineTo(off_x + i * tile_px, off_y + GRID * tile_px);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(off_x, off_y + i * tile_px);
        ctx.lineTo(off_x + GRID * tile_px, off_y + i * tile_px);
        ctx.stroke();
    }
}

// ── Agent layer ───────────────────────────────────────────────────────────────

/** Ensure chains pool matches current agent count. New slots initialized at agent position. */
function sync_chains(n, buf) {
    while (chains.length < n) {
        const i   = chains.length;
        const off = HEADER_LEN + i * AGENT_STRIDE;
        const wx  = buf[off];
        const wy  = buf[off + 1];
        const segs = Array.from({ length: NUM_SEGS }, () => ({ x: wx, y: wy }));
        chains.push({ segs });
    }
    if (chains.length > n) chains.length = n;
}

function draw_agents(buf, n, alpha, tile_px, off_x, off_y) {
    sync_chains(n, buf);

    // Additive blending gives the bioluminescent glow for free
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (let i = 0; i < n; i++) {
        const off = HEADER_LEN + i * AGENT_STRIDE;

        // Lerp position between prev tick and current tick
        const hx_w = lerp_wrap(buf[off + 2], buf[off],     alpha);
        const hy_w = lerp_wrap(buf[off + 3], buf[off + 1], alpha);

        const energy_norm    = buf[off + 4];
        const vel_x          = buf[off + 5];
        const vel_y          = buf[off + 6];
        const genome_cluster = buf[off + 7] | 0;
        const aggression     = buf[off + 10];

        // Update kinematic chain
        const chain = chains[i];
        chain.segs[0].x = hx_w;
        chain.segs[0].y = hy_w;

        for (let s = 1; s < NUM_SEGS; s++) {
            const ps = chain.segs[s - 1];
            const cs = chain.segs[s];
            let dx = cs.x - ps.x;
            let dy = cs.y - ps.y;
            // Shortest path wrap
            if (dx >  GRID * 0.5) dx -= GRID;
            if (dx < -GRID * 0.5) dx += GRID;
            if (dy >  GRID * 0.5) dy -= GRID;
            if (dy < -GRID * 0.5) dy += GRID;
            const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001;
            if (dist > SEG_DIST) {
                const ratio = SEG_DIST / dist;
                cs.x = ps.x + dx * ratio;
                cs.y = ps.y + dy * ratio;
            }
        }

        // Cluster color
        const [cr, cg, cb] = CLUSTER_RGB[genome_cluster % CLUSTER_RGB.length];

        // Base radius scales with tile size, energy, and aggression
        const base_r = tile_px * (0.07 + energy_norm * 0.05 + aggression * 0.04);

        // Draw segments back-to-front (tail first) so head is on top
        for (let s = NUM_SEGS - 1; s >= 0; s--) {
            const seg = chain.segs[s];
            const sx  = off_x + seg.x * tile_px;
            const sy  = off_y + seg.y * tile_px;
            const r   = base_r * SEG_ENV[s];

            // Soft outer glow
            const glow_r = r * 2.8;
            const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, glow_r);
            const a_glow = (0.06 + energy_norm * 0.05) * (1.0 - s / NUM_SEGS * 0.4);
            grd.addColorStop(0,   `rgba(${cr},${cg},${cb},${a_glow})`);
            grd.addColorStop(1,   'rgba(0,0,0,0)');
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.arc(sx, sy, glow_r, 0, Math.PI * 2);
            ctx.fill();

            // Bright core
            const a_core = (0.55 + energy_norm * 0.35) * (1.0 - s * 0.07);
            ctx.fillStyle = `rgba(${cr},${cg},${cb},${a_core})`;
            ctx.beginPath();
            ctx.arc(sx, sy, r, 0, Math.PI * 2);
            ctx.fill();
        }

        // Eyes: two small dots offset perpendicular to velocity at head segment
        const vlen = Math.sqrt(vel_x * vel_x + vel_y * vel_y) || 1;
        const vnx = vel_x / vlen;
        const vny = vel_y / vlen;
        const head_sx = off_x + chain.segs[0].x * tile_px;
        const head_sy = off_y + chain.segs[0].y * tile_px;
        const eye_r   = base_r * SEG_ENV[0] * 0.28;
        const eye_fwd = base_r * SEG_ENV[0] * 0.55;
        const eye_lat = base_r * SEG_ENV[0] * 0.45;

        for (const side of [-1, 1]) {
            const ex = head_sx + vnx * eye_fwd + (-vny) * eye_lat * side;
            const ey = head_sy + vny * eye_fwd + ( vnx) * eye_lat * side;
            ctx.fillStyle = 'rgba(255,255,255,0.90)';
            ctx.beginPath();
            ctx.arc(ex, ey, eye_r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    ctx.restore();
}

// ── Stir indicator ────────────────────────────────────────────────────────────

function draw_stir(tile_px, off_x, off_y) {
    const sx = off_x + mouse_world.x * tile_px;
    const sy = off_y + mouse_world.y * tile_px;
    const r  = 1.8 * tile_px;

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
