import init, {
    species_stride,
    predator_state_stride,
    species_membership_radius,
    WasmWorld,
    state_header_len,
    state_agent_stride,
    state_tile_stride,
    state_death_stride,
    stats_sample_stride,
    trait_bounds,
    tunable_ranges,
    schema_version,
    brain_layer_sizes,
} from '../pond_core/pkg/pond_core.js';
import { decodeAgent } from './decode.js';
import { createChain, updateChain } from './chain.js';
import { deriveMorphology } from './morphology.js';
import { drawBody } from './body.js';
import { oklchToRgb, genomeColor } from './color.js';
import { initLegend, initGenomePanel } from './panels.js';
import { initArchetypes, archetypeColor, summarize } from './archetypes.js';
import {
    parseSpecies, initSpeciesPanel, initToast, centroidDistance,
} from './species.js';
import { initGraphs } from './graphs.js';
import { initSetup } from './setup.js';
import { initGod } from './god.js';
import { initInspector } from './inspector.js';
import { openPhylogeny, refreshPhylogeny } from './phylogeny.js';
import { closeFloatingPrefix } from './floating.js';

// Wire format this page was written against. The engine reports its own; a
// mismatch means pond_core/pkg and pond_web were built from different commits,
// and every flat buffer the page reads would be off by some number of floats —
// silently, producing plausible wrong numbers rather than an error. See
// pond_core/src/schema.rs.
const EXPECTED_SCHEMA = 2;

// ── Sim config ────────────────────────────────────────────────────────────────
// Set from the setup panel (`N`) and fixed for the life of a run — changing any
// of them means building a new World, which restart() does.
let GRID = 12;
let POPULATION = 100;
let SEED = 42n;

// Fallback family swatches. Bodies no longer use these — colour comes from the
// genome (see genomeColor) — but the legend needs a swatch before any member of
// a family has been seen, and these seed it.
// Fallback family swatches, sampled along the same passive → middle →
// aggressive palette the bodies use (see genomeColor).
const CLUSTER_RGB = [
    [0xF8, 0xFF, 0x00],  // lime — passive
    [0x69, 0xFF, 0xAA],
    [0x00, 0xF8, 0xFF],  // cyan — middle
    [0x56, 0x8F, 0xFF],
    [0xFF, 0x00, 0xF8],  // magenta — aggressive
    [0xCB, 0x00, 0xC5],
];


// ── Runtime state ─────────────────────────────────────────────────────────────
let canvas, ctx, world;
let HEADER_LEN, AGENT_STRIDE, TILE_STRIDE, DEATH_STRIDE;
// id → true while that predator is swimming off, rebuilt each frame from
// predators_state(). There can be several at once, so this is a map rather than
// a single id.
let predators = new Map();
// Kinematic chains for the predators, keyed by id like the agents' own. Three
// segments, drawn as triangles that pivot on their links.
let predator_chains = new Map();
const PREDATOR_SEGS = 3;
// World units between segments. A hunter's head covers 0.95 world units per sim
// tick — six times what a prey animal does — so at the old 0.42 the followers sat
// permanently pinned at the far end of their slack and were dragged in hard
// increments instead of trailing. Longer links than a frame's worth of travel
// give the body something to trail with.
const PREDATOR_SEG_DIST = 0.9;
// Per-frame easing on each segment's drawn angle. The link constraint is
// geometric and reacts instantly; a hunter that changes heading should look like
// it is swinging its body round, not like the triangles teleported.
const PREDATOR_ANGLE_EASE = 0.18;
// Must match RECTANGLE_HALF_WIDTH in world.rs — the top tier is drawn at the
// exact extent it kills at.
const RECT_HALF_WIDTH_WORLD = 0.85;

let chains = new Map();   // agent id → kinematic chain (stable across swap_remove reshuffles)
let morph_cache = new Map(); // agent id → derived MorphSpec (traits are immutable per life)
let color_state = new Map(); // agent id → current displayed color [L,C,h] (Oklch)
let dying = [];           // death-dissolve effects in flight (see reap_stale)
const DEATH_SEC = 0.85;   // death dissolve duration
let prev_ts = 0;
let paused = false;
// The world exists from boot (the panels and the first frame need something to
// read), but it does not advance until a run is actually started. Choosing
// parameters while the pond you're configuring runs behind the panel would mean
// the run you start is never the run you were looking at.
let sim_running = false;
let speed_mult = 1;       // applied to delta_ms before world.update()
let automatic_predators = true;
let frame_delta = 16.67;  // last frame's raw delta, for color smoothing

let stir_active = false;
let mouse_down = null;    // {x, y} canvas px at mousedown; null when button up
let mouse_world = { x: 0, y: 0 };
const DRAG_THRESHOLD_PX = 4;   // below = click (select), above = drag (stir)

// Camera: cam.x/cam.y are the world coordinates at the centre of the window,
// cam.zoom is a multiple of fit_scale(). Panning and zooming are view-only —
// nothing here reaches the sim.
const cam = { x: GRID / 2, y: GRID / 2, zoom: 1 };
const MIN_ZOOM = 1;       // whole pond visible, letterboxed
const MAX_ZOOM = 10;      // ~1 tile fills a quarter of the window
const ZOOM_STEP = 1.12;   // per wheel notch
const KEY_PAN_TILES = 0.6;
let pan_drag = null;      // {sx, sy, cam_x, cam_y} while right/middle-dragging
let cam_default = true;   // false once the user pans or zooms; see resize()

// Selection / inspector
let selected_id = null;
let inspector;            // initInspector() handle
let insp_first = false;   // next inspector update should (re)fill trait rows
let last_agents = [];     // {id, x, y, cluster} decoded this frame, for click hit-test
let selected_pos = null;  // interpolated world pos of selected agent this frame

// Panels
let update_legend_counts, update_genome_panel, update_graphs, setup, god;
// The rule dials, fixed for the life of a run and set from the setup panel.
// Held here so a restart can re-apply them to the new world.
let dials = null;
let graphs_visible = false;
let debug_visible = false;
let archetypes_visible = false;
// Overlay colour per agent id while the archetype view is open. Cleared on
// toggle-off so bodies revert to their genome palette (color.js) — this is an
// overlay, not a replacement for the trait-derived colour.
const arch_color = new Map();
let update_archetypes = null;
let arch_timer = null;
let update_species = null;
let announce = null;
let species_rows = [];
// Ids already announced, so a toast fires once per promotion rather than on
// every refresh that still sees the species.
const announced = new Set();
const extinct_announced = new Set();
// Set once the first roster has been seen, so restoring a run in progress does
// not toast every species that already exists.
let species_seeded = false;
let current_step = 0;
let last_species_tick = -1;
let TRAIT_BOUNDS = null;
let hint_visible = true;   // the bottom-left controls key; click it or the ? chip
let zen = false;           // C hides the whole UI (see toggle_zen)
let graphs_timer = null;
const GRAPH_REFRESH_MS = 1000;   // series only gain a sample every 10 sim steps
let last_panel_step = -1;
let last_genome_step = -1;
const GENOME_SAMPLE_EVERY = 10;   // sim steps between average-genome samples

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
    await init();

    const engine_schema = schema_version();
    if (engine_schema !== EXPECTED_SCHEMA) {
        // Loud, and before anything reads a buffer. A stale pkg/ against a fresh
        // page reads the right number of floats from the wrong places, which
        // looks like a simulation bug for as long as it takes to notice.
        const msg = `schema mismatch: engine reports ${engine_schema}, ` +
            `this page expects ${EXPECTED_SCHEMA}. Rebuild with ` +
            `\`wasm-pack build pond_core --target web --features wasm\`.`;
        document.body.innerHTML =
            `<pre style="color:#ff69a5;font:13px/1.6 'Courier New',monospace;padding:24px">` +
            `${msg}</pre>`;
        throw new Error(msg);
    }

    world = new WasmWorld(GRID, POPULATION, SEED);
    HEADER_LEN  = state_header_len();
    AGENT_STRIDE = state_agent_stride();
    TILE_STRIDE  = state_tile_stride();
    DEATH_STRIDE = state_death_stride();

    canvas = document.getElementById('c');
    ctx = canvas.getContext('2d');

    build_panels();
    setup = initSetup(document.getElementById('setup'), restart, {
        grid: GRID, population: POPULATION, seed: SEED,
    }, tunable_ranges());
    god = initGod(document.getElementById('god'), {
        smiteRadius: (x, y, r) => world.smite_radius(x, y, r),
        smiteBand: (x0, x1) => world.smite_band(x0, x1),
        smiteAll: () => world.smite_all(),
        setImmortal: on => world.set_immortal(on),
        summonOctagon: () => world.summon_octagon(),
        summonRectangle: () => world.summon_rectangle(),
        dismissHunters: () => world.dismiss_summoned_predators(),
        summonedHunterCount: () => world.summoned_predator_count(),
        gridSize: () => GRID,
        onChange: update_cursor,
    });
    inspector = initInspector(Array.from(brain_layer_sizes()));

    resize();
    layout_right_column();
    window.addEventListener('resize', () => { resize(); layout_right_column(); });

    canvas.addEventListener('mousedown',  on_mousedown);
    canvas.addEventListener('mousemove',  on_mousemove);
    canvas.addEventListener('mouseup',    on_mouseup);
    canvas.addEventListener('mouseleave', on_mouseup);
    canvas.addEventListener('dblclick',   on_dblclick);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('wheel', on_wheel, { passive: false });

    document.getElementById('h-newrun').addEventListener('click', open_setup);
    document.getElementById('h-predators').addEventListener('click', toggle_predators);
    document.getElementById('hint').addEventListener('click', toggle_hint_click);
    document.getElementById('hint-show').addEventListener('click', toggle_hint_click);

    window.addEventListener('keydown', on_key);

    requestAnimationFrame(frame);
}

/** (Re)build the panels that render per-run data.
 *
 *  They fill their containers on init, so a rebuild has to clear them first or
 *  a restart would stack a second copy of every row underneath the first. */
function build_panels() {
    for (const id of ['legend-colors', 'legend-shapes', 'legend-tiles',
                      'legend-deaths', 'legend-composite', 'genome-panel', 'graphs',
                      'species-list']) {
        document.getElementById(id).innerHTML = '';
    }
    TRAIT_BOUNDS = trait_bounds();
    update_legend_counts = initLegend(
        family_palette(),
        i => world.cluster_composite(i),
        TRAIT_BOUNDS,
    );
    update_genome_panel = initGenomePanel(TRAIT_BOUNDS);
    update_graphs = initGraphs(document.getElementById('graphs'));
    update_archetypes = initArchetypes(document.getElementById('archetypes'));
    update_species = initSpeciesPanel(
        document.getElementById('species-list'),
        s => species_swatch(s),
        TRAIT_BOUNDS,
    );
    announce = initToast(document.getElementById('species-toast'));
}

/** Tear down the current run and build a new World from the given parameters.
 *
 *  Everything keyed by agent id — chains, morphology, colour — has to go: ids
 *  restart from 0 in the new world, so a stale entry would hand a fresh agent
 *  the body of a dead one. */
function restart({ grid, population, seed, dials: chosen }) {
    GRID = grid;
    POPULATION = population;
    SEED = seed;
    if (chosen) dials = chosen;

    world = new WasmWorld(GRID, POPULATION, SEED);
    // Rules are applied once, at construction, and never touched again for the
    // life of this world — see setup.js for why they are start-of-run only.
    if (dials) {
        world.set_food_regen_scale(dials.regen);
        world.set_hunt_aggression_threshold(dials.hunt);
        world.set_cluster_k(Math.round(dials.k));
    }
    // A fresh world starts mortal, so an immortality toggle left on has to be
    // re-applied or the panel would claim a state the sim isn't in.
    if (god.isImmortal()) world.set_immortal(true);
    world.set_automatic_predators(automatic_predators);

    chains.clear();
    predator_chains.clear();
    morph_cache.clear();
    color_state.clear();
    death_cause.clear();
    dying = [];
    last_agents = [];
    deselect();
    last_panel_step = -1;
    last_genome_step = -1;
    species_rows = [];
    announced.clear();
    extinct_announced.clear();
    species_seeded = false;
    last_species_tick = -1;

    build_panels();
    if (graphs_visible) refresh_graphs();
    reset_camera();
    close_setup();
}

/** Open the setup panel and freeze the sim while it's up. */
function open_setup() {
    if (setup.isOpen()) return;
    sim_running = false;
    setup.show();
    document.getElementById('setup-banner').style.display = 'block';

    // Opening the panel ends the run there and then. There is no resume: the
    // pond is replaced by the idle scene, the panels that described it are
    // emptied, and Escape does not close this. The only way out is Start Run.
    //
    // Otherwise the previous run stays clickable behind the idle scene — its
    // agents selectable, its families and graphs still on screen — and the
    // "new run" screen is a lie about a sim that is merely paused.
    deselect();
    last_agents = [];
    clear_run_panels();
}

/** Empty every panel that describes a run. They are rebuilt by build_panels()
 *  when the next run starts. */
function clear_run_panels() {
    graphs_visible = false;
    if (graphs_timer) {
        clearInterval(graphs_timer);
        graphs_timer = null;
    }
    document.getElementById('graphs').style.display = 'none';
    closeFloatingPrefix('graph:');
    closeFloatingPrefix('species:');
    closeFloatingPrefix('tree:');
    for (const id of ['legend-colors', 'legend-shapes', 'legend-tiles',
                      'legend-deaths', 'legend-composite', 'genome-panel', 'graphs',
                      'species-list']) {
        document.getElementById(id).innerHTML = '';
    }
    update_legend_counts = null;
    update_genome_panel = null;
    update_graphs = null;
}

/** Close the setup panel and let the new world run. Only `restart` calls this —
 *  the panel is not dismissable by any other means. */
function close_setup() {
    setup.hide();
    document.getElementById('setup-banner').style.display = 'none';
    sim_running = true;
}

/** Crosshair while a god tool is armed, so it's obvious the next click smites. */
function update_cursor() {
    if (!canvas) return;
    canvas.style.cursor = god?.armedTool() ? 'crosshair' : 'default';
    layout_right_column();
}

function toggle_predators() {
    automatic_predators = !automatic_predators;
    world.set_automatic_predators(automatic_predators);
    const button = document.getElementById('h-predators');
    button.textContent = `predators: ${automatic_predators ? 'on' : 'off'}`;
    button.classList.toggle('off', !automatic_predators);
}

/** Stack the legend under the god panel. The god panel grows and shrinks as it
 *  is enabled, so the offset is measured rather than fixed. */
function layout_right_column() {
    const god_el = document.getElementById('god');
    const side = document.getElementById('side-right');
    const top = god_el.getBoundingClientRect().height + 26;
    side.style.top = top + 'px';
    side.style.maxHeight = `calc(100vh - ${top + 24}px)`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    // A window that changes aspect changes what "fills the window" means, so a
    // camera still at its default framing follows it; one the user has moved is
    // left alone beyond re-clamping to the new bounds.
    if (cam_default) reset_camera();
    else clamp_camera();
}

/** Convert canvas pixel → world coordinates (float). */
function screen_to_world(sx, sy) {
    const { tile_w, tile_h, off_x, off_y } = layout();
    return { x: (sx - off_x) / tile_w, y: (sy - off_y) / tile_h };
}

/** True if a canvas pixel lands on the pond rather than the letterbox margin.
 *
 *  The margin used to be cosmetic: screen_to_world happily returned world
 *  coordinates outside [0, GRID), and stir's rem_euclid then wrapped them onto
 *  the far side of the torus — so dragging in the dead space stirred a part of
 *  the pond nowhere near the cursor. Every pointer handler now gates on this. */
function in_pond(sx, sy) {
    const { tile_w, tile_h, off_x, off_y } = layout();
    return sx >= off_x && sx < off_x + GRID * tile_w
        && sy >= off_y && sy < off_y + GRID * tile_h;
}

/** Return rendering layout params from current canvas size and camera.
 *
 *  The pond is square and drawn at one uniform scale on both axes. It
 *  previously stretched to fill the window, giving 1.78:1 tiles on a 16:9
 *  display — which is what made the food glow read as squares (a round gradient
 *  clipped to a wide tile rect) and forced body.js to build hull geometry in
 *  screen space to undo the distortion.
 *
 *  `fit_scale` shows the whole pond with letterbox margins; `cover_scale` fills
 *  the window with no margin at all and is the default, since a 16:9 window at
 *  fit scale spent ~44% of its width on dead space. Zooming below cover brings
 *  the margins back, and they are genuinely inert — see in_pond. */
function layout() {
    const W = canvas.width, H = canvas.height;
    const s = fit_scale() * cam.zoom;
    const off_x = W / 2 - cam.x * s;
    const off_y = H / 2 - cam.y * s;
    return { W, H, tile_w: s, tile_h: s, scale_px: s, off_x, off_y };
}

/** Pixels per tile at which the whole pond just fits the smaller window axis. */
function fit_scale() {
    return Math.min(canvas.width, canvas.height) / GRID;
}

/** Zoom multiple (relative to fit) at which the pond covers the whole window. */
function cover_zoom() {
    return Math.max(canvas.width, canvas.height) / Math.min(canvas.width, canvas.height);
}

/** Keep the camera pointed at the pond.
 *
 *  On an axis where the pond is wider than the window, the centre is clamped so
 *  the view can't slide off the edge into empty space; where it is narrower, the
 *  pond is centred and that axis simply letterboxes. The world is a torus, but
 *  the renderer draws exactly one copy of it, so panning past the edge would
 *  show nothing rather than wrapping around. */
function clamp_camera() {
    const s = fit_scale() * cam.zoom;
    const half_w = canvas.width / (2 * s);
    const half_h = canvas.height / (2 * s);
    cam.x = half_w * 2 >= GRID ? GRID / 2 : Math.min(GRID - half_w, Math.max(half_w, cam.x));
    cam.y = half_h * 2 >= GRID ? GRID / 2 : Math.min(GRID - half_h, Math.max(half_h, cam.y));
}

/** Zoom about a fixed canvas point, so the world under the cursor stays put. */
function zoom_at(sx, sy, factor) {
    cam_default = false;
    const before = screen_to_world(sx, sy);
    cam.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, cam.zoom * factor));
    clamp_camera();
    const after = screen_to_world(sx, sy);
    cam.x += before.x - after.x;
    cam.y += before.y - after.y;
    clamp_camera();
}

/** Reset to the default framing: pond filling the window, centred. */
function reset_camera() {
    cam.zoom = cover_zoom();
    cam.x = GRID / 2;
    cam.y = GRID / 2;
    cam_default = true;
    clamp_camera();
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
    if (setup.isOpen()) return;   // no pond to click while choosing parameters
    // Right or middle button pans, at any zoom level.
    if (e.button === 1 || e.button === 2) {
        pan_drag = { sx: e.clientX, sy: e.clientY, cam_x: cam.x, cam_y: cam.y };
        return;
    }
    if (e.button !== 0) return;
    if (!in_pond(e.clientX, e.clientY)) return;

    // An armed god tool takes the click outright — no select, no stir.
    const w = screen_to_world(e.clientX, e.clientY);
    if (god.useAt(w.x, w.y, performance.now() / 1000)) return;

    // Don't stir yet — a short press-and-release is a select click; stirring
    // starts only once the cursor drags past DRAG_THRESHOLD_PX.
    mouse_down = { x: e.clientX, y: e.clientY };
    mouse_world = screen_to_world(e.clientX, e.clientY);
}

function on_mousemove(e) {
    if (setup.isOpen()) return;
    if (pan_drag) {
        const s = fit_scale() * cam.zoom;
        cam.x = pan_drag.cam_x - (e.clientX - pan_drag.sx) / s;
        cam.y = pan_drag.cam_y - (e.clientY - pan_drag.sy) / s;
        cam_default = false;
        clamp_camera();
        return;
    }

    mouse_world = screen_to_world(e.clientX, e.clientY);
    if (mouse_down && !stir_active) {
        const dx = e.clientX - mouse_down.x;
        const dy = e.clientY - mouse_down.y;
        if (dx * dx + dy * dy > DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) stir_active = true;
    }
    // A drag that leaves the pond stops stirring at the edge instead of wrapping
    // the impulse round to the far side of the torus.
    if (stir_active && in_pond(e.clientX, e.clientY)) {
        world.stir(mouse_world.x, mouse_world.y, 1.8, 0.45);
    }
}

function on_mouseup(e) {
    if (pan_drag && (e.button === 1 || e.button === 2 || e.type !== 'mouseup')) {
        pan_drag = null;
        return;
    }
    // Input state is released in `finally`, before anything downstream can
    // fail. Selecting used to run first: a throw inside it left `mouse_down`
    // set, and the next mousemove latched `stir_active` on with no way to
    // clear it — a click that failed turned the stir on permanently.
    try {
        if (mouse_down && !stir_active && e && e.type === 'mouseup'
            && in_pond(e.clientX, e.clientY)) {
            select_agent_at(screen_to_world(e.clientX, e.clientY));
        }
    } catch (err) {
        report_frame_error(err);
    } finally {
        mouse_down = null;
        stir_active = false;
    }
}

function on_wheel(e) {
    e.preventDefault();
    zoom_at(e.clientX, e.clientY, e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
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
        inspector.show(best.id, best.rgb);
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
        inspector.showDead(death_cause.get(selected_id));
        selected_pos = null;
        selected_id = null;
        return;
    }
    inspector.update(buf, insp_first);

    // Species membership. Traits sit at [60..69) in the inspect buffer.
    const agent = last_agents.find(a => a.id === selected_id);
    const sp = agent ? species_rows.find(s => s.id === agent.species) : null;
    if (sp) {
        const traits = Array.from(buf.slice(60, 69));
        inspector.setSpecies(
            sp.name,
            centroidDistance(traits, sp.centroid, TRAIT_BOUNDS),
            species_membership_radius(),
        );
    } else {
        inspector.setSpecies(null);
    }
    insp_first = false;
}

function on_dblclick(e) {
    if (setup.isOpen()) return;
    if (!in_pond(e.clientX, e.clientY)) return;
    const w = screen_to_world(e.clientX, e.clientY);
    world.pour_agents(w.x, w.y, 12);
}

function on_key(e) {
    // Typing in a panel field is not a shortcut. Without this, entering a seed
    // toggles half the UI on the way through — and now that C clears the screen
    // and P opens a window, a keystroke in a text box is expensive.
    const el = e.target;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) {
        return;
    }
    if (e.key === ' ') {
        e.preventDefault();
        paused = !paused;
        document.getElementById('paused-banner').style.display = paused ? 'block' : 'none';
    }
    if (e.key === '+' || e.key === '=') speed_mult = Math.min(speed_mult * 2, 16);
    if (e.key === '-')                  speed_mult = Math.max(speed_mult / 2, 0.25);
    if (e.key === 'g' || e.key === 'G') toggle_graphs();
    if (e.key === 'b' || e.key === 'B') toggle_archetypes();
    if (e.key === 'd' || e.key === 'D') toggle_debug();
    // Zen while the setup panel is up would hide the only way to start a run.
    if ((e.key === 'c' || e.key === 'C') && !setup.isOpen()) toggle_zen();
    if (e.key === 'p' || e.key === 'P') toggle_phylogeny();
    if (e.key === 'n' || e.key === 'N') open_setup();

    // Camera. Arrows pan by a fixed world distance, so a keypress covers the
    // same ground regardless of zoom.
    const pan = KEY_PAN_TILES;
    const arrows = { ArrowLeft: [-pan, 0], ArrowRight: [pan, 0], ArrowUp: [0, -pan], ArrowDown: [0, pan] };
    if (arrows[e.key]) {
        cam.x += arrows[e.key][0];
        cam.y += arrows[e.key][1];
        cam_default = false;
        clamp_camera();
        e.preventDefault();
    }
    if (e.key === '[') zoom_at(canvas.width / 2, canvas.height / 2, 1 / ZOOM_STEP);
    if (e.key === ']') zoom_at(canvas.width / 2, canvas.height / 2, ZOOM_STEP);
    if (e.key === '0') reset_camera();
    if (e.key === 'f' || e.key === 'F') {
        // Fit: whole pond visible, real letterbox on the long axis.
        cam.zoom = MIN_ZOOM;
        cam_default = false;
        clamp_camera();
    }

    // Escape does not dismiss the setup panel: opening it ended the run, and
    // there is nothing to go back to.
    if (e.key === 'Escape' && !setup.isOpen()) deselect();
    document.getElementById('h-speed').textContent = `speed ×${speed_mult}`;
}

/** The controls key is still dismissable by clicking it, and the ? chip brings
 *  it back — it is the one panel you stop needing within a minute. Only the
 *  keyboard binding went; C now clears everything at once. */
function toggle_hint_click() {
    hint_visible = !hint_visible;
    document.getElementById('hint').style.display = hint_visible ? 'block' : 'none';
    document.getElementById('hint-show').style.display = hint_visible ? 'none' : 'block';
}

/** Clear the UI — zen mode. One key for "get out of the way", replacing the
 *  separate hint and legend toggles, which nobody wanted individually.
 *
 *  It is a single class on <body> and the stylesheet does the hiding. Nothing
 *  else is touched: every panel's own visible flag, the graph timers and the
 *  engine-side brain clustering all stay exactly as they were, so leaving zen
 *  restores precisely what was open with no state to fall out of sync. */
function toggle_zen() {
    zen = !zen;
    document.body.classList.toggle('zen', zen);
}

/** The phylogeny window. Built from the roster on open, refreshed on the same
 *  cadence as the species panel. Closing it is the window's own × button. */
function toggle_phylogeny() {
    openPhylogeny(phylogeny_source, species_swatch);
}

/** What the tree reads. A function, not a snapshot, so the window tracks the
 *  run instead of freezing at the moment it was opened. */
function phylogeny_source() {
    return { rows: species_rows, step: current_step, seed: String(SEED) };
}

/** Show/hide the stat panel. Redraw only runs while it is visible — reading and
 *  plotting 600 samples for a hidden panel is pure waste. */
// Behavioural archetype overlay. Brain clustering is off in the engine until
// this is opened — it is the most expensive work in the sim and nothing else
// consumes it, so visitors who never open the panel pay nothing for it.
function toggle_archetypes() {
    archetypes_visible = !archetypes_visible;
    document.getElementById('archetypes').style.display =
        archetypes_visible ? 'block' : 'none';
    world.set_brain_clustering(archetypes_visible);
    if (archetypes_visible) {
        refresh_archetypes();
        arch_timer = setInterval(refresh_archetypes, GRAPH_REFRESH_MS);
    } else {
        clearInterval(arch_timer);
        arch_timer = null;
        // Drop the overlay colours so bodies return to their genome palette.
        arch_color.clear();
    }
}

/** One swatch per family, cycling the fallback palette when k exceeds it.
 *  Bodies colour themselves from the genome; these only seed a legend row that
 *  has not seen a member yet. */
function family_palette() {
    const k = world.cluster_k();
    return Array.from({ length: k }, (_, i) => CLUSTER_RGB[i % CLUSTER_RGB.length]);
}

// The raw k-means family legend. Behind a key because it is speciation's input
// rather than a claim about the pond: there are always k families whether or not
// the population has k distinct groups, so presenting it as the pond's structure
// overstates it. Useful for telling "speciation is wrong" apart from "clustering
// is wrong", which is exactly a debug question.
function toggle_debug() {
    debug_visible = !debug_visible;
    document.getElementById('legend-debug').style.display =
        debug_visible ? 'block' : 'none';
    // The legend lives in the right panel, so make sure that panel is up too.
    if (debug_visible) {
        document.getElementById('side-right').style.display = 'block';
    }
}

function toggle_graphs() {
    graphs_visible = !graphs_visible;
    document.getElementById('graphs').style.display = graphs_visible ? 'block' : 'none';
    if (graphs_visible) {
        refresh_graphs();
        if (!graphs_timer) graphs_timer = setInterval(refresh_graphs, GRAPH_REFRESH_MS);
    }
    // Once graph detail windows can exist independently, hiding the source
    // strip must not freeze them. The 1 Hz updater stays alive until the run is
    // cleared; clear_run_panels owns timer teardown.
}

/** Swatch colour for a species row: its centroid through the body colour path. */
function species_swatch(s) {
    const m = world.species_morph(s.index);
    if (!m || m.length < 7) return [104, 116, 124];
    const lch = genomeColor({
        pointiness: m[0], elongation: m[1], bulk: m[2], ornament: m[3],
        eyeSize: m[4], pulseRate: m[5], belly: m[6],
    });
    return oklchToRgb(lch);
}

function refresh_species() {
    const flat = world.species_list();
    const stride = species_stride();
    species_rows = parseSpecies(flat, stride, world.species_names());
    species_rows.forEach((s, i) => { s.index = i; });

    const assigned = new Set(species_rows.filter(s => s.extinctAt === null).map(s => s.id));
    const unassigned = last_agents.filter(a => !assigned.has(a.species)).length;
    update_species?.(species_rows, current_step, unassigned);
    // The tree reads the same roster, so it refreshes on the same tick rather
    // than polling for a change it cannot see.
    refreshPhylogeny(phylogeny_source, species_swatch);

    // First roster of a run seeds the "already seen" sets silently; only genuinely
    // new promotions and extinctions after that are worth interrupting for.
    if (!species_seeded) {
        species_seeded = true;
        for (const s of species_rows) {
            announced.add(s.id);
            if (s.extinctAt !== null) extinct_announced.add(s.id);
        }
        return;
    }

    for (const s of species_rows) {
        if (s.extinctAt === null && !announced.has(s.id)) {
            announced.add(s.id);
            announce?.(`${s.name} emerged — step ${s.founded}, ${s.members} members`,
                       species_swatch(s));
        }
        if (s.extinctAt !== null && !extinct_announced.has(s.id)) {
            extinct_announced.add(s.id);
            announce?.(`${s.name} is extinct — lived ${s.extinctAt - s.founded} steps`,
                       [150, 150, 155]);
        }
    }
}

function refresh_archetypes() {
    if (!archetypes_visible) return;
    // Lineage is the species id once speciation exports it; genome family is the
    // stand-in until then, so the cross-tab is useful before that lands.
    const by_id = new Map(species_rows.map(s => [s.id, s.name]));
    const rows = last_agents.map(a => ({
        brainCluster: a.brainCluster,
        lineage: by_id.get(a.species) ?? 'unassigned',
    }));
    update_archetypes?.(rows);

    // Overlay palette, keyed by cluster id so the draw loop is a map lookup.
    // Ranks come from the same summary the panel shows, so a swatch in the list
    // is the colour that agent is actually drawn in.
    const { ranks } = summarize(rows);
    arch_color.clear();
    for (const [id, rank] of ranks) arch_color.set(id, archetypeColor(rank));
}

function refresh_graphs() {
    update_graphs?.({
        flat: world.stats_history(),
        stride: stats_sample_stride(),
        totals: world.death_totals(),
        peak: world.peak_population(),
        startPop: POPULATION,
    });
}

// ── Main loop ─────────────────────────────────────────────────────────────────

function frame(ts) {
    // The loop is rescheduled in `finally`, unconditionally. Previously the
    // reschedule was the last statement of the body, so a single thrown error
    // anywhere — a panel bug, a wasm trap — stopped requestAnimationFrame for
    // good and the whole pond froze with no indication why. A frame is allowed
    // to fail; the loop is not allowed to die with it.
    try {
        frame_body(ts);
    } catch (err) {
        report_frame_error(err);
    } finally {
        requestAnimationFrame(frame);
    }
}

function frame_body(ts) {
    const raw_delta = prev_ts ? ts - prev_ts : 16.67;
    prev_ts = ts;
    frame_delta = Math.min(raw_delta, 200);

    // While parameters are being chosen there is no run to show, so the pond is
    // replaced by an idle scene rather than a frozen still of a world that is
    // about to be thrown away.
    if (!sim_running && setup.isOpen()) {
        draw_idle_scene(ts / 1000);
        return;
    }

    god.update(ts / 1000);

    if (!paused && sim_running) {
        // Cap delta to 200ms to avoid spiral of death after tab-switch
        const delta = frame_delta * speed_mult;
        world.update(delta);
    }

    const buf = world.get_state();
    render(buf, ts / 1000);

    // Panels are observation. A panel that throws must not take the pond down
    // with it, so they are isolated from the render path.
    try {
        update_panels(buf[2] | 0);
    } catch (err) {
        report_frame_error(err);
    }
}

// Errors are logged once per distinct message. At 60 fps a recurring fault
// would otherwise bury the console in thousands of identical lines, which is
// how a single real error becomes impossible to read.
const reported_errors = new Set();

function report_frame_error(err) {
    const key = String(err && err.stack ? err.stack : err);
    if (reported_errors.has(key)) return;
    reported_errors.add(key);
    console.error('[pond] frame error (loop continues):', err);
    // Surface it without needing devtools open — a fault that used to freeze
    // the pond silently should at least say so.
    announce?.(`render error — see console: ${String(err).slice(0, 80)}`, [255, 90, 90]);
}

// ── Idle scene ────────────────────────────────────────────────────────────────
//
// Shown while the setup panel is open. One creature swimming a slow figure
// loop on plain dark purple: enough to show what the pond is made of, with
// nothing on screen that the run about to start will contradict.

// Wide enough that the creature reads as small and distant, and its loop passes
// clear of the setup panel in the middle of the screen.
const IDLE_GRID = 18;
const IDLE_MORPH = {             // mid-range knobs — a generic, readable body
    pointiness: 0.55, elongation: 0.5, bulk: 0.45,
    ornament: 0.5, eyeSize: 0.6, pulseRate: 0.5, belly: 0.45,
};
// Fixed acid-lime for the title-screen creature. Not drawn from genomeColor:
// it belongs to no run and has no genome, and the arc's colours all mean
// something about a lineage that doesn't exist yet.
const IDLE_RGB = [0xC4, 0xFE, 0x01];
let idle_spec = null, idle_chain = null;

// The README's ASCII wordmark, drawn on the canvas rather than in the DOM —
// canvas paints the idle background, so a DOM logo would sit above it and the
// creature could never swim over the letters.
const LOGO = [
    '██████╗  ██████╗ ███╗   ██╗██████╗       ███████╗██████╗  █████╗ ██╗    ██╗███╗   ██╗',
    '██╔══██╗██╔═══██╗████╗  ██║██╔══██╗      ██╔════╝██╔══██╗██╔══██╗██║    ██║████╗  ██║',
    '██████╔╝██║   ██║██╔██╗ ██║██║  ██║      ███████╗██████╔╝███████║██║ █╗ ██║██╔██╗ ██║',
    '██╔═══╝ ██║   ██║██║╚██╗██║██║  ██║      ╚════██║██╔═══╝ ██╔══██║██║███╗██║██║╚██╗██║',
    '██║     ╚██████╔╝██║ ╚████║██████╔╝      ███████║██║     ██║  ██║╚███╔███╔╝██║ ╚████║',
    '╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝',
];

function draw_idle_scene(time_sec) {
    ctx.fillStyle = '#0d0524';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    draw_logo();

    if (!idle_spec) {
        idle_spec = deriveMorphology(IDLE_MORPH);
        idle_chain = createChain(IDLE_GRID / 2, IDLE_GRID / 2, idle_spec.segCount);
    }

    // Lissajous loop — never repeats its heading exactly, so the body keeps
    // flexing instead of settling into a rigid circle.
    const t = time_sec * 0.28;
    const cx = IDLE_GRID / 2 + Math.cos(t) * IDLE_GRID * 0.36;
    const cy = IDLE_GRID / 2 + Math.sin(t * 1.4) * IDLE_GRID * 0.30;
    const prev = { x: idle_chain.segs[0].x, y: idle_chain.segs[0].y };
    updateChain(idle_chain, cx, cy, {
        segCount: idle_spec.segCount, segDist: idle_spec.segDist, gridSize: IDLE_GRID,
    });

    const s = Math.min(canvas.width, canvas.height) / IDLE_GRID;
    const xform = {
        tile_w: s, tile_h: s, scale_px: s,
        off_x: canvas.width / 2 - (IDLE_GRID / 2) * s,
        off_y: canvas.height / 2 - (IDLE_GRID / 2) * s,
        gridSize: IDLE_GRID,
    };

    drawBody(
        ctx, idle_chain, idle_spec, IDLE_RGB, xform,
        {
            baseR: s * 0.16,
            energyNorm: 1,
            velX: (cx - prev.x) * 60,
            velY: (cy - prev.y) * 60,
            timeSec: time_sec,
        },
    );
}


/** Wordmark on its own smoked panel, matching the setup panel's chrome.
 *
 *  Drawn before the idle creature, so the creature passes in front of it. */
function draw_logo() {
    const W = canvas.width, H = canvas.height;
    const target_w = Math.min(W * 0.72, 860);

    // Size the glyphs so the widest line fills the target width.
    ctx.font = '10px "Courier New", monospace';
    const unit = ctx.measureText(LOGO[0]).width;
    const size = Math.max(4, (target_w / unit) * 10);
    ctx.font = `${size}px "Courier New", monospace`;

    const line_h = size * 1.02;
    const text_w = ctx.measureText(LOGO[0]).width;
    const text_h = line_h * LOGO.length;
    const cx = W / 2;
    const top = Math.max(H * 0.10, H * 0.24 - text_h / 2);

    const pad_x = size * 1.6;
    const pad_y = size * 1.2;
    const box = {
        x: cx - text_w / 2 - pad_x,
        y: top - pad_y,
        w: text_w + pad_x * 2,
        h: text_h + pad_y * 2,
    };

    ctx.save();
    ctx.fillStyle = 'rgba(4, 10, 18, 0.88)';
    ctx.strokeStyle = 'rgba(120, 255, 245, 0.18)';
    ctx.lineWidth = 1;
    const r = 6;
    ctx.beginPath();
    ctx.moveTo(box.x + r, box.y);
    ctx.arcTo(box.x + box.w, box.y, box.x + box.w, box.y + box.h, r);
    ctx.arcTo(box.x + box.w, box.y + box.h, box.x, box.y + box.h, r);
    ctx.arcTo(box.x, box.y + box.h, box.x, box.y, r);
    ctx.arcTo(box.x, box.y, box.x + box.w, box.y, r);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    ctx.shadowColor = 'rgba(120, 255, 245, 0.55)';
    ctx.shadowBlur = size * 0.9;
    ctx.fillStyle = 'rgba(150, 255, 245, 0.95)';
    for (let i = 0; i < LOGO.length; i++) {
        ctx.fillText(LOGO[i], cx, top + i * line_h);
    }
    ctx.restore();
}

/** Sim-step-gated panel refreshes — activations and means change per tick
 *  (20 Hz max), not per frame, so skip when the step hasn't advanced. */
function update_panels(step) {
    if (step === last_panel_step) return;
    last_panel_step = step;

    // Species change only on cluster ticks, and species_names() allocates a JS
    // string per species, so refresh on the cluster boundary rather than per
    // step or per frame.
    const cluster_tick = Math.floor(step / 50);
    if (cluster_tick !== last_species_tick) {
        last_species_tick = cluster_tick;
        refresh_species();
    }
    // The inspector consumes species_rows, so it must run after a roster refresh
    // on promotion ticks or a new name appears one tick late.
    refresh_inspector();

    // Legend counts: tally decoded this frame in draw_agents
    // Families no longer have a fixed colour — each swatch shows the mean
    // genome colour of its live members, so a family that has genetically
    // converged with another honestly looks like it.
    const k = world.cluster_k();
    const counts = new Array(k).fill(0);
    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    for (const a of last_agents) {
        const c = a.cluster % k;
        counts[c]++;
        for (let i = 0; i < 3; i++) sums[c][i] += a.rgb[i];
    }
    const mean_rgb = sums.map((sum, c) =>
        counts[c] > 0 ? sum.map(v => Math.round(v / counts[c])) : null);
    update_legend_counts?.(counts, mean_rgb);

    if (step - last_genome_step >= GENOME_SAMPLE_EVERY) {
        last_genome_step = step;
        update_genome_panel?.(world.trait_means());
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function render(buf, time_sec) {
    const L = layout();
    const n     = buf[0] | 0;   // agent count
    const step  = buf[2] | 0;
    current_step = step;
    const food  = buf[3] | 0;
    const avgE  = buf[4].toFixed(1);
    const alpha = buf[5];        // interpolation factor

    // Letterbox margin around the square pond — kept darker than the deepest
    // water so the dish reads as a framed object rather than a cropped view.
    ctx.fillStyle = '#06010f';
    ctx.fillRect(0, 0, L.W, L.H);

    // Before draw_agents, whose reap_stale clears the caches this reads.
    collect_deaths(buf, n, time_sec, L.scale_px);

    // Everything the pond contains is clipped to the pond. Without this the
    // water and food layers paint over the margin, which is what made the
    // letterbox look like part of the playfield.
    ctx.save();
    ctx.beginPath();
    ctx.rect(L.off_x, L.off_y, GRID * L.tile_w, GRID * L.tile_h);
    ctx.clip();

    draw_water(buf, n, L);
    draw_shimmer(buf, n, L, time_sec);
    draw_food(buf, n, L, time_sec);
    draw_agents(buf, n, alpha, L, time_sec);
    draw_dying(L, time_sec);
    draw_god_effects(L, time_sec);

    if (stir_active) draw_stir(L);

    ctx.restore();


    // HUD
    document.getElementById('h-step').textContent   = `step   ${step}`;
    document.getElementById('h-agents').textContent = `agents ${n}`;
    document.getElementById('h-energy').textContent = `energy ${avgE}`;
    document.getElementById('h-food').textContent   = `food   ${food}`;
}

// ── Tile layer ────────────────────────────────────────────────────────────────

// Water is drawn by rasterizing the fertility field into a GRID×GRID offscreen
// canvas and letting drawImage's bilinear filter upscale it to the pond. At ~90×
// magnification the tile boundaries dissolve into soft organic gradients, so the
// desert/oasis blobs the sim already generates read as fluid water rather than a
// checkerboard — for the cost of one drawImage per frame.
//
// The source is only 144 pixels, so it's rebuilt every frame rather than cached.
// That's cheaper than tracking invalidation and it picks up stir damage (which
// permanently lowers tile fertility) for free.
// Neon palette: barren water is near-black violet, fertile water an electric
// indigo. Both stay dark and blue-violet on purpose — the creatures and food are
// the only saturated, light things on screen, so they read as emissive.
const WATER_DESERT = [10, 4, 26];    // barren: deep violet-black, matte
const WATER_FERTILE = [46, 22, 138]; // max fertility: electric indigo
const MAX_FERTILITY = 1.6;

// Upscaling 12×12 straight to the window is bilinear across a ~100× jump, and
// bilinear is only C0 — the interpolation derivative jumps at every sample, which
// the eye reads as faint grid creases. Going through an intermediate at 8× with a
// small blur first makes the final upscale a fine-grained ~13× step instead, so
// the tile lattice disappears entirely.
const WATER_MID_SCALE = 8;
const WATER_BLUR_PX = 2.2;

let terrain_canvas = null, terrain_ctx = null, terrain_img = null;
let water_mid = null, water_mid_ctx = null;
// Grid size the offscreen canvases above were built for. They used to be built
// once and kept forever, so starting a run on any grid other than the boot
// default wrote GRID² tiles into a 12×12 ImageData and the water came out
// truncated. Rebuilt whenever the run's grid size changes.
let terrain_grid = 0;

function draw_water(buf, n, { tile_w, tile_h, off_x, off_y }) {
    if (!terrain_canvas || terrain_grid !== GRID) {
        terrain_grid = GRID;

        terrain_canvas = document.createElement('canvas');
        terrain_canvas.width = GRID;
        terrain_canvas.height = GRID;
        terrain_ctx = terrain_canvas.getContext('2d');
        terrain_img = terrain_ctx.createImageData(GRID, GRID);

        water_mid = document.createElement('canvas');
        water_mid.width = GRID * WATER_MID_SCALE;
        water_mid.height = GRID * WATER_MID_SCALE;
        water_mid_ctx = water_mid.getContext('2d');
    }

    const tile_base = HEADER_LEN + n * AGENT_STRIDE;
    const px = terrain_img.data;

    for (let ti = 0; ti < GRID * GRID; ti++) {
        const fertility = buf[tile_base + ti * TILE_STRIDE + 1];
        // Barren tiles are exactly 0.0 (see assign_barren_tiles), so this ramp
        // keeps deserts pinned at the matte end rather than blending into them.
        const t = Math.min(fertility / MAX_FERTILITY, 1);
        const o = ti * 4;
        for (let c = 0; c < 3; c++) {
            px[o + c] = WATER_DESERT[c] + (WATER_FERTILE[c] - WATER_DESERT[c]) * t;
        }
        px[o + 3] = 255;
    }

    terrain_ctx.putImageData(terrain_img, 0, 0);

    // Pass 1: 12×12 → 96×96, blurred. The blur runs at this small size, so it
    // costs almost nothing and still smooths a full tile's worth of lattice.
    const m = GRID * WATER_MID_SCALE;
    water_mid_ctx.clearRect(0, 0, m, m);
    water_mid_ctx.imageSmoothingEnabled = true;
    water_mid_ctx.filter = `blur(${WATER_BLUR_PX}px)`;
    water_mid_ctx.drawImage(terrain_canvas, 0, 0, GRID, GRID, 0, 0, m, m);
    water_mid_ctx.filter = 'none';

    // Pass 2: 96×96 → pond.
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(water_mid, 0, 0, m, m, off_x, off_y, GRID * tile_w, GRID * tile_h);
}

// ── Shimmer (6b) ──────────────────────────────────────────────────────────────
//
// Caustics masked to fertile water. Summed sine waves at different frequencies
// and drift directions give an interference pattern that reads as light through
// moving water; deserts get zero shimmer, which is what makes them read matte
// against it.
//
// Computed at CAUSTIC_PX² and upscaled, same trick as the water: the per-pixel
// cost stays trivial and the upscale blur is what makes it look liquid rather
// than like a sine grid.
const CAUSTIC_PX = 96;
const SHIMMER_ALPHA = 0.5;

let caustic_canvas = null, caustic_ctx = null, caustic_img = null;

function draw_shimmer(buf, n, { tile_w, tile_h, off_x, off_y }, time_sec) {
    if (!caustic_canvas) {
        caustic_canvas = document.createElement('canvas');
        caustic_canvas.width = caustic_canvas.height = CAUSTIC_PX;
        caustic_ctx = caustic_canvas.getContext('2d');
        caustic_img = caustic_ctx.createImageData(CAUSTIC_PX, CAUSTIC_PX);
    }

    const tile_base = HEADER_LEN + n * AGENT_STRIDE;
    const px = caustic_img.data;
    const t = time_sec;

    for (let py = 0; py < CAUSTIC_PX; py++) {
        // Nearest-tile lookup is enough: the upscale smooths the mask edges.
        const ty = Math.min(GRID - 1, (py * GRID / CAUSTIC_PX) | 0);
        for (let pxi = 0; pxi < CAUSTIC_PX; pxi++) {
            const tx = Math.min(GRID - 1, (pxi * GRID / CAUSTIC_PX) | 0);
            const toff = tile_base + (ty * GRID + tx) * TILE_STRIDE;
            const fert = buf[toff + 1];
            const o = (py * CAUSTIC_PX + pxi) * 4;

            if (fert <= 0) { px[o + 3] = 0; continue; }   // desert: matte, no shimmer

            const u = pxi * 0.19, v = py * 0.19;
            const w1 = Math.sin(u + t * 0.7);
            const w2 = Math.sin(v * 0.9 - t * 0.5);
            const w3 = Math.sin((u + v) * 0.6 + t * 0.9);
            // Sharpened toward the crests so it reads as caustic banding.
            let c = (w1 + w2 + w3) / 3;
            c = Math.pow(Math.max(0, c), 2.2);

            // Food presence brightens the local shimmer without driving the
            // base colour, which would strobe as agents eat.
            const food = buf[toff];
            const mask = Math.min(1, fert / 1.6) * (0.6 + Math.min(food, 3) * 0.18);

            px[o] = 120; px[o + 1] = 255; px[o + 2] = 245;
            px[o + 3] = c * mask * 255 * SHIMMER_ALPHA;
        }
    }

    caustic_ctx.putImageData(caustic_img, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(caustic_canvas, 0, 0, CAUSTIC_PX, CAUSTIC_PX,
                  off_x, off_y, GRID * tile_w, GRID * tile_h);
    ctx.restore();
}

// Each food *unit* is its own drifting node rather than a per-tile glow. The old
// pass clipped a radial gradient to the tile rect, which chopped the round falloff
// into a hard-edged square — the single loudest grid artifact on screen.
//
// Node positions are derived renderer-side from a hash of (tile, unit index), so
// they need no sim change: food stays an integer count per tile, which is what
// eating operates on. Drift is bounded well inside the tile so a node never
// floats somewhere the agent can't actually eat from.
const ORB_TEX_PX = 64;
const FOOD_DRIFT = 0.16;    // bob amplitude, tiles
const FOOD_INSET = 0.26;    // keeps nodes off tile edges

let orb_tex = null;

function build_orb() {
    orb_tex = document.createElement('canvas');
    orb_tex.width = orb_tex.height = ORB_TEX_PX;
    const c = orb_tex.getContext('2d');
    const h = ORB_TEX_PX / 2;
    const g = c.createRadialGradient(h, h, 0, h, h, h);
    // Acid-lime core into an electric green halo — the brightest thing in the
    // pond after the creatures themselves.
    g.addColorStop(0.00, 'rgba(236,255,190,1.00)');
    g.addColorStop(0.18, 'rgba(150,255,60,0.85)');
    g.addColorStop(0.45, 'rgba(70,240,120,0.32)');
    g.addColorStop(1.00, 'rgba(30,200,110,0)');
    c.fillStyle = g;
    c.fillRect(0, 0, ORB_TEX_PX, ORB_TEX_PX);
}

/** Stable pseudo-random in [0,1) from two small integers. */
function hash01(a, b) {
    let h = Math.imul(a + 1, 73856093) ^ Math.imul(b + 1, 19349663);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function draw_food(buf, n, { tile_w, tile_h, off_x, off_y }, time_sec) {
    if (!orb_tex) build_orb();
    const tile_base = HEADER_LEN + n * AGENT_STRIDE;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (let ty = 0; ty < GRID; ty++) {
        for (let tx = 0; tx < GRID; tx++) {
            const ti = ty * GRID + tx;
            const food = buf[tile_base + ti * TILE_STRIDE];   // 0–3
            if (food <= 0) continue;

            for (let u = 0; u < food; u++) {
                // Fixed anchor inside the tile, then a slow independent bob.
                const rx = FOOD_INSET + hash01(ti, u) * (1 - 2 * FOOD_INSET);
                const ry = FOOD_INSET + hash01(ti, u + 97) * (1 - 2 * FOOD_INSET);
                const ph = hash01(ti, u + 421) * Math.PI * 2;
                const wob = 0.55 + hash01(ti, u + 733) * 0.5;

                const wx = tx + rx + Math.cos(time_sec * wob + ph) * FOOD_DRIFT;
                const wy = ty + ry + Math.sin(time_sec * wob * 0.83 + ph) * FOOD_DRIFT;

                const pulse = 0.82 + 0.18 * Math.sin(time_sec * 1.7 + ph);
                const r = Math.min(tile_w, tile_h) * 0.19 * pulse;

                ctx.drawImage(
                    orb_tex,
                    off_x + wx * tile_w - r, off_y + wy * tile_h - r, r * 2, r * 2,
                );
            }
        }
    }
    ctx.restore();
}

// ── Agent layer ───────────────────────────────────────────────────────────────

/** Drop chain/morph-spec/color cache entries for agents no longer alive. */
function reap_stale(chains_map, morph_map, color_map, predator_map, live_ids) {
    for (const id of chains_map.keys()) {
        if (!live_ids.has(id)) {
            chains_map.delete(id);
            morph_map.delete(id);
            color_map.delete(id);
        }
    }
    // Predators keep their own chains and never appear in chains_map, so they
    // need their own sweep or a departed hunter's body would leak.
    for (const id of predator_map.keys()) {
        if (!live_ids.has(id)) predator_map.delete(id);
    }
}

// Epitaph per cause of death. Codes come from CauseOfDeath::code() in world.rs
// and must stay in sync with it.
const EPITAPH = {
    0: ':/',      // Starvation — in a pond this full? really?
    1: '[RIP]',   // OldAge
    2: 'X_X',     // KilledInCombat
    3: 'X_X',     // EatenAlive
};

const DEATH_CAUSE = {
    0: 'starved',
    1: 'old age',
    2: 'killed in combat',
    3: 'eaten alive',
};

// Cause by agent id, kept only long enough for the inspector to report it. The
// agent is gone from the sim by the time the panel notices, so inspect_agent
// can't be asked — this is the only surviving record.
let death_cause = new Map();

/** Seed death effects from the sim's death queue. Must run before reap_stale
 *  clears the caches, since colour and size are looked up by id from them. */
function collect_deaths(buf, n, time_sec, scale_px) {
    const count = buf[6] | 0;                 // H_DEATH_COUNT
    if (count === 0) return;

    const base = HEADER_LEN + n * AGENT_STRIDE + GRID * GRID * TILE_STRIDE;
    for (let d = 0; d < count; d++) {
        const off = base + d * DEATH_STRIDE;
        const id = buf[off] | 0;
        death_cause.set(id, DEATH_CAUSE[buf[off + 3] | 0] ?? 'unknown');
        if (death_cause.size > 512) {
            // Bounded: only the selected agent's entry is ever read.
            death_cause.delete(death_cause.keys().next().value);
        }
        const lch = color_state.get(id);
        const spec = morph_cache.get(id);
        dying.push({
            x: buf[off + 1],
            y: buf[off + 2],
            rgb: lch ? oklchToRgb(lch) : [180, 220, 255],
            r: scale_px * 0.09 * (spec ? spec.envelope[2] : 1.25),
            glyph: EPITAPH[buf[off + 3] | 0] ?? '[RIP]',
            t0: time_sec,
        });
    }
}

/** Death dissolve: the body's glow blows outward and fades while its core
 *  collapses inward — a bioluminescent puff rather than an agent teleporting
 *  out of existence. Purely cosmetic; the sim has already reaped the agent. */
function draw_dying({ tile_w, tile_h, off_x, off_y }, time_sec) {
    if (dying.length === 0) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const d of dying) {
        const p = (time_sec - d.t0) / DEATH_SEC;
        if (p < 0 || p >= 1) continue;

        const [r8, g8, b8] = d.rgb;
        const sx = off_x + d.x * tile_w;
        const sy = off_y + d.y * tile_h;
        const ease = 1 - Math.pow(1 - p, 2);   // quick burst, slow settle
        const fade = (1 - p) * (1 - p);

        // Expanding shell
        const rad = d.r * (1 + ease * 3.6);
        const g = ctx.createRadialGradient(sx, sy, rad * 0.25, sx, sy, rad);
        g.addColorStop(0, `rgba(${r8},${g8},${b8},0)`);
        g.addColorStop(0.55, `rgba(${r8},${g8},${b8},${fade * 0.45})`);
        g.addColorStop(1, `rgba(${r8},${g8},${b8},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(sx, sy, rad, 0, Math.PI * 2);
        ctx.fill();

        // Collapsing core
        const cr = d.r * (1 - ease);
        if (cr > 0.4) {
            ctx.fillStyle = `rgba(${r8},${g8},${b8},${fade * 0.8})`;
            ctx.beginPath();
            ctx.arc(sx, sy, cr, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // Epitaphs go in a second pass with normal compositing — additive text over
    // a bright shell washes out to unreadable white.
    ctx.globalCompositeOperation = 'source-over';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const d of dying) {
        const p = (time_sec - d.t0) / DEATH_SEC;
        if (p < 0 || p >= 1) continue;

        const size = Math.max(11, d.r * 1.5);
        // Rises as it fades, like a bubble leaving the body.
        const rise = d.r * 2.2 * (1 - Math.pow(1 - p, 2));
        const a = p < 0.15 ? p / 0.15 : Math.pow(1 - (p - 0.15) / 0.85, 1.5);

        const gx = off_x + d.x * tile_w;
        const gy = off_y + d.y * tile_h - rise;
        // A violent death reads red and burns. The other epitaphs stay cool
        // white — the point is that X_X is the one you notice.
        const violent = d.glyph === 'X_X';

        ctx.font = `bold ${size}px ui-monospace, monospace`;
        ctx.lineWidth = Math.max(2, size * 0.18);
        ctx.strokeStyle = `rgba(2,6,14,${a * 0.85})`;
        ctx.strokeText(d.glyph, gx, gy);

        if (violent) {
            ctx.save();
            ctx.shadowColor = `rgba(255,40,60,${a})`;
            ctx.shadowBlur = size * 0.9;
            ctx.fillStyle = `rgba(255,70,86,${a})`;
            // Twice, so the bloom stacks into a genuine glow rather than a halo.
            ctx.fillText(d.glyph, gx, gy);
            ctx.shadowBlur = size * 0.45;
            ctx.fillStyle = `rgba(255,170,178,${a})`;
            ctx.fillText(d.glyph, gx, gy);
            ctx.restore();
        } else {
            ctx.fillStyle = `rgba(236,248,255,${a})`;
            ctx.fillText(d.glyph, gx, gy);
        }
    }

    ctx.restore();
    dying = dying.filter(d => time_sec - d.t0 < DEATH_SEC);
}

function draw_agents(buf, n, alpha, L, time_sec) {
    const { tile_w, tile_h, scale_px, off_x, off_y } = L;
    const live_ids = new Set();
    predators.clear();
    const pstate = world.predators_state();
    const pstride = predator_state_stride();
    for (let i = 0; i < pstate.length; i += pstride) {
        predators.set(pstate[i], {
            leaving: pstate[i + 1] === 1,
            tier: pstate[i + 2] | 0,
            angle: pstate[i + 3],
            reach: pstate[i + 4],
        });
    }

    // Clip to the pond (= the whole window now); seam copies from wrapping
    // bodies would otherwise paint past the canvas edges.
    ctx.save();
    ctx.beginPath();
    ctx.rect(off_x, off_y, GRID * tile_w, GRID * tile_h);
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

        if (a.id === selected_id) selected_pos = { x: hx_w, y: hy_w };

        // Colour is a pure function of the (immutable) genome, so it's derived
        // once per agent and cached. Live energy only dims it — no crossfade
        // needed, because there is no longer any discontinuity to hide.
        let lch = color_state.get(a.id);
        if (!lch) {
            lch = genomeColor(a.morph);
            color_state.set(a.id, lch);
        }
        // Energy dims the body, but only down to 72% lightness — at the old 55%
        // floor a starving neon body desaturated into brown, which read as a
        // rendering fault rather than as a hungry creature.
        let palette = oklchToRgb([
            lch[0] * (0.72 + a.energyNorm * 0.28),
            lch[1],
            lch[2],
        ]);
        // Archetype overlay. Temporary by design — toggling off restores the
        // trait-derived colour, which is the pond's default reading.
        if (archetypes_visible) {
            const arch = arch_color.get(a.brainCluster);
            if (arch) {
                const dim = 0.72 + a.energyNorm * 0.28;
                palette = [arch[0] * dim | 0, arch[1] * dim | 0, arch[2] * dim | 0];
            }
        }
        last_agents.push({
            id: a.id, x: hx_w, y: hy_w,
            cluster: a.genomeCluster, brainCluster: a.brainCluster,
            species: a.species, rgb: palette,
        });
        // Creatures are the subject; food orbs were drawn ~4x their size and the
        // pond read as a field of green dots with specks swimming in it.
        const base_r = scale_px * (0.105 + a.energyNorm * 0.07 + a.morph.pointiness * 0.05);

        // Apex predators do not use the body pipeline at all — they are hard
        // geometric shapes, so they can never be mistaken for one of their prey.
        if (predators.has(a.id)) {
            draw_predator(a.id, hx_w, hy_w, L, time_sec, predators.get(a.id));
            continue;
        }

        drawBody(
            ctx, chain, spec, palette,
            { tile_w, tile_h, scale_px, off_x, off_y, gridSize: GRID },
            { baseR: base_r, energyNorm: a.energyNorm, velX: a.velX, velY: a.velY, timeSec: time_sec },
        );
    }

    if (selected_pos) draw_selection_ring(L, time_sec);

    ctx.restore();
    reap_stale(chains, morph_cache, color_state, predator_chains, live_ids);
}

/** Announce the hunt, and say whether it was summoned or triggered itself.
 *
 *  The automatic cull is the one event in the sim the player didn't ask for, so
 *  it has to be legible: without this, a pond that suddenly halves looks like a
 *  crash rather than a mechanic. */
/** Comet impacts, spreading salt, and the sweep wipe.
 *
 *  Purely presentational: the kills have already happened engine-side by the
 *  time these draw, so an effect that is mid-animation is showing you what was
 *  done, not doing it. */
function draw_god_effects({ tile_w, tile_h, scale_px, off_x, off_y }, time_sec) {
    const fx = god.effects(time_sec);
    const sx = wx => off_x + wx * tile_w;
    const sy = wy => off_y + wy * tile_h;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const c of fx.comets) {
        const fade = 1 - c.t;
        const r = c.radius * scale_px * (0.6 + c.t * 0.9);
        const g = ctx.createRadialGradient(sx(c.x), sy(c.y), 0, sx(c.x), sy(c.y), r);
        g.addColorStop(0.00, `rgba(255,255,235,${0.95 * fade})`);
        g.addColorStop(0.30, `rgba(255,120,240,${0.75 * fade})`);
        g.addColorStop(0.65, `rgba(150,60,255,${0.35 * fade})`);
        g.addColorStop(1.00, 'rgba(70,0,120,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(sx(c.x), sy(c.y), r, 0, Math.PI * 2);
        ctx.fill();

        // Shock ring, expanding past the blast so the radius is legible.
        ctx.strokeStyle = `rgba(240,200,255,${0.8 * fade})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx(c.x), sy(c.y), c.radius * scale_px * (0.4 + c.t * 1.6), 0, Math.PI * 2);
        ctx.stroke();
    }

    for (const s of fx.salts) {
        const fade = 1 - s.t * 0.7;
        const r = s.radius * scale_px;
        const g = ctx.createRadialGradient(sx(s.x), sy(s.y), r * 0.35, sx(s.x), sy(s.y), r);
        g.addColorStop(0.00, 'rgba(180,240,255,0)');
        g.addColorStop(0.75, `rgba(220,250,255,${0.18 * fade})`);
        g.addColorStop(1.00, `rgba(255,255,255,${0.55 * fade})`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(sx(s.x), sy(s.y), r, 0, Math.PI * 2);
        ctx.fill();

        // Crystals scattered on the advancing front.
        ctx.fillStyle = `rgba(255,255,255,${0.7 * fade})`;
        for (let i = 0; i < 22; i++) {
            const a = hash01(i, 7) * Math.PI * 2 + s.t * 0.6;
            const rr = r * (0.82 + hash01(i, 13) * 0.18);
            const px = sx(s.x) + Math.cos(a) * rr;
            const py = sy(s.y) + Math.sin(a) * rr;
            const sz = scale_px * 0.02 * (0.6 + hash01(i, 29));
            ctx.fillRect(px - sz, py - sz, sz * 2, sz * 2);
        }
    }

    if (fx.sweep) {
        const edge_w = sx(fx.sweep.t * GRID);
        const left = off_x;
        const g = ctx.createLinearGradient(edge_w - scale_px * 2, 0, edge_w, 0);
        g.addColorStop(0, 'rgba(120,255,245,0)');
        g.addColorStop(1, 'rgba(200,255,250,0.55)');
        ctx.fillStyle = g;
        ctx.fillRect(left, off_y, edge_w - left, GRID * tile_h);

        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(edge_w, off_y);
        ctx.lineTo(edge_w, off_y + GRID * tile_h);
        ctx.stroke();
    }

    ctx.restore();
}

// Predator tiers.
//
// Deliberately not creatures. Everything else in the pond is a soft, evolved
// body; these are hard geometric shapes several times their size, so at a glance
// they are obviously not part of the ecology.
//
// The pond escalates when a wave cannot clear the field, and the shape is the
// tell: you should know which tier is in the water at a glance, without a
// banner.
//
//   0  grey triangle pack    resident, weakest, hunts in numbers
//   1  red octagon           hit and run, spins
//   2  rainbow rectangle     hit and run, slow sweep, kills along its edges
const TIER_RGB = [
    [0xE4, 0xEA, 0xFF],   // grey-white
    [0xFF, 0x36, 0x48],   // red
    null,                 // rainbow, built per frame
];

function draw_predator(id, wx, wy, L, time_sec, p) {
    const tier = p?.tier ?? 0;
    if (tier === 0) return draw_predator_triangles(id, wx, wy, L, time_sec, p);
    if (tier === 1) return draw_predator_octagon(wx, wy, L, time_sec, p);
    return draw_predator_rectangle(wx, wy, L, time_sec, p);
}

/** Shared aura under any predator, sized to what it actually kills at. */
function aura(sx, sy, radius_px, rgb, strength) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, radius_px);
    g.addColorStop(0.00, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.34 * strength})`);
    g.addColorStop(0.45, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.14 * strength})`);
    g.addColorStop(1.00, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx, sy, radius_px, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
}

/** Regular polygon path, centred at the origin. */
function polygon(sides, r, rotation) {
    ctx.beginPath();
    for (let i = 0; i < sides; i++) {
        const a = rotation + (i / sides) * Math.PI * 2;
        const x = Math.cos(a) * r;
        const y = Math.sin(a) * r;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
}

// Tier 0 — the original articulated triangle chain, unchanged.
function draw_predator_triangles(id, wx, wy, { tile_w, tile_h, scale_px, off_x, off_y }, time_sec, p) {
    let chain = predator_chains.get(id);
    if (!chain) {
        chain = createChain(wx, wy, PREDATOR_SEGS);
        // Smoothed draw angles, one per segment. Null until the first frame
        // gives each segment something to smooth from.
        chain.angles = new Array(PREDATOR_SEGS).fill(null);
        predator_chains.set(id, chain);
    }
    if (!chain.angles) chain.angles = new Array(chain.segs.length).fill(null);
    updateChain(chain, wx, wy, {
        segCount: PREDATOR_SEGS, segDist: PREDATOR_SEG_DIST, gridSize: GRID,
    });

    const pulse = 0.6 + 0.25 * Math.sin(time_sec * 5);
    // A sated resident dims down — it is still here, but it is not hunting.
    const alpha = p?.leaving ? 0.55 : 1.0;
    const [r8, g8, b8] = TIER_RGB[0];

    const sx = w => off_x + w * tile_w;
    const sy = w => off_y + w * tile_h;
    const head = chain.segs[0];

    aura(sx(head.x), sy(head.y), scale_px * 1.5, TIER_RGB[0], pulse * alpha);

    ctx.save();
    ctx.globalAlpha = alpha;
    for (let i = 0; i < chain.segs.length; i++) {
        const seg = chain.segs[i];
        const next = chain.segs[Math.min(i + 1, chain.segs.length - 1)];

        // Toroidal delta: a segment across the wrap seam must not flip the
        // triangle round to face the long way across the pond.
        let dx = seg.x - next.x;
        let dy = seg.y - next.y;
        if (dx > GRID * 0.5) dx -= GRID; else if (dx < -GRID * 0.5) dx += GRID;
        if (dy > GRID * 0.5) dy -= GRID; else if (dy < -GRID * 0.5) dy += GRID;
        // Coincident segments (the tail, before the chain has stretched out)
        // carry no direction — hold the last angle rather than snapping to zero.
        const settled = dx === 0 && dy === 0;
        const prev_angle = chain.angles[i];
        let angle;
        if (settled) {
            angle = prev_angle ?? 0;
        } else if (prev_angle === null) {
            angle = Math.atan2(dy, dx);
        } else {
            // Shortest way round, eased: the wrap at ±π must not spin the body
            // the long way when a hunter turns through it.
            let d = Math.atan2(dy, dx) - prev_angle;
            d = ((d + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
            angle = prev_angle + d * PREDATOR_ANGLE_EASE;
        }
        chain.angles[i] = angle;

        const taper = 1 - i * 0.22;              // head largest, tail smallest
        const len = scale_px * 0.52 * taper;
        const wide = scale_px * 0.34 * taper;

        ctx.save();
        ctx.translate(sx(seg.x), sy(seg.y));
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(len, 0);
        ctx.lineTo(-len * 0.55, wide);
        ctx.lineTo(-len * 0.55, -wide);
        ctx.closePath();
        ctx.fillStyle = `rgba(${r8},${g8},${b8},${0.92 - i * 0.12})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(255,255,255,${0.75 * pulse})`;
        ctx.lineWidth = Math.max(1, scale_px * 0.02);
        ctx.stroke();
        ctx.restore();
    }
    ctx.restore();
}

// Tier 1 — a rotating red octagon. Hit and run.
function draw_predator_octagon(wx, wy, { tile_w, tile_h, scale_px, off_x, off_y }, time_sec, p) {
    const sx = off_x + wx * tile_w;
    const sy = off_y + wy * tile_h;
    const pulse = 0.7 + 0.25 * Math.sin(time_sec * 7);
    const alpha = p?.leaving ? 0.55 : 1.0;
    const r = Math.max(scale_px * 0.6, p.reach * scale_px);

    aura(sx, sy, r * 1.45, TIER_RGB[1], pulse * alpha);

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(sx, sy);
    // Engine-side angle, not a render clock: what you see spinning is the same
    // number the kill test uses.
    ctx.rotate(p.angle);
    polygon(8, r, 0);
    ctx.fillStyle = 'rgba(255,54,72,0.82)';
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, scale_px * 0.035);
    ctx.strokeStyle = `rgba(255,200,205,${0.9 * pulse})`;
    ctx.stroke();
    // Inner ring, so the spin is legible on a near-regular polygon.
    polygon(8, r * 0.55, Math.PI / 8);
    ctx.strokeStyle = `rgba(255,120,130,${0.7 * pulse})`;
    ctx.lineWidth = Math.max(1, scale_px * 0.02);
    ctx.stroke();
    ctx.restore();
}

// Tier 2 — R E C T A N G L E. Sweeps slowly and kills everything any edge
// touches, so it is drawn at exactly the extent of that sweep.
function draw_predator_rectangle(wx, wy, { tile_w, tile_h, scale_px, off_x, off_y }, time_sec, p) {
    const sx = off_x + wx * tile_w;
    const sy = off_y + wy * tile_h;
    const alpha = p?.leaving ? 0.6 : 1.0;
    const half_len = p.reach * scale_px;
    const half_w = RECT_HALF_WIDTH_WORLD * scale_px;

    aura(sx, sy, half_len * 1.25, [255, 255, 255], 0.5 * alpha);

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(sx, sy);
    ctx.rotate(p.angle);

    // Rainbow along the long axis, cycling so it never reads as a flat bar.
    const grd = ctx.createLinearGradient(-half_len, 0, half_len, 0);
    const shift = (time_sec * 0.25) % 1;
    for (let i = 0; i <= 6; i++) {
        const t = i / 6;
        const hue = ((t + shift) % 1) * 360;
        grd.addColorStop(t, `hsl(${hue.toFixed(0)}, 95%, 58%)`);
    }
    ctx.fillStyle = grd;
    ctx.fillRect(-half_len, -half_w, half_len * 2, half_w * 2);

    ctx.lineWidth = Math.max(2, scale_px * 0.04);
    ctx.strokeStyle = `rgba(255,255,255,${0.85 * alpha})`;
    ctx.strokeRect(-half_len, -half_w, half_len * 2, half_w * 2);
    ctx.restore();
}

/** Pulsing ring around the selected agent (drawn inside the grid clip). */
function draw_selection_ring({ tile_w, tile_h, scale_px, off_x, off_y }, time_sec) {
    const sx = off_x + selected_pos.x * tile_w;
    const sy = off_y + selected_pos.y * tile_h;
    const r = scale_px * (0.30 + 0.03 * Math.sin(time_sec * 4));
    ctx.strokeStyle = 'rgba(255,60,220,0.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.stroke();
}

// ── Stir indicator ────────────────────────────────────────────────────────────

function draw_stir({ tile_w, tile_h, scale_px, off_x, off_y }) {
    const sx = off_x + mouse_world.x * tile_w;
    const sy = off_y + mouse_world.y * tile_h;
    const r  = 1.8 * scale_px;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
    grd.addColorStop(0,   'rgba(120,255,245,0.10)');
    grd.addColorStop(0.5, 'rgba(180,60,255,0.06)');
    grd.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();

    // Ripple ring
    ctx.strokeStyle = 'rgba(120,255,245,0.28)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, r * 0.65, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}

// ── Launch ────────────────────────────────────────────────────────────────────
boot();
