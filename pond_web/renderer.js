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
    disease_stride,
    disease_species_columns,
} from '../pond_core/pkg/pond_core.js';
import { decodeAgent } from './decode.js';
import { createChain, updateChain } from './chain.js';
import { deriveMorphology } from './morphology.js';
import { drawBody, PASS_GLOW, PASS_CORE } from './body.js';
import {
    ATLAS_PPT, SPRITE_LOD_MAX_SCALE_PX, spriteKey, spriteFor, energyRep,
    beginAtlasFrame, resetAtlas, resetAtlasStats, atlasCanvas, atlasStats,
} from './atlas.js';
import { oklchToRgb, speciesColor, unassignedColor, strategyGlow } from './color.js';
import { initLegend, initGenomePanel } from './panels.js';
import { initArchetypes, archetypeColor, summarize } from './archetypes.js';
import {
    parseSpecies, initSpeciesPanel, initToast, centroidDistance,
} from './species.js';
import { initGraphs } from './graphs.js';
import { initSetup } from './setup.js';
import { initSplash } from './splash.js';
import { initGod } from './god.js';
import { initInspector } from './inspector.js';
import { openPhylogeny, refreshPhylogeny } from './phylogeny.js';
import { initDiseasePanel, parseDiseases } from './disease.js';
import { closeFloatingPrefix } from './floating.js';

// Wire format this page was written against. The engine reports its own; a
// mismatch means pond_core/pkg and pond_web were built from different commits,
// and every flat buffer the page reads would be off by some number of floats —
// silently, producing plausible wrong numbers rather than an error. See
// pond_core/src/schema.rs.
const EXPECTED_SCHEMA = 7;

// ── Sim config ────────────────────────────────────────────────────────────────
// Set from the setup panel (`N`) and fixed for the life of a run — changing any
// of them means building a new World, which restart() does.
//
// These are the opening pond's parameters, and they are chosen rather than
// arbitrary. The old 12×12/100 default went extinct in roughly a third of seeds
// inside 10,000 ticks and in most of them eventually: every pond overshoots its
// food early, crashes, and bottoms out in a bottleneck, and on 144 tiles that
// bottleneck is a handful of animals that drift to zero. The pond is not
// short of food and predation is not the cause — the same crash happens with
// ambient predators switched off. It is the area.
//
// Measured, 10,000 ticks, 8 seeds each: at 12×12 survival is ~2/3 at any
// starting density between 0.26 and 0.69 agents per tile, and every grid from
// 14 up survives 8/8 across that whole density range. 24×24 with 150 agents
// (0.26/tile) then survived 10/10 seeds to 120,000 ticks — 100 minutes of real
// time at 20 Hz — averaging 60–320 animals and turning over dozens of species.
const GRID_DEFAULT = 24;
const POPULATION_DEFAULT = 150;
// Chosen for where it is at the warm-start mark: ~100 animals at avg energy 58,
// two live species — one of 60 and one of 12 — a third already extinct, and a
// disease circulating with 48 carriers. So the pond opens mid-story rather than
// mid-bloom: something has already won, something is already losing, and
// something is already dead.
//
// Verified to 60,000 ticks (50 minutes at 20 Hz): never dips below 58 animals,
// finishes at 63 on 75 average energy, and turns over 7 species along the way.
// Any seed survives; this one arrives interesting.
const SEED_DEFAULT = 21n;
// Ticks the opening pond is wound forward before it is shown. Speciation needs
// generations — the first promotion lands around tick 1,700–3,200 — so a pond
// shown from tick 0 is unnamed grey soup for several minutes, which is longer
// than anyone gives it. Only the auto-started pond is warmed: a run started
// from the setup panel begins at tick 0, because that one was asked for.
const WARM_START_STEPS = 4200;
// Milliseconds per frame spent winding forward, so the page keeps painting and
// the card stays clickable while it happens.
const WARM_BUDGET_MS = 12;
// The white curtain over the opening pond is a five-second CSS fade (see #veil
// in index.html). It is deliberately not tied to warm-start progress: driving
// it from the tick count made it stutter with the frame budget and finish
// whenever the wasm happened to finish, which is not a timing anyone chose.
// This copy of its length only schedules the fallback reveal of the welcome
// card; the animation itself is the stylesheet's.
const VEIL_MS = 5000;
// How long the curtain is solid white, matching the 30% stop in `veil-fade`.
// A run started by hand is held still for exactly this long: the first ticks of
// a fresh pond are the ones worth seeing, and running them behind an opaque
// sheet spends them on nobody. The opening pond is exempt — it is being wound
// forward on purpose.
const VEIL_HOLD_MS = 1500;
// Bumped on every restart, so a hold timer left over from a run that has since
// been replaced cannot start the wrong world.
let run_epoch = 0;

let GRID = GRID_DEFAULT;
let POPULATION = POPULATION_DEFAULT;
let SEED = SEED_DEFAULT;

// Fallback family swatches. Bodies no longer use these — colour comes from the
// genome — but the legend needs a swatch before any member of
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
// The opening pond runs from boot. It stops only while the setup panel is up:
// choosing parameters while the pond you're configuring runs behind the panel
// would mean the run you start is never the run you were looking at.
let sim_running = true;
// Ticks still owed to the warm start. Counted down in frame_body a chunk at a
// time; zero means the pond is running at ordinary speed.
let warm_remaining = 0;
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
let splash;
let update_diseases;
// The rule dials, fixed for the life of a run and set from the setup panel.
// Held here so a restart can re-apply them to the new world.
let dials = null;
let graphs_visible = false;
let debug_visible = false;
let archetypes_visible = false;

// ── Render timing (M) ────────────────────────────────────────────────────────
//
// Every optimisation below this line is measured against these five numbers, so
// they exist before any of it: op-count arithmetic said the agent layer was the
// whole bill, and an estimate is not an acceptance test. Exponential moving
// average rather than a per-frame readout, because a number that changes 60
// times a second cannot be read, and the thing being chased is the *typical*
// frame, not the worst one.
//
// The spans are wall-clock around each pass, so they include rasterisation the
// browser may defer — Canvas2D is pipelined and a `fill` can bill to whichever
// call next forces a flush. Treat the split as apportionment, not as isolation:
// the totals are honest, and a pass that dominates really is where the work is.
let perf_visible = false;
// Sprite LOD, **off by default**, toggleable with L.
//
// Off because the atlas does not scale to a diverse pond and shipping it armed
// would hand the bug to anyone on a short window. Its key multiplies colour by
// silhouette, and silhouette varies per *agent* — pointiness, armour, fins and
// spikes are continuous mutable traits — so a mature pond has hundreds of live
// silhouettes against ~100 colours, thousands of keys, against a cache that
// holds 448. It overflows, wipes, and refills forever: 282 wipes in a few
// minutes on a *paused* grid-128 pond, which is the clean proof, since a frozen
// population cannot have a changing working set.
//
// Eviction does not save it either: when every key is drawn every frame every
// key is hot, and an LRU thrashes identically. The only real fix is to stop
// keying on colour — bake shape-only sprites and tint at blit time — and
// Canvas2D can only tint through per-agent composite switches, which is the
// exact cost the atlas exists to avoid.
//
// The grid ceiling is 64 (setup.js), where on most windows the LOD threshold is
// never reached anyway. Kept behind the key, not deleted, because the pipeline
// is sound and the measurements around it are worth not re-deriving; if the
// renderer ever moves to WebGL2 the tinting problem disappears and this becomes
// straightforwardly correct.
let sprites_enabled = false;
// Last frame's zoom, for the HUD only. Not smoothed — it is a setting, not a
// measurement, and an averaged one would lag the key you just pressed.
let perf_scale_px = 0;
const PERF_EMA = 0.1;
const perf = { water: 0, shimmer: 0, food: 0, agents: 0, frame: 0, sim: 0 };
function perf_mark(key, t0) {
    perf[key] += (performance.now() - t0 - perf[key]) * PERF_EMA;
}
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
    }, tunable_ranges(), cancel_setup);
    splash = initSplash(document.getElementById('splash'), {
        // The pond is already running behind the card; continuing only has to
        // get out of the way, and take the setup panel with it if it is up.
        onContinue: cancel_setup,
        onNewRun: open_setup,
    });
    // Wind the opening pond forward to where it has lineages in it. Started
    // here and finished a chunk per frame, so the card is readable and
    // clickable throughout — with the curtain down over it meanwhile.
    warm_remaining = WARM_START_STEPS;
    run_veil(true);
    god = initGod(document.getElementById('god'), {
        smiteRadius: (x, y, r) => world.smite_radius(x, y, r),
        smiteBand: (x0, x1) => world.smite_band(x0, x1),
        smiteAll: () => world.smite_all(),
        setImmortal: on => world.set_immortal(on),
        setAutomaticPredators: on => {
            automatic_predators = on;
            world.set_automatic_predators(on);
        },
        setDiseaseEnabled: on => world.set_disease_enabled(on),
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
                      'species-list', 'disease-panel']) {
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
    update_diseases = initDiseasePanel(
        document.getElementById('disease-panel'),
        id => species_rows.find(s => s.id === id)?.name ?? (id === 0 ? 'unassigned' : `species ${id}`),
    );
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
    // A run that was asked for starts where a run starts, and is watched from
    // the first tick: the warm start belongs to the opening pond only (see
    // WARM_START_STEPS). The curtain is not the warm start's — every run comes
    // up behind it — so it plays here too, without the welcome card.
    warm_remaining = 0;
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
    // A new pond has new lineages and new colours, so every cached sprite is
    // about to be an unreachable key. The stats go with it — a high-water mark
    // carried over from the last run would describe the wrong pond.
    resetAtlas();
    resetAtlasStats();
    water_step = -1;   // new fertility field, and the step counter restarts at 0
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

    // The old run's panels and floating windows describe a pond that no longer
    // exists — they go before the new ones are built.
    clear_run_panels();
    build_panels();
    reset_camera();
    close_setup();
    run_veil(false);
    hold_for_curtain();
}

/** Keep a freshly started run still until the curtain starts to lift, so its
 *  opening ticks happen in view rather than behind solid white. */
function hold_for_curtain() {
    const epoch = ++run_epoch;
    sim_running = false;
    setTimeout(() => {
        // Not if this run has been replaced, or the setup panel is up: both
        // mean the world this timer was started for is not the one on screen.
        if (epoch !== run_epoch || setup.isOpen()) return;
        sim_running = true;
    }, VEIL_HOLD_MS);
}

/** Open the setup panel and freeze the sim while it's up.
 *
 *  Freeze, not end. The panel used to tear the run down on the way in, on the
 *  grounds that a "new run" screen over a live pond is a lie about a sim that
 *  is merely paused — but that left no way back out of a panel you opened by
 *  mistake, and no way to change your mind. It closes now (`close`, Escape, or
 *  the opening card's continue button) and the pond picks up where it stopped.
 *  Starting a run from it still discards everything; that is what start does. */
function open_setup() {
    if (setup.isOpen()) return;
    splash?.hide();
    // The setup panel stops the world, so a warm start still in flight would
    // stall half-faded behind it. Abandon it, and lift the curtain — what is
    // behind this panel is the title scene, which needs no covering.
    warm_remaining = 0;
    clear_veil();
    sim_running = false;
    setup.show();
    document.getElementById('setup-banner').style.display = 'block';
}

/** Dismiss the setup panel without starting anything: the run that was frozen
 *  when it opened resumes. */
function cancel_setup() {
    if (!setup.isOpen()) return;
    setup.hide();
    document.getElementById('setup-banner').style.display = 'none';
    sim_running = true;
}

/** Empty every panel and floating window that describes a run, and stop the
 *  graph timer. Called on restart, before build_panels() refills them for the
 *  run that replaces it. */
function clear_run_panels() {
    // Announcements outlive their run otherwise: the toast queue is timed, and
    // a promotion from the pond you just discarded would scroll past over the
    // one that replaced it.
    // Guarded rather than called straight: a browser holding a cached
    // species.js from before `clear` existed would otherwise take the restart
    // down with a TypeError, and a stale toast is a much smaller problem than
    // a run that will not start.
    announce?.clear?.();
    announce = null;
    graphs_visible = false;
    if (graphs_timer) {
        clearInterval(graphs_timer);
        graphs_timer = null;
    }
    document.getElementById('graphs').style.display = 'none';
    closeFloatingPrefix('graph:');
    closeFloatingPrefix('species:');
    closeFloatingPrefix('tree:');
    closeFloatingPrefix('disease:');
    for (const id of ['legend-colors', 'legend-shapes', 'legend-tiles',
                      'legend-deaths', 'legend-composite', 'genome-panel', 'graphs',
                      'species-list', 'disease-panel']) {
        document.getElementById(id).innerHTML = '';
    }
    update_legend_counts = null;
    update_diseases = null;
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
        // Stirring permanently lowers tile fertility, and it is the one thing
        // that changes the water without the engine stepping — so on a paused
        // pond the cached water layer would not show the damage until the sim
        // was resumed. Invalidate it directly.
        water_step = -1;
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

    // Species membership. The trait block sits after the whole traced forward
    // pass plus energy, age and kills — derived from the engine's own layer
    // sizes rather than written down. It was hardcoded as [60..69), which after
    // the input vector widened was feeding age, kills and traits 0-6 into the
    // distance calculation.
    const agent = last_agents.find(a => a.id === selected_id);
    const sp = agent ? species_rows.find(s => s.id === agent.species) : null;
    if (sp) {
        const sizes = Array.from(brain_layer_sizes());
        const traits_at = sizes.reduce((a, b) => a + b, 0)   // inputs + hidden + logits
            + sizes[sizes.length - 1]                        // sigmoid gates
            + 3;                                             // energy, age, kills
        const traits = Array.from(buf.slice(traits_at));
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
    if (e.key === 'm' || e.key === 'M') toggle_perf();
    if (e.key === 'l' || e.key === 'L') toggle_sprites();
    // Zen while the setup panel is up would hide the only way to start a run.
    if ((e.key === 'c' || e.key === 'C') && !setup.isOpen()) toggle_zen();
    if (e.key === 'p' || e.key === 'P') toggle_phylogeny();
    // N toggles: the panel it opens is dismissable now, so the same key closes it.
    if (e.key === 'n' || e.key === 'N') setup.isOpen() ? cancel_setup() : open_setup();

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

    // Escape backs out of the setup panel — it freezes the run rather than
    // ending it, so there is something to go back to.
    if (e.key === 'Escape' && setup.isOpen()) { cancel_setup(); return; }
    if (e.key === 'Escape') deselect();
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
    // Every body is about to change colour, so every cached sprite is about to
    // become an unreachable key holding atlas space. Wipe rather than let the
    // overlay's palette and the genome palette share a 512-entry budget.
    resetAtlas();
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

/** Render timings, behind M. Off by default and gated at every call site, so an
 *  unopened HUD costs nothing — `performance.now()` twelve times a frame is not
 *  free, and a profiler that shows up in its own numbers is worse than none.
 *  The averages reset on open so a reading always describes the run you are
 *  looking at, not the frame the panel happened to be opened on. */
function toggle_perf() {
    perf_visible = !perf_visible;
    if (perf_visible) for (const k in perf) perf[k] = 0;
    document.getElementById('h-perf').style.display = perf_visible ? 'block' : 'none';
}

/** Sprite LOD on/off, behind L. Resets the perf averages on the way through:
 *  an EMA that straddles the switch is a reading of neither build. */
function toggle_sprites() {
    sprites_enabled = !sprites_enabled;
    for (const k in perf) perf[k] = 0;
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

// Hue registry: species id → [L, C, h]. Built from the roster in promotion
// order, so a genus takes the next free hue the first time one of its species
// is promoted and every later sibling shares it, varying by lightness and
// chroma. Order of first appearance is stable for a run, so a lineage's colour
// never changes under it.
const species_lch = new Map();
const genus_index = new Map();

function rebuild_species_colors(rows) {
    species_lch.clear();
    const within = new Map();
    for (const s of rows) {
        const genus = String(s.name).split(' ')[0];
        if (!genus_index.has(genus)) genus_index.set(genus, genus_index.size);
        const gi = genus_index.get(genus);
        const wi = within.get(genus) ?? 0;
        within.set(genus, wi + 1);
        species_lch.set(s.id, speciesColor(gi, wi));
    }
}

/** Swatch colour for a species row: the lineage's own hue. */
function species_swatch(s) {
    const lch = species_lch.get(s.id);
    if (!lch) return [104, 116, 124];
    return oklchToRgb(lch);
}

function refresh_species() {
    const flat = world.species_list();
    const stride = species_stride();
    species_rows = parseSpecies(flat, stride, world.species_names());
    rebuild_species_colors(species_rows);
    species_rows.forEach((s, i) => { s.index = i; });

    const assigned = new Set(species_rows.filter(s => s.extinctAt === null).map(s => s.id));
    const unassigned = last_agents.filter(a => !assigned.has(a.species)).length;
    update_species?.(species_rows, current_step, unassigned);
    // The tree reads the same roster, so it refreshes on the same tick rather
    // than polling for a change it cannot see. Diseases refresh here too: the
    // panel names species, so it wants the roster that was just decoded.
    refreshPhylogeny(phylogeny_source, species_swatch);
    update_diseases?.(
        parseDiseases(world.disease_list(), disease_stride(),
                      disease_species_columns(), world.disease_names()),
        current_step,
    );

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

    // While parameters are being chosen, the pond behind the panel is the
    // title scene — one creature and the wordmark — not the run itself. The run
    // is only frozen, and `close` gives it straight back; showing a still of it
    // under a panel that says "new run" reads as though it were already gone.
    if (!sim_running && setup.isOpen()) {
        draw_idle_scene(ts / 1000);
        return;
    }

    god.update(ts / 1000);

    const t_frame = performance.now();
    if (warm_remaining > 0) {
        run_warm_start_chunk();
    } else if (!paused && sim_running) {
        // `frame_delta` is already capped at 200 ms to avoid a spiral of death
        // after a tab-switch. The speed multiply moved engine-side: the substep
        // cap scales with the dial, so the engine has to see the dial rather
        // than a pre-multiplied number it cannot take apart.
        const t_sim = performance.now();
        world.update(frame_delta, speed_mult);
        if (perf_visible) perf_mark('sim', t_sim);
    }

    const buf = world.get_state();
    render(buf, ts / 1000);
    if (perf_visible) perf_mark('frame', t_frame);

    // Panels are observation. A panel that throws must not take the pond down
    // with it, so they are isolated from the render path.
    try {
        update_panels(buf[2] | 0);
    } catch (err) {
        report_frame_error(err);
    }
}

/** Drop the white curtain and fade it out. Every run starts behind it: the
 *  opening pond because it is being wound forward, a run started by hand
 *  because the first hundred ticks of a fresh pond are the least interesting
 *  it will ever be.
 *
 *  `reveal_card` brings the welcome card up when the curtain lifts, and is for
 *  the opening pond only — someone who just filled in the setup panel has
 *  answered everything the card asks. Belt and braces on the timing:
 *  `animationend` is the real signal, the timer is there so a browser that
 *  never fires it (a background tab at load, reduced motion) does not leave the
 *  card permanently hidden. */
function run_veil(reveal_card) {
    const veil = document.getElementById('veil');
    if (!veil) { if (reveal_card) splash?.show(); return; }

    // Restart the animation rather than assume it has never run: re-adding a
    // class the element already has does nothing at all.
    veil.classList.remove('fade');
    veil.style.opacity = '';
    void veil.offsetWidth;   // reflow, so the removal is committed first
    veil.classList.add('fade');

    if (!reveal_card) return;
    let shown = false;
    const reveal = () => {
        if (shown) return;
        shown = true;
        // Not if the run has already been taken over by hand — opening the
        // setup panel is a decision the card has nothing left to offer.
        if (!setup.isOpen()) splash?.show();
    };
    veil.addEventListener('animationend', reveal, { once: true });
    setTimeout(reveal, VEIL_MS + 250);
}

/** Lift the curtain at once, mid-fade or not. */
function clear_veil() {
    const veil = document.getElementById('veil');
    if (!veil) return;
    veil.classList.remove('fade');
    veil.style.opacity = '0';
}

/** One frame's worth of warm start: step the world until the budget is spent,
 *  Chunked rather than run in one go because 4,200 ticks is a second or two of wasm, and a page that stops
 *  painting for that long on load reads as one that has failed to load.
 *
 *  A pond that goes extinct mid-warm-start ends it early — `fast_forward`
 *  returns short, and there is nothing left to wind forward. */
function run_warm_start_chunk() {
    const budget_end = performance.now() + WARM_BUDGET_MS;
    // 25 at a time: enough that the per-call overhead is noise, short enough
    // that the budget is honoured to within a fraction of a frame.
    const chunk = 25;
    while (warm_remaining > 0 && performance.now() < budget_end) {
        const want = Math.min(chunk, warm_remaining);
        const ran = world.fast_forward(want);
        warm_remaining -= want;
        if (ran < want) { warm_remaining = 0; break; }   // extinct
    }
    if (warm_remaining <= 0) {
        warm_remaining = 0;
        splash?.setStatus(null);
        return;
    }
    splash?.setStatus('warming the pond');
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
// Unassigned creatures stay neutral in their body colour, but this cool-blue
// edge keeps them legible against the pond without borrowing the title card's
// acid-lime identity.
const UNASSIGNED_OUTLINE_RGB = [0x4D, 0xA3, 0xFF];
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

    let t0 = performance.now();
    draw_water(buf, n, L);
    if (perf_visible) { perf_mark('water', t0); t0 = performance.now(); }
    draw_shimmer(buf, n, L, time_sec);
    if (perf_visible) { perf_mark('shimmer', t0); t0 = performance.now(); }
    draw_food(buf, n, L, time_sec);
    if (perf_visible) { perf_mark('food', t0); t0 = performance.now(); }
    draw_agents(buf, n, alpha, L, time_sec);
    if (perf_visible) perf_mark('agents', t0);

    draw_dying(L, time_sec);
    draw_god_effects(L, time_sec);

    if (stir_active) draw_stir(L);

    ctx.restore();


    // HUD
    document.getElementById('h-step').textContent   = `step   ${step}`;
    document.getElementById('h-agents').textContent = `agents ${n}`;
    document.getElementById('h-energy').textContent = `energy ${avgE}`;
    document.getElementById('h-food').textContent   = `food   ${food}`;

    if (perf_visible) {
        // `frame` is the whole callback, so the gap between it and the four
        // spans is everything unattributed: get_state, the dying/god/stir
        // layers, and whatever rasterisation the browser deferred past the last
        // measured pass. A large gap is a finding, not an error.
        const rest = Math.max(0, perf.frame - perf.water - perf.shimmer - perf.food - perf.agents - perf.sim);
        const atlas_stat = atlasStats();
        const ms = v => v.toFixed(1).padStart(6);
        document.getElementById('h-perf').textContent =
            `sim     ${ms(perf.sim)}\n` +
            `water   ${ms(perf.water)}\n` +
            `shimmer ${ms(perf.shimmer)}\n` +
            `food    ${ms(perf.food)}\n` +
            `agents  ${ms(perf.agents)}\n` +
            `other   ${ms(rest)}\n` +
            `frame   ${ms(perf.frame)}  ${(1000 / Math.max(perf.frame, 0.01)).toFixed(0)} fps\n` +
            // The sprite line is what makes the L toggle readable: `drawn`
            // splits the population between the two pipelines, so a reading
            // where sprites are on but `drawn` is near zero is a threshold
            // problem, not an atlas problem.
            `sprite  ${sprites_enabled ? 'on ' : 'off'} drawn ${sprite_queue_len}/${sprite_queue_len + body_queue_len} ` +
            // `peak` against the cap is the whole question: if the working set
            // fits, `wipes` stops climbing and the atlas stops cycling. A rising
            // wipe count on a paused pond means the key space is still too wide,
            // not that the cache is too small.
            `atlas ${atlas_stat.entries}/${atlas_stat.peak} wipes ${atlas_stat.wipes}\n` +
            // `zoom` against the LOD threshold, because "sprites on, nothing
            // drawn" has two very different causes and no way to tell them
            // apart from the outside: over the threshold means zoom out, under
            // it means the atlas is broken.
            `zoom    ${perf_scale_px.toFixed(1)} px/tile  lod ≤${SPRITE_LOD_MAX_SCALE_PX}  ` +
            `×${cam.zoom.toFixed(2)}${cam.zoom <= MIN_ZOOM ? ' (fit)' : ''}\n` +
            // The grid line exists because px/tile at full zoom-out is
            // `min(W,H) / GRID` and nothing else — so a small pond can be as far
            // out as the camera goes and still be nowhere near the LOD
            // threshold. `fit` is the floor this run can reach: if that number
            // is above `lod`, no amount of zooming will engage sprites and the
            // run needs a bigger grid, not a different camera. That is exactly
            // the guess this line is here to stop.
            `grid    ${GRID}  canvas ${canvas.width}×${canvas.height}  ` +
            `fit ${fit_scale().toFixed(1)} px/tile` +
            (fit_scale() > SPRITE_LOD_MAX_SCALE_PX ? '  ← never reaches lod' : '');
    }
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
// Ceiling on the mid canvas's side length. At 8× per tile a 512×512 pond would
// ask for a 4096² canvas and blur it every frame, which is minutes-per-frame
// territory; the upscale to screen looks the same from a smaller one, since the
// pond is only ever a window's worth of pixels wide anyway.
const WATER_MID_MAX_PX = 1024;

/** Mid-canvas pixels per tile for the current grid — WATER_MID_SCALE until the
 *  grid is large enough that it would blow WATER_MID_MAX_PX. */
function water_mid_scale() {
    return Math.max(1, Math.min(WATER_MID_SCALE, Math.floor(WATER_MID_MAX_PX / GRID)));
}
const WATER_BLUR_PX = 2.2;

let terrain_canvas = null, terrain_ctx = null, terrain_img = null;
let water_mid = null, water_mid_ctx = null;
// Grid size the offscreen canvases above were built for. They used to be built
// once and kept forever, so starting a run on any grid other than the boot
// default wrote GRID² tiles into a 12×12 ImageData and the water came out
// truncated. Rebuilt whenever the run's grid size changes.
let terrain_grid = 0;
// Sim step the blurred mid-canvas was built from. The fertility field only
// changes when the engine steps, so rebuilding it per *frame* redid the same
// work up to 350 times a second — measured at 4.6 ms a frame in Firefox against
// 0.1 ms in Edge, because the blur filter is far more expensive on Firefox's
// Canvas2D backend. Rebuilding on step instead ties the cost to 20 Hz.
//
// -1 forces a rebuild: set on a new run, and used by the initial build below.
let water_step = -1;

function draw_water(buf, n, { tile_w, tile_h, off_x, off_y }) {
    if (!terrain_canvas || terrain_grid !== GRID) {
        terrain_grid = GRID;
        water_step = -1;

        terrain_canvas = document.createElement('canvas');
        terrain_canvas.width = GRID;
        terrain_canvas.height = GRID;
        terrain_ctx = terrain_canvas.getContext('2d');
        terrain_img = terrain_ctx.createImageData(GRID, GRID);

        water_mid = document.createElement('canvas');
        water_mid.width = GRID * water_mid_scale();
        water_mid.height = GRID * water_mid_scale();
        water_mid_ctx = water_mid.getContext('2d');
    }

    const mid_scale = water_mid_scale();
    const m = GRID * mid_scale;

    // Everything from here to the final upscale is fertility-only, so it is
    // redone on a step change and skipped otherwise. Stir damage and god-mode
    // salt both land through the engine, so both arrive with a step and are
    // picked up — the old comment claiming a per-frame rebuild was needed to
    // catch stir was wrong about the mechanism, and was written when the grid
    // was 12×12 and the whole pass was 144 pixels.
    if (water_step !== current_step) {
        water_step = current_step;
        rebuild_water_mid(buf, n, m, mid_scale);
    }

    // Pass 2: mid → pond. One drawImage, and the only part that depends on the
    // camera, so it stays per-frame.
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(water_mid, 0, 0, m, m, off_x, off_y, GRID * tile_w, GRID * tile_h);
}

/** Rasterise the fertility field and blur it into the mid canvas. */
function rebuild_water_mid(buf, n, m, mid_scale) {
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

    // Pass 1: GRID² → mid, blurred. This is the expensive half — the blur is
    // what costs 4.6 ms a frame in Firefox — and it is what the step check
    // above exists to skip.
    water_mid_ctx.clearRect(0, 0, m, m);
    water_mid_ctx.imageSmoothingEnabled = true;
    // Blur is specified in mid-canvas pixels, so it tracks the scale — a large
    // grid softens by the same fraction of a tile as a small one.
    water_mid_ctx.filter = `blur(${WATER_BLUR_PX * mid_scale / WATER_MID_SCALE}px)`;
    water_mid_ctx.drawImage(terrain_canvas, 0, 0, GRID, GRID, 0, 0, m, m);
    water_mid_ctx.filter = 'none';
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
// Rebuild rate for the caustic mask. Well under the frame rate on any machine
// that is running well, and the drift is slow enough that 30 Hz is
// indistinguishable from per-frame.
const SHIMMER_HZ = 30;
let shimmer_built_at = -1;
const SHIMMER_ALPHA = 0.5;

let caustic_canvas = null, caustic_ctx = null, caustic_img = null;

function draw_shimmer(buf, n, { tile_w, tile_h, off_x, off_y }, time_sec) {
    if (!caustic_canvas) {
        caustic_canvas = document.createElement('canvas');
        caustic_canvas.width = caustic_canvas.height = CAUSTIC_PX;
        caustic_ctx = caustic_canvas.getContext('2d');
        caustic_img = caustic_ctx.createImageData(CAUSTIC_PX, CAUSTIC_PX);
    }

    // Caustics drift; they cannot key off the sim step like the water does. But
    // they drift *slowly*, and regenerating 9,216 pixels of summed sine at 350
    // fps is work nobody can see. Capped at SHIMMER_HZ — the canvas is retained
    // between rebuilds and still composited every frame, so the layer is always
    // present and only its animation is quantised.
    //
    // Measured at 3.6 ms a frame in Firefox against 0.5 in Edge before this.
    if (time_sec - shimmer_built_at >= 1 / SHIMMER_HZ) {
        shimmer_built_at = time_sec;
        rebuild_caustics(buf, n, time_sec);
    }

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(caustic_canvas, 0, 0, CAUSTIC_PX, CAUSTIC_PX,
                  off_x, off_y, GRID * tile_w, GRID * tile_h);
    ctx.restore();
}

/** Redraw the caustic mask at `time_sec`. */
function rebuild_caustics(buf, n, time_sec) {
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

function draw_food(buf, n, { W, H, tile_w, tile_h, off_x, off_y }, time_sec) {
    if (!orb_tex) build_orb();
    const tile_base = HEADER_LEN + n * AGENT_STRIDE;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Only the tiles on screen. Whole-pond views of a small grid are unaffected
    // (the bounds land on the whole grid), but a 512×512 pond has 262,144 tiles
    // and up to three nodes each, and drawing the ones behind the viewport is
    // the difference between a frame and a stall. One tile of margin covers the
    // drift and radius of a node anchored just outside.
    const tx0 = Math.max(0, Math.floor((0 - off_x) / tile_w) - 1);
    const tx1 = Math.min(GRID, Math.ceil((W - off_x) / tile_w) + 1);
    const ty0 = Math.max(0, Math.floor((0 - off_y) / tile_h) - 1);
    const ty1 = Math.min(GRID, Math.ceil((H - off_y) / tile_h) + 1);

    for (let ty = ty0; ty < ty1; ty++) {
        for (let tx = tx0; tx < tx1; tx++) {
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
// Codes mirror CauseOfDeath::code() in world.rs. Both maps were missing 4
// (Smitten) and fell through to 'unknown'; 5 is Disease.
const EPITAPH = {
    0: ':/',      // Starvation — in a pond this full? really?
    1: '[RIP]',   // OldAge
    2: 'X_X',     // KilledInCombat
    3: 'X_X',     // EatenAlive
    4: '*',       // Smitten
    5: '///',     // Disease
};

const DEATH_CAUSE = {
    0: 'starved',
    1: 'old age',
    2: 'killed in combat',
    3: 'eaten alive',
    4: 'smitten',
    5: 'disease',
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

// Draw queues for the batched agent passes. Grown once and never shrunk, and
// the records are mutated in place rather than replaced — a pond that booms to
// 20,000 would otherwise allocate 20,000 objects a frame purely to describe work
// it is about to do, and the collector is already the suspected reason a busy
// pond stutters instead of merely running slow.
const body_queue = [];
let body_queue_len = 0;
const predator_queue = [];
let predator_queue_len = 0;

function queue_body(chain, spec, palette, base_r, a, glow, outline, time_sec) {
    let b = body_queue[body_queue_len];
    if (!b) {
        b = { chain: null, spec: null, palette: null, glow: null, outline: null,
              motion: { baseR: 0, energyNorm: 0, velX: 0, velY: 0, timeSec: 0 } };
        body_queue[body_queue_len] = b;
    }
    b.chain = chain; b.spec = spec; b.palette = palette;
    b.glow = glow; b.outline = outline;
    b.motion.baseR = base_r;
    b.motion.energyNorm = a.energyNorm;
    b.motion.velX = a.velX;
    b.motion.velY = a.velY;
    b.motion.timeSec = time_sec;
    body_queue_len++;
}

// Sprite draw queue. Same pooling discipline as the body queue, and flatter:
// a sprite record is five numbers and a reference, so this stays a fixed set of
// objects however far the pond booms.
const sprite_queue = [];
let sprite_queue_len = 0;

function queue_sprite(entry, x, y, cos, sin) {
    let s = sprite_queue[sprite_queue_len];
    if (!s) { s = { entry: null, x: 0, y: 0, cos: 1, sin: 0 }; sprite_queue[sprite_queue_len] = s; }
    s.entry = entry; s.x = x; s.y = y; s.cos = cos; s.sin = sin;
    sprite_queue_len++;
}

/** Shortest signed distance across the toroidal seam. */
function wrap_delta(d) {
    if (d >  GRID * 0.5) return d - GRID;
    if (d < -GRID * 0.5) return d + GRID;
    return d;
}

/** One sprite, rotated about the agent's head and scaled from atlas resolution
 *  to the current zoom.
 *
 *  `setTransform` rather than translate/rotate/scale inside a save/restore: it
 *  is one call instead of four and, unlike a composite change, it does not flush
 *  the canvas batch. The clip set by draw_agents is in device space and survives
 *  it. */
function blit_sprite(img, r, a, b, x, y) {
    ctx.setTransform(a, b, -b, a, x, y);
    ctx.drawImage(img, r.sx, r.sy, r.sw, r.sh, -r.px, -r.py, r.sw, r.sh);
}

/** One batched pass over the sprite queue. `which` is 'glow' or 'core'; the
 *  caller owns the composite mode, exactly as with the vector passes. */
function draw_sprites(which, { tile_w, tile_h, scale_px, off_x, off_y }) {
    if (sprite_queue_len === 0) return;
    const img = atlasCanvas();
    if (!img) return;

    const k = scale_px / ATLAS_PPT;
    const pond_w = GRID * tile_w, pond_h = GRID * tile_h;

    ctx.save();
    for (let q = 0; q < sprite_queue_len; q++) {
        const s = sprite_queue[q];
        const r = s.entry[which];
        const a = s.cos * k, b = s.sin * k;
        blit_sprite(img, r, a, b, s.x, s.y);

        // Seam copies, same job as body.js's wrap loop: a body near an edge has
        // to appear on the far side too. The margin is the sprite's own extent
        // at this zoom, so a sprite that cannot possibly cross pays one compare.
        const m = (r.sw > r.sh ? r.sw : r.sh) * k;
        const wx = (s.x - off_x < m) ? pond_w : (off_x + pond_w - s.x < m) ? -pond_w : 0;
        const wy = (s.y - off_y < m) ? pond_h : (off_y + pond_h - s.y < m) ? -pond_h : 0;
        if (wx) blit_sprite(img, r, a, b, s.x + wx, s.y);
        if (wy) blit_sprite(img, r, a, b, s.x, s.y + wy);
        if (wx && wy) blit_sprite(img, r, a, b, s.x + wx, s.y + wy);
    }
    ctx.restore();
}

function queue_predator(id, x, y, state) {
    let p = predator_queue[predator_queue_len];
    if (!p) { p = { id: 0, x: 0, y: 0, state: null }; predator_queue[predator_queue_len] = p; }
    p.id = id; p.x = x; p.y = y; p.state = state;
    predator_queue_len++;
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
    body_queue_len = 0;
    predator_queue_len = 0;
    sprite_queue_len = 0;

    // One decision for the whole frame. Per-agent LOD would put articulated and
    // frozen bodies side by side at the same size, which reads as some of the
    // animals having stopped moving.
    const use_sprites = sprites_enabled && scale_px <= SPRITE_LOD_MAX_SCALE_PX;
    perf_scale_px = scale_px;
    beginAtlasFrame();

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

        // Colour is the lineage's, not the strategy's. An agent takes its
        // species' hue; an unassigned one is near-colourless, so promotion
        // visibly confers an identity and a pond going from grey to coloured is
        // the speciation story at a glance.
        //
        // Not cached by id like it used to be: an agent's species can change
        // under it — membership is by nearest centroid and is recomputed every
        // cluster tick — and a cached colour would leave it wearing a lineage it
        // has drifted out of.
        const lch = species_lch.get(a.species) ?? unassignedColor();
        // Energy dims the body, but only down to 72% lightness — at the old 55%
        // floor a starving neon body desaturated into brown, which read as a
        // rendering fault rather than as a hungry creature.
        //
        // On the sprite path the dim is quantised to the same four steps the
        // atlas bakes at. Dimming continuously would smear one species across
        // ~16 colour keys, which is what made the atlas overflow and cycle even
        // on a paused pond. Four steps of lightness on a body a few pixels wide
        // is not a distinction anyone was reading anyway.
        const dim_e = use_sprites ? energyRep(a.energyNorm) : a.energyNorm;
        let palette = oklchToRgb([
            lch[0] * (0.72 + dim_e * 0.28),
            lch[1],
            lch[2],
        ]);
        // Archetype overlay. Temporary by design — toggling off restores the
        // trait-derived colour, which is the pond's default reading.
        if (archetypes_visible) {
            const arch = arch_color.get(a.brainCluster);
            if (arch) {
                const dim = 0.72 + dim_e * 0.28;
                palette = [arch[0] * dim | 0, arch[1] * dim | 0, arch[2] * dim | 0];
            }
        }
        last_agents.push({
            id: a.id, x: hx_w, y: hy_w,
            cluster: a.genomeCluster, brainCluster: a.brainCluster,
            species: a.species, rgb: palette,
        });
        // Strategy moved off hue and onto the halo when species took the hue.
        const glow = strategyGlow(a.morph);
        // Agents with no lineage wear a blue edge. Grey
        // alone read as something failing to render; an outline reads as a
        // creature waiting for a name, which is what it is.
        const outline = a.species === 0 ? UNASSIGNED_OUTLINE_RGB : null;
        // Creatures are the subject; food orbs were drawn ~4x their size and the
        // pond read as a field of green dots with specks swimming in it.
        const base_r = scale_px * (0.105 + a.energyNorm * 0.07 + a.morph.pointiness * 0.05);

        // Viewport cull. Off-screen agents still decode, still update their
        // chain, and still land in `last_agents` — the panels count the whole
        // population and the chain must not be stale when the camera comes back
        // — but they issue no draw calls, which is the only part that costs
        // ~8 µs each.
        //
        // This does nothing at fit zoom, where every agent is on screen by
        // definition. It is the zoomed-in case it pays for, and it explains a
        // measurement that looked backwards: zoomed 5× the pond cost 6.3 µs per
        // agent against 8.0 at fit, because the rasteriser was already rejecting
        // off-screen paths against the clip — cheaply, but not for free. This
        // stops them at the source.
        //
        // The margin covers the glow hull and a body length, so nothing pops in
        // at the edge. Seam copies are safe: `clamp_camera` keeps the view inside
        // the pond, so a body close enough to an edge to need its far-side copy
        // is on screen itself whenever both edges are.
        const sx_px = off_x + hx_w * tile_w;
        const sy_px = off_y + hy_w * tile_h;
        const cull_m = tile_w * 2 + base_r * 4;
        if (sx_px < -cull_m || sx_px > L.W + cull_m ||
            sy_px < -cull_m || sy_px > L.H + cull_m) {
            continue;
        }

        // Apex predators do not use the body pipeline at all — they are hard
        // geometric shapes, so they can never be mistaken for one of their prey.
        // Deferred to after the crowd so they stay on top of it: a hunter drawn
        // under its prey reads as background.
        if (predators.has(a.id)) {
            queue_predator(a.id, hx_w, hy_w, predators.get(a.id));
            continue;
        }

        // Sprite path. `combat` is recovered from the halo weight rather than
        // recomputed — strategyGlow derives both colour and weight from that one
        // scalar, so inverting the weight gets it back exactly and colour.js
        // keeps a single definition of what "combat" means.
        if (use_sprites) {
            const combat = (glow.weight - 0.35) / 0.65;
            const key = spriteKey(spec, palette, combat, a.energyNorm, outline !== null);
            const entry = key < 0 ? null
                : spriteFor(key, spec, palette, glow, outline, a.energyNorm);
            // A null entry means the frame's build budget is spent. That agent
            // takes the vector path this frame and gets its sprite on the next
            // one — a few slow bodies beat a stall or a hole in the pond.
            if (entry) {
                // Heading: velocity when there is any, else the chain's own
                // spine. A stationary agent still has an orientation, and
                // snapping it to +x would make every idle animal face east.
                let dx = a.velX, dy = a.velY;
                if (dx * dx + dy * dy < 1e-8 && chain.segs.length > 1) {
                    dx = wrap_delta(chain.segs[0].x - chain.segs[1].x);
                    dy = wrap_delta(chain.segs[0].y - chain.segs[1].y);
                }
                const len = Math.sqrt(dx * dx + dy * dy);
                const c = len > 1e-6 ? dx / len : 1;
                const s = len > 1e-6 ? dy / len : 0;
                queue_sprite(entry, off_x + hx_w * tile_w, off_y + hy_w * tile_h, c, s);
                continue;
            }
        }

        queue_body(chain, spec, palette, base_r, a, glow, outline, time_sec);
    }

    // Two batched passes instead of two per agent. Every body used to flip
    // `globalCompositeOperation` twice and save/restore once, and a composite
    // change flushes the canvas batch — at 5,600 agents that is ~11,300 flushes
    // a frame, which measured at ~24 µs per fill against a 3–10 µs cost for the
    // path itself. Now the mode is set twice, total.
    //
    // It does change the layering: every glow now sits behind every body,
    // where a glow used to sit behind its own body but in front of any body
    // drawn earlier. In a crowd that reads as one shared haze under the pond
    // rather than per-animal halos jostling for depth.
    const xform = { tile_w, tile_h, scale_px, off_x, off_y, gridSize: GRID };
    ctx.globalCompositeOperation = 'lighter';
    draw_sprites('glow', L);
    for (let q = 0; q < body_queue_len; q++) {
        const b = body_queue[q];
        drawBody(ctx, b.chain, b.spec, b.palette, xform, b.motion, b.glow, b.outline, PASS_GLOW);
    }
    ctx.globalCompositeOperation = 'source-over';
    draw_sprites('core', L);
    for (let q = 0; q < body_queue_len; q++) {
        const b = body_queue[q];
        drawBody(ctx, b.chain, b.spec, b.palette, xform, b.motion, b.glow, b.outline, PASS_CORE);
    }

    for (let q = 0; q < predator_queue_len; q++) {
        const p = predator_queue[q];
        draw_predator(p.id, p.x, p.y, L, time_sec, p.state);
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
