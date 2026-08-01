// Run-parameter panel: grid size, starting population, seed, and the three rule
// dials.
//
// The first three are exactly the arguments World::new takes. The dials — food
// regen, the hunt threshold, clustering k — are engine values that could be
// moved at any time, and deliberately are not: they are fixed for the life of a
// run, set here and applied once at construction.
//
// That is the point of putting them here. A pond whose rules changed halfway
// through is not one experiment, it is two halves of different ones, and no
// screenshot, stat graph or exported tree taken afterwards says which rules
// produced which part of it. Fixed at the start, the run parameters plus the
// seed reproduce it exactly, which is the property the seed exists for.

const LIMITS = {
    grid: { min: 6, max: 512, default: 24 },
    population: { min: 1, max: 5000, default: 150 },
};

// Below this many tiles a side a run is unlikely to last. Every pond overshoots
// its food in the first few hundred ticks and crashes into a bottleneck; on a
// small grid that bottleneck is a handful of animals, and a handful of animals
// drift to zero. Measured at 10,000 ticks: 12×12 survived roughly two thirds of
// seeds at any starting density, 14×14 and up survived all of them. Small ponds
// are still worth running — they are where a crash is legible — so this warns
// rather than clamps.
const FRAGILE_GRID = 14;

// Above this, expect the frame rate to drop — every agent is drawn as a full
// kinematic body, with no culling or instancing (see draw_agents).
const HEAVY_POPULATION = 1200;

// Above this many tiles a side the terrain passes start to cost real frame
// time: the water layer rebuilds a GRID×GRID image every frame, and food nodes
// are drawn per unit (culled to the viewport, so zooming in stays cheap while
// the whole-pond view is the expensive one). The engine itself is unbothered —
// 512×512 with 5,000 agents steps in ~6ms — so this warns rather than clamps.
const HEAVY_GRID = 160;

// The rule dials. Defaults and bounds come from the engine (tunable_ranges), so
// no number here is written down twice.
const DIALS = [
    {
        key: 'regen',
        label: 'food regen',
        step: 0.001,
        format: v => v.toFixed(3),
        blurb: 'How fast tiles grow food back, scaled by each tile’s own ' +
               'fertility. Higher means a richer pond and more agents feeding ' +
               'without moving; at zero nothing ever regrows and the pond runs ' +
               'down to the food it already has.',
    },
    {
        key: 'hunt',
        label: 'hunt threshold',
        step: 0.01,
        format: v => v.toFixed(2),
        blurb: 'How aggressive an agent must be before it hunts other agents ' +
               'instead of grazing. Lower turns more of the pond predatory; ' +
               'above the trait’s maximum of 1.05 nobody qualifies and ' +
               'agent-on-agent combat stops entirely. Summoned hunters are ' +
               'unaffected — they are not agents.',
    },
    {
        key: 'k',
        label: 'families (k)',
        step: 1,
        format: v => String(Math.round(v)),
        blurb: 'How many genome families the k-means pass sorts the pond into, ' +
               'for the family legend and as speciation’s input. This changes ' +
               'only how you read the pond, never how it behaves — more ' +
               'families means finer splits between lineages that are nearly ' +
               'alike.',
    },
];

/**
 * Build the setup panel and return a handle.
 *
 * @param {HTMLElement} root  container element (the #setup panel)
 * @param {(p: {grid: number, population: number, seed: bigint, dials: object}) => void} onStart
 * @param {{grid: number, population: number, seed: bigint}} defaults
 * @param {Float32Array} ranges  [default, min, max] × 3, from tunable_ranges()
 * @param {() => void} onCancel  back out without starting anything; the run
 *                               that was frozen when the panel opened resumes
 */
export function initSetup(root, onStart, defaults, ranges, onCancel) {
    const dials = DIALS.map((d, i) => ({
        ...d, def: ranges[i * 3], min: ranges[i * 3 + 1], max: ranges[i * 3 + 2],
    }));

    root.innerHTML =
        `<h2>new run</h2>` +
        row('grid', 'grid size', 'number', defaults.grid, `${LIMITS.grid.min}–${LIMITS.grid.max} tiles per side`) +
        row('population', 'population', 'number', defaults.population, `starting agents`) +
        row('seed', 'seed', 'text', String(defaults.seed), `same seed = same pond`) +
        `<h2>rules</h2>` +
        dials.map(dial_row).join('') +
        `<div class="setup-warn" id="setup-warn"></div>` +
        `<div class="setup-actions">` +
        `<button id="setup-random">random seed</button>` +
        `<button id="setup-defaults">default rules</button>` +
        `</div>` +
        `<div class="setup-actions">` +
        `<button id="setup-close">close</button>` +
        `<button id="setup-start" class="primary">start run</button>` +
        `</div>` +
        `<div class="legend-note">start run rebuilds the world from scratch — ` +
        `stats, families and the current pond are discarded. close leaves the ` +
        `current pond running.</div>`;

    const el = id => root.querySelector('#' + id);
    const grid_in = el('setup-grid');
    const pop_in = el('setup-population');
    const seed_in = el('setup-seed');
    const warn = el('setup-warn');
    const seed_note = el('setup-seed-note');

    function read() {
        const values = {};
        for (const d of dials) {
            const raw = parseFloat(el(`setup-${d.key}`).value);
            values[d.key] = Number.isFinite(raw)
                ? Math.min(d.max, Math.max(d.min, raw))
                : d.def;
        }
        return {
            grid: clamp(parseInt(grid_in.value, 10), LIMITS.grid),
            population: clamp(parseInt(pop_in.value, 10), LIMITS.population),
            seed: parse_seed(seed_in.value, defaults.seed),
            dials: values,
        };
    }

    /** Show the clamped values back, so a rejected entry is visible rather than
     *  silently replaced at start time. */
    function reflect() {
        const p = read();
        grid_in.value = p.grid;
        pop_in.value = p.population;
        seed_in.value = String(p.seed);
        for (const d of dials) {
            el(`setup-${d.key}`).value = p.dials[d.key];
            el(`setup-v-${d.key}`).textContent = d.format(p.dials[d.key]);
        }
        // Say plainly when the seed alone no longer describes the run: with a
        // dial moved, "same seed = same pond" needs the dials to match too.
        const tuned = dials.some(d => Math.abs(p.dials[d.key] - d.def) > 1e-6);
        seed_note.textContent = tuned
            ? 'same seed + same rules = same pond'
            : 'same seed = same pond';
        warn.textContent =
            p.population >= HEAVY_POPULATION
                ? `${p.population} agents will run slowly — every agent is drawn individually`
            : p.grid >= HEAVY_GRID
                ? `a ${p.grid}×${p.grid} pond draws slowly zoomed all the way out — ` +
                  `the sim keeps up, the terrain passes are what cost frames`
            : p.grid < FRAGILE_GRID
                ? `a ${p.grid}×${p.grid} pond usually goes extinct — too small to survive ` +
                  `the first crash`
            : '';
        return p;
    }

    for (const input of [grid_in, pop_in, seed_in]) {
        input.addEventListener('change', reflect);
        input.addEventListener('keydown', e => {
            e.stopPropagation();   // keep sim keybinds out of the text fields
            if (e.key === 'Enter') start();
        });
    }

    for (const d of dials) {
        el(`setup-${d.key}`).addEventListener('input', reflect);
    }

    el('setup-defaults').addEventListener('click', () => {
        for (const d of dials) el(`setup-${d.key}`).value = d.format(d.def);
        reflect();
    });

    el('setup-random').addEventListener('click', () => {
        seed_in.value = String(BigInt(Math.floor(Math.random() * 2 ** 32)));
        reflect();
    });

    function start() {
        onStart(reflect());
    }
    el('setup-start').addEventListener('click', start);
    // Backing out is not the same as starting a run with the current values —
    // it leaves the pond behind the panel exactly as it was.
    el('setup-close').addEventListener('click', () => onCancel?.());

    reflect();
    // Closed on load — the pond auto-starts and the splash card covers it. The
    // stylesheet says the same thing, but isOpen() reads the inline style, so
    // the initial state is set explicitly rather than left '' and read wrong.
    root.style.display = 'none';

    return {
        show() { root.style.display = 'block'; },
        hide() { root.style.display = 'none'; },
        toggle() {
            root.style.display = root.style.display === 'block' ? 'none' : 'block';
        },
        isOpen() { return root.style.display === 'block'; },
    };
}

function row(key, label, type, value, note) {
    return `<div class="setup-row">` +
        `<label for="setup-${key}">${label}</label>` +
        `<input id="setup-${key}" type="${type}" value="${value}" spellcheck="false">` +
        `</div><div class="setup-note" id="setup-${key}-note">${note}</div>`;
}

/** A dial: label, the value it currently reads, a slider, and an `ⓘ` that
 *  explains what it changes. The blurb opens on focus as well as hover, so it is
 *  reachable without a mouse. */
function dial_row(d) {
    return `<div class="setup-row dial">` +
        `<label for="setup-${d.key}">${d.label}</label>` +
        `<span class="tune-info" tabindex="0" role="note" ` +
              `aria-label="${d.label}: ${d.blurb}">i` +
            `<span class="tune-blurb">${d.blurb}</span>` +
        `</span>` +
        `<span class="tune-value" id="setup-v-${d.key}">${d.format(d.def)}</span>` +
        `</div>` +
        `<input class="dial-slider" id="setup-${d.key}" type="range" ` +
               `min="${d.min}" max="${d.max}" step="${d.step}" value="${d.format(d.def)}">`;
}

function clamp(v, { min, max, default: dflt }) {
    if (!Number.isFinite(v)) return dflt;
    return Math.min(max, Math.max(min, Math.round(v)));
}

/** Seeds are u64 on the Rust side, so they're carried as BigInt, not Number —
 *  past 2^53 a Number silently loses the low bits and the "same seed, same
 *  pond" guarantee with them. */
function parse_seed(text, fallback) {
    const cleaned = String(text).trim().replace(/[^0-9]/g, '');
    if (cleaned === '') return fallback;
    try {
        return BigInt(cleaned) % (2n ** 64n);
    } catch {
        return fallback;
    }
}
