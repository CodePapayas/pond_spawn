// Run-parameter panel: grid size, starting population, seed.
//
// These are exactly the three arguments World::new takes, so they are the only
// things that can be set before a run without touching sim rules. Everything
// else about a pond (fertility layout, barren clusters, genomes, death ages) is
// derived from the seed, which is what makes a seed worth exposing at all: the
// same three numbers reproduce the same pond exactly.

const LIMITS = {
    grid: { min: 6, max: 64, default: 12 },
    population: { min: 1, max: 5000, default: 100 },
};

// Above this, expect the frame rate to drop — every agent is drawn as a full
// kinematic body, with no culling or instancing (see draw_agents).
const HEAVY_POPULATION = 1200;

/**
 * Build the setup panel and return a handle.
 *
 * @param {HTMLElement} root  container element (the #setup panel)
 * @param {(p: {grid: number, population: number, seed: bigint}) => void} onStart
 * @param {{grid: number, population: number, seed: bigint}} defaults
 */
export function initSetup(root, onStart, defaults) {
    root.innerHTML =
        `<h2>new run</h2>` +
        row('grid', 'grid size', 'number', defaults.grid, `${LIMITS.grid.min}–${LIMITS.grid.max} tiles per side`) +
        row('population', 'population', 'number', defaults.population, `starting agents`) +
        row('seed', 'seed', 'text', String(defaults.seed), `same seed = same pond`) +
        `<div class="setup-warn" id="setup-warn"></div>` +
        `<div class="setup-actions">` +
        `<button id="setup-random">random seed</button>` +
        `<button id="setup-start" class="primary">start run</button>` +
        `</div>` +
        `<div class="legend-note">restarting rebuilds the world from scratch — ` +
        `stats, families and the current pond are discarded</div>`;

    const el = id => root.querySelector('#' + id);
    const grid_in = el('setup-grid');
    const pop_in = el('setup-population');
    const seed_in = el('setup-seed');
    const warn = el('setup-warn');

    function read() {
        return {
            grid: clamp(parseInt(grid_in.value, 10), LIMITS.grid),
            population: clamp(parseInt(pop_in.value, 10), LIMITS.population),
            seed: parse_seed(seed_in.value, defaults.seed),
        };
    }

    /** Show the clamped values back, so a rejected entry is visible rather than
     *  silently replaced at start time. */
    function reflect() {
        const p = read();
        grid_in.value = p.grid;
        pop_in.value = p.population;
        seed_in.value = String(p.seed);
        warn.textContent = p.population >= HEAVY_POPULATION
            ? `${p.population} agents will run slowly — every agent is drawn individually`
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

    el('setup-random').addEventListener('click', () => {
        seed_in.value = String(BigInt(Math.floor(Math.random() * 2 ** 32)));
        reflect();
    });

    function start() {
        onStart(reflect());
    }
    el('setup-start').addEventListener('click', start);

    reflect();
    // The stylesheet opens the panel, but isOpen() reads the inline style, so
    // make the initial state explicit rather than leaving it '' and reading as
    // closed on the first frame.
    root.style.display = 'block';

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
        `</div><div class="setup-note">${note}</div>`;
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
