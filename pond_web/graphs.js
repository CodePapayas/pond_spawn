// Live stat graphs — population, food, energy, lifespan band, deaths by cause.
//
// Restores the analytical read of the old Python matplotlib panel
// (legacy_python/cli/cli_sim_starter.py:plot_simulation_stats) as a live
// overlay. All series come from the engine-side ring buffer (pond_core/src/
// stats.rs) via world.stats_history(), so the browser holds no history of its
// own and a dropped frame loses nothing.
//
// Pure DOM + Canvas2D, redrawn on a 1 Hz timer rather than per frame — a
// population curve has nothing to say at 60 fps, and five canvases per frame is
// exactly the cost the renderer cannot absorb.

// Field offsets within one sample; mirrors StatSample::write_to in stats.rs.
const S_STEP = 0;
const S_ALIVE = 1;
const S_FOOD = 2;
const S_ENERGY = 3;
const S_MEDIAN_LIFESPAN = 4;
const S_MIN_AGE = 5;
const S_MAX_AGE = 6;
const S_DEATHS = 7;   // 5 consecutive; see DEATH_SERIES for the order

const PANEL_H = 74;
const CELL_GAP = 10;   // must match .graph-strip's gap in the stylesheet

// Death series, in `CauseOfDeath::code()` order — the same order the engine
// writes them, so index i here is cause i there. The legend panel codes causes
// as glyphs rather than colors, so there was nothing to inherit; these are
// chosen to stay distinguishable at a 1px stroke on the dark background.
const DEATH_SERIES = [
    { label: 'starve',  color: 'rgba(255, 80, 120, 0.95)' },
    { label: 'old age', color: 'rgba(120, 230, 140, 0.95)' },
    { label: 'combat',  color: 'rgba(120, 160, 255, 0.95)' },
    { label: 'eaten',   color: 'rgba(190, 120, 255, 0.95)' },
    { label: 'smitten', color: 'rgba(255, 60, 220, 0.95)' },
];

const PANELS = [
    { key: 'pop',      title: 'population', color: 'rgba(60, 220, 180, 0.95)' },
    { key: 'food',     title: 'food',       color: 'rgba(110, 230, 120, 0.95)' },
    { key: 'energy',   title: 'avg energy', color: 'rgba(255, 185, 0, 0.95)' },
    { key: 'lifespan', title: 'lifespan',   color: 'rgba(190, 150, 255, 0.95)' },
    { key: 'deaths',   title: 'deaths / interval', color: null },
];

/**
 * Build the graph panel once and return an updater.
 *
 * @param {HTMLElement} root  container element (the #graphs panel)
 * @returns {(data: {flat: Float32Array, stride: number, totals: Float32Array,
 *                   peak: number, startPop: number}) => void}
 */
export function initGraphs(root) {
    const strip = document.createElement('div');
    strip.className = 'graph-strip';
    root.appendChild(strip);

    const ctxs = {};
    const canvases = [];
    for (const p of PANELS) {
        const cell = document.createElement('div');
        cell.className = 'graph-cell';

        const head = document.createElement('div');
        head.className = 'graph-title';
        head.innerHTML = `<span>${p.title}</span><span class="graph-cur" data-cur="${p.key}"></span>`;

        const cv = document.createElement('canvas');
        cell.appendChild(head);
        cell.appendChild(cv);
        strip.appendChild(cell);
        ctxs[p.key] = cv.getContext('2d');
        canvases.push(cv);
    }

    const footer = document.createElement('div');
    footer.className = 'graph-footer';
    root.appendChild(footer);

    const deathKey = DEATH_SERIES
        .map(d => `<span style="color:${d.color}">${d.label}</span>`)
        .join(' · ');

    /** Size every canvas to its cell.
     *
     *  The cells flex to equal widths, so this is what makes the row fill the
     *  panel instead of leaving a gap at the right that reads as a sixth,
     *  missing graph. Backing-store pixels are set from the CSS width and the
     *  device pixel ratio, so the lines stay sharp on a HiDPI display. */
    function size_canvases() {
        const dpr = window.devicePixelRatio || 1;
        const total = strip.clientWidth;
        if (total <= 0) return false;
        const cell_w = Math.max(60, Math.floor((total - CELL_GAP * (PANELS.length - 1)) / PANELS.length));
        for (const cv of canvases) {
            cv.style.width = cell_w + 'px';
            cv.style.height = PANEL_H + 'px';
            cv.width = Math.round(cell_w * dpr);
            cv.height = Math.round(PANEL_H * dpr);
            const c = cv.getContext('2d');
            c.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        return true;
    }

    // The panel is display:none until first opened, so clientWidth is 0 at
    // build time; sizing is retried on every update, and re-run on resize.
    let sized = size_canvases();
    window.addEventListener('resize', () => { sized = size_canvases(); redraw(); });

    let last = null;

    function redraw() {
        if (last) update(last);
    }

    function update(payload) {
        last = payload;
        if (!sized) sized = size_canvases();
        const { flat, stride, totals, peak, startPop } = payload;
        const n = stride > 0 ? Math.floor(flat.length / stride) : 0;
        if (n === 0) {
            footer.innerHTML = '<span class="graph-note">no samples yet</span>';
            return;
        }

        const at = (i, field) => flat[i * stride + field];
        const series = field => {
            const out = new Array(n);
            for (let i = 0; i < n; i++) out[i] = at(i, field);
            return out;
        };

        const alive = series(S_ALIVE);
        const food = series(S_FOOD);
        const energy = series(S_ENERGY);
        const median = series(S_MEDIAN_LIFESPAN);
        const minAge = series(S_MIN_AGE);
        const maxAge = series(S_MAX_AGE);
        const deaths = DEATH_SERIES.map((_, d) => {
            const out = new Array(n);
            for (let i = 0; i < n; i++) out[i] = at(i, S_DEATHS + d);
            return out;
        });

        drawLine(ctxs.pop, [{ data: alive, color: PANELS[0].color }]);
        drawLine(ctxs.food, [{ data: food, color: PANELS[1].color }]);
        drawLine(ctxs.energy, [{ data: energy, color: PANELS[2].color }]);
        drawLine(
            ctxs.lifespan,
            [{ data: median, color: PANELS[3].color }],
            { band: { lo: minAge, hi: maxAge, color: 'rgba(190, 150, 255, 0.16)' } },
        );
        drawLine(ctxs.deaths, deaths.map((data, i) => ({ data, color: DEATH_SERIES[i].color })));

        setCur('pop', alive[n - 1]);
        setCur('food', food[n - 1]);
        setCur('energy', energy[n - 1].toFixed(0));
        setCur('lifespan', median[n - 1].toFixed(0));
        setCur('deaths', deaths.reduce((s, d) => s + d[n - 1], 0));

        const firstStep = at(0, S_STEP);
        const lastStep = at(n - 1, S_STEP);
        const totalDeaths = Array.from(totals).reduce((a, b) => a + b, 0);

        footer.innerHTML =
            `<span class="graph-note">steps ${firstStep | 0}–${lastStep | 0}` +
            ` &middot; start ${startPop} &middot; now ${alive[n - 1] | 0} &middot; peak ${peak}</span>` +
            `<span class="graph-note">deaths ${totalDeaths | 0} total &mdash; ` +
            DEATH_SERIES.map((d, i) =>
                `<span style="color:${d.color}">${d.label} ${totals[i] | 0}</span>`).join(' · ') +
            `</span>` +
            `<span class="graph-note">per-interval rates above; ${deathKey}</span>`;
    }

    function setCur(key, value) {
        const el = root.querySelector(`[data-cur="${key}"]`);
        if (el) el.textContent = value;
    }

    return update;
}

/**
 * Draw one panel: optional filled band behind, then one polyline per series.
 * Y autoscales to the window max with a floor, so an all-zero series draws flat
 * along the bottom instead of dividing by zero or filling the panel.
 */
function drawLine(ctx, seriesList, opts = {}) {
    // CSS pixels, not backing-store pixels: the context is pre-scaled by the
    // device pixel ratio, so drawing coordinates are in CSS space.
    const dpr = window.devicePixelRatio || 1;
    const W = ctx.canvas.width / dpr;
    const H = ctx.canvas.height / dpr;
    ctx.clearRect(0, 0, W, H);

    const n = seriesList[0]?.data.length ?? 0;
    if (n === 0) return;

    let max = 0;
    for (const s of seriesList) for (const v of s.data) if (v > max) max = v;
    if (opts.band) for (const v of opts.band.hi) if (v > max) max = v;
    if (max <= 0) max = 1;

    const pad = 3;
    const x = i => (n === 1 ? W / 2 : (i / (n - 1)) * (W - 2 * pad) + pad);
    const y = v => H - pad - (v / max) * (H - 2 * pad);

    // Baseline
    ctx.strokeStyle = 'rgba(60, 220, 180, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H - pad + 0.5);
    ctx.lineTo(W, H - pad + 0.5);
    ctx.stroke();

    if (opts.band) {
        const { lo, hi, color } = opts.band;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x(0), y(hi[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(hi[i]));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(x(i), y(lo[i]));
        ctx.closePath();
        ctx.fill();
    }

    for (const s of seriesList) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const px = x(i);
            const py = y(s.data[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    // Y-max label, top-left. No axis framework — at 168px wide, tick labels
    // would be unreadable, and unreadable ticks are worse than none.
    ctx.fillStyle = 'rgba(120, 160, 175, 0.65)';
    ctx.font = '9px "Courier New", monospace';
    ctx.fillText(formatMax(max), 3, 10);
}

function formatMax(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
    if (v >= 10) return String(Math.round(v));
    return v.toFixed(1);
}
