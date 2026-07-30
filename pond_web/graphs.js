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
import { openFloating, updateFloating } from './floating.js';

// Field offsets within one sample; mirrors StatSample::write_to in stats.rs.
const S_STEP = 0;
const S_ALIVE = 1;
const S_FOOD = 2;
const S_ENERGY = 3;
const S_MEDIAN_LIFESPAN = 4;   // per-interval, not cumulative
const S_AGE_P10 = 5;
const S_AGE_P90 = 6;
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
    { label: 'disease', color: 'rgba(180, 230, 60, 0.95)' },
];

// `log: true` panels autoscale logarithmically. Counts in this sim are spiky by
// nature — an extinction event or a predator cull puts a single interval an order
// of magnitude above the rest of the run, and on a linear axis that one spike
// sets the maximum and flattens everything else into the baseline. Log keeps the
// spike on screen and the quiet stretches readable at the same time.
//
// `avg energy` stays linear: it is bounded at ~100 by capacity, has no spikes to
// absorb, and log would squash exactly the mid-range where the interesting
// movement is.
const PANELS = [
    { key: 'pop',      title: 'population', color: 'rgba(60, 220, 180, 0.95)', log: true },
    { key: 'food',     title: 'food',       color: 'rgba(110, 230, 120, 0.95)', log: true },
    { key: 'energy',   title: 'avg energy', color: 'rgba(255, 185, 0, 0.95)' },
    { key: 'lifespan', title: 'lifespan p10\u2013p90', color: 'rgba(190, 150, 255, 0.95)', log: true },
    { key: 'deaths',   title: 'deaths / interval', color: null, log: true },
];

/** Log scaling is on by default and can be switched off — comparing two heights
 *  directly is a linear question, and a reader doing that deserves a linear
 *  axis rather than a mental antilog. */
let logScale = true;

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
    let expanded = null;
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
        cell.addEventListener('mouseenter', e => {
            if (!expanded) return;
            showHover(p, e);
        });
        cell.addEventListener('mousemove', positionHover);
        cell.addEventListener('mouseleave', hideHover);
        cell.addEventListener('click', () => {
            if (!expanded) return;
            openFloating({
                key: `graph:${p.key}`,
                title: p.title,
                className: 'graph-window',
                render: body => renderExpanded(body, p, expanded),
            });
        });
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
        const ageP10 = series(S_AGE_P10);
        const ageP90 = series(S_AGE_P90);
        const deaths = DEATH_SERIES.map((_, d) => {
            const out = new Array(n);
            for (let i = 0; i < n; i++) out[i] = at(i, S_DEATHS + d);
            return out;
        });
        expanded = { alive, food, energy, median, ageP10, ageP90, deaths, n };

        const scaleOf = key => ({ log: logScale && !!PANELS.find(p => p.key === key)?.log });
        drawLine(ctxs.pop, [{ data: alive, color: PANELS[0].color }], scaleOf('pop'));
        drawLine(ctxs.food, [{ data: food, color: PANELS[1].color }], scaleOf('food'));
        drawLine(ctxs.energy, [{ data: energy, color: PANELS[2].color }], scaleOf('energy'));
        drawLine(
            ctxs.lifespan,
            [{ data: median, color: PANELS[3].color }],
            {
                ...scaleOf('lifespan'),
                band: { lo: ageP10, hi: ageP90, color: 'rgba(190, 150, 255, 0.16)' },
            },
        );
        drawLine(ctxs.deaths,
            deaths.map((data, i) => ({ data, color: DEATH_SERIES[i].color })),
            scaleOf('deaths'));
        for (const p of PANELS) {
            updateFloating(`graph:${p.key}`, body => renderExpanded(body, p, expanded));
        }

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
            `<span class="graph-note">per-interval rates above; ${deathKey}` +
            ` &middot; <button class="graph-scale" data-scale>` +
            `${logScale ? 'log' : 'linear'} scale</button></span>`;

        footer.querySelector('[data-scale]')?.addEventListener('click', () => {
            logScale = !logScale;
            redraw();
        });
    }

    function setCur(key, value) {
        const el = root.querySelector(`[data-cur="${key}"]`);
        if (el) el.textContent = value;
    }

    return update;

    function showHover(panel, event) {
        let hover = document.getElementById('graph-hover-preview');
        if (!hover) {
            hover = document.createElement('div');
            hover.id = 'graph-hover-preview';
            document.body.appendChild(hover);
        }
        hover.style.display = 'block';
        renderExpanded(hover, panel, expanded, true);
        positionHover(event);
    }

    function positionHover(event) {
        const hover = document.getElementById('graph-hover-preview');
        if (!hover || hover.style.display === 'none') return;
        const gap = 14;
        const w = hover.offsetWidth, h = hover.offsetHeight;
        hover.style.left = `${Math.max(8, Math.min(event.clientX + gap, window.innerWidth - w - 8))}px`;
        hover.style.top = `${Math.max(8, Math.min(event.clientY - h - gap, window.innerHeight - h - 8))}px`;
    }

    function hideHover() {
        const hover = document.getElementById('graph-hover-preview');
        if (hover) hover.style.display = 'none';
    }
}

function renderExpanded(body, panel, data, transient = false) {
    if (!data) return;
    body.innerHTML = transient ? `<div class="float-title">${panel.title}</div>` : '';
    const cv = document.createElement('canvas');
    cv.className = 'graph-expanded';
    body.appendChild(cv);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.min(520, Math.max(300, window.innerWidth - 48));
    const height = 220;
    cv.style.width = `${width}px`;
    cv.style.height = `${height}px`;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);

    const log = logScale && !!panel.log;
    if (panel.key === 'pop') drawLine(c, [{ data: data.alive, color: panel.color }], { log });
    if (panel.key === 'food') drawLine(c, [{ data: data.food, color: panel.color }], { log });
    if (panel.key === 'energy') drawLine(c, [{ data: data.energy, color: panel.color }], { log });
    if (panel.key === 'lifespan') {
        drawLine(c, [{ data: data.median, color: panel.color }], {
            log,
            band: { lo: data.ageP10, hi: data.ageP90, color: 'rgba(190, 150, 255, 0.16)' },
        });
    }
    if (panel.key === 'deaths') {
        drawLine(c, data.deaths.map((series, i) => ({
            data: series, color: DEATH_SERIES[i].color,
        })), { log });
        const key = document.createElement('div');
        key.className = 'graph-expanded-key';
        key.innerHTML = DEATH_SERIES
            .map(d => `<span style="color:${d.color}">${d.label}</span>`).join(' · ');
        body.appendChild(key);
    }
}

/**
 * Draw one panel: optional filled band behind, then one polyline per series.
 * Y autoscales to the window max with a floor, so an all-zero series draws flat
 * along the bottom instead of dividing by zero or filling the panel.
 *
 * `opts.log` switches the y axis to log. It is log1p, not log: these series are
 * counts that legitimately hit zero — no deaths in an interval, a pond with no
 * food left — and log(0) is negative infinity. log1p maps 0 to 0, keeps the
 * mapping monotone, and costs only a mild compression of the range 0–1, where
 * counts are not interesting anyway. Decade gridlines are drawn so the axis
 * cannot be mistaken for linear.
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
    const span = opts.log ? Math.log1p(max) : max;
    const norm = v => (opts.log ? Math.log1p(Math.max(0, v)) : Math.max(0, v)) / span;
    const y = v => H - pad - norm(v) * (H - 2 * pad);

    // Baseline
    ctx.strokeStyle = 'rgba(60, 220, 180, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H - pad + 0.5);
    ctx.lineTo(W, H - pad + 0.5);
    ctx.stroke();

    // Decade gridlines. On a log axis these are what show the compression is
    // there — without them a reader measures two peaks against each other and
    // gets the ratio badly wrong.
    if (opts.log) {
        ctx.strokeStyle = 'rgba(120, 160, 175, 0.16)';
        ctx.lineWidth = 1;
        for (let decade = 10; decade <= max; decade *= 10) {
            const py = Math.round(y(decade)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(W, py);
            ctx.stroke();
        }
    }

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
    // would be unreadable, and unreadable ticks are worse than none. The scale
    // is named, though: a log axis that does not say so is a lie about the
    // shape of the curve.
    ctx.fillStyle = 'rgba(120, 160, 175, 0.65)';
    ctx.font = '9px "Courier New", monospace';
    ctx.fillText(formatMax(max) + (opts.log ? ' log' : ''), 3, 10);
}

function formatMax(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
    if (v >= 10) return String(Math.round(v));
    return v.toFixed(1);
}
