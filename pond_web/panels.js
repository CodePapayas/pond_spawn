// Right-side panels: color/shape legend and running average-genome display.
// Pure DOM + tiny sparkline canvases — nothing here touches the main render
// loop except the update functions called on sim-step boundaries.

const TRAIT_NAMES = [
    'vision', 'speed', 'metabolism', 'energy cap', 'mutation', 'repro cost',
    'attack', 'defense', 'aggression',
];
// energy_capacity and mutation_rate are locked (no mutation drift) — display
// them dimmed and exclude them from "dominant" highlighting.
const LOCKED = new Set([3, 4]);

const HISTORY_LEN = 120;   // samples; at 1 sample / 10 ticks ≈ 1200 steps

// ── Legend ────────────────────────────────────────────────────────────────────

const SHAPE_ROWS = [
    ['aggression', 'M2,8 L20,3 L30,8 L20,13 Z', 'pointed head, swept fins'],
    ['speed',      'M2,8 Q10,5 30,7 Q10,11 2,8 Z', 'long slender body'],
    ['defense',    'M6,8 a7,6 0 1,0 14,0 a7,6 0 1,0 -14,0 M8,8 a5,4 0 1,0 10,0', 'wide body, white rings'],
    ['attack',     'M8,8 L2,3 M8,8 L2,13 M8,8 a6,5 0 1,0 12,0 a6,5 0 1,0 -12,0', 'head spikes'],
    ['vision',     'M10,6 a2,2 0 1,0 0.1,0 M20,6 a2,2 0 1,0 0.1,0', 'big, wide-set eyes'],
    ['metabolism', 'M2,8 h4 l3,-5 l4,10 l3,-5 h14', 'glow pulse speed'],
    ['energy cap', 'M4,8 a8,5 0 1,0 16,0 a8,5 0 1,0 -16,0 M8,8 a12,7 0 1,0 0,0.1', 'belly bulge'],
    ['energy',     'M2,12 h8 v-3 h8 v-4 h8', 'body brightness'],
];

const TILE_ROWS = [
    ['rgba(80,255,160,0.8)',  'floating node = one food unit'],
    ['rgba(26,72,112,0.95)',  'lighter blue = fertile water'],
    ['rgba(9,15,28,0.95)',    'dark matte = barren desert'],
    ['rgba(150,225,255,0.5)', 'shimmer = caustics over fertile water'],
    ['rgba(100,200,255,0.6)', 'ripple ring = stir'],
];

// Matches EPITAPH in renderer.js and CauseOfDeath::code() in world.rs.
const DEATH_ROWS = [
    ['X_X',   'killed in combat / eaten'],
    ['[RIP]', 'died of old age'],
    [':/',    'starved'],
];

const COMPOSITE_TRAITS = [0, 1, 2, 5, 6, 7, 8];   // skip the two locked traits
const LAYER_NAMES = ['in→h0', 'h0→h1', 'h1→h2', 'h2→out'];

/**
 * Build legend DOM once.
 * `onFamily(i)` is called when a family row is clicked (null = deselected),
 * and should return that family's composite buffer from cluster_composite().
 * Returns updateCounts(counts, meanRgb).
 */
export function initLegend(clusterRgb, onFamily, bounds) {
    const colorBox = document.getElementById('legend-colors');
    const compBox = document.getElementById('legend-composite');
    let selected = null;

    const rows = clusterRgb.map(([r, g, b], i) => {
        const row = document.createElement('div');
        row.className = 'legend-row clickable';
        row.innerHTML =
            `<div class="legend-swatch" style="background:rgb(${r},${g},${b});color:rgb(${r},${g},${b})"></div>` +
            `<span>family ${i + 1}</span><span class="legend-count">—</span>`;
        row.addEventListener('click', () => {
            selected = selected === i ? null : i;
            for (const other of rows) other.row.classList.remove('selected');
            if (selected !== null) row.classList.add('selected');
            renderComposite();
        });
        colorBox.appendChild(row);
        return { row, count: row.querySelector('.legend-count'), swatch: row.querySelector('.legend-swatch') };
    });

    function renderComposite() {
        if (selected === null || !onFamily) { compBox.innerHTML = ''; return; }
        const c = onFamily(selected);
        if (!c || c.length === 0) {
            compBox.innerHTML = `<div class="comp-head">family ${selected + 1}</div>` +
                `<div class="comp-note">no living members</div>`;
            return;
        }
        // [0]=count, [1..10)=trait means, [10..19)=trait sd,
        // [19..23)=layer magnitude means, [23..27)=layer sd
        let html = `<div class="comp-head">family ${selected + 1} — ${c[0] | 0} members</div>`;
        for (const t of COMPOSITE_TRAITS) {
            const lo = bounds[t * 2], hi = bounds[t * 2 + 1];
            const mean = c[1 + t], sd = c[10 + t];
            const nm = Math.max(0, Math.min(1, (mean - lo) / (hi - lo)));
            const nsd = Math.max(0, Math.min(1, sd / (hi - lo)));
            const left = Math.max(0, (nm - nsd) * 100);
            html +=
                `<div class="comp-row"><span class="k">${TRAIT_NAMES[t]}</span>` +
                `<div class="comp-track">` +
                // Spread band drawn last so it stays visible over the mean bar.
                `<div class="comp-fill" style="left:0;width:${(nm * 100).toFixed(1)}%"></div>` +
                `<div class="comp-sd" style="left:${left}%;width:${Math.min(100 - left, nsd * 200)}%"></div>` +
                `</div><span class="v">${mean.toFixed(2)}</span></div>`;
        }
        const maxMag = Math.max(1e-6, ...LAYER_NAMES.map((_, l) => c[19 + l]));
        for (let l = 0; l < 4; l++) {
            const mag = c[19 + l];
            html +=
                `<div class="comp-row"><span class="k">${LAYER_NAMES[l]}</span>` +
                `<div class="comp-track">` +
                `<div class="comp-fill" style="left:0;width:${(mag / maxMag * 100).toFixed(1)}%"></div>` +
                `</div><span class="v">${mag.toFixed(3)}</span></div>`;
        }
        html += `<div class="comp-note">bar = family mean &middot; magenta band = spread (±1sd); ` +
                `a narrow band means the family has genetically converged. ` +
                `brain rows are mean |weight| per layer.</div>`;
        compBox.innerHTML = html;
    }

    const countEls = rows.map(r => r.count);

    const shapeBox = document.getElementById('legend-shapes');
    for (const [name, path, desc] of SHAPE_ROWS) {
        const row = document.createElement('div');
        row.className = 'legend-row';
        row.innerHTML =
            `<svg class="legend-glyph" viewBox="0 0 34 16">` +
            `<path d="${path}" fill="none" stroke="rgba(60,220,180,0.8)" stroke-width="1.3"/></svg>` +
            `<span>${name}</span><span class="legend-count">${desc}</span>`;
        shapeBox.appendChild(row);
    }

    const tileBox = document.getElementById('legend-tiles');
    for (const [color, desc] of TILE_ROWS) {
        const row = document.createElement('div');
        row.className = 'legend-row';
        row.innerHTML =
            `<div class="legend-swatch" style="background:${color};color:${color}"></div>` +
            `<span>${desc}</span>`;
        tileBox.appendChild(row);
    }

    const deathBox = document.getElementById('legend-deaths');
    for (const [glyph, desc] of DEATH_ROWS) {
        const row = document.createElement('div');
        row.className = 'legend-row';
        row.innerHTML = `<span class="legend-glyph-txt">${glyph}</span><span>${desc}</span>`;
        deathBox.appendChild(row);
    }

    return function updateCounts(counts, meanRgb) {
        for (let i = 0; i < countEls.length; i++) {
            countEls[i].textContent = counts[i] > 0 ? `×${counts[i]}` : '—';
            // Swatch tracks the family's live mean genome colour; families with
            // no members keep their seed colour rather than going black.
            const c = meanRgb && meanRgb[i];
            if (c) {
                const css = `rgb(${c[0]},${c[1]},${c[2]})`;
                rows[i].swatch.style.background = css;
                rows[i].swatch.style.color = css;
            }
        }
        renderComposite();
    };
}

// ── Average genome panel ──────────────────────────────────────────────────────

/**
 * Build the average-genome rows. `bounds` = flat [lo,hi]×9 from wasm
 * trait_bounds(). Returns update(means) — call on a sampling cadence
 * (every ~10 sim steps), not per frame.
 */
export function initGenomePanel(bounds) {
    const box = document.getElementById('genome-panel');
    const rows = TRAIT_NAMES.map((name, i) => {
        const row = document.createElement('div');
        row.className = 'trait-row';
        row.innerHTML =
            `<span class="trait-name${LOCKED.has(i) ? ' locked' : ''}">${name}</span>` +
            `<div class="trait-track"><div class="trait-fill"></div></div>` +
            `<canvas class="trait-spark" width="46" height="14"></canvas>` +
            `<span class="trait-val">—</span>`;
        box.appendChild(row);
        return {
            row,
            fill: row.querySelector('.trait-fill'),
            spark: row.querySelector('.trait-spark').getContext('2d'),
            val: row.querySelector('.trait-val'),
            history: [],
        };
    });

    return function update(means) {
        // Normalized position within trait bounds, per trait
        const norms = TRAIT_NAMES.map((_, i) => {
            const lo = bounds[i * 2], hi = bounds[i * 2 + 1];
            return Math.max(0, Math.min(1, (means[i] - lo) / (hi - lo)));
        });

        // Dominant = top 3 unlocked traits by distance above range midpoint
        const dominant = new Set(
            norms.map((v, i) => [v - 0.5, i])
                .filter(([d, i]) => d > 0 && !LOCKED.has(i))
                .sort((a, b) => b[0] - a[0])
                .slice(0, 3)
                .map(([, i]) => i)
        );

        for (let i = 0; i < rows.length; i++) {
            const r = rows[i];
            r.fill.style.width = `${(norms[i] * 100).toFixed(1)}%`;
            r.val.textContent = means[i].toFixed(2);
            r.row.classList.toggle('dominant', dominant.has(i));

            r.history.push(norms[i]);
            if (r.history.length > HISTORY_LEN) r.history.shift();
            drawSpark(r.spark, r.history, dominant.has(i));
        }
    };
}

function drawSpark(ctx, history, dominant) {
    const W = 46, H = 14;
    ctx.clearRect(0, 0, W, H);
    if (history.length < 2) return;
    ctx.strokeStyle = dominant ? 'rgba(255,185,0,0.8)' : 'rgba(60,220,180,0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
        const x = (i / (HISTORY_LEN - 1)) * W;
        const y = H - 1 - history[i] * (H - 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}
