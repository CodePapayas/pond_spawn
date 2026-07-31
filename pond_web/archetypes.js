// Behavioural archetype view — k-means over the 488 brain weights, made visible.
//
// Genome clusters answer *who is related*. Species answer *which lineages locked
// in*. Brain clusters answer a third question — **who behaves alike** — and it is
// the one worth looking at, because nothing forces behaviour to follow ancestry.
// Two unrelated lineages converging on the same strategy, or one lineage hedging
// across several, are both invisible without this.
//
// Two rules the panel follows:
//
//  1. It is an *overlay*, not a change of palette. Default agent colour is
//     trait-derived through the three ramps (see color.js) and stays that way;
//     toggling recolours temporarily and toggling off restores it. Label-keyed
//     colour was deliberately abandoned once before and this does not undo that.
//
//  2. Archetypes are not named. They are unlabelled directions in 488-dimensional
//     weight space; calling one "forager" would assert an interpretation the data
//     does not carry. Size, colour, and the cross-tab are claims that hold.

/// What the panel is, in one paragraph, for the `i` popover.
const KEY_TEXT =
    'Every agent has a small neural network. This clusters those networks by ' +
    'their weights, so each group is a set of agents that would react to the ' +
    'same situation the same way. It is deliberately unrelated to genes or ' +
    'family: two unrelated lineages can end up in one group, and one lineage ' +
    'can spread across several. The groups have no names because they are ' +
    'directions in weight space — calling one "forager" would claim an ' +
    'interpretation the data does not support. While this panel is open the ' +
    'agents in the pond are tinted by group, so you can see where each one ' +
    'swims. Clustering is off until you open it: it is the most expensive ' +
    'thing in the simulation.';

/// Letters for the groups, so the grid can be read out loud.
const LETTERS = 'ABCDEFGH';

// Clusters coloured individually. k is 24 in the engine, and 24 hues sit ~15°
// apart, which reads as noise — so the largest few are coloured and the rest are
// pooled into one grey row whose size is still shown.
const COLOURED = 8;

// Golden angle: maximal separation for any count, deterministic, and stable per
// cluster id. A hash would let two archetypes land 5° apart.
const GOLDEN_ANGLE = 137.507764;
const TAIL_RGB = [104, 116, 124];

/** Stable colour for one archetype rank. `rank` is 0-based, by size. */
export function archetypeColor(rank) {
    if (rank >= COLOURED) return TAIL_RGB;
    const h = (rank * GOLDEN_ANGLE) % 360;
    return hslToRgb(h, 0.62, 0.60);
}

function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    if (h < 60)       { r = c; g = x; }
    else if (h < 120) { r = x; g = c; }
    else if (h < 180) { g = c; b = x; }
    else if (h < 240) { g = x; b = c; }
    else if (h < 300) { r = x; b = c; }
    else              { r = c; b = x; }
    return [(r + m) * 255 | 0, (g + m) * 255 | 0, (b + m) * 255 | 0];
}

/**
 * Rank archetypes by size and build the cross-tab against lineage.
 *
 * @param {Array} agents  decoded agents, needing `brainCluster` and `lineage`
 * @returns {{ranks: Map<number,number>, rows: Array, matrix: Array, total: number}}
 */
export function summarize(agents) {
    const counts = new Map();
    for (const a of agents) {
        counts.set(a.brainCluster, (counts.get(a.brainCluster) || 0) + 1);
    }
    // Rank by size, breaking ties on cluster id so the order is stable frame to
    // frame — otherwise equal-sized archetypes would swap colours as they jitter.
    const ordered = [...counts.entries()].sort((p, q) => q[1] - p[1] || p[0] - q[0]);

    const ranks = new Map();
    ordered.forEach(([id], i) => ranks.set(id, i));

    const rows = ordered.slice(0, COLOURED).map(([id, n], i) => ({
        id, count: n, rank: i, rgb: archetypeColor(i),
    }));
    const tail = ordered.slice(COLOURED).reduce((s, [, n]) => s + n, 0);

    // Cross-tab: lineage → per-archetype counts, tail pooled into one column.
    const byLineage = new Map();
    for (const a of agents) {
        if (!byLineage.has(a.lineage)) {
            byLineage.set(a.lineage, new Array(COLOURED + 1).fill(0));
        }
        const rank = ranks.get(a.brainCluster);
        byLineage.get(a.lineage)[Math.min(rank, COLOURED)] += 1;
    }
    const matrix = [...byLineage.entries()]
        .map(([lineage, cells]) => ({
            lineage, cells, total: cells.reduce((s, c) => s + c, 0),
        }))
        .sort((p, q) => q.total - p.total);

    return { ranks, rows, tail, matrix, total: agents.length };
}

/**
 * Build the panel once and return an updater.
 *
 * @param {HTMLElement} root  container element (the #archetypes panel)
 * @returns {(agents: Array) => void}
 */
export function initArchetypes(root) {
    root.innerHTML =
        `<div class="arch-title">behavioural archetypes` +
        `<span class="tune-info" tabindex="0" role="note" aria-label="${KEY_TEXT}">i` +
        `<span class="tune-blurb">${KEY_TEXT}</span></span></div>` +
        `<div class="arch-key">Each row is a group of agents whose <b>brains</b> are ` +
        `similar — not their genes, not their family. Bar = share of the pond. ` +
        `Letters name the groups so the grid below can refer to them; the colours ` +
        `are the same ones the agents are tinted with while this panel is open.</div>` +
        `<div class="arch-note" data-note></div>` +
        `<div data-list></div>` +
        `<div class="arch-title arch-title-sub">archetype &times; lineage</div>` +
        `<div class="arch-key">One row per lineage, one column per group above. ` +
        `Brighter = more of that lineage behaves that way. A row split across ` +
        `several columns is a lineage hedging; the same column bright on two ` +
        `unrelated rows is convergent evolution — two families that arrived at ` +
        `the same strategy independently.</div>` +
        `<div class="arch-note" data-matrix-note></div>` +
        `<div data-matrix></div>`;

    const noteEl = root.querySelector('[data-note]');
    const listEl = root.querySelector('[data-list]');
    const mNoteEl = root.querySelector('[data-matrix-note]');
    const matrixEl = root.querySelector('[data-matrix]');

    return function update(agents) {
        if (!agents || agents.length === 0) {
            noteEl.textContent = 'no agents';
            listEl.innerHTML = '';
            mNoteEl.textContent = '';
            matrixEl.innerHTML = '';
            return;
        }

        const { rows, tail, matrix, total } = summarize(agents);

        if (rows.length === 0) {
            // Clustering is on but the first pass has not finished yet.
            noteEl.textContent = 'clustering…';
            listEl.innerHTML = '';
            return;
        }
        noteEl.textContent = `${total} agents`;

        const pct = n => ((n / total) * 100).toFixed(0);
        const swatch = rgb => `background: rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;

        listEl.innerHTML = rows.map((r, i) => `
            <div class="arch-row">
                <span class="arch-letter">${LETTERS[i] ?? '·'}</span>
                <span class="arch-swatch" style="${swatch(r.rgb)}"></span>
                <span class="arch-bar"><i style="${swatch(r.rgb)};width:${pct(r.count)}%"></i></span>
                <span class="arch-n">${r.count}</span>
                <span class="arch-pct">${pct(r.count)}%</span>
            </div>`).join('')
            + (tail > 0 ? `
            <div class="arch-row arch-tail">
                <span class="arch-letter">·</span>
                <span class="arch-swatch" style="${swatch(TAIL_RGB)}"></span>
                <span class="arch-bar"><i style="${swatch(TAIL_RGB)};width:${pct(tail)}%"></i></span>
                <span class="arch-n">${tail}</span>
                <span class="arch-pct">${pct(tail)}%</span>
            </div>` : '');

        // Read a row across to see whether a lineage converged on one strategy or
        // hedges; read a column down to see unrelated lineages that found the
        // same one. The second is convergent evolution, and it is the whole
        // reason this panel exists.
        mNoteEl.textContent = matrix.length <= 1
            ? 'one lineage — nothing to compare yet'
            : 'row = lineage, column = archetype';

        const cols = rows.length + (tail > 0 ? 1 : 0);
        // Column headers, so a cell can be identified without counting across.
        const header = `<div class="arch-mrow arch-mhead"><span class="arch-mlabel"></span>` +
            Array.from({ length: cols }, (_, i) =>
                `<span class="arch-letter">${i < rows.length ? (LETTERS[i] ?? '·') : '·'}</span>`
            ).join('') + `</div>`;
        matrixEl.innerHTML = header + matrix.map(m => {
            const cells = m.cells.slice(0, cols).map((c, i) => {
                const rgb = i < rows.length ? rows[i].rgb : TAIL_RGB;
                // Alpha by share *within the lineage*: the question is how this
                // lineage splits, not how big it is next to the others.
                const a = m.total > 0 ? (c / m.total) : 0;
                return `<span class="arch-cell" title="${c}"
                    style="background: rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a.toFixed(3)})"></span>`;
            }).join('');
            return `<div class="arch-mrow"><span class="arch-mlabel">${m.lineage}</span>${cells}</div>`;
        }).join('');
    };
}
