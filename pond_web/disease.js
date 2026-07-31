// Pathogen roster — what is loose in the pond, and in whom.
//
// Diseases arrive with a lineage: a promoted species has a flat chance of
// having been carrying something, and it spreads by contact from there. Until
// this panel existed the only trace of an outbreak was a line on the death
// graph, which tells you that agents died without telling you what of, whose
// lineage it started in, or whether it has crossed into another.
//
// Burnt-out pathogens stay listed. A disease that killed its way through a
// lineage and ran out of hosts is part of the run's history in the same way an
// extinct species is, and dropping it would make the roster lie about what
// happened.

import { openFloating, updateFloating } from './floating.js';

// Row layout in the disease_list() buffer; mirrors DISEASE_STRIDE in wasm.rs.
const D_ID = 0;
const D_ORIGIN = 1;
const D_EMERGED = 2;
const D_SEVERITY = 3;
const D_CONTAGION = 4;
const D_DURATION = 5;       // ticks an unprotected agent stays ill
const D_JUMPED = 6;         // 1 once it has crossed into a second species
const D_CARRIERS = 7;
const D_BY_SPECIES = 8;     // disease_species_columns() entries; column 0 = unassigned

/** Parse the flat buffer into rows. */
export function parseDiseases(flat, stride, columns, names) {
    const n = stride > 0 ? Math.floor(flat.length / stride) : 0;
    const out = [];
    for (let i = 0; i < n; i++) {
        const off = i * stride;
        out.push({
            id: flat[off + D_ID] | 0,
            name: names?.[i] ?? `pathogen ${flat[off + D_ID] | 0}`,
            origin: flat[off + D_ORIGIN] | 0,
            emerged: flat[off + D_EMERGED] | 0,
            severity: flat[off + D_SEVERITY],
            contagion: flat[off + D_CONTAGION],
            duration: flat[off + D_DURATION] | 0,
            jumped: flat[off + D_JUMPED] > 0.5,
            carriers: flat[off + D_CARRIERS] | 0,
            bySpecies: Array.from(
                flat.slice(off + D_BY_SPECIES, off + D_BY_SPECIES + columns), v => v | 0),
        });
    }
    return out;
}

/**
 * Build the disease panel once and return an updater.
 *
 * @param {HTMLElement} root
 * @param {(id: number) => string} speciesName  species id → display name
 * @returns {(rows: Array, step: number) => void}
 */
export function initDiseasePanel(root, speciesName) {
    root.innerHTML = `
        <h2>pathogens</h2>
        <div data-live></div>
        <div class="sp-empty" data-empty>no disease in this pond</div>
        <div class="legend-note">emerged in a lineage &middot; carriers now &middot;
            severity is energy drained per tick</div>`;

    const liveEl = root.querySelector('[data-live]');
    const emptyEl = root.querySelector('[data-empty]');
    let latestRows = [];
    let latestStep = 0;

    root.onclick = event => {
        const row = event.target.closest('[data-disease-id]');
        if (!row) return;
        const d = latestRows.find(x => x.id === Number(row.dataset.diseaseId));
        if (d) openDisease(d, latestStep, speciesName);
    };

    return function update(rows, step) {
        latestRows = rows;
        latestStep = step;
        emptyEl.style.display = rows.length === 0 ? 'block' : 'none';

        liveEl.innerHTML = rows.map(d => {
            // Dead pathogens are struck through rather than dropped, like
            // extinct species in the roster next door.
            const gone = d.carriers === 0;
            return `
                <div class="legend-row sp-selectable${gone ? ' sp-dead' : ''}"
                     data-disease-id="${d.id}">
                    <span class="dz-dot${d.jumped ? ' jumped' : ''}"></span>
                    <span class="sp-name">${d.name}</span>
                    <span class="legend-count">${gone ? '—' : d.carriers}</span>
                </div>`;
        }).join('');

        for (const d of rows) {
            updateFloating(`disease:${d.id}`, body => renderDisease(body, d, step, speciesName));
        }
    };
}

function openDisease(d, step, speciesName) {
    openFloating({
        key: `disease:${d.id}`,
        title: d.name,
        className: 'species-window',
        render: body => renderDisease(body, d, step, speciesName),
    });
}

function renderDisease(body, d, step, speciesName) {
    const rows = d.bySpecies
        .map((count, i) => ({ count, label: i === 0 ? 'unassigned' : speciesName(i) }))
        .filter(r => r.count > 0)
        .sort((a, b) => b.count - a.count);

    const bar = (count) => {
        const frac = d.carriers > 0 ? (count / d.carriers) * 100 : 0;
        return `<div class="comp-track"><div class="comp-fill" ` +
               `style="left:0;width:${frac.toFixed(1)}%"></div></div>`;
    };

    body.innerHTML =
        `<div class="comp-head">${d.name}</div>` +
        `<div class="comp-row"><span class="k">emerged</span>` +
        `<span class="v">step ${d.emerged}</span></div>` +
        `<div class="comp-row"><span class="k">from</span>` +
        `<span class="v">${speciesName(d.origin)}</span></div>` +
        `<div class="comp-row"><span class="k">severity</span>` +
        `<span class="v">${d.severity.toFixed(3)}/tick</span></div>` +
        `<div class="comp-row"><span class="k">contagion</span>` +
        `<span class="v">${d.contagion.toFixed(3)}</span></div>` +
        `<div class="comp-row"><span class="k">illness</span>` +
        `<span class="v">${d.duration} ticks</span></div>` +
        `<div class="comp-row"><span class="k">age</span>` +
        `<span class="v">${step - d.emerged} steps</span></div>` +
        (d.jumped
            ? `<div class="comp-note dz-jumped">has jumped species — it is no longer ` +
              `anyone's disease in particular and spreads at full contagion to anything</div>`
            : `<div class="comp-note">still confined to its host lineage</div>`) +
        `<div class="comp-head" style="margin-top:8px">carriers — ${d.carriers}</div>` +
        (rows.length === 0
            ? `<div class="comp-note">nobody is carrying it. An outbreak ends when ` +
              `it runs out of new hosts faster than its carriers recover — it can ` +
              `come back if a carrier appears again.</div>`
            : rows.map(r =>
                `<div class="comp-row"><span class="k">${r.label}</span>` +
                bar(r.count) + `<span class="v">${r.count}</span></div>`).join('')) +
        `<div class="comp-note">severity drains energy, so deaths land as ` +
        `<em>disease</em> rather than starvation — but an outbreak still hits ` +
        `hardest where the pond is already hungry. Illness runs its length and ` +
        `ends; immunity both shortens it and blunts the drain, and recovery ` +
        `confers nothing — the same animal can catch it again.</div>`;
}
