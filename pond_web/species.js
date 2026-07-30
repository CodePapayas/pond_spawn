// Species roster — the promoted lineages, live and extinct.
//
// This replaces "family N" as the pond's headline grouping. A k-means family is
// a slot that exists because k = 6 was chosen; it always shows six rows whether
// or not the population has six distinct groups, and family 3 at step 200 shares
// nothing with family 3 at step 2000 but the number. A species is a lineage that
// held its shape and its share, survived having its mutability clamped, and
// earned a name it keeps until it dies (see species.rs, RULES.md).
//
// Species are therefore sparse and meaningful: a young pond has none, and a
// settled one has a handful. Everything not in a species is one grey
// "unassigned" row — which is honest, and is most of the pond most of the time.

import { openFloating, updateFloating } from './floating.js';

// Row layout in the species_list() buffer; mirrors SPECIES_STRIDE in wasm.rs.
const S_ID = 0;
const S_MEMBERS = 1;
const S_PEAK = 2;
const S_FOUNDED = 3;
const S_EXTINCT = 4;      // -1 while alive — Float32Array has no null
const S_FOUNDER_MEMBERS = 5;
const S_FOUNDER_POPULATION = 6;
const S_FOUNDER_GENERATION = 7;
const S_STREAK = 8;
const S_DRIFT = 9;
const S_SPREAD = 10;
const S_ENTRY_GENERATIONS = 11;
const S_PROBATION_GENERATIONS = 12;
// The species this one split from — nearest kin at promotion, 0 for a root.
// Same lookup that lends the genus, so name and ancestry always agree.
const S_PARENT = 13;
const S_CENTROID = 14;
const SIG_LEN = 7;
const S_FOUNDING_CENTROID = S_CENTROID + SIG_LEN;
const S_POPULATION_CENTROID = S_FOUNDING_CENTROID + SIG_LEN;
const SIGNATURE_NAMES = [
    'vision', 'speed', 'metabolism', 'reproduction cost',
    'attack', 'defense', 'aggression',
];
const SIGNATURE_DIMS = [0, 1, 2, 5, 6, 7, 8];

/** How long an extinction stays in the live list, struck through, before it
 *  drops into the collapsed fossil record. Long enough to notice, short enough
 *  not to clutter. */
const EXTINCT_LINGER_MS = 30_000;

/** Parse the flat buffer into rows, newest first among the living. */
export function parseSpecies(flat, stride, names) {
    const n = stride > 0 ? Math.floor(flat.length / stride) : 0;
    const out = [];
    for (let i = 0; i < n; i++) {
        const off = i * stride;
        const extinct = flat[off + S_EXTINCT];
        out.push({
            id: flat[off + S_ID] | 0,
            name: names?.[i] ?? `species ${flat[off + S_ID] | 0}`,
            members: flat[off + S_MEMBERS] | 0,
            peak: flat[off + S_PEAK] | 0,
            founded: flat[off + S_FOUNDED] | 0,
            extinctAt: extinct < 0 ? null : (extinct | 0),
            founderMembers: flat[off + S_FOUNDER_MEMBERS] | 0,
            founderPopulation: flat[off + S_FOUNDER_POPULATION] | 0,
            founderGeneration: flat[off + S_FOUNDER_GENERATION],
            streak: flat[off + S_STREAK] | 0,
            drift: flat[off + S_DRIFT],
            spread: flat[off + S_SPREAD],
            entryGenerations: flat[off + S_ENTRY_GENERATIONS],
            probationGenerations: flat[off + S_PROBATION_GENERATIONS],
            parentId: flat[off + S_PARENT] | 0,
            centroid: Array.from(flat.slice(off + S_CENTROID, off + S_CENTROID + SIG_LEN)),
            foundingCentroid: Array.from(flat.slice(
                off + S_FOUNDING_CENTROID, off + S_FOUNDING_CENTROID + SIG_LEN)),
            populationCentroid: Array.from(flat.slice(
                off + S_POPULATION_CENTROID, off + S_POPULATION_CENTROID + SIG_LEN)),
        });
    }
    return out;
}

// Species swatch colour comes from the centroid pushed through the same
// trait→colour path the bodies use (see `species_morph` in wasm.rs, then
// genomeColor in color.js). Deliberately *not* an arbitrary per-species hue:
// agent colour was moved off cluster labels onto continuous trait colour on
// purpose, and inventing hues would undo that — a passive species could come out
// magenta, breaking the "strategy is readable at a glance" property the ramps
// exist for. Deriving it from the centroid means the legend swatch shows what
// that lineage actually looks like swimming around. The renderer supplies the
// lookup as `colorFor`.

/**
 * Build the species panel once and return an updater.
 *
 * @returns {(rows: Array, step: number) => void}
 */
export function initSpeciesPanel(root, colorFor, bounds) {
    root.innerHTML = `
        <div data-live></div>
        <div class="sp-empty" data-empty></div>
        <details class="sp-fossils" data-fossils>
            <summary>fossil record (<span data-fossil-n>0</span>)</summary>
            <div data-fossil-list></div>
        </details>`;

    const liveEl = root.querySelector('[data-live]');
    const emptyEl = root.querySelector('[data-empty]');
    const fossilsEl = root.querySelector('[data-fossils]');
    const fossilNEl = root.querySelector('[data-fossil-n]');
    const fossilListEl = root.querySelector('[data-fossil-list]');

    // When each extinction was first seen, so it can linger before being filed.
    const extinctSeen = new Map();
    let latestRows = [];
    let latestStep = 0;

    root.onclick = event => {
        const row = event.target.closest('[data-species-id]');
        if (!row) return;
        const species = latestRows.find(s => s.id === Number(row.dataset.speciesId));
        if (species) openSpecies(species, latestStep, bounds);
    };

    return function update(rows, step, unassigned) {
        latestRows = rows;
        latestStep = step;
        const now = performance.now();
        for (const s of rows) {
            if (s.extinctAt !== null && !extinctSeen.has(s.id)) {
                extinctSeen.set(s.id, now);
            }
        }

        const lingering = s =>
            s.extinctAt !== null && (now - (extinctSeen.get(s.id) ?? now)) < EXTINCT_LINGER_MS;

        const live = rows.filter(s => s.extinctAt === null || lingering(s));
        const fossils = rows.filter(s => s.extinctAt !== null && !lingering(s));

        if (rows.length === 0) {
            // Honest empty state. No species yet is the normal condition early
            // in a run, not a failure, and saying so beats an empty box.
            emptyEl.textContent =
                'no species yet — clusters are still being tested';
            emptyEl.style.display = 'block';
        } else {
            emptyEl.style.display = 'none';
        }

        liveEl.innerHTML = live.map(s => {
            const [r, g, b] = colorFor(s);
            const age = (s.extinctAt ?? step) - s.founded;
            const dead = s.extinctAt !== null;
            return `
                <div class="legend-row sp-selectable${dead ? ' sp-dead' : ''}"
                     data-species-id="${s.id}">
                    <div class="legend-swatch" style="background:rgb(${r},${g},${b})"></div>
                    <span class="sp-name">${s.name}</span>
                    <span class="legend-count">${dead ? '—' : s.members}</span>
                    <span class="sp-age">${age}</span>
                </div>`;
        }).join('') + (unassigned > 0 ? `
                <div class="legend-row sp-unassigned">
                    <div class="legend-swatch" style="background:rgb(104,116,124)"></div>
                    <span class="sp-name">unassigned</span>
                    <span class="legend-count">${unassigned}</span>
                    <span class="sp-age"></span>
                </div>` : '');

        fossilNEl.textContent = fossils.length;
        fossilsEl.style.display = fossils.length > 0 ? 'block' : 'none';
        fossilListEl.innerHTML = fossils.map(s => `
            <div class="legend-row sp-fossil sp-selectable" data-species-id="${s.id}">
                <span class="sp-name">${s.name}</span>
                <span class="sp-age">${s.founded}–${s.extinctAt} &middot; peak ${s.peak}</span>
            </div>`).join('');
        for (const species of rows) {
            updateFloating(`species:${species.id}`,
                body => renderSpecies(body, species, step, bounds));
        }
    };
}

function openSpecies(species, step, bounds) {
    openFloating({
        key: `species:${species.id}`,
        title: species.name,
        className: 'species-window',
        render: body => renderSpecies(body, species, step, bounds),
    });
}

function renderSpecies(body, s, step, bounds) {
    const age = (s.extinctAt ?? step) - s.founded;
    const minimumMembers = Math.max(6, Math.round(s.founderPopulation * 0.05));
    const deviations = s.foundingCentroid.map((v, i) => v - s.populationCentroid[i]);
    let strongest = 0;
    for (let i = 1; i < deviations.length; i++) {
        if (Math.abs(deviations[i]) > Math.abs(deviations[strongest])) strongest = i;
    }
    const specialized = Math.abs(deviations[strongest]) >= 0.12;
    const evidence = [
        ['members', s.founderMembers, `>= ${minimumMembers}`,
            s.founderMembers >= minimumMembers],
        ['stable runs', s.streak, '>= 5', s.streak >= 5],
        ['centroid drift', s.drift.toFixed(3), '< 0.040', s.drift < 0.04],
        ['within spread', s.spread.toFixed(3), '< 0.250', s.spread < 0.25],
        ['generation advance', s.entryGenerations.toFixed(2), '>= 3.00',
            s.entryGenerations >= 3],
        ['probation advance', s.probationGenerations.toFixed(2), '>= 1.00',
            s.probationGenerations >= 1],
    ];

    const traitRows = SIGNATURE_NAMES.map((name, i) => {
        const trait = SIGNATURE_DIMS[i];
        const lo = bounds?.[trait * 2] ?? 0;
        const hi = bounds?.[trait * 2 + 1] ?? 1;
        const raw = lo + s.foundingCentroid[i] * (hi - lo);
        const delta = deviations[i];
        return `<div class="species-trait">
            <span>${name}</span><b>${raw.toFixed(2)}</b>
            <em class="${delta >= 0 ? 'up' : 'down'}">${delta >= 0 ? '+' : ''}${delta.toFixed(2)}</em>
        </div>`;
    }).join('');

    body.innerHTML = `
        <div class="species-summary">
            <span>species ${s.id}</span><span>${s.extinctAt === null ? 'living' : `extinct at ${s.extinctAt}`}</span>
            <span>founded ${s.founded}</span><span>age ${age} steps</span>
            <span>members ${s.members}</span><span>peak ${s.peak}</span>
            <span>founder generation ${s.founderGeneration.toFixed(2)}</span>
        </div>
        <h3>why it speciated</h3>
        <div class="species-evidence">${evidence.map(([label, actual, required, pass]) => `
            <div><span>${label}</span><b>${actual}</b><em class="${pass ? 'pass' : 'fail'}">${required}</em></div>
        `).join('')}</div>
        <div class="species-probation">survived probation with mutation clamped to 0.15x</div>
        <h3>founding traits</h3>
        <div class="species-traits">${traitRows}</div>
        <div class="species-name-reason">${specialized
            ? `epithet reflects ${SIGNATURE_NAMES[strongest]} ${deviations[strongest] > 0 ? 'above' : 'below'} the founding pond`
            : 'no trait cleared the 0.12 specialization threshold; epithet is nonspecialized'}</div>
        <div class="species-current-drift">current drift from founder:
            ${vectorDistance(s.centroid, s.foundingCentroid).toFixed(3)}</div>`;
}

function vectorDistance(a, b) {
    return Math.sqrt(a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0));
}

/**
 * Promotion toast. A species emerging is the moment the whole feature exists to
 * produce, and it should be visible without the panel open.
 */
export function initToast(root) {
    const queue = [];
    let showing = false;

    function next() {
        if (queue.length === 0) { showing = false; return; }
        showing = true;
        const { text, rgb } = queue.shift();
        root.innerHTML =
            `<span class="toast-dot" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>${text}`;
        root.style.display = 'block';
        root.style.opacity = '1';
        setTimeout(() => {
            root.style.opacity = '0';
            setTimeout(() => { root.style.display = 'none'; next(); }, 400);
        }, 4200);
    }

    return function announce(text, rgb) {
        queue.push({ text, rgb });
        if (!showing) next();
    };
}

/** Euclidean distance from an agent's signature to its species centroid. */
export function centroidDistance(traits, centroid, bounds) {
    let sum = 0;
    SIGNATURE_DIMS.forEach((t, i) => {
        const lo = bounds[t * 2], hi = bounds[t * 2 + 1];
        const norm = Math.max(0, Math.min(1, (traits[t] - lo) / (hi - lo)));
        sum += (norm - centroid[i]) ** 2;
    });
    return Math.sqrt(sum);
}
