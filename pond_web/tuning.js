// Live rule dials: food regen, the hunt threshold, and clustering k.
//
// Separate from the setup panel on purpose. Setup takes exactly the three
// arguments World::new takes and can only apply them by building a new world;
// these are live, and watching a pond respond to a regen change without losing
// the run is the whole point of exposing them.
//
// Defaults and ranges come from the engine (tunable_ranges), so no number here
// is written down twice.

const ROWS = [
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
               'for the family legend and as speciation’s input. This ' +
               'changes only how you read the pond, never how it behaves — ' +
               'more families means finer splits between lineages that are ' +
               'nearly alike.',
    },
];

/**
 * Build the tuning panel.
 *
 * @param {HTMLElement} root  the #tuning panel
 * @param {object} api  { ranges(), get(), set(key, value), modified(), onChange(key, value) }
 *   ranges() → Float32Array of [default, min, max] × 3 in ROWS order
 *   get()    → {regen, hunt, k} as the engine currently holds them
 * @returns {{refresh: () => void}}
 */
export function initTuning(root, api) {
    const r = api.ranges();
    const spec = ROWS.map((row, i) => ({
        ...row,
        def: r[i * 3], min: r[i * 3 + 1], max: r[i * 3 + 2],
    }));

    root.innerHTML =
        `<h2>tuning</h2>` +
        spec.map(s =>
            `<div class="tune-row">` +
              `<div class="tune-head">` +
                `<span class="tune-label">${s.label}</span>` +
                // tabindex so the blurb is reachable without a mouse — the panel
                // is keyboard-usable and a hover-only explanation would not be.
                `<span class="tune-info" tabindex="0" role="note" ` +
                      `aria-label="${s.label}: ${s.blurb}">i` +
                  `<span class="tune-blurb">${s.blurb}</span>` +
                `</span>` +
                `<span class="tune-value" id="tune-v-${s.key}">${s.format(s.def)}</span>` +
              `</div>` +
              `<div class="tune-controls">` +
                `<input type="range" id="tune-${s.key}" min="${s.min}" max="${s.max}" ` +
                       `step="${s.step}" value="${s.def}">` +
                `<button class="tune-reset" id="tune-r-${s.key}" title="back to default (${s.format(s.def)})">↺</button>` +
              `</div>` +
            `</div>`
        ).join('') +
        `<div class="tune-note" id="tune-note"></div>`;

    const el = id => root.querySelector('#' + id);

    for (const s of spec) {
        const slider = el(`tune-${s.key}`);
        const value = el(`tune-v-${s.key}`);
        const apply = v => {
            api.set(s.key, v);
            // Read back rather than trusting the slider: the engine clamps, and
            // k in particular is rounded on the way in.
            refresh();
        };
        slider.addEventListener('input', () => apply(parseFloat(slider.value)));
        el(`tune-r-${s.key}`).addEventListener('click', () => apply(s.def));
        value.textContent = s.format(s.def);
    }

    function refresh() {
        const cur = api.get();
        for (const s of spec) {
            el(`tune-${s.key}`).value = cur[s.key];
            el(`tune-v-${s.key}`).textContent = s.format(cur[s.key]);
        }
        // Honesty about reproducibility: once a dial has moved, the seed alone
        // no longer describes this run, and the setup panel's "same seed = same
        // pond" is no longer true of it.
        el('tune-note').textContent = api.modified()
            ? 'dials moved — this run is no longer reproducible from its seed alone'
            : 'at defaults — seed alone reproduces this run';
        el('tune-note').classList.toggle('tuned', api.modified());
    }

    refresh();
    return {
        refresh,
        /** Current slider positions, for re-applying them to a new world. */
        values: () => Object.fromEntries(
            spec.map(s => [s.key, parseFloat(el(`tune-${s.key}`).value)])),
    };
}
