// Left-side brain inspector. Renders the traced forward pass of the selected
// agent (from WasmWorld.inspect_agent) as node columns on a small dedicated
// canvas — completely separate from the main grid canvas.
//
// inspect_agent buffer layout, in blocks:
//   [inputs | h0 | h1 | h2 | logits | gates | energy_norm | age_norm | kills |
//    9 traits]
//
// Every offset below is *derived* from the layer sizes the engine reports
// (brain_layer_sizes), not written down. They used to be literals — buf[57],
// buf[58], buf[59], slice(49,57), buf[60+i] — and widening the input vector
// from 5 to 7 would have shifted all of them by two, silently: energy would
// have read the last logit, and nothing would have thrown.

const INPUT_LABELS = [
    'energy', 'food dist', 'food angle', 'crowding', 'speed',
    'threat dist', 'threat angle',
];
const OUTPUT_LABELS = ['seek', 'wander', 'separate', 'flee', 'eat', 'reproduce', 'attack', 'sleep'];
const DORMANT_OUTPUTS = new Set([6]);   // attack: routed through passive combat only

const TRAIT_NAMES = [
    'vision', 'speed', 'metabolism', 'energy cap', 'mutation', 'repro cost',
    'attack', 'defense', 'aggression', 'intelligence', 'immunity',
];

// Columns pushed apart and shifted left on the widened canvas: output labels
// ("reproduce" is the longest) were running into the right edge.
const LAYER_X = [62, 104, 142, 180, 214];

/**
 * @param {number[]} layerSizes  from brain_layer_sizes(): [inputs, h0, h1, h2, outputs]
 */
export function initInspector(layerSizes) {
    const LAYER_SIZES = layerSizes;
    // Block offsets into the inspect buffer, derived once.
    const OFF = {};
    {
        let o = 0;
        for (const [i, size] of LAYER_SIZES.entries()) { OFF[`l${i}`] = o; o += size; }
        OFF.gates = o;                       // sigmoid gates follow the logits
        o += LAYER_SIZES[LAYER_SIZES.length - 1];
        OFF.energy = o++;
        OFF.age = o++;
        OFF.kills = o++;
        OFF.traits = o;
    }
    const panel = document.getElementById('inspector');
    const swatch = document.getElementById('insp-swatch');
    const idEl = document.getElementById('insp-id');
    const statusEl = document.getElementById('insp-status');
    const energyFill = document.getElementById('insp-energy');
    const ageFill = document.getElementById('insp-age');
    const traitsBox = document.getElementById('insp-traits');
    const speciesNameEl = document.getElementById('insp-species-name');
    const speciesDistEl = document.getElementById('insp-species-dist');
    const killsEl = document.getElementById('insp-kills');
    const net = document.getElementById('insp-net');
    const ctx = net.getContext('2d');

    function setTraits(buf) {
        traitsBox.innerHTML = '';
        for (let i = 0; i < TRAIT_NAMES.length; i++) {
            const row = document.createElement('div');
            row.className = 'insp-bar-row';
            row.innerHTML =
                `<span class="trait-name">${TRAIT_NAMES[i]}</span>` +
                `<span class="trait-val" style="width:auto">${buf[OFF.traits + i].toFixed(2)}</span>`;
            traitsBox.appendChild(row);
        }
    }

    return {
        show(id, rgb) {
            panel.style.display = 'block';
            idEl.textContent = `agent ${id}`;
            statusEl.textContent = '';
            killsEl.textContent = '0';
            swatch.style.background = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
            swatch.style.color = swatch.style.background;
            traitsBox.innerHTML = '';
        },

        /** Species membership for this agent.
         *
         *  The distance is the interesting number: it is how you watch an
         *  individual drift out of its species before the population does. A
         *  member sitting near the membership radius is on its way out. */
        setSpecies(name, distance, radius) {
            if (!name) {
                speciesNameEl.textContent = 'unassigned';
                speciesNameEl.style.opacity = '0.5';
                speciesDistEl.textContent = '';
                return;
            }
            speciesNameEl.textContent = name;
            speciesNameEl.style.opacity = '0.9';
            const frac = radius > 0 ? distance / radius : 0;
            speciesDistEl.textContent = `${distance.toFixed(2)} from centroid`;
            // Fades toward the edge of membership, so "about to drift out" reads
            // without having to know what the radius is.
            speciesDistEl.style.opacity = (0.35 + Math.min(1, frac) * 0.5).toFixed(2);
        },

        /** New inspect buffer for the currently shown agent. */
        update(buf, isFirst) {
            statusEl.textContent = '';
            energyFill.style.width = `${(buf[OFF.energy] * 100).toFixed(0)}%`;
            ageFill.style.width = `${(buf[OFF.age] * 100).toFixed(0)}%`;
            killsEl.textContent = `${buf[OFF.kills] | 0}`;
            if (isFirst) setTraits(buf);   // traits are immutable per life
            drawNetwork(ctx, net.width, net.height, buf, LAYER_SIZES, OFF);
        },

        /** `cause` is the human-readable phrase from the sim's death event;
         *  the agent is already reaped by now, so it can't be re-queried. */
        showDead(cause) {
            statusEl.textContent = cause ? `died — ${cause}` : 'died';
        },
        hide() { panel.style.display = 'none'; },
    };
}

function nodeY(count, i, H) {
    const top = 18, bottom = H - 18;
    return count === 1 ? H / 2 : top + (i / (count - 1)) * (bottom - top);
}

function drawNetwork(ctx, W, H, buf, LAYER_SIZES, OFF) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = '9px "Courier New", monospace';

    // Per-layer values; hidden ReLU layers normalized by their own max for
    // display (unbounded), inputs are already ~[-1,1], gates already [0,1].
    const layers = [];
    let off = 0;
    for (const size of LAYER_SIZES) {
        layers.push(buf.slice(off, off + size));
        off += size;
    }
    // Display the gates, not the raw logits — same values the sim acts on.
    layers[LAYER_SIZES.length - 1] =
        buf.slice(OFF.gates, OFF.gates + LAYER_SIZES[LAYER_SIZES.length - 1]);

    for (let l = 0; l < LAYER_SIZES.length; l++) {
        const vals = layers[l];
        const n = LAYER_SIZES[l];
        let max = 1;
        if (l >= 1 && l <= 3) {
            max = Math.max(1e-6, ...vals);
        }

        for (let i = 0; i < n; i++) {
            const x = LAYER_X[l];
            const y = nodeY(n, i, H);
            const v = vals[i] / max;
            const mag = Math.min(1, Math.abs(v));

            // Positive teal, negative magenta (only inputs can go negative)
            const col = v >= 0 ? '60,220,180' : '255,60,140';
            ctx.fillStyle = `rgba(${col},${0.15 + mag * 0.85})`;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'rgba(60,220,180,0.25)';
            ctx.lineWidth = 0.75;
            ctx.stroke();

            if (l === 0) {
                ctx.fillStyle = 'rgba(140,180,190,0.8)';
                ctx.textAlign = 'right';
                ctx.fillText(INPUT_LABELS[i], x - 9, y + 3);
            }
            if (l === 4) {
                const dormant = DORMANT_OUTPUTS.has(i);
                ctx.textAlign = 'left';
                ctx.fillStyle = dormant ? 'rgba(100,120,130,0.45)' : 'rgba(140,180,190,0.85)';
                ctx.fillText(OUTPUT_LABELS[i], x + 26, y + 3);
                // Gate bar: the actual steering weight the sim applies
                ctx.fillStyle = 'rgba(60,220,180,0.12)';
                ctx.fillRect(x + 8, y - 3, 16, 6);
                ctx.fillStyle = dormant ? 'rgba(100,120,130,0.5)' : 'rgba(60,220,180,0.8)';
                ctx.fillRect(x + 8, y - 3, 16 * vals[i], 6);
            }
        }
    }
}
