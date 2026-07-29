// Left-side brain inspector. Renders the traced forward pass of the selected
// agent (from WasmWorld.inspect_agent) as node columns on a small dedicated
// canvas — completely separate from the main grid canvas.
//
// inspect_agent buffer layout (68 floats):
//   [0..5)   inputs        [5..17)  h0        [17..29) h1
//   [29..41) h2            [41..49) logits    [49..57) sigmoid gates
//   [57] energy_norm  [58] age_norm  [59] kills  [60..69) traits

const INPUT_LABELS = ['energy', 'food dist', 'food angle', 'crowding', 'speed'];
const OUTPUT_LABELS = ['seek', 'wander', 'separate', 'flee', 'eat', 'reproduce', 'attack', 'sleep'];
const DORMANT_OUTPUTS = new Set([3, 6]);   // flee/attack: no live mechanic yet

const TRAIT_NAMES = [
    'vision', 'speed', 'metabolism', 'energy cap', 'mutation', 'repro cost',
    'attack', 'defense', 'aggression',
];

const LAYER_SIZES = [5, 12, 12, 12, 8];
// Columns pushed apart and shifted left on the widened canvas: output labels
// ("reproduce" is the longest) were running into the right edge.
const LAYER_X = [62, 104, 142, 180, 214];

export function initInspector() {
    const panel = document.getElementById('inspector');
    const swatch = document.getElementById('insp-swatch');
    const idEl = document.getElementById('insp-id');
    const statusEl = document.getElementById('insp-status');
    const energyFill = document.getElementById('insp-energy');
    const ageFill = document.getElementById('insp-age');
    const traitsBox = document.getElementById('insp-traits');
    const killsEl = document.getElementById('insp-kills');
    const net = document.getElementById('insp-net');
    const ctx = net.getContext('2d');

    function setTraits(buf) {
        traitsBox.innerHTML = '';
        for (let i = 0; i < 9; i++) {
            const row = document.createElement('div');
            row.className = 'insp-bar-row';
            row.innerHTML =
                `<span class="trait-name">${TRAIT_NAMES[i]}</span>` +
                `<span class="trait-val" style="width:auto">${buf[60 + i].toFixed(2)}</span>`;
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

        /** New inspect buffer for the currently shown agent. */
        update(buf, isFirst) {
            statusEl.textContent = '';
            energyFill.style.width = `${(buf[57] * 100).toFixed(0)}%`;
            ageFill.style.width = `${(buf[58] * 100).toFixed(0)}%`;
            killsEl.textContent = `${buf[59] | 0}`;
            if (isFirst) setTraits(buf);   // traits are immutable per life
            drawNetwork(ctx, net.width, net.height, buf);
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

function drawNetwork(ctx, W, H, buf) {
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
    layers[4] = buf.slice(49, 57);   // display gates, not raw logits

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
