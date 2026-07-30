// The pond's family tree, drawn as a pine.
//
// Built at runtime from the species roster — the same `species_list()` buffer the
// roster panel decodes — so it costs nothing until it is opened and is never
// stale by more than one refresh.
//
// Why a tree rather than the list we already have: the roster shows *that* a
// lineage existed; only ancestry shows the thing speciation is for, which is
// lineages splitting from lineages. `parentId` is nearest kin at promotion (see
// Species::parent_id), so an edge means "closest relative when it earned a
// name", not an observed birth. The panel says so out loud, because a tree
// implies a certainty the inference does not have.
//
// Two renderers, one layout. `layoutTree` emits primitives in tree space;
// `drawTree` paints them to a canvas (panel and PNG) and `treeToSvg` serialises
// the same list. Nothing about the drawing is written twice, so the export can
// never drift from what is on screen.

import { openFloating, updateFloating } from './floating.js';

// ── Geometry ──────────────────────────────────────────────────────────────────

const W = 780;
const PAD_X = 212;         // room for labels either side of the trunk
const PAD_TOP = 40;        // the leader and the "now" tick
const PAD_BOTTOM = 34;
const MIN_H = 320;
const ROW_H = 46;          // vertical room per species, before time scaling
const TRUNK_W = 14;
const SPREAD_MIN = 34;     // a species that died immediately still shows
const SPREAD_MAX = 132;
const BOW = 0.82;          // how far out a bough sweeps before it rises
const NEEDLE_STEP = 7;
const NEEDLE_LEN = 6;
const LABEL_GAP = 22;      // minimum vertical space between two labels

const BARK = '#4a3626';
const BARK_LIT = '#5d442f';
const DEAD = '#6b5b4d';
const INK = 'rgba(190, 245, 255, 0.85)';
const DIM = 'rgba(120, 150, 175, 0.75)';

/**
 * Lay the roster out as a pine.
 *
 * Trunk is time: base = step 0, tip = `step`. Each species leaves its parent at
 * its own founding step, so a descendant physically grows out of its ancestor's
 * bough. Bough length is lifespan, thickness is peak members.
 *
 * @param {Array} rows     parsed species rows (see parseSpecies in species.js)
 * @param {number} step    current sim step
 * @param {(s:object)=>number[]} colorFor  species → [r,g,b]
 * @returns {{w:number, h:number, prims:Array, meta:object}}
 */
export function layoutTree(rows, step, colorFor = () => [120, 200, 140]) {
    const species = [...rows].sort((a, b) => a.founded - b.founded || a.id - b.id);
    const now = Math.max(step, ...species.map(s => s.extinctAt ?? s.founded), 1);

    const h = Math.max(MIN_H, PAD_TOP + PAD_BOTTOM + species.length * ROW_H);
    const w = W;
    const cx = w / 2;
    const yOf = s => PAD_TOP + (1 - s / now) * (h - PAD_TOP - PAD_BOTTOM);

    const prims = [];
    const meta = {
        step, count: species.length,
        live: species.filter(s => s.extinctAt === null).length,
    };

    // Trunk: a tapered trunk reads as a tree where a rectangle reads as an axis.
    prims.push({
        kind: 'trunk',
        x: cx, y0: yOf(0), y1: yOf(now),
        w0: TRUNK_W, w1: TRUNK_W * 0.45,
        fill: BARK, edge: BARK_LIT,
    });
    prims.push({ kind: 'leader', x: cx, y: yOf(now), h: PAD_TOP * 0.6, fill: BARK_LIT });

    // Time ticks — the trunk is a timeline, and without marks it is decoration.
    const tickEvery = tickInterval(now);
    for (let t = 0; t <= now; t += tickEvery) {
        prims.push({ kind: 'tick', x: cx, y: yOf(t), label: String(t) });
    }

    if (species.length === 0) {
        prims.push({
            kind: 'note', x: cx, y: h / 2,
            text: 'no species yet — the pond is still one population',
        });
        return { w, h, prims, meta };
    }

    // Where each species' bough starts, so a child can leave its parent rather
    // than the trunk. Missing parents (id not in the roster) fall back to the
    // trunk: a partial roster should still draw.
    const byId = new Map(species.map(s => [s.id, s]));
    const anchor = new Map();
    let sideFlip = 1;

    for (const s of species) {
        const parent = s.parentId ? byId.get(s.parentId) : null;
        const y = yOf(s.founded);
        // Alternate sides so boughs do not stack; a child prefers its parent's
        // side, which keeps a lineage visually together.
        let side;
        if (parent && anchor.has(parent.id)) {
            side = anchor.get(parent.id).side;
        } else {
            sideFlip = -sideFlip;
            side = sideFlip;
        }
        const x0 = parent ? pointOnBough(anchor.get(parent.id), s.founded, cx) : cx;

        // Vertical is always time — a bough leaves at its founding height and
        // ends at its death (or at now). Horizontal spread is decoration, scaled
        // by lifespan so a long-lived lineage reaches further out. Boughs
        // therefore rise as they extend, which is what lets a descendant attach
        // at the exact point on its parent matching its own founding step.
        const end = s.extinctAt ?? now;
        const life = end - s.founded;
        const spread = SPREAD_MIN
            + (SPREAD_MAX - SPREAD_MIN) * Math.min(1, life / Math.max(1, now));
        const x1 = clamp(x0 + side * spread, PAD_X * 0.55, w - PAD_X * 0.55);
        const y1 = yOf(end);
        const dead = s.extinctAt !== null;
        const rgb = dead ? [107, 91, 77] : colorFor(s);

        // Control point: the bough sweeps outward first, then rises to its tip.
        //
        // The curve must never dip below where it started. Vertical is time, so
        // a sagging bough would draw a lineage running backwards — the first
        // version did exactly that and looked, correctly, wrong. Keeping the
        // control point between the endpoints in y makes the curve monotone in
        // time while still bowing in x, which is where the conifer sweep comes
        // from.
        const seat = {
            x0, y0: y, x1, y1, side, t0: s.founded, t1: end,
            cx: x0 + (x1 - x0) * BOW,
            cy: y + (y1 - y) * 0.18,
        };
        anchor.set(s.id, seat);

        prims.push({
            kind: 'bough',
            ...seat,
            width: 2 + 5 * Math.min(1, s.peak / 40),
            color: dead ? DEAD : `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`,
            dead,
        });
        if (!dead) {
            prims.push({
                kind: 'needles', ...seat,
                color: `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`,
            });
        }
        // Labels live in a fixed column outside the canopy rather than floating
        // at each tip: tips cluster wherever promotions clustered, and text over
        // the boughs made both unreadable. A leader line ties each back to its
        // own tip.
        prims.push({
            kind: 'label',
            x: side > 0 ? w - PAD_X + 12 : PAD_X - 12,
            y: y1 + 3,
            align: side > 0 ? 'left' : 'right',
            tip: { x: x1, y: y1 },
            text: s.name,
            sub: dead
                ? `${s.founded}–${s.extinctAt} · peak ${s.peak}`
                : `${s.members} alive · ${life} steps · peak ${s.peak}`,
            strike: dead,
        });
    }

    spaceLabels(prims, h);
    return { w, h, prims, meta };
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Push labels apart so a burst of promotions in a short window is still
 *  readable. Each side is spaced independently, and a nudged label keeps a
 *  leader line back to its bough tip so it never looks orphaned. */
function spaceLabels(prims, h) {
    for (const side of [-1, 1]) {
        const labels = prims.filter(p => p.kind === 'label'
            && (side > 0 ? p.align === 'left' : p.align === 'right'));
        labels.sort((a, b) => a.y - b.y);
        let prev = -Infinity;
        for (const l of labels) {
            l.y = Math.min(Math.max(l.y, prev + LABEL_GAP), h - 12);
            l.leader = l.tip;
            prev = l.y;
        }
    }
}

/** Round time ticks to something readable at this run length. */
function tickInterval(now) {
    for (const t of [50, 100, 250, 500, 1000, 2500, 5000]) {
        if (now / t <= 8) return t;
    }
    return Math.ceil(now / 8 / 5000) * 5000;
}

/** Where a child leaves its parent's bough: the point matching the child's own
 *  founding step. Interpolated in time, not pixels, so it stays right whatever
 *  the vertical scale is. A child founded after its parent died — possible, the
 *  parent may be a fossil — hangs off the tip. */
function pointOnBough(seat, founded, cx) {
    if (!seat) return cx;
    const span = seat.t1 - seat.t0;
    const t = span <= 0 ? 1 : Math.max(0, Math.min(1, (founded - seat.t0) / span));
    return seat.x0 + (seat.x1 - seat.x0) * t;
}

// ── Canvas ────────────────────────────────────────────────────────────────────

/** Paint a layout. Coordinates are CSS pixels; the caller owns DPR scaling,
 *  following the convention in graphs.js. */
export function drawTree(ctx, layout) {
    const { w, h, prims } = layout;
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = 'round';

    for (const p of prims) {
        switch (p.kind) {
            case 'trunk': {
                ctx.fillStyle = p.fill;
                ctx.beginPath();
                ctx.moveTo(p.x - p.w0 / 2, p.y0);
                ctx.lineTo(p.x + p.w0 / 2, p.y0);
                ctx.lineTo(p.x + p.w1 / 2, p.y1);
                ctx.lineTo(p.x - p.w1 / 2, p.y1);
                ctx.closePath();
                ctx.fill();
                break;
            }
            case 'leader':
                ctx.strokeStyle = p.fill;
                ctx.lineWidth = 3;
                line(ctx, p.x, p.y, p.x, p.y - p.h);
                break;
            case 'tick':
                ctx.strokeStyle = 'rgba(120, 150, 175, 0.35)';
                ctx.lineWidth = 1;
                line(ctx, p.x - 11, p.y, p.x + 11, p.y);
                ctx.fillStyle = DIM;
                ctx.font = '9px "Courier New", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(p.label, p.x, p.y - 3);
                break;
            case 'bough':
                ctx.strokeStyle = p.color;
                ctx.lineWidth = p.width;
                if (p.dead) ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(p.x0, p.y0);
                ctx.quadraticCurveTo(p.cx, p.cy, p.x1, p.y1);
                ctx.stroke();
                ctx.setLineDash([]);
                // A snapped-off tip: the lineage ends here and nothing grows on.
                if (p.dead) {
                    ctx.lineWidth = 1;
                    const t = 3.5;
                    line(ctx, p.x1 - t, p.y1 - t, p.x1 + t, p.y1 + t);
                    line(ctx, p.x1 - t, p.y1 + t, p.x1 + t, p.y1 - t);
                }
                break;
            case 'needles':
                ctx.strokeStyle = p.color;
                ctx.lineWidth = 1;
                for (const n of needles(p)) line(ctx, n.x0, n.y0, n.x1, n.y1);
                break;
            case 'label':
                if (p.leader) {
                    ctx.strokeStyle = 'rgba(120, 150, 175, 0.28)';
                    ctx.lineWidth = 1;
                    line(ctx, p.leader.x, p.leader.y, p.x - (p.align === 'left' ? 5 : -5), p.y - 3);
                }
                ctx.textAlign = p.align;
                ctx.fillStyle = p.strike ? DIM : INK;
                ctx.font = '11px "Courier New", monospace';
                ctx.fillText(p.text, p.x, p.y);
                if (p.strike) {
                    const wl = ctx.measureText(p.text).width;
                    ctx.strokeStyle = DIM;
                    ctx.lineWidth = 1;
                    const x0 = p.align === 'left' ? p.x : p.x - wl;
                    line(ctx, x0, p.y - 4, x0 + wl, p.y - 4);
                }
                ctx.fillStyle = DIM;
                ctx.font = '9px "Courier New", monospace';
                ctx.fillText(p.sub, p.x, p.y + 10);
                break;
            case 'note':
                ctx.textAlign = 'center';
                ctx.fillStyle = DIM;
                ctx.font = '11px "Courier New", monospace';
                ctx.fillText(p.text, p.x, p.y);
                break;
        }
    }
    ctx.textAlign = 'left';
}

function line(ctx, x0, y0, x1, y1) {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
}

/** A point on the bough's curve, and the tangent there. */
function onCurve(p, t) {
    const u = 1 - t;
    const x = u * u * p.x0 + 2 * u * t * p.cx + t * t * p.x1;
    const y = u * u * p.y0 + 2 * u * t * p.cy + t * t * p.y1;
    const dx = 2 * u * (p.cx - p.x0) + 2 * t * (p.x1 - p.cx);
    const dy = 2 * u * (p.cy - p.y0) + 2 * t * (p.y1 - p.cy);
    return { x, y, a: Math.atan2(dy, dx) };
}

/** Needle pairs along a bough — the thing that makes it a pine rather than a
 *  dendrogram. Needles sweep back toward the trunk and hang below, the way a
 *  conifer's do. Shared by both renderers so they cannot disagree. */
function needles(p) {
    const span = Math.hypot(p.x1 - p.x0, p.y1 - p.y0) || 1;
    const steps = Math.max(2, Math.floor(span / NEEDLE_STEP));
    const out = [];
    for (let i = 1; i <= steps; i++) {
        const t = i / (steps + 1);
        const { x, y, a } = onCurve(p, t);
        // Longer needles toward the middle of the bough, tapering at the tip.
        const len = NEEDLE_LEN * (0.55 + 0.75 * Math.sin(Math.PI * t));
        for (const sweep of [2.3, -2.3]) {
            const na = a + sweep * p.side;
            out.push({ x0: x, y0: y, x1: x + Math.cos(na) * len, y1: y + Math.sin(na) * len });
        }
    }
    return out;
}

// ── SVG ───────────────────────────────────────────────────────────────────────

const esc = s => String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** The same layout as a standalone SVG document. */
export function treeToSvg(layout, meta = {}) {
    const { w, h, prims } = layout;
    const parts = [];
    parts.push(`<rect width="${w}" height="${h}" fill="#040a12"/>`);

    for (const p of prims) {
        switch (p.kind) {
            case 'trunk':
                parts.push(`<polygon points="${p.x - p.w0 / 2},${p.y0} ${p.x + p.w0 / 2},${p.y0} ` +
                    `${p.x + p.w1 / 2},${p.y1} ${p.x - p.w1 / 2},${p.y1}" fill="${p.fill}"/>`);
                break;
            case 'leader':
                parts.push(svgLine(p.x, p.y, p.x, p.y - p.h, p.fill, 3));
                break;
            case 'tick':
                parts.push(svgLine(p.x - 11, p.y, p.x + 11, p.y, 'rgba(120,150,175,0.35)', 1));
                parts.push(svgText(p.x, p.y - 3, p.label, 9, DIM, 'middle'));
                break;
            case 'bough':
                parts.push(`<path d="M ${r(p.x0)} ${r(p.y0)} Q ${r(p.cx)} ${r(p.cy)} ` +
                    `${r(p.x1)} ${r(p.y1)}" fill="none" stroke="${p.color}" ` +
                    `stroke-width="${r(p.width)}" stroke-linecap="round"` +
                    `${p.dead ? ' stroke-dasharray="5 4"' : ''}/>`);
                if (p.dead) {
                    const t = 3.5;
                    parts.push(svgLine(p.x1 - t, p.y1 - t, p.x1 + t, p.y1 + t, DEAD, 1));
                    parts.push(svgLine(p.x1 - t, p.y1 + t, p.x1 + t, p.y1 - t, DEAD, 1));
                }
                break;
            case 'needles':
                for (const n of needles(p)) {
                    parts.push(svgLine(n.x0, n.y0, n.x1, n.y1, p.color, 1));
                }
                break;
            case 'label': {
                if (p.leader) {
                    parts.push(svgLine(p.leader.x, p.leader.y,
                        p.x - (p.align === 'left' ? 5 : -5), p.y - 3,
                        'rgba(120,150,175,0.28)', 1));
                }
                const anchor = p.align === 'left' ? 'start' : 'end';
                const deco = p.strike ? ' text-decoration="line-through"' : '';
                parts.push(svgText(p.x, p.y, p.text, 11, p.strike ? DIM : INK, anchor, deco));
                parts.push(svgText(p.x, p.y + 10, p.sub, 9, DIM, anchor));
                break;
            }
            case 'note':
                parts.push(svgText(p.x, p.y, p.text, 11, DIM, 'middle'));
                break;
        }
    }

    const caption = `pond_spawn · seed ${meta.seed ?? '?'} · step ${layout.meta?.step ?? '?'} · ` +
        `${layout.meta?.live ?? 0} live of ${layout.meta?.count ?? 0} lineages`;
    parts.push(svgText(w / 2, h - 8, caption, 9, DIM, 'middle'));

    return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" ` +
        `viewBox="0 0 ${w} ${h}" font-family="Courier New, monospace">\n` +
        parts.join('\n') + '\n</svg>\n';
}

function svgLine(x0, y0, x1, y1, stroke, width, extra = '') {
    return `<line x1="${r(x0)}" y1="${r(y0)}" x2="${r(x1)}" y2="${r(y1)}" ` +
        `stroke="${stroke}" stroke-width="${r(width)}" stroke-linecap="round"${extra}/>`;
}

function svgText(x, y, text, size, fill, anchor, extra = '') {
    return `<text x="${r(x)}" y="${r(y)}" font-size="${size}" fill="${fill}" ` +
        `text-anchor="${anchor}"${extra}>${esc(text)}</text>`;
}

const r = n => Math.round(n * 10) / 10;

// ── Download ──────────────────────────────────────────────────────────────────

/** Hand a blob to the browser as a file. The one download idiom in the app;
 *  the object URL is revoked so a long session does not leak them. */
export function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function exportSvg(layout, meta) {
    downloadBlob(new Blob([treeToSvg(layout, meta)], { type: 'image/svg+xml' }),
        filename(meta, 'svg'));
}

/** PNG at 2× so an exported tree survives being looked at. */
export function exportPng(layout, meta) {
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = layout.w * scale;
    canvas.height = layout.h * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#040a12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    drawTree(ctx, layout);
    canvas.toBlob(blob => blob && downloadBlob(blob, filename(meta, 'png')));
}

function filename(meta, ext) {
    return `pond_phylogeny_seed${meta?.seed ?? 'x'}_step${meta?.step ?? 0}.${ext}`;
}

// ── Window ────────────────────────────────────────────────────────────────────

const KEY = 'tree:phylogeny';

/**
 * Open (or refresh) the phylogeny window.
 *
 * `source()` is called on every render and returns `{rows, step, seed}`, so the
 * tree tracks the run rather than freezing at the moment it was opened.
 */
export function openPhylogeny(source, colorFor) {
    openFloating({
        key: KEY,
        title: 'phylogeny',
        className: 'float-tree',
        render: body => renderInto(body, source, colorFor),
    });
}

/** Refresh an already-open window; no-op when it is closed. */
export function refreshPhylogeny(source, colorFor) {
    updateFloating(KEY, body => renderInto(body, source, colorFor));
}

function renderInto(body, source, colorFor) {
    const { rows, step, seed } = source();
    const layout = layoutTree(rows, step, colorFor);
    const meta = { seed, step };

    // Rebuild the shell only once; redrawing into the existing canvas keeps the
    // export buttons from being torn out from under a click.
    let canvas = body.querySelector('canvas');
    if (!canvas) {
        body.innerHTML =
            `<div class="tree-actions">` +
              `<button data-png>export png</button>` +
              `<button data-svg>export svg</button>` +
            `</div>` +
            `<canvas></canvas>` +
            `<div class="tree-note">` +
              `Trunk is time, base = step 0. Each bough leaves its parent at its ` +
              `founding step; length is lifespan, thickness is peak members, ` +
              `bare and dashed means extinct.<br>` +
              `Founding and extinction steps are multiples of 50 — the registry ` +
              `only advances on a cluster run — so the trunk is quantised. ` +
              `Parentage is nearest kin at promotion, not an observed split: ` +
              `it is the best available inference, not a genealogy.` +
            `</div>`;
        canvas = body.querySelector('canvas');
        body.querySelector('[data-png]').addEventListener('click',
            () => exportPng(currentLayout, currentMeta));
        body.querySelector('[data-svg]').addEventListener('click',
            () => exportSvg(currentLayout, currentMeta));
    }
    currentLayout = layout;
    currentMeta = meta;

    // DPR handling as in graphs.js: draw in CSS pixels, scale the backing store.
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = `${layout.w}px`;
    canvas.style.height = `${layout.h}px`;
    canvas.width = Math.round(layout.w * dpr);
    canvas.height = Math.round(layout.h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTree(ctx, layout);
}

// The layout the buttons export — always what was last drawn, so the file
// matches the picture rather than re-deriving it from a newer roster.
let currentLayout = null;
let currentMeta = null;
