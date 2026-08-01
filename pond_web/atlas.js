// Pre-rendered body sprites, for the zoomed-out crowd.
//
// Why this exists: a Firefox profile at ~5,700 agents put `fill` + `stroke` at
// 72% of all render time, and the arithmetic underneath it said the cost is
// per-*call*, not per-pixel. That was later confirmed directly on the dev
// machine: per-agent cost is ~9 µs at 5,000+ agents whether bodies are 20 px or
// 55 px long — 11% apart across a 7× difference in pixel coverage. Two earlier
// interventions that removed large amounts of per-agent *work* (ornament gating,
// composite batching) each moved per-agent cost by under 3%, which is what you
// would expect if the cost is the call and not the work.
//
// `drawImage` does not pay that overhead. This module bakes a body once into an
// atlas and blits it, taking an agent from ~2 hull fills plus ornaments down to
// one `drawImage` per pass.
//
// **What is given up.** A sprite is a frozen pose: the kinematic wiggle, the
// per-agent envelope, and the glow pulse are all baked. That is the trade the
// LOD threshold governs — below ~20 px/tile an agent is ~13 px long and its
// spine curvature is under a pixel, so the wiggle is not visible to give up.
// Above the threshold the vector pipeline in body.js runs unchanged.
//
// **This threshold is deliberately conservative and was tried the other way.**
// Raising it to 56 px/tile — justified by the ~9 µs figure above, which says
// sprites pay off at every zoom — put frozen bodies on screen at sizes where the
// spine flex reads, and the pond visibly stiffened. Quantising the spine into
// curvature buckets did not save it. If this is revisited, the thing to fix is
// the stiffness, not the threshold: the threshold is only the symptom's volume
// knob.
//
// **What is kept.** Lineage hue, strategy halo colour and weight, silhouette
// (segment count, plan proportions, fins, spikes), the unassigned outline, and
// the additive glow — the glow is a separate sprite drawn under `lighter`, not
// baked flat into the body, so the pond keeps the shared haze that the look
// depends on.

import { drawBody, PASS_GLOW, PASS_CORE } from './body.js';

/** Pixels per world tile the atlas is rendered at.
 *
 *  Sprites are drawn scaled by `scale_px / ATLAS_PPT`, so this is a resolution
 *  choice, not a size one — any zoom reproduces the vector geometry exactly,
 *  just resampled. Set a little above the LOD threshold so the common case is a
 *  mild *down*scale (smooth) rather than an upscale (mush), and not much above:
 *  every extra pixel here is atlas area, and the atlas is a fixed budget. */
export const ATLAS_PPT = 28;

/** Zoom at or below which the crowd is drawn from the atlas.
 *
 *  In `scale_px` (screen pixels per world tile) because that is what the whole
 *  renderer is scaled by and a *global* switch keeps the pond visually of one
 *  piece — a mixed frame, some bodies articulated and some frozen, reads as a
 *  bug. At 20 px/tile a body is ~13 px long and ~5 px wide.
 *
 *  Note this is a floor set by grid size, not by the camera: at full zoom-out
 *  `scale_px` is `min(canvasW, canvasH) / GRID`, so a small pond never reaches
 *  this however far out you go. The M HUD prints that floor. */
export const SPRITE_LOD_MAX_SCALE_PX = 20;

const ATLAS_W = 2048;
const ATLAS_H = 2048;

// Ceiling on distinct sprites. Overflow wipes the atlas and lets it refill,
// which is cheap (a build is a handful of fills) and self-correcting: the keys
// that come back are the ones actually on screen. A hard cap that *stopped*
// building would instead leave whichever agents lost the race permanently on the
// slow path.
const MAX_ENTRIES = 448;

// Sprites built per frame. A pond that speciates mid-frame, or the first frame
// after a reset, would otherwise stall building hundreds of sprites at once;
// past this the remaining agents fall back to the vector path for that frame and
// get their sprite on the next one. Spreads a spike over a few frames instead of
// dropping one.
const BUILDS_PER_FRAME = 12;

// Canonical pose lives at (ORIGIN, ORIGIN) in a world large enough that body.js
// never sees the toroidal seam: a body at the origin of a size-0 grid would
// trigger its wrap-copy branch and paint itself three extra times off-canvas.
const ORIGIN = 100;
const GRID_BIG = 1000;

// Mirrors body.js. Only used for bounding-box arithmetic here; the actual glow
// geometry is body.js's.
const GLOW_SCALE = 1.9;

let atlas = null, actx = null;
// Shelf packer: rows filled left to right, a new row started when one is full.
// Sprites are wide and short and arrive in no particular order, so a shelf
// wastes some tail on each row — cheap next to the complexity of anything
// better, and the atlas is only ever a few percent full in practice.
let shelf_x = 0, shelf_y = 0, shelf_h = 0;
let entries = new Map();
let builds_this_frame = 0;

function ensure_atlas() {
    if (atlas) return;
    atlas = document.createElement('canvas');
    atlas.width = ATLAS_W;
    atlas.height = ATLAS_H;
    actx = atlas.getContext('2d');
}

/** Drop every sprite and start the atlas over. Called on overflow, and by the
 *  renderer when the palette wholesale changes (archetype overlay on or off) —
 *  the old sprites are still *correct*, but they are all about to be unreachable
 *  keys taking up shelf space. */
export function resetAtlas() {
    if (actx) actx.clearRect(0, 0, ATLAS_W, ATLAS_H);
    shelf_x = 0; shelf_y = 0; shelf_h = 0;
    entries = new Map();
}

/** Call once per frame before queueing agents. Resets the build budget. */
export function beginAtlasFrame() {
    builds_this_frame = 0;
}

export function atlasStats() {
    return { entries: entries.size, shelf_y, built: builds_this_frame };
}

// ── Keying ───────────────────────────────────────────────────────────────────
//
// One 29-bit integer, not a string: this is computed per agent per frame, and
// 6,000 template strings a frame to look up a Map is exactly the kind of
// incidental allocation this whole exercise is trying to delete.
//
// The buckets are deliberately coarse where a difference is unreadable at 13 px
// and fine where it is not:
//
// - **Colour, 4 bits per channel.** Lineage hue is the single most legible
//   signal on screen, so it gets the most bits. 16 levels per channel is a
//   ~4% error on any one channel, which is under the difference between two
//   adjacent species' hues.
// - **Strategy, 4 bits.** `strategyGlow` is a one-dimensional ramp (combat
//   0→1 walks a cool-teal → hot-orange arc and sets the halo weight from the
//   same number), so one bucket index captures both colour and weight exactly.
// - **Silhouette:** segment count is exact (it is already quantised to 5/7/9/11
//   by morphology.js), fins and spikes are exact counts, and the two continuous
//   proportion knobs get 2 bits each.
// - **Energy, 2 bits.** Drives body radius and fill alpha. Four steps is visible
//   as "bright and fat" vs "dim and thin" without banding, because the colour
//   itself already carries the energy dimming at 4-bit resolution.
//
// Everything *not* in the key — belly, elongation's effect on the envelope, the
// exact eye size — rides along on whichever agent's spec happened to build the
// sprite first. Those are sub-pixel at this zoom.
const SEG_TO_IDX = { 5: 0, 7: 1, 9: 2, 11: 3 };

function bucket(v, n) {
    const b = (v * n) | 0;
    return b < 0 ? 0 : b >= n ? n - 1 : b;
}

/** Representative value at the centre of bucket `b` of `n`. */
function unbucket(b, n) { return (b + 0.5) / n; }

const ENERGY_BUCKETS = 4;
const COMBAT_BUCKETS = 16;

/** @returns {number} 29-bit sprite key, or -1 if this body has no sprite form. */
export function spriteKey(spec, palette, combat, energyNorm, outlined) {
    const seg = SEG_TO_IDX[spec.segCount];
    if (seg === undefined) return -1;
    const qr = palette[0] >> 4, qg = palette[1] >> 4, qb = palette[2] >> 4;
    const pb = bucket(spec.headPointiness, 4);
    const bb = bucket((spec.armorBumps) / 4, 4);       // armour count stands in for bulk
    const fb = Math.min(3, spec.fins.count >> 1);      // 0, 2, 4, 6
    const ob = Math.min(3, spec.ornamentPairs);
    const eb = bucket(energyNorm, ENERGY_BUCKETS);
    const cb = bucket(combat, COMBAT_BUCKETS);
    return (
        seg |
        (pb << 2) | (bb << 4) | (fb << 6) | (ob << 8) |
        (qr << 10) | (qg << 14) | (qb << 18) |
        (cb << 22) |
        (eb << 26) |
        ((outlined ? 1 : 0) << 28)
    );
}

// ── Building ─────────────────────────────────────────────────────────────────

function pack(w, h) {
    if (shelf_x + w > ATLAS_W) { shelf_x = 0; shelf_y += shelf_h; shelf_h = 0; }
    if (shelf_y + h > ATLAS_H) return null;
    const r = { x: shelf_x, y: shelf_y };
    shelf_x += w;
    if (h > shelf_h) shelf_h = h;
    return r;
}

/** Screen-space extent of this body at ATLAS_PPT, plus where its head sits.
 *
 *  Analytic rather than measured: body.js's geometry is all derived from
 *  `baseR`, `segDist` and the ornament lengths, so the box can be computed
 *  without a trial render. Erring wide costs atlas area; erring narrow clips a
 *  fin off, so every term is the outer bound and there is a 2 px cushion. */
function spriteBox(spec, baseR) {
    let maxEnv = 0;
    for (const e of spec.envelope) if (e > maxEnv) maxEnv = e;
    const r = maxEnv * baseR;
    const spike = spec.ornamentPairs > 0 ? spec.ornamentLen * ATLAS_PPT : 0;
    const fin = spec.fins.count > 0 ? spec.fins.len * ATLAS_PPT : 0;
    // Half-height: the glow hull, or the core hull plus whatever sticks out of
    // it — ornaments are not scaled by the glow, so these compete rather than
    // stack.
    const lat = Math.max(r * GLOW_SCALE, r + Math.max(spike, fin)) + 2;
    const bodyLen = (spec.segCount - 1) * spec.segDist * ATLAS_PPT;
    // The head apex runs ahead of segment 0 by the pointiness wedge, at glow
    // scale on the glow pass.
    const head = spec.envelope[0] * baseR * GLOW_SCALE * (0.5 + spec.headPointiness * 1.4)
        + spike + 2;
    // The chain trails the head in -x, so the head sits near the *right* edge:
    // `lat` of tail cap, then the body, then the head and its wedge.
    return {
        w: Math.ceil(lat + bodyLen + head),
        h: Math.ceil(2 * lat),
        px: lat + bodyLen, // head (rotation pivot) offset from the cell's left edge
        py: lat,           // …and from its top
    };
}

/** Render one pass of a canonical, straight, +x-facing body into a packed cell. */
function bake(spec, palette, glow, outline, energyNorm, baseR, box, pass) {
    const cell = pack(box.w, box.h);
    if (!cell) return null;

    actx.clearRect(cell.x, cell.y, box.w, box.h);

    // Batched mode (`pass` given) so body.js touches neither the composite mode
    // nor save/restore. Both passes bake under `source-over`: the glow is a
    // single non-self-overlapping fill, and additive-over-transparent is
    // identical to source-over-over-transparent, so the sprite is byte-identical
    // to what the additive pass would have produced. The *drawing* of it is
    // still additive — see draw_sprites in renderer.js.
    actx.globalCompositeOperation = 'source-over';

    const chain = { segs: new Array(spec.segCount) };
    for (let s = 0; s < spec.segCount; s++) {
        chain.segs[s] = { x: ORIGIN - s * spec.segDist, y: ORIGIN };
    }
    const xform = {
        tile_w: ATLAS_PPT, tile_h: ATLAS_PPT, scale_px: ATLAS_PPT,
        off_x: cell.x + box.px - ORIGIN * ATLAS_PPT,
        off_y: cell.y + box.py - ORIGIN * ATLAS_PPT,
        gridSize: GRID_BIG,
    };
    // velX pins the head direction to +x regardless of what the chain's own
    // finite-difference would say at the head. timeSec 0 puts the glow pulse at
    // the centre of its swing — a baked sprite cannot pulse, so it sits at the
    // mean rather than at a phase extreme.
    const motion = { baseR, energyNorm, velX: 1, velY: 0, timeSec: 0 };

    drawBody(actx, chain, spec, palette, xform, motion, glow, outline, pass);
    return { sx: cell.x, sy: cell.y, sw: box.w, sh: box.h, px: box.px, py: box.py };
}

/**
 * Sprite pair for a body, built on first sight.
 *
 * @param {number} key       from `spriteKey`, already computed by the caller
 * @param {object} spec      the requesting agent's MorphSpec — only the first
 *                           agent to ask for a key contributes its spec, so the
 *                           unkeyed proportions are that agent's
 * @param {number[]} palette the agent's exact core rgb (quantised here)
 * @param {object} glow      `{rgb, weight}` from strategyGlow
 * @param {number[]|null} outline
 * @param {number} energyNorm
 * @returns {{glow: object, core: object}|null} null when the build budget is
 *          spent this frame — the caller draws that agent the slow way instead.
 */
export function spriteFor(key, spec, palette, glow, outline, energyNorm) {
    let e = entries.get(key);
    if (e) return e;
    if (builds_this_frame >= BUILDS_PER_FRAME) return null;

    ensure_atlas();
    if (entries.size >= MAX_ENTRIES) resetAtlas();

    // Build from the bucket centres, not from the requesting agent's exact
    // values — otherwise the sprite every one of a thousand agents shares would
    // be whichever one of them happened to be decoded first, and the pond's
    // colour would visibly jump each time a cache reset handed the slot to a
    // different animal.
    const qp = [
        (palette[0] >> 4) * 16 + 8,
        (palette[1] >> 4) * 16 + 8,
        (palette[2] >> 4) * 16 + 8,
    ];
    const eN = unbucket(bucket(energyNorm, ENERGY_BUCKETS), ENERGY_BUCKETS);
    const baseR = ATLAS_PPT * (0.105 + eN * 0.07 + spec.headPointiness * 0.05);
    const box = spriteBox(spec, baseR);

    const g = bake(spec, qp, glow, outline, eN, baseR, box, PASS_GLOW);
    const c = g && bake(spec, qp, glow, outline, eN, baseR, box, PASS_CORE);
    if (!c) {
        // Atlas full mid-pair. Wipe and let the next frame refill; returning
        // null keeps this agent on the vector path rather than drawing half of
        // it.
        resetAtlas();
        return null;
    }

    builds_this_frame++;
    e = { glow: g, core: c };
    entries.set(key, e);
    return e;
}

/** The atlas canvas, for `drawImage`. Null until the first sprite is built. */
export function atlasCanvas() { return atlas; }
