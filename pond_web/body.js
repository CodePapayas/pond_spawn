// Draws a MorphSpec body along a kinematic chain as one continuous hull
// silhouette instead of per-segment circles. Knows nothing about traits —
// consumes only { chain, spec, palette, xform, motion }.
//
// The world→screen transform is uniform (square pond, letterboxed), so the
// per-axis projection below collapses to a single scale. Hull geometry is still
// built in screen space from `scale_px` rather than in world units, which keeps
// body proportions independent of the transform.
//
// Perf: one glow hull path + one core hull path (+ small trait ornaments),
// ~2 draw calls/agent regardless of segment count.

const GLOW_SCALE = 1.9;

function perp(dir) { return { x: -dir.y, y: dir.x }; }

function wrapDelta(d, gridSize) {
    if (d >  gridSize * 0.5) return d - gridSize;
    if (d < -gridSize * 0.5) return d + gridSize;
    return d;
}

/** Per-segment forward direction (screen space, unit length). */
function segmentDirs(centers, segCount, velXpx, velYpx) {
    const dirs = new Array(segCount);
    for (let s = 0; s < segCount; s++) {
        const a = centers[Math.max(s - 1, 0)];
        const b = centers[Math.min(s + 1, segCount - 1)];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        dirs[s] = { x: dx / len, y: dy / len };
    }
    const vlen = Math.sqrt(velXpx * velXpx + velYpx * velYpx);
    if (vlen > 0.01) dirs[0] = { x: velXpx / vlen, y: velYpx / vlen };
    return dirs;
}

/** Build the closed hull path (head cap → left side → tail → right side). */
function buildHullPath(ctx, left, right, headCenter, headDir, headWidth, headPointiness, widthScale) {
    const n = left.length;
    const apexDist = headWidth * widthScale * (0.5 + headPointiness * 1.4);
    const apex = { x: headCenter.x + headDir.x * apexDist, y: headCenter.y + headDir.y * apexDist };

    ctx.beginPath();
    ctx.moveTo(right[0].x, right[0].y);
    ctx.quadraticCurveTo(apex.x, apex.y, left[0].x, left[0].y);

    for (let s = 1; s < n; s++) {
        const midX = (left[s - 1].x + left[s].x) / 2;
        const midY = (left[s - 1].y + left[s].y) / 2;
        ctx.quadraticCurveTo(left[s - 1].x, left[s - 1].y, midX, midY);
    }
    ctx.lineTo(left[n - 1].x, left[n - 1].y);
    ctx.lineTo(right[n - 1].x, right[n - 1].y);

    for (let s = n - 2; s >= 0; s--) {
        const midX = (right[s + 1].x + right[s].x) / 2;
        const midY = (right[s + 1].y + right[s].y) / 2;
        ctx.quadraticCurveTo(right[s + 1].x, right[s + 1].y, midX, midY);
    }
    ctx.closePath();
}

/**
 * chain:  { segs: [{x,y}, ...] } in world units
 * spec:   MorphSpec from morphology.js
 * palette: [r, g, b]
 * xform:  { tile_w, tile_h, scale_px, off_x, off_y, gridSize }
 * motion: { baseR, energyNorm, velX, velY, timeSec }
 */
/**
 * @param {{rgb: number[], weight: number}} [glow]  strategy halo — warm and
 *   bright for an aggressive lineage, cool and faint for a passive one. Omitted
 *   (title screen, previews) the halo falls back to the body colour, which is
 *   what it always used to be.
 */
export function drawBody(ctx, chain, spec, palette, xform, motion, glow) {
    const { segCount, envelope, headPointiness, fins, ornamentLen, ornamentPairs, armorBumps, eyeSize, eyeSpread, pulseRate } = spec;
    const { tile_w, tile_h, scale_px, off_x, off_y, gridSize } = xform;
    const { baseR, energyNorm, velX, velY, timeSec } = motion;
    const [cr, cg, cb] = palette;

    const toScreen = (wx, wy) => ({ x: off_x + wx * tile_w, y: off_y + wy * tile_h });

    // Unwrap segments relative to the head so a body straddling the toroidal
    // seam projects as one contiguous shape instead of a hull stretched
    // across the whole grid. Unwrapped coords may run slightly outside
    // [0, gridSize); seam copies below cover the wrapped side.
    const world = new Array(segCount);
    world[0] = { x: chain.segs[0].x, y: chain.segs[0].y };
    for (let s = 1; s < segCount; s++) {
        world[s] = {
            x: world[s - 1].x + wrapDelta(chain.segs[s].x - chain.segs[s - 1].x, gridSize),
            y: world[s - 1].y + wrapDelta(chain.segs[s].y - chain.segs[s - 1].y, gridSize),
        };
    }
    const centers = world.map(seg => toScreen(seg.x, seg.y));
    const dirs = segmentDirs(centers, segCount, velX * tile_w, velY * tile_h);

    const left = new Array(segCount);
    const right = new Array(segCount);
    for (let s = 0; s < segCount; s++) {
        const w = envelope[s] * baseR;
        const p = perp(dirs[s]);
        left[s]  = { x: centers[s].x + p.x * w, y: centers[s].y + p.y * w };
        right[s] = { x: centers[s].x - p.x * w, y: centers[s].y - p.y * w };
    }

    const pulse = 0.5 + 0.5 * Math.sin(timeSec * pulseRate * Math.PI * 2);
    const headWidth = envelope[0] * baseR;

    const paint = () => {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Glow pass: same hull, scaled outward, low alpha.
    //
    // This carries *strategy* now. Hue belongs to the lineage, so the halo is
    // where "what does this animal do" went: warm and strong for an aggressive
    // one, cool and faint for a passive one. It used to be a brighter copy of
    // the body, which carried no information at all.
    const glowLeft = left.map((pt, s) => scaleFromCenter(pt, centers[s], GLOW_SCALE));
    const glowRight = right.map((pt, s) => scaleFromCenter(pt, centers[s], GLOW_SCALE));
    buildHullPath(ctx, glowLeft, glowRight, centers[0], dirs[0], headWidth, headPointiness, GLOW_SCALE);
    const [gr, gg, gb] = glow?.rgb ?? [cr, cg, cb];
    const glowA = (0.10 + energyNorm * 0.08 + pulse * 0.06) * (glow?.weight ?? 1);
    ctx.fillStyle = `rgba(${gr},${gg},${gb},${glowA})`;
    ctx.fill();

    // Core pass. Additive compositing saturates a bright fill straight to white
    // over lit water, which erased the genome colour entirely — so only the glow
    // above is additive and the body itself paints normally.
    ctx.globalCompositeOperation = 'source-over';
    buildHullPath(ctx, left, right, centers[0], dirs[0], headWidth, headPointiness, 1.0);
    const coreA = 0.72 + energyNorm * 0.28;
    ctx.fillStyle = `rgba(${cr},${cg},${cb},${coreA})`;
    ctx.fill();

    // Armor scalloping (defense): rings around the mid-body hull points.
    if (armorBumps > 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.16)';
        ctx.lineWidth = 0.8;
        for (let b = 0; b < armorBumps; b++) {
            // One ring per segment, walking back from the shoulder. This used to
            // be `2 + (b % 2)`, so every count above two drew over the same two
            // segments and the number was decorative — three rings and four
            // rings were the same animal.
            const s = Math.min(2 + b, segCount - 1);
            // Hugs the hull. At 1.08 the widened bulk/belly envelope pushed these
            // into big free-floating rings that read as UI, not anatomy.
            const r = envelope[s] * baseR * 0.92;
            ctx.beginPath();
            ctx.arc(centers[s].x, centers[s].y, r, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // Head spikes (attack): short strokes outward from the head hull points, one
    // pair per segment walking back. `ornamentPairs` is 0-3 and **0 means no
    // spikes at all** — presence or absence is the categorical read, where
    // "slightly longer spikes" is not something anyone can see.
    if (ornamentPairs > 0) {
        const spikeLen = ornamentLen * scale_px;
        ctx.strokeStyle = `rgba(${cr},${cg},${cb},0.9)`;
        ctx.lineWidth = 1.2;
        const anchors = [];
        for (let p = 0; p < ornamentPairs && p < segCount; p++) {
            anchors.push({ hp: left[p], seg: p }, { hp: right[p], seg: p });
        }
        for (let ai = 0; ai < anchors.length; ai++) {
            const { hp, seg } = anchors[ai];
            const c = centers[seg];
            const dx = hp.x - c.x, dy = hp.y - c.y;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            ctx.beginPath();
            ctx.moveTo(hp.x, hp.y);
            ctx.lineTo(hp.x + (dx / len) * spikeLen, hp.y + (dy / len) * spikeLen);
            ctx.stroke();
        }
    }

    // Fins (aggression): swept-back triangles. `fins.count` is a total, drawn as
    // count/2 pairs spread down the rear half of the body — a 4-fin spear reads
    // very differently from a 2-fin eel at the same proportions.
    // Finless is a real body plan now, so this is gated on the count rather than
    // on a length that was never zero.
    if (fins.count > 0 && fins.len > 0.01) {
        const pairs = Math.round(fins.count / 2);
        const finLen = fins.len * scale_px;
        const sweep = fins.rake;
        for (let f = 0; f < pairs; f++) {
            const frac = pairs === 1 ? 0.62 : 0.45 + (f / (pairs - 1)) * 0.35;
            const s = Math.min(segCount - 1, Math.max(1, Math.round(frac * (segCount - 1))));
            const p = perp(dirs[s]);
            for (const side of [-1, 1]) {
                const baseX = centers[s].x + p.x * envelope[s] * baseR * side;
                const baseY = centers[s].y + p.y * envelope[s] * baseR * side;
                const tipX = baseX + (p.x * side * (1 - sweep) - dirs[s].x * sweep) * finLen;
                const tipY = baseY + (p.y * side * (1 - sweep) - dirs[s].y * sweep) * finLen;
                ctx.fillStyle = `rgba(${cr},${cg},${cb},0.5)`;
                ctx.beginPath();
                ctx.moveTo(baseX, baseY);
                ctx.lineTo(tipX, tipY);
                ctx.lineTo(centers[s].x + dirs[s].x * baseR * 0.3, centers[s].y + dirs[s].y * baseR * 0.3);
                ctx.closePath();
                ctx.fill();
            }
        }
    }

    // Eyes (vision): two dots offset perpendicular to heading at the head.
    const eyeR = headWidth * eyeSize;
    const eyeFwd = headWidth * 0.55;
    const eyeLat = headWidth * eyeSpread;
    const p0 = perp(dirs[0]);
    for (const side of [-1, 1]) {
        const ex = centers[0].x + dirs[0].x * eyeFwd + p0.x * eyeLat * side;
        const ey = centers[0].y + dirs[0].y * eyeFwd + p0.y * eyeLat * side;
        ctx.fillStyle = 'rgba(255,255,255,0.90)';
        ctx.beginPath();
        ctx.arc(ex, ey, eyeR, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.restore();
    };

    // Paint once, plus shifted copies when the unwrapped body extends past a
    // grid edge, so seam-crossing bodies appear on both sides. The caller
    // clips to the grid rect, so off-grid copies cost nothing visible.
    const margin_x = (baseR * 3.5) / tile_w;
    const margin_y = (baseR * 3.5) / tile_h;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of world) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }
    const xOffs = [0];
    if (minX - margin_x < 0) xOffs.push(gridSize);
    if (maxX + margin_x > gridSize) xOffs.push(-gridSize);
    const yOffs = [0];
    if (minY - margin_y < 0) yOffs.push(gridSize);
    if (maxY + margin_y > gridSize) yOffs.push(-gridSize);

    for (const ox of xOffs) {
        for (const oy of yOffs) {
            if (ox === 0 && oy === 0) { paint(); continue; }
            ctx.save();
            ctx.translate(ox * tile_w, oy * tile_h);
            paint();
            ctx.restore();
        }
    }
}

function scaleFromCenter(pt, center, scale) {
    return {
        x: center.x + (pt.x - center.x) * scale,
        y: center.y + (pt.y - center.y) * scale,
    };
}
