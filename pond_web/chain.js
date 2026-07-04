// Kinematic follow-chain: each segment trails the previous one, clamped to a
// fixed link distance. Pure — no globals, no canvas, no wasm buffer access.

/** New chain with all segments pinned at (x, y). */
export function createChain(x, y, segCount) {
    return { segs: Array.from({ length: segCount }, () => ({ x, y })) };
}

/**
 * Move the head to (hx, hy) and let the rest of the chain follow.
 * `spec` = { segCount, segDist, gridSize } — segCount/segDist are per-agent
 * (from MorphSpec), gridSize is world size for toroidal wrap.
 */
export function updateChain(chain, hx, hy, spec) {
    const { segCount, segDist, gridSize } = spec;

    if (chain.segs.length !== segCount) {
        // Morph spec changed segment count (shouldn't happen mid-life, but
        // keep the chain consistent rather than indexing out of bounds).
        chain.segs = Array.from({ length: segCount }, (_, s) =>
            chain.segs[Math.min(s, chain.segs.length - 1)] ?? { x: hx, y: hy });
    }

    chain.segs[0].x = hx;
    chain.segs[0].y = hy;

    for (let s = 1; s < segCount; s++) {
        const ps = chain.segs[s - 1];
        const cs = chain.segs[s];
        let dx = cs.x - ps.x;
        let dy = cs.y - ps.y;
        if (dx >  gridSize * 0.5) dx -= gridSize;
        if (dx < -gridSize * 0.5) dx += gridSize;
        if (dy >  gridSize * 0.5) dy -= gridSize;
        if (dy < -gridSize * 0.5) dy += gridSize;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001;
        if (dist > segDist) {
            const ratio = segDist / dist;
            // Keep stored coords in [0, gridSize) so downstream consumers can
            // rely on wrapped positions; seam-crossing is handled by wrapDelta.
            cs.x = (((ps.x + dx * ratio) % gridSize) + gridSize) % gridSize;
            cs.y = (((ps.y + dy * ratio) % gridSize) + gridSize) % gridSize;
        }
    }

    return chain;
}
