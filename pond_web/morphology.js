// Trait → shape mapping. Single place where "what does aggression look like"
// lives. Input is the already-normalized [0,1] morphology knobs computed
// Rust-side (pond_core/src/morphology.rs) from raw genome traits — this file
// never sees a raw trait value, only shape-ready scalars.

// Segment count varies with body plan instead of being fixed at 7. This is the
// single largest change available to the silhouette — a 5-segment stub and an
// 11-segment eel read as different animals, where proportion tweaks on a fixed
// skeleton only ever read as the same animal resized.
const MIN_SEG_COUNT = 5;
const MAX_SEG_COUNT = 11;
// Body length = (segCount-1) * segDist world tiles. Kept near the old ~0.66–1.06
// range at the short end; long bodies get a shorter per-segment distance so
// eels stay swimmable rather than overcrowding the grid.
const BASE_SEG_DIST = 0.11;

// Profile control points, resampled to whatever segCount the plan asks for.
const ROUND_PROFILE = [0.85, 1.05, 1.25, 1.15, 0.95, 0.65, 0.35];
const SPEAR_PROFILE = [0.55, 0.75, 0.95, 0.85, 0.55, 0.30, 0.12];

function lerp(a, b, t) { return a + (b - a) * t; }

/** Sample a 7-point profile at `n` evenly spaced positions. */
function resample(profile, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        const t = (i / (n - 1)) * (profile.length - 1);
        const lo = Math.floor(t);
        const hi = Math.min(lo + 1, profile.length - 1);
        out[i] = lerp(profile[lo], profile[hi], t - lo);
    }
    return out;
}

// Discrete body plans. The dominant trait picks the family, so a lineage that
// evolves toward aggression doesn't just get pointier — it becomes a different
// silhouette. Each entry biases segment count and proportions on top of the
// continuous knobs.
const PLANS = {
    spear: { segBias: -0.25, distMul: 1.30, widthMul: 0.82, finCount: 4 },
    eel:   { segBias: +0.45, distMul: 0.72, widthMul: 0.70, finCount: 2 },
    tank:  { segBias: -0.30, distMul: 1.05, widthMul: 1.35, finCount: 2 },
    scout: { segBias: +0.05, distMul: 0.95, widthMul: 0.95, finCount: 2 },
};

/** Dominant trait → body plan. Ties resolve in listed order. */
function choosePlan(pointiness, elongation, bulk, eyeSize) {
    const ranked = [
        ['spear', pointiness],
        ['eel', elongation],
        ['tank', bulk],
        ['scout', eyeSize],
    ].sort((a, b) => b[1] - a[1]);
    return ranked[0][0];
}

/** morph: { pointiness, elongation, bulk, ornament, eyeSize, pulseRate, belly } (all 0..1) */
export function deriveMorphology(morph) {
    const { pointiness, elongation, bulk, ornament, eyeSize, pulseRate, belly } = morph;

    const planName = choosePlan(pointiness, elongation, bulk, eyeSize);
    const plan = PLANS[planName];

    const segT = Math.max(0, Math.min(1, elongation + plan.segBias));
    const segCount = Math.round(lerp(MIN_SEG_COUNT, MAX_SEG_COUNT, segT));

    // aggression: mass shifts forward + narrows toward a spear profile as it rises
    const round = resample(ROUND_PROFILE, segCount);
    const spear = resample(SPEAR_PROFILE, segCount);
    const mid = (segCount - 1) * 0.35;   // where the midsection sits for this length
    const envelope = round.map((r, s) => {
        let v = lerp(r, spear[s], pointiness) * plan.widthMul;
        // Bell around the midsection rather than fixed indices, so bulk and
        // belly land in the right place at any segment count.
        const near = Math.max(0, 1 - Math.abs(s - mid) / 1.6);
        v *= 1 + bulk * 0.45 * near;     // defense: wide midsection
        v *= 1 + belly * 0.35 * near;    // energy_capacity: belly bulge
        return v;
    });

    return {
        plan: planName,
        segCount,
        segDist: BASE_SEG_DIST * (1 + elongation * 1.4) * plan.distMul,
        envelope,
        headPointiness: pointiness,                        // aggression: wedge vs round cap
        taperPower: 1 + elongation * 1.5,                  // speed: harder tail falloff
        fins: {
            count: plan.finCount,
            rake: pointiness * 0.6,                        // aggression: swept-back fins
            len: 0.14 + pointiness * 0.12,
        },
        ornamentLen: ornament * 0.16,                      // attack: head spikes
        ornamentPairs: ornament > 0.55 ? 2 : 1,            // attack: spike count
        armorBumps: bulk > 0.4 ? Math.round(bulk * 4) : 0, // defense: scalloped hull
        eyeSize: 0.28 * (1 + eyeSize * 0.6),               // vision: eye size
        eyeSpread: 0.45 * (1 + eyeSize * 0.4),             // vision: lateral spread
        glowIntensity: 1.0,
        pulseRate: 0.5 + pulseRate * 2.0,                  // metabolism: glow pulse rate
    };
}
