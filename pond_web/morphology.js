// Trait → shape mapping. Single place where "what does aggression look like"
// lives. Input is the already-normalized [0,1] morphology knobs computed
// Rust-side (pond_core/src/morphology.rs) from raw genome traits — this file
// never sees a raw trait value, only shape-ready scalars.

const BASE_SEG_COUNT = 7;
// Body length = (segCount-1) * segDist * (1 + elongation*0.6) world tiles.
// 0.11 gives ~0.66–1.06 tiles head-to-tail; 0.28 produced 1.7–2.7-tile bodies
// that overcrowded the grid.
const BASE_SEG_DIST = 0.11;

// Size envelope: head(0) → body peak → tail(6), relative to base_r.
const ROUND_ENVELOPE = [0.85, 1.05, 1.25, 1.15, 0.95, 0.65, 0.35];
const SPEAR_ENVELOPE = [0.55, 0.75, 0.95, 0.85, 0.55, 0.30, 0.12];

function lerp(a, b, t) { return a + (b - a) * t; }

/** morph: { pointiness, elongation, bulk, ornament, eyeSize, pulseRate, belly } (all 0..1) */
export function deriveMorphology(morph) {
    const { pointiness, elongation, bulk, ornament, eyeSize, pulseRate, belly } = morph;

    // aggression: mass shifts forward + narrows toward a spear profile as it rises
    const envelope = ROUND_ENVELOPE.map((r, s) => {
        let v = lerp(r, SPEAR_ENVELOPE[s], pointiness);
        if (s === 2 || s === 3) v *= 1 + bulk * 0.35;   // defense: wide midsection
        if (s === 2) v *= 1 + belly * 0.25;             // energy_capacity: belly bulge
        return v;
    });

    return {
        segCount: BASE_SEG_COUNT,
        segDist: BASE_SEG_DIST * (1 + elongation * 0.6),   // speed: elongation
        envelope,
        headPointiness: pointiness,                        // aggression: wedge vs round cap
        taperPower: 1 + elongation * 1.5,                  // speed: harder tail falloff
        fins: {
            count: 2,
            rake: pointiness * 0.6,                        // aggression: swept-back fins
            len: 0.14 + pointiness * 0.12,
        },
        ornamentLen: ornament * 0.16,                      // attack: head spikes
        armorBumps: bulk > 0.4 ? Math.round(bulk * 4) : 0, // defense: scalloped hull
        eyeSize: 0.28 * (1 + eyeSize * 0.6),               // vision: eye size
        eyeSpread: 0.45 * (1 + eyeSize * 0.4),             // vision: lateral spread
        glowIntensity: 1.0,
        pulseRate: 0.5 + pulseRate * 2.0,                  // metabolism: glow pulse rate
    };
}
