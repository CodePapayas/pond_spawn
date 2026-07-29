// Oklab/Oklch ↔ sRGB conversion + perceptual color smoothing. Cluster colors
// crossfade through Oklch (polar Oklab) along the shortest hue arc, so even
// near-complementary transitions (teal→magenta) rotate through vivid hues
// instead of passing through gray mud (which straight-line lerp does).

function srgbToLinear(c) {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c) {
    const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/** [r,g,b] 0–255 → [L,a,b] Oklab. */
export function rgbToOklab([r8, g8, b8]) {
    const r = srgbToLinear(r8), g = srgbToLinear(g8), b = srgbToLinear(b8);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ];
}

/** [L,a,b] Oklab → [r,g,b] 0–255. */
export function oklabToRgb([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
    return [
        linearToSrgb(+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
    ];
}

/** [r,g,b] 0–255 → [L,C,h] Oklch (h in radians). */
export function rgbToOklch(rgb) {
    const [L, a, b] = rgbToOklab(rgb);
    return [L, Math.sqrt(a * a + b * b), Math.atan2(b, a)];
}

/** [L,C,h] Oklch → [r,g,b] 0–255. */
export function oklchToRgb([L, C, h]) {
    return oklabToRgb([L, C * Math.cos(h), C * Math.sin(h)]);
}

// ── Genome → colour ───────────────────────────────────────────────────────────
//
// Colour is a continuous function of the genome rather than a lookup on the
// k-means cluster label. Two consequences worth knowing:
//
//  1. Within-family variation becomes visible. Keyed to the label, ~75% of a
//     converged pond rendered as one identical RGB triple.
//  2. Colour can no longer "switch". A label is a step function, so any
//     reassignment was a jump that smoothing could soften but not remove;
//     traits only drift by small mutation steps, so hue drifts with them.
//
// Traits are immutable for an agent's life, so the base colour is constant and
// needs no temporal smoothing at all — only the live energy term varies, and
// that already moves continuously.
//
// Agent colour comes from three fixed ramps rather than a free hue sweep:
//
//   lime    — passive
//   cyan    — middle
//   magenta — aggressive
//
// A free arc put most of the pond in whatever hues sat mid-sweep, which is how
// it ended up orange. Anchoring to three ramps means the strategy a lineage has
// evolved is readable at a glance, and no colour outside the palette can occur.
//
// Ramps are interpolated, not snapped to: traits drift by small mutation steps,
// so an agent between passive and middling reads as a blend rather than jumping
// between bands. Interpolation runs in Oklch, so lime→cyan passes through
// vivid greens instead of the grey mud a straight RGB lerp would give.
const RAMP_LIME = ['#F8FF00', '#D0D600', '#A9AE00', '#878B00', '#636600', '#3F4100'];
const RAMP_CYAN = ['#00F8FF', '#00CDD2', '#00A3A8', '#007E82', '#00585B', '#003234'];
const RAMP_MAGENTA = ['#FFDFFB', '#FF98F7', '#FF00F8', '#CB00C5', '#950091', '#5F005C'];

function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Pre-converted once: the ramps are constant, and rgbToOklch per agent per
// frame would be pure waste.
const RAMPS = [RAMP_LIME, RAMP_CYAN, RAMP_MAGENTA]
    .map(ramp => ramp.map(hex => rgbToOklch(hexToRgb(hex))));

/** Sample one ramp at a continuous position, blending its two nearest stops. */
function sampleRamp(ramp, pos) {
    const t = Math.min(ramp.length - 1, Math.max(0, pos));
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, ramp.length - 1);
    return lerpOklch(ramp[lo], ramp[hi], t - lo);
}

/** Blend two Oklch colours, hue along the shortest arc. */
function lerpOklch(a, b, t) {
    let dh = b[2] - a[2];
    if (dh > Math.PI) dh -= 2 * Math.PI;
    if (dh < -Math.PI) dh += 2 * Math.PI;
    return [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + dh * t,
    ];
}

// Where each ramp's vivid anchor sits. The lime and cyan ramps open on their
// brightest stop, but the magenta ramp opens on two pale tints — starting there
// made aggressive lineages read as washed-out pink rather than hot magenta, so
// magenta starts two stops in and its light tints go unused.
const RAMP_START = [0, 0, 2];
// How far past that anchor a slow-metabolism agent sits. Kept shallow: the
// bottom stops are dark enough to disappear against the water, and live energy
// already dims the colour further at render time.
const SHADE_DEPTH = 2.0;

/**
 * morph: normalized [0,1] knobs from pond_core (see morphology.rs).
 * Returns base [L, C, h] — multiply L by live energy before rendering.
 */
export function genomeColor(morph) {
    const { pointiness, ornament, bulk, pulseRate } = morph;

    // Strategy ← combat profile. Aggression dominates because it's the trait
    // that most defines a lineage's strategy; attack and defense shade it.
    const combat = Math.min(1, Math.max(0,
        pointiness * 0.55 + ornament * 0.25 + (1 - bulk) * 0.20));

    // Position along its ramp ← burn rate. Fast-metabolism lineages sit at the
    // bright end, slow ones deeper down.
    const depth = (1 - pulseRate) * SHADE_DEPTH;

    // lime → cyan over the first half of the range, cyan → magenta over the
    // second, so the three anchors land at passive / middle / aggressive.
    const half = combat < 0.5 ? 0 : 1;
    const t = combat < 0.5 ? combat * 2 : (combat - 0.5) * 2;
    return lerpOklch(
        sampleRamp(RAMPS[half], RAMP_START[half] + depth),
        sampleRamp(RAMPS[half + 1], RAMP_START[half + 1] + depth),
        t,
    );
}

/**
 * Frame-rate-independent exponential smoother over Oklch.
 * `state` = current [L,C,h] (mutated in place), `target` = target [L,C,h],
 * `dtMs` = frame delta, `tauMs` = time constant (~63% converged per tau).
 * Hue takes the shortest arc, so chroma stays high mid-fade.
 */
export function smoothOklch(state, target, dtMs, tauMs) {
    const k = 1 - Math.exp(-dtMs / tauMs);
    state[0] += (target[0] - state[0]) * k;
    state[1] += (target[1] - state[1]) * k;
    let dh = target[2] - state[2];
    if (dh > Math.PI) dh -= 2 * Math.PI;
    if (dh < -Math.PI) dh += 2 * Math.PI;
    state[2] += dh * k;
    return state;
}
