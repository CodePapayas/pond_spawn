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
