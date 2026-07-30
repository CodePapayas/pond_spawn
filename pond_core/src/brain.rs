use rand::Rng;

/// Layer widths, input first. One source for the shape: the weight count, every
/// slice offset, the wasm export the inspector draws from, and `initial_weights`
/// all derive from this array.
///
/// It was eight hand-computed offset constants with their arithmetic in
/// trailing comments. That is fine until the input width changes, at which point
/// all eight move and any one of them getting missed reads the next layer's
/// weights as this layer's biases — silently, since the shapes still line up.
pub const LAYER_SIZES: [usize; 5] = [INPUT_COUNT, 12, 12, 12, 8];

/// Inputs to the network. See `World::perceive` for what each one carries.
pub const INPUT_COUNT: usize = 7;
/// Outputs. Indices are named in `world.rs` (`OUT_SEEK`…`OUT_SLEEP`).
pub const OUTPUT_COUNT: usize = 8;

/// Total parameter count: `7→12→12→12→8` with biases, so
/// `(7×12+12) + (12×12+12) + (12×12+12) + (12×8+8)` = 604.
pub const WEIGHT_COUNT: usize = weight_count();

const fn weight_count() -> usize {
    let mut total = 0;
    let mut l = 0;
    while l + 1 < LAYER_SIZES.len() {
        total += LAYER_SIZES[l] * LAYER_SIZES[l + 1] + LAYER_SIZES[l + 1];
        l += 1;
    }
    total
}

/// Start of layer `l`'s weight block in the flat buffer.
const fn layer_w(l: usize) -> usize {
    let mut off = 0;
    let mut i = 0;
    while i < l {
        off += LAYER_SIZES[i] * LAYER_SIZES[i + 1] + LAYER_SIZES[i + 1];
        i += 1;
    }
    off
}
/// Start of layer `l`'s bias block: its weights end there.
const fn layer_b(l: usize) -> usize {
    layer_w(l) + LAYER_SIZES[l] * LAYER_SIZES[l + 1]
}

const L0_W: usize = layer_w(0);
const L0_B: usize = layer_b(0);
const L1_W: usize = layer_w(1);
const L1_B: usize = layer_b(1);
const L2_W: usize = layer_w(2);
const L2_B: usize = layer_b(2);
const L3_W: usize = layer_w(3);
const L3_B: usize = layer_b(3);

/// Weights stored row-major [out, in] matching PyTorch nn.Linear weight layout.
/// `output[i] = sum_j(w[i * IN + j] * input[j]) + bias[i]`
#[inline(always)]
fn linear<const IN: usize, const OUT: usize>(
    input: &[f32; IN],
    w: &[f32],
    b: &[f32],
    out: &mut [f32; OUT],
) {
    for i in 0..OUT {
        let mut sum = b[i];
        for j in 0..IN {
            sum += w[i * IN + j] * input[j];
        }
        out[i] = sum;
    }
}

#[inline(always)]
fn relu_inplace<const N: usize>(x: &mut [f32; N]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Forward pass: 7 → 12 (ReLU) → 12 (ReLU) → 12 (ReLU) → 8 logits.
/// Returns raw logits; the caller applies `sigmoid_outputs`.
pub fn forward(weights: &[f32; WEIGHT_COUNT], input: [f32; INPUT_COUNT]) -> [f32; 8] {
    forward_traced(weights, input).3
}

/// Same forward pass, but returns every layer's post-activation values.
/// Shares the exact code path with `forward` (which delegates here) so the
/// inspector can never drift from what the sim actually computes.
pub fn forward_traced(
    weights: &[f32; WEIGHT_COUNT],
    input: [f32; INPUT_COUNT],
) -> ([f32; 12], [f32; 12], [f32; 12], [f32; 8]) {
    let mut h0 = [0f32; 12];
    linear::<INPUT_COUNT, 12>(&input, &weights[L0_W..L0_B], &weights[L0_B..L1_W], &mut h0);
    relu_inplace(&mut h0);

    let mut h1 = [0f32; 12];
    linear::<12, 12>(&h0, &weights[L1_W..L1_B], &weights[L1_B..L2_W], &mut h1);
    relu_inplace(&mut h1);

    let mut h2 = [0f32; 12];
    linear::<12, 12>(&h1, &weights[L2_W..L2_B], &weights[L2_B..L3_W], &mut h2);
    relu_inplace(&mut h2);

    let mut logits = [0f32; 8];
    linear::<12, 8>(&h2, &weights[L3_W..L3_B], &weights[L3_B..WEIGHT_COUNT], &mut logits);
    (h0, h1, h2, logits)
}

/// Initialize weights matching `Brain.initial_weights()` in brain.py.
/// Weights uniform(-0.5, 0.5); biases fixed at 0.001.
/// Draw order: for each linear layer — weight floats first, then bias floats.
pub fn initial_weights(rng: &mut impl Rng) -> Vec<f32> {
    let mut buf = Vec::with_capacity(WEIGHT_COUNT);
    for l in 0..LAYER_SIZES.len() - 1 {
        let (in_size, out_size) = (LAYER_SIZES[l], LAYER_SIZES[l + 1]);
        for _ in 0..(in_size * out_size) {
            buf.push(rng.gen_range(-0.5_f32..=0.5));
        }
        for _ in 0..out_size {
            buf.push(0.001_f32);
        }
    }
    debug_assert_eq!(buf.len(), WEIGHT_COUNT);
    buf
}

/// Softmax over a fixed-size slice (numerically stable).
///
/// Dead in the live sim: the continuous-space physics refactor (Pass A)
/// replaced discrete softmax-sampled action selection with independent
/// sigmoid gates (`sigmoid_outputs`), because steering needs several
/// simultaneous continuous force weights, not one mutually-exclusive action
/// drawn from a probability simplex. Kept for the Python-parity golden harness
/// and possible future argmax/discrete-mode experiments.
#[allow(dead_code)]
pub fn softmax(logits: [f32; 8]) -> [f32; 8] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = [0f32; 8];
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        exps[i] = (v - max).exp();
        sum += exps[i];
    }
    for v in exps.iter_mut() {
        *v /= sum;
    }
    exps
}

/// Multinomial sample from a probability distribution (softmax output).
///
/// Dead in the live sim alongside [`softmax`] — see its note. The sigmoid-gate
/// steering path never samples a single action.
#[allow(dead_code)]
pub fn sample_action(probs: [f32; 8], rng: &mut impl Rng) -> usize {
    let roll: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if roll < cumsum {
            return i;
        }
    }
    7 // fallback: last action
}

/// Element-wise sigmoid over raw logits — used by steering system.
/// Each output is independent in [0, 1]; represents a behavior weight or trigger gate.
pub fn sigmoid_outputs(logits: [f32; 8]) -> [f32; 8] {
    logits.map(|x| 1.0 / (1.0 + (-x).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn weight_count_matches_the_layer_table() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let w = initial_weights(&mut rng);
        assert_eq!(w.len(), WEIGHT_COUNT);
        // Derived, not asserted against a literal — but pin the current shape so
        // a change to LAYER_SIZES is a deliberate act with a schema bump, not a
        // silent one. 7→12→12→12→8 with biases: two extra inputs over the old
        // 488 add 2×12 weights, so 512. (pond_core/README said 604 for this
        // shape; that number was never right.)
        assert_eq!(WEIGHT_COUNT, 512);
        assert_eq!(LAYER_SIZES, [7, 12, 12, 12, 8]);
    }

    #[test]
    fn layer_offsets_tile_the_buffer_exactly() {
        // Each block starts where the previous one ended, and the last ends at
        // the buffer's end. This is what the eight hand-computed constants used
        // to promise in comments.
        let bounds = [
            (L0_W, L0_B, L1_W), (L1_W, L1_B, L2_W),
            (L2_W, L2_B, L3_W), (L3_W, L3_B, WEIGHT_COUNT),
        ];
        let mut cursor = 0;
        for (l, &(w_start, b_start, next)) in bounds.iter().enumerate() {
            assert_eq!(w_start, cursor, "layer {} weights do not follow the previous block", l);
            assert_eq!(b_start - w_start, LAYER_SIZES[l] * LAYER_SIZES[l + 1]);
            assert_eq!(next - b_start, LAYER_SIZES[l + 1], "layer {} bias block is wrong", l);
            cursor = next;
        }
        assert_eq!(cursor, WEIGHT_COUNT);
    }

    #[test]
    fn initial_weights_ranges() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let w = initial_weights(&mut rng);
        // Bias blocks, derived from the layer table.
        let bias_ranges = [(L0_B, L1_W), (L1_B, L2_W), (L2_B, L3_W), (L3_B, WEIGHT_COUNT)];
        let bias_positions: std::collections::HashSet<usize> =
            bias_ranges.iter().flat_map(|&(s, e)| s..e).collect();
        for (i, &v) in w.iter().enumerate() {
            if bias_positions.contains(&i) {
                assert!((v - 0.001).abs() < 1e-6, "bias[{}]={} != 0.001", i, v);
            } else {
                assert!((-0.5..=0.5).contains(&v), "weight[{}]={} out of range", i, v);
            }
        }
    }

    #[test]
    fn forward_output_shape() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let w: Vec<f32> = initial_weights(&mut rng);
        let weights: &[f32; WEIGHT_COUNT] = w.as_slice().try_into().unwrap();
        let input = [0.5f32, 0.3, 0.7, 0.4, 0.6, 1.0, 0.0];
        let out = forward(weights, input);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn forward_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let w: Vec<f32> = initial_weights(&mut rng);
        let weights: &[f32; WEIGHT_COUNT] = w.as_slice().try_into().unwrap();
        let input = [1.0f32, 0.0, 0.5, 0.2, 0.8, 1.0, 0.0];
        let a = forward(weights, input);
        let b = forward(weights, input);
        assert_eq!(a, b);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = [1.0f32, 2.0, 3.0, 0.5, -1.0, 0.0, 1.5, 2.5];
        let probs = softmax(logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for &p in &probs {
            assert!(p >= 0.0);
        }
    }
}
