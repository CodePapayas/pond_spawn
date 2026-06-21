use rand::Rng;

/// Total parameter count matching `brains/brain.json`: 5→12→12→12→8 with biases.
/// Layout: [w(5×12)=60, b(12)=12, w(12×12)=144, b(12)=12, w(12×12)=144, b(12)=12, w(12×8)=96, b(8)=8]
pub const WEIGHT_COUNT: usize = 488;

// Slice offsets into the flat weight buffer
const L0_W: usize = 0;
const L0_B: usize = 60;   // 60 + 12 = 72
const L1_W: usize = 72;
const L1_B: usize = 216;  // 72 + 144 + 12 = 228
const L2_W: usize = 228;
const L2_B: usize = 372;  // 228 + 144 + 12 = 384
const L3_W: usize = 384;
const L3_B: usize = 480;  // 384 + 96 + 8 = 488

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

/// Forward pass: 5 → 12 (ReLU) → 12 (ReLU) → 12 (ReLU) → 8 logits.
/// Returns raw logits; caller applies softmax + multinomial for action selection.
pub fn forward(weights: &[f32; WEIGHT_COUNT], input: [f32; 5]) -> [f32; 8] {
    let mut h0 = [0f32; 12];
    linear::<5, 12>(&input, &weights[L0_W..L0_B], &weights[L0_B..L1_W], &mut h0);
    relu_inplace(&mut h0);

    let mut h1 = [0f32; 12];
    linear::<12, 12>(&h0, &weights[L1_W..L1_B], &weights[L1_B..L2_W], &mut h1);
    relu_inplace(&mut h1);

    let mut h2 = [0f32; 12];
    linear::<12, 12>(&h1, &weights[L2_W..L2_B], &weights[L2_B..L3_W], &mut h2);
    relu_inplace(&mut h2);

    let mut logits = [0f32; 8];
    linear::<12, 8>(&h2, &weights[L3_W..L3_B], &weights[L3_B..WEIGHT_COUNT], &mut logits);
    logits
}

/// Initialize weights matching `Brain.initial_weights()` in brain.py.
/// Weights uniform(-0.5, 0.5); biases fixed at 0.001.
/// Draw order: for each linear layer — weight floats first, then bias floats.
pub fn initial_weights(rng: &mut impl Rng) -> Vec<f32> {
    // (in_size, out_size) for each linear layer
    const LAYERS: &[(usize, usize)] = &[(5, 12), (12, 12), (12, 12), (12, 8)];
    let mut buf = Vec::with_capacity(WEIGHT_COUNT);
    for &(in_size, out_size) in LAYERS {
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
    fn weight_count_is_488() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let w = initial_weights(&mut rng);
        assert_eq!(w.len(), WEIGHT_COUNT);
    }

    #[test]
    fn initial_weights_ranges() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let w = initial_weights(&mut rng);
        // Bias positions: 60..72, 216..228, 372..384, 480..488
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
        let input = [0.5f32, 0.3, 0.7, 0.4, 0.6];
        let out = forward(weights, input);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn forward_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let w: Vec<f32> = initial_weights(&mut rng);
        let weights: &[f32; WEIGHT_COUNT] = w.as_slice().try_into().unwrap();
        let input = [1.0f32, 0.0, 0.5, 0.2, 0.8];
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
