use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::genome::Genome;

const KMEANS_ITERS: usize = 15;

/// Dual k-means cluster assignments, one entry per alive agent.
/// Rebuilt every 50 steps by World::step().
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    pub genome_cluster_ids: Vec<u8>,
    pub brain_cluster_ids: Vec<u8>,
}

impl ClusterState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run both clustering passes over the current alive agent set.
    /// k_genome: 4–8, k_brain: 16–32. Uses a fresh RNG seeded from step count
    /// so results are deterministic given the same world state.
    pub fn run(genomes: &[Genome], k_genome: usize, k_brain: usize, step: u32) -> Self {
        let n = genomes.len();
        if n == 0 {
            return Self::new();
        }

        let mut rng = ChaCha8Rng::seed_from_u64(step as u64 ^ 0xdeadbeef_cafebabe);

        let genome_cluster_ids = kmeans_genome(genomes, k_genome.min(n), &mut rng);
        let brain_cluster_ids = kmeans_brain(genomes, k_brain.min(n), &mut rng);

        Self { genome_cluster_ids, brain_cluster_ids }
    }
}

// ── Genome clustering (12 traits, euclidean, k=6) ────────────────────────────

fn trait_vec(g: &Genome) -> [f64; 12] {
    let t = &g.traits;
    [
        t.vision, t.speed, t.metabolism, t.daily_nutrition_minimum,
        t.energy_capacity, t.mutation_rate, t.clone_energy_threshold,
        t.reproduction_cost, t.intelligence, t.attack, t.defense, t.aggression,
    ]
}

fn euclidean_sq(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans_genome(genomes: &[Genome], k: usize, rng: &mut ChaCha8Rng) -> Vec<u8> {
    let n = genomes.len();
    let points: Vec<[f64; 12]> = genomes.iter().map(trait_vec).collect();

    // k-means++ initialization
    let mut centroids: Vec<[f64; 12]> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(points[first]);

    for _ in 1..k {
        let dists: Vec<f64> = points.iter().map(|p| {
            centroids.iter().map(|c| euclidean_sq(p, c)).fold(f64::MAX, f64::min)
        }).collect();
        let total: f64 = dists.iter().sum();
        let mut target = rng.gen::<f64>() * total;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            target -= d;
            if target <= 0.0 { chosen = i; break; }
        }
        centroids.push(points[chosen]);
    }

    let mut labels = vec![0u8; n];
    for _ in 0..KMEANS_ITERS {
        // Assign
        for (i, p) in points.iter().enumerate() {
            labels[i] = centroids.iter().enumerate()
                .map(|(ci, c)| (ci, euclidean_sq(p, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci as u8)
                .unwrap_or(0);
        }
        // Update centroids
        let mut sums = vec![[0f64; 12]; k];
        let mut counts = vec![0usize; k];
        for (i, &label) in labels.iter().enumerate() {
            let c = label as usize;
            for d in 0..12 { sums[c][d] += points[i][d]; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..12 { centroids[c][d] = sums[c][d] / counts[c] as f64; }
            }
        }
    }
    labels
}

// ── Brain clustering (488 weights, cosine via normalized euclidean, k=24) ────

fn brain_vec(g: &Genome) -> Vec<f32> {
    g.brain_weights.clone()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 { return vec![0.0; v.len()]; }
    v.iter().map(|x| x / norm).collect()
}

fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans_brain(genomes: &[Genome], k: usize, rng: &mut ChaCha8Rng) -> Vec<u8> {
    let n = genomes.len();
    let dim = genomes[0].brain_weights.len();
    let points: Vec<Vec<f32>> = genomes.iter().map(|g| normalize(&brain_vec(g))).collect();

    // k-means++ initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(points[first].clone());

    for _ in 1..k {
        let dists: Vec<f32> = points.iter().map(|p| {
            centroids.iter().map(|c| euclidean_sq_f32(p, c)).fold(f32::MAX, f32::min)
        }).collect();
        let total: f32 = dists.iter().sum();
        let mut target = rng.gen::<f32>() * total;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            target -= d;
            if target <= 0.0 { chosen = i; break; }
        }
        centroids.push(points[chosen].clone());
    }

    let mut labels = vec![0u8; n];
    for _ in 0..KMEANS_ITERS {
        // Assign
        for (i, p) in points.iter().enumerate() {
            labels[i] = centroids.iter().enumerate()
                .map(|(ci, c)| (ci, euclidean_sq_f32(p, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci as u8)
                .unwrap_or(0);
        }
        // Update centroids (mean of normalized vectors, then re-normalize)
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &label) in labels.iter().enumerate() {
            let c = label as usize;
            for d in 0..dim { sums[c][d] += points[i][d]; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim { sums[c][d] /= counts[c] as f32; }
                centroids[c] = normalize(&sums[c]);
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    fn make_genomes(n: usize) -> Vec<Genome> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        (0..n).map(|_| Genome::generate(&mut rng)).collect()
    }

    #[test]
    fn cluster_output_length_matches_input() {
        let genomes = make_genomes(30);
        let state = ClusterState::run(&genomes, 6, 8, 100);
        assert_eq!(state.genome_cluster_ids.len(), 30);
        assert_eq!(state.brain_cluster_ids.len(), 30);
    }

    #[test]
    fn cluster_ids_in_range() {
        let genomes = make_genomes(20);
        let state = ClusterState::run(&genomes, 4, 6, 50);
        for &id in &state.genome_cluster_ids { assert!(id < 4); }
        for &id in &state.brain_cluster_ids  { assert!(id < 6); }
    }

    #[test]
    fn cluster_handles_small_population() {
        let genomes = make_genomes(3);
        let state = ClusterState::run(&genomes, 6, 16, 50);
        assert_eq!(state.genome_cluster_ids.len(), 3);
        // k clamped to n=3
        for &id in &state.genome_cluster_ids { assert!(id < 3); }
        for &id in &state.brain_cluster_ids  { assert!(id < 3); }
    }

    #[test]
    fn cluster_empty_population() {
        let state = ClusterState::run(&[], 6, 24, 50);
        assert!(state.genome_cluster_ids.is_empty());
        assert!(state.brain_cluster_ids.is_empty());
    }

    #[test]
    fn cluster_deterministic() {
        let genomes = make_genomes(15);
        let a = ClusterState::run(&genomes, 4, 8, 200);
        let b = ClusterState::run(&genomes, 4, 8, 200);
        assert_eq!(a.genome_cluster_ids, b.genome_cluster_ids);
        assert_eq!(a.brain_cluster_ids, b.brain_cluster_ids);
    }
}
