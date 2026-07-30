//! Behavioural clustering — k-means over the 488 brain weights.
//!
//! Split out of `cluster.rs` because it costs ~99.5% of what the cluster tick
//! costs and needs a completely different execution model. The genome pass is
//! nine dimensions and finishes in microseconds, so it runs synchronously on the
//! tick. This pass is `n × k(24) × 488 dims × iters` and, run the same way, was
//! a 14–164 ms spike every 50 steps — a visible stutter at any population, since
//! a 60 fps frame budget is 16.7 ms.
//!
//! Three things make it affordable:
//!
//! 1. **Warm start.** Centroids are retained between runs and reused as the next
//!    run's initialization. The population changes by a few births and deaths
//!    between runs, not wholesale, so the previous centroids are already close
//!    to converged. This removes k-means++ from the steady-state path entirely —
//!    it was 42% of the cost, 24 sequential passes over every 488-dim vector
//!    purely to choose starting points — and lets the iteration count drop from
//!    15 to a handful. k-means++ remains the cold-start path and the reseeding
//!    path for clusters that go empty.
//!
//! 2. **Amortization.** With no init phase to complete first, k-means is
//!    naturally incremental: one iteration per step across several steps rather
//!    than all of them in one tick. The spike becomes a ripple.
//!
//! 3. **Flat buffers.** Points are one contiguous `Vec<f32>` reused across runs
//!    rather than a `Vec<Vec<f32>>` allocating per agent per run, and the
//!    accumulator is reused across iterations rather than reallocated.
//!
//! Iteration runs against a **snapshot** taken when the pass begins, so agents
//! born or killed mid-window cannot corrupt the partition. The final assignment
//! is made against the live population, which is both what keeps `labels`
//! aligned to the current agent arrays and the pass's last iteration — no work
//! is duplicated to achieve it.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::genome::Genome;

/// Iterations for a warm-started pass. Centroids begin near-converged, so this
/// is refinement rather than a search.
pub const WARM_ITERS: u32 = 3;
/// Iterations for a cold pass, where centroids come from k-means++ and there is
/// a real partition to find.
pub const COLD_ITERS: u32 = 12;

/// Behavioural k-means over brain weights: retained centroids, incremental
/// execution, and an on/off switch.
#[derive(Debug, Clone, Default)]
pub struct BrainClusters {
    /// Cluster label per agent, aligned to the live agent arrays as of the last
    /// completed pass. Empty when disabled or before the first pass finishes.
    pub labels: Vec<u8>,
    /// Retained centroids, flat `k × dim`. The warm start.
    centroids: Vec<f32>,
    k: usize,
    dim: usize,
    /// `false` skips the pass entirely — nothing looks at behavioural clusters
    /// unless the view is open, and this is the most expensive thing in the sim.
    enabled: bool,
    pass: Option<Pass>,
    /// Scratch, reused across runs to keep the pass allocation-free.
    scratch: Scratch,
}

/// A pass in flight, iterating against the snapshot it began with.
#[derive(Debug, Clone)]
struct Pass {
    points: Vec<f32>,
    n: usize,
    labels: Vec<u8>,
    iters_left: u32,
}

#[derive(Debug, Clone, Default)]
struct Scratch {
    sums: Vec<f32>,
    counts: Vec<usize>,
    dists: Vec<f32>,
}

impl BrainClusters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Turn behavioural clustering on or off. Switching off drops the labels and
    /// abandons any pass in flight; consumers already index defensively.
    pub fn set_enabled(&mut self, on: bool) {
        if self.enabled == on {
            return;
        }
        self.enabled = on;
        if !on {
            self.labels.clear();
            self.pass = None;
        }
    }

    /// True while a pass is mid-flight. Diagnostic.
    pub fn in_progress(&self) -> bool {
        self.pass.is_some()
    }

    /// Begin a pass. Called on the cluster tick; no-op when disabled, when the
    /// population is empty, or when the previous pass has not finished.
    pub fn begin(&mut self, genomes: &[Genome], k: usize, step: u32) {
        if !self.enabled || genomes.is_empty() || self.pass.is_some() {
            return;
        }
        let n = genomes.len();
        let dim = genomes[0].brain_weights.len();
        let k = k.min(n);

        let mut points = vec![0f32; n * dim];
        for (i, g) in genomes.iter().enumerate() {
            normalize_into(&g.brain_weights, &mut points[i * dim..(i + 1) * dim]);
        }

        // Warm start only if the retained centroids still describe the same
        // problem — a changed k or dim means they are meaningless.
        let warm = self.k == k && self.dim == dim && self.centroids.len() == k * dim;
        if !warm {
            let mut rng = ChaCha8Rng::seed_from_u64(step as u64 ^ 0xb7a1_c105_7e51_u64);
            self.centroids = kmeans_pp_init(&points, n, dim, k, &mut rng, &mut self.scratch.dists);
            self.k = k;
            self.dim = dim;
        }

        self.pass = Some(Pass {
            points,
            n,
            labels: vec![0u8; n],
            iters_left: if warm { WARM_ITERS } else { COLD_ITERS },
        });
    }

    /// Advance a pass in flight by one iteration. Called every step; no-op when
    /// idle. The last iteration assigns against the live population rather than
    /// the snapshot, which is what keeps `labels` aligned to the current agent
    /// arrays.
    pub fn advance(&mut self, genomes: &[Genome]) {
        let Some(pass) = self.pass.as_mut() else { return };
        if !self.enabled {
            self.pass = None;
            return;
        }

        pass.iters_left -= 1;
        let finishing = pass.iters_left == 0;

        assign(&pass.points, pass.n, self.dim, &self.centroids, self.k, &mut pass.labels);
        update_centroids(
            &pass.points, pass.n, self.dim, &pass.labels, self.k,
            &mut self.centroids, &mut self.scratch,
        );

        if !finishing {
            return;
        }

        // Final assignment against the live population. Agents born during the
        // window get a label, agents that died take theirs with them, and the
        // result indexes the same way the agent arrays do.
        let n_live = genomes.len();
        self.labels.clear();
        self.labels.resize(n_live, 0);
        if n_live > 0 && genomes[0].brain_weights.len() == self.dim {
            let mut buf = vec![0f32; self.dim];
            for (i, g) in genomes.iter().enumerate() {
                normalize_into(&g.brain_weights, &mut buf);
                self.labels[i] = nearest(&buf, &self.centroids, self.k, self.dim);
            }
        }
        self.pass = None;
    }
}

/// L2-normalize into a destination slice. Cosine distance over unit vectors is
/// monotone in euclidean distance, so the rest of the algorithm is plain
/// k-means and never computes a cosine.
fn normalize_into(v: &[f32], out: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        out.fill(0.0);
        return;
    }
    for (o, x) in out.iter_mut().zip(v) {
        *o = x / norm;
    }
}

fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn nearest(p: &[f32], centroids: &[f32], k: usize, dim: usize) -> u8 {
    let mut best = (f32::MAX, 0usize);
    for c in 0..k {
        let d = dist_sq(p, &centroids[c * dim..(c + 1) * dim]);
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1 as u8
}

fn assign(points: &[f32], n: usize, dim: usize, centroids: &[f32], k: usize, labels: &mut [u8]) {
    for i in 0..n {
        labels[i] = nearest(&points[i * dim..(i + 1) * dim], centroids, k, dim);
    }
}

/// Mean of each cluster's members, re-normalized. Empty clusters are reseeded
/// onto the point furthest from its own centroid — without this a warm-started
/// centroid that loses all its members would sit dead forever, since nothing
/// else would ever move it.
fn update_centroids(
    points: &[f32], n: usize, dim: usize, labels: &[u8], k: usize,
    centroids: &mut [f32], scratch: &mut Scratch,
) {
    scratch.sums.clear();
    scratch.sums.resize(k * dim, 0.0);
    scratch.counts.clear();
    scratch.counts.resize(k, 0);

    for i in 0..n {
        let c = labels[i] as usize;
        let (dst, src) = (c * dim, i * dim);
        for d in 0..dim {
            scratch.sums[dst + d] += points[src + d];
        }
        scratch.counts[c] += 1;
    }

    for c in 0..k {
        if scratch.counts[c] == 0 {
            continue;
        }
        let inv = 1.0 / scratch.counts[c] as f32;
        for d in 0..dim {
            scratch.sums[c * dim + d] *= inv;
        }
        let (lo, hi) = (c * dim, (c + 1) * dim);
        let norm: f32 = scratch.sums[lo..hi].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            centroids[lo..hi].fill(0.0);
        } else {
            for d in 0..dim {
                centroids[lo + d] = scratch.sums[lo + d] / norm;
            }
        }
    }

    // Reseed the empty.
    for c in 0..k {
        if scratch.counts[c] > 0 || n == 0 {
            continue;
        }
        let mut worst = (-1.0f32, 0usize);
        for i in 0..n {
            let p = &points[i * dim..(i + 1) * dim];
            let own = labels[i] as usize;
            let d = dist_sq(p, &centroids[own * dim..(own + 1) * dim]);
            if d > worst.0 {
                worst = (d, i);
            }
        }
        let src = worst.1 * dim;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[src..src + dim]);
    }
}

/// k-means++ over the flat point buffer. Cold-start path only.
fn kmeans_pp_init(
    points: &[f32], n: usize, dim: usize, k: usize,
    rng: &mut ChaCha8Rng, dists: &mut Vec<f32>,
) -> Vec<f32> {
    let mut centroids = vec![0f32; k * dim];
    let first = rng.gen_range(0..n);
    centroids[0..dim].copy_from_slice(&points[first * dim..(first + 1) * dim]);

    dists.clear();
    dists.resize(n, f32::MAX);

    for c in 1..k {
        // Each round only needs the distance to the centroid just added, folded
        // into the running minimum — the original recomputed against every
        // centroid chosen so far, making init quadratic in k for no reason.
        let prev = &centroids[(c - 1) * dim..c * dim];
        let mut total = 0f32;
        for i in 0..n {
            let d = dist_sq(&points[i * dim..(i + 1) * dim], prev);
            if d < dists[i] {
                dists[i] = d;
            }
            total += dists[i];
        }
        let mut target = rng.gen::<f32>() * total;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            target -= d;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids[c * dim..(c + 1) * dim]
            .copy_from_slice(&points[chosen * dim..(chosen + 1) * dim]);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn genomes(n: usize, seed: u64) -> Vec<Genome> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| Genome::generate(&mut rng)).collect()
    }

    /// Drive a full pass to completion.
    fn full_pass(bc: &mut BrainClusters, g: &[Genome], k: usize, step: u32) {
        bc.begin(g, k, step);
        while bc.in_progress() {
            bc.advance(g);
        }
    }

    #[test]
    fn disabled_costs_nothing_and_produces_nothing() {
        let mut bc = BrainClusters::new();
        let g = genomes(40, 1);
        full_pass(&mut bc, &g, 8, 50);
        assert!(bc.labels.is_empty());
        assert!(!bc.in_progress());
    }

    #[test]
    fn a_pass_labels_every_agent_in_range() {
        let mut bc = BrainClusters::new();
        bc.set_enabled(true);
        let g = genomes(40, 1);
        full_pass(&mut bc, &g, 8, 50);
        assert_eq!(bc.labels.len(), 40);
        assert!(bc.labels.iter().all(|&l| (l as usize) < 8));
    }

    #[test]
    fn a_pass_is_spread_over_several_steps() {
        // The whole point: no single step carries the entire cost.
        let mut bc = BrainClusters::new();
        bc.set_enabled(true);
        let g = genomes(40, 1);
        bc.begin(&g, 8, 50);
        let mut steps: u32 = 0;
        while bc.in_progress() {
            bc.advance(&g);
            steps += 1;
        }
        assert_eq!(steps, COLD_ITERS, "cold pass spreads over COLD_ITERS steps");

        bc.begin(&g, 8, 100);
        let mut warm_steps: u32 = 0;
        while bc.in_progress() {
            bc.advance(&g);
            warm_steps += 1;
        }
        assert_eq!(warm_steps, WARM_ITERS, "warm pass is shorter");
        assert!(warm_steps < steps);
    }

    #[test]
    fn warm_start_agrees_with_cold_start() {
        // Warm starting must not park the partition in a worse local optimum.
        // Compare against a cold run on the same population: the *grouping*
        // must agree, though label numbering need not.
        let g = genomes(120, 7);

        let mut cold = BrainClusters::new();
        cold.set_enabled(true);
        full_pass(&mut cold, &g, 8, 50);

        let mut warm = BrainClusters::new();
        warm.set_enabled(true);
        full_pass(&mut warm, &g, 8, 50);
        for step in 1..6 {
            full_pass(&mut warm, &g, 8, 50 + step * 50);
        }

        let agree = pair_agreement(&cold.labels, &warm.labels);
        assert!(agree > 0.85, "warm/cold partition agreement only {:.3}", agree);
    }

    /// Fraction of agent pairs the two labelings agree about (same cluster or
    /// different cluster). Label-permutation invariant, unlike a direct compare.
    fn pair_agreement(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len();
        let (mut same, mut total) = (0u64, 0u64);
        for i in 0..n {
            for j in (i + 1)..n {
                if (a[i] == a[j]) == (b[i] == b[j]) {
                    same += 1;
                }
                total += 1;
            }
        }
        same as f64 / total as f64
    }

    #[test]
    fn empty_clusters_are_reseeded_not_left_dead() {
        // Warm start keeps centroids across runs, so a cluster that loses every
        // member would sit dead forever if nothing moved it.
        let mut bc = BrainClusters::new();
        bc.set_enabled(true);
        let many = genomes(80, 3);
        full_pass(&mut bc, &many, 16, 50);

        // Collapse to a handful of agents, far fewer than k.
        let few = genomes(6, 9);
        full_pass(&mut bc, &few, 16, 100);
        assert_eq!(bc.labels.len(), 6);

        // Grow again: every label must still be reachable, i.e. no centroid is
        // stranded somewhere nothing will ever match.
        let many2 = genomes(80, 11);
        full_pass(&mut bc, &many2, 16, 150);
        assert_eq!(bc.labels.len(), 80);
    }

    #[test]
    fn labels_track_a_changing_population() {
        let mut bc = BrainClusters::new();
        bc.set_enabled(true);
        let a = genomes(50, 1);
        full_pass(&mut bc, &a, 8, 50);

        // A pass that begins on one population and ends on a smaller one still
        // returns labels aligned to the live agents.
        bc.begin(&a, 8, 100);
        let b = genomes(30, 2);
        while bc.in_progress() {
            bc.advance(&b);
        }
        assert_eq!(bc.labels.len(), 30);
    }

    #[test]
    fn deterministic_for_a_given_seed() {
        let g = genomes(60, 5);
        let mut a = BrainClusters::new();
        let mut b = BrainClusters::new();
        a.set_enabled(true);
        b.set_enabled(true);
        for step in 0..4 {
            full_pass(&mut a, &g, 8, 50 + step * 50);
            full_pass(&mut b, &g, 8, 50 + step * 50);
        }
        assert_eq!(a.labels, b.labels);
    }

    #[test]
    fn toggling_off_clears_state() {
        let mut bc = BrainClusters::new();
        bc.set_enabled(true);
        let g = genomes(40, 1);
        full_pass(&mut bc, &g, 8, 50);
        assert!(!bc.labels.is_empty());
        bc.set_enabled(false);
        assert!(bc.labels.is_empty());
        assert!(!bc.in_progress());
    }
}
