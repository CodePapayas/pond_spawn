use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::genome::Genome;

const KMEANS_ITERS: usize = 15;

/// Dual k-means cluster assignments, one entry per alive agent.
/// Rebuilt every 50 steps by World::step().
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    pub genome_cluster_ids: Vec<u8>,
    /// Final centroid per genome-cluster label (None = label unused this run).
    /// Kept so the next run can match its centroids against these and keep
    /// label→lineage assignment stable — labels drive renderer colors, and
    /// without matching a re-run may permute them (color flash).
    pub genome_centroids: Vec<Option<[f64; TRAIT_DIMS]>>,
}

impl ClusterState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run both clustering passes over the current alive agent set.
    /// k_genome: 4–8. Uses a fresh RNG seeded from step count
    /// so results are deterministic given the same world state.
    /// `prev` is the previous run's state; genome labels are remapped so that
    /// clusters persisting across runs keep their old label.
    pub fn run(genomes: &[Genome], k_genome: usize, step: u32, prev: Option<&ClusterState>) -> Self {
        let n = genomes.len();
        if n == 0 {
            return Self::new();
        }

        let mut rng = ChaCha8Rng::seed_from_u64(step as u64 ^ 0xdeadbeef_cafebabe);

        let k_g = k_genome.min(n);
        let (mut genome_cluster_ids, centroids) = kmeans_genome(genomes, k_g, &mut rng);

        // Remap raw kmeans labels onto previous labels by centroid proximity.
        let map = match prev {
            Some(p) if !p.genome_centroids.is_empty() =>
                match_labels(&centroids, &p.genome_centroids, k_genome),
            _ => (0..centroids.len() as u8).collect(),
        };
        for l in genome_cluster_ids.iter_mut() {
            *l = map[*l as usize];
        }
        let mut genome_centroids = vec![None; k_genome.max(centroids.len())];
        for (ni, c) in centroids.iter().enumerate() {
            genome_centroids[map[ni] as usize] = Some(*c);
        }

        Self { genome_cluster_ids, genome_centroids }
    }
}

/// Greedy nearest-centroid matching: returns map[new_label] → stable label.
/// Persisting clusters inherit the previous label; genuinely new clusters get
/// the free labels. k=6, so greedy over all pairs is plenty.
fn match_labels(
    new: &[[f64; TRAIT_DIMS]],
    prev: &[Option<[f64; TRAIT_DIMS]>],
    k: usize,
) -> Vec<u8> {
    // Labels must stay below the current `k`: everything downstream — the legend
    // rows, the renderer palette, the headless histogram — sizes itself by k.
    // Widening this to `prev.len()` let a cluster keep a label from a wider
    // earlier run after `k` was lowered, which is a label no consumer expects
    // and which indexed out of bounds. A family whose label is now out of range
    // gets remapped, so it changes colour once; that is the honest reading, as
    // lowering k genuinely merged the families away.
    //
    // Nothing changes while k is held constant: `prev.len()` is then k already.
    let slots = k.max(new.len());
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (ni, nc) in new.iter().enumerate() {
        // Previous labels at or above `slots` are gone with the k they came
        // from — a cluster near one of them still gets a fresh in-range label
        // below rather than inheriting an unusable one.
        for (oi, oc) in prev.iter().enumerate().take(slots) {
            if let Some(oc) = oc {
                pairs.push((euclidean_sq(nc, oc), ni, oi));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut map = vec![u8::MAX; new.len()];
    let mut new_taken = vec![false; new.len()];
    let mut old_taken = vec![false; slots];
    for (_, ni, oi) in pairs {
        if new_taken[ni] || old_taken[oi] { continue; }
        map[ni] = oi as u8;
        new_taken[ni] = true;
        old_taken[oi] = true;
    }
    // Unmatched new clusters take the lowest free labels.
    let mut free = (0..slots).filter(|&l| !old_taken[l]);
    for (ni, m) in map.iter_mut().enumerate() {
        if *m == u8::MAX {
            *m = free.next().unwrap_or(ni) as u8;
        }
    }
    map
}

// ── Genome clustering (7 normalized traits, euclidean, k=6) ──────────────────

/// Clustering runs on the species signature — the seven mutable traits, each
/// rescaled to [0, 1] by its bounds (`species::signature`).
///
/// Both changes matter, and both were wrong here before:
///
/// - **Normalized.** Raw trait values have bounds differing by an order of
///   magnitude, so euclidean distance over them let `reproduction_cost`
///   (0.75–1.50) count for several times what `defense` (0.5–1.07) did, and
///   `mutation_rate` (0.01–0.25) contributed almost nothing to a partition.
/// - **Mutable traits only.** `energy_capacity` and `mutation_rate` are locked
///   (D3) — never mutated, inherited exactly — so clustering on them yields
///   founder-descent groups rather than shapes selection built.
///
/// Sharing the signature with `species.rs` is the point: the cluster colors on
/// screen and the species a member is assigned to are now the same space. Before
/// this they were two different readings of "family" that did not agree.
const TRAIT_DIMS: usize = crate::species::SIG_LEN;

fn trait_vec(g: &Genome) -> [f64; TRAIT_DIMS] {
    crate::species::signature(&g.traits)
}

fn euclidean_sq(a: &[f64; TRAIT_DIMS], b: &[f64; TRAIT_DIMS]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Move every empty cluster's centroid onto the point furthest from its own
/// centroid — the same policy `brain_cluster::update_centroids` uses, and the
/// reason the two implementations no longer disagree.
///
/// A cluster that loses every member has no sum to average, so without this it
/// keeps whatever centroid it held when the last member left: nothing can ever
/// move it again, it is written into `genome_centroids`, and it goes on
/// claiming a stable label in `match_labels` against a shape no live agent has.
/// Reseeding onto the worst-fit point instead gives it the split most worth
/// making. Several empty clusters in one pass may land on the same point; the
/// next assign pass separates them, exactly as in `brain_cluster`.
fn reseed_empty(
    points: &[[f64; TRAIT_DIMS]],
    labels: &[u8],
    counts: &[usize],
    centroids: &mut [[f64; TRAIT_DIMS]],
) {
    if points.is_empty() { return; }
    for c in 0..centroids.len() {
        if counts[c] > 0 { continue; }
        let mut worst = (-1.0f64, 0usize);
        for (i, p) in points.iter().enumerate() {
            let d = euclidean_sq(p, &centroids[labels[i] as usize]);
            if d > worst.0 { worst = (d, i); }
        }
        centroids[c] = points[worst.1];
    }
}

fn kmeans_genome(
    genomes: &[Genome],
    k: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<u8>, Vec<[f64; TRAIT_DIMS]>) {
    let n = genomes.len();
    let points: Vec<[f64; TRAIT_DIMS]> = genomes.iter().map(trait_vec).collect();

    // k-means++ initialization
    let mut centroids: Vec<[f64; TRAIT_DIMS]> = Vec::with_capacity(k);
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
        let mut sums = vec![[0f64; TRAIT_DIMS]; k];
        let mut counts = vec![0usize; k];
        for (i, &label) in labels.iter().enumerate() {
            let c = label as usize;
            for d in 0..TRAIT_DIMS { sums[c][d] += points[i][d]; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..TRAIT_DIMS { centroids[c][d] = sums[c][d] / counts[c] as f64; }
            }
        }
        reseed_empty(&points, &labels, &counts, &mut centroids);
    }
    (labels, centroids)
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
        let state = ClusterState::run(&genomes, 6, 100, None);
        assert_eq!(state.genome_cluster_ids.len(), 30);
    }

    #[test]
    fn cluster_ids_in_range() {
        let genomes = make_genomes(20);
        let state = ClusterState::run(&genomes, 4, 50, None);
        for &id in &state.genome_cluster_ids { assert!(id < 4); }
    }

    #[test]
    fn cluster_handles_small_population() {
        let genomes = make_genomes(3);
        let state = ClusterState::run(&genomes, 6, 50, None);
        assert_eq!(state.genome_cluster_ids.len(), 3);
        // k clamped to n=3
        for &id in &state.genome_cluster_ids { assert!(id < 3); }
    }

    #[test]
    fn cluster_empty_population() {
        let state = ClusterState::run(&[], 6, 50, None);
        assert!(state.genome_cluster_ids.is_empty());
    }

    #[test]
    fn labels_stable_across_reruns_with_prev() {
        // Three well-separated trait blobs: every re-run finds the same
        // partition, but a different step seed may permute the labels.
        // With prev passed, matching must map them back to the same labels.
        let mut genomes = make_genomes(60);
        for (i, g) in genomes.iter_mut().enumerate() {
            let level = match i % 3 { 0 => 0.5, 1 => 0.75, _ => 1.0 };
            let t = &mut g.traits;
            t.vision = level; t.speed = level.min(1.0); t.metabolism = level;
            t.attack = level; t.defense = level; t.aggression = level;
            t.reproduction_cost = level + 0.25;
        }
        let a = ClusterState::run(&genomes, 6, 100, None);
        let b = ClusterState::run(&genomes, 6, 150, Some(&a));
        assert_eq!(a.genome_cluster_ids, b.genome_cluster_ids);
    }

    #[test]
    fn clustering_ignores_locked_traits_and_normalizes_the_rest() {
        let mut a = make_genomes(1).pop().unwrap();
        let mut b = a.clone();
        // Locked traits (D3) are founder tags, not evolved shape — they must not
        // move a genome in cluster space.
        b.traits.energy_capacity = 1.05;
        b.traits.mutation_rate = 0.25;
        a.traits.energy_capacity = 0.95;
        a.traits.mutation_rate = 0.01;
        assert_eq!(trait_vec(&a), trait_vec(&b));

        // Every dimension is on the same [0, 1] scale, so no trait outweighs
        // another purely because its bounds are wider.
        for v in trait_vec(&a) {
            assert!((0.0..=1.0).contains(&v), "dimension {} outside [0,1]", v);
        }
    }

    #[test]
    fn empty_clusters_are_reseeded_not_left_stale() {
        // Cluster 1 has lost every member; its centroid is a mean of agents that
        // no longer exist, and nothing in the update step can move it.
        let points = vec![
            [0.0; TRAIT_DIMS],
            [0.1; TRAIT_DIMS],
            [0.9; TRAIT_DIMS],
        ];
        let labels = vec![0u8, 0, 0];
        let counts = vec![3usize, 0];
        let stale = [0.5; TRAIT_DIMS];
        let mut centroids = vec![[0.05; TRAIT_DIMS], stale];

        reseed_empty(&points, &labels, &counts, &mut centroids);

        // The worst-fit live point is [0.9…], furthest from its own centroid.
        assert_eq!(centroids[1], points[2]);
        // The occupied cluster is untouched.
        assert_eq!(centroids[0], [0.05; TRAIT_DIMS]);
    }

    #[test]
    fn labels_stay_below_k_when_k_shrinks() {
        // `k` is a dial, and every consumer — legend rows, the renderer palette,
        // the headless histogram — sizes itself by it. A label inherited from a
        // wider previous run would index past all three.
        let genomes = make_genomes(60);
        let wide = ClusterState::run(&genomes, 8, 100, None);
        assert!(wide.genome_cluster_ids.iter().any(|&id| id >= 3));

        let narrow = ClusterState::run(&genomes, 3, 150, Some(&wide));
        for &id in &narrow.genome_cluster_ids { assert!(id < 3, "label {} >= k", id); }
        assert!(narrow.genome_centroids.len() <= 3);
    }

    #[test]
    fn cluster_deterministic() {
        let genomes = make_genomes(15);
        let a = ClusterState::run(&genomes, 4, 200, None);
        let b = ClusterState::run(&genomes, 4, 200, None);
        assert_eq!(a.genome_cluster_ids, b.genome_cluster_ids);
    }
}

