use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::brain::{initial_weights, WEIGHT_COUNT};

/// Genome traits. Bounds match `genomes/genome.json`.
/// `daily_nutrition_minimum` and `clone_energy_threshold` were removed: generated
/// and mutated but never read anywhere, so they only caused mutation drift on a
/// non-selected gene and wasted RNG draws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Traits {
    pub vision: f64,
    pub speed: f64,
    pub metabolism: f64,
    pub energy_capacity: f64,    // locked (D3)
    pub mutation_rate: f64,      // locked (D3)
    pub reproduction_cost: f64,
    pub attack: f64,
    pub defense: f64,
    pub aggression: f64,
    /// How often the agent thinks, how quickly it notices a threat, and what
    /// that costs. See `DECISION_INTERVAL_MAX`, `THREAT_LAG_MAX` and
    /// `INTELLIGENCE_UPKEEP` in `world.rs`.
    ///
    /// Appended rather than slotted in beside `reproduction_cost` where it was
    /// first drafted: every index after it is mirrored by hand in `wasm.rs`,
    /// `species.rs`, the inspector and the panels, and shifting `attack`,
    /// `defense` and `aggression` by one would have been a silent misread in
    /// each of them.
    pub intelligence: f64,
}

/// Number of genome traits. One source for every `[f64; N]` that mirrors the
/// trait list — the population means, the composite panel, the inspector row.
pub const TRAIT_COUNT: usize = Traits::BOUNDS.len();

impl Traits {
    /// `[lo, hi]` per trait in field order. Canonical — `generate()` draws from
    /// these, `mutate()` clamps to them, `species.rs` normalizes by them, and
    /// `wasm::trait_bounds()` exports them. One table, so a bounds change can't
    /// half-land.
    pub const BOUNDS: [(f64, f64); 10] = [
        (0.5, 1.05),   // vision
        (0.5, 1.0),    // speed
        (0.5, 1.05),   // metabolism
        (0.95, 1.05),  // energy_capacity (locked, D3)
        (0.01, 0.25),  // mutation_rate  (locked, D3)
        (0.75, 1.50),  // reproduction_cost
        (0.5, 1.25),   // attack
        (0.5, 1.07),   // defense
        (0.0, 1.05),   // aggression
        (0.5, 1.05),   // intelligence
    ];

    /// Generate random founding trait values within JSON-defined bounds.
    /// Draw order matches Python's dict iteration order (insertion order, Python 3.7+).
    pub fn generate(rng: &mut impl Rng) -> Self {
        Self {
            vision:                  rng.gen_range(0.5_f64..=1.05),
            speed:                   rng.gen_range(0.5_f64..=1.0),
            metabolism:              rng.gen_range(0.5_f64..=1.05),
            energy_capacity:         rng.gen_range(0.95_f64..=1.05),
            mutation_rate:           rng.gen_range(0.01_f64..=0.25),
            reproduction_cost:       rng.gen_range(0.75_f64..=1.50),
            attack:                  rng.gen_range(0.5_f64..=1.25),
            defense:                 rng.gen_range(0.5_f64..=1.07),
            aggression:              rng.gen_range(0.0_f64..=1.05),
            intelligence:            rng.gen_range(0.5_f64..=1.05),
        }
    }

    /// Mutate mutable traits. Locked traits (energy_capacity, mutation_rate) skip
    /// the RNG draw entirely — matching Python D3 behavior for golden-seed parity.
    fn mutate(&self, eff_rate: f32, rng: &mut impl Rng) -> Self {
        let rate = eff_rate as f64;
        let magnitude = rate * 0.5;

        macro_rules! maybe_mutate {
            ($val:expr, $min:expr, $max:expr) => {{
                if rng.gen::<f64>() < rate {
                    let factor = rng.gen_range((1.0 - magnitude)..=(1.0 + magnitude));
                    ($val * factor).clamp($min, $max)
                } else {
                    $val
                }
            }};
        }

        Self {
            vision:                  maybe_mutate!(self.vision, 0.5, 1.05),
            speed:                   maybe_mutate!(self.speed, 0.5, 1.0),
            metabolism:              maybe_mutate!(self.metabolism, 0.5, 1.05),
            // Locked — no RNG draw (D3)
            energy_capacity:         self.energy_capacity,
            mutation_rate:           self.mutation_rate,
            reproduction_cost:       maybe_mutate!(self.reproduction_cost, 0.75, 1.50),
            attack:                  maybe_mutate!(self.attack, 0.5, 1.25),
            defense:                 maybe_mutate!(self.defense, 0.5, 1.07),
            aggression:              maybe_mutate!(self.aggression, 0.0, 1.05),
            intelligence:            maybe_mutate!(self.intelligence, 0.5, 1.05),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub id: String,
    pub traits: Traits,
    /// Flat weight buffer, length WEIGHT_COUNT (488). Layout matches brain.py.
    pub brain_weights: Vec<f32>,
    /// Heritable effective mutation rate (D4). Separate from locked `traits.mutation_rate`.
    /// Starts at `traits.mutation_rate`; suppressed at reproduction by AgentMemory success count.
    pub effective_mutation_rate: f32,
    /// Reproductive depth from a founding genome: 0 for a founder, parent + 1 for
    /// offspring. Not age — an agent that lives forever without reproducing stays
    /// at its birth generation, which is the point. Speciation needs "this trait
    /// signature was passed down and re-selected", and only generation says that.
    /// `serde(default)` so pre-existing golden-harness traces still deserialize.
    #[serde(default)]
    pub generation: u32,
}

impl Genome {
    /// Generate a new founding genome. Matches `Genome.generate()` in genome.py.
    pub fn generate(rng: &mut impl Rng) -> Self {
        let traits = Traits::generate(rng);
        let eff_rate = traits.mutation_rate as f32;
        let brain_weights = initial_weights(rng);
        Self {
            id: genome_id(rng),
            traits,
            brain_weights,
            effective_mutation_rate: eff_rate,
            generation: 0,
        }
    }

    /// Produce a mutated offspring genome.
    ///
    /// `suppression` = 1.0 / (1.0 + parent_success_count * k) from AgentMemory.
    /// It is **heritable**: it multiplies into the child's
    /// `effective_mutation_rate` and so compounds down the generations (D4).
    ///
    /// `clamp` is the species probation clamp, and is **not** heritable: it
    /// scales the rate used for this one set of draws and is discarded. The
    /// distinction is the whole mechanic — probation asks whether a lineage
    /// survives having its mutability taken away, and a lineage that passes has
    /// to get it back. Routing the clamp through the heritable path instead
    /// would sterilize the lineage permanently, which tests something else.
    pub fn mutate(&self, rng: &mut impl Rng, suppression: f32, clamp: f32) -> Self {
        let eff_rate = self.effective_mutation_rate * suppression;
        let rate_used = eff_rate * clamp;
        let rate_f64 = rate_used as f64;
        let magnitude = rate_f64 * 0.5;

        let new_traits = self.traits.mutate(rate_used, rng);

        let new_weights: Vec<f32> = self
            .brain_weights
            .iter()
            .map(|&w| {
                if rng.gen::<f64>() < rate_f64 {
                    // Additive perturbation (was multiplicative w * factor): lets
                    // weight signs flip and zero weights revive. Diverges from
                    // legacy Python mutation — pond_core is canonical (RULES.md).
                    w + rng.gen_range((-magnitude * 0.5)..=(magnitude * 0.5)) as f32
                } else {
                    w
                }
            })
            .collect();

        debug_assert_eq!(new_weights.len(), WEIGHT_COUNT);

        Self {
            id: genome_id(rng),
            traits: new_traits,
            brain_weights: new_weights,
            // Deliberately `eff_rate`, not `rate_used`: the clamp does not ride
            // into the child.
            effective_mutation_rate: eff_rate,
            generation: self.generation + 1,
        }
    }

    /// Convenience: mutate with no suppression and no clamp (founding
    /// generations, tests).
    pub fn mutate_unsuppressed(&self, rng: &mut impl Rng) -> Self {
        self.mutate(rng, 1.0, 1.0)
    }

    pub fn weights_array(&self) -> &[f32; WEIGHT_COUNT] {
        self.brain_weights.as_slice().try_into().expect("brain_weights len != 488")
    }
}

/// ID format matches Python: `"g_{:08x}"` over a 32-bit random value.
fn genome_id(rng: &mut impl Rng) -> String {
    format!("g_{:08x}", rng.gen::<u32>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    fn seeded() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn generation_counts_reproductive_depth() {
        let mut rng = seeded();
        let founder = Genome::generate(&mut rng);
        assert_eq!(founder.generation, 0);

        let mut g = founder;
        for expected in 1..=3 {
            g = g.mutate_unsuppressed(&mut rng);
            assert_eq!(g.generation, expected);
        }
    }

    #[test]
    fn generate_trait_bounds() {
        let mut rng = seeded();
        for _ in 0..50 {
            let g = Genome::generate(&mut rng);
            let t = &g.traits;
            assert!((0.5..=1.05).contains(&t.vision));
            assert!((0.5..=1.0).contains(&t.speed));
            assert!((0.5..=1.05).contains(&t.metabolism));
            assert!((0.95..=1.05).contains(&t.energy_capacity));
            assert!((0.01..=0.25).contains(&t.mutation_rate));
            assert!((0.75..=1.50).contains(&t.reproduction_cost));
            assert!((0.5..=1.25).contains(&t.attack));
            assert!((0.5..=1.07).contains(&t.defense));
            assert!((0.0..=1.05).contains(&t.aggression));
        }
    }

    #[test]
    fn generate_weight_count() {
        let mut rng = seeded();
        let g = Genome::generate(&mut rng);
        assert_eq!(g.brain_weights.len(), WEIGHT_COUNT);
    }

    #[test]
    fn effective_mutation_rate_initialized_from_trait() {
        let mut rng = seeded();
        let g = Genome::generate(&mut rng);
        assert!((g.effective_mutation_rate as f64 - g.traits.mutation_rate).abs() < 1e-6);
    }

    #[test]
    fn mutate_locked_traits_unchanged() {
        let mut rng = seeded();
        let parent = Genome::generate(&mut rng);
        let child = parent.mutate_unsuppressed(&mut rng);
        assert_eq!(child.traits.energy_capacity, parent.traits.energy_capacity);
        assert_eq!(child.traits.mutation_rate, parent.traits.mutation_rate);
    }

    #[test]
    fn mutate_child_traits_in_bounds() {
        let mut rng = seeded();
        let parent = Genome::generate(&mut rng);
        for _ in 0..20 {
            let child = parent.mutate_unsuppressed(&mut rng);
            let t = &child.traits;
            assert!((0.5..=1.05).contains(&t.vision));
            assert!((0.5..=1.0).contains(&t.speed));
            assert!((0.75..=1.50).contains(&t.reproduction_cost));
            assert!((0.0..=1.05).contains(&t.aggression));
        }
    }

    #[test]
    fn suppression_compresses_effective_rate() {
        let mut rng = seeded();
        let parent = Genome::generate(&mut rng);
        let suppression = 1.0 / (1.0 + 5.0 * 0.05_f32);
        let child = parent.mutate(&mut rng, suppression, 1.0);
        assert!(child.effective_mutation_rate < parent.effective_mutation_rate);
    }

    #[test]
    fn probation_clamp_is_not_heritable() {
        // The clamp throttles this set of draws only. If it leaked into the
        // child's effective rate it would compound down the generations and
        // sterilize the lineage permanently, which is a different mechanic.
        let mut rng = seeded();
        let parent = Genome::generate(&mut rng);

        let clamped = parent.mutate(&mut rng, 1.0, 0.15);
        let free = parent.mutate(&mut rng, 1.0, 1.0);
        assert_eq!(clamped.effective_mutation_rate, free.effective_mutation_rate);
        assert_eq!(clamped.effective_mutation_rate, parent.effective_mutation_rate);
    }

    #[test]
    fn probation_clamp_suppresses_drift() {
        // Under the clamp far fewer brain weights move per birth.
        let mut rng = seeded();
        let parent = Genome::generate(&mut rng);
        let changed = |child: &Genome| {
            child.brain_weights.iter().zip(&parent.brain_weights)
                .filter(|(a, b)| a != b).count()
        };

        let mut clamped_total = 0;
        let mut free_total = 0;
        for _ in 0..20 {
            clamped_total += changed(&parent.mutate(&mut rng, 1.0, 0.15));
            free_total += changed(&parent.mutate(&mut rng, 1.0, 1.0));
        }
        assert!(clamped_total * 2 < free_total, "clamped {} vs free {}", clamped_total, free_total);
    }

    #[test]
    fn weights_array_succeeds() {
        let mut rng = seeded();
        let g = Genome::generate(&mut rng);
        let _arr: &[f32; WEIGHT_COUNT] = g.weights_array();
    }
}
