use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::brain::{initial_weights, WEIGHT_COUNT};

/// All 12 genome traits. Bounds match `genomes/genome.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Traits {
    pub vision: f64,
    pub speed: f64,
    pub metabolism: f64,
    pub daily_nutrition_minimum: f64,
    pub energy_capacity: f64,    // locked (D3)
    pub mutation_rate: f64,      // locked (D3)
    pub clone_energy_threshold: f64,
    pub reproduction_cost: f64,
    pub intelligence: f64,
    pub attack: f64,
    pub defense: f64,
    pub aggression: f64,
}

impl Traits {
    /// Generate random founding trait values within JSON-defined bounds.
    /// Draw order matches Python's dict iteration order (insertion order, Python 3.7+).
    pub fn generate(rng: &mut impl Rng) -> Self {
        Self {
            vision:                  rng.gen_range(0.5_f64..=1.05),
            speed:                   rng.gen_range(0.5_f64..=1.0),
            metabolism:              rng.gen_range(0.5_f64..=1.05),
            daily_nutrition_minimum: rng.gen_range(0.95_f64..=1.0),
            energy_capacity:         rng.gen_range(0.95_f64..=1.05),
            mutation_rate:           rng.gen_range(0.01_f64..=0.25),
            clone_energy_threshold:  rng.gen_range(0.5_f64..=1.05),
            reproduction_cost:       rng.gen_range(0.75_f64..=1.50),
            intelligence:            rng.gen_range(0.5_f64..=1.05),
            attack:                  rng.gen_range(0.5_f64..=1.25),
            defense:                 rng.gen_range(0.5_f64..=1.07),
            aggression:              rng.gen_range(0.0_f64..=1.05),
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
            daily_nutrition_minimum: maybe_mutate!(self.daily_nutrition_minimum, 0.95, 1.0),
            // Locked — no RNG draw (D3)
            energy_capacity:         self.energy_capacity,
            mutation_rate:           self.mutation_rate,
            clone_energy_threshold:  maybe_mutate!(self.clone_energy_threshold, 0.5, 1.05),
            reproduction_cost:       maybe_mutate!(self.reproduction_cost, 0.75, 1.50),
            intelligence:            maybe_mutate!(self.intelligence, 0.5, 1.05),
            attack:                  maybe_mutate!(self.attack, 0.5, 1.25),
            defense:                 maybe_mutate!(self.defense, 0.5, 1.07),
            aggression:              maybe_mutate!(self.aggression, 0.0, 1.05),
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
        }
    }

    /// Produce a mutated offspring genome.
    /// `suppression` = 1.0 / (1.0 + parent_success_count * k) from AgentMemory.
    pub fn mutate(&self, rng: &mut impl Rng, suppression: f32) -> Self {
        let eff_rate = self.effective_mutation_rate * suppression;
        let rate_f64 = eff_rate as f64;
        let magnitude = rate_f64 * 0.5;

        let new_traits = self.traits.mutate(eff_rate, rng);

        let new_weights: Vec<f32> = self
            .brain_weights
            .iter()
            .map(|&w| {
                if rng.gen::<f64>() < rate_f64 {
                    let factor = rng.gen_range((1.0 - magnitude)..=(1.0 + magnitude)) as f32;
                    w * factor
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
            effective_mutation_rate: eff_rate,
        }
    }

    /// Convenience: mutate with no suppression (founding generations, tests).
    pub fn mutate_unsuppressed(&self, rng: &mut impl Rng) -> Self {
        self.mutate(rng, 1.0)
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
    fn generate_trait_bounds() {
        let mut rng = seeded();
        for _ in 0..50 {
            let g = Genome::generate(&mut rng);
            let t = &g.traits;
            assert!((0.5..=1.05).contains(&t.vision));
            assert!((0.5..=1.0).contains(&t.speed));
            assert!((0.5..=1.05).contains(&t.metabolism));
            assert!((0.95..=1.0).contains(&t.daily_nutrition_minimum));
            assert!((0.95..=1.05).contains(&t.energy_capacity));
            assert!((0.01..=0.25).contains(&t.mutation_rate));
            assert!((0.5..=1.05).contains(&t.clone_energy_threshold));
            assert!((0.75..=1.50).contains(&t.reproduction_cost));
            assert!((0.5..=1.05).contains(&t.intelligence));
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
        let child = parent.mutate(&mut rng, suppression);
        assert!(child.effective_mutation_rate < parent.effective_mutation_rate);
    }

    #[test]
    fn weights_array_succeeds() {
        let mut rng = seeded();
        let g = Genome::generate(&mut rng);
        let _arr: &[f32; WEIGHT_COUNT] = g.weights_array();
    }
}
