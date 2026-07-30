use rand::Rng;
use serde::{Deserialize, Serialize};

// Matches landscape.py constants
const MAX_FERTILITY: f64 = 1.6;
pub const MAX_FOOD_PER_TILE: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeTile {
    pub fertility: f64,
    pub food_units: u32,
    pub movement_speed: f64,
    pub visibility: f64,
}

impl BiomeTile {
    /// Generate a random biome tile. Matches `Biome.generate()` in landscape.py.
    /// Python draw order: movement_speed, visibility, food_units (choice), fertility.
    pub fn generate(rng: &mut impl Rng) -> Self {
        let movement_speed = rng.gen_range(0.8_f64..=1.05);
        let visibility = rng.gen_range(0.25_f64..=1.0);
        // Python: r.choice([1, 2, 3]) — uniform over {1, 2, 3}
        let food_units = rng.gen_range(1_u32..=3);
        let fertility = rng.gen_range(0.01_f64..=1.6);
        Self { fertility, food_units, movement_speed, visibility }
    }

    /// Per-tick food regen probability. Matches `Biome.get_regen_rate()`.
    ///
    /// `scale` is the world's `Tunables::food_regen_scale` (0.012 by default,
    /// the old hardcoded constant). Fertility normalization stays here because
    /// it is the tile's own business; the rate is the world's, and is a dial.
    pub fn regen_rate(&self, scale: f64) -> f64 {
        (self.fertility / MAX_FERTILITY * scale).min(1.0)
    }

    /// Barren tile: zero fertility and food. Applied by the desert-cluster pass.
    pub fn make_barren(&mut self) {
        self.fertility = 0.0;
        self.food_units = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::DEFAULT_FOOD_REGEN_SCALE;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn generate_values_in_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let t = BiomeTile::generate(&mut rng);
            assert!((0.01..=1.6).contains(&t.fertility));
            assert!((1..=3).contains(&t.food_units));
            assert!((0.8..=1.05).contains(&t.movement_speed));
            assert!((0.25..=1.0).contains(&t.visibility));
        }
    }

    #[test]
    fn regen_rate_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..100 {
            let t = BiomeTile::generate(&mut rng);
            let r = t.regen_rate(DEFAULT_FOOD_REGEN_SCALE);
            assert!((0.0..=1.0).contains(&r));
        }
    }

    #[test]
    fn regen_rate_max_fertility() {
        let tile = BiomeTile { fertility: 1.6, food_units: 1, movement_speed: 1.0, visibility: 1.0 };
        let expected = 0.012_f64;
        assert!((tile.regen_rate(DEFAULT_FOOD_REGEN_SCALE) - expected).abs() < 1e-10);
    }

    #[test]
    fn regen_rate_barren() {
        let mut tile = BiomeTile { fertility: 0.5, food_units: 2, movement_speed: 1.0, visibility: 0.5 };
        tile.make_barren();
        assert_eq!(tile.fertility, 0.0);
        assert_eq!(tile.food_units, 0);
        assert_eq!(tile.regen_rate(DEFAULT_FOOD_REGEN_SCALE), 0.0);
    }
}
