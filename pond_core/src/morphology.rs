use crate::genome::Traits;

/// Trait-derived shape knobs for the renderer body silhouette. Each field is
/// normalized to [0, 1] against the trait's bounds in RULES.md, so the
/// renderer never needs to know raw trait ranges — this is the single place
/// where "what does trait X look like" is decided.
#[derive(Debug, Clone, Copy, Default)]
pub struct MorphParams {
    pub pointiness: f32,  // aggression: head cap wedge vs round, forward mass, fin rake
    pub elongation: f32,  // speed: body length / taper sharpness
    pub bulk: f32,        // defense: mid-body width, armor scalloping
    pub ornament: f32,    // attack: head spike length
    pub eye_size: f32,    // vision: eye radius / lateral spread
    pub pulse_rate: f32,  // metabolism: glow pulse speed
    pub belly: f32,       // energy_capacity: belly bulge amplitude
}

impl MorphParams {
    pub fn from_traits(t: &Traits) -> Self {
        Self {
            pointiness: norm(t.aggression, 0.0, 1.05),
            elongation: norm(t.speed, 0.5, 1.0),
            bulk: norm(t.defense, 0.5, 1.07),
            ornament: norm(t.attack, 0.5, 1.25),
            eye_size: norm(t.vision, 0.5, 1.05),
            pulse_rate: norm(t.metabolism, 0.5, 1.05),
            belly: norm(t.energy_capacity, 0.95, 1.05),
        }
    }
}

fn norm(v: f64, lo: f64, hi: f64) -> f32 {
    (((v - lo) / (hi - lo)).clamp(0.0, 1.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_clamp_to_unit_range() {
        let t = Traits {
            vision: 1.05, speed: 1.0, metabolism: 1.05,
            energy_capacity: 1.05, mutation_rate: 0.1,
            reproduction_cost: 1.0, attack: 1.25, defense: 1.07,
            aggression: 1.05, intelligence: 1.05, immunity: 1.0,
        };
        let m = MorphParams::from_traits(&t);
        for v in [m.pointiness, m.elongation, m.bulk, m.ornament, m.eye_size, m.pulse_rate, m.belly] {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn min_traits_give_zero() {
        let t = Traits {
            vision: 0.5, speed: 0.5, metabolism: 0.5,
            energy_capacity: 0.95, mutation_rate: 0.1,
            reproduction_cost: 1.0, attack: 0.5, defense: 0.5,
            aggression: 0.0, intelligence: 0.5, immunity: 0.0,
        };
        let m = MorphParams::from_traits(&t);
        for v in [m.pointiness, m.elongation, m.bulk, m.ornament, m.eye_size, m.pulse_rate, m.belly] {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }
}
