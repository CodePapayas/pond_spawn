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

/// How much of an individual's own deviation from its species' founding shape
/// survives into its body.
///
/// Below 1.0 so a species reads as a *kind*: every member is drawn around the
/// shape the lineage had when it was promoted, with personal variation as a
/// perturbation rather than as the whole signal. At 1.0 this would be identical
/// to `from_traits` and a species would have no look of its own.
const INDIVIDUAL_VARIATION: f32 = 0.35;

impl MorphParams {
    /// Shape from raw traits. Used for agents with no species — they have no
    /// lineage to vary around, so they are simply themselves.
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

    /// Shape anchored to the species' **founding** centroid, with the agent's
    /// own deviation from it as variation around that fixed base.
    ///
    /// The anchor has to be the founding centroid and not `Species::centroid`,
    /// which EMA-tracks the live member mean every cluster tick: keyed to that,
    /// an animal's appearance changes when its neighbours' does, and a species
    /// has no stable look at all. Anchored at promotion, a lineage keeps the
    /// shape it earned its name with, and individuals vary around it.
    ///
    /// `founding` is a species signature — normalized [0,1] per dim, indexed by
    /// `species::SIGNATURE_DIMS` — so the trait dims it does not cover
    /// (`energy_capacity`, `mutation_rate`) fall back to the agent's own value.
    pub fn from_species_deviation(t: &Traits, founding: &[f64]) -> Self {
        let own = Self::from_traits(t);
        // Signature position of each trait index we care about, or None if the
        // trait is not in the signature.
        let slot = |trait_dim: usize| -> Option<usize> {
            crate::species::SIGNATURE_DIMS.iter().position(|&d| d == trait_dim)
        };
        let anchored = |trait_dim: usize, own_value: f32| -> f32 {
            match slot(trait_dim).and_then(|i| founding.get(i)) {
                Some(&base) => {
                    let base = base as f32;
                    (base + (own_value - base) * INDIVIDUAL_VARIATION).clamp(0.0, 1.0)
                }
                None => own_value,
            }
        };

        Self {
            pointiness: anchored(8, own.pointiness),   // aggression
            elongation: anchored(1, own.elongation),   // speed
            bulk: anchored(7, own.bulk),               // defense
            ornament: anchored(6, own.ornament),       // attack
            eye_size: anchored(0, own.eye_size),       // vision
            pulse_rate: anchored(2, own.pulse_rate),   // metabolism
            belly: own.belly,                          // energy_capacity: locked, not in the signature
        }
    }
}

fn norm(v: f64, lo: f64, hi: f64) -> f32 {
    (((v - lo) / (hi - lo)).clamp(0.0, 1.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::SIGNATURE_DIMS;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// A founding signature with every dim at `v`.
    fn founding(v: f64) -> Vec<f64> {
        vec![v; SIGNATURE_DIMS.len()]
    }

    #[test]
    fn a_species_shape_dominates_the_individual() {
        // Two very different animals of the same lineage must still look like
        // that lineage — that is the whole point of anchoring.
        let mut a = Traits::generate(&mut ChaCha8Rng::seed_from_u64(1));
        let mut b = a.clone();
        a.aggression = 0.0;   // opposite extremes on a signature trait
        b.aggression = 1.05;

        let base = founding(0.5);
        let ma = MorphParams::from_species_deviation(&a, &base);
        let mb = MorphParams::from_species_deviation(&b, &base);

        let raw_gap = (MorphParams::from_traits(&a).pointiness
            - MorphParams::from_traits(&b).pointiness).abs();
        let anchored_gap = (ma.pointiness - mb.pointiness).abs();
        assert!(anchored_gap < raw_gap,
            "anchoring did not pull members together: {} vs {}", anchored_gap, raw_gap);
        // Both sit near the lineage's own shape rather than at their own extremes.
        assert!((ma.pointiness - 0.5).abs() < 0.2 && (mb.pointiness - 0.5).abs() < 0.2);
    }

    #[test]
    fn two_lineages_look_different_even_with_identical_members() {
        // The same animal, promoted into two different lineages, is drawn as two
        // different kinds. Species identity is in the anchor, not the genome.
        let t = Traits::generate(&mut ChaCha8Rng::seed_from_u64(2));
        let lean = MorphParams::from_species_deviation(&t, &founding(0.1));
        let heavy = MorphParams::from_species_deviation(&t, &founding(0.9));
        assert!((lean.bulk - heavy.bulk).abs() > 0.4,
            "two lineages produced the same body: {} vs {}", lean.bulk, heavy.bulk);
    }

    #[test]
    fn an_unassigned_agent_is_simply_itself() {
        let t = Traits::generate(&mut ChaCha8Rng::seed_from_u64(3));
        let m = MorphParams::from_traits(&t);
        assert_eq!(m.pointiness, norm(t.aggression, 0.0, 1.05));
    }

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
