//! Diseases: the second disturbance regime.
//!
//! A disease arrives with a lineage. When a species is promoted there is a flat,
//! low chance it turns out to have been carrying something, and that pathogen
//! then spreads by contact for the rest of the run.
//!
//! Three rules keep it a disturbance rather than a population controller, and
//! all three are easy to break by being helpful:
//!
//! 1. **The origin roll is flat.** Not weighted by population, species age, or
//!    how dominant the lineage is. A disease that is more likely to appear when
//!    the pond is crowded is a density-dependent cull wearing a costume.
//! 2. **Nothing scales with total population.** Transmission depends on *local*
//!    crowding only. An outbreak in a tight cluster runs the same whether the
//!    pond holds 40 agents or 400, which is what lets it overshoot and crash
//!    instead of trimming toward a setpoint.
//! 3. **There is no recovery.** An infected agent carries it until it dies.
//!    Recovery would give the system a restoring force and turn every outbreak
//!    into a damped oscillation converging on an equilibrium — the exact
//!    behaviour these mechanics exist to prevent.
//!
//! Severity is an energy drain, so an outbreak interacts with the food economy:
//! it kills the already-marginal first and hits hardest exactly when the pond is
//! hungry. Death is still attributed to `Disease`, not starvation, so the cause
//! breakdown stays honest.

use serde::{Deserialize, Serialize};

/// A pathogen. Created at a promotion, then never modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disease {
    /// Monotonic, never reused. 0 means "not infected" in `World::infection`.
    pub id: u32,
    /// Latin-ish name derived from the host species — see `naming::disease_name`.
    pub name: String,
    /// The species it emerged in. Members of that species catch it at full
    /// contagion; everything else needs the cross-species jump.
    pub origin_species: u32,
    /// Energy drained per tick from an infected agent, scaled by metabolism.
    pub severity: f64,
    /// Base per-contact infection probability at full local crowding.
    pub contagion: f64,
    /// Step it first appeared.
    pub emerged_step: u32,
    /// Set once it has crossed into a second species. After that it is no longer
    /// a disease *of* one lineage and spreads at full contagion to anything.
    ///
    /// The jump is a single rare event rather than a permanent low leak. A
    /// per-contact probability, however small, is not rare in aggregate — every
    /// carrier rolls it against every neighbour every tick, so over an outbreak
    /// it fires thousands of times and the disease is simply endemic everywhere.
    /// Measured with a 2% leak: 60 of 60 agents in a scrum, in 40 ticks.
    pub jumped: bool,
}

/// Chance that a newly promoted species turns out to be carrying something.
///
/// Flat by design. See the module docs: weighting this by population or
/// dominance is the single easiest way to turn disease into a cull.
///
/// Read per *promotion*, and promotions are rare — a 3000-step run at
/// 12×12/400 sees about two. At 0.18 that is a two-in-three chance of a run
/// having no disease at all, which is a mechanic that does not exist. 0.30 puts
/// a pathogen in roughly half of runs while leaving any individual lineage
/// mostly clean.
pub const DISEASE_CHANCE: f64 = 0.30;

/// Severity range: energy per tick, scaled by metabolism. The top of this is
/// comparable to `BASE_DRAIN`, i.e. a doubling of the cost of being alive.
pub const SEVERITY_RANGE: (f64, f64) = (0.02, 0.14);
/// Contagion range: per-contact infection chance at full local crowding.
pub const CONTAGION_RANGE: (f64, f64) = (0.02, 0.30);
/// Radius, in tiles, within which contact can transmit.
pub const CONTACT_RADIUS: f32 = 1.1;
/// Neighbours within `CONTACT_RADIUS` at which crowding is considered maximal.
/// Transmission scales linearly up to this and clamps, so an outbreak is driven
/// by how tightly packed a lineage is, not by how many agents exist.
pub const CROWDING_FULL: f64 = 6.0;
/// Per-contact chance that a pathogen crosses into a second species, once, and
/// stops being a disease of one lineage.
///
/// Deliberately tiny: this fires against every susceptible neighbour of every
/// carrier on every tick, so an outbreak of any size rolls it thousands of
/// times. It is the probability that a *run* sees a jump, spread thin.
///
/// A real outbreak — 40 carriers, a handful of susceptible neighbours each,
/// a thousand ticks — makes on the order of 100k rolls. At 4e-5 that is eight
/// expected jumps, i.e. a certainty, and two of the first three measured
/// outbreaks did jump. At 1.5e-6 it is a rare event that reshapes the odd run,
/// which is what it is for.
pub const CROSS_SPECIES_JUMP: f64 = 0.0000015;
