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
//! 3. **Recovery is fixed by the pathogen, not by the pond.** Each disease has
//!    a `duration`; immunity shortens an individual's share of it and blunts
//!    the drain, and nothing else touches either. There is no acquired
//!    immunity: surviving is not a permanent ticket, so an outbreak can come
//!    back through the same animals.
//!
//!    Illness used to be terminal, which sounded like the anti-equilibrium
//!    choice and was the opposite. Measured over 100k ticks, seed 7 recorded
//!    15,268 disease deaths out of 18,176 — 84%. An infection that always kills
//!    is a death sentence with a variable delay, so a pathogen that jumps
//!    species eventually owns the pond and *is* the equilibrium.
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
    /// How long the illness runs, in ticks, before an agent shrugs it off.
    /// Rolled once at creation and fixed — a property of the pathogen, never of
    /// the population, or the mechanic becomes a controller.
    pub duration: u32,
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
/// How long an illness lasts, in ticks, before recovery.
///
/// Bounded under the longest death age the pool produces (~674, see
/// `create_death_range`): a disease must be survivable in principle, or
/// "recovery" is a word for a slower death.
pub const ILLNESS_TICKS: (u32, u32) = (150, 520);
/// Share of an illness's length immunity can remove. At full immunity an agent
/// is ill for `1 - this` of the pathogen's duration — still ill, briefly.
pub const IMMUNITY_DURATION_RELIEF: f64 = 0.75;
/// Share of the per-tick drain immunity can remove.
pub const IMMUNITY_SEVERITY_RELIEF: f64 = 0.70;
/// Chance of coming out of an illness permanently resistant to *that* pathogen,
/// at full immunity. Scaled by the gene, so an animal with no immune system
/// never acquires anything — there is nothing there to remember with.
///
/// Acquired resistance is per-disease, per-animal, and dies with the animal: it
/// is not inherited. The gene is the heritable half and this is the earned
/// half, and keeping them separate is what stops one outbreak vaccinating a
/// whole lineage forever.
pub const ACQUIRED_IMMUNITY_CHANCE: f64 = 0.6;
/// Diseases a single agent can be resistant to, set by the bitmask width. Runs
/// produce a handful of pathogens, so 64 is not a limit anything reaches.
pub const MAX_TRACKED_DISEASES: u32 = 64;

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
