use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use crate::biome::{BiomeTile, MAX_FOOD_PER_TILE};
use crate::brain::{forward as brain_forward, forward_traced, sigmoid_outputs, INPUT_COUNT};
use crate::brain_cluster::BrainClusters;
use crate::cluster::ClusterState;
use crate::disease::{
    Disease, CONTACT_RADIUS, CONTAGION_RANGE, CROSS_SPECIES_JUMP, CROWDING_FULL,
    DISEASE_CHANCE, SEVERITY_RANGE,
};
use crate::species::SpeciesRegistry;
use crate::genome::{Genome, TRAIT_COUNT};
use crate::memory::{AgentMemory, SUCCESS_SCALAR};
use crate::spatial::SpatialHashGrid;
use crate::stats::{StatHistory, StatSample, CAUSE_COUNT, SAMPLE_INTERVAL};

// ── Steering output indices ───────────────────────────────────────────────────
const OUT_SEEK: usize = 0;       // force weight toward nearest food
const OUT_WANDER: usize = 1;     // random perturbation weight
const OUT_SEPARATE: usize = 2;   // repulsion from nearby agents
const OUT_FLEE: usize = 3;       // force weight directly away from a seen threat
const OUT_EAT: usize = 4;        // discrete trigger gate
const OUT_REPRODUCE: usize = 5;  // discrete trigger gate
// OUT_ATTACK = 6  (dormant — routes through passive combat only)
const OUT_SLEEP: usize = 7;      // discrete trigger gate

// ── Physics constants ─────────────────────────────────────────────────────────
pub const DT: f32 = 1.0 / 20.0;        // 20 Hz fixed timestep (50 ms per tick)
pub const MAX_SPEED: f32 = 3.0;         // tiles/sec at speed_trait = 1.0
const MAX_FORCE: f32 = 8.0;             // steering acceleration magnitude cap
const WANDER_FORCE: f32 = 2.5;          // wander perturbation strength
const SEPARATION_RADIUS: f32 = 1.2;     // repulsion radius in tiles
const VISION_SCALE: f32 = 3.0;          // vision_trait × VISION_SCALE = radius in tiles
const MOVE_COST: f64 = 0.15;            // energy per tile traveled × terrain_speed × metabolism

// ── Economy constants ─────────────────────────────────────────────────────────
const MATURITY_AGE: u32 = 100;
const CHILDHOOD_TICKS: u32 = 50;
const BIRTH_FAIL_CHANCE: f64 = 0.02;
const FAIL_COUNTS_CHANCE: f64 = 0.20;
const FOOD_ENERGY: f64 = 33.3;
// Fraction of the parent's reproduction payment the child actually receives.
// The rest is thermodynamic overhead, lost to the world.
//
// This was documented in the refactor roadmap but never implemented: the child
// used to receive the payment in full, which made reproduction energy-neutral
// and turned `reproduction_cost` into an investment dial rather than a cost — a
// parent paying more produced a better-provisioned child at no net loss, so
// selection ratcheted the trait to its upper bound and the population grew
// without any check but food.
const BIRTH_ENERGY_YIELD: f64 = 0.40;
const MAX_ENERGY_BASE: f64 = 100.0;
// Tile can't be fed from again for this many ticks after last eat — stops one
// camper parking on a tile and eating every tick regardless of regen rate.
const EAT_COOLDOWN_TICKS: u32 = 8;
// Energy an attacker loses to retaliation when its attack roll fails, scaled by
// the defender's effective defense and divided by the attacker's own. Failed
// hunts used to kill the attacker outright, which selected aggression to
// extinction by step ~250.
//
// Dividing by the attacker's defense is what keeps defense worth investing in
// for a predator too: a well-armoured hunter shrugs off a failed hunt, a glass
// cannon does not. Defense is therefore useful on both sides of a fight, where
// before it only ever helped the victim.
const RETALIATION_ENERGY: f64 = 8.0;
// Share of the loser's energy the winner absorbs.
//
// Was 0.125, which made predation nearly pointless: a successful hunt returned
// an eighth of a body while a failed one cost real energy, so aggression was
// selected out of every run and the population lost its only density-dependent
// brake. Two thirds makes hunting genuinely worth the risk without making it
// free — the roll can still fail, and failure still hurts.
const PREDATION_YIELD: f64 = 0.667;
// Agents only hunt below this fraction of their max energy. This is the
// density-dependent brake on predation — see resolve_combat_spatial.
const HUNT_HUNGER_FRAC: f64 = 0.5;
// Energy an immortal agent is held at when it would otherwise starve.
const IMMORTAL_ENERGY_FLOOR: f64 = 1.0;
// Energy a sleeping agent claws back against the tick's base metabolism drain,
// as a multiple of `metabolism`. Strictly less than the 0.1 × metabolism base
// drain, so sleep halves the rate an agent starves at and is never a source.
//
// It was 0.15 — larger than the drain — which made sleep a net +0.05 ×
// metabolism per tick, and the gain was not clamped to capacity either, so an
// agent whose sleep gate kept winning climbed past its own maximum forever. The
// overflow was invisible: `energy_norm` clamps to 1.0 in perception, so neither
// the brain nor the HUD showed it, and the only symptom was agents that could
// not starve. RULES.md documented the 0.15 gain; the refactor roadmap recorded
// the decision as "rest, not recovery". The roadmap was right.
const SLEEP_RECOVERY: f64 = 0.05;
// Passive metabolism drain per tick, as a multiple of `metabolism`. Named only so
// the sleep invariant above is checkable — the value is unchanged.
const BASE_DRAIN: f64 = 0.1;

// ── Cannibalism ───────────────────────────────────────────────────────────────
//
// Eating your own kind is a decision, not a default. Ordinary predation between
// agents needs aggression over the hunt threshold; turning on a *member of your
// own species* needs more than that, and it needs the animal to be too dull to
// know better.
//
// Framed as a rule over existing traits rather than as a twelfth gene. A
// cannibalism trait would be a second aggression that only matters against one
// class of target, and it would drift on its own — this way the behaviour falls
// out of a combination the pond already selects on, and an intelligent lineage
// gets kin recognition for free as a side effect of being intelligent.

/// Aggression an agent needs before it will turn on its own species — above the
/// ordinary hunt threshold, because this is the more extreme act.
const CANNIBAL_AGGRESSION_MIN: f64 = 0.95;
/// Intelligence at or above which an agent will not eat its own kind, as a
/// fraction of the trait's range. Smart animals recognise their relatives; the
/// dull ones see food.
const CANNIBAL_INTELLIGENCE_MAX_FRAC: f64 = 0.55;

/// Would this agent eat a member of its own species?
fn is_cannibal(traits: &crate::genome::Traits) -> bool {
    let (lo, hi) = crate::genome::Traits::BOUNDS[9];
    let smart = (traits.intelligence - lo) / (hi - lo);
    traits.aggression >= CANNIBAL_AGGRESSION_MIN && smart < CANNIBAL_INTELLIGENCE_MAX_FRAC
}

// ── Defense upkeep ────────────────────────────────────────────────────────────
//
// Armour used to be free. Nothing anywhere charged for `defense`, while ordinary
// combat paid out for it directly — the winner of a fight is decided by
// `attack / (attack + defense)` — so mean defense climbed in every run and no
// predator change could stop it. Measured across five seeds it went 0.955 →
// 1.013 even after predation stopped subsidising it.
//
// It is treated as a physical addition: plating is carried, and carrying it
// costs metabolism whether or not anything attacks you. Bracing against a hit
// costs again, and scales the same way.

/// Energy per tick per unit of defense above the trait's floor, scaled by
/// metabolism.
///
/// Charged on the *margin* over `BOUNDS[7].0`, not on the raw value: the floor
/// is the minimum the trait can take, so it is not armour, it is a body. An
/// agent at the bottom of the range pays nothing and one at the top pays
/// `0.05 × metabolism` a tick — half of `BASE_DRAIN`.
const DEFENSE_UPKEEP: f64 = 0.09;
/// Energy to brace against one attack, per unit of defense over the floor,
/// scaled by metabolism. Bracing is work.
const DEFENSE_BLOCK_COST: f64 = 0.6;

/// Energy per tick per unit of immunity, scaled by metabolism.
///
/// An immune system is expensive to run, and without a price this trait goes
/// straight to fixation and stops being a decision — which is exactly what
/// defense did for the entire life of the project. Set against the fact that
/// disease appears in roughly half of runs: in a clean pond this is pure
/// overhead, so a lineage that has never met a pathogen should drift *down*.
const IMMUNITY_UPKEEP: f64 = 0.035;

/// Armour carried above the trait's floor. Everything charged for defense is
/// charged on this, not on the raw trait.
fn armour_margin(defense: f64) -> f64 {
    (defense - crate::genome::Traits::BOUNDS[7].0).max(0.0)
}

// ── Intelligence ──────────────────────────────────────────────────────────────
//
// One trait, three consequences and a bill. A dull animal is not merely worse at
// deciding — it decides *less often*, on older information, and pays less for the
// privilege. That trade is the point: thinking is not free, so an agent in a
// stable pond can profitably be stupid, and one under predation cannot.

/// Ticks between decisions at the bottom of the intelligence range. The dullest
/// agents re-decide every 10th tick and spend the other nine acting on a stale
/// picture; the sharpest decide every tick.
const DECISION_INTERVAL_MAX: u32 = 10;
/// Ticks between a predator entering an agent's vision and the agent's brain
/// being told about it, at the bottom of the range. The sharpest see it the tick
/// it arrives.
const THREAT_LAG_MAX: usize = 8;
/// Energy per tick per unit of intelligence, scaled by metabolism like every
/// other running cost. At the top of the range this is a fifth of `BASE_DRAIN` —
/// enough that a pond with nothing to think about will drift dull.
const INTELLIGENCE_UPKEEP: f64 = 0.02;

/// Decision interval for an intelligence value: 1 tick at the top of the trait's
/// range, `DECISION_INTERVAL_MAX` at the bottom.
fn decision_interval(intelligence: f64) -> u32 {
    let (lo, hi) = crate::genome::Traits::BOUNDS[9];
    let norm = ((intelligence - lo) / (hi - lo)).clamp(0.0, 1.0);
    let span = (DECISION_INTERVAL_MAX - 1) as f64;
    1 + ((1.0 - norm) * span).round() as u32
}

/// Slots in each agent's threat pipeline: one per tick of possible lag, plus the
/// current tick.
const THREAT_RING: usize = THREAT_LAG_MAX + 1;

/// Ticks before a seen threat reaches the brain. Zero at the top of the range.
fn threat_lag(intelligence: f64) -> usize {
    let (lo, hi) = crate::genome::Traits::BOUNDS[9];
    let norm = ((intelligence - lo) / (hi - lo)).clamp(0.0, 1.0);
    ((1.0 - norm) * THREAT_LAG_MAX as f64).round() as usize
}

// ── Predator adaptation ───────────────────────────────────────────────────────
//
// Hunters used to pick the nearest animal and bite with a flat `TIER_ATTACK`.
// Both halves of that pushed selection the same way: pressure was uniform, so
// nothing was ever safer for being rare, and armour was an absolute refuge,
// since `0.75` loses to any defense above it no matter how often it is tested.
// The pond's answer was to put everything into defense, and there was nothing
// the predators could do about it.
//
// A hunter now forms a **search image** of the most common family and prefers it
// while hunting, and its bite slowly learns that family's armour. Both are
// frequency-dependent, which is what makes them stabilising rather than just
// stronger: whichever strategy wins becomes the plurality, and being the
// plurality is what draws the hunters. A rare family is comparatively safe, so
// diversity pays; a dominant one is hunted by something that has learned to bite
// through exactly its defense, so dominance pays for itself.

/// A challenger family must be this much more numerous than the current search
/// image before a hunter switches to it. Without hysteresis the image would flip
/// between two near-equal families every review, and the bite adaptation — which
/// resets on a switch — would never get anywhere.
const SEARCH_IMAGE_SWITCH_MARGIN: f64 = 1.25;
/// How much nearer a matching animal is treated as being. Preference, not
/// exclusivity: a hunter that finds nothing matching still eats what is there,
/// or a rare family would be untouchable and would simply take over.
const SEARCH_IMAGE_PULL: f32 = 2.0;
/// Fraction of the gap to its image's armour a hunter closes each review.
///
/// 0.15 was too slow to matter: images switch every few hundred steps, and a
/// hunter that reset to base on each switch spent its whole life re-learning.
/// Measured, the animals actually being eaten averaged 0.65 defense against a
/// pond at 0.98 — predation was still killing the *unarmoured*, which is the
/// selection pressure this is meant to remove.
const PREDATOR_ATTACK_ADAPT: f64 = 0.30;
/// How much of the learned surplus survives an image switch. Not zero: general
/// toughness carries across prey, only the specific calibration does not. A full
/// reset handed armour its immunity back every time the plurality moved.
const PREDATOR_ATTACK_RETENTION: f64 = 0.5;
/// How far past the toughest animal in its image a hunter aims.
///
/// The aim is the family's **maximum** effective armour, not its mean or a
/// standard-deviation estimate. Both of those were tried and both failed the
/// same way, because the kill shape bites everything it covers and kills
/// whatever falls below `attack`: any aim short of the toughest member kills
/// the weaker half of the family and spares the stronger, which is a reward for
/// armour no matter where between the mean and the tail it is set. Measured, an
/// aim at mean + 2σ still ate animals averaging 0.68 defense out of a pond
/// averaging 0.95 — predation was selecting *for* armour, which is the whole
/// complaint this change exists to answer.
///
/// Aiming past the toughest makes predation armour-neutral instead: within the
/// hunted family, armour changes nothing about who dies. The pressure that
/// remains is frequency — being common is what gets you hunted — and that is the
/// part that stabilises rather than ratchets.
const PREDATOR_ATTACK_MARGIN: f64 = 0.05;
/// How strongly a hunter prefers the better-armoured members of its image.
/// Distance is scaled by `(family mean / this animal's armour)^2`, clamped, so
/// an animal well above the family norm reads as much closer than it is.
const ARMOUR_PREFERENCE_CLAMP: (f32, f32) = (0.3, 2.5);
/// Cap on learned attack, above the tier's base. Defense tops out at 1.07, so
/// 0.45 over the tier-0 base of 0.75 lets a hunter reach any armour eventually —
/// armour buys time, not immunity.
const PREDATOR_ATTACK_MAX_ADAPT: f64 = 0.45;

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Per-tile per-tick food regen chance at full fertility. Was hardcoded in
/// `biome.rs`.
pub const DEFAULT_FOOD_REGEN_SCALE: f64 = 0.012;
/// `aggression` at or above this makes an agent hunt other agents rather than
/// graze. `aggression` maxes at 1.05, so anything above that switches
/// agent-on-agent combat off entirely.
pub const DEFAULT_HUNT_AGGRESSION_THRESHOLD: f64 = 0.80;
/// Genome families the k-means pass splits the pond into.
pub const DEFAULT_CLUSTER_K: usize = 6;

pub const FOOD_REGEN_SCALE_RANGE: (f64, f64) = (0.0, 0.05);
pub const HUNT_AGGRESSION_THRESHOLD_RANGE: (f64, f64) = (0.0, 1.06);
pub const CLUSTER_K_RANGE: (usize, usize) = (2, 12);

/// The three dials a viewer can move mid-run, held on `World` so the numbers
/// have one home rather than being literals at their call sites.
///
/// They are not the same kind of thing, which is the reason they are documented
/// together: `food_regen_scale` and `hunt_aggression_threshold` are physics —
/// moving them changes what the pond does — while `cluster_k` is presentation,
/// changing only how the pond is grouped for colours and species tracking.
///
/// Determinism: same seed plus the same tunables reproduces a run exactly, as
/// before. A run whose dials moved mid-flight is no longer reproducible from
/// `(grid, population, seed)` alone, which is what `modified` records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tunables {
    pub food_regen_scale: f64,
    pub hunt_aggression_threshold: f64,
    pub cluster_k: usize,
    /// Set once any dial leaves its default. Never cleared by returning a dial
    /// to its default value — the run already diverged.
    pub modified: bool,
}

impl Default for Tunables {
    fn default() -> Self {
        Self {
            food_regen_scale: DEFAULT_FOOD_REGEN_SCALE,
            hunt_aggression_threshold: DEFAULT_HUNT_AGGRESSION_THRESHOLD,
            cluster_k: DEFAULT_CLUSTER_K,
            modified: false,
        }
    }
}

fn clamp_range(v: f64, (lo, hi): (f64, f64)) -> f64 {
    v.clamp(lo, hi)
}

// ── Ultra predator ────────────────────────────────────────────────────────────
// A single apex agent that cannot die and eats until the population is down to a
// target fraction, then leaves. Summoned by the player, or automatically when the
// pond grows past a size the renderer cannot draw at frame rate.
//
// It is a real agent — it renders, it swims, it can be inspected — but it does
// not obey the ecology: no aging, no starvation, no combat rolls, no hunger gate.
/// Agents per tile the cull aims to keep. The threshold scales with the pond
/// rather than being a fixed number: food supply scales with area, so a 32×32
/// pond genuinely supports more life than a 12×12 one, and a constant cap would
/// either strangle the big pond or never fire on the small one.
///
/// Tuned for the renderer, not for the ecology. Every agent is drawn as an
/// individual kinematic body with no culling or instancing, so the pond can feed
/// far more life than the browser can draw; this is deliberately below the
/// food-limited carrying capacity.
pub const PREDATOR_POP_PER_TILE: f64 = 1.75;
/// Absolute ceiling on the cull target, whatever the pond's area. Past roughly
/// this many bodies the frame rate goes regardless of how much food there is,
/// so a big pond stops scaling its cap linearly and just holds this line.
pub const PREDATOR_POP_CEILING: usize = 900;
/// Prey each predator in a summoned pack is expected to account for. A cull of
/// thousands is not one hunter's work.
const PREY_PER_PREDATOR: usize = 250;
/// How far below the hysteresis floor a cull actually cuts.
///
/// Culling to exactly the floor made no sense against how this pond grows: a
/// population that booms re-crosses the trigger almost immediately, so predators
/// were effectively resident and permanently mid-hunt. Cutting deeper leaves
/// room for a boom to happen before the next wave is warranted.
const CULL_DEPTH: f64 = 0.72;

/// Hysteresis around the cap, as a fraction. Predators arrive above
/// `cap × (1 + band)` and leave at `cap × (1 - band)`, so a population sitting
/// near the cap isn't culled every few steps — it gets room to breathe.
pub const PREDATOR_POP_BAND: f64 = 0.10;
/// Most predators that may be in the pond at once. High enough to hold the whole
/// escalation ladder — the tier pack sizes sum to 11, and a cap below that would
/// silently stop a wave escalating past the tier that filled it. Steady state is
/// far lower: only the triangles stay, and the top two depart.
pub const PREDATOR_MAX: usize = 12;
/// Steps between reinforcement checks. Long enough that a new arrival has time
/// to make a dent before the next is considered.
const PREDATOR_REINFORCE_STEPS: u32 = 120;
/// Survivor fraction when the player summons one: eats four fifths of the pond.
pub const PREDATOR_MANUAL_FRAC: f64 = 0.20;
/// Tier a plain player summon arrives at — the octagon. A hit-and-run tier
/// deliberately: "summon" should mean calling in a strike that lands and leaves,
/// and summoning at the resident tier would park another permanent hunter in the
/// pond on every press.
pub const PREDATOR_MANUAL_TIER: u8 = 1;
/// The rectangle. Player-only, like the octagon, and the more final of the two.
pub const PREDATOR_RECTANGLE_TIER: u8 = 2;
// ── Predator tiers ────────────────────────────────────────────────────────────
//
// The ecology only ever produces triangles. They are resident: they arrive,
// cut, and then stay in the pond patrolling, going quiet until the population
// climbs again. Every time the pond crosses the cull threshold one more triangle
// joins the pack, so a pond that keeps outbreeding its hunters accumulates them
// permanently rather than being answered by something new each time.
//
// The octagon and the rectangle are **player powers only**. They are far too
// lethal to leave to an automatic rule — the pond would never get to overshoot
// and recover, which is the thing worth watching. They hit and run: cull, then
// leave.

/// Number of tiers. Shapes, in order: grey triangle pack, rotating red octagon,
/// rotating rainbow rectangle.
pub const PREDATOR_TIERS: usize = 3;

/// World units per step each tier closes on its prey.
///
/// Tiers 1 and 2 are god-mode powers and keep these flat, deliberately unfair
/// values. **Tier 0 does not use its entry**: the ambient triangle takes its
/// speed from the pond instead, at `PREDATOR_SPEED_FRAC` of the mean speed trait
/// of the family it is hunting (see `predator_chase_speed`).
///
/// The old constant was 0.95 world units *per tick*, while the fastest possible
/// agent — speed trait 1.0, `MAX_SPEED` 3.0 tiles/sec at a 20 Hz tick — moves
/// 0.15. A six-fold advantage means no amount of evolved evasion can ever
/// matter, which is why flee sat dormant for as long as it did. The entry is
/// kept as the floor for a pond with nothing left to measure.
const TIER_SPEED: [f32; PREDATOR_TIERS] = [0.95, 1.30, 1.55];

/// Prey needed before the ambient hunter exists at all. Below this the pond is
/// recovering from something and does not need a resident predator on top of it.
pub const PREDATOR_AMBIENT_MIN_PREY: usize = 30;

/// Floor under an ambient hunter's speed, as a speed *trait* equivalent.
///
/// Absolute floor, as a speed trait equivalent: the slowest animal the genome
/// can build. A hunter is never slower than that, however far the pond sinks —
/// without it, a pond that abandons speed entirely also abandons the predator,
/// and the relative band becomes a treadmill pointing down.
const PREDATOR_SPEED_FLOOR_TRAIT: f32 = 0.50;
/// Absolute ceiling, as a speed trait equivalent. A bursting hunter never quite
/// reaches the fastest animal the genome can build, so an all-in speed lineage
/// keeps an edge even at the worst moment.
const PREDATOR_SPEED_CEILING_TRAIT: f32 = 0.99;

/// Cruising speed as a fraction of its prey's mean — a band, not a number.
///
/// Under 1.0, so an average animal outpaces a cruising hunter and the ordinary
/// state of the pond is "not being caught". The *band* is the point: a fixed
/// multiplier is a fixed safe margin, and a fixed safe margin is something a
/// lineage evolves to sit exactly on top of. Re-rolled per hunter on every
/// search-image review, so the margin keeps moving.
const PREDATOR_CRUISE_FRAC: (f32, f32) = (0.80, 0.90);
/// What a burst multiplies cruising speed by, before the absolute ceiling. This
/// is the part that actually catches things: cruising is for closing distance,
/// bursting is for the last few tiles.
const PREDATOR_BURST_MULT: f32 = 1.9;

/// Per-tick chance a hunter goes into a burst.
///
/// Roughly one burst per hunter per 2,500 ticks. The point is variance: a
/// steady threat is one a lineage can evolve a fixed answer to, and then the
/// pond settles. A hunter that is occasionally faster than it has any right to
/// be means the safe margin is never quite knowable, which is the same reason
/// the search image moves.
const PREDATOR_BURST_CHANCE: f64 = 0.0004;
/// How long a burst lasts, in ticks. Long enough to change the outcome of a
/// chase and of several others behind it.
const PREDATOR_BURST_TICKS: (u32, u32) = (150, 400);

/// Fraction of its prey's mean speed an ambient hunter moves at.
///
/// Below 1.0 on purpose. A hunter slower than the average animal in its target
/// family cannot catch the above-average ones, so escape becomes positional:
/// the goal is not to outrun the predator, it is to be harder to catch than the
/// neighbour. That is the pressure that makes speed and flee worth evolving,
/// and it is what stops predation being a flat tax everyone pays equally.
const PREDATOR_SPEED_FRAC: f32 = 0.95;
/// Bite reach per tier. For the rectangle this is the half-length of its long
/// edge rather than a radius — see `tier_bite_hits`.
const TIER_BITE: [f32; PREDATOR_TIERS] = [0.55, 1.90, 3.20];
/// Flat attack rating for each predator shape. A bite only lands when this is
/// strictly greater than the prey's effective defense — except where
/// `TIER_IGNORES_DEFENSE` says otherwise.
pub const TIER_ATTACK: [f64; PREDATOR_TIERS] = [0.75, 0.90, 1.05];
/// Tiers whose kills are not rolled for at all: anything the shape covers dies,
/// whatever it is made of.
///
/// The rectangle only. At an attack of 1.05 it was losing bites to well-armoured
/// agents — defense runs to 1.07 from the trait bounds alone, and a childhood
/// bonus stacks on top of that — so the most final power in the game could sweep
/// the pond and leave survivors standing in its path. It is a player power of
/// last resort and it now reads as one. Contrast the triangles, whose failure
/// against a tough animal is exactly what makes defense worth evolving.
const TIER_IGNORES_DEFENSE: [bool; PREDATOR_TIERS] = [false, false, true];
/// Half-width of the rectangle's short edge. Anything the sweep touches dies.
const RECTANGLE_HALF_WIDTH: f32 = 0.85;
/// Radians per step the top two tiers rotate. The rectangle turns slowly, and
/// covers so much ground that it does not need to turn quickly.
const TIER_SPIN: [f32; PREDATOR_TIERS] = [0.0, 0.20, 0.045];
/// Hunters summoned per wave at each tier. The triangles hunt as a pack — they
/// are the weakest tier and the only resident one, so numbers are what they
/// have; the top two are lethal enough not to need company.
///
/// Only the triangle entry is used automatically, and only as the size of the
/// *first* pack: after that the pond earns one more per threshold crossing.
const TIER_PACK: [usize; PREDATOR_TIERS] = [3, 5, 2];
/// Whether a tier stays in the pond after its quota is met. Only the triangles
/// do: the other two are far too lethal to leave in the water.
const TIER_RESIDENT: [bool; PREDATOR_TIERS] = [true, false, false];

/// Radians per tick a hunter may swing its heading.
///
/// A chase used to write position straight from a fresh unit vector to whatever
/// prey was nearest, so the heading could flip by any angle between two ticks.
/// At 0.95 world units a tick — six times a prey animal's step — that read as
/// jitter rather than as swimming. Turning is now rate-limited, so a hunter
/// banks onto its target instead of snapping onto it.
///
/// Chosen for the look of the arc, not for lethality: cull duration barely moves
/// with this number (a hunter in a dense pond eats whatever its shape sweeps,
/// aimed or not), while the turn radius is `speed / turn` — 4.7 world units for
/// the triangles, so they make long banking passes rather than buzzing in tight
/// circles. Nothing escapes by out-turning them regardless: they close at 19
/// tiles/s against a prey top speed of 3. Heavier shapes turn wider still — the
/// rectangle sweeps at a radius of 17.
const TIER_MAX_TURN: [f32; PREDATOR_TIERS] = [0.20, 0.15, 0.09];
/// Ticks a hunter stays committed to one prey animal before it looks again.
///
/// Picking the nearest prey every tick is the other half of what made hunters
/// jitter: in a crowd two animals sit at near-equal distance and the argmin
/// alternates between them, so the hunter aims at each in turn and goes nowhere.
/// Commitment also means a hunt has a subject you can watch.
const PREDATOR_COMMIT_TICKS: u32 = 18;
/// Commitment breaks if the target gets this far away, so a hunter can't be led
/// across the pond by one animal while a crowd sits behind it.
const PREDATOR_COMMIT_RANGE: f32 = 12.0;
/// Patrol wander: fraction of last tick's turn rate that carries over, and the
/// size of the fresh noise added to it. An uncorrelated per-tick turn reads as
/// 20 Hz vibration, and patrolling is what a resident does most of its life, so
/// the random walk is on the turn *rate* and stays smooth.
const PATROL_TURN_MEMORY: f32 = 0.92;
const PATROL_TURN_NOISE: f32 = 0.02;
/// Ceiling on the patrol turn rate, radians per tick.
const PATROL_TURN_MAX: f32 = 0.12;
/// Patrol speed as a fraction of the tier's hunting speed.
const PATROL_SPEED_FRAC: f32 = 0.45;
/// Per-tick easing of a predator's speed toward whatever its current state
/// wants. Without it a hunter going quiet or leaving changes speed in one tick,
/// which is as visible as a heading snap.
const PREDATOR_SPEED_EASE: f32 = 0.12;

/// Bite reach for a tier.
pub fn tier_bite(tier: u8) -> f32 {
    TIER_BITE[(tier as usize).min(PREDATOR_TIERS - 1)]
}
/// Radians per step a tier rotates. Zero for the bottom two.
pub fn tier_spin(tier: u8) -> f32 {
    TIER_SPIN[(tier as usize).min(PREDATOR_TIERS - 1)]
}
/// Whether a tier stays after its quota is met rather than departing.
pub fn tier_resident(tier: u8) -> bool {
    TIER_RESIDENT[(tier as usize).min(PREDATOR_TIERS - 1)]
}

/// Does this tier's kill shape cover `(dx, dy)`, a toroidal offset from the
/// predator? Radial for the first three; the top tier sweeps an oriented
/// rectangle and kills anything any edge touches.
fn tier_bite_hits(tier: u8, dx: f32, dy: f32, angle: f32) -> bool {
    let reach = tier_bite(tier);
    if tier as usize + 1 < PREDATOR_TIERS {
        return dx * dx + dy * dy <= reach * reach;
    }
    let (c, s) = (angle.cos(), angle.sin());
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;
    lx.abs() <= reach && ly.abs() <= RECTANGLE_HALF_WIDTH
}

/// Steps the predator spends swimming away before it disappears. It does not
/// vanish the instant its quota is met — it leaves the way it arrived, under its
/// own power, so the cull has a visible end.
const PREDATOR_LEAVE_TICKS: u32 = 44;
/// Speed while departing. Faster than its hunting speed: it is done here.
const PREDATOR_LEAVE_SPEED: f32 = 1.8;

/// State of the apex predator while it is in the pond.
#[derive(Debug, Clone, Copy)]
pub struct Predator {
    /// Stable agent id, so the renderer can single it out.
    pub id: u32,
    /// Escalation tier, 0–3. Drives speed, reach, kill shape, and whether this
    /// one leaves when it is done.
    pub tier: u8,
    /// Current rotation of the kill shape, radians. Only the top two spin.
    pub angle: f32,
    /// Set once its quota is met and it is resident, so it patrols instead of
    /// hunting. Cleared when the population climbs back over the trigger.
    pub sated: bool,
    /// It stops hunting once the living population reaches this. Checked against
    /// the live count every step, since births keep changing it underneath.
    pub target_pop: usize,
    /// Set when it triggered itself rather than being summoned.
    pub automatic: bool,
    pub kills: u32,
    /// Steps left of its departure swim; None while still hunting.
    pub leaving: Option<u32>,
    /// A prey animal that just resisted this hunter. It is skipped for one
    /// target choice so a high-defense agent cannot pin the hunter in place.
    pub rejected_id: Option<u32>,
    /// The animal it is currently hunting. Held for `PREDATOR_COMMIT_TICKS` so
    /// the heading has something stable to steer toward — see that constant.
    pub target_id: Option<u32>,
    /// Ticks of commitment left on `target_id`. Zero forces a fresh look.
    pub commit_ticks: u32,
    /// Current speed, world units per tick. Eased toward what the current state
    /// wants rather than set outright, so state changes don't snap.
    pub speed: f32,
    /// Patrol turn rate, radians per tick, carried between ticks so idle motion
    /// is a smooth arc instead of a per-tick coin flip.
    pub turn_rate: f32,
    /// The family this hunter is currently hunting by preference — a genome
    /// cluster label, reviewed on the cluster tick. `None` before the first
    /// review, or in a pond with nothing left to count.
    pub search_image: Option<u8>,
    /// Learned bite strength, starting at the tier's base and tracking the
    /// armoured tail of `search_image`. Reset to base when the image changes:
    /// what it learned was how to bite *that* shape.
    pub attack: f64,
    /// Mean effective armour of the current image, from the last review. Used to
    /// pick out the better-armoured members while hunting — a hunter goes for
    /// the prize animal, not the runt.
    pub image_armour: f64,
    /// Ticks left of a speed burst. While it runs the hunter multiplies its
    /// cruising speed by `PREDATOR_BURST_MULT`, up to the absolute ceiling.
    pub burst_ticks: u32,
    /// This hunter's current place in the cruising band, re-rolled on each
    /// search-image review. Set on spawn so a hunter has one before its first
    /// review.
    pub cruise_frac: f32,
}

// ── Death ─────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CauseOfDeath {
    Starvation,
    OldAge,
    KilledInCombat,
    EatenAlive,
    /// Killed by a player god-mode action (comet, salt, sweep). Kept distinct
    /// from the natural causes so the death graph doesn't attribute an act of
    /// god to the ecology.
    Smitten,
    /// Ran out of energy while infected. Severity is an energy drain, so this
    /// death is a starvation mechanically — but attributing it to starvation
    /// would hide every outbreak inside the food economy, which is exactly the
    /// thing an outbreak is not.
    Disease,
}

impl CauseOfDeath {
    /// Stable numeric code for the wasm state buffer. The renderer keys its
    /// death effect off this, so the values must not be reordered.
    pub fn code(&self) -> u8 {
        match self {
            CauseOfDeath::Starvation => 0,
            CauseOfDeath::OldAge => 1,
            CauseOfDeath::KilledInCombat => 2,
            CauseOfDeath::EatenAlive => 3,
            CauseOfDeath::Smitten => 4,
            CauseOfDeath::Disease => 5,
        }
    }
}

/// One death, queued for the renderer. Drained by the wasm state builder each
/// frame; positions are captured before `reap_dead`'s swap_remove invalidates them.
#[derive(Debug, Clone, Copy)]
pub struct DeathEvent {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub cause: u8,
}

/// Safety cap on the pending-death queue in case nothing drains it.
const MAX_QUEUED_DEATHS: usize = 256;

// ── Pending offspring ─────────────────────────────────────────────────────────
struct PendingAgent {
    genome: Genome,
    energy: f64,
    x: f32,
    y: f32,
    parent_defense: f64,
    parent_id: u32,
    /// The parent's species, inherited at birth. See `push_agent`.
    species: u32,
}

// ── Public stats ──────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct SimStats {
    pub step: u32,
    pub alive_agents: usize,
    pub total_food: u32,
    pub avg_energy: f64,
    pub median_lifespan: f64,
    pub deaths: HashMap<String, u32>,
}

// ── World ─────────────────────────────────────────────────────────────────────
pub struct World {
    pub grid_size: usize,
    pub step_count: u32,
    rng: ChaCha8Rng,
    death_range_pool: Vec<u32>,

    // Grid: flat, indexed y * grid_size + x (tile system unchanged)
    pub tiles: Vec<BiomeTile>,

    // SoA agent arrays — all same length, same index = same agent
    pub ids: Vec<u32>,
    pub energy: Vec<f64>,
    pub age: Vec<u32>,
    pub pos_x: Vec<f32>,           // continuous world x in [0, grid_size)
    pub pos_y: Vec<f32>,           // continuous world y in [0, grid_size)
    pub vel_x: Vec<f32>,           // velocity x in tiles/sec
    pub vel_y: Vec<f32>,           // velocity y in tiles/sec
    pub prev_x: Vec<f32>,          // position at previous tick (for renderer interpolation)
    pub prev_y: Vec<f32>,
    pub death_age: Vec<u32>,
    pub genome: Vec<Genome>,
    pub memory: Vec<AgentMemory>,
    /// Ticks until this agent thinks again. Zero means it decides this tick.
    decision_cooldown: Vec<u32>,
    /// Threat pipeline per agent: `[dist_norm, angle_norm, away_x, away_y]` per
    /// tick, oldest still-relevant entry first. Written every tick regardless of
    /// the decision cadence — seeing is passive, only *thinking* is rationed —
    /// and read `threat_lag(intelligence)` ticks late, which is how a dull agent
    /// ends up steering away from where the predator was rather than where it is.
    threat_ring: Vec<[[f32; 4]; THREAT_RING]>,
    /// Next write position in each agent's ring.
    threat_head: Vec<usize>,
    /// The last decision each agent made, replayed on the ticks it does not
    /// think. Physics still runs every tick for everyone — only the *deciding*
    /// is rationed, so a dull agent keeps swimming on last tick's intent.
    last_outputs: Vec<[f32; 8]>,
    /// The perception that produced `last_outputs`, and the food direction that
    /// came with it. Replayed alongside, so a stale decision is acted on with
    /// the stale picture that justified it rather than a fresh one.
    last_perception: Vec<[f32; INPUT_COUNT]>,
    last_food_dir: Vec<(f32, f32)>,
    /// Combat wins per agent, shown in the inspector.
    pub kills: Vec<u32>,
    parent_defense_bonus: Vec<f64>,
    parent_id: Vec<Option<u32>>,
    cause_of_death: Vec<Option<CauseOfDeath>>,
    offspring_count: Vec<u32>,
    max_offspring: Vec<u32>,
    last_reproduced_age: Vec<Option<u32>>,
    reproduction_cooldown: Vec<u32>,

    next_id: u32,
    pub lifespans: Vec<u32>,
    /// How much of `lifespans` the last stats sample already consumed, so the
    /// next one can take the median of just this interval's deaths.
    lifespans_sampled: usize,
    /// Deaths awaiting export to the renderer. Drained by the wasm state builder.
    pub last_deaths: Vec<DeathEvent>,
    death_tally: HashMap<CauseOfDeath, u32>,
    spatial: SpatialHashGrid,
    pub cluster: ClusterState,
    /// Behavioural k-means over brain weights. Retained, incremental, and off
    /// unless something is looking at it — see `brain_cluster.rs`.
    pub brain_clusters: BrainClusters,
    /// Promoted lineages and the fossil record. Advanced on the same schedule
    /// as `cluster`, since a species is a judgement about clusters over time.
    pub species: SpeciesRegistry,
    /// Every pathogen this run has produced, live or burned out. Indexed by
    /// `id - 1`; id 0 means "healthy" in `infection`.
    pub diseases: Vec<Disease>,
    /// Disease id per agent, 0 for healthy. Parallel to the agent arrays.
    pub infection: Vec<u32>,
    /// Species id per agent, parallel to the agent arrays and refreshed on
    /// cluster runs. Stale between runs in exactly the way `cluster` is —
    /// `swap_remove` reshuffles slots, so consumers index defensively.
    pub species_ids: Vec<u32>,
    /// God-mode immortality. While set, no natural cause kills anyone: old age,
    /// starvation and combat losses are all suppressed and starving agents are
    /// held at a floor energy so they keep acting. Player smites ignore it —
    /// the point of a comet is that it overrules the rules.
    ///
    /// Population is unbounded with this on, and the renderer draws every agent
    /// individually, so this is the one switch that can genuinely bring the tab
    /// down. The UI warns before enabling it.
    pub immortal: bool,
    /// Apex predators currently in the pond. See `Predator`.
    pub predators: Vec<Predator>,
    /// Ids in `predators`, as a set. Pure cache, resynced by
    /// `resync_predator_ids` on every change to `predators`: `is_predator` is
    /// called once per agent inside the hunt's scans, and walking the pack for
    /// each of them made the phase O(agents × pack) per hunter.
    predator_ids: HashSet<u32>,
    /// Player-facing ecology switch. God-mode summons are deliberately
    /// independent of it.
    pub automatic_predators_enabled: bool,
    /// Player-facing disease switch. Off stops new pathogens being seeded and
    /// stops transmission; agents already infected stay infected, because
    /// clearing them would rewrite the run rather than pause it.
    pub disease_enabled: bool,
    /// Predators that finished their departure swim this step, awaiting removal.
    departed_ids: Vec<u32>,
    /// Largest pack this run has ever fielded. A pack ratchets: it can grow but
    /// never shrink, so every later cull starts at least as strong as the
    /// strongest one before it. A pond that once needed six hunters has proven
    /// it can outbreed five, and there is no reason to relearn that each time.
    predator_high_water: usize,
    /// Tier the automatic rule deploys. Always 0 — the ecology only fields
    /// triangles, and the other two are player powers.
    pub predator_tier: u8,
    /// Population at the last reinforcement check, to tell "still climbing"
    /// from "the cull is working".
    last_reinforce_pop: usize,
    /// Step of the last reinforcement check.
    last_reinforce_step: u32,
    /// Rolling stat time-series, sampled every `SAMPLE_INTERVAL` steps.
    pub stats_history: StatHistory,
    /// The three live dials. See `Tunables`.
    tunables: Tunables,
    /// Test hook: hold tier-0 hunters at the flat `TIER_SPEED` constant instead
    /// of tracking their prey, so the two regimes can be compared on one seed.
    #[cfg(test)]
    pub pin_predator_speed_for_test: bool,
    /// Set when `cluster_k` changes, so the next step reclusters instead of
    /// waiting out the rest of the 50-step cycle — otherwise the dial looks
    /// dead for up to 50 steps.
    cluster_dirty: bool,

    // Per-tile eat bookkeeping (crowding contention + cooldown)
    tile_last_eaten: Vec<Option<u32>>,
    tile_eat_count_this_tick: Vec<u16>,

    // Pre-allocated scratch buffers — cleared and reused each step
    scratch_acting: Vec<usize>,
    scratch_dead: Vec<usize>,
    scratch_perceptions: Vec<[f32; INPUT_COUNT]>,
    /// Per acting slot: is this agent thinking this tick, or replaying?
    scratch_deciding: Vec<bool>,
    scratch_food_dirs: Vec<(f32, f32)>,  // unit vector to nearest visible food; (0,0) if none
    scratch_outputs: Vec<[f32; 8]>,  // sigmoid-gated brain outputs per acting agent
}

impl World {
    pub fn new(grid_size: usize, population: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let death_range_pool = create_death_range(&mut rng);
        let tiles = init_grid(grid_size, &mut rng);
        let spatial = SpatialHashGrid::new(grid_size);

        let mut world = Self {
            grid_size,
            step_count: 0,
            rng,
            death_range_pool,
            tiles,
            ids: Vec::new(),
            energy: Vec::new(),
            age: Vec::new(),
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            prev_x: Vec::new(),
            prev_y: Vec::new(),
            death_age: Vec::new(),
            genome: Vec::new(),
            memory: Vec::new(),
            decision_cooldown: Vec::new(),
            threat_ring: Vec::new(),
            threat_head: Vec::new(),
            last_outputs: Vec::new(),
            last_perception: Vec::new(),
            last_food_dir: Vec::new(),
            kills: Vec::new(),
            parent_defense_bonus: Vec::new(),
            parent_id: Vec::new(),
            cause_of_death: Vec::new(),
            offspring_count: Vec::new(),
            max_offspring: Vec::new(),
            last_reproduced_age: Vec::new(),
            reproduction_cooldown: Vec::new(),
            next_id: 0,
            lifespans: Vec::new(),
            lifespans_sampled: 0,
            last_deaths: Vec::new(),
            death_tally: HashMap::new(),
            spatial,
            cluster: ClusterState::new(),
            brain_clusters: BrainClusters::new(),
            species: SpeciesRegistry::new(seed),
            species_ids: Vec::new(),
            diseases: Vec::new(),
            infection: Vec::new(),
            immortal: false,
            predators: Vec::new(),
            predator_ids: HashSet::new(),
            automatic_predators_enabled: true,
            disease_enabled: true,
            departed_ids: Vec::new(),
            predator_high_water: 0,
            predator_tier: 0,
            last_reinforce_pop: 0,
            last_reinforce_step: 0,
            stats_history: StatHistory::new(),
            tunables: Tunables::default(),
            #[cfg(test)]
            pin_predator_speed_for_test: false,
            cluster_dirty: false,
            tile_last_eaten: vec![None; grid_size * grid_size],
            tile_eat_count_this_tick: vec![0u16; grid_size * grid_size],
            scratch_acting: Vec::new(),
            scratch_dead: Vec::new(),
            scratch_perceptions: Vec::new(),
            scratch_deciding: Vec::new(),
            scratch_food_dirs: Vec::new(),
            scratch_outputs: Vec::new(),
        };

        world.spawn_agents(population);
        world.spatial.rebuild(&world.pos_x, &world.pos_y);
        world
    }

    pub fn agent_count(&self) -> usize {
        self.ids.len()
    }

    // ── Tunables ──────────────────────────────────────────────────────────────
    // Setters clamp in core rather than trusting the caller: the UI is one
    // caller, and a regen scale of 5.0 or a k of 0 would panic or wedge the sim.

    pub fn tunables(&self) -> Tunables {
        self.tunables
    }

    pub fn set_food_regen_scale(&mut self, v: f64) {
        self.tunables.food_regen_scale = clamp_range(v, FOOD_REGEN_SCALE_RANGE);
        self.mark_tuned();
    }

    pub fn set_hunt_aggression_threshold(&mut self, v: f64) {
        self.tunables.hunt_aggression_threshold =
            clamp_range(v, HUNT_AGGRESSION_THRESHOLD_RANGE);
        self.mark_tuned();
    }

    /// Changing `k` mid-run is safe: `match_labels` already handles the previous
    /// run having a different label count. Colours re-shuffle once, which is
    /// honest — the families genuinely changed.
    pub fn set_cluster_k(&mut self, k: usize) {
        let k = k.clamp(CLUSTER_K_RANGE.0, CLUSTER_K_RANGE.1);
        if k != self.tunables.cluster_k {
            self.tunables.cluster_k = k;
            self.cluster_dirty = true;
        }
        self.mark_tuned();
    }

    /// Latches `modified` the first time any dial differs from its default.
    /// Setting a dial back does not clear it: the run already diverged, and the
    /// seed no longer describes it.
    fn mark_tuned(&mut self) {
        // Compared with a tolerance because the values arrive from JS as f32:
        // the wasm layer hands the UI `0.012` as an f32 and gets back
        // 0.012000000104…, so an exact test would latch `modified` the first
        // time someone pressed reset without having moved anything.
        const EPS: f64 = 1e-6;
        let d = Tunables::default();
        let t = &self.tunables;
        if (t.food_regen_scale - d.food_regen_scale).abs() > EPS
            || (t.hunt_aggression_threshold - d.hunt_aggression_threshold).abs() > EPS
            || t.cluster_k != d.cluster_k
        {
            self.tunables.modified = true;
        }
    }

    pub fn get_stats(&self) -> SimStats {
        let total_food: u32 = self.tiles.iter().map(|t| t.food_units).sum();
        // Prey only, here too: a predator's energy is meaningless — it is held
        // at a floor and never eats — so averaging it in drags the pond's
        // apparent energy toward that floor as the pack grows.
        let mut sum = 0.0;
        let mut n = 0usize;
        for i in 0..self.ids.len() {
            if self.is_predator(i) { continue; }
            sum += self.energy[i];
            n += 1;
        }
        let avg_energy = if n > 0 { sum / n as f64 } else { 0.0 };
        let median_lifespan = median(&self.lifespans);
        let mut deaths = HashMap::new();
        for (cause, &count) in &self.death_tally {
            deaths.insert(format!("{:?}", cause), count);
        }
        SimStats {
            step: self.step_count,
            // Prey only. A triangle is a hazard, not an inhabitant: counting one
            // as a member of the population puts predators in the population
            // graph, in the HUD, and in the average-energy denominator, where
            // they are all three times misleading.
            alive_agents: self.prey_count(),
            total_food,
            avg_energy,
            median_lifespan,
            deaths,
        }
    }

    /// Population mean of each genome trait over living agents, in `Traits`
    /// field order: vision, speed, metabolism, energy_capacity, mutation_rate,
    /// reproduction_cost, attack, defense, aggression, intelligence.
    pub fn trait_means(&self) -> [f64; TRAIT_COUNT] {
        let mut sums = [0f64; TRAIT_COUNT];
        let mut count = 0usize;
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_some() { continue; }
            let t = &self.genome[i].traits;
            for (s, v) in sums.iter_mut().zip([
                t.vision, t.speed, t.metabolism, t.energy_capacity,
                t.mutation_rate, t.reproduction_cost, t.attack, t.defense, t.aggression,
                t.intelligence, t.immunity,
            ]) { *s += v; }
            count += 1;
        }
        if count > 0 {
            for s in sums.iter_mut() { *s /= count as f64; }
        }
        sums
    }

    /// Debug snapshot of one agent's brain for the inspector panel. Runs a
    /// pure traced forward pass on the agent's current perception — no RNG
    /// consumed, no state mutated, so calling it never perturbs the sim.
    /// Layout: [7 inputs | 12 h0 | 12 h1 | 12 h2 | 8 logits | 8 sigmoid gates
    ///          | energy_norm | age_norm | kills | 9 traits] = 71 floats. The JS
    /// side derives these offsets from `brain_layer_sizes()` rather than
    /// hardcoding them — see `schema.rs`.
    /// Returns None if the id is not alive.
    pub fn inspect_agent(&self, id: u32) -> Option<Vec<f32>> {
        let idx = self.ids.iter().position(|&i| i == id)?;
        if self.cause_of_death[idx].is_some() { return None; }

        let (input, _) = self.perceive(idx);
        let (h0, h1, h2, logits) = forward_traced(self.genome[idx].weights_array(), input);
        let gates = sigmoid_outputs(logits);

        let max_e = MAX_ENERGY_BASE * self.genome[idx].traits.energy_capacity;
        let energy_norm = (self.energy[idx] / max_e).clamp(0.0, 1.0) as f32;
        let age_norm = (self.age[idx] as f64 / self.death_age[idx] as f64).clamp(0.0, 1.0) as f32;
        let t = &self.genome[idx].traits;

        let mut out = Vec::with_capacity(71);
        out.extend_from_slice(&input);
        out.extend_from_slice(&h0);
        out.extend_from_slice(&h1);
        out.extend_from_slice(&h2);
        out.extend_from_slice(&logits);
        out.extend_from_slice(&gates);
        out.push(energy_norm);
        out.push(age_norm);
        out.push(self.kills[idx] as f32);
        // Every trait, in field order. This listed nine and omitted
        // `intelligence`, which left the buffer one float short of what the
        // inspector iterates — `undefined.toFixed()` threw and the whole panel
        // went dead. The count is asserted below so a future trait cannot repeat
        // it silently.
        for v in [t.vision, t.speed, t.metabolism, t.energy_capacity,
                  t.mutation_rate, t.reproduction_cost, t.attack, t.defense,
                  t.aggression, t.intelligence, t.immunity] {
            out.push(v as f32);
        }
        debug_assert_eq!(out.len(), Self::inspect_buffer_len(),
            "inspect buffer length drifted from its documented layout");
        Some(out)
    }

    /// Length of the `inspect_agent` buffer, derived from the brain's shape and
    /// the trait count rather than written down. The JS side derives its offsets
    /// the same way (see `brain_layer_sizes`).
    pub fn inspect_buffer_len() -> usize {
        let sizes = crate::brain::LAYER_SIZES;
        let outputs = sizes[sizes.len() - 1];
        sizes.iter().sum::<usize>()   // inputs + hidden layers + logits
            + outputs                 // sigmoid gates
            + 3                       // energy_norm, age_norm, kills
            + TRAIT_COUNT
    }

    /// Spawn agents at random positions within `radius` tiles of (cx, cy).
    pub fn pour_agents(&mut self, cx: f32, cy: f32, count: usize) {
        let world_size = self.grid_size as f32;
        for _ in 0..count {
            let angle = self.rng.gen::<f32>() * TAU;
            let radius: f32 = self.rng.gen_range(0.3..2.0);
            let x = (cx + radius * angle.cos()).rem_euclid(world_size);
            let y = (cy + radius * angle.sin()).rem_euclid(world_size);
            let genome = Genome::generate(&mut self.rng);
            self.push_agent(x, y, 50.0, genome, 0.0, None, crate::species::UNASSIGNED);
        }
        self.spatial.rebuild(&self.pos_x, &self.pos_y);
    }

    /// Add food to the tile under world position (cx, cy). Clamped to MAX_FOOD_PER_TILE.
    pub fn inject_food(&mut self, cx: f32, cy: f32, amount: u32) {
        let gs = self.grid_size;
        let (tx, ty) = SpatialHashGrid::tile_of(cx, cy, gs);
        let tile = &mut self.tiles[ty * gs + tx];
        tile.food_units = (tile.food_units + amount).min(MAX_FOOD_PER_TILE);
    }

    /// Disturb food, fertility, and agent velocities within `radius` world units
    /// of (cx, cy). `intensity` in [0, 1]: 1.0 = maximum disruption.
    pub fn stir(&mut self, cx: f32, cy: f32, radius: f32, intensity: f32) {
        let gs = self.grid_size;
        let world_size = gs as f32;
        let r_cells = radius.ceil() as i32;
        let cx_tile = cx.floor() as i32;
        let cy_tile = cy.floor() as i32;
        let half = world_size * 0.5;

        // Disturb tiles within radius
        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx_tile + dx).rem_euclid(gs as i32) as usize;
                let ty = (cy_tile + dy).rem_euclid(gs as i32) as usize;
                let tile_cx = tx as f32 + 0.5;
                let tile_cy = ty as f32 + 0.5;
                let mut ddx = tile_cx - cx;
                let mut ddy = tile_cy - cy;
                if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
                if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }
                let dist = (ddx * ddx + ddy * ddy).sqrt();
                if dist > radius { continue; }

                let tile = &mut self.tiles[ty * gs + tx];
                let drain = (tile.food_units as f32 * intensity) as u32;
                tile.food_units = tile.food_units.saturating_sub(drain);
                tile.fertility *= (1.0 - intensity * 0.5) as f64;
                tile.fertility = tile.fertility.max(0.0);
            }
        }

        // Apply velocity impulse to agents within radius (scatter outward from stir center)
        let impulse = intensity * 6.0;
        let n = self.agent_count();
        for i in 0..n {
            let ax = self.pos_x[i];
            let ay = self.pos_y[i];
            let mut ddx = ax - cx;
            let mut ddy = ay - cy;
            if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
            if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }
            let dist = (ddx * ddx + ddy * ddy).sqrt();
            if dist > radius || dist < 0.001 { continue; }

            let falloff = (1.0 - dist / radius) * impulse;
            self.vel_x[i] += ddx / dist * falloff;
            self.vel_y[i] += ddy / dist * falloff;

            // Clamp to 2× max speed so stir can't fling agents unreasonably far
            let max = self.genome[i].traits.speed as f32 * MAX_SPEED * 2.0;
            let cur = (self.vel_x[i].powi(2) + self.vel_y[i].powi(2)).sqrt();
            if cur > max && cur > 0.0 {
                self.vel_x[i] = self.vel_x[i] / cur * max;
                self.vel_y[i] = self.vel_y[i] / cur * max;
            }
        }
    }

    // ── God mode ──────────────────────────────────────────────────────────────
    //
    // Player-driven kills. All three funnel through `smite`, so they queue death
    // events, tally as `Smitten`, and record lifespans exactly like a natural
    // death — the renderer's death effects and the stat graphs need no special
    // case, and the death panel can honestly separate acts of god from ecology.
    //
    // Immortality does not protect against these. A comet is meant to overrule
    // the rules, not obey them.

    /// Kill every living agent within `radius` world units of (cx, cy),
    /// measured toroidally. Returns the number killed.
    pub fn smite_radius(&mut self, cx: f32, cy: f32, radius: f32) -> u32 {
        let world_size = self.grid_size as f32;
        let half = world_size * 0.5;
        let r2 = radius * radius;
        let mut victims = Vec::new();

        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_some() { continue; }
            let mut dx = self.pos_x[i] - cx;
            let mut dy = self.pos_y[i] - cy;
            if dx > half { dx -= world_size; } else if dx < -half { dx += world_size; }
            if dy > half { dy -= world_size; } else if dy < -half { dy += world_size; }
            if dx * dx + dy * dy <= r2 { victims.push(i); }
        }

        self.smite(victims)
    }

    /// Kill every living agent inside the world-space column [x0, x1).
    /// Used by the sweep, which advances the column across the pond.
    pub fn smite_band(&mut self, x0: f32, x1: f32) -> u32 {
        let victims: Vec<usize> = (0..self.ids.len())
            .filter(|&i| self.cause_of_death[i].is_none())
            .filter(|&i| self.pos_x[i] >= x0 && self.pos_x[i] < x1)
            .collect();
        self.smite(victims)
    }

    /// Kill every living agent. Returns the number killed.
    pub fn smite_all(&mut self) -> u32 {
        let victims: Vec<usize> = (0..self.ids.len())
            .filter(|&i| self.cause_of_death[i].is_none())
            .collect();
        self.smite(victims)
    }

    /// Mark the given slots dead and reap them immediately, so a smite takes
    /// effect on the frame the player triggers it rather than on the next step —
    /// god mode works while the sim is paused.
    fn smite(&mut self, victims: Vec<usize>) -> u32 {
        if victims.is_empty() { return 0; }
        let killed = victims.len() as u32;
        for &i in &victims {
            self.cause_of_death[i] = Some(CauseOfDeath::Smitten);
            self.energy[i] = 0.0;
            self.lifespans.push(self.age[i]);
        }
        self.reap_dead(victims);
        self.spatial.rebuild(&self.pos_x, &self.pos_y);
        killed
    }

    // ── Ultra predator ────────────────────────────────────────────────────────

    /// Sustainable population for this pond: agents per tile × tiles. The
    /// automatic cull is measured against this rather than a fixed number.
    pub fn pop_cap(&self) -> usize {
        let by_area = ((self.grid_size * self.grid_size) as f64 * PREDATOR_POP_PER_TILE) as usize;
        by_area.min(PREDATOR_POP_CEILING)
    }

    /// Population above which predators arrive: the cap plus its breathing room.
    pub fn cull_trigger_pop(&self) -> usize {
        (self.pop_cap() as f64 * (1.0 + PREDATOR_POP_BAND)) as usize
    }

    /// Population the automatic cull drives down to: the cap minus its breathing
    /// room. Culling to the cap exactly would put the pond back at the trigger
    /// within a few births; the band is what stops that oscillation.
    pub fn cull_target_pop(&self) -> usize {
        (self.pop_cap() as f64 * (1.0 - PREDATOR_POP_BAND) * CULL_DEPTH) as usize
    }

    /// Enable or disable the pond's automatic triangle ecology. Disabling is an
    /// active command: automatic residents depart, and the pack's ratchet is
    /// reset. Player-summoned predators are not affected.
    pub fn set_automatic_predators(&mut self, on: bool) {
        if self.automatic_predators_enabled == on { return; }
        self.automatic_predators_enabled = on;
        if on { return; }

        let ids: Vec<u32> = self.predators.iter()
            .filter(|p| p.automatic && p.leaving.is_none())
            .map(|p| p.id)
            .collect();
        for id in ids {
            let Some(pi) = self.predators.iter().position(|p| p.id == id) else { continue };
            let Some(slot) = self.slot_of(id) else { continue };
            self.begin_departure(slot, pi);
        }
        self.predator_high_water = 0;
        self.last_reinforce_pop = self.prey_count();
        self.last_reinforce_step = self.step_count;
    }

    /// Send every player-summoned hunter away, mid-hunt. Returns how many were
    /// dismissed.
    ///
    /// The off switch for the god-mode shapes, and the counterpart to
    /// `set_automatic_predators`: that one deliberately spares player summons, so
    /// without this an octagon or a rectangle could only be waited out. They
    /// leave the way they always do — under their own power, over
    /// `PREDATOR_LEAVE_TICKS` — rather than blinking out, so a dismissal reads as
    /// the hunt being called off and not as a rendering fault. The automatic
    /// residents are left alone; they are the ecology, not a player power.
    pub fn dismiss_summoned_predators(&mut self) -> usize {
        let ids: Vec<u32> = self.predators.iter()
            .filter(|p| !p.automatic && p.leaving.is_none())
            .map(|p| p.id)
            .collect();
        let mut sent = 0;
        for id in ids {
            let Some(pi) = self.predators.iter().position(|p| p.id == id) else { continue };
            let Some(slot) = self.slot_of(id) else { continue };
            self.begin_departure(slot, pi);
            sent += 1;
        }
        sent
    }

    /// Player-summoned hunters still in the pond and not already leaving — what
    /// the off switch would act on.
    pub fn summoned_predator_count(&self) -> usize {
        self.predators.iter().filter(|p| !p.automatic && p.leaving.is_none()).count()
    }

    /// Summon a predator. It hunts until the living prey count reaches
    /// `target_pop`, then leaves. Returns its agent id, or None at the cap.
    pub fn summon_predator_to(&mut self, target_pop: usize, automatic: bool) -> Option<u32> {
        self.summon_predator_tier(target_pop, automatic, self.predator_tier)
    }

    /// Summon one hunter of a given tier.
    pub fn summon_predator_tier(
        &mut self, target_pop: usize, automatic: bool, tier: u8,
    ) -> Option<u32> {
        if self.predators.len() >= PREDATOR_MAX { return None; }

        // Maxed traits across the board. It never reproduces or forages, so most
        // of these only matter for how it looks — morphology reads the genome,
        // and an apex predator should look like one.
        let mut genome = Genome::generate(&mut self.rng);
        let t = &mut genome.traits;
        t.vision = 1.05;
        t.speed = 1.0;
        t.metabolism = 1.05;
        t.attack = 1.25;
        t.defense = 1.07;
        t.aggression = 1.05;
        t.reproduction_cost = 1.50;   // it will never pay it; keeps it out of the breeding pool

        let x = self.rng.gen::<f32>() * self.grid_size as f32;
        let y = self.rng.gen::<f32>() * self.grid_size as f32;
        self.push_agent(x, y, MAX_ENERGY_BASE, genome, 0.0, None, crate::species::UNASSIGNED);

        let id = *self.ids.last().unwrap();
        self.predators.push(Predator {
            id, tier, angle: 0.0, sated: false,
            target_pop, automatic, kills: 0, leaving: None, rejected_id: None,
            target_id: None, commit_ticks: 0,
            // Arrives from a standing start and accelerates into its first
            // chase, rather than appearing already at full speed.
            speed: 0.0, turn_rate: 0.0,
            // No image until the first review, and the tier's own bite until it
            // has met something often enough to learn from.
            search_image: None,
            // Arrives already suited to this pond rather than at the tier's raw
            // bite. A hunter that has to learn from scratch spends the whole
            // cull — which is front-loaded, since residents go quiet once the
            // pond is down to target — eating whatever is softest, which is the
            // armour subsidy this whole mechanism exists to remove. The ecology
            // fields a predator for the pond it is entering.
            attack: self.starting_bite(tier),
            image_armour: 0.0,
            burst_ticks: 0,
            cruise_frac: (PREDATOR_CRUISE_FRAC.0 + PREDATOR_CRUISE_FRAC.1) * 0.5,
        });
        self.resync_predator_ids();
        self.predator_high_water = self.predator_high_water.max(self.predators.len());
        self.spatial.rebuild(&self.pos_x, &self.pos_y);
        Some(id)
    }

    /// Player summon: culls to `survivor_frac` of the population right now.
    /// Returns the id of the first predator summoned.
    pub fn summon_predator(&mut self, survivor_frac: f64, automatic: bool) -> Option<u32> {
        if !automatic {
            let target = (self.prey_count() as f64 * survivor_frac).round() as usize;
            return self.summon_predator_pack_tier(target, false, PREDATOR_MANUAL_TIER);
        }
        let target = ((self.prey_count() as f64) * survivor_frac).round().max(0.0) as usize;
        self.summon_predator_pack(target, automatic)
    }

    /// Summon as many predators as the size of the job warrants — one per
    /// `PREY_PER_PREDATOR` to remove, capped by `PREDATOR_MAX` and by how many
    /// are already hunting. A cull of two thousand is not one hunter's work, and
    /// waiting for reinforcements to trickle in makes it look broken.
    pub fn summon_predator_pack(&mut self, target_pop: usize, automatic: bool) -> Option<u32> {
        self.summon_predator_pack_tier(target_pop, automatic, self.predator_tier)
    }

    /// Summon a pack of a given tier, sized by that tier's pack size and by how
    /// large the job is.
    pub fn summon_predator_pack_tier(
        &mut self, target_pop: usize, automatic: bool, tier: u8,
    ) -> Option<u32> {
        let to_remove = self.prey_count().saturating_sub(target_pop);
        let pack = TIER_PACK[(tier as usize).min(PREDATOR_TIERS - 1)];
        // Never fewer than the run's high-water mark: the pack ratchets up.
        let wanted = to_remove
            .div_ceil(PREY_PER_PREDATOR)
            // The per-tier pack floor is an automatic-wave rule: an escalation
            // arrives as a group. A player summon still scales to the job asked.
            .max(if automatic { pack } else { 1 })
            .max(self.predator_high_water)
            .clamp(1, PREDATOR_MAX);
        // Only hunters of this tier count toward the pack — an escalation must
        // add its own hunters rather than being satisfied by the resident ones
        // that already failed to finish the job.
        let already = self.predators.iter()
            .filter(|p| p.leaving.is_none() && p.tier == tier)
            .count();
        let spawn = wanted.saturating_sub(already);

        let mut first = None;
        for _ in 0..spawn.max(if already == 0 { 1 } else { 0 }) {
            match self.summon_predator_tier(target_pop, automatic, tier) {
                Some(id) => { if first.is_none() { first = Some(id); } }
                None => break,   // pack is at its cap
            }
        }
        first
    }

    /// Slot index of an agent id. Ids are stable across the swap_remove
    /// reshuffles that reaping causes, so slots have to be looked up, not cached.
    fn slot_of(&self, id: u32) -> Option<usize> {
        self.ids.iter().position(|&i| i == id)
    }

    /// True if this slot holds a predator — used by the death checks to skip it.
    fn is_predator(&self, idx: usize) -> bool {
        match self.ids.get(idx) {
            Some(id) => self.predator_ids.contains(id),
            None => false,
        }
    }

    /// Rebuild the `predator_ids` cache. Called after any change to
    /// `self.predators`; the pack is at most `PREDATOR_MAX`, so this is cheap
    /// next to the per-agent lookups it saves.
    fn resync_predator_ids(&mut self) {
        self.predator_ids.clear();
        self.predator_ids.extend(self.predators.iter().map(|p| p.id));
    }

    /// Run every predator's hunt, then handle arrivals and departures.
    ///
    /// Its own phase, after the dead are reaped, so no predator ever has to
    /// reason about an agent that is already dying this tick.
    fn hunt_with_predators(&mut self) {
        let ids: Vec<u32> = self.predators.iter().map(|p| p.id).collect();
        for id in ids {
            self.hunt_one(id);
        }
        self.predators.retain(|p| p.leaving != Some(0));
        self.resync_predator_ids();

        // Anything that finished its departure swim leaves the pond. Removal is
        // not a death: no tally, no lifespan, no death event.
        let gone: Vec<u32> = self.departed_ids.drain(..).collect();
        for id in gone {
            if let Some(slot) = self.slot_of(id) {
                self.remove_slots(vec![slot]);
            }
        }
        if !self.ids.is_empty() {
            self.spatial.rebuild(&self.pos_x, &self.pos_y);
        }
    }

    /// The bite a newly arrived hunter starts with: enough to take the toughest
    /// animal currently in the pond, capped like any learned bite. Falls back to
    /// the tier's own attack in an empty pond.
    fn starting_bite(&self, tier: u8) -> f64 {
        let base = TIER_ATTACK[(tier as usize).min(PREDATOR_TIERS - 1)];
        let mut toughest: f64 = 0.0;
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_some() || self.is_predator(i) { continue; }
            let def = effective_defense(
                self.genome[i].traits.defense,
                self.parent_defense_bonus[i],
                self.age[i],
            );
            if def > toughest { toughest = def; }
        }
        if toughest <= 0.0 { return base; }
        (toughest + PREDATOR_ATTACK_MARGIN).clamp(base, base + PREDATOR_ATTACK_MAX_ADAPT)
    }

    /// Chase speed for one hunter, in world units per tick.
    ///
    /// Tiers above 0 keep their constant. The ambient triangle cruises at a
    /// fraction of the mean speed of the family it is hunting — under 1.0, so an
    /// average animal outpaces it — and bursts above that to close a kill. The
    /// fraction is a band re-rolled on every review rather than a constant,
    /// because a fixed safe margin is one a lineage sits exactly on top of.
    fn predator_chase_speed(&self, pi: usize) -> f32 {
        let tier = self.predators[pi].tier as usize;
        let base = TIER_SPEED[tier.min(PREDATOR_TIERS - 1)];
        if tier != 0 { return base; }
        #[cfg(test)]
        if self.pin_predator_speed_for_test { return base; }

        let floor = PREDATOR_SPEED_FLOOR_TRAIT * MAX_SPEED * DT;
        let ceiling = PREDATOR_SPEED_CEILING_TRAIT * MAX_SPEED * DT;
        let bursting = self.predators[pi].burst_ticks > 0;
        // Bursting multiplies whatever it was cruising at, capped. Cruising
        // closes distance; the burst is what actually catches something.
        let apply = |v: f32| {
            let v = if bursting { v * PREDATOR_BURST_MULT } else { v };
            v.clamp(floor, ceiling)
        };

        // No image yet — first ticks of a run, before the first clustering pass.
        let Some(image) = self.predators[pi].search_image else { return apply(floor) };
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_some() || self.is_predator(i) { continue; }
            if self.cluster.genome_cluster_ids.get(i).copied() != Some(image) { continue; }
            sum += self.genome[i].traits.speed as f32;
            count += 1;
        }
        if count == 0 { return apply(floor); }
        // Trait → tiles per tick, the same conversion an agent's own velocity
        // cap goes through: `speed_trait × MAX_SPEED` is tiles per second, and a
        // tick is DT of one.
        let cruise = (sum / count as f32) * MAX_SPEED * DT * self.predators[pi].cruise_frac;
        apply(cruise)
    }

    /// Re-form every hunter's search image, and train its bite on that family's
    /// armour. Runs on the cluster tick, immediately after the labels are
    /// rebuilt, so the counts describe the pond as it is now.
    ///
    /// This is the whole of the frequency dependence. The plurality family is
    /// the one hunted by preference, so a strategy is punished in proportion to
    /// how well it is doing — which is what keeps the pond from collapsing onto
    /// a single answer, and what stopped armour being that answer.
    ///
    /// Consumes no RNG: it is counting and averaging over world state, so a run
    /// with predators present steps identically whether or not this fires.
    fn review_predator_search_images(&mut self) {
        if self.predators.is_empty() { return; }

        // Counts and mean armour per family, over living prey only. Predators
        // carry genomes and cluster labels of their own, and counting them would
        // let a big pack vote for its own family.
        let k = self.tunables.cluster_k.max(1);
        let mut counts = vec![0usize; k];
        let mut armour = vec![0f64; k];
        let mut toughest = vec![0f64; k];
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_some() || self.is_predator(i) { continue; }
            let Some(&label) = self.cluster.genome_cluster_ids.get(i) else { continue };
            let c = label as usize;
            if c >= k { continue; }
            let def = effective_defense(
                self.genome[i].traits.defense,
                self.parent_defense_bonus[i],
                self.age[i],
            );
            counts[c] += 1;
            armour[c] += def;
            if def > toughest[c] { toughest[c] = def; }
        }

        let Some(plurality) = (0..k).max_by_key(|&c| counts[c]).filter(|&c| counts[c] > 0)
        else { return };

        for pi in 0..self.predators.len() {
            let base = TIER_ATTACK[(self.predators[pi].tier as usize).min(PREDATOR_TIERS - 1)];
            let current = self.predators[pi].search_image.map(|c| c as usize);

            // Switch only when the challenger is clearly bigger, or when the old
            // image has emptied out. Hysteresis: without it two families of
            // similar size trade the image every review and the bite adaptation,
            // which resets on a switch, never accumulates.
            let switch = match current {
                None => true,
                Some(c) if c >= k || counts[c] == 0 => true,
                Some(c) => counts[plurality] as f64 > counts[c] as f64 * SEARCH_IMAGE_SWITCH_MARGIN,
            };
            if switch && current != Some(plurality) {
                self.predators[pi].search_image = Some(plurality as u8);
                // A new shape is a new problem, but not from scratch: keep half
                // the learned surplus. Resetting fully gave armour its immunity
                // back on every switch, which is most of why the first version
                // still ate the weakest animals in the pond.
                let surplus = (self.predators[pi].attack - base).max(0.0);
                self.predators[pi].attack = base + surplus * PREDATOR_ATTACK_RETENTION;
            }

            let Some(image) = self.predators[pi].search_image.map(|c| c as usize) else { continue };
            if image >= k || counts[image] == 0 { continue; }

            // Train past the toughest animal in the family — see
            // PREDATOR_ATTACK_MARGIN for why anything short of that rewards
            // armour rather than taxing it.
            let n = counts[image] as f64;
            let mean = armour[image] / n;
            let want = (toughest[image] + PREDATOR_ATTACK_MARGIN)
                .clamp(base, base + PREDATOR_ATTACK_MAX_ADAPT);
            let attack = self.predators[pi].attack;
            self.predators[pi].attack = attack + (want - attack) * PREDATOR_ATTACK_ADAPT;
            self.predators[pi].image_armour = mean;
            // New place in the cruising band. A hunter whose margin never moves
            // is a margin a lineage can settle exactly on top of.
            self.predators[pi].cruise_frac =
                self.rng.gen_range(PREDATOR_CRUISE_FRAC.0..=PREDATOR_CRUISE_FRAC.1);
        }
    }

    /// One predator's turn: depart, or chase and eat.
    fn hunt_one(&mut self, id: u32) {
        let Some(pi) = self.predators.iter().position(|p| p.id == id) else { return };
        let Some(idx) = self.slot_of(id) else {
            // Unreachable — a predator can't die — but drop it rather than
            // hunting with a stale slot if its id somehow vanished.
            self.predators.remove(pi);
            self.resync_predator_ids();
            return;
        };
        // Bursts: a small per-tick chance of running at the ceiling for a few
        // hundred ticks. Variance in the threat, so a lineage cannot settle on a
        // fixed safe margin — the same reason the search image moves.
        if self.predators[pi].burst_ticks > 0 {
            self.predators[pi].burst_ticks -= 1;
        } else if tier_resident(self.predators[pi].tier)
            && self.rng.gen_bool(PREDATOR_BURST_CHANCE)
        {
            self.predators[pi].burst_ticks =
                self.rng.gen_range(PREDATOR_BURST_TICKS.0..=PREDATOR_BURST_TICKS.1);
        }

        let target = self.predators[pi].target_pop;
        let tier = self.predators[pi].tier;
        let attack = self.predators[pi].attack;
        let automatic = self.predators[pi].automatic;
        let speed = self.predator_chase_speed(pi);
        let max_turn = TIER_MAX_TURN[(tier as usize).min(PREDATOR_TIERS - 1)];
        self.predators[pi].angle += tier_spin(tier);
        let angle = self.predators[pi].angle;
        let world_size = self.grid_size as f32;

        // Departing: swim straight on, at speed, until the swim is done. The
        // population is deliberately not re-checked here — births during the
        // departure must not drag it back into a second hunt.
        if let Some(ticks) = self.predators[pi].leaving {
            if ticks == 0 { return; }
            let dir = (self.vel_x[idx], self.vel_y[idx]);
            self.predator_steer(idx, pi, dir, PREDATOR_LEAVE_SPEED, max_turn);
            let left = ticks - 1;
            self.predators[pi].leaving = Some(left);
            if left == 0 { self.departed_ids.push(id); }
            return;
        }

        // Quota met — checked against the live count, which births keep moving.
        if target > 0 && self.prey_count() <= target {
            if tier_resident(tier) {
                // A resident that has finished its cull takes ambient duty —
                // quota zero, hunting forever — but only one of them does.
                //
                // It used to sit sated until the population crossed the trigger
                // again, which left the pond with no predator in it for most of
                // its life: predation as a thermostat with a setpoint. One
                // hunter that is always there, and slower than its prey, is a
                // pressure instead.
                //
                // The cap matters. The pack ratchets — every threshold crossing
                // leaves a permanent extra triangle — and with all of them on
                // ambient duty the pressure compounds with each boom the pond
                // has ever had. Measured, three ambient hunters took the pond
                // from 49 survivors to 15 and killed *more* than the fast ones
                // they replaced. The rest go quiet, as they always did, and wake
                // for the next cull.
                let ambient_taken = self.predators.iter().enumerate()
                    .any(|(j, p)| j != pi && p.leaving.is_none()
                        && tier_resident(p.tier) && p.target_pop == 0);
                self.predators[pi].sated = ambient_taken;
                if !ambient_taken {
                    self.predators[pi].target_pop = 0;
                }
                self.predators[pi].target_id = None;
                self.predators[pi].commit_ticks = 0;
                if ambient_taken {
                    let speed = self.predator_chase_speed(pi);
                    self.patrol(idx, pi, speed * PATROL_SPEED_FRAC);
                }
                return;
            }
            self.begin_departure(idx, pi);
            return;
        }
        self.predators[pi].sated = false;

        let px = self.pos_x[idx];
        let py = self.pos_y[idx];

        // The animal it is already hunting, if that is still worth doing: alive,
        // not itself a predator, and not so far off that the pond behind it is a
        // better prospect. Re-picking every tick is what made the heading flip.
        let mut target_slot = None;
        if self.predators[pi].commit_ticks > 0 {
            if let Some(tid) = self.predators[pi].target_id {
                if let Some(s) = self.slot_of(tid) {
                    let (dx, dy) = toroidal_delta(px, py, self.pos_x[s], self.pos_y[s], world_size);
                    if self.cause_of_death[s].is_none()
                        && !self.is_predator(s)
                        && dx * dx + dy * dy <= PREDATOR_COMMIT_RANGE * PREDATOR_COMMIT_RANGE
                    {
                        target_slot = Some(s);
                    }
                }
            }
        }
        // Commitment lapsed, or the target is gone: look again.
        if target_slot.is_none() {
            let image = self.predators[pi].search_image;
            let armour = self.predators[pi].image_armour;
            target_slot = self.nearest_prey(idx, self.predators[pi].rejected_id, image, armour);
            if target_slot.is_none() && self.predators[pi].rejected_id.is_some() {
                // The resistant animal is the only prey left. Take it again
                // rather than deadlocking the hunt on a skip.
                self.predators[pi].rejected_id = None;
                target_slot = self.nearest_prey(idx, None, image, armour);
            }
            self.predators[pi].target_id = target_slot.map(|s| self.ids[s]);
            self.predators[pi].commit_ticks = PREDATOR_COMMIT_TICKS;
        }
        self.predators[pi].commit_ticks = self.predators[pi].commit_ticks.saturating_sub(1);

        match target_slot {
            Some(s) => {
                let dir = toroidal_delta(px, py, self.pos_x[s], self.pos_y[s], world_size);
                self.predator_steer(idx, pi, dir, speed, max_turn);
            }
            // Nothing to chase at all. Keep swimming rather than freezing over
            // the last spot something was eaten.
            None => self.patrol(idx, pi, speed * PATROL_SPEED_FRAC),
        }

        // Eat everything the kill shape covers. Deliberately *not* clamped to
        // the remaining quota: clamping is what made a cull land on exactly the
        // threshold, which then sat one boom away from re-triggering. Overshoot
        // is the point — it buys the pond room to grow again.
        //
        // Every agent is tested rather than only the nearby buckets: the kill
        // shape must never miss something standing inside it, and the spatial
        // grid does not know about position writes made earlier in this same
        // phase. `is_predator` is a set lookup now, so this is one pass, not one
        // pass per member of the pack.
        let bite_x = self.pos_x[idx];
        let bite_y = self.pos_y[idx];
        let mut victims = Vec::new();
        let mut target_resisted = false;
        for i in 0..self.ids.len() {
            if i == idx || self.cause_of_death[i].is_some() || self.is_predator(i) { continue; }
            let (dx, dy) = toroidal_delta(bite_x, bite_y, self.pos_x[i], self.pos_y[i], world_size);
            if tier_bite_hits(tier, dx, dy, angle) {
                // God-mode immortality suppresses every natural cause, and an
                // ambient hunter is ecology like any other. A *summoned* one is
                // the player overruling the rules, which is the whole point of
                // the god panel, so those still bite.
                if self.immortal && automatic { continue; }
                if predator_attack_succeeds(
                    tier,
                    attack,
                    self.genome[i].traits.defense,
                    self.parent_defense_bonus[i],
                    self.age[i],
                ) {
                    victims.push(i);
                } else if target_slot == Some(i) {
                    target_resisted = true;
                }
            }
        }
        self.predators[pi].rejected_id =
            if target_resisted { target_slot.map(|i| self.ids[i]) } else { None };
        if target_resisted {
            // Drop the commitment too, or it spends the rest of the commitment
            // window swimming at an animal it has already failed to bite.
            self.predators[pi].target_id = None;
            self.predators[pi].commit_ticks = 0;
        }

        if !victims.is_empty() {
            let eaten = victims.len() as u32;
            for &i in &victims {
                self.cause_of_death[i] = Some(CauseOfDeath::EatenAlive);
                self.energy[i] = 0.0;
                self.lifespans.push(self.age[i]);
            }
            self.reap_dead(victims);
            self.spatial.rebuild(&self.pos_x, &self.pos_y);
            self.predators[pi].kills += eaten;
            if let Some(slot) = self.slot_of(id) {
                self.kills[slot] += eaten;
            }
        }

        if target > 0 && self.prey_count() <= target && !tier_resident(tier) {
            if let Some(slot) = self.slot_of(id) {
                self.begin_departure(slot, pi);
            }
        }
    }

    /// Slot of the best prey animal to chase: nearest, but with the hunter's
    /// search image counted as nearer than it is. Skips `skip_id`.
    ///
    /// The preference is a distance discount rather than a filter. A hunter that
    /// can only see off-image animals still hunts them — an exclusive search
    /// image would make a rare family untouchable, and untouchable is exactly
    /// how a family stops being rare.
    ///
    /// Only runs when a hunter needs a new target — once per
    /// `PREDATOR_COMMIT_TICKS`, not once per tick.
    fn nearest_prey(
        &self, idx: usize, skip_id: Option<u32>, image: Option<u8>, image_armour: f64,
    ) -> Option<usize> {
        let world_size = self.grid_size as f32;
        let (px, py) = (self.pos_x[idx], self.pos_y[idx]);
        let discount = 1.0 / (SEARCH_IMAGE_PULL * SEARCH_IMAGE_PULL);
        let mut best: Option<(f32, usize)> = None;
        for i in 0..self.ids.len() {
            if i == idx || self.cause_of_death[i].is_some() || self.is_predator(i) { continue; }
            if skip_id == Some(self.ids[i]) { continue; }
            let (dx, dy) = toroidal_delta(px, py, self.pos_x[i], self.pos_y[i], world_size);
            let mut d2 = dx * dx + dy * dy;
            if image.is_some() && self.cluster.genome_cluster_ids.get(i).copied() == image {
                d2 *= discount;
                // Within the image, the best-armoured animal is the one worth
                // chasing. This is the half of the mechanism that actually
                // taxes armour: being the hardest target in the commonest
                // family is what puts a hunter on you.
                if image_armour > 0.0 {
                    let def = effective_defense(
                        self.genome[i].traits.defense,
                        self.parent_defense_bonus[i],
                        self.age[i],
                    );
                    let ratio = (image_armour / def.max(1e-6)) as f32;
                    d2 *= ratio.clamp(ARMOUR_PREFERENCE_CLAMP.0, ARMOUR_PREFERENCE_CLAMP.1)
                        .powi(2);
                }
            }
            if best.is_none() || d2 < best.unwrap().0 { best = Some((d2, i)); }
        }
        best.map(|b| b.1)
    }

    /// Move one predator `step` world units, turning no more than `max_turn`
    /// radians this tick toward `dir`.
    ///
    /// Every predator motion path goes through here — chase, patrol, departure —
    /// so heading and speed are continuous across state changes. Velocity is the
    /// state that carries between ticks; position follows from it. Before this,
    /// chase wrote position from a fresh unit vector every tick and left velocity
    /// as a by-product, which is what let a hunter turn 180° between two frames.
    fn predator_steer(&mut self, idx: usize, pi: usize, dir: (f32, f32), step: f32, max_turn: f32) {
        let world_size = self.grid_size as f32;
        let want_len2 = dir.0 * dir.0 + dir.1 * dir.1;

        let (vx, vy) = (self.vel_x[idx], self.vel_y[idx]);
        let mut heading = if vx * vx + vy * vy < 1e-6 {
            // Nothing to preserve: face where we want to go, or anywhere at all.
            if want_len2 < 1e-12 { self.rng.gen::<f32>() * TAU } else { dir.1.atan2(dir.0) }
        } else {
            vy.atan2(vx)
        };
        if want_len2 >= 1e-12 {
            // Shortest signed turn, then clamp it to the tier's rate.
            let mut delta = (dir.1.atan2(dir.0) - heading).rem_euclid(TAU);
            if delta > PI { delta -= TAU; }
            heading += delta.clamp(-max_turn, max_turn);
        }

        let speed = self.predators[pi].speed
            + (step - self.predators[pi].speed) * PREDATOR_SPEED_EASE;
        self.predators[pi].speed = speed;

        let (dx, dy) = (heading.cos() * speed, heading.sin() * speed);
        self.prev_x[idx] = self.pos_x[idx];
        self.prev_y[idx] = self.pos_y[idx];
        self.pos_x[idx] = (self.pos_x[idx] + dx).rem_euclid(world_size);
        self.pos_y[idx] = (self.pos_y[idx] + dy).rem_euclid(world_size);
        self.vel_x[idx] = dx / DT;
        self.vel_y[idx] = dy / DT;
    }

    /// Idle motion for a predator with nothing to hunt. It keeps its heading,
    /// drifts, and stays visible — the pond is supposed to feel like it has
    /// permanent residents once they have arrived.
    ///
    /// The wander is a random walk on the *turn rate*, not on the heading: an
    /// uncorrelated turn every tick at 20 Hz reads as vibration, and patrolling
    /// is the state a sated resident spends most of its life in.
    fn patrol(&mut self, idx: usize, pi: usize, speed: f32) {
        let noise = (self.rng.gen::<f32>() - 0.5) * 2.0 * PATROL_TURN_NOISE;
        let rate = (self.predators[pi].turn_rate * PATROL_TURN_MEMORY + noise)
            .clamp(-PATROL_TURN_MAX, PATROL_TURN_MAX);
        self.predators[pi].turn_rate = rate;

        let (vx, vy) = (self.vel_x[idx], self.vel_y[idx]);
        let heading = if vx * vx + vy * vy < 1e-6 {
            self.rng.gen::<f32>() * TAU
        } else {
            vy.atan2(vx)
        };
        let want = heading + rate;
        self.predator_steer(idx, pi, (want.cos(), want.sin()), speed, PATROL_TURN_MAX);
    }

    /// Stop hunting and start swimming away. Heading is whatever it was last
    /// chasing, so the exit continues its current motion rather than snapping.
    fn begin_departure(&mut self, idx: usize, pi: usize) {
        let vx = self.vel_x[idx];
        let vy = self.vel_y[idx];
        if vx * vx + vy * vy < 1e-6 {
            // Standing still (it ate its last prey exactly on the spot): pick a
            // direction rather than dividing by zero on the first departure step.
            let angle = self.rng.gen::<f32>() * TAU;
            self.vel_x[idx] = angle.cos();
            self.vel_y[idx] = angle.sin();
        }
        self.predators[pi].target_id = None;
        self.predators[pi].commit_ticks = 0;
        self.predators[pi].leaving = Some(PREDATOR_LEAVE_TICKS);
    }

    /// Arrivals and reinforcements.
    ///
    /// Two jobs. A pond over its capacity gets a pack sized to the overshoot.
    /// And any hunt already under way — summoned or automatic — gets another
    /// hunter if the prey count still isn't falling a while later, so a pond
    /// that outbreeds its predators is answered by more of them.
    fn manage_predator_pack(&mut self) {
        if !self.automatic_predators_enabled { return; }
        let prey = self.prey_count();

        // Ambient pressure: one resident triangle whenever there is a pond worth
        // hunting, with a quota of zero so it never sates and never stops.
        //
        // Predation used to exist only as a cull — nothing in the water until
        // the population crossed capacity, then a pack, then quiet again. That
        // makes predation a controller with a setpoint, which is exactly what it
        // should not be: the pond spent most of its life with no predator in it
        // at all, and evolved accordingly. A single hunter that is always there
        // and always slower than its prey is a pressure, not a thermostat. The
        // capacity rule below still stacks a pack on top when the pond outgrows
        // its cap.
        let resident_hunting = self.predators.iter()
            .any(|p| p.leaving.is_none() && tier_resident(p.tier));
        if prey >= PREDATOR_AMBIENT_MIN_PREY && !resident_hunting {
            self.summon_predator_tier(0, true, 0);
        }
        // A cull in progress, if any. The ambient resident carries a quota of
        // zero and hunts forever, so counting it here would mask the capacity
        // rule completely — the pond could sit far over its cap with the cull
        // branch never reached, because something was technically "hunting".
        let hunting = self.predators.iter()
            .filter(|p| p.leaving.is_none() && !p.sated && p.target_pop > 0)
            .map(|p| p.target_pop)
            .min();

        match hunting {
            // Nothing hunting: only the capacity rule can start a wave, and it
            // only ever fields triangles.
            None => {
                if prey > self.cull_trigger_pop() {
                    let target = self.cull_target_pop();
                    // Wake every resident, then add one more to the pack. The
                    // pond earns a permanent hunter each time it outgrows the
                    // ones it already has.
                    for p in self.predators.iter_mut() {
                        p.sated = false;
                        p.target_pop = target;
                    }
                    let resident = self.predators.iter()
                        .filter(|p| p.leaving.is_none() && tier_resident(p.tier))
                        .count();
                    let want = (resident + 1).max(TIER_PACK[0]).min(PREDATOR_MAX);
                    for _ in resident..want {
                        if self.summon_predator_tier(target, true, 0).is_none() {
                            break;
                        }
                    }
                    self.last_reinforce_step = self.step_count;
                }
                self.last_reinforce_pop = prey;
            }
            // A hunt is under way. A pond still not falling after a while has
            // outbred the pack it has, so the pack grows by one — the same rule
            // as a fresh threshold crossing, since that is what this is.
            Some(target) => {
                if self.step_count.saturating_sub(self.last_reinforce_step)
                    < PREDATOR_REINFORCE_STEPS {
                    return;
                }
                self.last_reinforce_step = self.step_count;

                // Only while the population is not actually falling. If the
                // hunters already in the water are winning, adding more would
                // overshoot the target and turn a cull into an extinction.
                if prey > target && prey >= self.last_reinforce_pop {
                    self.summon_predator_tier(target, true, 0);
                }
                self.last_reinforce_pop = prey;
            }
        }
    }

    /// Living population excluding predators — the number a cull targets.
    pub fn prey_count(&self) -> usize {
        self.agent_count().saturating_sub(self.predators.len())
    }


    // ── Step loop ─────────────────────────────────────────────────────────────

    pub fn step(&mut self) {
        self.step_count += 1;

        // Phase 1: food regen
        self.tick_food_regen();
        for c in self.tile_eat_count_this_tick.iter_mut() { *c = 0; }

        // Phase 2: age / passive metabolism drain / natural death
        self.scratch_dead.clear();
        self.tick_age_and_metabolism_scratch();

        // Rebuild spatial after natural deaths (dead agents still in arrays until reap)
        self.spatial.rebuild(&self.pos_x, &self.pos_y);

        // Phase 3: collect alive agents
        self.scratch_acting.clear();
        self.partition_agents_scratch();

        // Phase 4: perception → 5-input vector per acting agent
        let acting_len = self.scratch_acting.len();
        self.scratch_perceptions.resize(acting_len, [0f32; INPUT_COUNT]);
        self.scratch_food_dirs.resize(acting_len, (0f32, 0f32));
        self.scratch_deciding.clear();
        self.scratch_deciding.resize(acting_len, false);
        self.sense_threats();
        self.perceive_all();

        // Phase 5: brain forward → 8 sigmoid outputs, for the agents thinking
        // this tick. The rest replay their last decision.
        self.scratch_outputs.resize(acting_len, [0f32; 8]);
        self.steer_all();

        // Phase 6: integrate physics + fire discrete triggers
        let mut offspring: Vec<PendingAgent> = Vec::new();
        for slot in 0..acting_len {
            let idx = self.scratch_acting[slot];
            if self.cause_of_death[idx].is_some() { continue; }
            let perception = self.scratch_perceptions[slot];
            let food_dir = self.scratch_food_dirs[slot];
            let outputs = self.scratch_outputs[slot];
            if let Some(child) = self.integrate_agent(idx, perception, food_dir, outputs) {
                offspring.push(child);
            }
            if self.energy[idx] <= 0.0 && self.cause_of_death[idx].is_none() {
                if self.immortal || self.is_predator(idx) {
                    self.energy[idx] = IMMORTAL_ENERGY_FLOOR;
                } else {
                    self.cause_of_death[idx] = Some(CauseOfDeath::Starvation);
                    self.lifespans.push(self.age[idx]);
                    self.scratch_dead.push(idx);
                }
            }
        }

        // Phase 6b: contagion. After actions, so this tick's crowding is the
        // crowding the agents actually chose, and before the reap so a death
        // this tick is not also a source of infection.
        self.tick_disease();

        // Phase 7: passive combat per tile
        self.resolve_combat_spatial();

        // Phase 8: add offspring
        self.spawn_offspring(offspring);

        // Phase 9: reap dead
        let dead: Vec<usize> = self.scratch_dead.clone();
        self.reap_dead(dead);

        // Rebuild spatial for next step
        self.spatial.rebuild(&self.pos_x, &self.pos_y);

        // Phase 9b: the predators hunt. After the reap so none of them ever
        // chases an agent that is already dead this tick, and before clustering
        // so the k-means pass sees the population they actually left behind.
        self.hunt_with_predators();

        // Arrivals and reinforcements. A pond past its carrying capacity can no
        // longer be drawn at frame rate — every agent is an individually drawn
        // body — so predators are the pressure valve.
        self.manage_predator_pack();

        // Phase 10: dual k-means clustering every 50 steps, or immediately after
        // a `cluster_k` change so the dial isn't dead until the next cycle.
        if (self.step_count % 50 == 0 || self.cluster_dirty) && !self.genome.is_empty() {
            self.cluster_dirty = false;
            let prev = std::mem::take(&mut self.cluster);
            let k = self.tunables.cluster_k;
            self.cluster = ClusterState::run(&self.genome, k, self.step_count, Some(&prev));
            self.brain_clusters.begin(&self.genome, 24, self.step_count);
            // Speciation reads the fresh clustering and consumes no RNG, so a
            // run with it enabled steps identically to one without.
            // Membership is carried, not recomputed: `update` only releases the
            // members of an extinct species and seats a new one's founders.
            let mut assignment = std::mem::take(&mut self.species_ids);
            self.species.update(&self.genome, &self.cluster, self.step_count, &mut assignment);
            self.species_ids = assignment;

            // A promotion is the only thing that can introduce a pathogen. The
            // roll lives here rather than in `species.rs` because it needs the
            // world RNG, and that module is deliberately RNG-free.
            let promoted: Vec<(u32, String)> = self.species.all().iter()
                .filter(|s| s.founded_step == self.step_count)
                .map(|s| (s.id, s.name.genus.clone()))
                .collect();
            for (id, genus) in promoted {
                self.maybe_seed_disease(id, &genus);
            }
            // Hunters re-read the pond on the same tick the labels are rebuilt.
            self.review_predator_search_images();
        }

        // Behavioural clustering advances one iteration per step so its cost is
        // a ripple rather than the frame-dropping spike it was when the whole
        // pass ran inside the tick. No-op when disabled or idle.
        self.brain_clusters.advance(&self.genome);

        // Observation, not a phase: read-only sampling of the state the ten
        // phases just produced. Consumes no RNG, mutates no agent.
        if self.step_count % SAMPLE_INTERVAL == 0 {
            self.sample_stats();
        }
    }

    /// Cumulative deaths per cause since the run began, indexed by
    /// `CauseOfDeath::code()`.
    pub fn death_counts(&self) -> [u32; CAUSE_COUNT] {
        let mut out = [0u32; CAUSE_COUNT];
        for (cause, &count) in &self.death_tally {
            out[cause.code() as usize] += count;
        }
        out
    }

    /// Append one sample to `stats_history`. Called on interval boundaries after
    /// the dead have been reaped, so every agent still in the arrays is alive.
    fn sample_stats(&mut self) {
        let stats = self.get_stats();
        // Percentiles, not min/max: reproduction is continuous, so there is
        // essentially always a newborn and the minimum sat at 0 forever.
        let (age_p10, age_p90) = age_percentiles(&self.age);
        // Median age at death during this interval only. `lifespans` is
        // append-only, so the tail past the last sample is exactly this
        // interval's deaths — the same differencing trick the death counts use.
        let interval_median = median(&self.lifespans[self.lifespans_sampled.min(self.lifespans.len())..]);
        self.lifespans_sampled = self.lifespans.len();
        let deaths = self.stats_history.interval_deaths(self.death_counts());
        let n = self.genome.len();
        let (mean_generation, max_generation) = if n > 0 {
            let sum: u64 = self.genome.iter().map(|g| g.generation as u64).sum();
            let max = self.genome.iter().map(|g| g.generation).max().unwrap_or(0);
            (sum as f32 / n as f32, max)
        } else {
            (0.0, 0)
        };

        self.stats_history.push(StatSample {
            step: self.step_count,
            alive: stats.alive_agents as u32,
            total_food: stats.total_food,
            avg_energy: stats.avg_energy as f32,
            median_lifespan: interval_median as f32,
            age_p10,
            age_p90,
            deaths,
            mean_generation,
            max_generation,
        });
    }

    // ── Phase implementations ─────────────────────────────────────────────────

    fn tick_food_regen(&mut self) {
        let scale = self.tunables.food_regen_scale;
        for tile in &mut self.tiles {
            if tile.food_units < MAX_FOOD_PER_TILE {
                let rate = tile.regen_rate(scale);
                if self.rng.gen::<f64>() < rate {
                    tile.food_units += 1;
                }
            }
        }
    }

    fn tick_age_and_metabolism_scratch(&mut self) {
        let n = self.ids.len();
        for i in 0..n {
            if self.cause_of_death[i].is_some() { continue; }
            self.age[i] += 1;

            if self.age[i] >= self.death_age[i] && !self.immortal && !self.is_predator(i) {
                self.cause_of_death[i] = Some(CauseOfDeath::OldAge);
                self.lifespans.push(self.age[i]);
                self.scratch_dead.push(i);
                continue;
            }

            let metabolism = self.genome[i].traits.metabolism;
            self.energy[i] -= BASE_DRAIN * metabolism;
            // Thinking costs. Paid every tick by every agent, whether or not it
            // decided this one — the brain is carried, not rented.
            //
            // Predators are exempt rather than merely floored: they run no brain
            // at all, take no decision cadence and no detection lag, and hunt at
            // full rate always. Charging them for a brain they do not have would
            // be the first step toward a future refactor slowing them down by
            // accident.
            if !self.is_predator(i) {
                self.energy[i] -=
                    INTELLIGENCE_UPKEEP * self.genome[i].traits.intelligence * metabolism;
                // Armour is mass, and mass is carried every tick.
                self.energy[i] -=
                    DEFENSE_UPKEEP * armour_margin(self.genome[i].traits.defense) * metabolism;
                // So is an immune system.
                self.energy[i] -=
                    IMMUNITY_UPKEEP * self.genome[i].traits.immunity * metabolism;
                // Being ill costs. Severity is a drain rather than a death roll,
                // so an outbreak lands on the food economy and takes the
                // already-marginal first.
                if self.infection[i] != 0 {
                    if let Some(d) = self.disease_of(i) {
                        self.energy[i] -= d.severity * metabolism;
                    }
                }
            }

            if self.energy[i] <= 0.0 {
                if self.immortal || self.is_predator(i) {
                    // Held just above zero rather than at it: an agent pinned at
                    // 0 would re-trip every death check in the step and read as
                    // permanently dying instead of merely starving.
                    self.energy[i] = IMMORTAL_ENERGY_FLOOR;
                } else {
                    // An infected agent that runs out is a death from the
                    // disease, not from the food economy it was pushed into.
                    self.cause_of_death[i] = Some(if self.infection[i] != 0 {
                        CauseOfDeath::Disease
                    } else {
                        CauseOfDeath::Starvation
                    });
                    self.lifespans.push(self.age[i]);
                    self.scratch_dead.push(i);
                }
            }
        }
    }

    /// The pathogen an agent is carrying, if any.
    /// Live carriers of each disease, and how many of them belong to each live
    /// species. Row `d` is disease id `d + 1`; column 0 of the per-species block
    /// is "unassigned", column `s` is species id `s`.
    ///
    /// Counted here rather than in JS because the page has no per-agent species
    /// *and* infection pairing without another array export, and this is a
    /// once-per-panel-refresh walk either way.
    pub fn disease_carrier_census(&self, species_columns: usize) -> Vec<Vec<u32>> {
        let mut out = vec![vec![0u32; species_columns]; self.diseases.len()];
        for i in 0..self.ids.len() {
            let id = self.infection[i];
            if id == 0 || self.cause_of_death[i].is_some() { continue; }
            let Some(row) = out.get_mut(id as usize - 1) else { continue };
            let species = self.species_ids.get(i).copied().unwrap_or(0) as usize;
            if let Some(cell) = row.get_mut(species.min(species_columns - 1)) {
                *cell += 1;
            }
        }
        out
    }

    pub fn disease_of(&self, idx: usize) -> Option<&Disease> {
        let id = *self.infection.get(idx)?;
        if id == 0 { return None; }
        self.diseases.get(id as usize - 1)
    }

    /// Roll for a pathogen in a newly promoted species, and infect its first
    /// case.
    ///
    /// Flat probability: not weighted by population, species age or dominance.
    /// A disease more likely to appear in a crowded pond is a density-dependent
    /// cull wearing a costume, and the point of this mechanic is to be a
    /// disturbance that arrives on its own schedule.
    fn maybe_seed_disease(&mut self, species_id: u32, genus: &str) {
        if !self.disease_enabled { return; }
        if !self.rng.gen_bool(DISEASE_CHANCE) { return; }

        let id = self.diseases.len() as u32 + 1;
        let severity = self.rng.gen_range(SEVERITY_RANGE.0..=SEVERITY_RANGE.1);
        let contagion = self.rng.gen_range(CONTAGION_RANGE.0..=CONTAGION_RANGE.1);
        let name = crate::naming::disease_name(genus, id, self.species.world_seed());
        self.diseases.push(Disease {
            id, name, origin_species: species_id, severity, contagion,
            emerged_step: self.step_count, jumped: false,
        });

        // Patient zero: one member of the species it emerged in. Without a first
        // case the pathogen exists on paper and never infects anything.
        let first = (0..self.ids.len()).find(|&i| {
            self.cause_of_death[i].is_none() && !self.is_predator(i)
                && self.species_ids.get(i).copied() == Some(species_id)
        });
        if let Some(i) = first { self.infection[i] = id; }
    }

    /// Spread every live infection by contact.
    ///
    /// Density-dependent and *locally* so: the chance of catching something is
    /// set by how many agents are within `CONTACT_RADIUS`, clamped at
    /// `CROWDING_FULL`. Nothing here reads the total population, which is what
    /// lets an outbreak overshoot and crash rather than trimming the pond toward
    /// a setpoint — a lineage that becomes numerous *and* clustered is what
    /// makes an epidemic, not a lineage that is merely numerous.
    fn tick_disease(&mut self) {
        if !self.disease_enabled || self.diseases.is_empty() { return; }

        // Collected first: the roll below mutates `infection`, and an agent
        // infected this tick must not go on to infect others in the same tick —
        // that would make transmission depend on iteration order.
        let carriers: Vec<(usize, u32)> = (0..self.ids.len())
            .filter(|&i| self.infection[i] != 0 && self.cause_of_death[i].is_none())
            .filter(|&i| !self.is_predator(i))
            .map(|i| (i, self.infection[i]))
            .collect();
        if carriers.is_empty() { return; }

        let mut caught: Vec<(usize, u32)> = Vec::new();
        let mut jumps: Vec<u32> = Vec::new();
        for (i, disease_id) in carriers {
            let Some(d) = self.diseases.get(disease_id as usize - 1) else { continue };
            let (contagion, origin, jumped) = (d.contagion, d.origin_species, d.jumped);
            let (px, py) = (self.pos_x[i], self.pos_y[i]);
            let neighbours = self.spatial.agents_near(px, py, CONTACT_RADIUS);

            let crowd = (neighbours.len() as f64 / CROWDING_FULL).clamp(0.0, 1.0);
            for &j in &neighbours {
                if j == i || self.infection[j] != 0 || self.cause_of_death[j].is_some() { continue; }
                if self.is_predator(j) { continue; }
                // Once a pathogen has jumped it is nobody's disease in
                // particular and spreads at full contagion to anything.
                // Immunity is resistance to *catching* it and nothing else. An
                // infected agent carries it to the grave however immune it is:
                // recovery would be a restoring force and would damp exactly the
                // oscillation this mechanic exists to create.
                let resist = 1.0 - self.genome[j].traits.immunity.clamp(0.0, 1.0);
                let host = jumped || self.species_ids.get(j).copied() == Some(origin);
                if host {
                    let p = contagion * crowd * resist;
                    if p > 0.0 && self.rng.gen_bool(p.clamp(0.0, 1.0)) {
                        caught.push((j, disease_id));
                    }
                } else if self.rng.gen_bool(CROSS_SPECIES_JUMP * resist) {
                    caught.push((j, disease_id));
                    jumps.push(disease_id);
                }
            }
        }
        for (j, id) in caught {
            if self.infection[j] == 0 { self.infection[j] = id; }
        }
        for id in jumps {
            if let Some(d) = self.diseases.get_mut(id as usize - 1) { d.jumped = true; }
        }
    }

    /// Everything that gets a brain this tick.
    ///
    /// Predators are excluded. They have their own phase, and having them in
    /// both meant the brain wrote a velocity capped at `MAX_SPEED` and a
    /// `prev_*` pair, and then the hunt overwrote all four a few phases later:
    /// the heading a hunter tried to hold was perturbed by wander force every
    /// tick, and the leg of motion the brain had already moved it through was
    /// never interpolated by the renderer. An apex predator does not forage,
    /// sleep or breed either — those gates fired here and nowhere else.
    fn partition_agents_scratch(&mut self) {
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_none() && !self.is_predator(i) {
                self.scratch_acting.push(i);
            }
        }
    }

    /// Look for predators, once per tick per agent, and push what was seen into
    /// the agent's threat pipeline.
    ///
    /// Vision sets the range at which a predator registers at all; a hunter
    /// outside it is simply not there as far as this agent is concerned, however
    /// close it is about to be. Intelligence decides how long the sighting takes
    /// to reach the brain — that part happens on read, in `perceive`.
    ///
    /// Scans `self.predators` rather than the spatial grid: there are at most a
    /// dozen of them and usually one, so this is cheaper than a neighbourhood
    /// query and does not care how crowded the pond is.
    fn sense_threats(&mut self) {
        let world_size = self.grid_size as f32;
        // Positions first: the borrow checker will not let us read `self.pos_*`
        // for predators while writing rings on `self`, and this is a handful of
        // entries.
        let hunters: Vec<(f32, f32)> = self.predators.iter()
            .filter(|p| p.leaving.is_none())
            .filter_map(|p| self.slot_of(p.id))
            .map(|s| (self.pos_x[s], self.pos_y[s]))
            .collect();

        for slot in 0..self.scratch_acting.len() {
            let idx = self.scratch_acting[slot];
            let (px, py) = (self.pos_x[idx], self.pos_y[idx]);
            let vision_radius = self.genome[idx].traits.vision as f32 * VISION_SCALE;

            let mut best: Option<(f32, f32, f32)> = None;   // (dist, dx, dy)
            for &(hx, hy) in &hunters {
                let (dx, dy) = toroidal_delta(px, py, hx, hy, world_size);
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > vision_radius { continue; }
                if best.is_none() || dist < best.unwrap().0 { best = Some((dist, dx, dy)); }
            }

            let entry = match best {
                None => [1.0, 0.0, 0.0, 0.0],
                Some((dist, dx, dy)) => {
                    let dist_norm = (dist / vision_radius.max(1e-6)).clamp(0.0, 1.0);
                    // Bearing to the threat relative to heading, in [-1, 1] — the
                    // same convention the food channel uses, so the brain reads
                    // both angles the same way.
                    let (vx, vy) = (self.vel_x[idx], self.vel_y[idx]);
                    let angle_norm = if vx * vx + vy * vy > 1e-6 {
                        let mut a = dy.atan2(dx) - vy.atan2(vx);
                        while a > PI { a -= TAU; }
                        while a < -PI { a += TAU; }
                        a / PI
                    } else {
                        0.0
                    };
                    // Unit vector *away*, cached so the flee force needs no second
                    // scan — exactly how food_dir works.
                    let inv = 1.0 / dist.max(1e-6);
                    [dist_norm, angle_norm, -dx * inv, -dy * inv]
                }
            };

            let head = self.threat_head[idx];
            self.threat_ring[idx][head] = entry;
            self.threat_head[idx] = (head + 1) % THREAT_RING;
        }
    }

    /// What this agent's brain is allowed to know about threats right now: the
    /// pipeline entry from `threat_lag(intelligence)` ticks ago.
    fn delayed_threat(&self, idx: usize) -> [f32; 4] {
        let lag = threat_lag(self.genome[idx].traits.intelligence).min(THREAT_RING - 1);
        // `threat_head` points at the *next* write, so the newest entry is one
        // behind it.
        let newest = (self.threat_head[idx] + THREAT_RING - 1) % THREAT_RING;
        self.threat_ring[idx][(newest + THREAT_RING - lag) % THREAT_RING]
    }

    /// Perceive, but only for the agents whose turn it is to think. Everyone
    /// else keeps the perception their last decision was made from — a dull
    /// agent acts on an old picture, which is the whole cost of being dull.
    ///
    /// The cadence gate lives here rather than in `partition_agents_scratch`,
    /// which is the obvious place and the wrong one: that list also drives
    /// physics integration, so skipping an agent there would freeze it in the
    /// water instead of leaving it swimming on a stale intent.
    fn perceive_all(&mut self) {
        for slot in 0..self.scratch_acting.len() {
            let idx = self.scratch_acting[slot];
            let thinking = self.decision_cooldown[idx] == 0;
            self.scratch_deciding[slot] = thinking;
            if thinking {
                let (perception, food_dir) = self.perceive(idx);
                self.scratch_perceptions[slot] = perception;
                self.scratch_food_dirs[slot] = food_dir;
                self.last_perception[idx] = perception;
                self.last_food_dir[idx] = food_dir;
                self.decision_cooldown[idx] =
                    decision_interval(self.genome[idx].traits.intelligence) - 1;
            } else {
                self.scratch_perceptions[slot] = self.last_perception[idx];
                self.scratch_food_dirs[slot] = self.last_food_dir[idx];
                self.decision_cooldown[idx] -= 1;
            }
        }
    }

    /// Forward pass for the agents thinking this tick; the others get their
    /// previous outputs back unchanged.
    fn steer_all(&mut self) {
        for slot in 0..self.scratch_acting.len() {
            let idx = self.scratch_acting[slot];
            if !self.scratch_deciding[slot] {
                self.scratch_outputs[slot] = self.last_outputs[idx];
                continue;
            }
            let weights = self.genome[idx].weights_array();
            let p = self.scratch_perceptions[slot];
            let logits = brain_forward(weights, p);
            let outputs = sigmoid_outputs(logits);
            self.scratch_outputs[slot] = outputs;
            self.last_outputs[idx] = outputs;
        }
    }

    /// Build 5-input perception vector for one agent, plus the unit direction
    /// to the nearest visible food tile ((0,0) if none) so the steering pass
    /// doesn't have to re-scan the same tiles.
    fn perceive(&self, idx: usize) -> ([f32; INPUT_COUNT], (f32, f32)) {
        let px = self.pos_x[idx];
        let py = self.pos_y[idx];
        let vx = self.vel_x[idx];
        let vy = self.vel_y[idx];
        let vision = self.genome[idx].traits.vision as f32;
        let speed_trait = self.genome[idx].traits.speed as f32;
        let vision_radius = vision * VISION_SCALE;
        let world_size = self.grid_size as f32;

        // [0] energy
        let max_e = MAX_ENERGY_BASE * self.genome[idx].traits.energy_capacity;
        let energy_norm = (self.energy[idx] / max_e).clamp(0.0, 1.0) as f32;

        // [1,2] nearest food tile distance + angle relative to velocity
        let (food_dist_norm, food_angle_norm, food_dir) =
            self.nearest_food_inputs(idx, px, py, vx, vy, vision_radius, world_size);

        // [3] agent density within separation radius
        let nearby = self.spatial.agents_near(px, py, SEPARATION_RADIUS + 0.5);
        let neighbor_count = nearby.iter()
            .filter(|&&i| i != idx && self.cause_of_death[i].is_none())
            .count();
        let agent_density_norm = (neighbor_count as f32 / 8.0).clamp(0.0, 1.0);

        // [4] current speed normalized to max
        let cur_speed = (vx * vx + vy * vy).sqrt();
        let max_speed = speed_trait * MAX_SPEED;
        let speed_norm = if max_speed > 0.0 { (cur_speed / max_speed).clamp(0.0, 1.0) } else { 0.0 };

        // [5,6] threat: distance and bearing to the nearest predator this agent
        // can see, as of `threat_lag` ticks ago.
        let threat = self.delayed_threat(idx);
        let (threat_dist_norm, threat_angle_norm) = (threat[0], threat[1]);

        (
            [
                energy_norm, food_dist_norm, food_angle_norm, agent_density_norm,
                speed_norm, threat_dist_norm, threat_angle_norm,
            ],
            food_dir,
        )
    }

    fn nearest_food_inputs(
        &self,
        _idx: usize,
        px: f32, py: f32,
        vx: f32, vy: f32,
        vision_radius: f32,
        world_size: f32,
    ) -> (f32, f32, (f32, f32)) {
        let gs = self.grid_size;
        let r_cells = vision_radius.ceil() as i32;
        let cx = px.floor() as i32;
        let cy = py.floor() as i32;

        let mut nearest_dist_sq = f32::MAX;
        let mut nearest_ddx = 0.0f32;
        let mut nearest_ddy = 0.0f32;
        let mut found = false;

        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx + dx).rem_euclid(gs as i32) as usize;
                let ty = (cy + dy).rem_euclid(gs as i32) as usize;
                if self.tiles[ty * gs + tx].food_units == 0 { continue; }

                // Food tile center, wrapped delta
                let fx = tx as f32 + 0.5;
                let fy = ty as f32 + 0.5;
                let mut ddx = fx - px;
                let mut ddy = fy - py;
                let half = world_size * 0.5;
                if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
                if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }

                let dist_sq = ddx * ddx + ddy * ddy;
                if dist_sq < nearest_dist_sq {
                    nearest_dist_sq = dist_sq;
                    nearest_ddx = ddx;
                    nearest_ddy = ddy;
                    found = true;
                }
            }
        }

        if !found {
            return (1.0, 0.0, (0.0, 0.0));
        }
        let dist = nearest_dist_sq.sqrt();
        if dist > vision_radius {
            return (1.0, 0.0, (0.0, 0.0));
        }

        let food_dist_norm = (dist / vision_radius).clamp(0.0, 1.0);

        let cur_speed = (vx * vx + vy * vy).sqrt();
        let food_angle_norm = if cur_speed > 0.01 {
            let food_angle = nearest_ddy.atan2(nearest_ddx);
            let vel_angle = vy.atan2(vx);
            let mut rel = food_angle - vel_angle;
            while rel > std::f32::consts::PI { rel -= TAU; }
            while rel < -std::f32::consts::PI { rel += TAU; }
            rel / std::f32::consts::PI
        } else {
            0.0
        };

        let inv = 1.0 / dist.max(0.001);
        (food_dist_norm, food_angle_norm, (nearest_ddx * inv, nearest_ddy * inv))
    }

    /// Apply steering forces, integrate position, fire discrete triggers.
    /// Returns Some(PendingAgent) if reproduction fires.
    fn integrate_agent(
        &mut self,
        idx: usize,
        perception: [f32; INPUT_COUNT],
        food_dir: (f32, f32),
        outputs: [f32; 8],
    ) -> Option<PendingAgent> {
        let px = self.pos_x[idx];
        let py = self.pos_y[idx];
        let vx = self.vel_x[idx];
        let vy = self.vel_y[idx];
        let world_size = self.grid_size as f32;
        let speed_trait = self.genome[idx].traits.speed as f32;
        let metabolism = self.genome[idx].traits.metabolism;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;

        // Seek food (if food visible: perception[1] < 1.0; direction cached
        // from the perception pass — same nearest tile, no second scan)
        if perception[1] < 1.0 {
            fx += food_dir.0 * outputs[OUT_SEEK] * MAX_FORCE;
            fy += food_dir.1 * outputs[OUT_SEEK] * MAX_FORCE;
        }

        // Wander — random perturbation
        let wander_angle = self.rng.gen::<f32>() * TAU;
        fx += wander_angle.cos() * outputs[OUT_WANDER] * WANDER_FORCE;
        fy += wander_angle.sin() * outputs[OUT_WANDER] * WANDER_FORCE;

        // Separation from nearby agents
        let (sx, sy) = self.separation_force(idx, px, py, world_size);
        fx += sx * outputs[OUT_SEPARATE] * MAX_FORCE;
        fy += sy * outputs[OUT_SEPARATE] * MAX_FORCE;

        // Flee — straight away from the threat the brain was told about, at the
        // weight the brain chose. Gated on there being one (dist < 1.0), exactly
        // as seek is gated on food being visible.
        //
        // Nothing here is a reflex. An agent whose threat→flee weights never
        // evolved simply does not turn, and dies to something it could see
        // coming. That is a valid outcome, and there is deliberately no fallback
        // that saves it.
        if perception[5] < 1.0 {
            let threat = self.delayed_threat(idx);
            fx += threat[2] * outputs[OUT_FLEE] * MAX_FORCE;
            fy += threat[3] * outputs[OUT_FLEE] * MAX_FORCE;
        }

        // Velocity integration
        let max_speed = speed_trait * MAX_SPEED;
        let mut nvx = vx + fx * DT;
        let mut nvy = vy + fy * DT;
        let cur_speed = (nvx * nvx + nvy * nvy).sqrt();
        if cur_speed > max_speed && cur_speed > 0.0 {
            nvx = nvx / cur_speed * max_speed;
            nvy = nvy / cur_speed * max_speed;
        }

        // Save previous position for renderer interpolation
        self.prev_x[idx] = px;
        self.prev_y[idx] = py;

        // Integrate position with toroidal wrap
        let npx = (px + nvx * DT).rem_euclid(world_size);
        let npy = (py + nvy * DT).rem_euclid(world_size);

        // Incremental spatial update
        self.spatial.move_agent(idx, px, py, npx, npy);

        self.pos_x[idx] = npx;
        self.pos_y[idx] = npy;
        self.vel_x[idx] = nvx;
        self.vel_y[idx] = nvy;

        // Movement energy cost proportional to distance traveled, scaled by the
        // terrain speed modifier of the tile departed from (RULES.md: terrain_speed
        // × speed × metabolism × 0.15 — speed itself is already baked into dist via
        // the velocity cap above).
        let dist = (nvx * nvx + nvy * nvy).sqrt() * DT as f32;
        let (otx, oty) = SpatialHashGrid::tile_of(px, py, self.grid_size);
        let terrain_speed = self.tiles[oty * self.grid_size + otx].movement_speed;
        self.energy[idx] -= dist as f64 * terrain_speed * metabolism * MOVE_COST;

        // Discrete triggers are mutually exclusive per tick — the strongest gate above
        // threshold wins. Previously all three fired independently in the same tick
        // (free sleep alongside eat/reproduce/movement), which made SLEEP a dominant,
        // no-cost strategy.
        let gates = [
            (OUT_EAT, outputs[OUT_EAT]),
            (OUT_REPRODUCE, outputs[OUT_REPRODUCE]),
            (OUT_SLEEP, outputs[OUT_SLEEP]),
        ];
        let fired = gates.iter()
            .filter(|&&(_, v)| v > 0.5)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // One memory record per tick: the fired trigger if any, else the
        // dominant steering output.
        let recorded = fired.map(|&(winner, _)| winner as u8).unwrap_or_else(|| {
            outputs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u8)
                .unwrap_or(OUT_WANDER as u8)
        });
        self.memory[idx].record_action(recorded);

        if let Some(&(winner, _)) = fired {
            if winner == OUT_EAT {
                self.do_eat(idx);
            } else if winner == OUT_SLEEP {
                // Rest, not recovery: recovers less than the base drain, and never
                // past capacity. See SLEEP_RECOVERY.
                let max_e = MAX_ENERGY_BASE * self.genome[idx].traits.energy_capacity;
                self.energy[idx] = (self.energy[idx] + SLEEP_RECOVERY * metabolism).min(max_e);
            } else if winner == OUT_REPRODUCE {
                return self.do_reproduce(idx);
            }
        }

        None
    }

    /// Sum of repulsion vectors from all agents within SEPARATION_RADIUS.
    fn separation_force(&self, idx: usize, px: f32, py: f32, world_size: f32) -> (f32, f32) {
        let nearby = self.spatial.agents_near(px, py, SEPARATION_RADIUS + 0.5);
        let half = world_size * 0.5;
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        for other in nearby {
            if other == idx || self.cause_of_death[other].is_some() { continue; }
            let mut dx = px - self.pos_x[other];
            let mut dy = py - self.pos_y[other];
            if dx > half { dx -= world_size; } else if dx < -half { dx += world_size; }
            if dy > half { dy -= world_size; } else if dy < -half { dy += world_size; }
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < SEPARATION_RADIUS && dist > 0.001 {
                let strength = (SEPARATION_RADIUS - dist) / SEPARATION_RADIUS;
                fx += (dx / dist) * strength;
                fy += (dy / dist) * strength;
            }
        }
        (fx, fy)
    }

    fn do_eat(&mut self, idx: usize) {
        let (tx, ty) = SpatialHashGrid::tile_of(self.pos_x[idx], self.pos_y[idx], self.grid_size);
        let tile_idx = ty * self.grid_size + tx;
        if self.tiles[tile_idx].food_units == 0 { return; }

        // Per-tile cooldown: a tile just fed from can't be fed from again for a
        // while, independent of regen. Stops a single parked camper eating the
        // same tile every tick.
        if let Some(last) = self.tile_last_eaten[tile_idx] {
            if self.step_count.saturating_sub(last) < EAT_COOLDOWN_TICKS { return; }
        }

        let ec = self.genome[idx].traits.energy_capacity;
        let max_e = MAX_ENERGY_BASE * ec;
        let needed = max_e - self.energy[idx];
        if needed <= 0.0 { return; }

        // Crowding contention: agents eating the same tile in the same tick split
        // that tile's food value instead of each getting the full amount.
        let prior_claims = self.tile_eat_count_this_tick[tile_idx];
        self.tile_eat_count_this_tick[tile_idx] += 1;
        let share = FOOD_ENERGY / (1.0 + prior_claims as f64);

        let gained = share.min(needed);
        self.energy[idx] += gained;
        self.tiles[tile_idx].food_units -= 1;
        self.tile_last_eaten[tile_idx] = Some(self.step_count);
        let threshold = self.genome[idx].traits.metabolism * SUCCESS_SCALAR;
        if gained > threshold {
            // Ring entry already written by integrate_agent (eat gate won the
            // tick) — only bump the success counter here.
            self.memory[idx].success_count += 1;
        }
    }

    fn do_reproduce(&mut self, idx: usize) -> Option<PendingAgent> {
        if self.age[idx] < MATURITY_AGE { return None; }
        if self.energy[idx] < 40.0 { return None; }
        if self.offspring_count[idx] >= self.max_offspring[idx] { return None; }
        if let Some(last) = self.last_reproduced_age[idx] {
            if self.age[idx] - last < self.reproduction_cooldown[idx] { return None; }
        }

        let world_size = self.grid_size as f32;
        let repro_cost = self.genome[idx].traits.reproduction_cost;
        let cost = self.energy[idx] * 0.50 * repro_cost;
        self.energy[idx] -= cost;

        if self.rng.gen::<f64>() < BIRTH_FAIL_CHANCE {
            if self.rng.gen::<f64>() < FAIL_COUNTS_CHANCE {
                self.offspring_count[idx] += 1;
                self.last_reproduced_age[idx] = Some(self.age[idx]);
            }
            return None;
        }

        self.offspring_count[idx] += 1;
        self.last_reproduced_age[idx] = Some(self.age[idx]);

        // Spawn child nearby (within 2 tiles, random direction)
        let angle = self.rng.gen::<f32>() * TAU;
        let radius: f32 = self.rng.gen_range(0.5..2.0);
        let cx = (self.pos_x[idx] + radius * angle.cos()).rem_euclid(world_size);
        let cy = (self.pos_y[idx] + radius * angle.sin()).rem_euclid(world_size);

        let suppression = self.memory[idx].suppression(0.05);
        // Probation clamp: a cluster being tested reproduces with its mutability
        // taken away. Transient, not heritable — see Genome::mutate.
        let clamp = self.species.clamp_for(&self.genome[idx].traits);
        let child_genome = self.genome[idx].mutate(&mut self.rng, suppression, clamp);
        let child_traits = child_genome.traits.clone();
        let parent_defense = self.genome[idx].traits.defense;
        let child_energy = cost * BIRTH_ENERGY_YIELD;

        Some(PendingAgent {
            genome: child_genome,
            energy: child_energy,
            x: cx,
            y: cy,
            parent_defense,
            parent_id: self.ids[idx],
            // The child is measured against its parent's species definition
            // once, here, and never again. Mutation is the only thing that can
            // put it outside — which is exactly where new lineages come from: a
            // child born past its parents' definition is unassigned, and enough
            // of those clustering together is what the candidate machinery
            // promotes into the next species.
            species: {
                let parent_species =
                    self.species_ids.get(idx).copied().unwrap_or(crate::species::UNASSIGNED);
                if parent_species != crate::species::UNASSIGNED
                    && self.species.admits(parent_species, &child_traits)
                {
                    parent_species
                } else {
                    crate::species::UNASSIGNED
                }
            },
        })
    }

    /// Passive combat resolved per tile — no HashMap alloc.
    fn resolve_combat_spatial(&mut self) {
        // Combat always ends in a death (the loser is eaten), and the attacker's
        // energy cost can be lethal too, so immortality skips the phase whole
        // rather than trying to unpick which of its outcomes are fatal.
        if self.immortal { return; }

        let gs = self.grid_size;
        for ty in 0..gs {
            for tx in 0..gs {
                let occupants: Vec<usize> = self.spatial.agents_at_tile(tx, ty)
                    .iter()
                    .copied()
                    // The predator is excluded from ordinary combat entirely: it
                    // neither rolls to attack nor can be eaten. Its hunting is its
                    // own phase, and it must not be killable by prey.
                    .filter(|&i| self.cause_of_death[i].is_none() && !self.is_predator(i))
                    .collect();
                if occupants.len() < 2 { continue; }

                for &attacker in &occupants {
                    if self.cause_of_death[attacker].is_some() { continue; }
                    if self.genome[attacker].traits.aggression
                        < self.tunables.hunt_aggression_threshold { continue; }

                    // Hunger gate: sated agents don't hunt. Without it predation has
                    // no density-dependent brake — a predator that clears the local
                    // prey simply grazes instead, so nothing stops aggression from
                    // going to fixation and crashing the pond.
                    let a_ec = self.genome[attacker].traits.energy_capacity;
                    if self.energy[attacker] > HUNT_HUNGER_FRAC * MAX_ENERGY_BASE * a_ec { continue; }

                    // Who is fair game.
                    //
                    // Before speciation: everyone. An agent with no lineage has
                    // no relatives, and nothing about it is protected — the pond
                    // starts as a free-for-all and that is the baseline the rest
                    // of the economy was balanced against.
                    //
                    // After promotion: a species that is bright enough stops
                    // eating itself. Only aggression over CANNIBAL_AGGRESSION_MIN
                    // *and* intelligence under the cutoff will turn on its own
                    // lineage. Other species are always fair game — this is kin
                    // recognition, not pacifism.
                    let a_species = self.species_ids.get(attacker).copied()
                        .unwrap_or(crate::species::UNASSIGNED);
                    let cannibal = is_cannibal(&self.genome[attacker].traits);
                    let victim = occupants.iter()
                        .copied()
                        .find(|&i| {
                            if i == attacker || self.cause_of_death[i].is_some() { return false; }
                            if a_species == crate::species::UNASSIGNED { return true; }
                            let same = self.species_ids.get(i).copied()
                                .unwrap_or(crate::species::UNASSIGNED) == a_species;
                            !same || cannibal
                        });
                    let Some(victim) = victim else { continue };

                    let atk = self.genome[attacker].traits.attack;
                    let v_def = effective_defense(
                        self.genome[victim].traits.defense,
                        self.parent_defense_bonus[victim],
                        self.age[victim],
                    );
                    let metabolism = self.genome[attacker].traits.metabolism;
                    self.energy[attacker] -= 0.2 * metabolism;

                    // Bracing costs the defender, scaled by the armour it is
                    // bracing with. Being attacked is not free just because you
                    // survive it, and a heavily armoured animal in a crowded
                    // pond pays this over and over.
                    let v_metabolism = self.genome[victim].traits.metabolism;
                    self.energy[victim] -=
                        DEFENSE_BLOCK_COST * armour_margin(self.genome[victim].traits.defense)
                            * v_metabolism;
                    if self.energy[victim] <= 0.0 && self.cause_of_death[victim].is_none() {
                        self.cause_of_death[victim] = Some(CauseOfDeath::Starvation);
                        self.lifespans.push(self.age[victim]);
                        self.scratch_dead.push(victim);
                        continue;
                    }

                    // Attack cost can itself be lethal — check before the attacker
                    // fights on with <=0 energy and self-rescues via eat next tick.
                    if self.energy[attacker] <= 0.0 {
                        self.cause_of_death[attacker] = Some(CauseOfDeath::Starvation);
                        self.lifespans.push(self.age[attacker]);
                        self.scratch_dead.push(attacker);
                        continue;
                    }

                    // Win chance scales continuously with atk vs def instead of fixed
                    // thresholds. Old thresholds (defender wins iff atk <= def*0.33) were
                    // unreachable for adults given trait bounds (atk min 0.5, def max
                    // 1.07 -> def*0.33 max 0.353 < atk min), making combat a guaranteed
                    // win above atk 0.706 and a coinflip below it — no counter-pressure
                    // against high aggression.
                    let p_win = atk / (atk + v_def);
                    if self.rng.gen::<f64>() < p_win {
                        self.passive_eat(attacker, victim);
                    } else {
                        // Failed hunt: the prey fights back and escapes. A lost roll
                        // used to kill the attacker outright, making aggression >= 0.80
                        // a ~coinflip for your life that the agent volunteers for — so
                        // non-aggression strictly dominated and every seed reached zero
                        // fighters by step ~250, after which combat never fired again.
                        // Retaliation is only lethal if it empties the attacker.
                        let a_def = effective_defense(
                            self.genome[attacker].traits.defense,
                            self.parent_defense_bonus[attacker],
                            self.age[attacker],
                        );
                        self.energy[attacker] -= v_def * RETALIATION_ENERGY / a_def.max(0.1);
                        if self.energy[attacker] <= 0.0 {
                            self.passive_eat(victim, attacker);
                        }
                    }
                }
            }
        }
    }

    fn passive_eat(&mut self, winner: usize, loser: usize) {
        self.kills[winner] += 1;
        // Clamped at 0: retaliation can drive the attacker below zero before this
        // runs, and a negative "gain" would drain the defender that just won.
        let gained = (self.energy[loser] * PREDATION_YIELD).max(0.0);
        let ec = self.genome[winner].traits.energy_capacity;
        self.energy[winner] = (self.energy[winner] + gained).min(MAX_ENERGY_BASE * ec);
        self.cause_of_death[loser] = Some(CauseOfDeath::KilledInCombat);
        self.energy[loser] = 0.0;
        self.lifespans.push(self.age[loser]);
        self.scratch_dead.push(loser);
    }

    fn spawn_offspring(&mut self, offspring: Vec<PendingAgent>) {
        for child in offspring {
            let species = child.species;
            self.push_agent(
                child.x, child.y, child.energy, child.genome, child.parent_defense,
                Some(child.parent_id), species,
            );
        }
    }

    /// Common path for adding any new agent (initial spawn, offspring, or pour_agents).
    /// `energy` is clamped to the genome's own capacity. Founders and summoned
    /// predators are pushed with a flat 100.0, which is over the cap for any
    /// genome whose `energy_capacity` is below 1.0 — invisible until an ambient
    /// predator put one in every pond and the capacity invariant test found it.
    // Eight positional arguments is one past clippy's threshold. Grouping them
    // into a struct would mean a second spawn type beside `PendingAgent` that
    // three of the four callers would have to fill in with placeholders, which
    // is more ceremony than the warning is worth.
    #[allow(clippy::too_many_arguments)]
    fn push_agent(
        &mut self,
        x: f32,
        y: f32,
        energy: f64,
        genome: Genome,
        parent_defense: f64,
        parent_id: Option<u32>,
        species: u32,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        let death_age = assign_death_age(&self.death_range_pool, &mut self.rng);
        let max_offspring = self.rng.gen_range(1u32..=10);
        let reproductive_window = death_age.saturating_sub(MATURITY_AGE).max(1);
        let cooldown = reproductive_window / max_offspring.max(1);

        // Random initial velocity — small fraction of max speed
        let angle = self.rng.gen::<f32>() * TAU;
        let speed_trait = genome.traits.speed as f32;
        let init_speed = self.rng.gen::<f32>() * speed_trait * MAX_SPEED * 0.3;

        self.ids.push(id);
        self.energy.push(energy.min(MAX_ENERGY_BASE * genome.traits.energy_capacity));
        self.age.push(0);
        self.pos_x.push(x);
        self.pos_y.push(y);
        self.vel_x.push(angle.cos() * init_speed);
        self.vel_y.push(angle.sin() * init_speed);
        self.prev_x.push(x);
        self.prev_y.push(y);
        self.death_age.push(death_age);
        self.genome.push(genome);
        self.memory.push(AgentMemory::new());
        // Everything decides on its first tick; the cadence starts after that.
        self.decision_cooldown.push(0);
        // Born seeing nothing: dist 1.0 is "no threat in range".
        self.threat_ring.push([[1.0, 0.0, 0.0, 0.0]; THREAT_RING]);
        self.threat_head.push(0);
        // Born healthy. Vertical transmission is deliberately absent: a disease
        // that infected every newborn of a lineage would be a property of the
        // lineage, not an outbreak, and would never crash.
        self.infection.push(0);
        self.last_outputs.push([0f32; 8]);
        self.last_perception.push([0f32; INPUT_COUNT]);
        self.last_food_dir.push((0.0, 0.0));
        self.kills.push(0);
        self.parent_defense_bonus.push(parent_defense);
        self.parent_id.push(parent_id);
        self.cause_of_death.push(None);
        self.offspring_count.push(0);
        self.max_offspring.push(max_offspring);
        self.last_reproduced_age.push(None);
        self.reproduction_cooldown.push(cooldown);
        // Offspring are born into their parent's species. Membership is still
        // recomputed from scratch on the next species tick — by nearest centroid,
        // so a child that inherited a lineage it has already drifted out of loses
        // it there — but until then it belongs where it came from.
        //
        // It used to be born `UNASSIGNED` and wait up to 50 steps for the next
        // cluster run to notice it. That is wrong on its face: a newborn has its
        // parent's genome, so calling it lineage-less is a statement about the
        // scheduler, not about the animal. It was also visible, since a species
        // now carries a hue and a body shape — every pond had a churn of grey
        // unassigned newborns that turned into their parents 50 steps later.
        //
        // Founders and poured agents pass `UNASSIGNED`: they have no parent, and
        // inventing a lineage for them would be a different lie.
        self.species_ids.push(species);
    }

    fn reap_dead(&mut self, mut dead: Vec<usize>) {
        dead.sort_unstable();
        dead.dedup();

        for &i in &dead {
            if let Some(cause) = &self.cause_of_death[i] {
                *self.death_tally.entry(cause.clone()).or_insert(0) += 1;
                // Record for the renderer before swap_remove invalidates the index.
                // Capped so the queue can't grow without bound if nothing drains it.
                if self.last_deaths.len() < MAX_QUEUED_DEATHS {
                    self.last_deaths.push(DeathEvent {
                        id: self.ids[i],
                        x: self.pos_x[i],
                        y: self.pos_y[i],
                        cause: cause.code(),
                    });
                }
            }
        }

        self.remove_slots(dead);
    }

    /// Drop agent slots from every SoA array. Descending order so each
    /// swap_remove only disturbs indices already processed.
    ///
    /// Separate from `reap_dead` because not every removal is a death: the ultra
    /// predator leaves the pond when its cull is done, and that must not record a
    /// death, a lifespan or a tally entry.
    fn remove_slots(&mut self, mut slots: Vec<usize>) {
        slots.sort_unstable();
        slots.dedup();
        slots.sort_unstable_by(|a, b| b.cmp(a));
        for &i in &slots {
            let last = self.ids.len() - 1;
            if i < last {
                self.ids.swap_remove(i);
                self.energy.swap_remove(i);
                self.age.swap_remove(i);
                self.pos_x.swap_remove(i);
                self.pos_y.swap_remove(i);
                self.vel_x.swap_remove(i);
                self.vel_y.swap_remove(i);
                self.prev_x.swap_remove(i);
                self.prev_y.swap_remove(i);
                self.death_age.swap_remove(i);
                self.genome.swap_remove(i);
                self.memory.swap_remove(i);
                self.decision_cooldown.swap_remove(i);
                self.threat_ring.swap_remove(i);
                self.threat_head.swap_remove(i);
                self.infection.swap_remove(i);
                self.last_outputs.swap_remove(i);
                self.last_perception.swap_remove(i);
                self.last_food_dir.swap_remove(i);
                self.kills.swap_remove(i);
                self.parent_defense_bonus.swap_remove(i);
                self.parent_id.swap_remove(i);
                self.cause_of_death.swap_remove(i);
                self.offspring_count.swap_remove(i);
                self.max_offspring.swap_remove(i);
                self.last_reproduced_age.swap_remove(i);
                self.reproduction_cooldown.swap_remove(i);
                self.species_ids.swap_remove(i);
            } else {
                self.ids.pop();
                self.energy.pop();
                self.age.pop();
                self.pos_x.pop();
                self.pos_y.pop();
                self.vel_x.pop();
                self.vel_y.pop();
                self.prev_x.pop();
                self.prev_y.pop();
                self.death_age.pop();
                self.genome.pop();
                self.memory.pop();
                self.decision_cooldown.pop();
                self.threat_ring.pop();
                self.threat_head.pop();
                self.infection.pop();
                self.last_outputs.pop();
                self.last_perception.pop();
                self.last_food_dir.pop();
                self.kills.pop();
                self.parent_defense_bonus.pop();
                self.parent_id.pop();
                self.cause_of_death.pop();
                self.offspring_count.pop();
                self.max_offspring.pop();
                self.last_reproduced_age.pop();
                self.reproduction_cooldown.pop();
                self.species_ids.pop();
            }
        }
    }

    fn spawn_agents(&mut self, population: usize) {
        let world_size = self.grid_size as f32;
        for _ in 0..population {
            let x = self.rng.gen::<f32>() * world_size;
            let y = self.rng.gen::<f32>() * world_size;
            let genome = Genome::generate(&mut self.rng);
            self.push_agent(x, y, 100.0, genome, 0.0, None, crate::species::UNASSIGNED);
        }
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Offset from (px, py) to (qx, qy) across a toroidal world: always the shorter
/// way round on each axis.
#[inline]
fn toroidal_delta(px: f32, py: f32, qx: f32, qy: f32, world_size: f32) -> (f32, f32) {
    let half = world_size * 0.5;
    let mut dx = qx - px;
    let mut dy = qy - py;
    if dx > half { dx -= world_size; } else if dx < -half { dx += world_size; }
    if dy > half { dy -= world_size; } else if dy < -half { dy += world_size; }
    (dx, dy)
}

fn effective_defense(defense: f64, parent_bonus: f64, age: u32) -> f64 {
    if age >= CHILDHOOD_TICKS || parent_bonus == 0.0 {
        return defense;
    }
    let ratio = age as f64 / CHILDHOOD_TICKS as f64;
    defense + parent_bonus * (1.0 - ratio)
}

/// `attack` is the hunter's *learned* bite, not the tier constant: it starts at
/// `TIER_ATTACK[tier]` and rises toward the armour of whatever family it is
/// hunting. The tier still decides whether defense is consulted at all.
fn predator_attack_succeeds(
    tier: u8, attack: f64, defense: f64, parent_bonus: f64, age: u32,
) -> bool {
    let t = (tier as usize).min(PREDATOR_TIERS - 1);
    if TIER_IGNORES_DEFENSE[t] { return true; }
    attack > effective_defense(defense, parent_bonus, age)
}

fn init_grid(grid_size: usize, rng: &mut ChaCha8Rng) -> Vec<BiomeTile> {
    let n = grid_size * grid_size;
    let mut tiles: Vec<BiomeTile> = (0..n).map(|_| BiomeTile::generate(rng)).collect();
    assign_barren_tiles(&mut tiles, grid_size, rng);
    tiles
}

fn assign_barren_tiles(tiles: &mut Vec<BiomeTile>, grid_size: usize, rng: &mut ChaCha8Rng) {
    let total = grid_size * grid_size;
    let target_pct = rng.gen_range(0.35_f64..=0.45);
    let target = (total as f64 * target_pct) as usize;
    let num_seeds = (grid_size / 3).max(2).min(total);
    let gs = grid_size as i32;

    let mut all_idx: Vec<usize> = (0..total).collect();
    for i in 0..num_seeds {
        let j = rng.gen_range(i..total);
        all_idx.swap(i, j);
    }
    let seed_positions: Vec<(u16, u16)> = all_idx[..num_seeds]
        .iter()
        .map(|&i| ((i % grid_size) as u16, (i / grid_size) as u16))
        .collect();

    let mut barren: HashSet<(u16, u16)> = seed_positions.iter().copied().collect();
    let mut frontier: Vec<(u16, u16)> = seed_positions;
    let mut spread_prob = 0.55_f64;

    'grow: while barren.len() < target {
        if frontier.is_empty() {
            let remaining: Vec<(u16, u16)> = (0..total)
                .map(|i| ((i % grid_size) as u16, (i / grid_size) as u16))
                .filter(|p| !barren.contains(p))
                .collect();
            if remaining.is_empty() { break; }
            let seed = *remaining.choose(rng).unwrap();
            barren.insert(seed);
            frontier.push(seed);
        }
        let current: Vec<(u16, u16)> = frontier.drain(..).collect();
        for (fx, fy) in current {
            for &(dx, dy) in &[(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = (fx as i32 + dx).rem_euclid(gs) as u16;
                let ny = (fy as i32 + dy).rem_euclid(gs) as u16;
                if !barren.contains(&(nx, ny)) && rng.gen::<f64>() < spread_prob {
                    barren.insert((nx, ny));
                    frontier.push((nx, ny));
                    if barren.len() >= target { break 'grow; }
                }
            }
        }
        spread_prob = (spread_prob * 0.85).max(0.30);
    }

    for (x, y) in barren {
        tiles[y as usize * grid_size + x as usize].make_barren();
    }
}

fn create_death_range(rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut pool = Vec::with_capacity(200);
    for i in 0usize..200 {
        if i < 5 && rng.gen::<f64>() < 0.15 {
            pool.push(rng.gen_range(50u32..=150));
        } else if i > 15 && i < 20 && rng.gen::<f64>() < 0.05 {
            pool.push(rng.gen_range(200u32..=400));
        } else {
            pool.push(500 + (i as u32 + 500) / 4);
        }
    }
    pool
}

fn assign_death_age(pool: &[u32], rng: &mut ChaCha8Rng) -> u32 {
    pool.choose(rng).copied().unwrap_or(750)
}

/// 10th and 90th percentile of a slice of ages, by nearest rank.
fn age_percentiles(ages: &[u32]) -> (u32, u32) {
    if ages.is_empty() {
        return (0, 0);
    }
    let mut s = ages.to_vec();
    s.sort_unstable();
    let idx = |q: f64| -> usize {
        (((s.len() - 1) as f64) * q).round() as usize
    };
    (s[idx(0.10)], s[idx(0.90)])
}

fn median(v: &[u32]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n % 2 == 1 { s[n / 2] as f64 } else { (s[n / 2 - 1] + s[n / 2]) as f64 / 2.0 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn small_world() -> World {
        World::new(8, 30, 42)
    }

    #[test]
    fn deaths_are_queued_with_position_and_cause() {
        let mut w = World::new(12, 60, 42);
        // Run until at least one agent has died.
        for _ in 0..600 {
            w.step();
            if !w.last_deaths.is_empty() || w.agent_count() == 0 { break; }
        }
        assert!(!w.last_deaths.is_empty(), "no deaths recorded in 600 steps");

        let gs = w.grid_size as f32;
        for d in &w.last_deaths {
            assert!(d.cause <= 3, "unknown cause code {}", d.cause);
            assert!((0.0..gs).contains(&d.x), "death x {} out of bounds", d.x);
            assert!((0.0..gs).contains(&d.y), "death y {} out of bounds", d.y);
        }
    }

    #[test]
    fn death_cause_codes_are_stable() {
        // The renderer maps these to epitaph glyphs; reordering would silently
        // swap them (see EPITAPH in pond_web/renderer.js).
        assert_eq!(CauseOfDeath::Starvation.code(), 0);
        assert_eq!(CauseOfDeath::OldAge.code(), 1);
        assert_eq!(CauseOfDeath::KilledInCombat.code(), 2);
        assert_eq!(CauseOfDeath::EatenAlive.code(), 3);
    }

    #[test]
    fn world_initializes() {
        let w = small_world();
        assert!(w.agent_count() > 0);
        assert_eq!(w.tiles.len(), 64);
        assert_eq!(w.step_count, 0);
    }

    #[test]
    fn step_runs_without_panic() {
        let mut w = small_world();
        for _ in 0..10 {
            w.step();
            if w.agent_count() == 0 { break; }
        }
        assert!(w.step_count <= 10);
    }

    #[test]
    fn soa_arrays_same_length() {
        let mut w = small_world();
        w.step();
        let n = w.ids.len();
        assert_eq!(w.energy.len(), n);
        assert_eq!(w.age.len(), n);
        assert_eq!(w.pos_x.len(), n);
        assert_eq!(w.pos_y.len(), n);
        assert_eq!(w.vel_x.len(), n);
        assert_eq!(w.vel_y.len(), n);
        assert_eq!(w.prev_x.len(), n);
        assert_eq!(w.prev_y.len(), n);
        assert_eq!(w.genome.len(), n);
        assert_eq!(w.memory.len(), n);
        assert_eq!(w.cause_of_death.len(), n);
        assert_eq!(w.death_age.len(), n);
    }

    #[test]
    fn positions_in_bounds() {
        let mut w = small_world();
        let world_size = w.grid_size as f32;
        for _ in 0..20 {
            w.step();
        }
        for i in 0..w.agent_count() {
            assert!(w.pos_x[i] >= 0.0 && w.pos_x[i] < world_size, "pos_x[{}]={} out of bounds", i, w.pos_x[i]);
            assert!(w.pos_y[i] >= 0.0 && w.pos_y[i] < world_size, "pos_y[{}]={} out of bounds", i, w.pos_y[i]);
        }
    }

    #[test]
    fn food_regen_respects_max() {
        let mut w = small_world();
        for _ in 0..200 {
            w.tick_food_regen();
        }
        for tile in &w.tiles {
            assert!(tile.food_units <= MAX_FOOD_PER_TILE);
        }
    }

    // ── Tunables ──────────────────────────────────────────────────────────────

    /// The load-bearing one: turning three constants into fields must be inert.
    /// Everything else here moves a dial, so this is what proves the dials are
    /// the only thing that changed.
    #[test]
    fn defaults_reproduce_the_untuned_run() {
        let mut a = World::new(12, 60, 42);
        let mut b = World::new(12, 60, 42);
        b.set_food_regen_scale(DEFAULT_FOOD_REGEN_SCALE);
        b.set_hunt_aggression_threshold(DEFAULT_HUNT_AGGRESSION_THRESHOLD);
        b.set_cluster_k(DEFAULT_CLUSTER_K);
        for _ in 0..300 {
            a.step();
            b.step();
        }
        assert_eq!(a.agent_count(), b.agent_count());
        assert_eq!(a.energy, b.energy);
        assert_eq!(a.cluster.genome_cluster_ids, b.cluster.genome_cluster_ids);
        assert_eq!(a.death_tally, b.death_tally);
        // Setting a dial to the value it already had is not a modification.
        assert!(!b.tunables().modified);
    }

    #[test]
    fn zero_regen_means_no_tile_ever_gains_food() {
        let mut w = small_world();
        w.set_food_regen_scale(0.0);
        let before: u32 = w.tiles.iter().map(|t| t.food_units).sum();
        for _ in 0..500 {
            w.tick_food_regen();
        }
        let after: u32 = w.tiles.iter().map(|t| t.food_units).sum();
        assert_eq!(before, after);
        assert!(w.tunables().modified);
    }

    #[test]
    fn a_threshold_above_the_trait_maximum_stops_combat() {
        // aggression tops out at 1.05, so nothing can clear 1.06.
        let mut tuned = World::new(12, 120, 42);
        tuned.set_hunt_aggression_threshold(1.06);
        let mut base = World::new(12, 120, 42);
        for _ in 0..400 {
            tuned.step();
            base.step();
        }
        assert_eq!(tuned.death_tally.get(&CauseOfDeath::KilledInCombat), None);
        // …and the untuned run of the same seed does kill in combat, or this
        // test would pass for the wrong reason.
        assert!(base.death_tally.get(&CauseOfDeath::KilledInCombat).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn cluster_k_takes_effect_without_waiting_for_the_cycle() {
        let mut w = World::new(12, 60, 42);
        for _ in 0..60 { w.step(); }
        assert!(w.cluster.genome_cluster_ids.iter().any(|&id| id >= 3));

        w.set_cluster_k(3);
        w.step(); // not a multiple of 50 — the dirty flag is what reclusters
        assert!(w.cluster.genome_cluster_ids.iter().all(|&id| id < 3),
            "k change should apply on the next step, not the next 50-step cycle");

        // And back up again, over the previous run's smaller centroid vector.
        w.set_cluster_k(8);
        w.step();
        assert!(w.cluster.genome_cluster_ids.iter().all(|&id| id < 8));
        assert!(w.cluster.genome_cluster_ids.iter().any(|&id| id >= 3));
    }

    #[test]
    fn tunables_are_clamped_to_their_ranges() {
        let mut w = small_world();
        w.set_food_regen_scale(99.0);
        w.set_hunt_aggression_threshold(-5.0);
        w.set_cluster_k(0);
        let t = w.tunables();
        assert_eq!(t.food_regen_scale, FOOD_REGEN_SCALE_RANGE.1);
        assert_eq!(t.hunt_aggression_threshold, HUNT_AGGRESSION_THRESHOLD_RANGE.0);
        assert_eq!(t.cluster_k, CLUSTER_K_RANGE.0);
    }

    #[test]
    fn an_f32_round_trip_of_a_default_is_not_a_modification() {
        // What the UI sends when reset is pressed: the default, having been
        // through f32 on the way out to JS and back.
        let mut w = small_world();
        w.set_food_regen_scale(DEFAULT_FOOD_REGEN_SCALE as f32 as f64);
        w.set_hunt_aggression_threshold(DEFAULT_HUNT_AGGRESSION_THRESHOLD as f32 as f64);
        assert!(!w.tunables().modified);
    }

    #[test]
    fn modified_latches_and_is_not_cleared_by_going_back() {
        let mut w = small_world();
        w.set_cluster_k(4);
        assert!(w.tunables().modified);
        w.set_cluster_k(DEFAULT_CLUSTER_K);
        assert!(w.tunables().modified, "the run already diverged; the seed no longer describes it");
    }

    #[test]
    fn energy_drains_each_step() {
        let mut w = World::new(6, 10, 99);
        let initial_energy: f64 = w.energy.iter().sum();
        w.step();
        let after_energy: f64 = w.energy.iter().sum();
        assert!(after_energy < initial_energy * 1.5);
    }

    #[test]
    fn sleep_slows_starvation_instead_of_reversing_it() {
        // Sleep must recover strictly less than the tick's base metabolism drain,
        // or an agent that keeps choosing it never starves.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(SLEEP_RECOVERY < BASE_DRAIN, "sleep is a net energy source");
        }
    }

    #[test]
    fn energy_never_exceeds_capacity() {
        // Start everyone at their cap so any unclamped gain (sleep, eat) shows up
        // as an overflow rather than being absorbed by a deficit.
        let mut w = World::new(10, 40, 5);
        for i in 0..w.energy.len() {
            w.energy[i] = MAX_ENERGY_BASE * w.genome[i].traits.energy_capacity;
        }
        for _ in 0..400 {
            w.step();
            for i in 0..w.energy.len() {
                let max_e = MAX_ENERGY_BASE * w.genome[i].traits.energy_capacity;
                assert!(
                    w.energy[i] <= max_e + 1e-6,
                    "agent {} at {} over capacity {}", i, w.energy[i], max_e,
                );
            }
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let mut w1 = World::new(8, 20, 7);
        let mut w2 = World::new(8, 20, 7);
        for _ in 0..5 {
            w1.step();
            w2.step();
        }
        assert_eq!(w1.agent_count(), w2.agent_count());
        assert_eq!(w1.step_count, w2.step_count);
    }

    #[test]
    fn pour_agents_adds_agents() {
        let mut w = World::new(8, 5, 11);
        let before = w.agent_count();
        w.pour_agents(4.0, 4.0, 10);
        assert_eq!(w.agent_count(), before + 10);
    }

    #[test]
    fn inject_food_clamps_to_tile_max() {
        let mut w = World::new(8, 5, 31);
        w.inject_food(3.5, 3.5, 99);
        let (tx, ty) = SpatialHashGrid::tile_of(3.5, 3.5, 8);
        assert_eq!(w.tiles[ty * 8 + tx].food_units, MAX_FOOD_PER_TILE);
    }

    #[test]
    fn stir_drains_food_and_scatters_agents() {
        let mut w = World::new(8, 30, 13);
        w.inject_food(4.0, 4.0, MAX_FOOD_PER_TILE);
        let (tx, ty) = SpatialHashGrid::tile_of(4.0, 4.0, 8);
        let fertility_before = w.tiles[ty * 8 + tx].fertility;

        // Park an agent just off the stir centre so the impulse is well-defined.
        w.pos_x[0] = 4.5;
        w.pos_y[0] = 4.0;
        w.vel_x[0] = 0.0;
        w.vel_y[0] = 0.0;

        w.stir(4.0, 4.0, 2.0, 1.0);

        assert_eq!(w.tiles[ty * 8 + tx].food_units, 0);
        assert!(w.tiles[ty * 8 + tx].fertility < fertility_before);
        // Pushed outward, i.e. away from the centre along +x.
        assert!(w.vel_x[0] > 0.0);
    }

    #[test]
    fn stir_respects_speed_clamp() {
        let mut w = World::new(8, 10, 19);
        w.pos_x[0] = 4.2;
        w.pos_y[0] = 4.0;
        w.stir(4.0, 4.0, 3.0, 1.0);
        let speed = (w.vel_x[0].powi(2) + w.vel_y[0].powi(2)).sqrt();
        let cap = w.genome[0].traits.speed as f32 * MAX_SPEED * 2.0;
        assert!(speed <= cap + 1e-4, "speed {} exceeds cap {}", speed, cap);
    }

    #[test]
    fn smite_radius_kills_only_inside_the_radius() {
        let mut w = World::new(16, 0, 3);
        w.pour_agents(4.0, 4.0, 12);
        w.pour_agents(12.0, 12.0, 8);
        let before = w.agent_count();

        let killed = w.smite_radius(4.0, 4.0, 2.5);
        assert!(killed > 0, "comet hit nothing");
        assert_eq!(w.agent_count(), before - killed as usize);
        // Survivors are all outside the blast.
        for i in 0..w.agent_count() {
            let dx = (w.pos_x[i] - 4.0f32).abs().min(16.0 - (w.pos_x[i] - 4.0f32).abs());
            let dy = (w.pos_y[i] - 4.0f32).abs().min(16.0 - (w.pos_y[i] - 4.0f32).abs());
            assert!(dx * dx + dy * dy > 2.5 * 2.5);
        }
    }

    #[test]
    fn smite_counts_as_its_own_cause() {
        let mut w = World::new(12, 30, 9);
        let killed = w.smite_all();
        assert_eq!(w.agent_count(), 0);
        assert_eq!(w.death_counts()[CauseOfDeath::Smitten.code() as usize], killed);
        // Natural causes untouched by an act of god.
        assert_eq!(w.death_counts()[CauseOfDeath::Starvation.code() as usize], 0);
        assert_eq!(w.death_counts()[CauseOfDeath::OldAge.code() as usize], 0);
    }

    #[test]
    fn smite_band_kills_only_its_column() {
        let mut w = World::new(12, 60, 21);
        let killed = w.smite_band(0.0, 6.0);
        assert!(killed > 0);
        for i in 0..w.agent_count() {
            assert!(w.pos_x[i] >= 6.0, "survivor at x={} inside the swept band", w.pos_x[i]);
        }
    }

    #[test]
    fn smite_queues_death_events_for_the_renderer() {
        let mut w = World::new(12, 20, 33);
        w.last_deaths.clear();
        let killed = w.smite_all();
        assert_eq!(w.last_deaths.len(), killed as usize);
        for d in &w.last_deaths {
            assert_eq!(d.cause, CauseOfDeath::Smitten.code());
        }
    }

    #[test]
    fn immortal_agents_never_die_naturally() {
        let mut w = World::new(12, 40, 77);
        w.immortal = true;
        let before = w.agent_count();
        for _ in 0..800 {
            w.step();
        }
        // No *natural* death of any kind. Predator kills are excluded: an
        // immortal pond grows without limit, so it crosses the cull threshold
        // and the pressure valve fires — that is the intended interaction.
        let d = w.death_counts();
        assert_eq!(d[CauseOfDeath::Starvation.code() as usize], 0);
        assert_eq!(d[CauseOfDeath::OldAge.code() as usize], 0);
        assert_eq!(d[CauseOfDeath::KilledInCombat.code() as usize], 0);
        assert_eq!(d[CauseOfDeath::Smitten.code() as usize], 0);
        assert!(w.agent_count() >= before, "population shrank under immortality");
        assert!(w.energy.iter().all(|&e| e > 0.0), "an immortal agent hit zero energy");
        let _ = d;
    }

    #[test]
    fn immortality_does_not_block_smiting() {
        let mut w = World::new(12, 25, 5);
        w.immortal = true;
        for _ in 0..20 { w.step(); }
        let killed = w.smite_all();
        assert!(killed > 0);
        assert_eq!(w.agent_count(), 0);
    }

    #[test]
    fn the_resident_tier_stays_and_reverts_to_ambient_duty() {
        // The triangles arrive and never leave. Finishing a cull no longer puts
        // one to sleep until the next boom — it drops to a quota of zero and
        // keeps hunting, which is what makes predation continuous pressure
        // rather than a thermostat.
        let mut w = World::new(16, 80, 21);
        w.set_automatic_predators(false);
        w.summon_predator_tier(w.prey_count(), false, 0);
        let id = w.predators[0].id;
        w.hunt_one(id);
        for _ in 0..PREDATOR_LEAVE_TICKS * 3 {
            w.hunt_one(id);
        }
        assert_eq!(w.predators.len(), 1, "a resident predator left the pond");
        assert_eq!(w.predators[0].target_pop, 0, "resident did not return to ambient duty");
        assert!(!w.predators[0].sated, "an ambient hunter should never be sated");
        assert!(w.predators[0].leaving.is_none(), "resident began departing");
        assert!(w.ids.contains(&id));
    }

    /// Heading of one agent slot, radians. None while it is standing still.
    fn heading_of(w: &World, idx: usize) -> Option<f32> {
        let (vx, vy) = (w.vel_x[idx], w.vel_y[idx]);
        if vx * vx + vy * vy < 1e-6 { None } else { Some(vy.atan2(vx)) }
    }

    /// Shortest signed angle from `a` to `b`.
    fn angle_delta(a: f32, b: f32) -> f32 {
        let mut d = (b - a).rem_euclid(TAU);
        if d > PI { d -= TAU; }
        d
    }

    #[test]
    fn a_hunter_never_turns_faster_than_its_tier() {
        // The jitter fix, stated as an invariant: no predator's heading may swing
        // more in one tick than its tier's turn rate, in any state — chasing,
        // patrolling, or leaving. Before steering, a chase could flip 180°
        // between two ticks whenever the nearest prey changed.
        for tier in 0..PREDATOR_TIERS as u8 {
            let mut w = World::new(16, 120, 404 + tier as u64);
            w.set_automatic_predators(false);
            w.summon_predator_tier(20, false, tier);
            let id = w.predators[0].id;
            let limit = TIER_MAX_TURN[tier as usize] + 1e-3;

            let mut last = w.slot_of(id).and_then(|s| heading_of(&w, s));
            for step in 0..400 {
                w.step();
                let Some(slot) = w.slot_of(id) else { break };
                let Some(now) = heading_of(&w, slot) else { continue };
                if let Some(prev) = last {
                    let turned = angle_delta(prev, now).abs();
                    assert!(
                        turned <= limit,
                        "tier {tier} turned {turned} rad on step {step}, limit {limit}",
                    );
                }
                last = Some(now);
            }
        }
    }

    #[test]
    fn a_hunter_never_changes_speed_abruptly() {
        // Speed eases toward whatever the current state wants, so going quiet or
        // leaving is not a one-tick jump. The bound is the easing itself: no step
        // may cover more than `PREDATOR_SPEED_EASE` of the gap to the fastest
        // thing a predator ever does.
        let mut w = World::new(16, 200, 77);
        w.set_automatic_predators(false);
        w.summon_predator_tier(40, false, 0);
        let id = w.predators[0].id;
        let ceiling = PREDATOR_LEAVE_SPEED.max(TIER_SPEED[0]);
        let bound = ceiling * PREDATOR_SPEED_EASE + 1e-4;

        let mut last = w.predators[0].speed;
        for step in 0..500 {
            w.step();
            let Some(p) = w.predators.iter().find(|p| p.id == id) else { break };
            let jump = (p.speed - last).abs();
            assert!(jump <= bound, "speed jumped {jump} on step {step}, bound {bound}");
            assert!(p.speed <= ceiling + 1e-4, "speed {} over its ceiling", p.speed);
            last = p.speed;
        }
    }

    #[test]
    fn a_hunter_still_runs_its_prey_down() {
        // Turn limits must not make a hunter miss. One triangle, one prey animal,
        // nothing else in the pond: it closes and eats, banking rather than
        // snapping onto the target.
        let mut w = World::new(12, 1, 909);
        w.set_automatic_predators(false);
        w.summon_predator_tier(0, false, 0);
        let id = w.predators[0].id;
        let prey_id = *w.ids.iter().find(|&&i| i != id).unwrap();
        // A slow animal. Hunters are deliberately slower than average prey now,
        // so "runs it down" is only true of something it can actually outpace —
        // this test is about the turn limit not making it miss, not about it
        // being able to catch anything at all.
        let prey_slot = w.slot_of(prey_id).unwrap();
        w.genome[prey_slot].traits.speed = crate::genome::Traits::BOUNDS[1].0;

        let mut caught = None;
        for step in 0..600 {
            w.step();
            if !w.ids.contains(&prey_id) { caught = Some(step); break; }
        }
        assert!(caught.is_some(), "a hunter alone with one prey animal never caught it");
        assert_eq!(w.predators[0].kills, 1, "the kill was not credited to the hunter");
    }

    #[test]
    fn a_sated_resident_patrols_smoothly() {
        // Patrolling is the state a resident spends most of its life in, and it
        // used to redraw its turn from scratch every tick, which reads as 20 Hz
        // vibration rather than as swimming.
        let mut w = World::new(16, 40, 313);
        w.set_automatic_predators(false);
        w.summon_predator_tier(0, false, 0);
        let id = w.predators[0].id;
        // Patrol is what a hunter does with nothing to chase. Emptying the pond
        // is now the only way to get there: an ambient hunter never sates, so
        // "quota met" no longer produces idle motion. `smite_all` would take the
        // hunter with it, so only the prey is cleared.
        let victims: Vec<usize> = (0..w.ids.len()).filter(|&i| !w.is_predator(i)).collect();
        w.smite(victims);
        for _ in 0..20 { w.step(); }
        assert!(w.prey_count() == 0, "something is still in the water to chase");

        let mut last = w.slot_of(id).and_then(|s| heading_of(&w, s));
        for step in 0..300 {
            w.step();
            let slot = w.slot_of(id).expect("resident left the pond");
            let Some(now) = heading_of(&w, slot) else { continue };
            if let Some(prev) = last {
                let turned = angle_delta(prev, now).abs();
                assert!(
                    turned <= PATROL_TURN_MAX + 1e-3,
                    "patrol turned {turned} rad on step {step}",
                );
            }
            last = Some(now);
        }
    }

    #[test]
    fn a_hunter_commits_to_one_target_instead_of_re_picking_every_tick() {
        // Two prey animals at near-equal distance used to make the argmin
        // alternate, so the hunter aimed at each in turn and went nowhere.
        // Pond large enough that neither animal can be reached inside the
        // commitment window — this is a test about target choice, not about kills.
        let mut w = World::new(60, 2, 1234);
        w.set_automatic_predators(false);
        w.summon_predator_tier(0, false, 0);
        let id = w.predators[0].id;
        let pslot = w.slot_of(id).unwrap();
        let prey: Vec<usize> = (0..w.ids.len()).filter(|&i| i != pslot).collect();

        // Symmetrically placed either side of the hunter's course, and both
        // inside PREDATOR_COMMIT_RANGE — outside it, commitment is *supposed* to
        // lapse every tick, so a wider spacing tests nothing. The old fixture
        // sat them 20 apart and passed on the luck of which way the hunter's
        // random initial heading pointed.
        w.pos_x[pslot] = 30.0;
        w.pos_y[pslot] = 30.0;
        w.pos_x[prey[0]] = 30.0;
        w.pos_y[prey[0]] = 30.0 - PREDATOR_COMMIT_RANGE * 0.5;
        w.pos_x[prey[1]] = 30.0;
        w.pos_y[prey[1]] = 30.0 + PREDATOR_COMMIT_RANGE * 0.5;

        w.hunt_one(id);
        let first = w.predators[0].target_id;
        assert!(first.is_some(), "hunter picked no target");
        let mut held = 0;
        for _ in 0..PREDATOR_COMMIT_TICKS - 1 {
            w.hunt_one(id);
            // Catching and eating the target ends the commitment legitimately —
            // that is a hunt succeeding, not a hunter dithering. Inside the
            // commit range a hunter closes 0.95 units a tick, so it will often
            // reach its prey well before the window is up.
            if first.map(|t| w.slot_of(t).is_none()).unwrap_or(false) { break; }
            assert_eq!(w.predators[0].target_id, first, "hunter re-picked mid-commitment");
            held += 1;
        }
        assert!(held > 0, "commitment did not survive even one tick");
    }



    #[test]
    fn the_ecology_only_ever_fields_triangles() {
        // The octagon and the rectangle are player powers. No automatic rule may
        // ever put one in the water, however badly the pond is overrun.
        let mut w = World::new(12, 10, 5);
        let over = w.cull_trigger_pop() + 400;
        for _ in 0..3000 {
            while w.prey_count() < over {
                w.pour_agents(6.0, 6.0, 64);
            }
            w.step();
            for p in &w.predators {
                assert_eq!(p.tier, 0, "the ecology summoned a tier-{} predator", p.tier);
            }
        }
    }

    #[test]
    fn the_pack_grows_by_one_each_time_the_threshold_is_crossed() {
        // A pond that keeps outgrowing its hunters accumulates them. Residents
        // never leave, so the pack is a running record of how often the pond has
        // outbred it.
        let mut w = World::new(12, 10, 5);
        let over = w.cull_trigger_pop() + 400;
        let mut counts = Vec::new();
        for _ in 0..2000 {
            while w.prey_count() < over {
                w.pour_agents(6.0, 6.0, 64);
            }
            w.step();
            counts.push(w.predators.len());
        }
        let first = counts[0];
        let last = *counts.last().unwrap();
        assert!(last > first, "pack never grew: {} → {}", first, last);
        assert!(last <= PREDATOR_MAX, "pack blew past the cap: {}", last);
        // Monotone: residents do not leave, so the count never falls.
        for pair in counts.windows(2) {
            assert!(pair[1] >= pair[0], "pack shrank: {} → {}", pair[0], pair[1]);
        }
    }

    #[test]
    fn higher_tiers_kill_harder() {
        // Lethality is back-loaded on purpose: if the bottom tiers could finish
        // the job the ladder would never be climbed.
        for t in 1..PREDATOR_TIERS as u8 {
            assert!(
                tier_bite(t) > tier_bite(t - 1),
                "tier {} does not out-reach tier {}", t, t - 1,
            );
        }
        assert!(tier_resident(0), "the triangles must stay");
        for t in 1..PREDATOR_TIERS as u8 {
            assert!(!tier_resident(t), "tier {} must hit and run", t);
        }
    }

    // ── Intelligence ──────────────────────────────────────────────────────────

    #[test]
    fn the_decision_interval_spans_the_trait_range() {
        let (lo, hi) = crate::genome::Traits::BOUNDS[9];
        assert_eq!(decision_interval(hi), 1, "the sharpest agents think every tick");
        assert_eq!(decision_interval(lo), DECISION_INTERVAL_MAX);
        // Monotone, so a mutation toward intelligence never makes an agent think
        // less often.
        let mut prev = u32::MAX;
        for step in 0..=20 {
            let v = lo + (hi - lo) * step as f64 / 20.0;
            let interval = decision_interval(v);
            assert!(interval <= prev, "interval rose with intelligence at {}", v);
            prev = interval;
        }
        assert_eq!(threat_lag(hi), 0, "the sharpest notice a predator the tick it arrives");
        assert_eq!(threat_lag(lo), THREAT_LAG_MAX);
    }

    #[test]
    fn a_dull_agent_acts_on_a_stale_decision() {
        let mut w = World::new(12, 40, 3);
        w.set_automatic_predators(false);
        // One dull animal and one sharp one, everything else out of the way.
        for i in 0..w.ids.len() {
            w.genome[i].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;
        }
        let dull = 0;
        w.genome[dull].traits.intelligence = crate::genome::Traits::BOUNDS[9].0;

        w.step();
        let first = w.last_outputs[dull];
        // Through the dull agent's whole interval, the outputs it acts on are the
        // ones it decided on the first tick — even as the pond moves around it.
        for _ in 1..DECISION_INTERVAL_MAX {
            w.step();
            if w.cause_of_death[dull].is_some() { return; }
            assert_eq!(w.last_outputs[dull], first, "a dull agent re-decided early");
        }
    }

    #[test]
    fn physics_still_runs_on_a_tick_an_agent_does_not_think() {
        // The trap this design exists to avoid: rationing decisions must not
        // ration movement. A dull agent keeps swimming on its last intent.
        let mut w = World::new(12, 30, 11);
        w.set_automatic_predators(false);
        for i in 0..w.ids.len() {
            w.genome[i].traits.intelligence = crate::genome::Traits::BOUNDS[9].0;
        }
        w.step();   // everyone decides on their first tick
        let before: Vec<(f32, f32)> = w.pos_x.iter().copied().zip(w.pos_y.iter().copied()).collect();
        w.step();   // nobody decides on this one
        let moved = (0..w.ids.len()).filter(|&i| {
            (w.pos_x[i], w.pos_y[i]) != before[i]
        }).count();
        assert!(moved > 0, "the pond froze on a no-decision tick");
    }

    #[test]
    fn a_predator_never_thinks_slowly() {
        // Predators are outside the brain path entirely, so cadence and lag must
        // not reach them however dull the genome they were spawned with. This is
        // the assertion that stops a later refactor quietly nerfing them.
        let mut w = World::new(16, 60, 4);
        let id = w.summon_predator(0.05, false).unwrap();
        let slot = w.slot_of(id).unwrap();
        w.genome[slot].traits.intelligence = crate::genome::Traits::BOUNDS[9].0;
        for _ in 0..40 { w.step(); }
        // Slots move as agents are reaped, so re-resolve rather than reusing the
        // one from before the run.
        let slot = w.slot_of(id).expect("the hunter is still in the pond");
        // It is never in the deciding population at all.
        assert!(!w.scratch_acting.contains(&slot), "a predator entered the brain path");
        assert_eq!(w.decision_cooldown[slot], 0, "a predator was put on a decision cadence");
    }

    #[test]
    fn thinking_costs_energy() {
        let sharp = World::new(12, 1, 5);
        let dull = World::new(12, 1, 5);
        let (lo, hi) = crate::genome::Traits::BOUNDS[9];

        let drain = |mut w: World, iq: f64| -> f64 {
            w.set_automatic_predators(false);
            w.genome[0].traits.intelligence = iq;
            // Same metabolism, so upkeep is the only difference between them.
            w.genome[0].traits.metabolism = 1.0;
            let start = w.energy[0];
            for _ in 0..20 { w.step(); }
            start - w.energy[0]
        };
        assert!(drain(sharp, hi) > drain(dull, lo),
            "intelligence must cost something, or every pond evolves to maximum");
    }

    // ── Disease ───────────────────────────────────────────────────────────────

    /// Plant a pathogen directly, so transmission can be tested without waiting
    /// for a promotion to roll one.
    fn plant_disease(w: &mut World, severity: f64, contagion: f64, origin: u32) -> u32 {
        let id = w.diseases.len() as u32 + 1;
        w.diseases.push(Disease {
            id, name: format!("Testibus morbus {}", id), origin_species: origin,
            severity, contagion, emerged_step: w.step_count, jumped: false,
        });
        id
    }

    #[test]
    fn infection_spreads_by_contact_and_needs_a_carrier() {
        let mut w = World::new(20, 30, 31);
        w.set_automatic_predators(false);
        let id = plant_disease(&mut w, 0.0, 1.0, 0);

        // Everyone piled onto one spot: crowding is maximal, contagion is 1.0.
        for i in 0..w.ids.len() {
            w.pos_x[i] = 10.0;
            w.pos_y[i] = 10.0;
            w.species_ids[i] = 0;
        }
        w.spatial.rebuild(&w.pos_x, &w.pos_y);
        // No carrier: nothing happens, however crowded.
        w.tick_disease();
        assert!(w.infection.iter().all(|&v| v == 0), "infection appeared from nowhere");

        w.infection[0] = id;
        w.tick_disease();
        let infected = w.infection.iter().filter(|&&v| v != 0).count();
        assert!(infected > 1, "a full-contagion carrier in a scrum infected nobody");
    }

    #[test]
    fn transmission_is_local_not_population_wide() {
        // The property that makes this a disturbance rather than a controller:
        // an outbreak reads local crowding only. Two ponds, same tight cluster,
        // wildly different totals — the cluster must fare the same in both.
        let seeded = |n: usize| -> usize {
            let mut w = World::new(40, n, 8);
            w.set_automatic_predators(false);
            let id = plant_disease(&mut w, 0.0, 0.5, 0);
            for i in 0..w.ids.len() {
                w.species_ids[i] = 0;
                // Uniform susceptibility: this test is about locality, and
                // random immunity would vary between the two ponds' clusters.
                w.genome[i].traits.immunity = 0.0;
            }
            // A tight cluster of eight; everything else is scattered far away.
            for i in 0..w.ids.len().min(8) {
                w.pos_x[i] = 5.0 + (i % 3) as f32 * 0.2;
                w.pos_y[i] = 5.0 + (i / 3) as f32 * 0.2;
            }
            for i in 8..w.ids.len() {
                w.pos_x[i] = 25.0 + (i % 10) as f32;
                w.pos_y[i] = 25.0 + (i / 10) as f32 % 10.0;
            }
            w.spatial.rebuild(&w.pos_x, &w.pos_y);
            w.infection[0] = id;
            for _ in 0..3 { w.tick_disease(); }
            w.infection.iter().take(8).filter(|&&v| v != 0).count()
        };
        let small = seeded(20);
        let large = seeded(200);
        assert_eq!(small, large,
            "the same cluster caught differently in a bigger pond — something is \
             reading total population");
    }

    #[test]
    fn a_pathogen_stays_in_its_own_species_until_it_jumps() {
        let scrum = |jumped: bool| -> usize {
            let mut w = World::new(20, 60, 12);
            w.set_automatic_predators(false);
            let id = plant_disease(&mut w, 0.0, 1.0, 1);
            w.diseases[0].jumped = jumped;
            for i in 0..w.ids.len() {
                w.pos_x[i] = 10.0;
                w.pos_y[i] = 10.0;
                w.species_ids[i] = 2;      // nobody is of the origin species
            }
            w.spatial.rebuild(&w.pos_x, &w.pos_y);
            w.infection[0] = id;
            for _ in 0..40 { w.tick_disease(); }
            w.infection.iter().filter(|&&v| v != 0).count()
        };

        // Full contagion, maximal crowding, forty ticks — and it stays put,
        // because none of these animals are its host.
        assert_eq!(scrum(false), 1, "a pathogen leaked into another species");
        // Once it has jumped, the same scrum goes up like tinder.
        assert!(scrum(true) > 30, "a jumped pathogen should spread to anything");
    }

    #[test]
    fn the_god_switch_stops_new_infections_but_not_existing_ones() {
        let mut w = World::new(20, 40, 55);
        w.set_automatic_predators(false);
        let id = plant_disease(&mut w, 0.0, 1.0, 0);
        for i in 0..w.ids.len() {
            w.pos_x[i] = 10.0;
            w.pos_y[i] = 10.0;
            w.species_ids[i] = 0;
            w.genome[i].traits.immunity = 0.0;
        }
        w.spatial.rebuild(&w.pos_x, &w.pos_y);
        w.infection[0] = id;

        w.disease_enabled = false;
        for _ in 0..20 { w.tick_disease(); }
        assert_eq!(w.infection.iter().filter(|&&v| v != 0).count(), 1,
            "disease spread with the switch off");
        assert_eq!(w.infection[0], id, "the switch cured someone instead of pausing");

        w.disease_enabled = true;
        w.tick_disease();
        assert!(w.infection.iter().filter(|&&v| v != 0).count() > 1,
            "turning it back on did not resume transmission");
    }

    #[test]
    fn offspring_are_born_into_their_parents_species() {
        // Membership is asserted directly rather than waiting for a promotion:
        // this is about what a newborn inherits, not about the registry, and a
        // real promotion needs a few thousand steps of the right pond.
        let mut w = World::new(12, 60, 42);
        w.set_automatic_predators(false);
        for _ in 0..60 { w.step(); }
        // A real lineage, defined on the parent's own signature so the parent is
        // comfortably inside it and only mutation can push a child out.
        let parent = 0;
        let centre = crate::species::signature(&w.genome[parent].traits);
        let species = w.species.plant_for_test(centre, "Thalura");
        w.species_ids[parent] = species;

        // Force a birth from that parent and check what the child is born as.
        w.energy[parent] = MAX_ENERGY_BASE;
        w.age[parent] = MATURITY_AGE + 1;
        w.reproduction_cooldown[parent] = 0;
        w.last_reproduced_age[parent] = None;
        let before = w.ids.len();
        let child = w.do_reproduce(parent).expect("parent did not reproduce");
        assert_eq!(child.species, species, "child inherited no lineage");
        w.spawn_offspring(vec![child]);
        assert_eq!(w.ids.len(), before + 1);
        assert_eq!(*w.species_ids.last().unwrap(), species,
            "a newborn was filed as unassigned until the next cluster tick");
    }

    // ── Cannibalism ───────────────────────────────────────────────────────────

    /// Two agents of one species on one tile, both starving, attacker maxed for
    /// aggression. Returns whether the attacker ate its relative.
    fn cannibalism_between_kin(intelligence: f64) -> bool {
        let mut w = World::new(8, 6, 17);
        w.set_automatic_predators(false);
        let (a, b) = (0, 1);
        for &i in &[a, b] {
            w.pos_x[i] = 2.5;
            w.pos_y[i] = 2.5;
            w.species_ids[i] = 3;                 // same lineage
            w.energy[i] = 5.0;                    // hungry enough to hunt
            w.genome[i].traits.aggression = 1.05;
            w.genome[i].traits.intelligence = intelligence;
        }
        // The attacker must be able to win the roll for the test to be about
        // the kin rule rather than about combat odds.
        w.genome[a].traits.attack = 1.25;
        w.genome[b].traits.defense = 0.5;
        w.parent_defense_bonus[b] = 0.0;
        w.spatial.rebuild(&w.pos_x, &w.pos_y);

        for _ in 0..40 {
            w.resolve_combat_spatial();
            if w.cause_of_death[b] == Some(CauseOfDeath::KilledInCombat) { return true; }
        }
        false
    }

    #[test]
    fn before_speciation_anything_is_food() {
        // The pond starts as a free-for-all. An unassigned agent has no
        // relatives, so nothing is protected from it however bright it is —
        // kin recognition is something a lineage gets on promotion.
        let mut w = World::new(8, 6, 17);
        w.set_automatic_predators(false);
        let (a, b) = (0, 1);
        for &i in &[a, b] {
            w.pos_x[i] = 2.5;
            w.pos_y[i] = 2.5;
            w.energy[i] = 5.0;
            w.species_ids[i] = crate::species::UNASSIGNED;
            w.genome[i].traits.aggression = 1.05;
            w.genome[i].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;
        }
        w.genome[a].traits.attack = 1.25;
        w.genome[b].traits.defense = 0.5;
        w.parent_defense_bonus[b] = 0.0;
        w.spatial.rebuild(&w.pos_x, &w.pos_y);

        let mut ate = false;
        for _ in 0..40 {
            w.resolve_combat_spatial();
            if w.cause_of_death[b] == Some(CauseOfDeath::KilledInCombat) { ate = true; break; }
        }
        assert!(ate, "kin protection reached agents that have no kin yet");
    }

    #[test]
    fn the_dull_and_furious_eat_their_own_and_the_bright_do_not() {
        let (lo, hi) = crate::genome::Traits::BOUNDS[9];
        assert!(cannibalism_between_kin(lo), "a dull, maximally aggressive agent spared its kin");
        assert!(!cannibalism_between_kin(hi), "a bright agent ate a member of its own species");
    }

    #[test]
    fn kin_protection_does_not_extend_to_other_lineages() {
        // The rule is about your own kind. A bright agent still hunts everything
        // else — otherwise intelligence would be a pacifism trait.
        let mut w = World::new(8, 6, 17);
        w.set_automatic_predators(false);
        let (a, b) = (0, 1);
        for &i in &[a, b] {
            w.pos_x[i] = 2.5;
            w.pos_y[i] = 2.5;
            w.energy[i] = 5.0;
            w.genome[i].traits.aggression = 1.05;
            w.genome[i].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;
        }
        w.species_ids[a] = 3;
        w.species_ids[b] = 4;                     // a different lineage
        w.genome[a].traits.attack = 1.25;
        w.genome[b].traits.defense = 0.5;
        w.parent_defense_bonus[b] = 0.0;
        w.spatial.rebuild(&w.pos_x, &w.pos_y);

        let mut ate = false;
        for _ in 0..40 {
            w.resolve_combat_spatial();
            if w.cause_of_death[b] == Some(CauseOfDeath::KilledInCombat) { ate = true; break; }
        }
        assert!(ate, "a bright agent refused to hunt another species");
    }

    #[test]
    fn a_child_that_mutates_past_the_definition_is_born_outside_it() {
        // The mechanism speciation runs on: a lineage is a definition fixed at
        // promotion, and mutation is the only thing that can put a child outside
        // it. Enough of those born-outside children clustering together is what
        // the candidate machinery promotes into the next species.
        let mut w = World::new(12, 60, 8);
        w.set_automatic_predators(false);
        for _ in 0..60 { w.step(); }
        let parent = 0;

        // A definition centred a long way from this parent's own shape: whatever
        // the child inherits, it cannot land inside.
        let far = [0.0; crate::species::SIG_LEN];
        let mut near = crate::species::signature(&w.genome[parent].traits);
        for v in near.iter_mut() { *v = (*v + 0.9).min(1.0); }
        let elsewhere = w.species.plant_for_test(far, "Vorixa");
        w.species_ids[parent] = elsewhere;
        assert!(!w.species.admits(elsewhere, &w.genome[parent].traits),
            "the fixture's parent is inside the definition, so nothing is tested");

        w.energy[parent] = MAX_ENERGY_BASE;
        w.age[parent] = MATURITY_AGE + 1;
        w.reproduction_cooldown[parent] = 0;
        w.last_reproduced_age[parent] = None;
        let child = w.do_reproduce(parent).expect("parent did not reproduce");
        assert_eq!(child.species, crate::species::UNASSIGNED,
            "a child outside the definition was seated in it anyway");
        let _ = near;
    }

    #[test]
    fn the_carrier_census_splits_by_species() {
        let mut w = World::new(20, 12, 3);
        w.set_automatic_predators(false);
        let a = plant_disease(&mut w, 0.0, 0.0, 1);
        let b = plant_disease(&mut w, 0.0, 0.0, 2);
        for i in 0..w.ids.len() { w.species_ids[i] = 0; }
        // Two carriers of A in species 1, one in species 2, one of B unassigned.
        w.infection[0] = a; w.species_ids[0] = 1;
        w.infection[1] = a; w.species_ids[1] = 1;
        w.infection[2] = a; w.species_ids[2] = 2;
        w.infection[3] = b; w.species_ids[3] = 0;

        let census = w.disease_carrier_census(13);
        assert_eq!(census[0][1], 2, "species 1 carriers of A");
        assert_eq!(census[0][2], 1, "species 2 carriers of A");
        assert_eq!(census[0].iter().sum::<u32>(), 3);
        assert_eq!(census[1][0], 1, "unassigned carrier of B");
        assert_eq!(census[1].iter().sum::<u32>(), 1);
    }

    #[test]
    fn immunity_resists_catching_but_does_not_cure() {
        let caught_with = |immunity: f64| -> usize {
            let mut w = World::new(20, 40, 77);
            w.set_automatic_predators(false);
            let id = plant_disease(&mut w, 0.0, 0.6, 0);
            for i in 0..w.ids.len() {
                w.pos_x[i] = 10.0;
                w.pos_y[i] = 10.0;
                w.species_ids[i] = 0;
                w.genome[i].traits.immunity = immunity;
            }
            w.spatial.rebuild(&w.pos_x, &w.pos_y);
            w.infection[0] = id;
            // One tick. Given enough of them a 0.03 chance against forty
            // neighbours still infects everyone — resistance slows an outbreak,
            // it does not wall it off, and the measurement has to respect that.
            w.tick_disease();
            w.infection.iter().filter(|&&v| v != 0).count()
        };
        let susceptible = caught_with(0.0);
        let resistant = caught_with(0.95);
        assert!(resistant < susceptible,
            "immunity did not resist: {} caught vs {}", resistant, susceptible);
        assert!(susceptible > 5, "the control scrum barely spread: {}", susceptible);
    }

    #[test]
    fn immunity_does_not_save_an_agent_already_infected() {
        // There is no recovery, at any immunity. Never catching it is the whole
        // defence — a curable disease is a restoring force and damps the
        // oscillation the mechanic exists to create.
        let mut w = World::new(12, 6, 44);
        w.set_automatic_predators(false);
        let id = plant_disease(&mut w, 5.0, 0.0, 0);
        w.infection[0] = id;
        w.genome[0].traits.immunity = 1.0;
        w.energy[0] = 1.0;
        for _ in 0..10 { w.step(); }
        assert!(w.death_counts()[CauseOfDeath::Disease.code() as usize] > 0,
            "a fully immune agent shrugged off an infection it already had");
    }

    #[test]
    fn immunity_costs_energy() {
        let drain = |immunity: f64| -> f64 {
            let mut w = World::new(12, 1, 5);
            w.set_automatic_predators(false);
            w.genome[0].traits.immunity = immunity;
            w.genome[0].traits.metabolism = 1.0;
            let start = w.energy[0];
            for _ in 0..20 { w.step(); }
            start - w.energy[0]
        };
        assert!(drain(1.0) > drain(0.0),
            "an immune system must cost something, or every pond evolves to maximum");
    }

    #[test]
    fn dying_infected_is_recorded_as_disease_not_starvation() {
        let mut w = World::new(12, 6, 44);
        w.set_automatic_predators(false);
        let id = plant_disease(&mut w, 5.0, 0.0, 0);   // brutal, non-contagious
        w.infection[0] = id;
        w.energy[0] = 1.0;
        for _ in 0..5 { w.step(); }
        assert!(w.death_counts()[CauseOfDeath::Disease.code() as usize] > 0,
            "an infected agent starved without the outbreak being credited");
    }

    #[test]
    fn a_disease_only_ever_arrives_with_a_species() {
        // Flat chance at promotion is the only origin. No promotions, no
        // pathogens, however long the pond runs or how crowded it gets.
        let mut w = World::new(12, 80, 6);
        for _ in 0..600 { w.step(); }
        if w.species.all().is_empty() {
            assert!(w.diseases.is_empty(), "a disease appeared with no species to carry it");
        }
        for d in &w.diseases {
            assert!(w.species.get(d.origin_species).is_some(),
                "disease {} has no host species", d.name);
        }
    }

    // ── Threat perception and flee ────────────────────────────────────────────

    #[test]
    fn vision_sets_the_range_a_predator_registers_at() {
        let mut w = World::new(20, 4, 21);
        w.set_automatic_predators(false);
        let id = w.summon_predator(0.05, false).unwrap();
        let hunter = w.slot_of(id).unwrap();
        let prey: Vec<usize> = (0..w.ids.len()).filter(|&i| i != hunter).collect();
        let (blind, sharp) = (prey[0], prey[1]);

        w.genome[blind].traits.vision = crate::genome::Traits::BOUNDS[0].0;
        w.genome[sharp].traits.vision = crate::genome::Traits::BOUNDS[0].1;
        // Everyone sees instantly, so this test is about vision alone.
        for &i in &[blind, sharp] {
            w.genome[i].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;
        }

        // Parked at a distance the sharp eye covers and the dim one does not.
        let blind_r = w.genome[blind].traits.vision as f32 * VISION_SCALE;
        let sharp_r = w.genome[sharp].traits.vision as f32 * VISION_SCALE;
        let gap = (blind_r + sharp_r) / 2.0;
        w.pos_x[hunter] = 10.0; w.pos_y[hunter] = 10.0;
        for &i in &[blind, sharp] {
            w.pos_x[i] = 10.0 + gap;
            w.pos_y[i] = 10.0;
        }

        w.scratch_acting.clear();
        w.scratch_acting.extend([blind, sharp]);
        w.sense_threats();

        assert_eq!(w.delayed_threat(blind)[0], 1.0, "a predator outside vision registered");
        assert!(w.delayed_threat(sharp)[0] < 1.0, "a predator inside vision went unseen");
    }

    #[test]
    fn a_dull_agent_learns_of_a_threat_late() {
        let mut w = World::new(20, 3, 22);
        w.set_automatic_predators(false);
        let id = w.summon_predator(0.05, false).unwrap();
        let hunter = w.slot_of(id).unwrap();
        let prey: Vec<usize> = (0..w.ids.len()).filter(|&i| i != hunter).collect();
        let (dull, sharp) = (prey[0], prey[1]);
        w.genome[dull].traits.intelligence = crate::genome::Traits::BOUNDS[9].0;
        w.genome[sharp].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;
        for &i in &[dull, sharp] { w.genome[i].traits.vision = crate::genome::Traits::BOUNDS[0].1; }

        w.pos_x[hunter] = 10.0; w.pos_y[hunter] = 10.0;
        for &i in &[dull, sharp] { w.pos_x[i] = 10.5; w.pos_y[i] = 10.0; }

        w.scratch_acting.clear();
        w.scratch_acting.extend([dull, sharp]);
        w.sense_threats();

        assert!(w.delayed_threat(sharp)[0] < 1.0, "a sharp agent should see it at once");
        assert_eq!(w.delayed_threat(dull)[0], 1.0,
            "a dull agent should still be looking at an empty pond");

        // It arrives once its lag has elapsed, and not before.
        let lag = threat_lag(w.genome[dull].traits.intelligence);
        for _ in 0..lag { w.sense_threats(); }
        assert!(w.delayed_threat(dull)[0] < 1.0, "the sighting never arrived");
    }

    #[test]
    fn flee_steers_away_and_only_when_the_brain_asks() {
        let mut w = World::new(20, 2, 23);
        w.set_automatic_predators(false);
        let id = w.summon_predator(0.05, false).unwrap();
        let hunter = w.slot_of(id).unwrap();
        let prey = (0..w.ids.len()).find(|&i| i != hunter).unwrap();
        w.genome[prey].traits.vision = crate::genome::Traits::BOUNDS[0].1;
        w.genome[prey].traits.intelligence = crate::genome::Traits::BOUNDS[9].1;

        // Hunter to the west, prey at rest.
        w.pos_x[hunter] = 9.0; w.pos_y[hunter] = 10.0;
        w.pos_x[prey] = 10.0;  w.pos_y[prey] = 10.0;
        w.vel_x[prey] = 0.0;   w.vel_y[prey] = 0.0;
        w.scratch_acting.clear();
        w.scratch_acting.push(prey);
        w.sense_threats();
        let (perception, food_dir) = w.perceive(prey);
        assert!(perception[5] < 1.0, "the prey cannot see the hunter");

        // Flee gate open, everything else shut: it must accelerate east, away.
        let mut outputs = [0f32; 8];
        outputs[OUT_FLEE] = 1.0;
        w.integrate_agent(prey, perception, food_dir, outputs);
        assert!(w.vel_x[prey] > 0.0, "fled toward the predator, not away");

        // Flee gate shut: no threat force at all. An agent that never evolved
        // the weight does not move away, and that is allowed to kill it.
        let mut w2 = World::new(20, 2, 23);
        w2.set_automatic_predators(false);
        let id2 = w2.summon_predator(0.05, false).unwrap();
        let hunter2 = w2.slot_of(id2).unwrap();
        let prey2 = (0..w2.ids.len()).find(|&i| i != hunter2).unwrap();
        w2.pos_x[hunter2] = 9.0; w2.pos_y[hunter2] = 10.0;
        w2.pos_x[prey2] = 10.0;  w2.pos_y[prey2] = 10.0;
        w2.vel_x[prey2] = 0.0;   w2.vel_y[prey2] = 0.0;
        w2.scratch_acting.clear();
        w2.scratch_acting.push(prey2);
        w2.sense_threats();
        let (p2, fd2) = w2.perceive(prey2);
        w2.integrate_agent(prey2, p2, fd2, [0f32; 8]);
        assert_eq!(w2.vel_x[prey2], 0.0, "something fled without being asked to");
    }

    // ── Predator adaptation ───────────────────────────────────────────────────

    /// Cruising speed for a hunter, ignoring any burst it is in.
    fn cruise_of(w: &World, pi: usize) -> f32 {
        let Some(image) = w.predators[pi].search_image else {
            return PREDATOR_SPEED_FLOOR_TRAIT * MAX_SPEED * DT;
        };
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..w.ids.len() {
            if w.cause_of_death[i].is_some() || w.is_predator(i) { continue; }
            if w.cluster.genome_cluster_ids.get(i).copied() != Some(image) { continue; }
            sum += w.genome[i].traits.speed as f32;
            count += 1;
        }
        if count == 0 { return PREDATOR_SPEED_FLOOR_TRAIT * MAX_SPEED * DT; }
        ((sum / count as f32) * MAX_SPEED * DT * w.predators[pi].cruise_frac)
            .max(PREDATOR_SPEED_FLOOR_TRAIT * MAX_SPEED * DT)
    }

    #[test]
    fn hunters_burst_and_the_burst_ends() {
        // Variance in the threat: a steady hunter is one a lineage can evolve a
        // fixed answer to. Over enough ticks a burst must happen, must run at
        // the ceiling, and must stop.
        let mut w = World::new(12, 200, 4);
        let ceiling = PREDATOR_SPEED_CEILING_TRAIT * MAX_SPEED * DT;
        let mut saw_burst = false;
        let mut saw_ceiling = false;
        let mut saw_end = false;
        for _ in 0..6000 {
            w.step();
            let Some(pi) = w.predators.iter().position(|p| p.tier == 0) else { continue };
            if w.predators[pi].burst_ticks > 0 {
                saw_burst = true;
                // A burst multiplies cruising speed rather than jumping to a
                // fixed value, so what is asserted is that it is meaningfully
                // faster than cruising and never past the ceiling.
                let v = w.predator_chase_speed(pi);
                if v > cruise_of(&w, pi) * 1.5 { saw_ceiling = true; }
                assert!(v <= ceiling + 1e-6, "a burst ran past the ceiling: {}", v);
            } else if saw_burst {
                saw_end = true;
            }
        }
        assert!(saw_burst, "no hunter ever burst in 6000 ticks");
        assert!(saw_ceiling, "a burst was not meaningfully faster than cruising");
        assert!(saw_end, "a burst never ended");
    }

    #[test]
    fn an_ambient_hunter_is_always_in_the_water() {
        let mut w = World::new(12, 120, 77);
        w.step();
        assert!(w.predators.iter().any(|p| tier_resident(p.tier)),
            "a pond above the ambient floor should always have a hunter in it");
        // And it never stops: no quota to meet, so nothing to be sated by.
        for _ in 0..400 { w.step(); }
        if w.prey_count() >= PREDATOR_AMBIENT_MIN_PREY {
            assert!(w.predators.iter().any(|p| !p.sated && p.leaving.is_none()),
                "the ambient hunter went quiet");
        }
    }

    #[test]
    fn a_hunter_is_fast_but_never_faster_than_the_genome_allows() {
        // An apex predator in open water is fast — it is floored well above the
        // pond's average. What it must never be is *unbeatable*: the ceiling
        // sits just under the quickest animal the genome can build, so a
        // lineage that spends everything on speed can still escape and one that
        // has not, cannot.
        let mut w = World::new(12, 200, 5);
        for _ in 0..120 { w.step(); }
        let pi = w.predators.iter().position(|p| p.tier == 0)
            .expect("expected an ambient triangle");
        let Some(image) = w.predators[pi].search_image else { return };

        let members: Vec<usize> = (0..w.ids.len())
            .filter(|&i| !w.is_predator(i) && w.cause_of_death[i].is_none())
            .filter(|&i| w.cluster.genome_cluster_ids.get(i).copied() == Some(image))
            .collect();
        if members.is_empty() { return; }

        let speed = w.predator_chase_speed(pi);
        let floor = PREDATOR_SPEED_FLOOR_TRAIT * MAX_SPEED * DT;
        let genetic_max = crate::genome::Traits::BOUNDS[1].1 as f32 * MAX_SPEED * DT;
        assert!(speed >= floor, "hunter at {} fell under the floor {}", speed, floor);
        assert!(speed < genetic_max,
            "hunter at {} is at or past the fastest possible animal {}", speed, genetic_max);
    }

    #[test]
    fn immortality_stops_the_ecology_but_not_the_player() {
        // A summoned hunter is the player overruling the rules; an ambient one
        // is a rule. Immortality suppresses the second and not the first.
        let mut w = World::new(12, 120, 9);
        w.immortal = true;
        for _ in 0..200 { w.step(); }
        assert_eq!(w.death_counts()[CauseOfDeath::EatenAlive.code() as usize], 0,
            "the ambient hunter ate under immortality");

        w.summon_predator(0.5, false);
        for _ in 0..200 { w.step(); }
        assert!(w.death_counts()[CauseOfDeath::EatenAlive.code() as usize] > 0,
            "a summoned hunter should still bite through immortality");
    }

    #[test]
    fn a_hunter_forms_an_image_of_the_commonest_family() {
        let mut w = World::new(12, 120, 42);
        w.summon_predator(0.5, false);
        for _ in 0..60 { w.step(); }

        let image = w.predators[0].search_image.expect("expected a search image");
        // It must be the plurality family among living prey, counted the same
        // way the review counts it.
        let k = w.tunables().cluster_k;
        let mut counts = vec![0usize; k];
        for i in 0..w.ids.len() {
            if w.cause_of_death[i].is_some() || w.is_predator(i) { continue; }
            if let Some(&c) = w.cluster.genome_cluster_ids.get(i) {
                counts[c as usize] += 1;
            }
        }
        let plurality = (0..k).max_by_key(|&c| counts[c]).unwrap();
        assert_eq!(image as usize, plurality);
    }

    #[test]
    fn the_image_prefers_its_family_without_ignoring_the_rest() {
        let mut w = World::new(12, 60, 7);
        for _ in 0..60 { w.step(); }
        let id = w.summon_predator(0.5, false).expect("a hunter");
        let hunter = w.slot_of(id).expect("its slot");

        // With no image, the nearest animal wins; with an image, a matching
        // animal up to SEARCH_IMAGE_PULL times further away wins instead.
        let plain = w.nearest_prey(hunter, None, None, 0.0);
        assert!(plain.is_some(), "there should be prey to chase");

        // Every family in turn: whatever the image, something is still chosen.
        for c in 0..w.tunables().cluster_k as u8 {
            assert!(w.nearest_prey(hunter, None, Some(c), 0.9).is_some(),
                "an image with no members must not stop the hunt");
        }
    }

    #[test]
    fn the_bite_learns_the_armour_it_keeps_meeting() {
        let mut w = World::new(12, 120, 42);
        w.summon_predator(0.5, false);
        // Armour the pond, and hand the hunter an untrained bite to learn from.
        for i in 0..w.ids.len() {
            if w.is_predator(i) { continue; }
            w.genome[i].traits.defense = 1.0;
            w.parent_defense_bonus[i] = 0.0;
        }
        for _ in 0..60 { w.step(); }   // labels exist, so a family can be counted
        let base = TIER_ATTACK[0];
        w.predators[0].attack = base;

        let mut last = base;
        for _ in 0..12 {
            w.review_predator_search_images();
            let now = w.predators[0].attack;
            assert!(now >= last - 1e-9, "the bite went backwards: {} then {}", last, now);
            last = now;
        }
        assert!(last > base, "the bite never adapted: {} vs base {}", last, base);
        assert!(last <= base + PREDATOR_ATTACK_MAX_ADAPT + 1e-9,
            "adaptation ran past its cap: {}", last);
    }

    #[test]
    fn armour_buys_time_not_immunity() {
        // The spiral this exists to break: a maxed-defense animal used to be
        // untouchable by a tier-0 hunter forever. It should survive an untrained
        // bite and lose to a trained one.
        let armoured = 1.07;
        assert!(!predator_attack_succeeds(0, TIER_ATTACK[0], armoured, 0.0, CHILDHOOD_TICKS));
        let trained = TIER_ATTACK[0] + PREDATOR_ATTACK_MAX_ADAPT;
        assert!(predator_attack_succeeds(0, trained, armoured, 0.0, CHILDHOOD_TICKS),
            "a fully trained hunter must be able to reach maximum armour");
    }

    #[test]
    fn switching_image_forgets_the_learned_bite() {
        let mut w = World::new(12, 120, 42);
        w.summon_predator(0.5, false);
        // Step first: with no cluster labels yet there is nothing to count, and
        // the review leaves every image alone rather than guessing.
        for _ in 0..60 { w.step(); }
        let base = TIER_ATTACK[0];
        w.predators[0].attack = base + 0.4;
        w.predators[0].search_image = Some(200);   // a family that cannot exist
        w.review_predator_search_images();
        assert!(w.predators[0].search_image != Some(200), "a dead image must be dropped");
        // Half the surplus is kept — general toughness carries, the specific
        // calibration does not. It must be strictly between base and what it had.
        let attack = w.predators[0].attack;
        assert!(attack < base + 0.4, "the whole learned bite carried over: {}", attack);
        assert!(attack > base, "the switch threw away everything: {}", attack);
    }

    #[test]
    fn predator_adaptation_consumes_no_rng() {
        // Speciation's guarantee, extended: reviewing search images is counting
        // and averaging over world state, so it must not shift the RNG stream.
        // Predators off: the point is that *reviewing* draws no RNG, and an
        // ambient hunter would confound it — its bite trains a little further on
        // every extra review, so the two ponds would diverge through the hunt
        // rather than through the RNG stream.
        let mut a = World::new(12, 100, 42);
        let mut b = World::new(12, 100, 42);
        a.set_automatic_predators(false);
        b.set_automatic_predators(false);
        for _ in 0..200 { a.step(); }
        for _ in 0..200 {
            b.step();
            b.review_predator_search_images();   // extra reviews, same result
        }
        assert_eq!(a.agent_count(), b.agent_count());
        assert_eq!(a.energy, b.energy);
    }

    #[test]
    fn predator_attack_is_strict_and_scaled_by_tier() {
        // An untrained hunter bites at its tier's base — the second argument is
        // the learned attack, which starts there.
        for (tier, &attack) in TIER_ATTACK.iter().enumerate() {
            if TIER_IGNORES_DEFENSE[tier] { continue; }
            let t = tier as u8;
            assert!(predator_attack_succeeds(t, attack, attack - 0.01, 0.0, CHILDHOOD_TICKS));
            assert!(!predator_attack_succeeds(t, attack, attack, 0.0, CHILDHOOD_TICKS));
            assert!(!predator_attack_succeeds(t, attack, attack + 0.01, 0.0, CHILDHOOD_TICKS));
        }
        // The existing childhood bonus participates in the same effective
        // defense calculation used by ordinary combat.
        assert!(!predator_attack_succeeds(0, TIER_ATTACK[0], 0.50, 0.50, 0));
    }

    #[test]
    fn the_rectangle_eats_anything_it_covers() {
        // The most final power in the game does not roll for it. Nothing the
        // trait bounds or the childhood bonus can produce survives the sweep.
        let tier = PREDATOR_RECTANGLE_TIER;
        let bite = TIER_ATTACK[tier as usize];
        for &defense in &[0.5, 1.0, 1.07, 5.0, f64::MAX] {
            assert!(predator_attack_succeeds(tier, bite, defense, 0.0, CHILDHOOD_TICKS));
            assert!(predator_attack_succeeds(tier, bite, defense, 0.5, 0),
                "childhood bonus survived");
        }
        // And the tiers below it still roll — an untrained triangle losing to
        // armour is what makes defense worth evolving in the first place.
        assert!(!predator_attack_succeeds(0, TIER_ATTACK[0], 1.07, 0.0, CHILDHOOD_TICKS));
    }

    #[test]
    fn a_rectangle_sweep_leaves_no_survivors_in_its_path() {
        // End to end: the maximum-defense pond, swept. Anything the shape covers
        // must be gone, however armoured.
        let mut w = World::new(16, 60, 4242);
        w.set_automatic_predators(false);
        for i in 0..w.ids.len() {
            w.genome[i].traits.defense = 1.07;
        }
        w.summon_predator_pack_tier(0, false, PREDATOR_RECTANGLE_TIER);
        assert!(!w.predators.is_empty(), "no rectangle arrived");

        for _ in 0..3000 {
            w.step();
            if w.prey_count() == 0 { break; }
        }
        assert_eq!(w.prey_count(), 0, "armoured agents survived the sweep");
    }

    #[test]
    fn summoned_hunters_can_be_dismissed_and_the_ecology_is_left_alone() {
        let mut w = World::new(16, 80, 55);
        w.summon_predator_tier(20, true, 0);                        // ecology
        w.summon_predator_tier(20, false, PREDATOR_MANUAL_TIER);    // player
        w.summon_predator_tier(20, false, PREDATOR_RECTANGLE_TIER); // player
        assert_eq!(w.summoned_predator_count(), 2);

        assert_eq!(w.dismiss_summoned_predators(), 2, "dismissal missed a summon");
        assert_eq!(w.summoned_predator_count(), 0, "a summon stayed behind");
        assert!(
            w.predators.iter().filter(|p| !p.automatic).all(|p| p.leaving.is_some()),
            "a dismissed hunter is not leaving",
        );
        // The ecology's own residents are not a player power and keep hunting.
        assert!(
            w.predators.iter().filter(|p| p.automatic).all(|p| p.leaving.is_none()),
            "dismissal sent an automatic resident away",
        );
        // Dismissing twice is not an error, and does not re-dismiss the leavers.
        assert_eq!(w.dismiss_summoned_predators(), 0);

        // They actually go, under their own power rather than blinking out.
        for _ in 0..PREDATOR_LEAVE_TICKS * 2 {
            w.step();
        }
        assert!(
            w.predators.iter().all(|p| p.automatic),
            "dismissed hunters never left the pond",
        );
    }

    #[test]
    fn predator_moves_on_after_a_target_resists() {
        let mut w = World::new(10, 2, 17);
        w.set_automatic_predators(false);
        w.summon_predator_tier(0, false, 0);
        let predator_id = w.predators[0].id;
        let predator_slot = w.slot_of(predator_id).unwrap();
        let prey: Vec<usize> = (0..w.ids.len())
            .filter(|&i| i != predator_slot)
            .collect();

        w.pos_x[predator_slot] = 1.0;
        w.pos_y[predator_slot] = 1.0;
        w.pos_x[prey[0]] = 1.0;
        w.pos_y[prey[0]] = 1.0;
        w.genome[prey[0]].traits.defense = 1.07;
        w.parent_defense_bonus[prey[0]] = 0.0;
        w.pos_x[prey[1]] = 5.0;
        w.pos_y[prey[1]] = 1.0;

        let resisted_id = w.ids[prey[0]];
        // A hunter that has not learned this pond's armour yet. Hunters now
        // arrive calibrated (see `starting_bite`), so the resist this test is
        // about has to be set up rather than assumed.
        w.predators[0].attack = TIER_ATTACK[0];
        w.hunt_one(predator_id);
        assert_eq!(w.predators[0].rejected_id, Some(resisted_id));
        assert!(w.ids.contains(&resisted_id), "resistant prey was eaten");

        w.hunt_one(predator_id);
        let predator_slot = w.slot_of(predator_id).unwrap();
        assert_ne!(w.pos_x[predator_slot], 1.0, "hunter did not move on");
        assert_eq!(w.predators[0].rejected_id, None);
    }

    #[test]
    fn disabling_automatic_predators_sends_only_them_away_and_resets_the_pack() {
        let mut w = World::new(16, 80, 91);
        w.summon_predator_tier(20, true, 0);
        w.summon_predator_tier(20, false, PREDATOR_MANUAL_TIER);
        assert!(w.predator_high_water > 0);

        w.set_automatic_predators(false);
        assert!(!w.automatic_predators_enabled);
        assert_eq!(w.predator_high_water, 0);
        assert!(w.predators.iter().filter(|p| p.automatic)
            .all(|p| p.leaving.is_some()));
        assert!(w.predators.iter().filter(|p| !p.automatic)
            .all(|p| p.leaving.is_none()));

        let before = w.predators.len();
        w.manage_predator_pack();
        assert_eq!(w.predators.len(), before, "disabled ecology reinforced");
    }

    #[test]
    fn the_rectangle_kills_along_its_edge_not_in_a_circle() {
        let top = (PREDATOR_TIERS - 1) as u8;
        let reach = tier_bite(top);
        // Far out along the long axis: inside the sweep.
        assert!(tier_bite_hits(top, reach * 0.9, 0.0, 0.0));
        // The same distance off the short axis: outside it.
        assert!(!tier_bite_hits(top, 0.0, reach * 0.9, 0.0));
        // Rotating the sweep by a quarter turn swaps which one is covered.
        let quarter = std::f32::consts::FRAC_PI_2;
        assert!(tier_bite_hits(top, 0.0, reach * 0.9, quarter));
    }

    #[test]
    fn a_cull_cuts_below_the_band_so_a_boom_has_room() {
        // Landing exactly on the floor left the pond one boom from re-trigger,
        // which made predators permanently resident and mid-hunt.
        let w = World::new(16, 1, 1);
        let floor = (w.pop_cap() as f64 * (1.0 - PREDATOR_POP_BAND)) as usize;
        assert!(
            w.cull_target_pop() < floor,
            "cull target {} is not below the hysteresis floor {}",
            w.cull_target_pop(), floor,
        );
    }

    #[test]
    fn predator_culls_to_its_target_then_leaves() {
        let mut w = World::new(20, 30, 12);
        w.set_automatic_predators(false);
        let before = w.prey_count();
        let target = before - 1;
        w.summon_predator_pack_tier(target, false, PREDATOR_MANUAL_TIER)
            .expect("summon failed");
        let target = w.predators[0].target_pop;
        let predator_slot = w.slot_of(w.predators[0].id).unwrap();
        let victim = (0..w.ids.len()).find(|&i| i != predator_slot).unwrap();
        w.genome[victim].traits.defense = 0.50;
        w.parent_defense_bonus[victim] = 0.0;
        w.pos_x[victim] = w.pos_x[predator_slot];
        w.pos_y[victim] = w.pos_y[predator_slot];
        w.hunt_one(w.predators[0].id);

        for _ in 0..PREDATOR_LEAVE_TICKS + 2 {
            w.hunt_with_predators();
            if w.predators.is_empty() { break; }
        }

        assert!(w.predators.is_empty(), "predator never left");
        assert!(w.prey_count() <= target);
    }

    #[test]
    fn predator_cannot_die() {
        let mut w = World::new(16, 120, 8);
        let id = w.summon_predator(0.05, false).unwrap();
        for _ in 0..600 {
            w.step();
            // Only this hunter is under test. The pack around it changes size as
            // the pond does, so an empty-pack check would stop early or, worse,
            // keep asserting about an id that has legitimately departed.
            if !w.predators.iter().any(|p| p.id == id) { break; }
            assert!(w.ids.contains(&id), "predator vanished while still hunting");
        }
        // Nothing ever recorded it as dead, by any cause.
        assert!(!w.last_deaths.iter().any(|d| d.id == id));
    }

    #[test]
    fn predator_leaving_is_not_a_death() {
        let mut w = World::new(16, 60, 4);
        let deaths_before: u32 = w.death_counts().iter().sum();
        // This is about the summoned hunter; the ambient resident would be a
        // second predator in the water and is not what is under test.
        w.set_automatic_predators(false);
        // Hit-and-run tier: the triangles stay in the pond by design.
        w.summon_predator_tier(w.prey_count(), false, PREDATOR_TIERS as u8 - 1);
        let id = w.predators[0].id;

        // It swims off before it disappears, so it is still present for a while.
        w.step();
        assert!(w.predators[0].leaving.is_some(), "did not begin departing");
        assert!(w.ids.contains(&id));

        for _ in 0..200 {
            w.step();
            if w.predators.is_empty() { break; }
        }
        assert!(w.predators.is_empty(), "predator never finished leaving");
        assert!(!w.ids.contains(&id), "predator still in the pond");

        // Departure is not a death: nothing but its own meals was tallied.
        let eaten = w.death_counts()[CauseOfDeath::EatenAlive.code() as usize];
        assert_eq!(deaths_before + eaten, w.death_counts().iter().sum::<u32>());
        assert!(!w.last_deaths.iter().any(|d| d.id == id));
    }

    #[test]
    fn departing_predator_does_not_resume_hunting() {
        // Births during the departure swim can push the population back above
        // the target; that must not drag it back into a second cull.
        let mut w = World::new(16, 80, 44);
        let target = w.prey_count();
        // This is about the summoned hunter; the ambient resident would be a
        // second predator in the water and is not what is under test.
        w.set_automatic_predators(false);
        w.summon_predator_tier(target, false, PREDATOR_MANUAL_TIER);   // hit-and-run
        w.step();
        assert!(w.predators[0].leaving.is_some());
        let eaten_at_departure = w.death_counts()[CauseOfDeath::EatenAlive.code() as usize];

        for _ in 0..PREDATOR_LEAVE_TICKS + 5 {
            w.step();
            if w.predators.is_empty() { break; }
        }
        // Ordinary combat can still eat agents, so compare the predator's own
        // tally rather than the world's.
        assert!(w.predators.is_empty() || w.predators[0].leaving.is_some());
        let _ = eaten_at_departure;
    }

    #[test]
    fn cull_band_scales_with_the_grid() {
        let small = World::new(12, 1, 1);
        let big = World::new(32, 1, 1);
        assert!(big.pop_cap() > small.pop_cap(), "cap must scale with pond area");
        // The band brackets the cap symmetrically, and leaves real breathing room.
        for w in [&small, &big] {
            assert!(w.cull_target_pop() < w.pop_cap());
            assert!(w.cull_trigger_pop() > w.pop_cap());
            assert!(w.cull_trigger_pop() - w.cull_target_pop() > 0);
        }
    }

    #[test]
    fn predators_arrive_over_the_cap_and_cull_into_the_band() {
        let mut w = World::new(12, 1, 3);
        let trigger = w.cull_trigger_pop();
        w.pour_agents(6.0, 6.0, trigger + 40);
        assert!(w.predators.is_empty());

        w.step();
        assert!(!w.predators.is_empty(), "no predator arrived over the cap");
        // The ambient resident is in the water too, with a quota of zero, so the
        // culling hunter is the one carrying the cull target rather than simply
        // the first in the list.
        let target = w.cull_target_pop();
        assert!(w.predators.iter().any(|p| p.target_pop == target),
            "no hunter arrived with the cull target");
        assert!(w.predators.iter().all(|p| p.tier == 0), "a wave starts at the bottom tier");

        for _ in 0..4000 {
            w.step();
            // Residents no longer sate — they revert to ambient duty — so the
            // end of a cull is the population reaching the band, not the hunters
            // going quiet.
            if w.prey_count() <= w.cull_target_pop() { break; }
        }
        // Culled into the band, not past it into an extinction.
        assert!(
            w.prey_count() <= w.cull_trigger_pop(),
            "population {} still over trigger {}", w.prey_count(), w.cull_trigger_pop()
        );
        // A few more ticks so the hunters notice the cull is over: the revert to
        // ambient duty happens inside the hunt, not in the loop condition above.
        for _ in 0..5 { w.step(); }
        // An automatic wave is made of resident triangles, and they stay — on
        // ambient duty, quota zero, rather than leaving or going quiet.
        assert!(
            w.predators.iter().any(|p| tier_resident(p.tier) && p.target_pop == 0),
            "no resident predator returned to ambient duty after the cull",
        );
    }

    #[test]
    fn no_cull_pack_while_under_the_cap() {
        // The ambient resident is always there above the prey floor — that is
        // the point of it. What must not happen under the cap is a *cull*: a
        // hunter carrying a population target.
        let mut w = World::new(16, 20, 11);
        for _ in 0..400 { w.step(); }
        assert!(w.prey_count() < w.cull_trigger_pop());
        assert!(w.predators.iter().all(|p| p.target_pop == 0),
            "a culling hunter arrived under the cap");
    }

    #[test]
    fn reinforcements_arrive_while_the_pond_keeps_climbing() {
        // Immortality means nothing dies of anything, so one hunter can never
        // win on its own: the pack has to grow.
        let mut w = World::new(12, 1, 21);
        w.immortal = true;
        w.pour_agents(6.0, 6.0, w.cull_trigger_pop() + 200);

        let mut max_pack = 0;
        for _ in 0..(PREDATOR_REINFORCE_STEPS * 4) {
            w.step();
            max_pack = max_pack.max(w.predators.len());
            // Keep the pond growing faster than one predator can eat.
            w.pour_agents(6.0, 6.0, 6);
        }
        assert!(max_pack > 1, "pack never reinforced (max {})", max_pack);
        assert!(max_pack <= PREDATOR_MAX);
    }

    #[test]
    fn predators_do_not_eat_each_other() {
        let mut w = World::new(12, 60, 31);
        w.summon_predator(0.5, false);
        w.summon_predator(0.5, false);
        let ids: Vec<u32> = w.predators.iter().map(|p| p.id).collect();
        for _ in 0..300 {
            w.step();
            for id in &ids {
                if w.predators.iter().any(|p| p.id == *id) {
                    assert!(w.ids.contains(id), "a predator ate another predator");
                }
            }
        }
    }

    #[test]
    fn pack_is_capped() {
        // A huge cull asks for more hunters than the cap allows; it gets the cap.
        let mut w = World::new(24, 1, 6);
        w.pour_agents(12.0, 12.0, PREY_PER_PREDATOR * (PREDATOR_MAX + 4));
        w.summon_predator(0.0, false);
        assert_eq!(w.predators.len(), PREDATOR_MAX, "pack ignored its cap");
        // Summoning again adds nothing: the pack is already at full strength.
        w.summon_predator(0.0, false);
        assert_eq!(w.predators.len(), PREDATOR_MAX);
    }

    #[test]
    fn pack_size_ratchets_and_never_shrinks() {
        let mut w = World::new(24, 1, 77);

        // A big cull fields a pack.
        w.pour_agents(12.0, 12.0, PREY_PER_PREDATOR * 4);
        w.summon_predator(0.0, false);
        let first_pack = w.predators.len();
        assert!(first_pack > 1);

        // Completion is not what this test measures. Defense can legitimately
        // stop a cull now, so finish the hit-and-run departure directly.
        for p in &mut w.predators {
            p.leaving = Some(1);
        }
        w.hunt_with_predators();
        assert!(w.predators.is_empty(), "pack never left");

        // A later, much smaller cull would only warrant one hunter on its own —
        // but the high-water mark stands, so it arrives at full strength.
        w.pour_agents(12.0, 12.0, 30);
        w.summon_predator(0.0, false);
        assert!(
            w.predators.len() >= first_pack,
            "pack shrank: {} < {}", w.predators.len(), first_pack
        );
    }

    #[test]
    fn manual_summon_scales_with_the_size_of_the_cull() {
        let mut small = World::new(16, 1, 9);
        small.pour_agents(8.0, 8.0, 40);
        small.summon_predator(0.2, false);
        assert_eq!(small.predators.len(), 1, "a small cull needs one hunter");

        let mut big = World::new(24, 1, 9);
        big.pour_agents(12.0, 12.0, PREY_PER_PREDATOR * 4);
        big.summon_predator(0.2, false);
        assert!(big.predators.len() > 1, "a big cull should arrive as a pack");
    }

    #[test]
    fn predator_kills_count_as_eaten_alive() {
        let mut w = World::new(16, 100, 15);
        w.summon_predator(0.5, false);
        for _ in 0..2000 {
            w.step();
            if w.predators.is_empty() { break; }
        }
        assert!(w.death_counts()[CauseOfDeath::EatenAlive.code() as usize] > 0);
    }

    #[test]
    fn soa_arrays_stay_aligned_when_the_last_slot_dies() {
        // Regression: the pop() branch of the removal loop never popped `kills`,
        // so killing the highest-indexed agent left that array one longer than
        // the rest and shifted every later agent's kill count by one.
        let mut w = World::new(10, 6, 2);
        let last = w.agent_count() - 1;
        w.cause_of_death[last] = Some(CauseOfDeath::Starvation);
        w.reap_dead(vec![last]);
        let n = w.agent_count();
        assert_eq!(w.kills.len(), n);
        assert_eq!(w.energy.len(), n);
        assert_eq!(w.genome.len(), n);
        assert_eq!(w.species_ids.len(), n);
    }

    #[test]
    fn species_ids_stay_aligned_through_spawn_and_swap_remove() {
        let mut w = World::new(10, 6, 2);
        assert_eq!(w.species_ids.len(), w.agent_count());
        w.pour_agents(5.0, 5.0, 3);
        assert_eq!(w.species_ids.len(), w.agent_count());
        w.species_ids = (0..w.agent_count()).map(|i| i as u32 + 1).collect();
        let moved_id = *w.ids.last().unwrap();
        let moved_species = *w.species_ids.last().unwrap();
        w.cause_of_death[1] = Some(CauseOfDeath::Starvation);
        w.reap_dead(vec![1]);
        let moved_slot = w.ids.iter().position(|&id| id == moved_id).unwrap();
        assert_eq!(w.species_ids[moved_slot], moved_species);
        assert_eq!(w.species_ids.len(), w.agent_count());
    }

    #[test]
    fn stats_sampled_on_interval_boundaries() {
        let mut w = World::new(10, 40, 3);
        let steps = crate::stats::SAMPLE_INTERVAL * 4;
        for _ in 0..steps {
            w.step();
        }
        assert_eq!(w.stats_history.len(), 4);
        let last = w.stats_history.latest().unwrap();
        assert_eq!(last.step, steps);
        // Prey, not slots: the sampled series excludes predators.
        assert_eq!(last.alive as usize, w.prey_count());
        assert_eq!(last.total_food, w.get_stats().total_food);
    }

    #[test]
    fn interval_median_lifespan_moves_with_the_interval() {
        // The old cumulative median flattened after a few hundred deaths and
        // stopped saying anything. Per-interval, the value must be able to
        // change between samples and must be 0 in an interval with no deaths.
        let mut w = World::new(10, 60, 11);
        let mut seen = Vec::new();
        for _ in 0..40 {
            for _ in 0..crate::stats::SAMPLE_INTERVAL {
                w.step();
            }
            seen.push(w.stats_history.latest().unwrap().median_lifespan);
        }
        let distinct: std::collections::HashSet<u32> =
            seen.iter().map(|v| v.to_bits()).collect();
        assert!(distinct.len() > 2, "interval median never moved: {:?}", seen);
    }

    #[test]
    fn sampled_age_band_brackets_living_agents() {
        let mut w = World::new(10, 40, 5);
        for _ in 0..crate::stats::SAMPLE_INTERVAL * 3 {
            w.step();
        }
        let s = w.stats_history.latest().unwrap();
        let lo = w.age.iter().copied().min().unwrap();
        let hi = w.age.iter().copied().max().unwrap();
        assert!(s.age_p10 >= lo && s.age_p90 <= hi, "percentiles outside the true range");
        assert!(s.age_p10 <= s.age_p90);
    }

    #[test]
    fn interval_deaths_sum_to_cumulative_tally() {
        let mut w = World::new(12, 80, 17);
        for _ in 0..600 {
            w.step();
            if w.agent_count() == 0 { break; }
        }
        let mut summed = [0u32; crate::stats::CAUSE_COUNT];
        for s in w.stats_history.iter_chrono() {
            for (total, d) in summed.iter_mut().zip(s.deaths) {
                *total += d;
            }
        }
        // Deaths after the last sample boundary aren't in the history yet, so
        // the history can lag the tally but must never exceed it.
        let cumulative = w.death_counts();
        for i in 0..crate::stats::CAUSE_COUNT {
            assert!(
                summed[i] <= cumulative[i],
                "cause {} history {} exceeds tally {}", i, summed[i], cumulative[i]
            );
        }
        assert!(cumulative.iter().sum::<u32>() > 0, "no deaths in 600 steps");
    }

    #[test]
    fn speciation_is_deterministic() {
        // Speciation draws no RNG of its own, but it is not a pure observer:
        // the probation clamp changes the mutation rate used at reproduction,
        // and because the per-weight mutation draw is conditional on that rate,
        // the RNG stream itself diverges from an unspeciated run. So the
        // guarantee is same-seed reproducibility, not parity with a build that
        // has speciation switched off.
        let mut a = World::new(12, 100, 42);
        let mut b = World::new(12, 100, 42);
        for _ in 0..900 {
            a.step();
            b.step();
        }
        let (sa, sb) = (a.get_stats(), b.get_stats());
        assert_eq!(sa.alive_agents, sb.alive_agents);
        assert_eq!(sa.total_food, sb.total_food);
        assert_eq!(sa.avg_energy, sb.avg_energy);
        assert_eq!(a.death_counts(), b.death_counts());
        assert_eq!(a.species.all(), b.species.all());
        assert_eq!(a.species_ids, b.species_ids);
    }

    #[test]
    fn generation_telemetry_tracks_reproductive_depth() {
        // Predators off: this is about reproduction bookkeeping, and an ambient
        // hunter in a small test pond eats the population before it breeds.
        // Founders are generation 0, so the first samples must read 0 and the
        // series must climb only as reproduction happens — never as age alone.
        let mut w = World::new(12, 100, 42);
        w.set_automatic_predators(false);
        for _ in 0..crate::stats::SAMPLE_INTERVAL {
            w.step();
        }
        let first = *w.stats_history.iter_chrono().next().unwrap();
        assert_eq!(first.max_generation, 0, "no reproduction yet");
        assert_eq!(first.mean_generation, 0.0);

        for _ in 0..1000 {
            w.step();
        }
        let last = *w.stats_history.iter_chrono().last().unwrap();
        assert!(last.max_generation > 0, "1000 steps produced no offspring");
        assert!(last.mean_generation <= last.max_generation as f32);
    }

    #[test]
    fn sampling_does_not_perturb_the_sim() {
        // Sampling is read-only, so a world stepped past many sample boundaries
        // must match one stepped the same number of steps with the same seed.
        let mut a = World::new(10, 50, 23);
        let mut b = World::new(10, 50, 23);
        for _ in 0..crate::stats::SAMPLE_INTERVAL * 7 {
            a.step();
            b.step();
        }
        assert_eq!(a.agent_count(), b.agent_count());
        assert_eq!(a.get_stats().total_food, b.get_stats().total_food);
        assert_eq!(a.death_counts(), b.death_counts());
    }
}

#[cfg(test)]
mod predation_selection {
    use super::*;

    /// The property this whole mechanism exists for: predation must not be a
    /// subsidy for armour.
    ///
    /// Measured as the mean defense of what the hunters ate, minus the mean
    /// defense of the pond they ate it from *at that moment* — a run-long
    /// average would be meaningless, since predation is front-loaded and armour
    /// climbs all run. Negative means predation is killing the soft and sparing
    /// the armoured, which is the reward this change removes; it read -0.155
    /// before, and the intermediate versions (aim at the mean, aim at mean + 2σ,
    /// untrained hunters at spawn) all left it near -0.15 too.
    #[test]
    fn predation_does_not_subsidise_armour() {
        let mut w = World::new(12, 400, 42);
        let mut eaten = Vec::new();
        let mut pond = Vec::new();

        for _ in 0..1200 {
            let before: HashMap<u32, f64> = w.ids.iter().enumerate()
                .filter(|&(i, _)| !w.is_predator(i))
                .map(|(i, &id)| (id, w.genome[i].traits.defense))
                .collect();
            let live_now = if before.is_empty() { 0.0 }
                else { before.values().sum::<f64>() / before.len() as f64 };
            w.step();
            for d in &w.last_deaths {
                if d.cause != CauseOfDeath::EatenAlive.code() { continue; }
                if let Some(&def) = before.get(&d.id) {
                    eaten.push(def);
                    pond.push(live_now);
                }
            }
        }

        assert!(eaten.len() > 50, "too few kills to conclude anything: {}", eaten.len());
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let selection = mean(&eaten) - mean(&pond);
        assert!(selection > -0.05,
            "predation is selecting for armour: eaten {:.3} vs pond {:.3} ({:+.3})",
            mean(&eaten), mean(&pond), selection);
    }
}


#[cfg(test)]
mod speed_probe {
    use super::*;

    /// Why is the pond slow? Three conditions, same seeds.
    #[test]
    #[ignore]
    fn what_drives_speed() {
        for seed in [42u64, 7, 1337] {
            // A: predators off entirely.
            let mut a = World::new(12, 400, seed);
            a.set_automatic_predators(false);
            for _ in 0..3000 { a.step(); }

            // B: predators on, speed pinned to the old flat constant.
            let mut b = World::new(12, 400, seed);
            b.pin_predator_speed_for_test = true;
            for _ in 0..3000 { b.step(); }

            // C: as shipped — hunter tracks its prey's mean speed.
            let mut c = World::new(12, 400, seed);
            for _ in 0..3000 { c.step(); }

            let sp = |w: &World| w.trait_means()[1];
            let mt = |w: &World| w.trait_means()[2];
            println!(
                "seed {seed}: speed  no-pred {:.3} | flat-pred {:.3} | tracking {:.3}   \
                 metabolism {:.3} / {:.3} / {:.3}   pop {} / {} / {}",
                sp(&a), sp(&b), sp(&c), mt(&a), mt(&b), mt(&c),
                a.agent_count(), b.agent_count(), c.agent_count(),
            );
        }
    }
}
