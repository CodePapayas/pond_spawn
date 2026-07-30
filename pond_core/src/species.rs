//! Named species — stable genome clusters promoted to lineages with identity.
//!
//! A k-means label is not a species. It exists because `k = 6` was chosen, it
//! permutes when clusters split or merge, and family 3 at step 200 shares
//! nothing with family 3 at step 2000 but a slot. This module watches those
//! clusters over time and *promotes* the ones that hold still long enough to be
//! real: a species gets a monotonic id that is never reused, a founding step,
//! and a record that survives its own extinction.
//!
//! Two design points carry most of the weight:
//!
//! 1. **The signature is the seven mutable traits, not all nine.**
//!    `energy_capacity` and `mutation_rate` are locked (D3): never mutated,
//!    inherited exactly. That makes them perfect founder tags with zero
//!    selection pressure, so clustering on them yields descent groups rather
//!    than ecotypes — and normalizing amplifies them, since `energy_capacity`
//!    spans a raw 0.1 that would otherwise stay negligible. A species should be
//!    a shape selection built, so the locked pair is excluded.
//!
//! 2. **Candidates are tracked by their own centroid, not by k-means label.**
//!    Label stability is exactly the unreliable thing here — `match_labels` is
//!    greedy, and a split or merge permutes labels. Keying candidacy by label
//!    would either reset the counter forever or silently hand one blob's
//!    accumulated stability to another. Post-promotion membership already works
//!    by nearest centroid; candidates use the same mechanism one stage earlier,
//!    so there is one matching path in the module rather than two.
//!
//! Nothing here draws from the world RNG. Promotion is a pure function of world
//! state, so a run with speciation enabled steps identically to one without.

use std::fmt;

use crate::cluster::ClusterState;
use crate::genome::{Genome, Traits};
use crate::naming::{self, Name};

/// Trait indices used for the species signature: the seven mutable traits.
/// Skips `energy_capacity` (3) and `mutation_rate` (4) — see module docs.
pub const SIGNATURE_DIMS: [usize; 8] = [0, 1, 2, 5, 6, 7, 8, 9];
/// Length of a signature vector.
pub const SIG_LEN: usize = SIGNATURE_DIMS.len();

/// Consecutive cluster runs a candidate must satisfy every criterion.
/// At the 50-step cluster interval this is 250 steps.
pub const STABILITY_RUNS: u32 = 5;
/// Generations a cluster must persist, holding its share and staying settled,
/// before it enters probation.
///
/// This is the criterion that makes the module *speciation* rather than
/// clustering: a cluster can sit still because its members are long-lived, not
/// because its shape is heritable, and only generation advance distinguishes
/// the two. It has to be set against real turnover — generations per step
/// varies with grid size, population and food pressure, and a value the entry
/// window cannot physically contain silently promotes nothing at all. Check
/// `mean_generation` in the stats CSV before raising it.
pub const PROBATION_ENTRY_GENERATIONS: f32 = 3.0;
/// Further generations a suppressed cluster must survive to be promoted.
pub const PROBATION_TEST_GENERATIONS: f32 = 1.0;
/// Floor on member count, regardless of population.
pub const MIN_MEMBERS_FLOOR: usize = 6;
/// Member count as a fraction of the living population.
///
/// Relative, not absolute. Population swings by large factors within a single
/// run — boom, crash, re-radiation — so an absolute floor means something
/// different at each phase: generous at peak, and at a trough large enough that
/// only one cluster can possibly clear it, which turns the criterion into
/// "promote only during a monoculture".
pub const MIN_MEMBERS_FRAC: f64 = 0.05;
/// Mutation rate multiplier applied to members of a cluster on probation.
///
/// **Economy lever.** This is the experiment: a cluster that has held its shape
/// and its share for `PROBATION_ENTRY_GENERATIONS` gets its mutability taken
/// away, and only a lineage that keeps holding both without it is promoted. A
/// fluke — a shape that was still riding mutation toward a fit, or that k-means
/// happened to bracket — loses share once frozen and never promotes.
///
/// Not zero: a total freeze is the purer experiment but leaves a lineage no
/// headroom at all, and the clamp already applies to trait and brain-weight
/// mutation alike (they share one rate), so 0.0 would stop behavioural
/// adaptation dead for the whole probation window.
pub const PROBATION_MUTATION_CLAMP: f32 = 0.15;
/// Max centroid movement per run for a candidate to count as settled,
/// in normalized signature space.
pub const DRIFT_EPS: f64 = 0.04;
/// Max mean per-trait standard deviation for a candidate to count as a cluster
/// rather than a bin.
pub const SPREAD_MAX: f64 = 0.25;
/// An agent belongs to the nearest species within this normalized distance.
/// Outside every radius it is unassigned (species 0).
pub const MEMBERSHIP_RADIUS: f64 = 0.35;
/// How far a candidate centroid may sit from a tracked candidate and still be
/// considered the same one. Wider than membership: candidates are still moving.
pub const CANDIDATE_MATCH_RADIUS: f64 = 0.5;
/// Rate at which a species centroid tracks its members, per run.
///
/// Fast tracking would let a species follow its own members anywhere and never
/// go extinct; zero tracking would strand it in place as they drift.
pub const CENTROID_TRACKING: f64 = 0.05;
/// Consecutive empty runs before a species is declared extinct.
pub const EXTINCTION_RUNS: u32 = 2;
/// How close a promoting species must sit to an existing one — live or extinct —
/// to be read as the same lineage and inherit its genus.
pub const GENUS_RADIUS: f64 = 0.45;
/// Cap on simultaneously live species. At the cap, promotion is refused and the
/// candidate keeps accumulating — evicting a live species would write a false
/// `extinct_at` and make the fossil record lie.
pub const MAX_SPECIES: usize = 12;

/// Species id reserved for agents matching no species.
pub const UNASSIGNED: u32 = 0;

/// A promoted lineage. Extinct entries are retained forever — the `species`
/// vector is the fossil record, not a live set.
#[derive(Debug, Clone, PartialEq)]
pub struct Species {
    /// Binomial. Genus is feminine and inherited from the nearest lineage;
    /// the epithet reports the trait this species most deviates from the pond
    /// on, or is nonsense when it deviates on nothing. See `naming.rs`.
    pub name: Name,
    /// Monotonic, never reused. An extinct species that reappears at the same
    /// centroid is a *new* species: convergent evolution is the more
    /// interesting reading, and resurrection would make the timeline lie.
    pub id: u32,
    /// Nearest kin at promotion — the species this one is read as having split
    /// from — or `UNASSIGNED` for a lineage that founded in empty trait space.
    ///
    /// It is the same nearest-within-`GENUS_RADIUS` neighbour whose genus this
    /// species inherits, so name and ancestry cannot disagree. Extinct species
    /// are eligible: a lineage that re-radiates after a bottleneck descends from
    /// the one that died.
    ///
    /// This is an inference, not an observed birth: nothing watches a population
    /// split, and a species promoting near an unrelated lineage that happens to
    /// have converged on the same shape will be recorded as its child. Read it
    /// as "nearest relative at the time it earned a name".
    pub parent_id: u32,
    /// Trait signature, tracked slowly toward the member mean.
    pub centroid: [f64; SIG_LEN],
    /// Signature at promotion, kept unchanged for the lineage record.
    pub founding_centroid: [f64; SIG_LEN],
    /// Whole-pond signature at promotion. The difference from
    /// `founding_centroid` is the evidence used to name the lineage.
    pub founding_population_centroid: [f64; SIG_LEN],
    pub founded_step: u32,
    /// Population mean generation at promotion.
    pub founder_generation: f32,
    /// Promotion snapshot. These values are immutable evidence for the UI:
    /// they explain why this cluster passed rather than merely stating that it
    /// did.
    pub founder_members: u32,
    pub founder_population: u32,
    pub promotion_streak: u32,
    pub promotion_drift: f64,
    pub promotion_spread: f64,
    pub entry_generation_advance: f32,
    pub probation_generation_advance: f32,
    pub extinct_at: Option<u32>,
    pub peak_members: u32,
    /// Current, recomputed each run.
    pub members: u32,
    /// Consecutive runs with zero members.
    empty_runs: u32,
}

impl Species {
    pub fn is_alive(&self) -> bool {
        self.extinct_at.is_none()
    }

    /// Steps from founding to extinction, or to `now` if still alive.
    pub fn age(&self, now: u32) -> u32 {
        self.extinct_at.unwrap_or(now).saturating_sub(self.founded_step)
    }
}

/// A cluster under observation, moving through observed → probation → promoted.
#[derive(Debug, Clone)]
struct Candidate {
    centroid: [f64; SIG_LEN],
    /// Consecutive qualifying runs. Reset to 0 on any failure — stability must
    /// be consecutive, or a cluster that oscillates gets promoted by accrual.
    streak: u32,
    /// Population mean generation when the current streak began.
    streak_start_generation: f32,
    /// Population mean generation when probation began, if it has.
    /// `Some` is what makes members' mutation rate clamped.
    probation_start_generation: Option<f32>,
    /// Size at the most recent sighting. Carried into the promotion event so it
    /// reports the members the species is founded on rather than 0 — membership
    /// is not recounted until the following run.
    members: u32,
    /// Most recent qualifying measurements, retained for the promotion record.
    drift: f64,
    spread: f64,
    /// Set each run; candidates not seen this run are dropped.
    seen_this_run: bool,
}

impl Candidate {
    fn on_probation(&self) -> bool {
        self.probation_start_generation.is_some()
    }
}

/// Something worth telling the user about. Drained by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeciesEvent {
    /// A cluster held long enough to have its mutability clamped. It is now
    /// being tested, not yet named.
    ProbationStarted { step: u32, members: u32 },
    /// A cluster on probation stopped qualifying. The clamp lifts and it goes
    /// back to being an ordinary cluster.
    ProbationFailed { step: u32, members: u32 },
    Promoted { id: u32, name: Name, step: u32, members: u32 },
    Extinct { id: u32, step: u32, age: u32, peak: u32 },
}

impl fmt::Display for SpeciesEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeciesEvent::ProbationStarted { step, members } =>
                write!(f, "cluster on probation — step {}, {} members, mutation clamped to {}×",
                       step, members, PROBATION_MUTATION_CLAMP),
            SpeciesEvent::ProbationFailed { step, members } =>
                write!(f, "probation failed — step {}, {} members, clamp lifted", step, members),
            SpeciesEvent::Promoted { id, name, step, members } =>
                write!(f, "{} emerged — species {}, step {}, {} members", name, id, step, members),
            SpeciesEvent::Extinct { id, step, age, peak } =>
                write!(f, "species {} extinct — step {}, lived {} steps, peaked at {}", id, step, age, peak),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeciesRegistry {
    /// Live and extinct, in promotion order.
    species: Vec<Species>,
    candidates: Vec<Candidate>,
    next_id: u32,
    events: Vec<SpeciesEvent>,
    /// Seeds name generation. Names must replay identically for a given world
    /// seed, and must never draw from the world's RNG stream.
    world_seed: u64,
}

impl SpeciesRegistry {
    pub fn new(world_seed: u64) -> Self {
        Self {
            species: Vec::new(),
            candidates: Vec::new(),
            next_id: 1,
            events: Vec::new(),
            world_seed,
        }
    }

    /// The seed this registry names from. Diseases derive their own names from
    /// it too, so a run's pathogens replay with its species.
    pub fn world_seed(&self) -> u64 { self.world_seed }

    pub fn all(&self) -> &[Species] {
        &self.species
    }

    pub fn get(&self, id: u32) -> Option<&Species> {
        self.species.iter().find(|s| s.id == id)
    }

    pub fn live_count(&self) -> usize {
        self.species.iter().filter(|s| s.is_alive()).count()
    }

    /// Take the accumulated events, leaving the queue empty.
    pub fn drain_events(&mut self) -> Vec<SpeciesEvent> {
        std::mem::take(&mut self.events)
    }

    /// Advance one cluster run. Returns the per-agent species id, parallel to
    /// `genomes` — `UNASSIGNED` for agents outside every species radius.
    ///
    /// Order matters: membership is resolved against the *existing* species
    /// first, then candidates are formed only from what is left over. Forming
    /// candidates over all agents would let a cluster that is already a species
    /// be promoted a second time at the same centroid.
    pub fn update(
        &mut self,
        genomes: &[Genome],
        cluster: &ClusterState,
        step: u32,
    ) -> Vec<u32> {
        let n = genomes.len();
        if n == 0 {
            self.retire_empty_species(step);
            self.candidates.clear();
            return Vec::new();
        }

        let points: Vec<[f64; SIG_LEN]> = genomes.iter().map(|g| signature(&g.traits)).collect();
        let mean_generation =
            genomes.iter().map(|g| g.generation as f64).sum::<f64>() / n as f64;

        // ── Membership ───────────────────────────────────────────────────────
        let assignment: Vec<u32> = points.iter().map(|p| self.nearest_species(p)).collect();

        // ── Species bookkeeping ──────────────────────────────────────────────
        self.update_species(&points, &assignment, step);

        // ── Candidates, over unassigned agents only ──────────────────────────
        let min_members = min_members(n);
        for c in self.candidates.iter_mut() {
            c.seen_this_run = false;
        }
        for group in unassigned_groups(&points, &assignment, cluster) {
            self.observe_candidate(group, min_members, mean_generation as f32, step);
        }
        // A candidate whose cluster vanished this run has broken its streak.
        for c in self.candidates.iter().filter(|c| !c.seen_this_run && c.on_probation()) {
            self.events.push(SpeciesEvent::ProbationFailed { step, members: c.members });
        }
        self.candidates.retain(|c| c.seen_this_run);

        // ── Promotion ────────────────────────────────────────────────────────
        // Population centroid, so a promoting species can be named for what
        // makes it different from the pond rather than for whatever trait
        // happens to sit high across the whole pond.
        let mut population_centroid = [0f64; SIG_LEN];
        for p in &points {
            for d in 0..SIG_LEN {
                population_centroid[d] += p[d];
            }
        }
        for v in population_centroid.iter_mut() {
            *v /= n as f64;
        }
        self.promote_ready(step, mean_generation as f32, n as u32, &population_centroid);

        // Promotion may have created a species the leftover agents now belong
        // to, so resolve membership once more for the returned assignment.
        points.iter().map(|p| self.nearest_species(p)).collect()
    }

    /// Mutation clamp for an agent about to reproduce: `PROBATION_MUTATION_CLAMP`
    /// if its genome sits inside a cluster currently on probation, else 1.0.
    ///
    /// Keyed by genome, not by array slot. Reproduction happens every step while
    /// candidates refresh only on cluster runs, and `swap_remove` reshuffles
    /// agent slots in between — a parallel per-agent vector would hand agents
    /// the wrong clamp as the population churns. That is survivable for a
    /// colour and not for a mechanic. There are at most a handful of
    /// candidates, so scanning them is cheaper than the bookkeeping would be.
    pub fn clamp_for(&self, traits: &Traits) -> f32 {
        if self.candidates.iter().all(|c| !c.on_probation()) {
            return 1.0;
        }
        let p = signature(traits);
        let inside = self
            .candidates
            .iter()
            .filter(|c| c.on_probation())
            .any(|c| dist_sq(&p, &c.centroid) < MEMBERSHIP_RADIUS * MEMBERSHIP_RADIUS);
        if inside { PROBATION_MUTATION_CLAMP } else { 1.0 }
    }

    /// Clusters currently under the clamp.
    pub fn probation_count(&self) -> usize {
        self.candidates.iter().filter(|c| c.on_probation()).count()
    }

    /// Nearest live species within `MEMBERSHIP_RADIUS`, else `UNASSIGNED`.
    fn nearest_species(&self, p: &[f64; SIG_LEN]) -> u32 {
        let mut best = (MEMBERSHIP_RADIUS * MEMBERSHIP_RADIUS, UNASSIGNED);
        for s in self.species.iter().filter(|s| s.is_alive()) {
            let d = dist_sq(p, &s.centroid);
            if d < best.0 {
                best = (d, s.id);
            }
        }
        best.1
    }

    /// Recount members, track centroids toward them, and retire the empty.
    fn update_species(&mut self, points: &[[f64; SIG_LEN]], assignment: &[u32], step: u32) {
        for s in self.species.iter_mut().filter(|s| s.is_alive()) {
            let mut sum = [0f64; SIG_LEN];
            let mut count = 0u32;
            for (p, &a) in points.iter().zip(assignment) {
                if a == s.id {
                    for d in 0..SIG_LEN {
                        sum[d] += p[d];
                    }
                    count += 1;
                }
            }
            s.members = count;
            s.peak_members = s.peak_members.max(count);

            if count == 0 {
                s.empty_runs += 1;
            } else {
                s.empty_runs = 0;
                for d in 0..SIG_LEN {
                    let mean = sum[d] / count as f64;
                    s.centroid[d] += CENTROID_TRACKING * (mean - s.centroid[d]);
                }
            }
        }
        self.retire_empty_species(step);
    }

    fn retire_empty_species(&mut self, step: u32) {
        for s in self.species.iter_mut() {
            if s.is_alive() && s.members == 0 && s.empty_runs >= EXTINCTION_RUNS {
                s.extinct_at = Some(step);
                self.events.push(SpeciesEvent::Extinct {
                    id: s.id,
                    step,
                    age: step.saturating_sub(s.founded_step),
                    peak: s.peak_members,
                });
            }
        }
    }

    /// Match one observed cluster against a tracked candidate (or start one),
    /// advance or reset its streak, and move it into probation once it has held
    /// its shape and its share long enough to be worth testing.
    fn observe_candidate(
        &mut self,
        group: Group,
        min_members: usize,
        mean_generation: f32,
        step: u32,
    ) {
        let qualifies_alone = group.count >= min_members && group.spread < SPREAD_MAX;

        let matched = self
            .candidates
            .iter_mut()
            .filter(|c| !c.seen_this_run)
            .map(|c| (dist_sq(&group.centroid, &c.centroid), c))
            .filter(|(d, _)| *d < CANDIDATE_MATCH_RADIUS * CANDIDATE_MATCH_RADIUS)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let Some((d_sq, c)) = matched else {
            self.candidates.push(Candidate {
                centroid: group.centroid,
                // A brand-new candidate has no measurable drift yet, so its
                // first run never counts toward the streak.
                streak: 0,
                streak_start_generation: mean_generation,
                probation_start_generation: None,
                members: group.count as u32,
                drift: f64::INFINITY,
                spread: group.spread,
                seen_this_run: true,
            });
            return;
        };

        let settled = d_sq < DRIFT_EPS * DRIFT_EPS;
        c.drift = d_sq.sqrt();
        c.spread = group.spread;
        c.centroid = group.centroid;
        c.members = group.count as u32;
        c.seen_this_run = true;

        if qualifies_alone && settled {
            c.streak += 1;
            // Entry: held its shape and its share long enough that the shape is
            // plausibly heritable rather than merely persistent.
            if !c.on_probation()
                && c.streak >= STABILITY_RUNS
                && mean_generation - c.streak_start_generation >= PROBATION_ENTRY_GENERATIONS
            {
                c.probation_start_generation = Some(mean_generation);
                self.events.push(SpeciesEvent::ProbationStarted { step, members: c.members });
            }
            return;
        }

        // Failed a criterion. Probation is over and the clamp lifts — a lineage
        // that could not hold its share while frozen is exactly what probation
        // exists to reject.
        if c.on_probation() {
            self.events.push(SpeciesEvent::ProbationFailed { step, members: c.members });
        }
        c.probation_start_generation = None;
        c.streak = 0;
        c.streak_start_generation = mean_generation;
    }

    /// The nearest species — live or extinct — close enough to be read as the
    /// same lineage. Extinct ones count: a lineage that re-radiates after a
    /// bottleneck should keep its family name even though the parent is gone.
    ///
    /// One lookup serves both the genus and `parent_id`, so a species can never
    /// carry one lineage's name and another's ancestry.
    fn nearest_kin(&self, centroid: &[f64; SIG_LEN]) -> Option<&Species> {
        self.species
            .iter()
            .map(|s| (dist_sq(centroid, &s.founding_centroid), s))
            .filter(|(d, _)| *d < GENUS_RADIUS * GENUS_RADIUS)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, s)| s)
    }

    fn promote_ready(
        &mut self,
        step: u32,
        mean_generation: f32,
        population: u32,
        population_centroid: &[f64; SIG_LEN],
    ) {
        let mut promoted: Vec<usize> = Vec::new();
        for (i, c) in self.candidates.iter().enumerate() {
            // Only a cluster that survived having its mutability taken away is
            // promoted, and only after holding on long enough under the clamp
            // for the freeze to have been a real test.
            let Some(start) = c.probation_start_generation else { continue };
            if mean_generation - start < PROBATION_TEST_GENERATIONS {
                continue;
            }
            // At the cap, refuse and let the candidate keep accumulating.
            if self.live_count() + promoted.len() >= MAX_SPECIES {
                continue;
            }
            promoted.push(i);
        }

        for &i in &promoted {
            let candidate = &self.candidates[i];
            let centroid = candidate.centroid;
            let members = candidate.members;
            let streak = candidate.streak;
            let drift = candidate.drift;
            let spread = candidate.spread;
            let streak_start_generation = candidate.streak_start_generation;
            let probation_start = candidate.probation_start_generation
                .expect("only probation candidates are promoted");
            let id = self.next_id;
            self.next_id += 1;

            let mut deviation = [0f64; SIG_LEN];
            for d in 0..SIG_LEN {
                deviation[d] = centroid[d] - population_centroid[d];
            }
            let taken: Vec<String> = self.species.iter().map(|s| s.name.full()).collect();
            // One lookup, two uses: the genus this lineage inherits and the id it
            // descends from. Borrow ends before the push below.
            let (parent_id, inherited_genus) = match self.nearest_kin(&centroid) {
                Some(kin) => (kin.id, Some(kin.name.genus.clone())),
                None => (UNASSIGNED, None),
            };
            let name = naming::generate(
                id,
                self.world_seed,
                &deviation,
                inherited_genus.as_deref(),
                &taken,
            );

            self.species.push(Species {
                name: name.clone(),
                id,
                parent_id,
                centroid,
                founding_centroid: centroid,
                founding_population_centroid: *population_centroid,
                founded_step: step,
                founder_generation: mean_generation,
                founder_members: members,
                founder_population: population,
                promotion_streak: streak,
                promotion_drift: drift,
                promotion_spread: spread,
                entry_generation_advance: probation_start - streak_start_generation,
                probation_generation_advance: mean_generation - probation_start,
                extinct_at: None,
                peak_members: members,
                members,
                empty_runs: 0,
            });
            self.events.push(SpeciesEvent::Promoted { id, name, step, members });
        }

        for &i in promoted.iter().rev() {
            self.candidates.remove(i);
        }
    }
}

/// One k-means cluster's worth of unassigned agents.
struct Group {
    centroid: [f64; SIG_LEN],
    spread: f64,
    count: usize,
}

/// Group the agents belonging to no species by their k-means label, merge the
/// labels that describe the same blob, and reduce each group to centroid,
/// spread, and size.
///
/// The merge step is load-bearing. `k = 6` is fixed while the number of real
/// blobs is not, so k-means routinely splits one lineage across two or three
/// labels; without merging, each fragment becomes its own candidate and the
/// same blob gets promoted two or three times at the same centroid. Merging
/// first is also what lets a species survive k-means reshuffling its slots,
/// which is the whole point of tracking centroids rather than labels.
fn unassigned_groups(
    points: &[[f64; SIG_LEN]],
    assignment: &[u32],
    cluster: &ClusterState,
) -> Vec<Group> {
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for (i, &a) in assignment.iter().enumerate() {
        if a != UNASSIGNED {
            continue;
        }
        // An agent with no cluster label (arrays out of sync between runs) is
        // skipped rather than defaulted into label 0, which would contaminate
        // that group's centroid.
        let Some(&label) = cluster.genome_cluster_ids.get(i) else { continue };
        let label = label as usize;
        if buckets.len() <= label {
            buckets.resize(label + 1, Vec::new());
        }
        buckets[label].push(i);
    }
    buckets.retain(|b| !b.is_empty());
    merge_nearby(&mut buckets, points);

    buckets
        .into_iter()
        .map(|members| {
            let count = members.len();
            let mut centroid = [0f64; SIG_LEN];
            for &i in &members {
                for d in 0..SIG_LEN {
                    centroid[d] += points[i][d];
                }
            }
            for c in centroid.iter_mut() {
                *c /= count as f64;
            }
            // Mean per-trait standard deviation: "is this a cluster or a bin".
            let mut spread = 0f64;
            for d in 0..SIG_LEN {
                let var = members
                    .iter()
                    .map(|&i| (points[i][d] - centroid[d]).powi(2))
                    .sum::<f64>()
                    / count as f64;
                spread += var.sqrt();
            }
            Group { centroid, spread: spread / SIG_LEN as f64, count }
        })
        .collect()
}

/// Agglomerative merge of buckets whose centroids sit within
/// `CANDIDATE_MATCH_RADIUS`. Closest pair first, repeated until nothing is close
/// enough; at k = 6 the pair scan is trivially cheap.
fn merge_nearby(buckets: &mut Vec<Vec<usize>>, points: &[[f64; SIG_LEN]]) {
    loop {
        let centroids: Vec<[f64; SIG_LEN]> =
            buckets.iter().map(|b| mean(b, points)).collect();

        let mut closest: Option<(f64, usize, usize)> = None;
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let d = dist_sq(&centroids[i], &centroids[j]);
                if d < CANDIDATE_MATCH_RADIUS * CANDIDATE_MATCH_RADIUS
                    && closest.is_none_or(|(best, _, _)| d < best)
                {
                    closest = Some((d, i, j));
                }
            }
        }

        match closest {
            Some((_, i, j)) => {
                let moved = buckets.remove(j);
                buckets[i].extend(moved);
            }
            None => return,
        }
    }
}

fn mean(members: &[usize], points: &[[f64; SIG_LEN]]) -> [f64; SIG_LEN] {
    let mut c = [0f64; SIG_LEN];
    if members.is_empty() {
        return c;
    }
    for &i in members {
        for d in 0..SIG_LEN {
            c[d] += points[i][d];
        }
    }
    for v in c.iter_mut() {
        *v /= members.len() as f64;
    }
    c
}

/// Member floor for the current population — see `MIN_MEMBERS_FRAC`.
pub fn min_members(alive: usize) -> usize {
    MIN_MEMBERS_FLOOR.max((alive as f64 * MIN_MEMBERS_FRAC).round() as usize)
}

/// The seven mutable traits, each rescaled to [0, 1] by its bounds.
///
/// Normalizing matters: raw euclidean distance lets wide-range traits dominate,
/// so `reproduction_cost` (0.75–1.50) would otherwise count for several times
/// what `defense` (0.5–1.07) does, and every threshold here would mean a
/// different thing per trait.
pub fn signature(t: &Traits) -> [f64; SIG_LEN] {
    let raw = [
        t.vision, t.speed, t.metabolism,
        t.energy_capacity, t.mutation_rate,
        t.reproduction_cost, t.attack, t.defense, t.aggression,
        t.intelligence,
    ];
    let mut out = [0f64; SIG_LEN];
    for (slot, &d) in out.iter_mut().zip(SIGNATURE_DIMS.iter()) {
        let (lo, hi) = Traits::BOUNDS[d];
        *slot = ((raw[d] - lo) / (hi - lo)).clamp(0.0, 1.0);
    }
    out
}

fn dist_sq(a: &[f64; SIG_LEN], b: &[f64; SIG_LEN]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Three well-separated trait blobs, the fixture shape from cluster.rs.
    /// `generation` is set explicitly so tests control the generation criterion.
    fn blobs(n: usize, generation: u32) -> Vec<Genome> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut genomes: Vec<Genome> = (0..n).map(|_| Genome::generate(&mut rng)).collect();
        for (i, g) in genomes.iter_mut().enumerate() {
            let level = match i % 3 { 0 => 0.5, 1 => 0.75, _ => 1.0 };
            let t = &mut g.traits;
            t.vision = level; t.speed = level.min(1.0); t.metabolism = level;
            t.attack = level; t.defense = level.min(1.07); t.aggression = level;
            t.intelligence = level.min(1.05);
            t.reproduction_cost = level + 0.75;
            g.generation = generation;
        }
        genomes
    }

    fn run(reg: &mut SpeciesRegistry, genomes: &[Genome], step: u32) -> Vec<u32> {
        let cluster = ClusterState::run(genomes, 6, step, None);
        reg.update(genomes, &cluster, step)
    }

    #[test]
    fn signature_excludes_locked_traits() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut g = Genome::generate(&mut rng);
        let before = signature(&g.traits);
        // Locked traits swing across their full range; the signature must not move.
        g.traits.energy_capacity = 1.05;
        g.traits.mutation_rate = 0.25;
        assert_eq!(before, signature(&g.traits));

        g.traits.aggression = 1.05;
        assert_ne!(before, signature(&g.traits));
    }

    #[test]
    fn signature_is_normalized() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..50 {
            let g = Genome::generate(&mut rng);
            for v in signature(&g.traits) {
                assert!((0.0..=1.0).contains(&v), "signature out of [0,1]: {}", v);
            }
        }
    }

    #[test]
    fn stable_blobs_promote_once_generations_advance() {
        let mut reg = SpeciesRegistry::new(42);
        // Hold the population still but advance generations, one per run.
        for r in 0..(STABILITY_RUNS + 3) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
        }
        assert_eq!(reg.live_count(), 3, "three separated blobs → three species");
        let ids: Vec<u32> = reg.all().iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2, 3], "ids are monotonic from 1");
        for species in reg.all() {
            assert_eq!(species.founder_population, 60);
            assert!(species.founder_members >= min_members(60) as u32);
            assert!(species.promotion_streak >= STABILITY_RUNS);
            assert!(species.promotion_drift < DRIFT_EPS);
            assert!(species.promotion_spread < SPREAD_MAX);
            assert!(species.entry_generation_advance >= PROBATION_ENTRY_GENERATIONS);
            assert!(species.probation_generation_advance >= PROBATION_TEST_GENERATIONS);
            assert_eq!(species.founding_centroid, species.centroid);
        }
    }

    #[test]
    fn probation_clamps_its_members_and_nobody_else() {
        let mut reg = SpeciesRegistry::new(42);
        // Advance to the point where clusters are under test but not yet named.
        let mut r = 0;
        while reg.probation_count() == 0 && r < 40 {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
            if reg.live_count() > 0 {
                break;
            }
            r += 1;
        }
        assert!(reg.probation_count() > 0, "no cluster reached probation");

        // A member of a probationary blob is clamped.
        let member = &blobs(60, r)[0];
        assert_eq!(reg.clamp_for(&member.traits), PROBATION_MUTATION_CLAMP);

        // An agent at the far corner of trait space belongs to no candidate.
        let mut outsider = member.clone();
        let t = &mut outsider.traits;
        t.vision = 1.05; t.speed = 1.0; t.metabolism = 0.5;
        t.reproduction_cost = 0.75; t.attack = 0.5; t.defense = 1.07; t.aggression = 1.05;
        assert_eq!(reg.clamp_for(&outsider.traits), 1.0);
    }

    #[test]
    fn promoted_species_are_named_and_names_are_unique() {
        let mut reg = SpeciesRegistry::new(42);
        for r in 0..(STABILITY_RUNS + 12) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
        }
        assert!(reg.live_count() > 0);
        let names: Vec<String> = reg.all().iter().map(|s| s.name.full()).collect();
        for n in &names {
            assert!(n.contains(' '), "not a binomial: {}", n);
            assert!(n.chars().next().unwrap().is_uppercase(), "genus not capitalized: {}", n);
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate names: {:?}", names);
    }

    #[test]
    fn names_replay_identically_for_a_seed() {
        let mut a = SpeciesRegistry::new(7);
        let mut b = SpeciesRegistry::new(7);
        let mut c = SpeciesRegistry::new(8);
        for r in 0..(STABILITY_RUNS + 12) {
            let genomes = blobs(60, r);
            run(&mut a, &genomes, 100 + r * 50);
            run(&mut b, &genomes, 100 + r * 50);
            run(&mut c, &genomes, 100 + r * 50);
        }
        let names = |reg: &SpeciesRegistry| -> Vec<String> {
            reg.all().iter().map(|s| s.name.full()).collect()
        };
        assert_eq!(names(&a), names(&b), "same seed must replay the same names");
        assert_ne!(names(&a), names(&c), "a different world seed must rename");
    }

    /// Build a species directly, to test genus inheritance without depending on
    /// two promotions happening to land near each other in a real run.
    fn planted(reg: &mut SpeciesRegistry, id: u32, centroid: [f64; SIG_LEN], genus: &str) {
        reg.species.push(Species {
            name: Name { genus: genus.to_string(), epithet: "ferox".into() },
            id,
            parent_id: UNASSIGNED,
            centroid,
            founding_centroid: centroid,
            founding_population_centroid: centroid,
            founded_step: 0,
            founder_generation: 0.0,
            founder_members: 1,
            founder_population: 1,
            promotion_streak: STABILITY_RUNS,
            promotion_drift: 0.0,
            promotion_spread: 0.0,
            entry_generation_advance: PROBATION_ENTRY_GENERATIONS,
            probation_generation_advance: PROBATION_TEST_GENERATIONS,
            extinct_at: None,
            peak_members: 1,
            members: 1,
            empty_runs: 0,
        });
    }

    #[test]
    fn a_nearby_lineage_lends_its_genus() {
        let mut reg = SpeciesRegistry::new(42);
        let base = [0.5; SIG_LEN];
        planted(&mut reg, 1, base, "Thalura");

        // Just inside the radius: same lineage, same genus.
        let mut near = base;
        near[0] += GENUS_RADIUS * 0.5;
        assert_eq!(reg.nearest_kin(&near).map(|s| s.name.genus.as_str()), Some("Thalura"));

        // Well outside: a new lineage, and a new genus.
        let mut far = base;
        for d in 0..SIG_LEN {
            far[d] += GENUS_RADIUS;
        }
        assert!(reg.nearest_kin(&far).is_none());
    }

    /// Drive a registry until something promotes, so ancestry is tested on real
    /// promotions rather than on planted rows.
    fn run_until_promotion(reg: &mut SpeciesRegistry) {
        for r in 0..(STABILITY_RUNS + 40) {
            let genomes = blobs(60, r);
            run(reg, &genomes, 100 + r * 50);
            if !reg.all().is_empty() { return; }
        }
    }

    #[test]
    fn a_lineage_founding_in_empty_space_is_a_root() {
        let mut reg = SpeciesRegistry::new(42);
        run_until_promotion(&mut reg);
        let first = reg.all().first().expect("expected at least one promotion");
        assert_eq!(first.parent_id, UNASSIGNED,
            "the first species in a pond has no kin to descend from");
    }

    #[test]
    fn ancestry_and_genus_name_the_same_kin() {
        // Whatever the parent is, the genus must have come from that same
        // species — one lookup feeds both, and this is what pins them together.
        //
        // Two passes: the first learns where this fixture promotes, the second
        // plants a lineage there so the promotion has kin to descend from. The
        // blobs are far enough apart in signature space that every promotion is
        // otherwise a root, which would prove nothing.
        let mut scout = SpeciesRegistry::new(42);
        run_until_promotion(&mut scout);
        let landing = scout.all().first().expect("expected a promotion").founding_centroid;

        let mut reg = SpeciesRegistry::new(42);
        // Far enough that the planted lineage does not simply absorb the members
        // (MEMBERSHIP_RADIUS), close enough to still be kin (GENUS_RADIUS).
        let mut planted_at = landing;
        planted_at[0] += (MEMBERSHIP_RADIUS + GENUS_RADIUS) / 2.0;
        planted(&mut reg, 900, planted_at, "Kinara");
        for r in 0..(STABILITY_RUNS + 40) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
            if reg.all().iter().any(|s| s.id != 900) { break; }
        }

        let child = reg.all().iter().find(|s| s.id != 900)
            .expect("expected a promotion beside the planted lineage");
        assert_eq!(child.parent_id, 900, "{} should descend from the planted lineage",
            child.name.full());
        assert_eq!(child.name.genus, "Kinara",
            "ancestry and genus must name the same kin");
    }

    #[test]
    fn an_extinct_lineage_can_be_a_parent() {
        // Same rule as genus inheritance: a lineage that re-radiates after a
        // bottleneck descends from the one that died.
        let mut reg = SpeciesRegistry::new(42);
        let base = [0.5; SIG_LEN];
        planted(&mut reg, 1, base, "Vorixa");
        reg.species[0].extinct_at = Some(500);

        let mut near = base;
        near[2] += GENUS_RADIUS * 0.4;
        assert_eq!(reg.nearest_kin(&near).map(|s| s.id), Some(1));
    }

    #[test]
    fn the_nearest_kin_wins_not_the_first() {
        let mut reg = SpeciesRegistry::new(42);
        let base = [0.5; SIG_LEN];
        planted(&mut reg, 1, base, "Thalura");
        let mut closer = base;
        closer[0] += GENUS_RADIUS * 0.2;
        planted(&mut reg, 2, closer, "Thalura");

        let mut probe = base;
        probe[0] += GENUS_RADIUS * 0.25;
        assert_eq!(reg.nearest_kin(&probe).map(|s| s.id), Some(2));
    }

    #[test]
    fn an_extinct_lineage_still_lends_its_genus() {
        // A lineage that re-radiates after a bottleneck keeps its family name
        // even though the parent is gone.
        let mut reg = SpeciesRegistry::new(42);
        let base = [0.5; SIG_LEN];
        planted(&mut reg, 1, base, "Vorixa");
        reg.species[0].extinct_at = Some(500);

        let mut near = base;
        near[2] += GENUS_RADIUS * 0.4;
        assert_eq!(reg.nearest_kin(&near).map(|s| s.name.genus.as_str()), Some("Vorixa"));
    }

    #[test]
    fn probation_precedes_every_promotion() {
        let mut reg = SpeciesRegistry::new(42);
        let mut saw_probation = false;
        for r in 0..(STABILITY_RUNS + 12) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
            for ev in reg.drain_events() {
                match ev {
                    SpeciesEvent::ProbationStarted { .. } => saw_probation = true,
                    SpeciesEvent::Promoted { .. } => assert!(
                        saw_probation,
                        "a species was promoted without ever being tested",
                    ),
                    _ => {}
                }
            }
        }
        assert!(reg.live_count() > 0, "nothing promoted at all");
    }

    #[test]
    fn no_promotion_without_generation_advance() {
        let mut reg = SpeciesRegistry::new(42);
        // Identical populations, frozen at generation 0: perfectly stable
        // clusters, but nothing was ever inherited, so nothing is a species.
        for r in 0..(STABILITY_RUNS + 5) {
            let genomes = blobs(60, 0);
            run(&mut reg, &genomes, 100 + r * 50);
        }
        assert_eq!(reg.live_count(), 0);
    }

    #[test]
    fn drifting_blobs_never_promote() {
        let mut reg = SpeciesRegistry::new(42);
        for r in 0..(STABILITY_RUNS + 5) {
            let mut genomes = blobs(60, r);
            // One tight blob walking steadily across trait space. Offsets are
            // chosen to stay well inside every bound: a drift that clamps stops
            // drifting, and a settled cluster is supposed to promote.
            let step = r as f64;
            for g in genomes.iter_mut() {
                let t = &mut g.traits;
                t.aggression = 0.05 * step;         // per-run drift 0.048 normalized
                t.attack = 0.5 + 0.03 * step;       //                0.040
                t.vision = 0.5 + 0.02 * step;       //                0.036
            }
            run(&mut reg, &genomes, 100 + r * 50);
        }
        assert_eq!(reg.live_count(), 0, "a moving cluster is not a lineage");
    }

    #[test]
    fn members_assign_to_nearest_species_and_outliers_stay_unassigned() {
        let mut reg = SpeciesRegistry::new(42);
        for r in 0..(STABILITY_RUNS + 3) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
        }
        assert_eq!(reg.live_count(), 3);

        let mut genomes = blobs(60, 20);
        // One agent parked at a corner of trait space belongs to nobody.
        let t = &mut genomes[0].traits;
        t.vision = 0.5; t.speed = 0.5; t.metabolism = 1.05;
        t.reproduction_cost = 1.50; t.attack = 1.25; t.defense = 0.5; t.aggression = 0.0;

        let assignment = run(&mut reg, &genomes, 1000);
        assert_eq!(assignment[0], UNASSIGNED);
        let assigned = assignment.iter().filter(|&&a| a != UNASSIGNED).count();
        assert!(assigned > 50, "the rest still belong somewhere: {}", assigned);
    }

    #[test]
    fn extinction_is_recorded_and_ids_are_never_reused() {
        let mut reg = SpeciesRegistry::new(42);
        for r in 0..(STABILITY_RUNS + 3) {
            let genomes = blobs(60, r);
            run(&mut reg, &genomes, 100 + r * 50);
        }
        let founded = reg.live_count();
        assert_eq!(founded, 3);
        let max_id = reg.all().iter().map(|s| s.id).max().unwrap();

        // Replace the population with agents far from every species centroid.
        for r in 0..EXTINCTION_RUNS + 1 {
            let mut genomes = blobs(60, 20 + r);
            for g in genomes.iter_mut() {
                let t = &mut g.traits;
                t.vision = 1.05; t.speed = 1.0; t.metabolism = 0.5;
                t.reproduction_cost = 0.75; t.attack = 0.5; t.defense = 1.07;
                t.aggression = 1.05;
            }
            run(&mut reg, &genomes, 1000 + r * 50);
        }
        assert_eq!(reg.live_count(), 0, "abandoned species go extinct");
        assert_eq!(reg.all().len(), 3, "extinct species stay in the record");
        assert!(reg.all().iter().all(|s| s.extinct_at.is_some()));

        // Bring the original blobs back: convergent, so new ids, not a revival.
        for r in 0..(STABILITY_RUNS + 3) {
            let genomes = blobs(60, 40 + r);
            run(&mut reg, &genomes, 2000 + r * 50);
        }
        assert!(reg.live_count() > 0, "the old shape re-promotes");
        assert!(
            reg.all().iter().filter(|s| s.is_alive()).all(|s| s.id > max_id),
            "revived shapes get fresh ids",
        );
    }

    #[test]
    fn live_species_never_exceed_the_cap() {
        let mut reg = SpeciesRegistry::new(42);
        for r in 0..40u32 {
            let mut rng = ChaCha8Rng::seed_from_u64(r as u64);
            let genomes: Vec<Genome> = (0..200)
                .map(|_| {
                    let mut g = Genome::generate(&mut rng);
                    g.generation = r;
                    g
                })
                .collect();
            run(&mut reg, &genomes, 100 + r * 50);
            assert!(reg.live_count() <= MAX_SPECIES, "cap breached: {}", reg.live_count());
        }
    }

    #[test]
    fn min_members_is_relative_with_a_floor() {
        // Derived from the constants, not hardcoded: the fraction is a tuning
        // knob and a test that pins its value would just have to be edited
        // alongside every change to it.
        assert_eq!(min_members(1), MIN_MEMBERS_FLOOR, "the floor holds at low population");
        let big = 10_000;
        assert_eq!(min_members(big), (big as f64 * MIN_MEMBERS_FRAC).round() as usize);
        assert!(min_members(big) > MIN_MEMBERS_FLOOR, "the fraction dominates at high population");
    }

    #[test]
    fn empty_population_is_handled() {
        let mut reg = SpeciesRegistry::new(42);
        let cluster = ClusterState::run(&[], 6, 50, None);
        assert!(reg.update(&[], &cluster, 50).is_empty());
    }
}
