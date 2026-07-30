# Plan — Named Species: Locking In Stable Lineages

Goal: make speciation legible. Right now a "family" is a k-means label that
exists because `k = 6` was chosen, not because anything in the population is
actually distinct. Labels persist across re-runs only by centroid matching
(`cluster.rs:64`), which prevents color flashing but confers no real identity —
family 3 at step 200 and family 3 at step 2000 may share nothing but a slot.

Target behavior: a genome cluster that stays put for several generations gets
**promoted** to a species with a generated name (e.g. *Vorix pallida*), a fixed
color, a founding step, and a lineage record that survives even after it goes
extinct. Unpromoted agents stay in a generic unassigned pool. The pond stops
looking like six arbitrary color bins and starts looking like it has a history.

---

## What exists

- `ClusterState::run` (`cluster.rs:31`) — dual k-means, genome (9 traits,
  euclidean, k=6) and brain (488 weights, cosine, k=24), rebuilt every 50 steps.
- `genome_centroids: Vec<Option<[f64; 9]>>` — retained per run, used only for
  label matching.
- `cluster_composite()` (`wasm.rs:206`) — per-family mean/sd for traits and
  per-layer brain weight magnitude. The "how converged is this family" number
  already exists; nothing consumes it as a *decision*.
- No generation counter on `Genome`. This is the one genuinely missing input.

---

## Step 0 — Generation counter

Add `pub generation: u32` to `Genome` (`genome.rs:78`). `generate()` sets 0;
`mutate()` sets `parent.generation + 1`. Two lines, and it makes "stable for
three generations" expressible rather than approximated by step count.

Do not confuse generation with age — a long-lived agent that never reproduces is
generation 0 forever, which is correct.

---

## Step 1 — Stability criteria

A cluster is a **species candidate** at each cluster run. Promotion requires all
of the following to hold for `STABILITY_RUNS` (default 5 runs = 250 steps):

| Criterion | Default | Why |
|---|---|---|
| Membership ≥ `MIN_MEMBERS` | 12 agents | Below this, k-means is fitting noise |
| Centroid drift < `DRIFT_EPS` per run | 0.04 (normalized trait space) | The lineage isn't still moving |
| Within-cluster spread < `SPREAD_MAX` | 0.25 mean per-trait sd | It's actually a cluster, not a bin |
| Mean generation advanced ≥ `MIN_GENERATIONS` | 3 | It survived reproduction, not just time |

The last one is the one that makes this *speciation* rather than *clustering*.
A cluster can sit still for 250 steps because its members are long-lived, not
because the shape is heritable. Requiring the mean generation to advance means
the trait signature was passed down and re-selected.

**Normalize trait space before measuring drift.** Traits have different ranges
(`trait_bounds()`, `wasm.rs:362`); raw euclidean distance in `cluster.rs:112`
lets wide-range traits dominate. Promotion thresholds must be range-normalized
or the criteria mean different things per trait. This also arguably improves the
clustering itself — worth benchmarking separately, but do not change
`kmeans_genome`'s metric in the same commit as this feature. One lever at a time.

---

## Step 2 — `pond_core/src/species.rs` (new module)

```rust
pub struct Species {
    pub id: u32,                       // monotonic, never reused
    pub name: String,                  // generated at promotion
    pub centroid: [f64; 9],            // trait signature at promotion
    pub color_seed: u32,               // stable color, independent of k-means slot
    pub founded_step: u32,
    pub founder_generation: u32,
    pub extinct_at: Option<u32>,
    pub peak_members: u32,
    pub members: u32,                  // current, recomputed per cluster run
}

pub struct SpeciesRegistry {
    species: Vec<Species>,             // includes extinct — this is the fossil record
    candidates: HashMap<u8, Candidate>,// keyed by current k-means label
    next_id: u32,
}
```

`Candidate` holds the consecutive-run counter, the last centroid, and the
generation at first sighting. Reset the counter whenever any criterion fails —
stability must be *consecutive*, otherwise a cluster that oscillates gets
promoted by accumulation.

### Assignment after promotion

Once species exist, agent → species is **not** the k-means label. Assign by
nearest species centroid within `MEMBERSHIP_RADIUS` (default 0.35 normalized);
agents outside every radius are unassigned (species id 0, rendered neutral grey).

This is the payoff of the whole design: a species keeps its identity even if
k-means reshuffles its labels, splits it across two slots, or merges it with a
neighbor. It also means an agent can *leave* a species by drifting, which is
what should happen.

### Drift and extinction

- Species centroid tracks its members slowly (`centroid += 0.05 * (member_mean - centroid)`
  per cluster run). Fast tracking would let a species follow its own members
  anywhere and never go extinct; zero tracking would strand it as members drift.
- `members == 0` for two consecutive runs → set `extinct_at`, stop tracking.
  Keep the record forever. An extinct species reappearing is a *new* species
  with a new id and name, even at the same centroid — convergent evolution is
  the more interesting reading, and resurrection would make the timeline lie.
- Cap live species at `MAX_SPECIES` (default 12). At the cap, promotion requires
  beating the weakest live species on membership. Without a cap, a long run
  accumulates species faster than the legend can show them.

---

## Step 3 — Name generation

Binomial, deterministic from `(species_id, world_seed)` — a replay of the same
seed must produce the same names, so use a `ChaCha8Rng` seeded from those two,
never a global RNG and never the world's RNG stream (drawing from it would shift
every subsequent simulation draw and break determinism against existing traces).

Construction: syllable tables, genus = 2–3 syllables capitalized, species
epithet = 1–2 syllables, biased by the promoted centroid's dominant trait so the
name carries information:

| Dominant trait | Epithet pool |
|---|---|
| aggression / attack | *ferox, rapax, atrox* |
| speed | *velox, fugax, celer* |
| vision | *lucida, vigilis* |
| metabolism | *ardens, avida* |
| defense | *thorax, munita* |
| energy_capacity | *opima, gravis* |

Genus syllables stay pure nonsense (*vor, ix, thal, mek, sura*) so the genus
reads as lineage identity and the epithet as ecology. Guarantee uniqueness by
re-drawing on collision within a run.

**Gender convention (decided):** genera are always **feminine** — the base
taxonomy is one gender, so genus names read as a consistent family. Epithets
**mix masculine and feminine suffixes** (*-us* / *-a*) for variety across
species names. This deliberately departs from strict Latin agreement, where the
epithet would have to match the genus; the variety is worth more here than the
grammar. Optional refinement if we want the suffix to carry information rather
than just vary: use the two genders to encode the **direction** of the trait
deviation (high vs low), which the epithet pools need to distinguish anyway.

**Genus is inherited, not per-species (decided):** at promotion, find the
nearest species by founding centroid — live or extinct — and within a
`GENUS_RADIUS`, inherit its genus and take a new epithet. Otherwise mint a new
genus. Deriving genus from `species_id` alone would give two species that split
from one ancestor unrelated genus names, making the naming actively lie about
descent; inheriting it means a re-radiation after a bottleneck reads as one
genus fanning out.

**Epithet is chosen by signed deviation from the population centroid**, not by
the largest normalized trait value. Argmax over the centroid just returns
whichever trait sits high in the population as a whole, so in an aggressive pond
every species ends up *ferox* and the epithet carries no information. Deviation
names what makes this lineage *different*, and the sign means each trait needs a
high word and a low word.

One correction to the table above: *thorax* is a noun, not an adjective, and
does not work as an epithet the way the others do. Use *munita*, *loricata*
(armoured), or *scutata* (shielded) for the defense slot. `reproduction_cost` is
in the signature and can be the most deviant trait, so it needs a pool too —
*fecunda* / *parca* (fertile / sparing) covers both directions.

Color: derive from `color_seed` through the existing OKLCH path
(`pond_web/color.js`), holding lightness and chroma in the band the renderer
already uses so species colors sit next to each other legibly. Reserve one grey
for unassigned.

---

## Step 4 — Export and UI

Add to the wasm boundary, flat-buffer convention as everywhere else:

```rust
pub fn species_list(&self) -> Vec<f32>;   // [id, color_seed, members, peak, founded, extinct, 9 centroid]
pub fn species_names(&self) -> Vec<JsValue>; // parallel, strings can't ride a Float32Array
```

Per-agent species id: add `A_SPECIES` to the agent stride in `wasm.rs`
(currently 18 → 19). `A_GENOME_CLUSTER` stays — the raw k-means label is still
useful for debugging the promotion logic, and removing it in the same change
would confound "is speciation wrong" with "is clustering wrong".

UI changes in `pond_web`:
- Legend (`panels.js:52`) lists **named species**, not "family N", with member
  count and age in steps. Unassigned agents get one grey row at the bottom.
- Promotion event: a one-line toast — *"Vorix pallida emerged — step 1240, 18 members"*.
  This is the moment the whole feature exists to produce; it should be visible
  without the panel open.
- Extinction: strike the legend row through, keep it visible for ~30 s, then
  move it to a collapsed "fossil record" list.
- Inspector (`inspector.js`) shows the agent's species name and its distance
  from the species centroid — that number is how you see an individual drifting
  out of its species before the population does.

---

## Step 5 — Tuning

Every threshold above is a guess. They interact with population dynamics that
already collapse around step 200 in some configs, so:

- Run headless at several seeds, log promotion/extinction events with the
  `PLAN_GRAPHS.md` CSV dump, and check the promotion rate. Target: 2–5 species
  alive in a healthy run, first promotion somewhere in steps 300–800.
- Too many promotions → raise `MIN_GENERATIONS` first, not `MIN_MEMBERS`.
  Generation depth is the criterion that distinguishes real lineages; member
  count just filters small ones.
- Zero promotions → the population may genuinely be panmictic, i.e. one
  interbreeding blob with no structure. Check `cluster_composite` spreads before
  loosening thresholds. If every family's spread is near the population spread,
  there is nothing to promote and lowering the bar would only invent species
  that aren't there.

### Tuning pass — run 2026-07-29

Six seeds, 12×12, 150 starting agents, 3000 steps, release build:

| seed | live species at 3000 | first promotion | final pop |
|---|---|---|---|
| 42 | 1 | step 2250 | 146 |
| 7 | 3 | step 2050 | 61 |
| 1337 | 1 | step 2750 | 66 |
| 2024 | 1 | step 1900 | 25 |
| 555 | 1 | step 2700 | 122 |
| 99 | 2 | step 1600 | 39 |

Re-run after `cluster.rs` moved to the normalized signature: this table is
unchanged, seed for seed. (At the smaller 12×12/100-agent config the change does
move things — seed 42 reaches probation at step 1050 instead of 1900 — so the
better-conditioned partition finds stable structure sooner where the population
is thinner.)

**Promotion count is fine; the expected window above was arithmetically
impossible.** Live counts land at 1–3 against a 2–5 target — slightly under, and
plausibly just the small grid. But no seed promoted before step 1600, and none
could have:

`mean_generation` from the stats CSV (seed 42) reaches 0.5 at step 250, 1.6 at
500, 4.0 at 750, and climbs about 1 generation per 250 steps after that.
Promotion needs `PROBATION_ENTRY_GENERATIONS` (3) of *advance* on top of a stable
streak, then `PROBATION_TEST_GENERATIONS` (1) more under the clamp — four
generations of advance measured from when the streak began, and the streak cannot
begin before the cluster holds still. Four generations of advance do not exist
until roughly step 1000 at the absolute earliest, and in practice the stability
streak starts later than that. **Steps 300–800 was never reachable** with
maturity at 100 ticks and per-agent reproduction cooldowns.

**Recommendation: leave the thresholds alone and revise the expectation to
~1600–2800.** The criteria are doing what they were designed to do — that a
lineage must persist across real generational turnover is the whole point, and
the fix for "promotion feels late" is a faster-turning population (lower
`MATURITY_AGE`), not a shorter generation requirement. If the window still wants
shortening, lower `PROBATION_ENTRY_GENERATIONS` from 3 to 2 first and re-run this
table; do not touch `MIN_MEMBERS_*`, which is not what is gating here.

Not yet measured: whether 1–3 live species is structure or is `k = 6` bracketing
one blob. That needs `cluster_composite` spreads compared against the population
spread, per the note above.

---

## Order

0 (generation counter, its own commit) → 1–2 (registry + promotion, headless,
logged to stdout only) → 3 (names/colors) → 4 (export + legend) → 4b (toast,
fossil record, inspector) → 5 (tuning pass).

Steps 1–2 should land with **no** renderer changes. Watching promotion events
scroll past in a headless run is how you find out whether the criteria are
sensible; wiring the UI first only makes bad thresholds prettier.

## Tests

- Synthetic populations with three well-separated blobs (the fixture in
  `cluster.rs:283` already builds one) promote exactly three species once
  generations advance past the threshold.
- A drifting blob never promotes while drift exceeds `DRIFT_EPS`.
- Species ids are never reused; an extinct species that reappears at the same
  centroid gets a new id.
- Name generation is deterministic given `(species_id, world_seed)` and unique
  within a run.
- Promotion consumes no draws from the world RNG: run 500 steps with speciation
  on and off, assert identical `get_stats()` output.
