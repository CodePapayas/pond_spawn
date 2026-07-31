# RULES.md

> **Canonical implementation: Rust `pond_core`.** These rules were refactored
> from the original Python sim to the Rust engine (see
> [`REFACTOR_RUST_ROADMAP.md`](REFACTOR_RUST_ROADMAP.md) and
> [`pond_core/README.md`](pond_core/README.md)). Where a mechanic differs
> between the two, **`pond_core` is authoritative** and the difference is
> tagged inline below. The biggest refactor change: discrete tile actions
> (MOVE/TURN/EAT/…) became **continuous-space steering forces + sigmoid trigger
> gates**. The action table below describes the legacy Python output surface;
> the Rust output contract is in the [Neural Network](#neural-network-brain)
> section and in `pond_core/README.md`.

## World

- Grid: Square grid, default 12x12
- Toroidal map (edges wrap)
- Each tile is a biome with properties:
  - `movement_speed`: 0.8–1.05
  - `visibility`: 0.25–1.0
  - `fertility`: 0.01–1.6
  - `food_units`: 0–3 (initial; barren tiles start at 0, fertile tiles 0–3)

## Food

- Each tile regenerates food passively every tick
- Regen rate per tile: `fertility / 1.6 × food_regen_scale` (per-tick probability of gaining 1 food unit)
- `food_regen_scale` is a **dial**, default **0.012**, range 0–0.05 (`Tunables`), set in run setup and fixed for the life of that run
- Max food per tile: **3**
- 35–45% of tiles are permanently barren (fertility = 0, never regenerate), arranged in contiguous desert clusters with fertile oases between them
- Each food unit provides **33.3 energy**

### Defense upkeep

Armour is a physical addition and costs metabolism to carry:

- **Per tick**: `DEFENSE_UPKEEP (0.09) × armour margin × metabolism`, where the
  margin is `defense - 0.5` — the floor of the trait's range is a body, not
  plating, so an agent at the bottom pays nothing and one at the top pays half of
  `BASE_DRAIN`.
- **Per attack survived**: `DEFENSE_BLOCK_COST (0.6) × armour margin ×
  metabolism`. Bracing is work, and a heavily armoured animal in a crowded pond
  pays it over and over.

Before this, nothing anywhere charged for `defense` while ordinary combat paid
out for it directly, so mean defense climbed in every run and no predator change
could stop it. Isolated over five seeds it took mean defense from **0.932 to
0.709**, down in every seed, and carried attack (1.041 → 0.870), aggression
(0.698 → 0.526) and speed (0.871 → 0.683) down with it — armour was propping up
a whole aggressive strategy. Starvation deaths doubled and the population rose,
which is what happens when the pond stops paying for plate it does not need.

## Agents ("Callums")

### Initialization
- Starting energy: 100
- Starting heading: random (N/E/S/W)
- Assigned a death age at birth

### Perception (Brain Inputs)

**Rust `pond_core` (canonical):**
1. `energy_norm` — own energy / (100 × energy_capacity trait), 0–1
2. `food_dist_norm` — distance to nearest food / vision radius (1.0 = none visible)
3. `food_angle_norm` — angle to food relative to velocity direction, in [−1, 1]
4. `agent_density_norm` — neighbours within separation radius / 8, 0–1
5. `speed_norm` — current speed / max speed, 0–1

**Legacy Python:**
1. Normalized energy (0–1)
2. Normalized food at tile (0–1)
3. Normalized nearby agent count (0–1)
4. Visibility factor
5. Movement factor

### Actions (Brain Outputs)

> **Legacy Python only — none of this table runs in `pond_core`.** The Rust
> engine's output surface is steering weights plus sigmoid trigger gates; index 7
> is the sleep gate, not ATTACK, and indices 3 and 6 are dormant. See
> [Neural Network](#neural-network-brain) for the live contract. The table is
> kept because the golden-parity harness still replays it.

| Index | Action | Notes |
|-------|--------|-------|
| 0 | MOVE | Move in heading direction |
| 1 | TURN_RIGHT | Rotate 90° clockwise |
| 2 | EAT | Consume food at current tile |
| 3 | REPRODUCE | Create offspring |
| 4 | SLEEP | Gain energy |
| 5 | NOTHING | Minimal energy burn; skip next tick |
| 6 | TURN_LEFT | Rotate 90° counter-clockwise |
| 7 | ATTACK | Attack an agent on the same tile |

### Decision Override Rules

**Legacy Python only.** `pond_core` has no override layer — crowding and hunger
reach behaviour through perception inputs and the separation force instead.

- Energy < 25% AND no food → forced MOVE
- Food > 0 AND nearby_agents > (food × 2 + 1) → forced MOVE

### Energy Costs
| Action | Cost |
|--------|------|
| Base metabolism (per tick) | `0.1 × metabolism` |
| Move | `terrain_speed × speed × metabolism × 0.15` |
| Turn | `0.14 × metabolism` (`turn()` = 0.1 + `execute_action` = 0.04) |
| Reproduce | `energy × 0.50 × reproduction_cost` |
| Sleep | Recover `0.05 × metabolism` (`SLEEP_RECOVERY`), clamped to capacity |
| Nothing | `0.005 × metabolism`; agent skips next tick |

### Sleep is rest, not recovery (Rust `pond_core`)
- Sleep recovers **less** than the tick's base metabolism drain
  (`0.05 × metabolism` against `0.1 × metabolism`), so choosing it halves the
  rate an agent starves at and can never be a net energy source. The gain is
  clamped to `100 × energy_capacity`.
- It was a gain of `0.15 × metabolism`, larger than the drain and unclamped, so
  an agent whose sleep gate kept winning gained energy indefinitely and past its
  own capacity. `energy_norm` clamps to 1.0 in perception, so neither the brain
  nor the HUD ever showed the overflow — the only symptom was agents that could
  not starve.

### Discrete Trigger Exclusivity (Rust `pond_core`)
- EAT, REPRODUCE, and SLEEP gates are mutually exclusive per tick: only the
  highest-value gate above 0.5 fires. An agent cannot sleep for free while also
  eating/reproducing/moving at full speed in the same tick.

### Eat Crowding & Cooldown (Rust `pond_core`)
- A tile that was just eaten from cannot be eaten from again for **8 ticks**
  (`EAT_COOLDOWN_TICKS`), independent of food regen — stops a single parked
  agent draining one tile every tick.
- Agents eating the same tile in the same tick split that tile's food value:
  `share = 33.3 / (1 + prior_claims_this_tick)`.

### Reproduction
- Minimum age: 100 ticks (no upper age limit — cooldown and cap govern timing)
- Cost: 50% of energy × `reproduction_cost` trait; paid before outcome is resolved
- **Max offspring cap**: assigned at birth, random 1–10 per agent; reproduction blocked once reached
- **Cooldown**: `(death_age - 100) // max_offspring` ticks between births
- **Birth failure**: 2% chance attempt produces no offspring; on failure, 20% chance it still burns one slot
- Offspring energy: **40% of the energy cost paid by the parent** (`BIRTH_ENERGY_YIELD`). The other 60% is thermodynamic overhead, lost to the world — reproduction is not an energy-neutral transfer. Without this loss a higher `reproduction_cost` bought a better-provisioned child for free, so selection drove the trait to its 1.50 bound and population growth had no brake but food.
- Offspring placed on random adjacent tile (wrapping)

### Death
- Energy ≤ 0
- Reaching assigned death age
- Killed in combat
- **Smitten** — killed by a player god-mode action (comet, salt, sweep). Tallied
  as its own cause so the death graph never attributes an act of god to the
  ecology.

### God mode (player powers, outside the simulation's rules)
- **Comet** — kills every agent within 2.2 world units of the click, instantly.
- **Salt** — kills in a ring that widens to 5.5 world units over 5 s, driven by
  wall-clock time rather than sim steps, so it keeps spreading while paused.
- **Sweep clean** — a column crosses the pond over 1.8 s, killing as it passes,
  then empties whatever remains.
- **Ultra predators** — apex agents, summoned by the player or triggered
  automatically. They cannot die (no aging, no starvation, and they are excluded
  from ordinary combat entirely, as attacker and as victim) and hunt down one
  agent at a time. Kills count as `EatenAlive`.

  - **Three shapes, and only one of them is the ecology's.**

    | Tier | Shape | Speed | Turn | Reach | Attack | Spin | Summoned by | Behaviour |
    |---|---|---|---|---|---|---|---|---|
    | 0 | grey triangle | 0.95 | 0.20 rad/step | 0.55 | 0.75 | — | the pond | resident |
    | 1 | red octagon | 1.30 | 0.15 rad/step | 1.90 | 0.90 | 0.20 rad/step | player only | hit and run |
    | 2 | rainbow rectangle | 1.55 | 0.09 rad/step | 3.20 | 1.05 | 0.045 rad/step | player only | hit and run |

    Speed is world units per step; `Turn` (`TIER_MAX_TURN`) is the most a hunter
    may swing its heading in one. `Spin` is rotation of the kill shape itself,
    which is independent of where the hunter is going.

    Every covered prey gets a deterministic defense check: the predator eats it
    only when the tier's flat attack is strictly greater than the prey's
    effective defense. Equality survives. **Tier 2 attacks along its edges**: an
    oriented rectangle of half-length `reach` and half-width 0.85.

    **The rectangle does not roll** (`TIER_IGNORES_DEFENSE`). Anything its edges
    touch dies, whatever its defense. At attack 1.05 it was losing bites to
    well-armoured agents — defense runs to 1.07 from the trait bounds alone, and
    the childhood bonus stacks on top — so the most final power in the game could
    sweep the pond and leave survivors standing in its path. The tiers below it
    still roll: a triangle losing to armour is what makes defense worth evolving.

  - **They swim, they do not teleport.** A hunter's velocity is state that
    carries between ticks. Every motion it makes — chase, patrol, departure —
    steers that velocity toward where it wants to go, turning at most the tier's
    `TIER_MAX_TURN` per step and easing its speed by `PREDATOR_SPEED_EASE` (0.12)
    of the remaining gap per step. So a hunter banks onto a target, accelerates
    into its first chase from a standing start, and slows as it goes quiet.

    Turn limits cost nothing in lethality — a cull of ~200 prey takes the same
    70–90 steps whatever the turn rate, because a hunter in a dense pond eats
    whatever its kill shape sweeps, aimed or not, and it closes at 19 tiles/s
    against a prey top speed of 3. What the number buys is the shape of the path:
    the turn radius is `speed / turn`, so the triangles bank through 4.7 world
    units and the rectangle through 17.

  - **A hunter commits to one animal.** It holds a target for
    `PREDATOR_COMMIT_TICKS` (18) before looking again, and drops it early only if
    the target dies, resists a bite, or gets more than `PREDATOR_COMMIT_RANGE`
    (12 world units) away. Re-picking the nearest prey every tick made the
    heading alternate between two equidistant animals; commitment also means a
    hunt has a subject you can watch.

  - **A resistant animal is skipped once, not forever.** A failed bite drops both
    the skip target and the commitment, so the hunter picks something else next
    tick. If the resistant animal is the only prey left, the skip is dropped and
    it is taken again rather than deadlocking the hunt.

  - **Patrol is a smooth arc.** An idle hunter random-walks its turn *rate*
    (`PATROL_TURN_MEMORY` 0.92 carry-over, `PATROL_TURN_NOISE` noise, clamped to
    `PATROL_TURN_MAX` 0.12 rad/step) at `PATROL_SPEED_FRAC` (0.45) of its hunting
    speed. Drawing a fresh turn each tick reads as vibration at 20 Hz, and
    patrolling is what a resident does most of its life.

  - **Predators have no brain phase.** They are excluded from perception,
    steering, and the discrete triggers — an apex predator does not forage, sleep
    or breed, and its hunt is its whole behaviour. They still age and still pay
    passive metabolism (floored at 1.0 energy, since they cannot starve).

  - **The ecology only fields triangles.** No automatic rule ever puts an octagon
    or a rectangle in the water. They are far too lethal to hand to a threshold —
    the pond would never get to overshoot and recover, which is the thing worth
    watching. They exist as god-mode powers and nothing else.

  - **Player toggle.** Automatic triangle ecology is on by default and can be
    disabled from the normal HUD. Turning it off prevents arrivals and
    reinforcements, sends existing automatic residents away, and resets the
    automatic pack ratchet. Player-summoned shapes are unaffected.

  - **Summoned hunts have an off switch.** `dismiss hunters` in the god panel
    sends every player-summoned shape away mid-hunt, over the usual departure
    swim rather than blinking them out. It is the counterpart to the HUD toggle,
    which spares player summons by design — without it an octagon or a rectangle
    could only be waited out. The automatic residents are left alone: they are
    the ecology, not a player power. Closing the god panel does *not* dismiss
    anything already in the water.

  - **The pack grows by one per threshold crossing.** Each time the pond climbs
    past the trigger, every resident wakes and one more triangle joins them
    permanently, up to `PREDATOR_MAX`. A pond that keeps outbreeding its hunters
    accumulates them, so the pack size is a running record of how often it has
    done so. The first wave arrives at `TIER_PACK[0]` (3).

  - **Residency.** The triangles never leave. Once their quota is met they go
    quiet and patrol, and wake when the population climbs back over the trigger —
    they are not aggressive unless the population needs culling. The two player
    shapes are far too lethal to leave in the water: they cull and depart.

  - **Summoned**: the octagon (`PREDATOR_MANUAL_TIER`) or the rectangle
    (`PREDATOR_RECTANGLE_TIER`), each culling to 20% of the population at the
    moment it is called. A summon is a strike that lands and leaves; summoning at
    the resident tier would park another permanent hunter in the pond on every
    press.

  - **Automatic**: the pond has a capacity of `1.75 × tiles`
    (`PREDATOR_POP_PER_TILE`), capped absolutely at 900 agents
    (`PREDATOR_POP_CEILING`) however large the pond is — so a 12×12 pond caps at
    252 and anything from 23×23 up holds the 900 line. Food supply scales with
    area, so a fixed number would either strangle the big pond or never fire on
    the small one; the ceiling exists because past roughly 900 individually drawn
    bodies the frame rate goes regardless of how much food there is. A wave
    arrives — the triangle pack — when the prey count passes `cap × 1.10`.

  - **Culls cut deep.** The target is `cap × 0.90 × 0.72` (`CULL_DEPTH`), well
    below the hysteresis floor. Landing on the floor exactly made no sense
    against how this pond grows — a boom re-crossed the trigger almost
    immediately, so predators were effectively permanent and always mid-hunt.
    Cutting deeper buys room for a boom to happen before the next wave is
    warranted.

  - **Bites are not clamped to the quota.** A predator eats everything its kill
    shape covers, even if that takes the population under the target. Clamping is
    what made a cull land on exactly the threshold; overshoot is the point.

  - **Reinforcements**: if the prey count is still not falling
    `PREDATOR_REINFORCE_STEPS` (120) after the last arrival, another triangle
    joins, up to `PREDATOR_MAX` (12). This is the same rule as a fresh threshold
    crossing — a pond that is over the line and not coming down has outbred the
    pack it has. Reinforcement is skipped while the population is already
    falling, so the cull doesn't overshoot into an extinction.

  - **The pack ratchets.** Its size can grow but never shrink: the largest pack
    a run has ever fielded is the minimum size of every later cull. A pond that
    once needed six hunters has proven it can outbreed five, and it does not get
    to relearn that from scratch each time. The per-tier pack floor applies to
    automatic waves only — a player summon still scales to the job asked.

  - Predators never hunt or eat each other.
  - The target is checked against the live population every step, since births
    keep moving it. It does overshoot: see "Bites are not clamped to the quota"
    above — the number taken in one bite is whatever the kill shape covers.
  - When the quota is met it stops hunting and swims off at speed for 44 steps
    before disappearing. Leaving is not a death — no tally, no lifespan, no death
    event. Births during the departure do not drag it back into a second hunt.
- **Immortality** — suppresses every natural death: old age and starvation are
  skipped, and the whole passive-combat phase is skipped (combat always ends in a
  death, and the attacker's energy cost can itself be lethal). A starving
  immortal agent is held at 1.0 energy rather than 0, so it keeps acting instead
  of re-tripping the death check every tick. Population is unbounded while this
  is on. Smites ignore it — a comet overrules the rules rather than obeying them.

## Combat

**Cannibalism.** Before speciation the pond is a free-for-all: an agent with no
lineage has no relatives, so nothing is protected from it. Once a lineage is
promoted, a member will only attack its *own species* if its aggression is at
least `CANNIBAL_AGGRESSION_MIN` (0.95) **and** its intelligence sits below 55% of
its range — bright species recognise their kin, dull furious ones do not. Other
species are always fair game; this is kin recognition, not pacifism. It is a rule
over existing traits rather than a twelfth gene, so the behaviour falls out of a
combination the pond already selects on.

Two combat paths exist:

### Passive combat phase (`_resolve_combat`, runs every tick after actions)
- Triggers when 2+ agents occupy the same tile
- Attacker must have `aggression >= hunt_aggression_threshold` to initiate. This
  is a **dial**, default **0.80**, range 0–1.06 (`Tunables`), set in run setup.
  `aggression` maxes at 1.05, so a threshold above that switches agent-on-agent
  combat off entirely
- Attacker must also be **hungry**: energy `< 0.5 × 100 × energy_capacity`
  (`HUNT_HUNGER_FRAC`). Sated agents do not hunt. This is the density-dependent
  brake on predation — without it a predator that clears the local prey simply
  grazes instead, so nothing stops aggression reaching fixation and crashing the
  population.
- Each eligible attacker picks one random co-tile target per step
- Outcome is probabilistic, scaling continuously with `attack` vs effective `defense`:
  `p_win = attack / (attack + defense)`
  (previous fixed thresholds — defender wins iff `attack <= defense × 0.33` — were
  unreachable for adult agents given trait bounds, making combat a guaranteed win
  above `attack` 0.706 and a coinflip below it)
- **Win:** attacker gains **66.7% of loser's current energy** (`PREDATION_YIELD`,
  capped at capacity); loser dies ("Killed in combat").
  This was 12.5%, which made predation close to pointless — a kill returned an
  eighth of a body while a failed hunt cost real energy, so aggression was
  selected out of every run and the population lost its only density-dependent
  brake. Two thirds makes hunting a genuinely worthwhile high-risk strategy
  without making it free: the roll can still fail, and failure still hurts.
- **Loss:** the prey fights back and escapes. Attacker loses
  `victim_effective_defense × 8.0 / attacker_effective_defense` energy
  (`RETALIATION_ENERGY`) and *survives* unless that empties it; only then does
  the defender take the yield and the attacker die.
  Dividing by the attacker's own defense is what keeps defense worth investing
  in for a hunter as well as a victim: a well-armoured predator shrugs off a
  failed hunt, a glass cannon does not.
  A failed attack previously killed the attacker outright, which made
  `aggression >= 0.80` a ~coinflip for your life that the agent volunteers for —
  every seed reached **zero** agents above the threshold by step ~250, after which
  combat never fired again for the rest of the run.
- Initiating attack costs `0.2 × metabolism` energy

### Chosen attack (ACTION_ATTACK output, index 7) — legacy Python only

**Dormant in `pond_core`.** Output index 7 is the sleep gate in the Rust engine,
and no brain-chosen attack exists: all agent-on-agent predation goes through the
passive combat phase above. Kept as the spec for the mechanic if it is revived.

- Agent targets a co-tile agent selected by the environment
- Aggression must be `> 0.55`; otherwise costs `0.1` energy and does nothing
- Initiating costs `0.5 × metabolism` even on a successful strike
- If `attack > target_defense`: steal `12.5%` of target's current energy; target dies if drained to 0 ("Eaten alive")
- Otherwise: lose `attacker_defense × attacker_energy`; attacker dies if drained to 0

## Genome Traits

| Trait | Min | Max | Mutable |
|-------|-----|-----|---------|
| vision | 0.5 | 1.05 | ✓ |
| speed | 0.5 | 1.0 | ✓ |
| metabolism | 0.5 | 1.05 | ✓ |
| energy_capacity | 0.95 | 1.05 | ✗ |
| mutation_rate | 0.01 | 0.25 | ✗ |
| reproduction_cost | 0.75 | 1.50 | ✓ |
| attack | 0.5 | 1.25 | ✓ |
| defense | 0.5 | 1.07 | ✓ |
| aggression | 0.0 | 1.05 | ✓ |

`daily_nutrition_minimum` and `clone_energy_threshold` removed (Rust `pond_core`) —
generated and mutated but never read anywhere, pure mutation drift + wasted RNG
draws. `intelligence` disabled (Rust `pond_core`) for the same reason but kept as a
commented-out TODO in `genome.rs` since it's planned to be wired into
decision-making later.

## Neural Network (Brain)

- Architecture: `7 → 12 → 12 → 12 → 8` (4 linear layers, width 12, 8 outputs)
- Activations: ReLU between each linear layer
- Weights loaded from genome (512 total; hand-rolled MLP in `pond_core/src/brain.rs`)

**Output selection differs between implementations (refactor change):**

- **Legacy Python:** softmax over the 8 logits, then a multinomial sample picks
  one discrete action (decision D1).
- **Rust `pond_core` (canonical):** the 8 logits are passed through
  element-wise **sigmoid** to independent `[0,1]` gates. Outputs 0–2 are
  continuous steering-force weights (seek / wander / separate), 4/5/7 are
  discrete triggers that fire when the gate `> 0.5` (eat / reproduce / sleep,
  mutually exclusive — only the highest fires per tick), and 3/6 (flee /
  attack) are dormant. No softmax, no multinomial sample. See the input/output
  contract in [`pond_core/README.md`](pond_core/README.md).

## Mutation

- Each trait has `mutation_rate` chance to mutate
- Mutation magnitude scales with `mutation_rate`
- Brain weights also mutate, per-weight with the same `mutation_rate` chance
  - **Rust `pond_core` (canonical):** additive — `w + uniform(−m·0.5, +m·0.5)`
    where `m = mutation_rate × 0.5`. Weight signs can flip; zero weights can
    revive.
  - **Legacy Python:** multiplicative — `w × uniform(1−m, 1+m)`. Signs never
    flip; zero weights stay dead.

### Rate modifiers

Two things scale the mutation rate at reproduction, and they behave differently:

| Modifier | Source | Heritable? |
|---|---|---|
| Memory suppression | `1 / (1 + parent_success_count × 0.05)` | **Yes** — multiplies into the child's `effective_mutation_rate` and compounds down the generations (D4) |
| Probation clamp | `0.15` while the parent's genome sits in a cluster on probation, else `1.0` | **No** — scales the rate used for one set of draws and is discarded |

The clamp is deliberately not heritable. Probation asks whether a lineage
survives having its mutability taken away; a lineage that passes has to get it
back. Routing it through the heritable path would sterilize the lineage
permanently, which tests something else entirely.

---

## Speciation

A k-means cluster is not a species. Clusters exist because `k` was chosen — a
dial, default **6**, range 2–12 (`Tunables`), set in run setup — and their labels permute when
clusters split or merge. Labels are always `< k`; lowering `k` mid-run remaps a
family whose label is now out of range, so it changes colour once. A cluster becomes a named
species only by passing a test.

**Signature.** Seven mutable traits, each normalized to `[0, 1]` by its bounds.
`energy_capacity` and `mutation_rate` are excluded: they are locked (D3), never
mutated and inherited exactly, so clustering on them yields founder-descent
groups rather than the shapes selection actually built.

**Lifecycle.** `observed → probation → promoted`, evaluated on the cluster tick
(every 50 steps).

| Stage | Requirement |
|---|---|
| Observed → probation | Holds ≥ `max(6, 5%)` of the population, centroid drift under `DRIFT_EPS`, within-cluster spread under `SPREAD_MAX`, for ≥ 5 consecutive runs **and** ≥ 3 generations of mean-generation advance |
| Probation | Members' mutation rate clamped to `0.15×`. Must keep meeting every criterion above |
| Probation → promoted | Survives a further 1 generation of advance under the clamp. Clamp lifts on promotion |
| Probation → observed | Fails any criterion. Clamp lifts, streak resets, may re-enter later |

Promotion is therefore an experiment, not an observation: a shape still riding
mutation toward a fit, or one k-means merely happened to bracket, loses its
share once frozen and never promotes.

**Membership is decided once and held for life.** There are exactly two ways
into a species:

1. **Birth.** A child is measured against its parent's species *definition* —
   `founding_centroid` within `MEMBERSHIP_RADIUS` — once, at birth, after
   mutation. Inside, it inherits the lineage; outside, it is born unassigned.
2. **Founding.** At promotion, the unassigned agents inside the new species'
   definition are seated in it. That is the only time an existing agent joins.

Nothing else moves an agent between species. Leaving happens by dying, or by the
species going extinct and releasing its members.

**Why not by proximity.** Membership used to be re-measured every cluster tick
as "nearest live centroid within the radius" — but `Species::centroid` tracks
the member mean at `CENTROID_TRACKING` per run, so an agent could lose its
species without changing at all: the lineage drifted off it, or a neighbouring
one drifted onto it. Genome and brain weights are fixed at birth and evolution
happens only through reproduction; membership had no business being the one
heritable thing that was continuously re-measured against the neighbours.

**What this buys.** A species is a definition fixed at promotion, and mutation is
the only thing that can put a child outside it. As a lineage drifts, more of its
offspring are born outside its own definition; those accumulate as unassigned,
cluster, and are promoted as the next species. Speciation by budding, rather than
by an accounting rule.

**Drift and extinction.** A species centroid tracks its members at `0.05` per
run. Zero members for 2 consecutive runs sets `extinct_at`; the record is kept
forever. A species that reappears at the same centroid is a **new** species with
a new id — convergent evolution is the more interesting reading, and
resurrection would make the timeline lie. Ids are monotonic and never reused.
Live species are capped at 12; at the cap promotion is refused rather than
evicting a live species, which would write a false `extinct_at`.

### Appearance

**Shape is anchored to the lineage.** An agent's morphology comes from its
species' `founding_centroid` plus `INDIVIDUAL_VARIATION` (0.35) of its own
deviation from it, so every member of a species is drawn around the shape that
lineage had when it was promoted. The anchor must not be `Species::centroid`,
which EMA-tracks the live member mean every cluster tick: keyed to that, an
animal's appearance changes when its *neighbours* change and a species has no
stable look at all. Unassigned agents have no lineage to vary around and are
drawn from their own traits.

**Parts are categorical.** Counts are integers at fixed thresholds — 0–3 spike
pairs, 0–3 armour rings, 0/2/4/6 fins, 5/7/9/11 segments — and absence is a real
state. Continuous knobs across eight dimensions read as "different sizes"; a
viewer can only say "different kinds" about things they can count. Proportions
(eye size, segment spacing, envelope) stay continuous, because they are
proportions and not parts.

**Unassigned agents wear the title screen's lime as an outline.** The body is
near-colourless — that is the point, promotion confers colour — but grey alone
reads as something failing to render. An acid-lime edge reads as a creature
waiting for a name, which is what it is.

**Hue belongs to the lineage.** Each genus takes the next golden-angle hue on
first promotion, so genera are maximally separated however many accumulate;
species within a genus vary by lightness and chroma, so siblings look related
and the taxonomy is legible in the palette. Unassigned agents are a desaturated
grey-blue — promotion visibly confers an identity.

**Strategy moved to the glow.** The additive halo carries the combat profile:
warm and strong for aggressive, cool and faint for passive. Hue previously
encoded that, which meant a converged pond was a single colour — a scalar that
converges cannot label anything, and colour is the strongest categorical channel
available.

### Ancestry

Every promoted species records a `parent_id`: the nearest species — live or
extinct — within `GENUS_RADIUS` of its founding centroid, or **0** if it founded
with nothing nearby. It is the *same* lookup that lends the genus, so a species
can never carry one lineage's name and another's ancestry.

**This is an inference, not an observed birth.** Nothing in the sim watches a
population split. A species promoting beside an unrelated lineage that happens to
have converged on the same shape is recorded as its child, and a lineage whose
real ancestor drifted out of range before it promoted is recorded as a root. Read
an edge as "nearest relative at the time it earned a name".

The phylogeny panel (`P` in the web build) draws these edges as a pine: the trunk
is time, each bough leaves its parent at its own founding step, bough length is
lifespan and thickness is peak members. Founding and extinction steps are always
multiples of 50 — the registry only advances on a cluster tick — so the tree's
time axis is quantised to that.

### Ambient predation

One resident triangle is in the water whenever prey clears
`PREDATOR_AMBIENT_MIN_PREY` (30), with a quota of zero: it never sates and never
stops. The capacity rule still stacks a cull pack on top when the population
crosses `cull_trigger_pop`, and when that cull finishes exactly one resident
takes ambient duty again — the rest go quiet as before. The pack still ratchets,
but ambient pressure does not: with every resident on ambient duty the pressure
compounds with each boom the pond has ever had, which measured as *more* kills
than the old fast hunters and a five-seed mean population of 15.

**Hunter speed comes from the prey.** A tier-0 hunter moves at
`PREDATOR_SPEED_FRAC` (0.95) of the mean speed trait of its search-image family,
converted through `MAX_SPEED × DT` like any agent's own velocity cap. It was a
flat 0.95 world units per *tick* against a fastest-possible agent's 0.15 — a
sixfold advantage that made evasion arithmetically impossible, which is why the
flee output sat dormant for as long as it did. Below 1.0 on purpose: a hunter
slower than the average animal in the family it hunts cannot catch the
above-average ones, so escape is positional. The goal is not to outrun the
predator, it is to be harder to catch than a neighbour.

**Floor and ceiling.** A hunter's speed is clamped to
`[PREDATOR_SPEED_FLOOR_TRAIT (0.95), PREDATOR_SPEED_CEILING_TRAIT (0.99)]` of
`MAX_SPEED`. It is an apex predator in open water: fast, always. The ceiling
sits just under the quickest animal the genome can build (trait max 1.0), so a
lineage that spends everything on speed can still escape and one that has not,
cannot.

Tracking the prey's mean *alone* made the pressure purely relative — you only
needed to be a little quicker than your neighbours, so the distribution slid
downward together while movement kept costing energy in absolute terms.

**Bursts.** Each resident hunter has a small per-tick chance
(`PREDATOR_BURST_CHANCE`, about one per 2,500 ticks) of running at the ceiling
for 150–400 ticks. Variance in the threat: a steady predator is one a lineage
can evolve a fixed answer to, after which the pond settles. An occasional
faster-than-it-should-be hunter means the safe margin is never quite knowable —
the same reason the search image moves.

Tiers 1 and 2 keep their flat constants. They are god-mode powers and are meant
to be unfair.

**Immortality suppresses ecology, not the player.** Under god-mode immortality an
automatic hunter does not bite; a summoned one still does.

### Predator adaptation

An automatic hunter is not a fixed threat. It carries two pieces of state that
track the pond:

- **Search image.** On every cluster tick it re-forms an image of the most
  numerous genome family and prefers that family while hunting — a matching
  animal is treated as `SEARCH_IMAGE_PULL`× nearer than it is. Preference, not
  exclusivity: with nothing matching in range it hunts whatever is there, since
  an absolute image would make a rare family untouchable, which is how a rare
  family stops being rare. Switching needs the challenger to be
  `SEARCH_IMAGE_SWITCH_MARGIN`× more numerous, so two similar families do not
  make the image oscillate. Within the image it prefers the better-armoured
  members.
- **Learned bite.** `attack` starts at the toughest animal in the pond it
  arrives into (`starting_bite`) and tracks the toughest member of its current
  image, capped at `TIER_ATTACK[tier] + 0.45`. A switch keeps half the surplus:
  general toughness carries across prey, the specific calibration does not.

**Why the toughest and not the average.** The bite kills everything the kill
shape covers whose effective defense falls below `attack`, so any aim short of
the family's toughest member kills the softer half and spares the harder — a
subsidy for armour however it is phrased. Measured as the mean defense of eaten
animals minus the pond's mean defense at that moment, the old flat bite scored
**−0.155**: predation was systematically killing the least armoured. Aiming at
the family mean, and at mean + 2σ, both stayed near −0.15. Aiming past the
toughest, with hunters arriving calibrated, brings it to **+0.018** — armour no
longer changes who dies, so it stops being bought as predator insurance.

What remains is frequency: being common is what draws the hunters. That is the
part that stabilises, since a strategy is punished in proportion to how well it
is doing.

**This does not, on its own, lower the pond's mean defense** — agent-on-agent
combat is the larger mortality source and still rewards armour directly. It
removes predation's contribution to the ratchet, no more. See DEVLOG for the
measured five-seed comparison.

## Disease

A disease arrives with a lineage, not with a population level.

**Origin.** On each promotion, a flat `DISEASE_CHANCE` (0.30) that the new
species turns out to be carrying something. Not weighted by population, species
age or dominance — a pathogen more likely to appear in a crowded pond is a
density-dependent cull wearing a costume. The rate is read per *promotion*, and
a 3000-step run sees about two, so this puts a disease in roughly half of runs.

**Severity** (0.02–0.14) is an energy drain per tick, scaled by metabolism.
Death is attributed to `Disease` rather than `Starvation`, so an outbreak is not
hidden inside the food economy — but mechanically it *is* starvation, which is
the point: an outbreak lands on the economy and takes the already-marginal first.

**Contagion** (0.02–0.30) is a per-contact chance at full local crowding.
Transmission reads neighbours within `CONTACT_RADIUS` (1.1 tiles), scaled by
crowding up to `CROWDING_FULL` (6) and clamped. **Nothing scales with total
population.** An outbreak in a tight cluster behaves identically in a pond of 40
and a pond of 400, which is what lets it overshoot and crash instead of trimming
toward a setpoint.

**Immunity** (trait 10, bounds 0–1, mutable, in the species signature) scales the
per-contact infection probability by `1 - immunity`. It is resistance to
*catching* something and nothing else — a fully immune agent that is already
infected dies of it like anyone else. It costs `IMMUNITY_UPKEEP (0.035) ×
immunity × metabolism` per tick, because an immune system a pond has never
needed should be a liability: without a price the trait goes to fixation and
stops being a decision, which is what happened to defense for the life of the
project. Measured over six seeds it settles at **0.45**, mid-range.

**No recovery.** An infected agent carries it until it dies, and offspring are
born clean. Recovery would be a restoring force and would turn every outbreak
into a damped oscillation converging on equilibrium.

**Cross-species jump** at `CROSS_SPECIES_JUMP` (1.5e-6) per contact, once per
pathogen: after it jumps it belongs to nobody and spreads at full contagion to
anything. The rate is small because the roll fires against every susceptible
neighbour of every carrier every tick — an outbreak makes on the order of 100k
rolls, and at 4e-5 two of the first three measured outbreaks jumped, which is
not the rare event this is meant to be.

**Off switch.** `disease_enabled` (god panel) stops new pathogens being seeded
and stops transmission. Agents already infected stay infected: clearing them
would rewrite the run rather than pause it, and there is no cure in this model.

**Names** are the host's genus mangled — stem, a nonsense infix, a pathological
ending, and a word for what it does: *Thalorandrpestis spumosa*, *Surnecrosis
maligna*, *Lumoxytabes vexans*. Deterministic from `(disease_id, world_seed)`
with a private RNG, like species names.

### Names

Promoted species get a binomial, generated deterministically from
`(species_id, world_seed)` with a private RNG — never the world's stream, since
drawing from that would shift every subsequent simulation draw.

- **Genus is always feminine** (*Vorixa*, *Thalura*, *Ixyria*) — invented
  syllables, one gender throughout, so the genus reads as a consistent family.
- **Genus is inherited.** A species promoting within `GENUS_RADIUS` of an
  existing one — live *or* extinct — takes its genus and a new epithet. Extinct
  counts: a lineage re-radiating after a bottleneck keeps its family name.
  Without inheritance the naming would actively lie about descent.
- **Epithet comes from signed deviation** from the population centroid, not from
  the largest trait value. Argmax over the centroid returns whichever trait sits
  high across the whole pond, so in an aggressive pond every species would end
  up *ferox*. Deviation names what makes the lineage *different*, and the sign
  matters — each trait has a high word and a low word.
- **Latin when specialized, nonsense when not.** A deviation clearing
  `STRONG_DEVIATION` earns a real Latin adjective for that trait and direction
  (*loricata*, *velox*, *frigida*); nothing standing out earns a nonsense
  epithet (*kyrnus*, *vonda*). The name reports whether the lineage specialized
  at all.
- **Epithet endings mix masculine and feminine** (*-us* / *-a*) rather than
  agreeing with the feminine genus as Latin grammar would require. Deliberate:
  the variety is worth more here than the agreement.

**Determinism.** Speciation draws no RNG of its own, but it is not a pure
observer — the probation clamp changes the rate used at reproduction, and the
per-weight mutation draw is conditional on that rate, so the RNG stream diverges
from a build with speciation disabled. Same-seed runs remain bit-identical.

---
