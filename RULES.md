# RULES.md

> **Canonical implementation: Rust `pond_core`.** These rules were refactored
> from the original Python sim to the Rust engine (see
> [`REFACTOR_RUST_BEVY.md`](REFACTOR_RUST_BEVY.md) and
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
- Regen rate per tile: `fertility / 1.6 × 0.012` (per-tick probability of gaining 1 food unit)
- Max food per tile: **3**
- 35–45% of tiles are permanently barren (fertility = 0, never regenerate), arranged in contiguous desert clusters with fertile oases between them
- Each food unit provides **33.3 energy**

## Agents ("Callums")

### Initialization
- Starting energy: 100
- Starting heading: random (N/E/S/W)
- Assigned a death age at birth

### Perception (Brain Inputs)
1. Normalized energy (0–1)
2. Normalized food at tile (0–1)
3. Normalized nearby agent count (0–1)
4. Visibility factor
5. Movement factor

### Actions (Brain Outputs)
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
- Energy < 25% AND no food → forced MOVE
- Food > 0 AND nearby_agents > (food × 2 + 1) → forced MOVE

### Energy Costs
| Action | Cost |
|--------|------|
| Base metabolism (per tick) | `0.1 × metabolism` |
| Move | `terrain_speed × speed × metabolism × 0.15` |
| Turn | `0.14 × metabolism` (`turn()` = 0.1 + `execute_action` = 0.04) |
| Reproduce | `energy × 0.50 × reproduction_cost` |
| Sleep | Gain `0.15 × metabolism` |
| Nothing | `0.005 × metabolism`; agent skips next tick |

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
- Offspring energy: equal to the energy cost paid by the parent
- Offspring placed on random adjacent tile (wrapping)

### Death
- Energy ≤ 0
- Reaching assigned death age
- Killed in combat

## Combat

Two combat paths exist:

### Passive combat phase (`_resolve_combat`, runs every tick after actions)
- Triggers when 2+ agents occupy the same tile
- Attacker must have `aggression >= 0.80` to initiate
- Each eligible attacker picks one random co-tile target per step
- Outcome is probabilistic, scaling continuously with `attack` vs effective `defense`:
  `p_win = attack / (attack + defense)`
  (previous fixed thresholds — defender wins iff `attack <= defense × 0.33` — were
  unreachable for adult agents given trait bounds, making combat a guaranteed win
  above `attack` 0.706 and a coinflip below it)
- Winner gains **12.5% of loser's current energy** (capped at capacity)
- Loser dies ("Killed in combat")
- Initiating attack costs `0.2 × metabolism` energy

### Chosen attack (ACTION_ATTACK output, index 7)
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

- Architecture: `5 → 12 → 12 → 12 → 8` (4 linear layers, width 12, 8 outputs)
- Activations: ReLU between each linear layer
- Weights loaded from genome (488 total; hand-rolled MLP in `pond_core/src/brain.rs`)

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
- Brain weights also mutate

---
