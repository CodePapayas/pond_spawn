# RULES.md

## World

- Grid: Square grid, default 12x12
- Bounded edges (no wrapping)
- Each tile is a biome with properties:
  - `movement_speed`: 0.8–1.05
  - `visibility`: 0.25–1.0
  - `fertility`: 0.01–1.6
  - `food_units`: 0–3 (initial)

## Food

- Food spawns based on biome fertility
- Resupply triggers when: `total_food < population / 50`
- Food added: `int(resupply_amount * fertility) % 100`
- Each food unit provides **40 energy**

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
| 1 | TURN | Rotate 90° clockwise |
| 2 | EAT | Consume food at current tile |
| 3 | REPRODUCE | Create offspring |
| 4 | SLEEP | Gain energy |
| 5 | NOTHING | Minimal energy burn |

### Decision Override Rules
- Energy < 25% AND no food → forced MOVE
- Food > 0 AND nearby_agents > (food × 2 + 1) → forced MOVE

### Energy Costs
| Action | Cost |
|--------|------|
| Base metabolism (per tick) | `0.1 × metabolism` |
| Move | `terrain_speed × speed × metabolism × 0.15` |
| Turn | `0.1 × metabolism` |
| Reproduce | `energy × 0.40 × reproduction_cost` |
| Sleep | Gain `20 × metabolism` |
| Nothing | `0.005 × metabolism` |

### Reproduction
- Minimum energy: 40
- Requires food on tile
- Cost: 40% of energy × `reproduction_cost` trait
- Offspring receives the energy spent
- Offspring placed on adjacent tile

### Death
- Energy ≤ 0
- Reaching assigned death age

## Genome Traits

| Trait | Min | Max | Mutable |
|-------|-----|-----|---------|
| vision | 0.5 | 1.05 | ✓ |
| speed | 0.5 | 1.0 | ✓ |
| metabolism | 0.5 | 1.05 | ✓ |
| daily_nutrition_minimum | 0.95 | 1.0 | ✓ |
| energy_capacity | 0.95 | 1.05 | ✗ |
| mutation_rate | 0.01 | 0.25 | ✗ |
| clone_energy_threshold | 0.5 | 1.05 | ✓ |
| reproduction_cost | 0.25 | 1.05 | ✓ |
| intelligence | 0.5 | 1.05 | ✓ |
| attack | 0.5 | 1.25 | ✓ |
| defense | 0.5 | 1.05 | ✓ |
| aggression | 0.0 | 1.05 | ✓ |

## Neural Network (Brain)

- Architecture: 5 → 8 → 8 → 8 → 6
- Activations: ReLU, ReLU, Sigmoid
- Weights loaded from genome
- Winner-takes-all output selection

## Mutation

- Each trait has `mutation_rate` chance to mutate
- Mutation magnitude scales with `mutation_rate`
- Brain weights also mutate

---

*Traits `attack`, `defense`, `aggression` defined but not yet implemented.*
