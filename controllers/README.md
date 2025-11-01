# Controllers

This directory contains the core logic for agents, their brains (neural networks), genomes, and environment interaction.

## Agent Perception and Normalization

The `Agent.perceive()` method gathers sensory information from the environment and normalizes it for neural network input. All perception values are normalized to the range **[0.0, 1.0]** using `np.clip()`.

### Normalization Scheme

| Perception Input | Raw Range | Normalization Formula | Normalized Range | Notes |
|-----------------|-----------|----------------------|------------------|-------|
| **Energy** | 0 - 100+ | `energy / 100.0` | [0.0, 1.0] | Base energy capacity is 100, can be modified by `energy_capacity` trait |
| **Food Available** | 0 - 3 | `food_available / 3.0` | [0.0, 1.0] | Food units per tile (from biome.json: randomized 0-3) |
| **Nearby Agents** | 0 - 10+ | `nearby_agents / 10.0` | [0.0, 1.0] | Count of agents within vision range, capped at 10 for normalization |
| **Visibility** | 0 - ∞ | `nearby_agents / (visibility × visual_range)` | [0.0, 1.0] | Relative agent density considering terrain visibility and agent vision trait |
| **Movement Speed** | 0 - ∞ | `terrain_speed × speed_stat` | [0.0, 1.0] | Combined terrain movement modifier and agent speed trait |

### Input Vector Structure

The perception input to the brain is a 5-element tensor:
```python
[normalized_energy, normalized_food, normalized_agent_count, normalized_visibility, normalized_movement]
```

### Why Normalization Matters

1. **Neural Network Stability**: Neural networks perform best with inputs in a consistent range (typically [0, 1] or [-1, 1])
2. **Fair Weighting**: Without normalization, features with larger ranges would dominate the network's learning
3. **Gradient Flow**: Normalized inputs help prevent vanishing/exploding gradients during training
4. **Comparable Scales**: Makes different sensory inputs directly comparable to the network

### Safety Features

- **Division by Zero Protection**: The visibility calculation checks for zero denominators
- **Clipping**: All values are clipped using `np.clip(value, 0.0, 1.0)` to ensure they stay within bounds
- **Default Values**: Traits return default values (usually 1.0) if not present in genome

## Decision Making

The `Agent.decide()` method uses a hybrid approach:

1. **Critical Survival Rules** (hard-coded, checked first):
   - Low energy + no food → MOVE (search for food)
   - High competition → MOVE (avoid crowding)
   
2. **Neural Network Decision** (winner-takes-all):
   - If no critical rules apply, the brain processes perception
   - Action with highest output value is selected
   - Actions: 0=MOVE, 1=TURN, 2=EAT, 3=REPRODUCE

This ensures agents always prioritize survival while allowing learned behavior for non-critical situations.
