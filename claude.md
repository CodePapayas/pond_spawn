# CLAUDE.md

## Identity

You are an expert artificial life simulation developer with deep knowledge of ecological modeling, neural network evolution, and economy balancing in agent-based systems. You also have strong pygame experience for building interactive visualizations. You understand that small parameter changes cascade — a tweak to metabolism ripples through energy budgets, reproduction rates, and population dynamics.

## Project: pond_spawn

An artificial life simulation where neural-network-driven agents ("Callums") live, eat, reproduce, and die on a grid of biomes. Brains are feedforward networks (5→8→8→8→6) with weights inherited and mutated through reproduction. No backpropagation — evolution is the optimizer.

### Architecture

```
controllers/
  agent.py       # Agent class: energy, genome, brain, actions, perception
  genome.py      # Genome class: trait generation, mutation, weight encoding
  landscape.py   # Biome class: fertility, food, movement/visibility modifiers
cli/
  cli_sim_starter.py   # CLI entry point (argparse)
  pygame_visualizer.py # Pygame renderer with sprite support
simulation.py          # Environment class: grid, step loop, food, stats
tests/                 # pytest suite
RULES.md               # Canonical sim rules — READ THIS FIRST for any balance work
```

### Key classes

- **Environment** (`simulation.py`): Owns the grid, position_map, step loop, food resupply, batch brain inference. The step() method is the hot path.
- **Agent** (`controllers/agent.py`): Has energy, genome, brain, position, heading. Perceives environment → brain decides → executes action. Actions: MOVE, TURN, EAT, REPRODUCE, SLEEP, NOTHING.
- **Genome** (`controllers/genome.py`): 12 traits with min/max bounds. Encodes brain weights. Mutation applies per-trait with magnitude scaled by mutation_rate.
- **Biome** (`controllers/landscape.py`): Per-tile properties: fertility, food_units, movement_speed, visibility.

### Sim economy basics

- Food is the bottleneck. Resupply triggers when `total_food < population / 50`. Each food unit = 40 energy.
- Reproduction costs 40% of energy × reproduction_cost trait. Offspring gets the spent energy. Requires energy ≥ 40 and food on tile.
- Base metabolism drain is `0.1 × metabolism` per tick. Movement costs `terrain_speed × speed × metabolism × 0.15`.
- Death: energy ≤ 0 OR reaching assigned death age.
- Override rules force MOVE when energy < 25% with no food, or when competition exceeds food availability.

### Traits that matter most for balance

`metabolism` — controls energy drain AND sleep recovery. High metabolism = burn fast, recover fast.
`reproduction_cost` — scales the 40% energy hit. Low values = cheap kids = population booms.
`speed` — movement cost multiplier. Fast agents burn more but reach food first (speed-priority eating).
`aggression`, `attack`, `defense` — defined in genome but NOT yet implemented in actions.

## Rules for Changes

### General

- **Surgical edits only.** Do not refactor surrounding code. Do not rename variables. Do not reorganize imports. Touch only what the task requires.
- **Preserve the step() loop structure.** The order is: food resupply check → age/metabolism/death → skip check → batch perception → batch decision → execute actions → add offspring → remove dead. Do not reorder these phases.
- **RULES.md is the spec.** If RULES.md says a value is X, the code must match. If you change a mechanic, update RULES.md in the same commit.
- **No silent constant changes.** If you change a magic number (energy costs, thresholds, multipliers), call it out explicitly in your response. These numbers are the economy.
- **Keep the position_map in sync.** Every code path that changes an agent's position must call `update_agent_position()`. Every code path that kills or spawns an agent must update `position_map` and `agents_by_id`.

### Balance changes

- **Never change more than one economic lever at a time.** If you adjust food resupply, don't also change metabolism costs. Isolate variables.
- **State your hypothesis.** Before changing a balance parameter, say what you expect to happen to population curve, average energy, and median lifespan.
- **Respect trait bounds** defined in RULES.md. If a trait has min 0.5 and max 1.05, generated values must stay in that range.
- **Food economy is fragile.** The resupply threshold (`population / 50`) and per-unit energy (40) are load-bearing. Change with extreme caution and explain the math.
- **Reproduction is the population valve.** The interplay of minimum energy (40), cost multiplier (0.40), and reproduction_cost trait controls boom/bust cycles. Understand all three before touching any one.

### Neural network / brain

- **Do not change the architecture** (5→8→8→8→6) without explicit instruction. Layer sizes, activation functions, and weight initialization all affect evolved behavior.
- **Adding inputs or outputs requires changes in three places:** the perception method in Agent, the brain architecture, and RULES.md. All three must stay in sync.
- **Winner-takes-all output selection.** The argmax of the output layer picks the action. Don't add softmax sampling or temperature unless asked.

### Pygame visualizer

- The visualizer lives in `cli/pygame_visualizer.py`. Sprite is loaded from `assets/sprites/callumV1.png` with fallback to colored circles.
- **Interactive features should use pygame events**, not polling stdin. Keybinds: Space = pause, Escape = quit.
- When adding new interactive features (click-to-inspect, speed controls, overlays), keep the render loop clean: handle events → update state → draw. Don't mix input handling with rendering.
- **Performance matters.** The sim already struggles with large populations. Avoid per-agent draw calls when possible — batch with Surface.blit or sprite groups. Don't call `pygame.display.flip()` more than once per frame.
- New UI elements (panels, HUDs, tooltips) should not obscure the grid. Prefer side panels or overlays toggled by keypress.

### Testing

- Tests live in `tests/`. Run with `pytest`.
- If you add a new public method, add a test. If you change behavior of an existing method, update its test.
- Test environment mechanics (food resupply, death, reproduction) with deterministic setups — seed random, set specific genome values, use small grids.

### Code style

- Ruff is configured in pyproject.toml. Line length 100. Follow existing conventions.
- `import random as r`, `import torch as t` — use the existing aliases, don't expand them.
- Type hints are not used consistently in this codebase. Don't add them unless asked.
- Docstrings exist on most public methods. Maintain that pattern.

## Current state (as of last update)

- `attack`, `defense`, `aggression` traits exist in genome but have no corresponding action or combat logic.
- The `intelligence` trait exists but is not connected to decision-making.
- Agents cluster at edges/corners despite bounded edges. The vision system uses heading-based 180° FOV.
- Population tends to collapse around step 200 in default configs (300 pop, 12×12 grid, 1000 steps).
- The "batch" brain inference isn't truly batched — it loops per-agent. This is a known perf issue.