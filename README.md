```
  ██████╗  ██████╗ ███╗   ██╗██████╗       ███████╗██████╗  █████╗ ██╗    ██╗███╗   ██╗
  ██╔══██╗██╔═══██╗████╗  ██║██╔══██╗      ██╔════╝██╔══██╗██╔══██╗██║    ██║████╗  ██║
  ██████╔╝██║   ██║██╔██╗ ██║██║  ██║      ███████╗██████╔╝███████║██║ █╗ ██║██╔██╗ ██║
  ██╔═══╝ ██║   ██║██║╚██╗██║██║  ██║      ╚════██║██╔═══╝ ██╔══██║██║███╗██║██║╚██╗██║
  ██║     ╚██████╔╝██║ ╚████║██████╔╝      ███████║██║     ██║  ██║╚███╔███╔╝██║ ╚████║
  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝
```

[![Coverage](https://raw.githubusercontent.com/codepapayas/pond_spawn/badges/badges/coverage.svg)](https://github.com/codepapayas/pond_spawn)
[![Ruff](https://github.com/codepapayas/pond_spawn/actions/workflows/ruff.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions/workflows/ruff.yml)
[![Pylint](https://github.com/codepapayas/pond_spawn/actions/workflows/pylint.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions/workflows/pylint.yml)
[![Tests](https://github.com/codepapayas/pond_spawn/actions/workflows/coverage.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions/workflows/coverage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Language](https://img.shields.io/badge/language-python-blue)

### With visuals
![Simulation Visualization GIF](assets/gifs/visual_sim.gif)

### Without visuals (print to console)
![Simulation GIF](assets/gifs/sim_gif_looped.gif)

*************************

## RUNNING THE SIMULATION
*************************

Install dependencies first:

```bash
pip install -r requirements.txt
```

### Interactive menu (recommended)

```bash
python -m cli.menu
```

Navigate with number keys. Press `[8]` to run. Settings persist between runs in the same session.

| Option | What it sets |
|--------|-------------|
| `[1]` | Grid size |
| `[2]` | Population (auto-capped to `2 × grid²`) |
| `[3]` | Steps per run |
| `[4]` | Number of runs |
| `[5]` | Step delay (seconds) |
| `[6]` | Visual mode: progress bar / stats each step / full grid |
| `[7]` | Toggle show-initial grid |
| `[8]` | Run |
| `[0]` | Quit |

### CLI (headless / scripted)

```bash
# Basic usage
python -m cli.cli_sim_starter

# Custom parameters
python -m cli.cli_sim_starter --grid-size 14 --population 80 --steps 1500

# Stats only, no grid dump
python -m cli.cli_sim_starter --stats-only --no-visual

# Show the randomized starting state before the sim runs
python -m cli.cli_sim_starter --show-initial

# Full help
python -m cli.cli_sim_starter --help
```

Current CLI flags for `cli_sim_starter`:

`--grid-size` · `--population` · `--steps` · `--delay` · `--no-visual` · `--show-initial` · `--stats-only`

Both launchers write a stats chart to `charts/simulation_stats_<timestamp>.png` when the run ends.

### Pygame Visualizer
```bash
# Basic usage
python -m cli.pygame_visualizer

# Custom parameters
python -m cli.pygame_visualizer --grid-size 20 --population 200 --cell-size 30 --fps 15 --max-ticks 2000

# Full help
python -m cli.pygame_visualizer --help
```

Current visualizer flags:

`--grid-size` · `--population` · `--cell-size` · `--fps` · `--max-ticks`

Controls:

`Space` pauses/resumes the sim. `Escape` quits and generates the final stats chart.

Place your agent sprite at `assets/sprites/callumV1.png`. The visualizer rotates it by heading and falls back to circles plus heading markers if the sprite is missing.

# OVERVIEW
*************************
<h2>An attempt to understand neural networks and artificial life simulations.</h2>

The current codebase is a toroidal grid-world sim with random biome generation, mutated genomes, PyTorch brains, fertility-scaled food regrowth, and both console and pygame front ends.

# CURRENT SPECS
*************************

**Brain**

The brain is built dynamically from `brains/brain.json`, not hard-coded in Python.

| Property | Value |
|----------|-------|
| Inputs | 5 normalized values |
| Current architecture | `5 → 12 → 12 → 12 → 8` |
| Hidden activations | `ReLU` |
| Output selection | `softmax` sampling (multinomial draw) |
| Learning | No backpropagation; weights change only through genome mutation |

Current perception vector:

`energy / 100` · `food_on_current_tile / 5` · `nearby_agents / 10` · `visibility-adjusted crowding` · `terrain_movement_speed × speed_trait`

The environment also applies two hard survival heuristics before brain output is used:

`low energy + no food => MOVE`

`too many nearby agents for the available food => MOVE`

**Action Surface**

The brain outputs 8 action slots:

| Index | Action | Status |
|-------|--------|--------|
| 0 | MOVE | Wired |
| 1 | TURN_RIGHT | Wired |
| 2 | EAT | Wired |
| 3 | REPRODUCE | Wired |
| 4 | SLEEP | Wired |
| 5 | DO_NOTHING | Wired |
| 6 | TURN_LEFT | Wired |
| 7 | ATTACK | Wired |

There is also a passive combat phase after actions execute: agents with `aggression >= 0.80` may attack another agent on the same tile regardless of their chosen action output.

**Genome**

The genome template lives in `genomes/genome.json` and defines 12 traits:

`vision` · `speed` · `metabolism` · `daily_nutrition_minimum` · `energy_capacity` · `mutation_rate` · `clone_energy_threshold` · `reproduction_cost` · `intelligence` · `attack` · `defense` · `aggression`

`daily_nutrition_minimum` and `intelligence` are defined in the genome but not currently consumed by the sim loop.

**Environment**

| Mechanic | Current behavior |
|----------|------------------|
| World topology | Toroidal wraparound grid |
| Biome features | `movement_speed`, `visibility`, `fertility`, `food_units` |
| Initial biome food | Barren tiles: `0`; fertile tiles: random `0–3` |
| Initial energy | `100.0` |
| Initial heading | Randomized |
| Environment defaults | `Environment(grid_size=12, num_agents=300)` |
| CLI defaults | `grid_size=10`, `population=50`, `steps=1000` |
| Visualizer defaults | `grid_size=12`, `population=100`, `fps=10`, `max_ticks=1000` |
| Spawn cap | Initial population capped to `2 * grid_size * grid_size` |

**Energy / Lifecycle Rules**

| Mechanic | Current behavior |
|----------|------------------|
| Passive metabolism drain | `0.1 × metabolism` every tick |
| Move cost | `terrain_speed × speed × metabolism × 0.15` |
| Turn-right cost | `0.1 × metabolism` inside `turn()` plus `0.04 × metabolism` in action execution |
| Turn-left cost | `0.1 × metabolism` inside `turn_left()` plus `0.04 × metabolism` in action execution |
| Eat gain | Up to `33.3` energy per food unit, capped by `energy_capacity` |
| Sleep effect | Adds `0.15 × metabolism` energy (nearly neutral vs passive drain) |
| Loaf / do nothing | Costs `0.005 × metabolism` and skips the next turn |
| Reproduction minimum age | 100 ticks |
| Reproduction minimum energy | 40 |
| Reproduction cost | `current_energy × (0.50 × reproduction_cost_trait)` |
| Max offspring per lifetime | Random 1–10, assigned at birth |
| Reproduction cooldown | `(death_age − 100) ÷ max_offspring` ticks between births |
| Birth failure chance | 2%; failed birth has 20% chance of still consuming one slot |
| Offspring spawn | One adjacent wrapped tile, chosen at random |
| Offspring starting energy | Equal to the energy cost paid by the parent |
| Natural death | Randomly assigned death age drawn from `create_death_range()` |
| Starvation | Agent dies at `energy <= 0` |

**Food Regrowth**

Food regenerates passively every tick. Each tile rolls against its `regen_rate`:

`regen_rate = (fertility / 1.6) × 0.012`

A successful roll adds 1 food unit, capped at **3 per tile**. High-fertility tiles grow food more often. 35–45% of tiles are permanently barren (fertility = 0) and never regenerate, arranged in contiguous desert clusters with fertile oases between them.

**Runtime Notes**

The simulation uses PyTorch for inference and will pick `cuda` automatically when available. The environment batches perception/decision work per tick, but each agent still owns its own `Brain` instance loaded from its genome weights.
