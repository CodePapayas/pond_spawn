# pond_core

Pure Rust simulation engine for pond_spawn. Compiles to native (headless) or WASM (browser renderer via `pond_web/`).

> **This crate is the product of two refactors.** (1) The **engine refactor**
> ported the simulation from Python + PyTorch to this deterministic Rust crate —
> full history in [`../REFACTOR_RUST_BEVY.md`](../REFACTOR_RUST_BEVY.md). (2) The
> **renderer/visualization refactor** rebuilt `pond_web/` around trait-driven
> agent morphology, Oklch color smoothing, stable cluster labels, and
> interactive inspection panels — summarized in
> [Renderer visual key](#renderer-visual-key) below and driven by the WASM
> exports (`inspect_agent`, `trait_means`, `trait_bounds`) added for it.

---

## Build

```bash
# WASM (browser)
wasm-pack build pond_core --target web --features wasm

# Headless native runner
cargo run --bin run --release --features native -- [grid] [pop] [steps] [seed]
# defaults: 12×12, 100 agents, 500 steps, seed 42
```

---

## Architecture

```
src/
  biome.rs    BiomeTile — fertility, food, movement_speed, visibility
  brain.rs    Hand-rolled 5→12→12→12→8 MLP, 488 weights, ReLU, sigmoid output gates
  genome.rs   Genome + Traits (9 trait fields), mutation, effective_mutation_rate
  morphology.rs MorphParams — maps genome traits → normalized [0,1] shape knobs for the renderer
  memory.rs   AgentMemory — 10-action ring buffer, success detection, mutation suppression
  cluster.rs  Dual k-means: genome traits (k=6) + brain weights (k=24), every 50 steps
  spatial.rs  SpatialHashGrid — tile-aligned buckets, f32 coords, toroidal agents_near()
  world.rs    World — SoA arrays, 20 Hz step loop, continuous-space physics
  wasm.rs     WasmWorld — wasm-bindgen wrapper, get_state() Float32Array, stir/pour,
              plus renderer-refactor exports: inspect_agent, trait_means, trait_bounds
  bin/run.rs  Headless runner: prints step table, death tallies, cluster distributions
```

### Step loop (order fixed)

1. Food regen (per-tile probabilistic)
2. Age + passive metabolism drain → natural deaths
3. Spatial rebuild
4. Partition alive agents into acting set
5. Perception → 5-input vector per agent
6. Brain forward → 8 sigmoid outputs
7. Integrate steering forces + fire discrete triggers (eat / reproduce / sleep)
8. Passive combat per tile (aggression-gated)
9. Spawn offspring
10. Reap dead (swap-remove)
11. Spatial rebuild
12. k-means clustering (every 50 steps)

---

## Neural net — input / output contract

**Inputs (5)**

| # | Name | Description |
|---|------|-------------|
| 0 | `energy_norm` | own energy / 100 |
| 1 | `food_dist_norm` | distance to nearest food / vision_radius (1.0 = none visible) |
| 2 | `food_angle_norm` | angle to food relative to velocity direction, in [−1, 1] |
| 3 | `agent_density_norm` | neighbours within 1.2 tiles / 8 |
| 4 | `speed_norm` | current speed / max speed |

**Outputs (8) — sigmoid gated**

| # | Name | Effect |
|---|------|--------|
| 0 | `seek_weight` | steering force toward nearest food |
| 1 | `wander_weight` | random perturbation force |
| 2 | `separation_weight` | repulsion from nearby agents |
| 3 | `flee_weight` | dormant (future threat system) |
| 4 | `eat_trigger` | > 0.5 → eat from current tile |
| 5 | `reproduce_trigger` | > 0.5 → attempt reproduction |
| 6 | `attack_weight` | dormant (combat routes through aggression trait) |
| 7 | `sleep_trigger` | > 0.5 → give back 0.05 × metabolism energy |

---

## Renderer visual key

The `pond_web/` Canvas2D renderer draws agents as 7-segment kinematic chains with additive blending. The renderer refactor made the body **trait-driven** — its silhouette reads the genome — and added color smoothing plus interactive panels. Here's how to read what you see.

### Agent color — genome cluster (with smoothing)

Color encodes which of 6 genome clusters the agent belongs to. Clusters are recomputed every 50 steps via k-means on the 9 trait values.

| Color | Cluster | Rough trait tendency* |
|-------|---------|----------------------|
| **Teal** `#00ffc8` | 0 | balanced generalists |
| **Violet** `#6440ff` | 1 | high metabolism / fast burners |
| **Magenta** `#ff3c8c` | 2 | aggressive / high attack |
| **Cyan** `#28d2ff` | 3 | high vision / explorers |
| **Amber** `#ffb900` | 4 | high reproduction rate |
| **Lime** `#50ff3c` | 5 | high defense / slow metabolism |

*Tendencies emerge from evolution — not hardcoded. Early runs may show all clusters as mixed.*

**Refactor detail — no color flashing.** Two mechanisms keep color changes smooth: (1) `cluster.rs` matches each k-means re-fit's centroids to the previous run's and remaps labels, so a persisting lineage keeps its label → keeps its color even though raw k-means labels are arbitrary; (2) the renderer crossfades each agent toward its cluster color in **Oklch** (`pond_web/color.js`) over ~1s, so genuine reassignments fade through vivid hues instead of popping.

### Agent shape — genome morphology

Body segments follow a size envelope (narrow head → widest mid-body → tapering tail), but the envelope and ornaments are now derived per-agent from traits via `morphology.rs` (`MorphParams`, normalized [0,1]) and `pond_web/morphology.js`.

| Visual property | Driven by |
|----------------|-----------|
| **Pointed wedge head + swept-back fins** | `aggression` |
| **Body length / taper sharpness** | `speed` (elongation) |
| **Mid-body width + white armor rings** | `defense` (bulk) |
| **Head spikes** | `attack` (ornament) |
| **Eye size + lateral spread** | `vision` |
| **Glow pulse rate** | `metabolism` |
| **Belly bulge** | `energy_capacity` |
| **Overall size / glow radius / brightness** | `energy_norm` — bright, large halo = well-fed; dims as it starves |
| **Eye + body orientation** | Velocity vector — the head faces the direction of travel |

### Interactive panels (renderer refactor)

| Panel | What it shows | Data source |
|-------|---------------|-------------|
| **Legend** (right, `L` toggles, on by default) | Cluster color swatches with live per-cluster counts, shape→trait glyph key, tile-color key | client-side tally + static key |
| **Average genome** (right, under legend) | Population mean of each of the 9 traits as bars normalized to trait bounds, with sparkline history; dominant traits highlighted | `trait_means()`, `trait_bounds()` |
| **Inspector** (left, click an agent) | Selected agent's live brain: 5→12→12→12→8 node activations (brightness = magnitude), input/output labels, sigmoid gate bars, energy/age/traits | `inspect_agent(id)` — a pure traced forward pass, no sim mutation |

### Tile colors

| Tile appearance | Meaning |
|----------------|---------|
| Very dark blue-black | Barren tile — low fertility, slow food regen |
| Slightly brighter blue | Fertile tile — faster food regen |
| Green radial glow | Food present — brighter = more food units (max 3) |
| Glow fades | Food was eaten — will slowly regenerate |

### Interactions

| Input | Effect |
|-------|--------|
| Click agent | Select + open the **inspector** panel (drag threshold distinguishes click from stir) |
| Drag (left mouse) | `stir` — scatters agents outward, drains food, suppresses fertility |
| Double-click | `pour_agents` — spawns 12 new agents at cursor position |
| Space | Pause / resume |
| `+` / `-` | Speed up / slow down simulation (×0.25 to ×16) |
| `L` | Toggle the legend / average-genome panel |
| `Esc` | Deselect the inspected agent |

---

## Economy constants

| Constant | Value | Role |
|----------|-------|------|
| `FOOD_ENERGY` | 33.3 | energy per food unit consumed |
| `MAX_ENERGY_BASE` | 100.0 | base energy cap (× `energy_capacity` trait) |
| `BIRTH_ENERGY_TRANSFER` | 0.40 | 40% of reproduction cost becomes child energy; 60% overhead |
| `DT` | 1/20 s | physics timestep (20 Hz) |
| `MAX_SPEED` | 3.0 | tiles/sec at `speed` trait = 1.0 |
| `SEPARATION_RADIUS` | 1.2 tiles | repulsion zone between agents |
| `VISION_SCALE` | 3.0 | `vision` trait × 3 = vision radius in tiles |
| Passive drain | `0.1 × metabolism` / tick | baseline energy cost of being alive |
| Movement cost | `distance × metabolism × 0.12` / tick | additional cost proportional to speed |
| Sleep net drain | `0.05 × metabolism` / tick | half passive drain when sleep trigger fires |

---

## Next steps

### 1 · Threat detection + flee mechanic

Currently `OUT_FLEE` (output index 3) is dormant — no threat inputs exist and the flee force is never computed.

**What's needed:**
- Add two new NN inputs: `threat_dist_norm` (distance to nearest high-aggression agent / vision_radius) and `threat_angle_norm` (angle to that agent relative to velocity, in [−1, 1]). This expands the input vector from 5 → 7 and changes the MLP to `7→12→12→12→8` (new weight count: 604).
- In `perceive()`: scan `spatial.agents_near()` for agents where `aggression > threshold` and `genome[other].traits.attack > own_defense`. Emit distance + angle to the nearest qualifying threat.
- In `integrate_agent()`: wire `outputs[OUT_FLEE]` as a steering force directly opposite the threat direction (flee = push away).
- Update `brain.rs` (`WEIGHT_COUNT`, layer constants), `genome.rs` (brain weight vec length), and this README's NN contract table.

**Files:** `src/brain.rs`, `src/genome.rs`, `src/world.rs` (perceive + integrate), `pond_web/renderer.js` (no change needed — flee manifests as movement).

---

### 2 · Kin detection

Agents currently have no way to sense relatedness. Kin detection enables kin-selection pressure: agents could avoid attacking relatives, preferentially reproduce near them, or share food tiles.

**What's needed:**
- Add NN input `kin_density_norm`: fraction of agents within separation radius that share the same `genome_cluster_id` as self, normalized to [0, 1]. Expands inputs by 1 (coordinate with threat detection above — do both expansions together).
- The cluster ID is already computed every 50 steps in `cluster.rs`. Between clustering ticks, each agent reads its cached `cluster.genome_cluster_ids[i]`.
- Behavioral effects emerge from evolution — no hardcoded rules needed. Kin density as an input lets the brain learn "lots of relatives nearby → act differently."
- Optional explicit rule: suppress `passive_eat` when `attacker` and `victim` share a genome cluster (soft kin protection without NN change).

**Files:** `src/world.rs` (perceive), optionally `src/world.rs` (resolve_combat_spatial for soft rule).

---

### 3 · Two-parent (sexual) reproduction

Currently reproduction is asexual — one parent mutates its own genome. Two-parent adds crossover: offspring genome is a blend of two parents, then mutated.

**What's needed:**
- **Crossover operator** in `genome.rs`: `Genome::crossover(parent_a: &Genome, parent_b: &Genome, rng) -> Genome`. Per-trait: randomly inherit from A or B (uniform crossover). Brain weights: single-point crossover at a random index. Apply `mutate()` after.
- **Trigger condition**: when `reproduce_trigger > 0.5` fires for agent A, scan the same tile for agent B where `B.reproduce_trigger` was also high last tick (or store a "ready to mate" flag). If found, crossover. If not, fall back to asexual.
- **Mate selection**: simplest = nearest agent on tile. Richer = prefer same cluster (kin detection above enables this).
- **Energy split**: both parents pay `reproduction_cost * 0.25` each; child energy = sum × `BIRTH_ENERGY_TRANSFER`.
- Requires `reproduce_trigger` state to persist one tick — add `wants_to_mate: Vec<bool>` scratch buffer cleared each step.

**Files:** `src/genome.rs` (crossover), `src/world.rs` (do_reproduce + new scratch buffer).

---

### 4 · Click-to-inspect agent panel — ✅ shipped in the renderer refactor

Delivered as the **inspector panel** (left side) rather than a radial menu:
click an agent to select it (a drag threshold distinguishes select from stir),
draw a pulsing ring on its head, and open a panel showing its live brain
(5→12→12→12→8 node activations + sigmoid gate bars), energy/age, and all 9
trait values. Backed by the `inspect_agent(id)` WASM export — a pure traced
forward pass that never mutates sim state.

**Files (shipped):** `pond_web/inspector.js`, `pond_web/renderer.js` (selection
+ ring), `src/wasm.rs` + `src/world.rs` (`inspect_agent`), `src/brain.rs`
(`forward_traced`).

**Still open:** cluster-highlight-on-select, and a `world.kill_agent(id)`
export for a "kill" affordance.

---

### 5 · k-means speciation

k-means already runs every 50 steps (`cluster.rs`) and assigns `genome_cluster_ids` and `brain_cluster_ids` to each agent. Currently those IDs only drive renderer color. Speciation wires them into behavior and tracking.

**What's needed:**

**Engine side:**
- **Intra-species reproduction preference**: in `do_reproduce()`, if two-parent is implemented, bias mate selection toward same genome cluster.
- **Inter-species aggression modifier**: in `resolve_combat_spatial`, scale attack probability by cluster distance — agents far apart in genome space fight more readily (or less — let evolution decide, expose it as a trait).
- **Speciation events**: track when cluster centroids diverge beyond a threshold between consecutive clustering runs. Record `(step, cluster_a, cluster_b, divergence)` to a `speciation_log: Vec<SpeciationEvent>` in `World`. Export via `get_speciation_log() -> Vec<f32>` in `wasm.rs`.

**Renderer side:**
- **Species HUD panel**: persistent sidebar showing 6 genome clusters as colored rows with live agent count per cluster. Toggle with `S` key.
- **Phylogeny sketch**: as speciation events accumulate, draw a simple branching diagram — horizontal lines per cluster, vertical connections when clusters split. Canvas2D only; updates every 50 steps.
- **Cluster highlight mode**: press a cluster color in the HUD to highlight all agents of that species and dim others.

**Files:** `src/cluster.rs` (divergence tracking), `src/world.rs` (combat modifier, reproduce bias), `src/wasm.rs` (speciation log export), `pond_web/renderer.js` (species HUD, phylogeny sketch).
