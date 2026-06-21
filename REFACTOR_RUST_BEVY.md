# pond_spawn → Rust Refactor

**Engine status: complete.** All phases shipped. Remaining work: interactive GUI.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Pin spec, golden harness (Python seed + JSON trace) | ✅ Done |
| 1 | `pond_core` crate — data structures, brain, genome, biome, spatial | ✅ Done |
| 2 | Step loop, single-threaded — all 9 phases, SoA World, combat, reproduction | ✅ Done |
| 3 | WASM exports — `WasmWorld`, `step_n`, `get_state`, `inject_food`, `stir` | ✅ Done |
| 4 | Memory + clustering — `AgentMemory` ring buffer, dual k-means, suppression | ✅ Done |
| 5 | Performance hardening — scratch buffers, spatial combat, rayon feature, native runner | ✅ Done |
| Pass A | Continuous-space physics — steering forces, velocity integration, SpatialHashGrid | ✅ Done |
| Pass B | Canvas2D renderer — kinematic body chains, additive glow, stir + pour interaction | ✅ Done |
| 6 | Interactive GUI — WebGL upgrade, morphology, cluster UI, full HUD | 🔲 TODO |

---

## Delivery target

```
pond_core (pure Rust, no Bevy)
    └─ wasm-pack → pond_core/pkg/*.wasm  (122 KB, optimized)
          └─ Phase 6: JS renderer (Canvas2D / WebGL + physics)
```

---

## What is built

### `pond_core` crate (`pond_core/`)

All simulation logic. WASM-safe, deterministic, seeded with `ChaCha8Rng`.

**Modules:**

| File | Contents |
|------|----------|
| `src/biome.rs` | `BiomeTile` — fertility, food, movement_speed, visibility; BFS barren cluster init |
| `src/brain.rs` | Hand-rolled 5→12→12→12→8 MLP, 488 weights, ReLU, softmax, multinomial sample |
| `src/genome.rs` | `Genome` + `Traits` (12 fields), mutation with D3 locked traits, `effective_mutation_rate` (D4) |
| `src/memory.rs` | `AgentMemory` — 10-action ring buffer, success detection, suppression factor |
| `src/cluster.rs` | Dual k-means: genome (euclidean, k=6) + brain (cosine via normalized euclidean, k=24), k-means++ init |
| `src/spatial.rs` | `SpatialHashGrid` — tile-aligned buckets, f32 coords, toroidal `agents_near(radius)` |
| `src/world.rs` | `World` — SoA arrays, full step loop, scratch buffers, `get_stats()` |
| `src/wasm.rs` | `WasmWorld` — `#[wasm_bindgen]` wrapper, `get_state() → Float32Array`, state buffer layout |
| `src/bin/run.rs` | Headless native runner: `cargo run -p pond_core --bin run --release` |

**Features:**

| Feature | Effect |
|---------|--------|
| *(default)* | Single-threaded, WASM-safe |
| `wasm` | Enables `wasm-bindgen` exports (`WasmWorld`) |
| `native` | Enables `rayon` for parallel perception |

**Test suite:** 31 tests across all modules. All pass.

**Release perf (12×12, 100 agents):** ~0.003 ms/step baseline, ~4 ms at clustering ticks (every 50 steps). 500 steps in 16 ms total.

### WASM package (`pond_core/pkg/`)

Built with `wasm-pack build pond_core --target web --features wasm`.

- `pond_core_bg.wasm` — 122 KB, wasm-opt optimized
- `pond_core.js` — ES module glue
- `pond_core.d.ts` — TypeScript types

**JS API:**
```js
import init, { WasmWorld, state_header_len, state_agent_stride, state_tile_stride } from './pond_core.js';
await init();

const world = new WasmWorld(12, 100, 42n);  // grid_size, population, seed

// Fixed-timestep loop (preferred):
function frame(ts) {
    world.update(deltaMs);          // drains accumulator in 50ms ticks
    const alpha = world.get_alpha(); // interpolation factor in [0,1)
    draw(world.get_state(), alpha);
    requestAnimationFrame(frame);
}

// Or legacy step-by-step:
world.step_n(1);

const buf = world.get_state();  // Float32Array
// buf[0..6]               → header: [agent_count, grid_size, step, total_food, avg_energy, alpha]
// buf[6 + i*12 .. +12]    → per agent: [x, y, prev_x, prev_y, energy_norm, vel_x, vel_y,
//                                        genome_cluster, brain_cluster, age_norm, aggression, speed]
// buf[6 + n*12 + t*3..]   → per tile:  [food_units, fertility, movement_speed]

world.inject_food(cx, cy, amount);               // float world coords
world.stir(cx, cy, radius, intensity);           // disturbs food, fertility, agent velocities (scatter outward)
world.pour_agents(cx, cy, count);               // spawn agents near a point
```

**Pass A changes (continuous-space physics):**

| Before | After |
|--------|-------|
| `pos_x/y: Vec<u16>` (tile grid) | `pos_x/y: Vec<f32>` (continuous [0, grid_size)) |
| `heading: Vec<u8>` (0–3 cardinal) | `vel_x/y: Vec<f32>` (tiles/sec) + `prev_x/y` for lerp |
| Discrete actions (MOVE/TURN/EAT/…) | Steering force weights (seek/wander/separate) + trigger gates (eat/reproduce/sleep) |
| `SpatialIndex` (u16 tile lookup) | `SpatialHashGrid` (f32, toroidal `agents_near(radius)`) |
| `step_n(n)` only | `update(delta_ms)` accumulator + `get_alpha()` for interpolation |
| `inject_food(x: u16, y: u16, …)` | `inject_food(cx: f32, cy: f32, …)` (world coords) |
| `stir` rotates heading | `stir` applies velocity impulse (scatter outward) |
| Agent stride = 8 | Agent stride = 12 (adds prev_x/y, vel_x/y, speed) |

**NN input/output contract (locked):**
```
Inputs  [0] energy_norm          own energy / 100
        [1] food_dist_norm        dist to nearest food / vision_radius (1.0 = none visible)
        [2] food_angle_norm       angle to food relative to velocity [-1, 1]
        [3] agent_density_norm    neighbors within 1.2 tiles / 8.0
        [4] speed_norm            |vel| / max_speed

Outputs [0] seek_weight          → force toward nearest food
        [1] wander_weight        → random perturbation force
        [2] separation_weight    → repulsion from nearby agents
        [3] flee_weight          → dormant (future)
        [4] eat_trigger          → gate > 0.5 fires do_eat()
        [5] reproduce_trigger    → gate > 0.5 fires do_reproduce()
        [6] attack_weight        → dormant (passive combat via aggression trait)
        [7] sleep_trigger        → gate > 0.5 gives back 0.05×metabolism
```

### Sim economy changes (vs Python)

| Mechanic | Before | After | Reason |
|----------|--------|-------|--------|
| Sleep | `+0.15 × metabolism` | `-0.05 × metabolism` | Rest, not recovery |
| Child energy | `= reproduction_cost` (full) | `= cost × 0.40` | Thermodynamic loss — 60% overhead |
| Memory suppression | — | `1 / (1 + success_count × 0.05)` | Successful lineages compress mutation rate |

---

## Locked-in decisions (carried from Python ratification)

| ID | Decision |
|----|----------|
| D1 | Softmax-sampled action (not argmax) |
| D2 | Food unit = 33.3 energy |
| D3 | `energy_capacity`, `mutation_rate` locked — skip RNG draw in mutate |
| D4 | `effective_mutation_rate: f32` separate heritable field; suppressed at reproduction by memory |

---

## TODO — Phase 6: Interactive GUI

The engine is done. The renderer doesn't exist yet. This is the remaining work.

### Goal
A fluid, physics-backed web UI that feels alive. The grid should look like a living pond — agents move through it, food ripples, the stir mechanic creates visible disturbance.

### What `stir` already does (engine side)
- Drains food on tiles within radius (scaled by intensity)
- Suppresses tile fertility within radius
- Rotates agent headings within radius

### What the renderer needs to implement
- **Canvas2D / WebGL base** — grid tiles colored by food density and fertility; agents as animated sprites or particles
- **Physics layer** — fluid or soft-body simulation for visual continuity between sim steps; agents should feel like they're swimming, not teleporting
- **Stir interaction** — pointer hold + drag calls `world.stir(cx, cy, radius, intensity)` and produces visible fluid disturbance (ripple, scatter, wake)
- **Cluster coloring** — use `genome_cluster_id` and `brain_cluster_id` from `get_state()` to color agents by lineage/brain type
- **HUD** — step counter, population, avg energy, food total; toggled overlays for cluster view, energy heatmap, fertility map

### Architecture notes
- Import `pond_core.js` as an ES module; call `step_n(1)` per animation frame
- State buffer is a flat `Float32Array` — parse with the layout constants (`state_header_len()`, `state_agent_stride()`, `state_tile_stride()`)
- Physics runs client-side in JS/WebGL; the Rust engine is authoritative for all sim state
- No Bevy — renderer is pure browser tech

---

## 1. Original Python Architecture (reference)

### Data model
- **`Environment`** owns: `grid` (`list[list[Biome]]`), `agents` (list), `agents_by_id` (dict), `position_map` (dict `(x,y)→set`)
- **`Agent`**: `energy`, `age`, `position`, `heading` 0–3, `skip_turn`, reproduction bookkeeping, `death_age`, `parent_id`, `parent_defense_bonus`, `cause_of_death`, owns a `Genome` and a `Brain` (`torch.nn.Module`)
- **`Genome`**: `id`, `traits` (12-entry dict `{min, max, mutable, value}`), `brain_weights` (flat list, 488 floats)
- **`Biome`**: `movement_speed`, `visibility`, `fertility`, `food_units`

### Performance bottlenecks (now resolved in Rust)
1. Per-agent PyTorch inference loop — 488-op MLP run through CUDA launch overhead
2. Per-agent `Brain` nn.Module construction on reproduction
3. Three-container bookkeeping (`agents`, `agents_by_id`, `position_map`) with per-step list rebuilds
4. `deepcopy` of genome dicts on every `mutate()`

### Neural net
- 5 → 12 → 12 → 12 → 8, ReLU between layers, 488 total params
- Weight layout per layer: `[weights(in×out), bias(out)]`
- Init: weights `uniform(-0.5, 0.5)`, biases `0.001`
- Output: softmax → multinomial sample (D1)

---

## 2. Crate dependencies

| Crate | Purpose |
|-------|---------|
| `rand` | RNG traits, distributions |
| `rand_chacha` | `ChaCha8Rng` — deterministic, WASM-safe |
| `serde` + `serde_json` | JSON config loading, golden snapshots |
| `smallvec` | Per-tile occupancy buckets |
| `wasm-bindgen` *(optional, `wasm` feature)* | JS↔WASM boundary |
| `rayon` *(optional, `native` feature)* | Parallel perception on native builds |
| `approx` *(dev)* | Float tolerance in tests |
