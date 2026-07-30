```
  ██████╗  ██████╗ ███╗   ██╗██████╗       ███████╗██████╗  █████╗ ██╗    ██╗███╗   ██╗
  ██╔══██╗██╔═══██╗████╗  ██║██╔══██╗      ██╔════╝██╔══██╗██╔══██╗██║    ██║████╗  ██║
  ██████╔╝██║   ██║██╔██╗ ██║██║  ██║      ███████╗██████╔╝███████║██║ █╗ ██║██╔██╗ ██║
  ██╔═══╝ ██║   ██║██║╚██╗██║██║  ██║      ╚════██║██╔═══╝ ██╔══██║██║███╗██║██║╚██╗██║
  ██║     ╚██████╔╝██║ ╚████║██████╔╝      ███████║██║     ██║  ██║╚███╔███╔╝██║ ╚████║
  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝
```

[![Rust Tests](https://github.com/codepapayas/pond_spawn/actions/workflows/rust-test.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions/workflows/rust-test.yml)
[![Clippy](https://github.com/codepapayas/pond_spawn/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions/workflows/rust-clippy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Language](https://img.shields.io/badge/language-rust-orange)

*************************

## ⚙️ RUST REFACTOR — READ THIS FIRST
*************************

**The simulation has been refactored from Python to Rust.** The canonical
engine is now [`pond_core`](pond_core/README.md) — a pure-Rust, deterministic,
WASM-compilable crate — driven in the browser by the [`pond_web`](pond_web/)
Canvas2D renderer. The original Python implementation is archived under
[`legacy_python/`](legacy_python/README.md) for reference and no longer
reflects the live sim.

See [`REFACTOR_RUST_ROADMAP.md`](REFACTOR_RUST_ROADMAP.md) for the full refactor log
and [`pond_core/README.md`](pond_core/README.md) for the engine + renderer spec.

What the refactor changed at a glance:

- **Engine:** Python + PyTorch → hand-rolled Rust MLP (`7→12→12→12→8`, 512
  weights), SoA `World`, `ChaCha8Rng` determinism, native + WASM targets.
- **Physics:** discrete tile actions (MOVE/TURN/EAT) → continuous-space
  steering forces (seek/wander/separate) + sigmoid trigger gates
  (eat/reproduce/sleep). Positions are `f32`, interpolated between 20 Hz ticks.
- **Renderer (`pond_web`):** trait-driven agent **morphology** (body shape
  reads genome traits), **Oklch color smoothing** so cluster reassignments fade
  instead of flashing, stable cluster labels across k-means re-fits, a
  click-to-inspect **neuron panel**, a color/shape **legend**, and a running
  **average-genome** panel. Pond fills the whole window.

*************************

### Pond Spawn
![Simulation Visualization GIF](assets/gifs/rust_pond_main_interface.gif)

## Neural Network and Genome Expanded View
![Simulation GIF](assets/gifs/rust_pond_nn_inspector.gif)
**Inspect individual agents genetic makeup and neural network**
Click on any agent and view their genome and neural network to see how an agent makes their decisions in real time.

> **Status: in development.** Not published anywhere yet — run it locally per
> below. An itch.io release is on the roadmap, see
> [`REFACTOR_RUST_ROADMAP.md`](REFACTOR_RUST_ROADMAP.md).
>
> **Next up: speciation.** Stable genome clusters will be promoted to named
> species with their own colour, founding step and lineage record, instead of
> today's k-means labels that carry no identity across re-fits. Full plan in
> [`PLAN_SPECIATION.md`](PLAN_SPECIATION.md).

*************************

## Features
*************************

- **Evolving neural-network agents** — feedforward brain (`7→12→12→12→8`,
  512 weights) per agent, weights inherited and mutated on reproduction. No
  backprop; evolution is the optimizer.
- **Continuous-space physics** — agents steer via seek/wander/separate forces
  and sigmoid trigger gates (eat/reproduce/sleep) instead of discrete grid
  steps; positions are `f32`, interpolated between fixed 20 Hz sim ticks.
- **Genome-driven morphology** — each agent's body shape, size, and coloring
  is derived directly from its own genome traits, so lineages are visually
  distinguishable at a glance. Colour reads the lineage's combat strategy on a
  three-ramp palette: lime is passive, cyan is middling, magenta is aggressive.
- **Run setup + live stat graphs** — choose grid size, starting population and
  seed before a run; a toggleable panel plots population, food, average energy,
  the lifespan band and per-cause deaths over the last ~6000 steps, sampled
  engine-side so nothing is lost to a dropped frame.
- **God mode** — comet, spreading salt, a sweep that empties the pond,
  immortality, and **ultra predators**: unkillable apex hunters that cull the
  pond back inside its capacity band and then swim off. Three shapes — the grey
  triangles are the ecology's own and summon themselves in packs when the
  population outgrows what the renderer can draw; the red octagon and the
  rainbow rectangle are player powers, and the rectangle kills everything its
  edges touch however armoured. Any summoned hunt can be called off with
  `dismiss hunters`, and the triangle ecology has its own on/off toggle.
- **Predators that swim** — a hunter's velocity is state, not a per-tick
  recomputation: it commits to one animal, banks onto it at a bounded turn rate,
  eases its speed between hunting, patrolling and leaving, and patrols on a
  smoothed wander when there is nothing left to cull.
- **Live lineage clustering** — dual k-means (genome traits + brain weights)
  re-fits every 50 steps; cluster colors smoothly fade between reassignments
  (Oklch interpolation) instead of flashing, and cluster labels stay stable
  across re-fits.
- **Click-to-inspect agent panel** — click any agent to see its energy, age,
  full trait bars, and a live-activation diagram of its neural network as it
  makes decisions in real time.
- **Running population panels** — a colors/shapes legend and an average-genome
  panel (population mean per trait, ~1200-step rolling history) update live
  alongside the sim.
- **Pond interaction** — drag to stir (disturbs food, fertility, and nearby
  agent velocity), double-click to pour in new agents at a point, wheel to zoom
  and right-drag to pan around the pond.
- **Deterministic engine** — seeded `ChaCha8Rng`, same seed reproduces the
  same run; compiles to a headless native binary or to WASM for the browser.

## Running locally
*************************

Prereqs: [Rust toolchain](https://rustup.rs/) + `wasm-pack`
(`cargo install wasm-pack`).

```bash
# Build the engine to WASM
wasm-pack build pond_core --target web --features wasm

# Serve the repo root and open /pond_web/ in a browser
python3 -m http.server
# then visit http://localhost:8000/pond_web/
```

Controls in the browser:

| Input | Action |
|-------|--------|
| Click agent | Open inspector panel (genome, energy, live neuron activations) |
| Click + drag | Stir the pond at that point |
| Double-click | Pour in a batch of new agents |
| `Space` | Pause / resume |
| `+` / `-` | Speed up / slow down |
| `l` | Toggle legend |
| `Esc` | Deselect agent |

To run the sim headless (no renderer, for testing/perf work):

```bash
cargo run -p pond_core --bin run --release
```

The original Python implementation (PyTorch brains, pygame visualizer, CLI
menu) is archived in [`legacy_python/`](legacy_python/README.md) — frozen,
unmaintained, not covered by CI.
