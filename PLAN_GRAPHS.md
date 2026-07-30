# Plan — Live Stat Graphs (parity with the Python matplotlib view)

> **Status: steps 1–6 shipped.** `pond_core/src/stats.rs`, `World::stats_history`,
> the `stats_history()` / `death_totals()` / `peak_population()` wasm exports,
> `pond_web/graphs.js` (`G` toggles), and `--dump-stats PATH` on the headless
> runner all exist. Two deviations from the plan below, both deliberate:
> the panel is a centred bottom strip rather than a left sidebar (the inspector
> already owns the left edge, and five panels read better wide than stacked),
> and the summary/death tables are a compact footer line rather than DOM tables.
> Death series colors are defined in `graphs.js`; the legend codes causes as
> glyphs, not colors, so there was nothing to inherit.


Goal: bring back the analytical readability of the old
`plot_simulation_stats()` panel (`legacy_python/cli/cli_sim_starter.py:183`),
but live and in-browser rather than a static figure printed after the run ends.

The web build currently shows four scalars in the HUD (`index.html:172`) and
trait sparklines in the genome panel (`panels.js:167`). Everything the Python
figure plotted over time — population, food, energy, lifespan band, deaths by
cause — is either not retained at all or retained only as a transient event
queue that the renderer drains and discards.

---

## What the Python view showed (the parity target)

Five stacked time-series plots, plus three tables, plus a genome panel:

| Panel | Series | Source today |
|---|---|---|
| Population | alive agents vs step | `SimStats::alive_agents` ✅ |
| Food | total food units vs step | `SimStats::total_food` ✅ |
| Energy | population mean energy vs step | `SimStats::avg_energy` ✅ |
| Lifespan | median line + min–max age band | median ✅, **min/max age missing** |
| Deaths | starvation / combat / old age, per cause | **tally is cumulative-only, not per-step** |
| Summary table | start pop, final pop, peak, steps survived | derivable from history |
| Death table | final breakdown by cause | `death_tally` ✅ |
| Genome panel | mean trait bars | `trait_means()` ✅ |

Two gaps to close in `pond_core`, everything else is plumbing.

Note the death causes differ: Python had three (starvation, combat, old age),
Rust has four — `CauseOfDeath` (`world.rs:52`) splits combat into
`KilledInCombat` and `EatenAlive`. Plot four series, not three. Keep the numeric
codes as-is; the renderer's death effect keys off them (`world.rs:63`).

---

## Design decision: history lives in Rust, not JS

The sparkline history in `panels.js:13` is a JS-side array, 120 samples, rebuilt
from scratch on reload. That is fine for one panel and wrong as the general
mechanism, for three reasons:

1. Any second consumer (the headless runner, a future renderer) needs the same
   series and would otherwise reimplement the sampling — two samplers, two
   chances to drift.
2. Per-cause death *rates* cannot be reconstructed in JS: the renderer sees
   individual death events only for frames it happens to observe, and
   `get_state()` drains the queue (`wasm.rs:128`). A frame drop loses data.
3. A headless run (golden harness, `--steps 5000`) should be able to dump the
   same series without a renderer attached at all.

So: a ring buffer in `pond_core`, one export to read it.

---

## Step 1 — `pond_core/src/stats.rs` (new module)

```rust
pub const HISTORY_LEN: usize = 600;      // samples
pub const SAMPLE_INTERVAL: u32 = 10;     // steps → 6000 steps of coverage

#[derive(Clone, Copy, Default)]
pub struct StatSample {
    pub step: u32,
    pub alive: u32,
    pub total_food: u32,
    pub avg_energy: f32,
    pub median_lifespan: f32,
    pub min_age: u32,          // over living agents
    pub max_age: u32,
    pub deaths: [u32; 4],      // per cause, THIS interval — not cumulative
}

pub struct StatHistory { /* fixed-size ring, head index, len */ }
impl StatHistory {
    pub fn push(&mut self, s: StatSample);
    pub fn iter_chrono(&self) -> impl Iterator<Item = &StatSample>;
}
```

Ring buffer, not `Vec` — a long unattended run must not grow memory without
bound, and 600 × ~9 fields is negligible.

`deaths` is per-interval, and that is the important part. The existing
`death_tally` (`world.rs:144`) is cumulative, which plots as a monotone staircase
where what you actually want to see is *when* a starvation wave hit. Keep the
cumulative tally for the summary table; derive per-interval counts by
differencing the tally at sample time.

## Step 2 — Wire into `World`

- Field `pub stats_history: StatHistory` on `World`.
- Sample at the **end** of `step()`, after `reap_dead`, when
  `step_count % SAMPLE_INTERVAL == 0`. Do not insert this into the middle of the
  step loop — the nine-phase order is load-bearing (project rules), and sampling is
  an observation, not a phase.
- `min_age`/`max_age` come from a single pass over `self.age` for living agents.
  At ~10k agents once per 10 steps this is free; do not add per-tick tracking.

## Step 3 — Export

```rust
#[wasm_bindgen]
pub fn stats_sample_stride() -> u32;   // = 11

impl WasmWorld {
    /// Flat Float32Array, chronological, `len / stride` samples.
    pub fn stats_history(&self) -> Vec<f32>;
}
```

Same flat-buffer convention as `get_state()`. Do not invent a second
serialization style — one buffer layout idiom for the whole boundary.

Call it once per second from JS (`setInterval`, not per frame). The graphs
redraw at 1 Hz; nothing about a population curve needs 60 fps, and redrawing
five canvases per frame is exactly the kind of cost the renderer cannot absorb.

## Step 4 — `pond_web/graphs.js` (new module)

One module, one exported `initGraphs(container) → update(historyBuffer)`.
Follows the existing `panels.js` shape: build DOM once, return an updater.

- Five stacked `<canvas>` elements, ~260 × 70 each, in a collapsible left panel
  (`#side-left`), mirroring `#side-right`. **Must not overlay the grid** — side
  panel only (the visualizer rule applies to the web renderer for the
  same reason it applied to pygame).
- Toggle with `G`. Default hidden, so first paint stays clean.
- Drawing: plain Canvas2D polyline per series, matching the existing sparkline
  code (`panels.js:216`) rather than pulling in a charting library. Five small
  line charts do not justify a dependency.
- Lifespan panel: median polyline plus a filled min–max band at ~15% alpha,
  same as the Python `fill_between`.
- Death panel: four polylines, colored by the renderer's existing death-cause
  colors (`#legend-deaths` already defines them — reuse, do not pick new ones).
- Axes: no full axis framework. Y-max label, current-value label at the right
  edge, and a step-range label under the bottom panel. The Python figure had
  room for real axes; a 260px panel does not, and unreadable tick labels are
  worse than none.
- Y scaling: per-panel autoscale to the max over the window, with a floor so an
  all-zero series doesn't produce a divide-by-zero or a full-height line.

## Step 5 — Summary + death tables

Below the graphs, two small DOM tables matching the Python ones: start / current
/ peak population and steps elapsed; then final counts per death cause from the
cumulative tally. Plain HTML, updated at the same 1 Hz.

## Step 6 — Headless dump (optional, cheap)

`--dump-stats out.csv` on the native runner, writing the same samples as CSV.
Makes the golden harness able to diff population curves between builds
numerically instead of by eye.

---

## Order

1 (stats.rs + tests) → 2 (World wiring) → 3 (export) → 4 (graphs.js, population
panel only) → 4b (remaining four panels) → 5 (tables) → 6 (CSV dump).

Stop after 4 and look at it before building the rest. If a 260px live population
curve doesn't read well, the answer is fewer panels drawn bigger, not five
cramped ones.

## Tests

- `stats.rs`: ring wraps correctly, `iter_chrono` returns oldest→newest across
  the wrap boundary, per-interval death differencing sums to the cumulative
  tally over a full run.
- `world.rs`: after N steps, `stats_history` has `N / SAMPLE_INTERVAL` samples
  and the last sample's `alive` equals `agent_count()`.
