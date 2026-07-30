# Plan — Brain Cluster View (and the 50-step stutter)

Goal: make behavioural archetypes visible, and stop the cluster tick from
dropping frames while doing it.

These are one plan rather than two because they constrain each other. The
brain clustering is currently ~99.5% of a 14–164 ms spike every 50 steps
(`REFACTOR_RUST_ROADMAP.md`, measured native release), and today nothing reads
the result — `brain_cluster_ids` crosses the wasm boundary, gets decoded into
`brainCluster` at `decode.js:36`, and no renderer code touches that field. The
most expensive operation in the simulation currently feeds nothing.

The cheap fix for the stutter was therefore "stop computing it". Once there is a
view, that answer is off the table, and the cost has to actually come down.

---

## What the view is for

Genome clusters answer *who is related*. Species answer *which lineages locked
in*. Brain clusters answer a different question — **who behaves alike** — and it
is the one the sim exists to ask. Two things are worth seeing:

1. **Behavioural archetypes in the pond.** Which strategies exist right now, and
   how big each one is.
2. **Archetype against lineage.** Whether a species converged on one strategy or
   hedges across several, and whether unrelated lineages converged on the *same*
   strategy. Convergent evolution across genomes is the most interesting thing
   this simulation can show, and right now it is invisible.

(2) is the payoff. (1) is table stakes and mostly exists already in the data.

---

## Order

Perf first, view second. Building the view on the current implementation would
just make the stutter something the user can summon on demand, and tuning a
visualization while its input costs 80 ms per refresh is miserable.

```
1 warm-start  →  2 amortize  →  3 scratch buffers  →  4 gate  →  5 view
                 └─ re-measure in the browser after 2 ─┘
```

---

## Step 1 — Warm-start the brain k-means

The single biggest win, and it improves quality at the same time.

`ClusterState` retains `genome_centroids` across runs for label matching but
retains **nothing** for the brain pass, so every run re-runs k-means++ from
scratch: 24 sequential passes over every agent's 488-dim vector purely to pick
starting points. That init is 42% of the cost, measured, at every population.

Retain `brain_centroids: Vec<Option<Vec<f32>>>` and seed the next run from them.
Consequences:

- k-means++ init drops out of the steady-state path entirely. It stays as the
  cold-start path (first run, and reseeding any cluster that goes empty).
- Iterations can fall from 15 to ~3. Centroids start near-converged, because the
  population changed by a few births and deaths, not wholesale.
- Label stability improves for free. `match_labels` currently exists to stop
  colour flashing when labels permute between runs; warm-started centroids
  mostly do not permute in the first place. Do **not** delete `match_labels` in
  this commit — it is still needed for cold start and for reseeded clusters, and
  removing it here would confound two changes.

Estimated ~7–8× on the brain pass: at n=600, 82 ms → ~10 ms.

**Verification:** cluster assignments must not become unstable. Compare
warm-started vs cold-started label distributions over a 3000-step run at several
seeds; the partition should agree closely, and if it does not, the warm start is
getting stuck in a worse local optimum and the iteration count is too low.

## Step 2 — Amortize across steps

With warm-start, k-means is naturally incremental: there is no init phase that
has to complete before iteration one. Run **one iteration per step across three
consecutive steps** rather than three iterations in one tick.

At n=600 that turns a ~10 ms spike into ~3.5 ms on each of three steps — inside
the frame budget. At n=1200, ~7 ms per step for three steps, still survivable.

Requires a snapshot of the normalized point set at the start of the window, so
agents born or killed mid-window cannot corrupt the partition. That snapshot is
also the natural home for Step 3's scratch buffer.

**Re-measure in the browser here.** Steps 1–2 should be enough on their own; if
the hitch is gone at realistic populations, Steps 3 and 4 are optimizations
rather than fixes, and the view can start.

## Step 3 — Kill the per-run allocation churn

`points: Vec<Vec<f32>>` allocates one 488-float `Vec` per agent per run, and the
update loop reallocates `sums` (24 × 488 floats) on every iteration. Replace
with flat scratch buffers owned by `ClusterState`, matching the scratch-buffer
approach `World` already uses.

Worth an estimated 10–20%. Real, but it is not the fix, and doing it before
Steps 1–2 would be optimizing code that is about to be restructured.

## Step 4 — Gate it on the view

Even cheap, brain clustering should not run when nobody is looking at it.

```rust
pub fn set_brain_clustering(&mut self, on: bool);
```

Off by default; the renderer turns it on when the view is toggled and off when
dismissed. When off the pass is skipped entirely and `brain_cluster_ids` is
empty — consumers already index defensively (`wasm.rs:160` uses
`.get(i).copied().unwrap_or(0)`).

Pay for what you look at. This also means the itch.io build costs nothing for
visitors who never open the panel.

---

## Step 5 — The view

### Palette

k = 24 is too many to colour distinguishably; 24 hues sit ~15° apart and read as
noise. **Colour the N largest clusters (N ≈ 8) and render the tail in neutral
grey**, with the legend showing the tail's total size. Aggregating in the UI
keeps `k = 24` intact in the engine — the underlying granularity is worth
having, and changing `k` to suit a palette would be letting the view dictate the
data.

Hues by golden angle (`id × 137.5° mod 360`) rather than a hash: maximal
separation for any count, deterministic, stable per cluster id.

**This is a toggled overlay, not a change to agent colour.** Default agent
colour stays trait-derived through the three ramps (`color.js:55` documents why
that decision was made and why label-keyed colour was abandoned). Toggling the
brain view temporarily recolours agents by archetype; toggling it off restores
the ramps. Same contract as an energy heatmap.

### Panel

Toggle with `B`, mirroring `G` for graphs. Side panel, never over the grid
(see the project rules).

- **Archetype list** — one row per coloured cluster: swatch, member count, share
  of population. Tail row at the bottom in grey.
- **Archetype × species matrix** — the payoff. Rows are live species, columns
  are the coloured archetypes, cells shaded by count. Read across a row to see
  whether a lineage converged or hedges; read down a column to see whether
  unrelated lineages found the same strategy.
  - Needs `A_SPECIES` from the speciation export (Phase 7, step 4) to exist
    first. Until then the matrix rows can fall back to genome cluster.
  - Computed in JS from the per-agent buffer already shipped each frame — no new
    export, no engine cost. Update on the 1 Hz timer, not per frame.
- **Empty state** matters here: before any species is promoted the matrix has no
  rows. Say so in words rather than rendering an empty grid.

### What not to build

No per-archetype behavioural *labels* ("forager", "aggressor"). The clusters are
unlabelled directions in 488-dim weight space; naming them would be asserting an
interpretation the data does not carry. Size, colour, and the cross-tab are
claims that hold.

---

## Tests

- Warm-started clustering produces a partition close to cold-start over a long
  run (agreement metric, several seeds).
- A cluster that goes empty is reseeded rather than left as a dead centroid.
- Amortized clustering produces the same result as the equivalent batched run,
  given no births or deaths in the window.
- `set_brain_clustering(false)` leaves `brain_cluster_ids` empty and consumers
  handle it.
- Determinism holds: same seed, same brain cluster assignments.

## Benchmarks to keep

The measurement that started this should not have to be re-derived. Add a
`--bench-cluster` mode to the headless runner printing the cluster-tick cost at
several populations, so a regression shows up as a number rather than as a
stutter someone notices months later.
