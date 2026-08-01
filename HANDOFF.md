# HANDOFF — 2026-07-30 → for 2026-07-31

Written against commit `f2a37f0` plus the uncommitted changes below. Two things
happened today: the grid ceiling went up, and the renderer was profiled because
the itch build is choppy. **Tomorrow is renderer work.** No engine change is
needed — the engine is not the problem, and there are numbers below saying so.

## Done today (uncommitted)

Grid is now sizeable up to **512×512**. `pond_core` needed nothing — it never had
a grid bound. The 64 ceiling was in the web setup panel, and two render passes
scaled with GRID².

- `pond_web/setup.js` — `LIMITS.grid.max` 64 → 512. New `HEAVY_GRID = 160`
  warning: says the cost is drawing, not simulating.
- `pond_web/renderer.js` — `draw_water` capped its mid canvas at
  `WATER_MID_MAX_PX = 1024` (at 8 px/tile a 512 pond asked for a 4096² canvas
  and blurred it every frame). Blur radius now scales with the derived
  px-per-tile so softening stays constant in tile space; grids ≤128 are byte-for-
  byte unchanged. `draw_food` culls to the visible tile rect (it looped all
  GRID² tiles and drew up to 3 orbs each — 262k iterations at 512).
- `RULES.md` — World section records the 6–512 range and that the ceiling is a
  rendering budget, not a rule.

Verified: `cargo run --release -p pond_core --bin run --features native -- 512
5000 60 42` → **6.5 ms/step**, 390 ms total. Not looked at in a browser (no
browser in this container — same constraint as the 2026-07-29 note below).

**Caveat worth acting on separately:** `PREDATOR_POP_CEILING` (900) caps
`pop_cap()` regardless of area, so above ~23×23 every pond has the same carrying
capacity. A 512×512 pond buys space, not animals. Untouched — it's an economic
lever and wants its own change with its own hypothesis.

## The chop: investigated, not fixed

Symptom: 64×64, 2,000+ agents, zoomed out, on itch — real choppy.

### The engine is not it

| measurement | value |
|---|---|
| sim step, 64×64 / 2,200 agents, native | **1.5–2.0 ms** |
| sim tick rate | 20 Hz (so most 60 fps frames step 0 or 1 times) |
| brain k-means, one cluster step @ 1,200 agents | **11.0 ms** native (`--bench-cluster`) |

Even at a 2–3× wasm penalty the steady-state sim is a couple of ms per frame
amortized. The **cluster pass is a separate, periodic hitch**: k=24 over 488
dims scales ~linearly with population, so ~18–20 ms native at 2,200 agents,
×2–3 in wasm, landing on a handful of steps every 50 ticks (~2.5 s). That's the
lurch, not the chop.

### The renderer is it

Ran the real `morphology.js` / `chain.js` / `body.js` against 2,000 randomized
genomes with a counting canvas stub (mean segCount 7.2):

```
agents=2000 grid=64 scale_px=16.9
quadraticCurveTo  55,594    fill                          13,497
lineTo            24,392    set:fillStyle                 13,497
beginPath         21,756    stroke                         8,259
moveTo            15,030    arc                            6,725
closePath          9,362    set:globalCompositeOperation   4,135
save/restore    2,135 ea    TOTAL 182,199  (91.1 ops/agent)
```

Geometry JS alone, rasterization excluded: **~4 ms/frame** on V8. That's the
floor. The bill is the 13,497 fills and 8,259 strokes — at 3–10 µs of Canvas2D
per-draw overhead for a small path, **40–130 ms/frame for the agent layer
alone**.

**Why zoomed out specifically.** `draw_agents` has no culling, so JS geometry
cost is zoom-independent — but rasterization isn't. Zoomed in, the GPU trivially
rejects paths outside the clip rect. At `MIN_ZOOM` all 2,000 bodies land inside a
~1080² box, every one rasterizes, and every glow hull is an additive (`lighter`)
fill, which cannot early-out on overlap. Zoomed out is the worst case for the one
part of the cost that zoom controls.

### Ranked causes

1. **No LOD.** A 5-px agent gets the full pipeline: 2 hull paths (~2×segCount
   quadratic curves), 2 eye arcs, up to 6 fin triangles, 3 armour rings, 6 spike
   strokes. At second-screen size none of the ornaments resolve — they cost about
   half the ops and render as noise.
2. **State thrash.** 4,135 `globalCompositeOperation` switches and 2,135
   save/restore per frame; every composite change flushes the batch.
3. **13,497 `fillStyle` assignments/frame**, each a freshly built `rgba(...)`
   template string the browser must CSS-parse.
4. **GC pressure.** `drawBody` allocates 7 arrays of `{x,y}` objects per agent
   (`world`, `centers`, `dirs`, `left`, `right`, `glowLeft`, `glowRight`) ≈ 51
   objects/agent → **~100k objects/frame**, ~6M/s at 60 fps. This is what makes
   it *choppy* rather than uniformly slow.
5. **`ctx.filter = blur(...)` on the water mid-canvas every frame** — expensive
   in Chrome, often a readback, redrawn at 60 Hz for terrain that changes at 20.
6. **`draw_food` over all GRID² tiles** in the shipped itch build — 4,096 tiles,
   up to ~3,500 additive `drawImage` calls on a fed pond. Fixed locally today,
   not deployed.
7. **Slow-frame amplifier.** `frame_delta` is capped at 200 ms then multiplied by
   `speed_mult` before `world.update` — a 200 ms frame runs 4 sim steps inside
   it, and 64 at `speed ×16`. A late frame gets the most work.
8. **`get_state` copy**: 7 + 2000×19 + 4096×3 ≈ 50k floats ≈ 200 KB/frame, fresh
   `Float32Array` every frame. Minor here, listed for completeness.

## Tomorrow, in order

Do these one at a time and re-measure between — they overlap, and stacking them
makes it impossible to say which one paid.

1. **Instrument first.** Four `performance.now()` spans around `draw_water` /
   `draw_shimmer` / `draw_food` / `draw_agents`, printed to the HUD behind a
   keybind. Everything below is inferred from op counts; this converts it to
   fact, and it's the acceptance test for the rest. If `draw_agents` is not the
   dominant span, stop and re-read.
2. **Sprite LOD.** Below ~3 px `baseR` (`scale_px` under ~20), blit one
   pre-rendered sprite from an atlas keyed by (plan, segCount bucket, hue bucket,
   energy bucket) instead of running the hull pipeline. 91 ops → 1 `drawImage`.
   Targets exactly the zoomed-out case; expect an order of magnitude. Biggest
   single win, and the only one that needs a design opinion — get the bucket
   count right or lineage colour goes coarse on screen.
3. **Ornament budget**, if LOD lands short or as a cheaper first cut: skip
   fins/spikes/armour/eyes when their pixel size is under ~1.5 px. ~50% of ops,
   no visible difference.
4. **Two-pass batching**: all glow hulls in one `lighter` pass, then all cores in
   `source-over`. Drops 4,135 composite switches to 2. Note it changes layering —
   all glows behind all bodies — which is a look change, arguably a better one,
   so it needs eyes on it before it stays.
5. **Cache colour strings** keyed by rgb + quantized alpha.
6. **Scratch geometry buffers** — one reusable `Float32Array` per max segCount
   instead of per-agent object arrays. Kills the GC sawtooth.
7. **Viewport-cull `draw_agents`** with a seam margin. Doesn't help zoomed out;
   helps everything else.
8. **Cache the blurred water canvas**, invalidate on step change not frame.
9. **Cluster hitch**, separate from the above: amortize the brain k-means over
   more steps, or subsample to a fixed ~600-agent sample above that population.
   The labels are a reading aid, not a mechanic.

### Reproducing the measurements

- Sim: `cargo run --release -p pond_core --bin run --features native -- 64 2200 400 7`
- Cluster: same binary with `--bench-cluster`
- Canvas op counts: a ~40-line node ESM harness that imports `morphology.js`,
  `chain.js`, `body.js` (copy them to `.mjs` — the repo `package.json` has no
  `"type": "module"`, so node treats `.js` as CJS), builds 2,000 random morph
  knob sets, and calls `drawBody` with a stub ctx whose methods tally calls. Use
  a plain object stub, not a Proxy — a Proxy triples the reported JS time. It
  lived in the session scratchpad and is gone; it's quicker to rewrite than to
  find.
- A Chrome performance profile on the itch build would settle the whole split in
  one capture, and is worth doing on any machine with a browser before item 2.

---

# HANDOFF — 2026-07-29

State of the project after an economy + clustering review pass. Written against
commit `fab393b` plus the uncommitted changes described below.

## Done this pass

Uncommitted, all tests green: **137 pass, up from 134** (three new tests, verified
by stashing the source changes and re-counting). No new clippy warnings.

### 1. Sleep was a net energy source, and uncapped (`world.rs`) — fixed

`SLEEP_RECOVERY = 0.05` replaces a hardcoded `+= 0.15 * metabolism`, and the
result is clamped to `MAX_ENERGY_BASE × energy_capacity`.

The old value exceeded the `0.1 × metabolism` base drain, so an agent whose sleep
gate kept winning netted `+0.05/tick` forever — and with no capacity clamp it
climbed past its own maximum indefinitely. The overflow was invisible because
`energy_norm` clamps to 1.0 in perception, so neither the brain nor the HUD ever
showed it; the only symptom was agents that could not starve. `RULES.md`
documented the 0.15 gain and the code comment claimed it was matching the spec,
but `REFACTOR_RUST_ROADMAP.md` recorded the actual refactor decision as
`-0.05 × metabolism`, "rest, not recovery". The roadmap was right and RULES.md is
now corrected.

**The measured effect was noise-level**, which is worth knowing:

| seed | final pop before → after | mean avg_energy before → after |
|---|---|---|
| 42 | 23 → 39 | 60.2 → 61.0 |
| 7 | 74 → 78 | 48.1 → 49.0 |
| 1337 | 125 → 15 | 58.7 → 56.9 |

(12×12, 100 agents, 2000 steps.) The spread across seeds is larger than the
change, so this was a correctness fix, not a rebalance — the sleep gate is
evidently not the winning gate often enough to matter. Do not credit any
population change to it.

New tests: `sleep_slows_starvation_instead_of_reversing_it`,
`energy_never_exceeds_capacity`.

### 2. Genome clustering now shares the species signature (`cluster.rs`) — fixed

`trait_vec` delegates to `species::signature`: seven mutable traits, each
normalized to `[0,1]` by its bounds. `TRAIT_DIMS` is `species::SIG_LEN`.

Two problems, both about *what* was being clustered:

- **Unnormalized.** Raw values went into euclidean distance while bounds differ
  by an order of magnitude, so `reproduction_cost` (0.75–1.50) outweighed
  `defense` (0.5–1.07) several times over and `mutation_rate` (0.01–0.25) barely
  registered.
- **Included the locked traits.** `energy_capacity` and `mutation_rate` are never
  mutated and inherited exactly (D3) — perfect founder tags with zero selection
  pressure. `species.rs` excludes them for exactly this reason, so cluster colors
  on screen were part lineage-of-origin while the species assigned next to them
  was pure evolved shape. Two readings of "family" that didn't agree; now one.

New test: `clustering_ignores_locked_traits_and_normalizes_the_rest`.

This changes the RNG stream, because cluster membership feeds the species
probation clamp which feeds the mutation draws. At 12×12/100 agents seed 42 now
reaches probation at step 1050 instead of 1900 — the better-conditioned partition
finds stable structure sooner in a thin population. The 150-agent speciation
table below is unchanged seed for seed. Same-seed runs remain bit-identical to
each other; they are just not comparable to pre-change logs.

### 3. Doc pass

- `RULES.md` — sleep spec corrected, plus a section on why. The three legacy
  Python sections (action table, decision overrides, chosen ATTACK at index 7)
  are now explicitly tagged as not running in `pond_core`; a reader was
  previously handed a fully specified combat mechanic that never fires.
- `REFACTOR_RUST_ROADMAP.md` — phase table corrected (7 and 8 are done, 9 is
  next), module table filled in with the five modules built since it was
  written, test count 31 → 137, pre-Phase-8 cluster perf numbers labelled as
  historical, and the "the renderer doesn't exist yet" Phase 6 TODO replaced with
  what shipped.
- `PLAN_SPECIATION.md` — Step 5 tuning pass recorded with data (below).

### 4. Speciation tuning pass — measured, thresholds left alone

Six seeds, 12×12, 150 agents, 3000 steps: 1–3 live species per seed against a
2–5 target, first promotion between steps 1600 and 2750 against a target window
of 300–800.

**The target window was arithmetically impossible, not the thresholds.**
`mean_generation` reaches 4.0 only around step 750 and advances ~1 per 250 steps
after. Promotion requires 3 generations of advance to enter probation plus 1 more
under the clamp — four generations from when a stability streak begins, and the
streak cannot begin before a cluster holds still. Nothing could promote before
~step 1000 even in the best case.

Recommendation recorded in the plan: revise the expectation to ~1600–2800 rather
than loosen criteria. If the window genuinely needs shortening, the lever is a
faster-turning population (`MATURITY_AGE`), or `PROBATION_ENTRY_GENERATIONS`
3 → 2 — not `MIN_MEMBERS_*`, which is not what gates here.

---

## What the baseline actually says about the economy

This changes the priority order from what I first assumed. Across seeds at
12×12/100 agents/2000 steps:

```
KilledInCombat: 387–465     OldAge: 12–35     Starvation: 20–39
```

**Combat is ~90% of all deaths.** Starvation is a rounding error, and total food
sits at 0–5 units for long stretches in some seeds — the pond is food-poor and
still barely starves anyone, because agents are killed before they can starve.
Population runs 15–146 with a peak around 120–134 near step 200, then decays.

So the pond is not "generous" via food. Any food-regen tuning would be measured
against a system whose mortality is dominated by predation, and would move the
wrong number. The open economy question is whether `PREDATION_YIELD` (0.667) plus
survivable retaliation has made hunting so profitable that combat crowds out
every other cause of death. Both numbers are deliberate corrections for
aggression being selected to extinction, and both are documented at length — so
this is a real design question, not a bug, and it needs a decision before a
change: **should starvation be a visible cause of death in a healthy run?**

If yes, the single lever to try first is `PREDATION_YIELD`, down from 0.667
toward ~0.4, hypothesis: combat share of deaths falls, starvation share rises,
mean lifespan rises slightly, population peak roughly unchanged. One lever, one
run of the seed table above.

---

## Decided since (2026-07-30)

- **Combat share is not a defect.** A mortality profile dominated by predation
  does not make the ecosystem unhealthy — crabs in a bucket prosper. There is no
  "healthy" distribution of causes of death to aim at, so `PREDATION_YIELD`
  stays at 0.667 and the question above is closed, not deferred.
- **Food regen is fine as it stands.** Not a rebalance target; it becomes a dial
  instead (below).
- **Empty-cluster handling — fixed.** `cluster.rs` now reseeds an empty cluster
  onto the point furthest from its own centroid, via `reseed_empty`, matching
  `brain_cluster::update_centroids`. The two implementations agree. New test
  `empty_clusters_are_reseeded_not_left_stale`; **138 pass**, no new clippy
  warnings. This can move labels in runs where a cluster empties, so logs from
  before it are not comparable — same-seed runs remain identical to each other.
- **Three dials shipped** — food regen scale, the hunt aggression threshold and
  clustering `k`, live, with hover blurbs, behind `T` in the web build. Plan and
  what it got wrong: [PLAN_TUNABLES.md](PLAN_TUNABLES.md). Building it exposed a
  real bug in `match_labels`: lowering `k` handed out labels above the new `k`,
  which indexed out of bounds in three places. Fixed and regression-tested.
  **146 pass.**

## Still open

- **Species structure unverified** — now testable directly, since `k` is a dial:
  sweep it and see whether the composite spreads move.
- **`intelligence` trait** — still commented out in `genome.rs:18`, `:53`, `:85`.
  Oldest TODO in the project (DEVLOG, Dec 23 2025); the intent was for it to
  modulate how often an agent re-evaluates rather than to add an action.
- **Phase 9 — itch.io** — packaging, CSP/iframe check on the wasm fetch path,
  page copy. Not started; needs you, since it publishes.

## Notes

- **The tuning panel has never been seen in a browser.** Headless Chromium will
  not start in this container (`libnspr4.so` missing, no sudo), so the panel was
  exercised under jsdom against a stub engine instead: rows, blurbs, clamping,
  reset and the reproducibility note all behave. Layout and the blurb popover
  are unverified. Look at it on a machine with a browser before it ships.
- `pond_core/pkg/` was rebuilt (`wasm-pack build pond_core --target web
  --features wasm`) so the browser build includes the clustering change. That
  directory is gitignored.
- Baseline and post-fix run logs are in the session scratchpad
  (`base_*.txt`, `sleepfix_*.txt`, `spec_*.txt`, `stats42.csv`); regenerate with
  `./target/release/run 12 100 2000 <seed>`.
