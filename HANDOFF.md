# HANDOFF — 2026-08-01 → for next time

Written against commit `992c8ee`, which is `main`, `origin/main` and the tag
`v1.0.1`. Everything below is committed and pushed; `dist/pond_spawn_itch.zip`
is built from that commit.

The renderer got faster, the sprite atlas turned out to work after all, and the
Firefox-vs-Chromium gap turned out to be a different thing than anyone thought.
**There is one decision already made and deliberately not implemented — see
"Do this first".**

---

## Do this first: default the sprite atlas on

**Decided, not done.** `sprites_enabled` in `renderer.js` is `false`; it should
be `true`. Measured on the itch build in Edge, grid 64, 2,964 agents, fit zoom:

```
agents  9.7 ms / 2,963 = 3.27 µs/agent      (vector path: ~9.4 µs)
sprite  on  drawn 2963/2963  atlas 68/351  wipes 0
frame   14.3   70 fps
```

**2.9× on the dominant cost, zero wipes, whole population on the sprite path.**

Two things to handle while doing it:

1. **Nudge `SPRITE_LOD_MAX_SCALE_PX` from 20 to ~22.** At fit zoom `scale_px` is
   `min(canvasW, canvasH) / GRID` and nothing else, so on a 1072×1788 itch embed
   grid 64 gives 16.8 (sprites on) while a 2552×1308 desktop gives 20.4 (sprites
   off). Same build, same grid, different machine, different renderer. That is
   fragile and it will produce bug reports nobody can reproduce.
   **Look at it before keeping it** — raising this threshold is what made the
   pond read as stiff on 2026-07-31, though that was at 39.9 px/tile with bodies
   twice this size. At 16.8 px/tile bodies are ~11 px long and the frozen pose
   was not noticeable.
2. **Headroom is 22%** — peak 351 against a 448 cap. A more speciated pond, or
   the archetype overlay's second palette, could cross it. That is no longer a
   visual defect (wipes are deferred to a frame boundary and overflow falls back
   to the vector path) but it would show as a periodic frame-time step. If it
   does, the next cardinality cut is the silhouette buckets — `headPointiness`
   and `armorBumps` are 4 steps each in `spriteKey` and are the widest remaining
   fields.

### And correct the record while you are in there

`atlas.js` and `PERF_PROBLEM.md` both say the atlas **"does not scale to a
diverse pond"** and explain at length why it can never work under Canvas2D.
That was written from a grid-128 test showing 282 wipes on a paused pond. **It
is true at 128 and wrong at 64**, which is the shipped ceiling. The commit
`28585cc` message says the same thing and cannot be edited, so the code comment
and the doc are the only places to fix it.

What actually happened: the fix in `cf98188` (quantise the energy dim *before*
computing body colour, pinning a species to 4 colours instead of ~16) worked —
it just could not close a 4× larger pond's key space. Capping the grid at 64 did
not sidestep the atlas problem, it put the atlas inside its budget.

---

## Firefox and Chromium have different cost models

This is the most useful thing learned today and it invalidates two earlier
conclusions, including one written into `PERF_PROBLEM.md`.

### The measurements

Same pond, same seed, same zoom, matched population, `M` open.

**Before the terrain caching — grid 32, 39.7 px/tile, ~250 agents:**

| | Edge | Firefox |
|---|---|---|
| sim | 0.5 | 0.6 |
| water | 0.1 | **4.6** |
| shimmer | 0.5 | **3.6** |
| agents | 2.0 | 12.0 |
| frame | 2.9 (351 fps) | 20.6 (39 fps) |
| **µs/agent** | **8.1** | **46.3** |

**After it — grid 64, 19.8 px/tile, ~2,200 agents:**

| | Edge | Firefox |
|---|---|---|
| sim | 2.3 | 5.7 |
| water | 0.1 | 0.1 |
| shimmer | 0.4 | 0.6 |
| agents | 20.7 | 22.3 |
| frame | ~24.5 (40 fps) | 29.9 (33 fps) |
| **µs/agent** | **9.4** | **10.1** |

### What that says

Between those two runs body area fell ~4× (39.7 → 19.8 px/tile). Firefox's
per-agent cost fell **4.6×**, tracking area almost exactly. Chromium's did not
move.

- **Chromium is per-call bound.** Cost is issuing `fill()`; area barely matters.
  This is why the grid-24-vs-64 comparison earlier in `PERF_PROBLEM.md` found
  7× the pixel coverage for 11% more cost.
- **Firefox is per-pixel bound.** Cost tracks the area actually rasterised.

**Two conclusions to throw out.** First, "Firefox is ~5.7× slower on this
renderer, permanently, and there is nothing to do about it" — wrong; at small
body sizes the two are 7% apart. Second, and more importantly:

### The superlinearity may be real after all, in Firefox only

`PERF_PROBLEM.md` records per-agent cost climbing 8.2 → 47.3 µs between 1,100
and 5,000 agents, and then records it being written off twice — first as the
removed 900-agent population cap, then as a Firefox reading sitting beside
Chromium ones. **Both of those were mine and both were probably wrong.**

If Firefox pays per rasterised pixel, then the additive glow matters: `lighter`
cannot early-out on overlap, so in a crowd each agent's glow hull is rasterised
over its neighbours'. Per-agent cost then *rises with density*, which is exactly
superlinear in population at fixed pond size. That fits every Firefox number on
record and contradicts none of them.

**Do not accept this without testing it.** The test is cheap and specific: fix
the agent count and the zoom, then compare a crowd packed into one corner
against the same agents spread out. If Firefox's per-agent cost moves and
Chromium's does not, it is overdraw. Anything less than that and it stays a
hypothesis — this particular question has now been answered wrongly three times.

---

## Next lever: collapse the glow into one layer

Every agent costs **two** hull fills, glow and core. Since the two-pass batching
landed, the glow is already drawn as one shared haze behind all bodies — so it
does not need to be per-agent geometry at all. Accumulate all glows into one
low-resolution offscreen canvas, blur it once, composite it once: **N fills → 1**.

Worth doing for two independent reasons:

- **Chromium:** ~50% fewer draw calls, which is the cost that binds there.
- **Firefox:** it collapses N overlapping additive fills into one composite,
  which attacks the density term directly — if the overdraw hypothesis holds,
  this is the fix for it.

No visual quantisation, no cardinality, no keying. Roughly a tenth the
complexity of the sprite atlas. This is the one to build next.

---

## Also open

- **`other` is 2.4 ms**, 17% of the frame, and unattributed — `get_state`, the
  dying layer, god effects, and whatever rasterisation the browser defers past
  the last measured span. Nobody has looked at it. It is now bigger than water,
  shimmer and food combined.
- **`sim` is 2.5× slower in Firefox** (5.7 ms vs 2.3 at 2,200 agents), which is
  SpiderMonkey vs V8 on wasm rather than anything in the engine. It was
  negligible when drawing dominated; at 19% of a Firefox frame it no longer is.
- **The atlas cannot serve grid 128+** without a different approach to colour.
  Keying on colour is what blows the working set up; the fix is shape-only
  sprites tinted at blit time, which Canvas2D can only do through per-agent
  composite switches — the exact cost the atlas exists to avoid. In WebGL2 the
  tint is a uniform and the problem disappears. That is the real argument for
  WebGL2, not raw throughput.
- **No population cap, deliberately.** `PREDATOR_POP_CEILING` is gone and is not
  coming back; carrying capacity is `1.75 × tiles` with nothing on top. A boomed
  pond zoomed out is allowed to crawl. Do not propose a cap; the owner's position
  is that letting the user melt their machine is what this genre does.

## What landed today

All on `main`, tagged `v1.0.1`, 197 tests green, no new clippy warnings.

| commit | what |
|---|---|
| `a5e53e3` | timing HUD (`M`), ornament gating, two-pass batching, substep cap, grid ceiling to 512, `PREDATOR_POP_CEILING` removed |
| `f6246f1` | atlas no longer wiped mid-frame — agents were phasing in and out, *including while paused*, which is what identified it |
| `112fb0c` | viewport culling |
| `cf98188` | atlas cardinality: species pinned to 4 colours, strategy 16 → 4 buckets |
| `28585cc` | grid ceiling back to 64, atlas disabled by default |
| `68e8cdd` | opens on seed 21 |
| `5ad6ddc` | docs describe the keys that exist |
| `2cff1e0` | water rebuilt on sim step, shimmer at 30 Hz |
| `992c8ee` | substep cap scales with the speed dial |

Two of those undo parts of `a5e53e3` — the 512 ceiling in particular. Left
unsquashed on purpose.

### Method note, worth keeping

Three separate things today were diagnosed from the *paused* pond: the mid-frame
atlas wipe (a frozen population still phased, so it had to be draw-loop driven),
the atlas cardinality overflow (a frozen population cannot have a changing
working set, so the set was simply too wide — which also ruled out LRU eviction,
since every key is hot every frame), and the confirmation that neither was a sim
problem. **Pause it first.** It removes every moving variable at once and it
found more today than any profiler did.

---

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
