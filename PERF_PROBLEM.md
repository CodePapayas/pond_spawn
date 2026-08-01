# Investigation brief: Canvas2D agent rendering costs ~47 µs/agent and scales superlinearly

**Status:** open. Three hypotheses tested and falsified. Written 2026-07-31.

You are being asked to investigate a rendering performance problem in a browser
artificial-life simulation. Read this whole brief before proposing anything —
the obvious answers have been tried and measured, and the measurements are
below. **Do not propose a fix without saying what measurement would falsify
it.** Three plausible fixes have already produced exactly zero improvement.

A profiler capture now exists and supersedes most of the reasoning here — read
"Profiler capture" before the hypothesis section, which is kept only to record
what has already been ruled out.

---

## The system

`pond_spawn` — a Rust/wasm ecology sim with a hand-written Canvas2D renderer.
Agents are neural-network-driven creatures on a toroidal grid. Population is
unbounded by design (a hard 900-agent cap was removed today, deliberately: the
pond is meant to be self-balancing, limited by food and predation, not by what
the renderer can afford). It has run 20,000+ agents. The renderer must degrade
gracefully at any population, not be sized for one.

- Engine: Rust → wasm (`pond_core/`), fixed 20 Hz sim tick.
- Renderer: vanilla JS Canvas2D, `requestAnimationFrame`, targets 60 fps
  (`pond_web/renderer.js`, `body.js`, `chain.js`, `morphology.js`).
- Ships as a static web build (itch.io). No build step for the JS — plain ES
  modules served directly.
- Canvas is created with `canvas.getContext('2d')`, no options, sized to
  `window.innerWidth/innerHeight` with **no `devicePixelRatio` scaling**
  (`renderer.js:507`). So the backing store is CSS-pixel sized.
  **Correction (2026-07-31): the "roughly 1600×900" above is wrong on the
  development machine — the M HUD reports 2552×1308, which is 3.34 M px, 2.3× the
  assumed area.** Every per-pixel figure in this document derived from the 1.44 M
  px estimate is off by that factor, including the ~122 ns/px arithmetic. The
  conclusion it supports survives (see below) but the number does not.

## How an agent is drawn

Per agent, per frame, in `body.js:drawBody`:

1. Unwrap `segCount` (5–11, mean ~7.2) chain segments across the toroidal seam.
2. Project to screen, compute per-segment forward directions and left/right hull
   offsets.
3. **Glow hull**: one closed path, `moveTo` + ~2×segCount `quadraticCurveTo` +
   2 `lineTo` + `closePath`, scaled outward 1.9×, filled with `globalCompositeOperation
   = 'lighter'` at low alpha.
4. **Core hull**: the same path at 1.0×, filled `source-over`.
5. Ornaments: armour rings (`arc` + `stroke`), head spikes (`moveTo`/`lineTo`/
   `stroke`), fins (3-point filled triangles), two eye `arc` fills.
6. Seam copies: the whole paint is repeated translated when the body straddles a
   grid edge (1–4 copies; the great majority of agents get 1).

Every `fillStyle` is a freshly built `rgba(...)` template string.

## Measurements

Instrumentation is live in the build: press `M` for per-pass `performance.now()`
spans, EMA over ~30 frames (`renderer.js`, search `perf_mark`). All readings
below are **Firefox on Windows**, 64×64 grid, zoomed out to fit the whole pond
(`scale_px` ≈ 17 px/tile, so agent `baseR` ≈ 2.5 px — creatures are ~13 px long
and ~5 px wide on screen).

### The pass split is unambiguous

At 4,995 agents:

```
sim      17.0    (artifact — see note)
water     0.0
shimmer   0.2
food      0.1
agents  236.5    ← 99% of the drawing
other     0.0
frame   238.6    4 fps
```

`draw_agents` is the entire problem. Water, shimmer and food are noise. (`sim`
is inflated because at 4 fps `frame_delta` clamps to 200 ms and the engine runs
4 sim steps inside one frame; true cost is ~4.2 ms/step, ~8% of wall clock at
20 Hz. The engine is not the problem and does not need work.)

### Per-agent cost is superlinear, then invariant to everything tried

| build | agents | `agents` ms | **µs/agent** |
|---|---|---|---|
| baseline | 392 | 3.8 | 9.7 |
| baseline | 1,100 | 9.0 | 8.2 |
| baseline | 4,995 | 236.5 | **47.3** |
| ornament gating | 5,652 | 276.1 | **48.8** |
| + two-pass batching | 5,733 | 271.3 | **47.3** |

Two separate facts in that table:

1. **Superlinearity.** Per-agent cost rises ~6× between 1,100 and 5,000 agents
   while the per-agent work is identical. Something scales with density or with
   total allocation, not with agent count.
2. **Invariance.** Two interventions that removed large amounts of per-agent
   work changed nothing.

### What was tried and falsified

**Hypothesis 1 — op count / no LOD.** Ranked #1 from an offline op-count
harness (2,000 agents → 182,199 canvas ops, 91.1/agent). Gated ornaments off
below pixel thresholds (fins/spikes < 1.5 px, eyes < 0.6 px, armour when
`baseR` < 3 px). At the measured zoom `baseR` ≈ 2.5 px so **all gates fire and
every ornament is skipped** — roughly half the ops per agent.
**Result: 47.3 → 48.8 µs/agent. No effect.**

**Hypothesis 2 — canvas state thrash.** Each agent flipped
`globalCompositeOperation` twice and did one `save`/`restore`; a composite
change flushes the canvas batch. ~11,300 flushes/frame at this population.
Restructured `draw_agents` to collect into a pooled queue then run two batched
passes (all glow hulls under `lighter`, then all cores under `source-over`),
taking composite switches to **2 per frame** and eliminating per-agent
save/restore entirely.
**Result: 48.8 → 47.3 µs/agent. No effect.**

**Hypothesis 3 — the water blur.** `ctx.filter = blur(...)` on a mid-canvas
every frame was ranked #5 as "expensive in Chrome, often a readback".
**Measured at 0.0–0.1 ms. Dead.**

### What both interventions failed to touch

Neither experiment reduced **the two hull path fills**. Ornaments are arcs,
triangles and strokes — the hull is ~14 `quadraticCurveTo` calls per path, two
paths per agent, and both experiments left that completely intact.

Arithmetic: 271.3 ms / 5,733 agents / 2 hull fills ≈ **23.7 µs per hull fill**,
for a path covering ~124 px². At 1,100 agents the same fill costs ~4.1 µs. Same
path complexity, same pixel size, 6× the cost.

## Profiler capture — this supersedes the hypotheses below

A Firefox profile at ~5,700 agents, inverted call stack, self time:

```
52%   2751 samples   CanvasRenderingContext2D.fill
20%   1066           CanvasRenderingContext2D.stroke
3.4%   182           CanvasRenderingContext2D.beginPath
3.3%   177           set CanvasRenderingContext2D.fillStyle
2.8%   150           wasm-function[54]
2.4%   127           paint  (pond_web/body.js:117)
1.9%   100           wasm-function[4]
1.4%    72           set CanvasRenderingContext2D.strokeStyle
1.0%    52           CanvasRenderingContext2D.quadraticCurveTo
```

**What this establishes:**

- **72% of render time is `fill` + `stroke`.** The cost is in issuing those
  calls, not in building the paths. The fix is to stop making them.
- **All JS geometry is 2.4%** (`paint` is the whole of `drawBody`'s per-agent
  computation — 7 arrays, ~51 objects/agent). **The GC hypothesis is dead and
  scratch-buffer work is cancelled.**
- **Path construction is ~4.4%** (`beginPath` + `quadraticCurveTo`). Note this
  does *not* clear tessellation: curve flattening happens inside `fill()`, so it
  is hidden in the 52%. Tessellation vs rasterization within `fill` remains
  unseparated.
- **Colour string parsing is 4.7%** (`set fillStyle` + `set strokeStyle`). Real
  but minor — a colour cache is worth ~5%, not more.
- **wasm is 4.7%.** The engine is confirmed cheap for the third time.
- **`stroke` at 20% was one ungated call.** Armour and spikes are size-gated at
  this zoom, leaving only the unassigned-lineage hull outline — one per agent on
  a pond where nearly every agent is unassigned. Now gated (`OUTLINE_MIN_BODY_PX`).
  Call-count arithmetic corroborates that the gates are live: gates firing
  predicts a 2:1 fill:stroke ratio, and the observed sample ratio is 2.58:1.

**Per-call overhead, not pixel count.** ~11,500 fills producing ~1.15M px of
coverage on a ~1.44M px canvas cost ~141 ms. That is ~122 ns per *pixel*, which
is ~100× slower than even software rasterization should manage — so the cost is
a fixed per-`fill()`-call overhead, roughly **12 µs a call**, largely independent
of the area covered. This is the single most important number in this document
and it is what makes a sprite atlas the indicated fix: `drawImage` does not pay
it.

**Still unexplained: the superlinearity.** Nothing in this profile accounts for
per-agent cost rising 8.2 → 47.3 µs between 1,100 and 5,000 agents. If per-call
overhead were a constant ~12 µs, cost would be linear. Candidates: additive
overdraw growing with density (`lighter` cannot early-out on overlap), or a
rasterizer that degrades as dirty regions merge. **This is still the open
question**, and the readings at 392 and 1,100 agents may be confounded by a
different zoom level (and therefore a different fill area per agent) — that
should be re-measured at a fixed zoom before anyone theorises on it.

## Cost is per call, not per pixel — now measured directly

2026-07-31, via the `M` HUD on the dev machine (2552×1308 canvas), both readings
at fit zoom with sprites not yet engaging, so both are pure vector baseline:

| grid | px/tile | agents | `agents` ms | **µs/agent** |
|---|---|---|---|---|
| 64 | 20.4 | 5,666 | 55.6 | **9.8** |
| 24 | 54.5 | 5,217 | 46.0 | **8.8** |

Bodies at grid 24 are **2.7× longer**, roughly **7× the pixel coverage**, with
identical path complexity — and per-agent cost differs by **11%**. Whatever the
per-pixel rate is, it is not what sets the bill. This is the single cleanest
confirmation that the fix is to issue fewer draw calls, and it was available from
two keypresses.

**It also fails to reproduce the 47.3 µs/agent reading** in the table above: the
same populations cost ~9 µs here, in both grids. Until that is seen again on a
known browser and canvas size, treat the superlinearity as *unconfirmed as well
as unexplained* — the candidate confounds are browser (that capture was Firefox),
canvas size, and the since-removed 900-agent cap.

**Consequence for the design, and the open tension:** a zoom-keyed LOD only ever
helps ponds zoomed far out, but ~9 µs/agent at every body size says sprites would
pay off at every zoom. Acting on that — threshold 56, curvature buckets — made
the pond look stiff and was reverted (below). So the perf argument for wide
sprite coverage stands and the visual one against it also stands, and they are
not yet reconciled. The threshold is back at 20, where on most screens sprites
engage only for large grids.

**A note on the threshold, since it cost a debugging round:** at full zoom-out
`scale_px` is `min(canvasW, canvasH) / GRID` and nothing else, so **grid size
alone decides whether a run can ever reach the LOD threshold**. A 24-tile pond on
a 1308 px-tall window sits at 54.5 px/tile with the camera as far out as it goes.
The `M` HUD now prints grid, canvas, and that floor with a `← never reaches lod`
marker, so this is visible rather than inferred.

Growing the grid with population was considered and rejected: `pop_cap` is
`density × area`, so it would be a positive feedback loop with no stop.

## Built, unmeasured: the sprite atlas (`pond_web/atlas.js`)

Written 2026-07-31 against the profile above, on the strength of one number:
**~12 µs of fixed overhead per `fill()` call, largely independent of area.**
`drawImage` does not pay it. Everything here follows from that and nothing here
has been seen in a browser — this container has none.

**Shape.** `atlas.js` bakes a body once into a 2048² offscreen canvas and blits
it. Per agent per frame the crowd goes from ~2 hull fills plus ornaments to
**one `setTransform` + one `drawImage` per pass**, two passes.

- **Two sprites per key, not one.** The glow is baked separately and drawn under
  `lighter`; the core under `source-over`. Baking the glow flat into the body
  would have saved a blit but lost the additive haze the pond's look rests on,
  and the constraint list says visual identity matters. Composite switches stay
  at 2 per frame.
- **Canonical pose, rotated at blit time.** Sprites face +x with the head at the
  rotation pivot; heading comes from velocity, or from the chain's own spine when
  an agent is still. `setTransform` is one call and does not flush the batch, so
  rotation is not baked into the key — 16 angle buckets would have multiplied the
  atlas by 16 to save a call that costs nothing.
- **LOD is global, not per agent** (`SPRITE_LOD_MAX_SCALE_PX = 20`). Mixed
  atlas and vector bodies at the same on-screen size read as some of the animals
  having died.
- **Key is one 29-bit int**: segCount (exact) · pointiness · armour · fin count ·
  spike pairs · rgb at 4 bits/channel · strategy at 4 bits · energy at 2 bits ·
  unassigned-outline flag. Colour gets the most bits because lineage hue is the
  most legible signal on screen. Built lazily, capped at 448 entries and 12
  builds/frame; overflow wipes the atlas and refills from what is actually on
  screen, and an agent that loses the build race takes the vector path for one
  frame.
- **Sprites are baked at full ornament detail** (`ATLAS_PPT = 28` clears every
  size gate in `body.js`), so the atlas view gains back the fins, spikes, armour
  and eyes that the ornament budget had been discarding — they resample into
  texture instead of being skipped. Ornaments are now free.

**What is given up:** the kinematic wiggle, the per-agent envelope, and the glow
pulse, all frozen. At ≤20 px/tile a body is ~13 px long and its spine curvature
is under a pixel.

### Tried and reverted: raising the threshold to 56 px/tile

The ~9 µs/agent finding above says sprites pay off at every zoom, not only when
zoomed out, so the threshold was raised to 56 to cover ordinary grid sizes — and
because a frozen pose does not survive at that size, the spine was quantised into
four curvature buckets plus a mirror, calibrated against `chain.js` driven by
pond_core's real motion model.

**It looked wrong. Bodies read as stiff, and the whole thing was reverted.**

Keep the negative result: **the problem is the frozen pose, not the threshold.**
Four curvature buckets were not enough to hide it, and the calibration says why —
real spines bend gently (median 2.2°/joint, p90 7.1°), so what the eye is picking
up is not the *amount* of bend but its *continuity*. Quantising a small,
continuously-varying motion into four steps reads worse than not having it. More
buckets would be the obvious next try and is probably still wrong; the honest
options are per-agent deformation the atlas cannot express, or accepting sprites
only where the pose genuinely is invisible, which is what 20 px/tile means.

Two measurement notes worth keeping from that attempt:

- Reading spine curvature as first-to-last-link angle ÷ joint count **aliases**
  (`atan2` is (-π, π], so a hard-turning 11-segment eel returns the wrong sign).
  The middle joint alone cannot alias and tracks the whole-spine mean to 0.1°
  median.
- An early harness that *invented* heading-change-per-tick produced bends up to
  180°/joint — motions the engine cannot produce. Driving `chain.js` from
  `MAX_FORCE`, the speed clamp and `DT` gave a distribution 3–5× gentler. If
  anyone re-derives motion statistics, drive them from the engine's constants.

**Offline checks** (node, counting-canvas stub — no rasterisation, so this proves
correctness, not speed): keys stay inside 30 bits and are deterministic; all
baked geometry lands inside its packed cell for the worst-case body; a repeat ask
builds nothing; the build budget refuses past 12; 448 worst-case sprites fit
before the atlas wraps; a build costs ~13.8 fills, paid once.

The bounding-box check earned its keep: it caught the pivot being placed from the
wrong edge (bodies trail the head in −x, so a straight body rendered 56 px
outside its cell).

### The measurement that decides this

**`L` toggles the atlas at runtime**, and `M` now reports `sprite on/off`, how
the population split between the two pipelines, and the live atlas entry count.
Flip `L` with `M` open, at a fixed zoom and within one run. That is deliberate:
the two previous interventions were compared across rebuilds at different
populations, which is how a 3% swing got treated as a result when it was probably
drift.

- **Confirms the diagnosis:** `agents` ms falls by roughly the ratio of fills
  removed (~10×) with population and zoom held still.
- **Falsifies it:** `agents` ms barely moves with `drawn` showing the whole
  population on the sprite path. That would mean per-`fill()` overhead was not
  the cost, the ~12 µs figure is an artefact of apportionment, and the remaining
  candidate is per-pixel compositing — at which point the answer is WebGL2 and
  not a better atlas.
- **Still open either way: the superlinearity.** The atlas does not explain it
  and was not built to. Re-measure at 1,100 and 5,000 agents *at a fixed zoom*,
  both with and without `L`; if per-agent cost still climbs on the sprite path,
  the growth term is compositing or dirty-region behaviour, not call count.

## Leading hypotheses (superseded by the profile above, kept for the record)

Ranked by fit to the data, not by confidence. All are guesses.

1. **Path tessellation per fill.** Canvas2D flattens Bézier curves to polygons on
   the CPU on every `fill()`, with no caching between frames. 14 quadratic
   curves × 2 paths × N agents, re-tessellated 60×/second. Fits the ~4 µs
   constant floor. Does **not** by itself explain the superlinearity.
2. **GC pressure.** `drawBody` allocates ~51 short-lived `{x,y}` objects per
   agent per paint (`world`, `centers`, `dirs`, `left`, `right`, `glowLeft`,
   `glowRight`), plus `decodeAgent` and a `last_agents` record per agent. At
   5,733 agents that is ~300k objects/frame. Major GC pauses landing inside the
   measured span would read exactly like this, and would plausibly be
   superlinear. **Best fit for the superlinear term.**
3. **Canvas falling out of GPU acceleration** above some path or complexity
   budget, silently reverting to software rasterization.
4. **Additive overdraw.** Glow hulls are `lighter` and cannot early-out on
   overlap. But area arithmetic says glow covers only ~50% of the screen once at
   this population, which should cost single-digit ms, so this looks too small
   to be the story. Worth confirming, not assuming.
5. **Measurement misattribution.** The spans are wall-clock and Canvas2D is
   pipelined; deferred rasterization bills to whichever later call forces a
   flush. `other` reading 0.0 means the spans already sum past the frame total.
   The split is apportionment, not isolation.

## What we need from you

In priority order:

1. **Explain the superlinearity.** Why does per-agent cost go 8.2 → 47.3 µs
   between 1,100 and 5,000 agents when per-agent work is constant? This is the
   central question. A correct answer probably determines the fix.
2. **Falsify or confirm hypothesis 1 and 2** with a measurement, not an
   argument. What experiment isolates tessellation cost from GC cost in a
   browser?
3. **Say whether Canvas2D can reach 60 fps at 20,000 agents at all**, or whether
   this must become WebGL2 instanced rendering. If WebGL: what is the minimum
   viable path that preserves per-agent shape variety (agents have distinct
   morphologies — segment count, envelope profile, body plan — derived from
   their genome)?
4. **Sanity-check the planned direction** before it gets built:
   - Pre-rendered sprite atlas keyed by (body plan, segCount bucket, hue bucket,
     energy bucket), blitted with one `drawImage` per agent below ~3 px `baseR`,
     with the glow baked into the sprite so there is no additive pass at all.
   - Below ~1 px, stop drawing individuals entirely: accumulate into a coarse
     offscreen density buffer, blur, composite — a swarm as a shimmering cloud.
   - Long term, a hybrid: WebGL2 instanced quads for the distant mass, the
     existing Canvas2D hull pipeline for the few agents large on screen.

   Does the atlas actually dodge both hypothesis 1 and 2? Is the bucket
   scheme sound, or will lineage colour go visibly coarse?

## Constraints

- **Ships as a static site on itch.io.** No server, no custom headers assumed.
  This rules out anything needing cross-origin isolation (SharedArrayBuffer,
  wasm threads) unless someone confirms itch.io's per-project setting works.
- Plain ES modules, no bundler, no build step for JS. `package.json` has no
  `"type": "module"` — node treats `.js` as CJS, so an offline harness must copy
  files to `.mjs`.
- Visual identity matters. Agents must remain visually distinct by lineage
  (hue), strategy (glow), and morphology (silhouette). "Draw everything as
  identical dots" is not an acceptable answer, though a density field for
  sub-pixel agents is explicitly welcome.
- Illusions and foreground/background tricks are explicitly sanctioned by the
  project owner. It does not have to be honest, it has to look smooth.
- The population cap is gone and is not coming back. Do not propose one.

## Reproducing

```bash
# serve the repo root
python3 -m http.server 8000
# open http://localhost:8000/pond_web/index.html
```

Rebuild wasm only if `pond_core/` changed (`pond_core/pkg/` is gitignored):

```bash
wasm-pack build pond_core --target web --features wasm
```

In the browser: press `N`, set grid 64 and a population in the thousands, start.
Press `M` for the timing HUD. Let the pond boom past ~5,000 agents, zoom out to
fit (`F`), and read the `agents` line.

A Firefox capture has been taken (see "Profiler capture"). **A Chrome capture
has not**, and is worth doing: Chrome and Firefox have different Canvas2D
backends, and if the ~12 µs per-`fill()` overhead is Firefox-specific the whole
diagnosis changes. The primary dev container has no browser (`libnspr4.so`
missing, no sudo), so everything not backed by the `M` HUD or the profile is
inference.

## Key files

| file | what |
|---|---|
| `pond_web/renderer.js` | frame loop, `render()`, `draw_agents()`, timing HUD (`perf_mark`) |
| `pond_web/body.js` | `drawBody()` — the hot function, hull construction, ornaments |
| `pond_web/atlas.js` | sprite atlas: keying, packing, baking (`L` toggles its use) |
| `pond_web/chain.js` | kinematic chain follow physics |
| `pond_web/morphology.js` | genome knobs → `MorphSpec` (segCount, envelope, plan) |
| `HANDOFF.md` | prior profiling pass, op-count harness results, original ranked causes (**note: that ranking is now known to be wrong**) |
| `RULES.md` | sim spec |
