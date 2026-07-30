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
- local editor settings was already modified before this pass.
- Baseline and post-fix run logs are in the session scratchpad
  (`base_*.txt`, `sleepfix_*.txt`, `spec_*.txt`, `stats42.csv`); regenerate with
  `./target/release/run 12 100 2000 <seed>`.
