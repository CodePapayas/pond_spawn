# PLAN_TUNABLES — three exposed dials

Status: **built, 2026-07-30.** Kept as the record of what was decided and why;
what shipped differs from the plan in two places, noted inline below.

Expose three numbers that are currently hardcoded, each with a hover blurb that
says what it changes. The point is not configurability for its own sake — it is
that these three are the levers a viewer would actually want to pull, and each
one currently requires a rebuild to try.

| Dial | Now on `Tunables` as | Default | Effect |
|---|---|---|---|
| food regen rate | `food_regen_scale` | 0.012 | per-tile per-tick chance of +1 food, scaled by fertility |
| predator/prey aggression threshold | `hunt_aggression_threshold` | 0.80 | `aggression` at or above this makes an agent hunt other agents |
| clustering `k` | `cluster_k` | 6 | how many genome families the k-means pass splits the pond into |

Note these are three different *kinds* of knob, which is why they need different
handling below: one is an economy rate, one is a behavioural gate that decides
who is a predator, and one changes nothing about the simulation at all — it only
changes how the sim is *read*.

## 1. Core: a `Tunables` struct on `World`

```rust
pub struct Tunables {
    pub food_regen_scale: f64,          // 0.0 – 0.05,  default 0.012
    pub hunt_aggression_threshold: f64, // 0.0 – 1.06,  default 0.80
    pub cluster_k: usize,               // 2 – 12,      default 6
    pub modified: bool,                 // latched once any dial leaves default
}
```

- The existing constants stay, as `Tunables::default()`. Nothing gets a magic
  number written in two places.
- `BiomeTile::regen_rate` currently owns `REGEN_RATE_SCALE`; it should take the
  scale as an argument and keep `MAX_FERTILITY`, since fertility normalization
  is the tile's business and the rate is the world's. Call site is
  `world.rs::tick_food_regen`.
- `cluster_k` feeds `ClusterState::run(&self.genome, k, …)`.

  **This is where the plan was wrong.** `match_labels` did not survive a `k`
  change: it sized its label space by `k.max(prev.len())`, so lowering `k` let a
  cluster inherit a label from the wider previous run — a label above the new
  `k`, which then indexed past `genome_centroids`, past the legend's rows, and
  past the headless histogram's fixed array. It panicked on the first shrink.
  Labels are now clamped below the current `k`, and previous labels at or above
  it are not offered for matching. A family whose label went out of range is
  remapped and changes colour once, which is the honest reading: lowering `k`
  merged those families away.
  Brain clustering keeps its own `k = 24`; not in scope.
- Bounds are clamped in core, not only in JS. The UI is one caller.

**Determinism.** Same seed + same tunables = same run, unchanged. A run whose
dials were moved mid-flight is no longer reproducible from `(grid, population,
seed)` alone. Two consequences: the setup panel's "same seed = same pond" line
stays true only if the dials are also at defaults, and any dial that has been
touched should be recorded so the HUD can say so.

Built as a latched `modified` flag on `Tunables`, surfaced as a line at the foot
of the tuning panel ("dials moved — this run is no longer reproducible from its
seed alone"). It does not clear when a dial goes back to its default: the run
already diverged. Still open: whether changed dials should also be written into
the stats CSV header.

## 2. Wasm surface

Mirror the existing pattern in `wasm.rs` — a getter, a setter, and a free
function for the default so JS never hardcodes a number the core owns (as
`species_membership_radius()` already does):

```
set_food_regen_scale(f32) / food_regen_scale() -> f32
set_hunt_aggression_threshold(f32) / hunt_aggression_threshold() -> f32
set_cluster_k(u32) / cluster_k() -> u32
tunables_modified() -> bool
tunable_ranges() -> Vec<f32>     // [default, min, max] × 3, for slider init
```

`set_cluster_k` should force the next step to re-run clustering rather than wait
out the 50-step cycle, or the dial appears dead for up to 50 steps.

## 3. UI: where the dials live

Put them in a **`tuning` panel**, a sibling of the god panel, not in `setup.js`.
Rationale: `setup.js` is documented as taking exactly the three arguments
`World::new` takes, and these are not that — they are live rules, and watching a
pond respond to a regen change in real time is the whole appeal. Keep that
distinction rather than blurring it.

Each row: label, slider, live numeric value, reset-to-default affordance, and an
`ⓘ` that reveals the blurb on hover **and on focus** (keyboard, touch). A bare
`title=` attribute is enough for the one-line case in `archetypes.js`, but these
blurbs are two sentences and `title` gives no styling, a ~1s delay, and nothing
on touch — so this wants the same treatment as `.god-hint`: a positioned
`div`, `.tune-info`, shown by CSS on `:hover`/`:focus-within`. Must not obscure
the grid — the tuning panel is a side panel like the others.

Draft blurb copy (revise in place when building):

- **food regen** — "How fast tiles grow food back. Higher means a richer pond
  and more agents feeding without moving; at zero nothing ever regrows and the
  pond runs down to whatever food it started with."
- **hunt threshold** — "How aggressive an agent must be before it hunts other
  agents instead of grazing. Lower turns more of the pond predatory; above the
  trait's maximum of 1.05, nobody hunts and combat stops entirely."
- **families (k)** — "How many genome families the pond is sorted into for
  colouring and species tracking. This changes only how you see the pond, never
  how it behaves — more families means finer splits between lineages that are
  nearly alike."

That last line matters and should survive editing: `k` is presentation, the
other two are physics. A viewer who does not know that will read a colour
re-shuffle as the sim having changed.

## 4. Tests (as built)

- `food_regen_scale = 0.0` → no tile ever gains food over N steps.
- `hunt_aggression_threshold` above 1.05 → zero `KilledInCombat` deaths in a run
  that otherwise produces hundreds.
- `cluster_k` set to 3 → every label `< 3`, applied on the next step rather than
  the next 50-step cycle, and the pass survives the change without panicking on
  the previous centroid vector. Plus `labels_stay_below_k_when_k_shrinks` in
  `cluster.rs`, which is the regression test for the bug above.
- An f32 round-trip of a default is not a modification: the UI reads defaults
  back through f32, so `0.012` returns as `0.012000000104…` and an exact
  comparison latched `modified` the first time anyone pressed reset.
- Defaults reproduce current behaviour bit-for-bit: an existing seeded run gives
  an identical population/energy trace with `Tunables::default()`.

That last one is the load-bearing test — the refactor from constants to fields
should be provably inert before any dial is ever moved.

## 5. Not verified in a real browser

The panel was exercised under jsdom against a stub engine with the same clamp
rules — rows render, the blurbs are present and focusable, values clamp (k 99 →
12), reset restores, the reproducibility note latches. That is DOM logic, not
the app: headless Chromium could not start in this container (`libnspr4.so`
missing, no sudo to install it), so **nothing here has been seen rendered**.
Layout, the blurb popover's position, and the panel over a live wasm world are
all unconfirmed. Worth a look on a machine with a browser.

Dials carry across a restart, like the god-mode switches: a fresh `World` starts
at defaults, and a panel still showing a moved slider over a world that had
discarded it would be lying.
