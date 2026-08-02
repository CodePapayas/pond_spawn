# PLAN_UI_QOL — window sizing, symmetric hotkeys, auto sprite LOD

**Status: implemented.** Two deviations from the plan below, both in §1:
width is set explicitly but height is left to the content (capped by the
stylesheet's `max-height`) except where a caller asks for one — a short species
card should stay short. And windows are positioned *after* their first render,
since a content-height window has no height before that and the clamp was
reading zero.

Scope: `pond_web/` only. No engine changes, no `pond_core` changes. Four
independent workstreams; each lands on its own.

---

## 1. Floating windows fit the viewport

### What is wrong now

`floating.js` never sets a size. `.float-window` is `width: min-content` with
`max-width: calc(100vw - 16px)`, so the window is as wide as its content wants
to be. The phylogeny tree is laid out at a **fixed 780 px** (`W` in
`phylogeny.js:22`) with height `PAD_TOP + PAD_BOTTOM + species.length * 46`.
On a 1000×650 laptop window that is a window wider than the pond and taller
than the screen — the `max-*` rules only turn the overflow into scrollbars,
they do not shrink the drawing.

The open position is also sized against constants that no longer match:
`floating.js:27-28` clamps with hardcoded `360`/`260` instead of the window's
real size, so a tall window opens with its bottom off-screen.

### Change

**`floating.js`**

- `openFloating({..., size })` where `size` is optional
  `{ w, h, minW, minH }` in CSS px, the window's *preferred* size.
- On create, compute and set explicit `style.width` / `style.height`:
  ```
  avail_w = window.innerWidth  - 32
  avail_h = window.innerHeight - 32
  w = clamp(minW, size.w ?? 420, avail_w)
  h = clamp(minH, size.h ?? 380, min(avail_h, 0.9 * window.innerHeight))
  ```
  Setting an explicit size is also what makes `resize: both` behave — a
  `min-content` box snaps oddly on first drag.
- Cascade offset uses the computed `w`/`h`, not `360`/`260`.
- `clamp()` gains a size clamp: if a window is larger than the viewport
  (viewport shrank, or the user resized it big then resized the browser),
  shrink it to `avail_w`/`avail_h` before re-clamping the position. Call it
  from the existing `window.resize` listener, which currently only moves.

**`index.html`**

- `.float-window`: drop `width: min-content`; keep `min-width: 340px`,
  `resize: both`, `max-width/height`. Add `display:flex; flex-direction:column`
  and `.float-body { flex: 1 1 auto; overflow: auto; min-height: 0 }` so the
  body scrolls inside the frame rather than the frame growing past the screen.

### Per-window content that must react to the frame

`resize: both` + the existing `ResizeObserver` already re-renders on drag; the
content has to actually use the new size.

- **graphs** (`graphs.js:323`) already measures `body.clientWidth`. No change.
- **phylogeny** (`phylogeny.js:517-527`) draws at `layout.w` × `layout.h`
  unconditionally. Change `renderInto` to fit:
  ```
  const avail = body.clientWidth - 20;            // body padding
  const scale = Math.min(1, avail / layout.w);
  canvas.style.width  = layout.w * scale + 'px';
  canvas.style.height = layout.h * scale + 'px';
  canvas.width  = round(layout.w * scale * dpr);
  canvas.height = round(layout.h * scale * dpr);
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
  ```
  Everything downstream (`drawTree`) keeps drawing in layout units, and the
  PNG/SVG exports keep using `currentLayout` at full 780 px — the export should
  not inherit the screen's constraint.
  Open size hint: `{ w: 820, h: 620, minW: 360, minH: 260 }`.
- **species** (`species.js:185`) is a fixed `.species-window { width: 430px }`
  in CSS, which fights an explicit inline width. Drop the CSS width, pass
  `size: { w: 430, h: 520 }` instead.

### Non-floating panels on a short window

Same class of bug, different mechanism — these are `.panel`, not
`.float-window`:

- `#archetypes` is `max-height: calc(100vh - 360px)`; under ~500 px tall that
  is a sliver. Change to `calc(100vh - 32px)` and let the left column stack:
  it is `top:16px; left:16px` and the keybind key is bottom-left, so also add
  `@media (max-height: 700px)` rules that hide the hint key's least useful
  half — or simply rely on §2, where the key is now individually closable.
- `#side-right` is `width: 230px` fixed; below ~900 px wide it eats a quarter
  of the pond. Add `@media (max-width: 900px) { #side-right { width: 190px } }`
  and the matching `#god` rule (they share the column and `layout_right_column`
  measures, so nothing else needs touching).

---

## 2. Every hotkey toggles both ways, and the legend/key are individually closable

### Audit of `on_key` (`renderer.js:802-854`)

| key | today | verdict |
|-----|-------|---------|
| `g` graphs | `toggle_graphs` | symmetric ✓ |
| `b` archetypes | `toggle_archetypes` | symmetric ✓ |
| `d` debug clusters | `toggle_debug` | symmetric ✓, but force-shows `#side-right` on open and never restores it |
| `m` perf | `toggle_perf` | symmetric ✓ |
| `n` setup | ternary | symmetric ✓ |
| `c` zen | `toggle_zen` | symmetric ✓ |
| `p` phylogeny | `openPhylogeny` | **open only** — closing is the × button |
| `l` sprites | `toggle_sprites` | symmetric, but being remapped (§3) |
| legend (`#side-right`) | — | **no key at all**, only `c` clears everything |
| controls key (`#hint`) | click / `?` chip | **no key at all** |

### Change

**`floating.js`** — export two helpers the renderer can drive keys from:
```js
export function isFloatingOpen(key) { return open.has(key); }
export function closeFloating(key)  { const w = open.get(key); if (w) { open.delete(key); w.remove(); } return !!w; }
```

**`phylogeny.js`** — export `PHYLO_KEY` (the existing `'tree:phylogeny'`) or a
`togglePhylogeny(source, colorFor)` that closes when open. Prefer the latter so
the key constant stays private.

**`renderer.js`**
- `toggle_phylogeny()` → close if open, else open.
- New `legend_visible` flag (default `true`, matching today's CSS) and
  `toggle_legend()` setting `#side-right` display. Bound to **`l`**.
- `toggle_hint_click()` already exists and is symmetric — bind **`k`** to it so
  the controls key has a keyboard toggle to match the `?` chip.
- `toggle_debug()`: when it force-shows `#side-right`, set `legend_visible =
  true` too, so `l` afterwards actually hides it (today the flag and the DOM
  would disagree).

Zen (`c`) is unchanged and stays the master switch: it is a body class, every
panel keeps its own flag, and leaving zen restores exactly what was open. The
new per-panel flags inherit that property for free.

---

## 3. Sprite LOD: `;` and an auto mode

### Constraint from the existing comment (`renderer.js:213-235`)

Sprites are off by default *for a reason*: the atlas key is colour × silhouette,
both continuous per-agent traits, so a mature pond blows past the 448-entry
budget and wipes forever. "Kick on automatically if it makes sense" therefore
has to mean *measure whether it is making sense and back out if not* — not
"turn on when the pond is big".

### Change

Replace the boolean with a tri-state:

```js
let sprite_mode = 'auto';         // 'auto' | 'on' | 'off'
let sprites_enabled = false;      // what the frame actually does; auto writes it
```

- **`;`** cycles `auto → on → off → auto`. (`'` is the other candidate; `;` is
  unshifted on every layout and next to nothing else bound.) Resets the perf
  EMAs exactly as `toggle_sprites` does today.
- `l` no longer touches sprites (§2).
- The HUD perf block (`renderer.js:1453`) prints the mode, not just on/off:
  `sprite  auto→on  drawn …`.

**Auto policy**, evaluated once a second (not per frame), all thresholds named
constants at the top of the file:

Enable only when *all* hold:
1. `perf_scale_px <= SPRITE_LOD_MAX_SCALE_PX` — below the threshold sprites are
   never used, so switching on is a no-op that only risks a wipe.
2. `agent_count >= SPRITE_AUTO_MIN_AGENTS` (start at 400) — under that, the
   body path is already fast enough.
3. `frame_ema >= SPRITE_AUTO_SLOW_MS` (start at 20 ms ≈ below 50 fps) — only
   pay for the atlas when there is a problem to fix.

`frame_ema` is a new EMA of the existing `frame_delta` in `frame_body`
(`renderer.js:1087-1089`) — one subtraction per frame, unlike `perf`, which
only accumulates while `m` is open.

Back out to off when, after enabling:
4. `atlasStats().wipes` grew by more than `SPRITE_AUTO_MAX_WIPES_PER_SEC` (1)
   across the last evaluation window — the exact thrash the comment describes;
   **or**
5. `frame_ema` did not improve by at least 10% over the pre-enable reading
   after ~3 s.

On backing out, latch: set an `auto_locked_until = step + N` (or a simple
"don't retry this run unless the population halves") so it does not oscillate.
Hysteresis on (1)–(3) as well: require the condition to hold for two
consecutive evaluations before switching either way.

`resetAtlasStats()` is already called on run start (`renderer.js:421`), so the
wipe delta is per-run. Call it also on a manual mode change so a reading always
describes the current mode.

Default: `sprite_mode = 'auto'`, which — given (1)–(3) — behaves as **off** on
every default-size pond, satisfying "off by default" while letting a heavy,
zoomed-out pond claw back frames.

---

## 4. Docs the keys live in

Three places list keys and all three must match, or the key list lies:

- `index.html` `#hint` block (lines 787-805) — add `l legend`, `k controls key`,
  `; sprites (auto)`.
- `splash.js` `KEYS` (lines 14-23) — it is deliberately partial; add
  `l · k` to the panel line, leave `;` out.
- `README.md` controls section, if it enumerates keys.

`RULES.md` is the sim spec and is untouched by all of this.

---

## Suggested landing order

1. §2 hotkeys (smallest, unblocks the `l` remap).
2. §3 sprite auto mode (depends on `l` being free).
3. §1 window sizing (largest; `floating.js` + three call sites).
4. §4 docs, in whichever commit changes the binding.

## Test notes

No test harness covers `pond_web` (the `tests/` suite is the legacy Python
sim). Verification is manual:

- Resize the browser to 1024×600 and open phylogeny with 12+ species — window
  must fit inside the viewport and the tree must scale, not scroll.
- Every key in the table above pressed twice returns to the starting state.
- `m` + `;` on a grid-64 pond at full zoom-out: watch `wipes` after auto
  enables; it must latch off rather than cycle.
