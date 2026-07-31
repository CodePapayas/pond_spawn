# Releasing to itch.io

Everything the browser needs is in one zip. Build it, upload it, set five
fields, done.

```bash
./scripts/package_itch.sh
# → dist/pond_spawn_itch.zip   (~210 KB)
```

The script builds `pond_core` to wasm, copies `pond_web/` and the two engine
files into a flat tree, rewrites the one import that points outside the bundle
(`../pond_core/pkg/pond_core.js` → `./pkg/pond_core.js`), checks that the page's
`EXPECTED_SCHEMA` matches the engine's `SCHEMA_VERSION`, and zips it. It refuses
to produce a bundle if either of those two checks fails, so a zip that exists is
a zip that is internally consistent.

Contents: `index.html` at the root, the 17 page modules beside it, and
`pkg/pond_core.js` + `pkg/pond_core_bg.wasm`. No CDN, no fonts, no network calls
of any kind — it runs entirely from what is in the zip.

## Test it before uploading

```bash
cd dist/pond_spawn_itch && python3 -m http.server 8000
# open http://localhost:8000/
```

Open it from the *bundle*, not from `pond_web/` — that is the tree itch will
serve, and the rewritten import is the part worth exercising. What you should
see: white screen reading "digging a pond for u ;)" for about a second and a
half, fading into a pond that already has a few named species in it, with the
welcome card over the top. Check the browser console is clean.

## Upload

1. itch.io → **Dashboard** → **Create new project** (or edit the existing one).
2. **Kind of project**: `HTML`.
3. Upload `pond_spawn_itch.zip`, then tick **"This file will be played in the
   browser"** on it. Without that tick it is offered as a download and nothing
   runs.
4. **Embed options**:
   - *Viewport dimensions*: **1280 × 720**. The layout is fixed-position panels
     around a canvas and it fills whatever it is given; below roughly 1000 px
     wide the side panels start crowding the pond.
   - *Fullscreen button*: **on**. It is the best way to see it and the panels
     lay out properly at any size.
   - *Mobile friendly*: **off**. Everything is click, drag, wheel and keyboard.
   - *Automatically start on page load*: your call. Off gives a click-to-play
     splash, which is polite about the ~2 s of wasm boot; on means the opening
     curtain is the first thing they see, which is the intended sequence.
5. **SharedArrayBuffer support**: leave **off**. The engine is single-threaded
   and does not need cross-origin isolation; turning it on only adds headers
   that can break embedding.
6. Set visibility to **Restricted** (or Draft with the secret link) while
   friends are testing, then Public when you are happy.

## What to tell testers

- It starts mid-run on purpose — that pond is ~4,200 ticks old so there are
  already named species to look at. `new run` starts a fresh one from tick 0.
- Click any animal to open the inspector: genome, live neuron activations, its
  lineage. `g` graphs, `p` the phylogeny tree, `b` behavioural archetypes,
  `c` clears the whole interface for a clean look at the pond.
- God mode is in the top-right panel and includes the hunters.
- The seed reproduces a run exactly. If something interesting or broken
  happens, the seed and the run parameters from the `new run` panel are enough
  for anyone else to see the same thing — ask for those in a bug report.

## Updating a release

Re-run the script and upload the new zip over the old one; itch keeps the same
URL and players get the new build on reload. Two things to remember:

- **Rebuild the wasm** — the script does this for you, so never hand-assemble a
  zip from a stale `pond_core/pkg/`.
- **Bump `SCHEMA_VERSION`** if you changed any buffer layout, or the page will
  read the right number of floats from the wrong places. The script's schema
  check catches a mismatch between page and engine, not a layout change you
  forgot to version.

`dist/` is gitignored; the zip is a build artifact, not something to commit.
