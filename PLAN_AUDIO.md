# PLAN_AUDIO — ambient track, played at random intervals

Scope: `pond_web/` plus the itch packaging script. No engine changes. The audio
files themselves are supplied; this plan covers everything around them.

The shape: silence most of the time, and every so often a track fades in, plays
once, and fades out. Not a loop. The pond is the thing being watched, and
continuous music turns an observation into a screensaver — a track that arrives
occasionally reads as weather.

---

## 1. Where the files live

```
pond_web/audio/
  <whatever they are named>.mp3
```

Two constraints worth knowing before the files are cut:

- **`.mp3`, not `.ogg`**, unless you want two copies of everything. Safari
  still has no Ogg Vorbis; mp3 is the one format every target decodes. If you
  already have Ogg, ship both and let an `<audio>` element pick — but that means
  two `<source>` children per track and doubles the bundle.
- **This is the first external asset the page has ever loaded.** Sprites are
  procedural, the wasm is the only other file. So the packaging script has to
  learn about a new directory (§6) and the page has to survive the files being
  missing (§5).

A manifest const in the module lists the tracks. A static host has no directory
listing, so the list is written down — there is no discovering it at runtime:

```js
const TRACKS = ['pond-a.mp3', 'pond-b.mp3', 'pond-c.mp3'];
```

## 2. New module: `pond_web/audio.js`

```js
export function initAudio({ isPaused }) → {
    toggleMute(),        // returns the new muted state
    isMuted(),
    isPlaying(),         // for the HUD chip
    setVolume(v),
}
```

`HTMLAudioElement`, not Web Audio. An ambient bed is a long file played
start-to-finish with a volume ramp; `new Audio(src)` streams it, decodes off the
main thread, and `.volume` is the only control needed. Web Audio would mean
fetching and decoding the whole file into memory before the first note, which
for a two-minute bed is megabytes of PCM for nothing.

One element per track, created lazily on first play and kept — `preload="none"`
so an unplayed track never costs a byte.

## 3. The scheduler

```
first gesture ──▶ wait FIRST_DELAY ──▶ play a track ──▶ wait GAP ──▶ play ──▶ …
```

- `FIRST_DELAY`: 20–45 s. Long enough that the audio is not part of the page
  loading, which is the difference between "the pond has a mood" and "a website
  played a sound at me".
- `GAP`: uniform in 90–240 s, redrawn each time. Uniform rather than fixed is
  the whole point of the feature; make both ends constants at the top of the
  file so they are tunable without reading the logic.
- Track choice: random, never the same one twice running (keep `lastIndex`).
  With three tracks a naive random repeats about a third of the time, which is
  exactly often enough to be noticed.
- Fade in over ~3 s, out over ~4 s, via `setInterval` on `.volume`. A track that
  arrives at full volume is a jump scare; a linear ramp on a log-perceived
  quantity is fine at these levels, no need for an equal-power curve.
- `setTimeout`, **not** the render loop. The speed dial multiplies sim time, and
  ambience is wall-clock — a pond at ×16 should not get sixteen times the music.

**Randomness must be `Math.random()`, not the engine's RNG.** Determinism is a
property this project actually holds: same seed, same run. Anything that draws
from the engine's stream makes the audio schedule part of the simulation and a
muted run diverge from an unmuted one. This is the same rule the renderer's
cosmetic jitter already follows.

## 4. When it must not play

| Condition | Behaviour |
|-----------|-----------|
| Before any user gesture | Nothing at all. Browsers block autoplay; a blocked `play()` throws a caught `NotAllowedError` and, worse, some browsers flag the tab. Arm on the first `pointerdown`/`keydown`, once, then start the timer. |
| Tab hidden | Fade out and pause on `visibilitychange`; resume the *schedule*, not the track, when visible. An itch page left open in a background tab playing music is how a person hunts down which tab to close. |
| Sim paused (`space`) | Let it finish. Pausing is "let me look at this", not "stop everything" — the `isPaused` hook is passed in so this can be reversed in one line if it feels wrong. |
| Muted | Timer keeps running, `play()` is skipped. Cheaper than tearing the schedule down and rebuilding it on unmute, and unmuting mid-silence is the honest behaviour. |
| Zen mode (`c`) | Unaffected. Zen clears the *interface*; the pond keeps running and so does its weather. |

## 5. Failure is silence, not a broken page

A missing or unplayable file fires `error` on the element. Handle it: mark that
track dead, drop it from the rotation, and if every track is dead, disable the
subsystem. One `console.warn`, not one per attempt.

The page must work with the `audio/` directory entirely absent — that is the
state of the repo until the files land, and it is also what a bad zip looks
like. Nothing about audio may run inside the frame loop, so a fault there cannot
reach the renderer's `report_frame_error` path.

## 6. Packaging (`scripts/package_itch.sh`)

The script copies `pond_web/*.html` and `*.js` and nothing else, so today a
bundle would ship silently. Add:

```bash
mkdir -p "$stage/audio"
cp "$root"/pond_web/audio/*.mp3 "$stage/audio/"
```

and a count check in the same spirit as the existing schema check — the script
already refuses to build a bundle whose wasm and page disagree, and a bundle
that is silently missing its audio is the same class of mistake:

```bash
n=$(ls "$stage/audio" | wc -l)
[ "$n" -ge 1 ] || { echo "no audio staged" >&2; exit 1; }
```

Watch the zip size: itch's HTML5 limit is generous but the whole bundle is
currently a couple of hundred KB, and three minutes of 128 kbps stereo is ~3 MB
per track. 96 kbps mono is inaudible-different for an ambient bed and a third
of the size.

## 7. Controls

- **`a`** toggles mute. Free — currently bound: `space +- g b p d m l k ; x c n
  f 0 [ ] arrows esc`.
- Default: **unmuted**, since the feature is silence-by-default anyway and a
  muted default means most visitors never learn it exists.
- Persist the mute in `localStorage` (`pond_audio_muted`). Nothing else in this
  page persists anything, so this is a new dependency on storage — worth it,
  because someone who muted once meant it. Wrap in try/catch: storage throws
  outright in some embedded/iframe contexts, and itch runs the page in an
  iframe.
- HUD: one chip near the speed readout, showing muted/playing. It is the only
  way to tell "no track is scheduled right now" from "audio is broken", which is
  otherwise indistinguishable and will absolutely be asked about.

## 8. Docs to update in the same commit

- `README.md` controls table — `a`.
- `pond_web/index.html` `#hint` column — `a  ambience`.
- `splash.js KEYS` — leave it out; the card is deliberately partial.
- `ITCH_RELEASE.md` — a line about audio being in the bundle and the size note.

## 9. Landing order

1. `audio.js` with the manifest empty and the subsystem inert — proves the
   no-files case works, which is the state the repo is in.
2. Scheduler + gesture arming + mute key + HUD chip, tested with one placeholder
   file.
3. Packaging + docs, once the real tracks exist.

## 10. Verification

No JS test harness in this repo, so: load the page and confirm nothing plays
before a click; confirm the first track arrives on schedule and fades rather
than cuts; background the tab mid-track and confirm it stops and does not
resume mid-phrase; mute, reload, confirm it is still muted; delete the audio
directory and confirm the page is unchanged.
