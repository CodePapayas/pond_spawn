#!/usr/bin/env bash
#
# Build the browser bundle and zip it for itch.io.
#
# itch serves an HTML5 project from the root of the uploaded zip, so the tree
# has to be flat and self-contained: `index.html` at the top, every module
# beside it, and the wasm under `pkg/`. The repo layout is not that — the page
# lives in `pond_web/` and imports the engine from `../pond_core/pkg/` — so this
# copies rather than zips in place, and rewrites the one import that crosses the
# boundary.
#
# Usage: ./scripts/package_itch.sh
# Output: dist/pond_spawn_itch.zip

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/dist"
stage="$out/pond_spawn_itch"
zip_path="$out/pond_spawn_itch.zip"

command -v wasm-pack >/dev/null || {
    echo "wasm-pack not found — cargo install wasm-pack" >&2; exit 1; }
command -v python3 >/dev/null || {
    echo "python3 not found — needed to write the zip" >&2; exit 1; }

echo "==> building the engine to wasm (release)"
# wasm-pack is release by default; passing --release as well is an error.
wasm-pack build "$root/pond_core" --target web --features wasm >/dev/null

echo "==> staging"
rm -rf "$stage" "$zip_path"
mkdir -p "$stage/pkg"

cp "$root"/pond_web/*.html "$root"/pond_web/*.js "$stage/"
# Only the two files the page actually loads. The .d.ts and package.json are for
# bundler consumers and would just be dead weight in the upload.
cp "$root"/pond_core/pkg/pond_core.js "$root"/pond_core/pkg/pond_core_bg.wasm "$stage/pkg/"

# The one path that only makes sense in the repo. Fails loudly if the import
# ever moves, rather than shipping a bundle that 404s on the engine.
grep -q "\.\./pond_core/pkg/pond_core\.js" "$stage/renderer.js" || {
    echo "engine import not found in renderer.js — packaging is out of date" >&2
    exit 1; }
sed -i "s#\.\./pond_core/pkg/pond_core\.js#./pkg/pond_core.js#" "$stage/renderer.js"

# The schema check the page does at boot, done here too: a bundle whose wasm and
# page disagree fails on someone else's machine otherwise.
engine_schema="$(grep -oP 'SCHEMA_VERSION: u32 = \K[0-9]+' "$root/pond_core/src/schema.rs")"
page_schema="$(grep -oP 'EXPECTED_SCHEMA = \K[0-9]+' "$stage/renderer.js")"
[ "$engine_schema" = "$page_schema" ] || {
    echo "schema mismatch: engine $engine_schema, page $page_schema" >&2; exit 1; }
echo "    schema $engine_schema, engine and page agree"

# Written with python's zipfile rather than `zip`, which is not installed
# everywhere and is the only reason this script would need a package manager.
echo "==> zipping"
python3 - "$stage" "$zip_path" <<'PY'
import os, sys, zipfile
stage, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for dirpath, _, files in os.walk(stage):
        for f in sorted(files):
            full = os.path.join(dirpath, f)
            z.write(full, os.path.relpath(full, stage))
PY

echo
echo "    $zip_path  ($(du -h "$zip_path" | cut -f1))"
python3 -c "import zipfile,sys; [print('     ', n) for n in zipfile.ZipFile(sys.argv[1]).namelist()]" "$zip_path"
echo
echo "Upload it as an HTML project — see ITCH_RELEASE.md."
