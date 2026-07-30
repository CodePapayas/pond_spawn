//! Wire format version for everything the browser reads out of the engine.
//!
//! The JS side does not parse a self-describing format. It reads flat
//! `Float32Array`s and indexes them by hand: `decode.js` mirrors the agent
//! stride's field offsets, `species.js` mirrors the species row, `panels.js`
//! mirrors the 27-float composite, and `inspector.js` mirrors the traced
//! forward pass. Every one of those is a comment asking the next person to keep
//! two files in sync.
//!
//! Which mostly worked, and then did not: three separate comments disagreed
//! with the code about whether the inspect buffer was 68 or 69 floats, and the
//! header was documented as 6 floats while `HEADER_LEN` was 7. Nobody noticed,
//! because a stale offset does not fail — it silently reads the neighbouring
//! field, and a plausible-looking number appears in the UI.
//!
//! So: one integer, checked at boot. A mismatch between the wasm bundle and the
//! page that loads it becomes a loud failure instead of a wrong number.
//!
//! **Bump `SCHEMA_VERSION` whenever any of these change:**
//!
//! - any stride (`AGENT_STRIDE`, `TILE_STRIDE`, `SPECIES_STRIDE`, …) or the
//!   order of fields within one,
//! - the header layout, the death record, or the stats sample,
//! - the brain's shape — layer sizes or input count,
//! - the number or order of `CauseOfDeath` variants,
//! - anything else `wasm.rs` hands over as a bare array of floats.
//!
//! It is a version, not a migration: there is no persistence in this project,
//! so nothing needs converting. The only job is to stop a stale `pond_core/pkg`
//! from being read by a fresh `pond_web`, or the reverse.

/// Bumped by hand. See the module docs for what counts as a change.
///
/// - 1 — first versioned build. Baseline: 7-float header, 19-float agent
///   stride, 35-float species row, 5 death causes, brain `5→12→12→12→8`.
/// - 2 — brain gains two threat inputs: `7→12→12→12→8`, 512 weights. Inspect
///   buffer and its input labels grow with it.
/// - 3 — `intelligence` becomes a live trait: 10 traits rather than 9, and it
///   joins the species signature, so the composite buffer, the species row and
///   the inspector's trait block all grow.
/// - 4 — `immunity` joins the traits and the species signature: 11 traits,
///   `SIG_LEN` 9, so the species row and the composite buffer grow again.
/// - 5 — disease roster exported: `disease_list` / `disease_names` and the
///   `disease_stride` layout.
pub const SCHEMA_VERSION: u32 = 5;
