use wasm_bindgen::prelude::*;

use crate::morphology::MorphParams;
use crate::world::{World, DT};

// ── State buffer layout constants (exported so JS can read them) ──────────────
//
// get_state() returns a flat Float32Array with this layout:
//
//   [0..HEADER_LEN]                          — sim-wide header (6 floats)
//   [HEADER_LEN .. HEADER_LEN + n*AGENT_STRIDE] — per-agent data (AGENT_STRIDE floats each)
//   [above + gs*gs*TILE_STRIDE]              — per-tile data (3 floats each)

#[wasm_bindgen]
pub fn state_header_len() -> u32 { HEADER_LEN as u32 }
#[wasm_bindgen]
pub fn state_agent_stride() -> u32 { AGENT_STRIDE as u32 }
#[wasm_bindgen]
pub fn state_tile_stride() -> u32 { TILE_STRIDE as u32 }
#[wasm_bindgen]
pub fn state_death_stride() -> u32 { DEATH_STRIDE as u32 }
#[wasm_bindgen]
pub fn species_stride() -> u32 { SPECIES_STRIDE as u32 }
#[wasm_bindgen]
pub fn predator_state_stride() -> u32 { PREDATOR_STATE_STRIDE as u32 }
/// Distance at which an agent stops belonging to a species. The inspector shows
/// an agent's distance from its centroid against this, so drifting out is
/// visible before it happens.
#[wasm_bindgen]
pub fn species_membership_radius() -> f32 { crate::species::MEMBERSHIP_RADIUS as f32 }

const HEADER_LEN: usize = 7;
const AGENT_STRIDE: usize = 19;
const TILE_STRIDE: usize = 3;
/// Death record: [id, x, y, cause]. Appended after the tile block.
const DEATH_STRIDE: usize = 4;
/// Species row:
/// [id, members, peak, founded, extinct,
///  founder_members, founder_population, founder_generation,
///  promotion_streak, promotion_drift, promotion_spread,
///  entry_generation_advance, probation_generation_advance,
///  7 × current centroid, 7 × founding centroid,
///  7 × founding population centroid].
/// `extinct` is -1 while alive — a Float32Array has no null, and f32 represents
/// integers exactly far past any step count.
const SPECIES_STRIDE: usize = 13 + crate::species::SIG_LEN * 3;
/// Predator record: [id, leaving, tier, angle, reach].
const PREDATOR_STATE_STRIDE: usize = 5;

// Header field indices
const H_AGENT_COUNT: usize = 0;
const H_GRID_SIZE: usize = 1;
const H_STEP: usize = 2;
const H_TOTAL_FOOD: usize = 3;
const H_AVG_ENERGY: usize = 4;
const H_ALPHA: usize = 5;  // renderer interpolation factor [0,1)
const H_DEATH_COUNT: usize = 6;  // deaths since the last get_state() call

// Agent field offsets within stride
const A_X: usize = 0;
const A_Y: usize = 1;
const A_PREV_X: usize = 2;    // position at previous tick (for interpolation)
const A_PREV_Y: usize = 3;
const A_ENERGY_NORM: usize = 4;
const A_VEL_X: usize = 5;     // velocity (for direction/orientation in renderer)
const A_VEL_Y: usize = 6;
const A_GENOME_CLUSTER: usize = 7;
const A_BRAIN_CLUSTER: usize = 8;
const A_AGE_NORM: usize = 9;
const A_ID: usize = 10;       // stable agent id — swap_remove reshuffles array slots, id doesn't
// Trait-derived morphology knobs (see morphology.rs) — replaces raw trait export.
const A_MORPH_POINTINESS: usize = 11;
const A_MORPH_ELONGATION: usize = 12;
const A_MORPH_BULK: usize = 13;
const A_MORPH_ORNAMENT: usize = 14;
const A_MORPH_EYE_SIZE: usize = 15;
const A_MORPH_PULSE_RATE: usize = 16;
const A_MORPH_BELLY: usize = 17;
/// Promoted species id, 0 = unassigned. `A_GENOME_CLUSTER` stays alongside it:
/// the raw k-means label is how you tell "speciation is wrong" apart from
/// "clustering is wrong" when something looks off.
const A_SPECIES: usize = 18;

// Tile field offsets within stride
const T_FOOD: usize = 0;
const T_FERTILITY: usize = 1;
const T_MOVE_SPEED: usize = 2;

// ── WasmWorld ─────────────────────────────────────────────────────────────────

const TICK_MS: f32 = 1000.0 * DT; // 50 ms per physics tick (20 Hz)

#[wasm_bindgen]
pub struct WasmWorld {
    inner: World,
    accumulator: f32,
    last_cluster_step: u32,
}

#[wasm_bindgen]
impl WasmWorld {
    /// Create a new simulation world.
    #[wasm_bindgen(constructor)]
    pub fn new(grid_size: usize, population: usize, seed: u64) -> WasmWorld {
        console_error_panic_hook();
        WasmWorld {
            inner: World::new(grid_size, population, seed),
            accumulator: 0.0,
            last_cluster_step: 0,
        }
    }

    /// Fixed-timestep update. Call from requestAnimationFrame with elapsed ms.
    /// Drains the accumulator in 50 ms ticks; may advance 0 or more sim steps per call.
    /// Use `get_alpha()` to interpolate renderer positions between ticks.
    pub fn update(&mut self, delta_ms: f32) {
        self.accumulator += delta_ms;
        while self.accumulator >= TICK_MS && self.inner.agent_count() > 0 {
            self.inner.step();
            self.accumulator -= TICK_MS;
            self.last_cluster_step = self.inner.step_count
                - (self.inner.step_count % 50);
        }
        // Clamp accumulator to one tick so a long stall doesn't cause a burst
        if self.accumulator > TICK_MS * 3.0 {
            self.accumulator = TICK_MS * 3.0;
        }
    }

    /// All species, live and extinct, in promotion order. Flat, same convention
    /// as `get_state()`: `len / species_stride()` rows.
    ///
    /// Extinct species are included — the roster is the fossil record, not a
    /// live set, and the UI wants to show what died.
    pub fn species_list(&self) -> Vec<f32> {
        let all = self.inner.species.all();
        let mut buf = Vec::with_capacity(all.len() * SPECIES_STRIDE);
        for s in all {
            buf.push(s.id as f32);
            buf.push(s.members as f32);
            buf.push(s.peak_members as f32);
            buf.push(s.founded_step as f32);
            buf.push(s.extinct_at.map(|e| e as f32).unwrap_or(-1.0));
            buf.push(s.founder_members as f32);
            buf.push(s.founder_population as f32);
            buf.push(s.founder_generation);
            buf.push(s.promotion_streak as f32);
            buf.push(s.promotion_drift as f32);
            buf.push(s.promotion_spread as f32);
            buf.push(s.entry_generation_advance);
            buf.push(s.probation_generation_advance);
            for v in s.centroid {
                buf.push(v as f32);
            }
            for v in s.founding_centroid {
                buf.push(v as f32);
            }
            for v in s.founding_population_centroid {
                buf.push(v as f32);
            }
        }
        buf
    }

    /// Morphology knobs for species row `index`, so the legend swatch can be
    /// drawn through the same trait→colour path the bodies use.
    ///
    /// The centroid is a seven-dimensional signature, not a full genome, so the
    /// two locked traits are filled with the midpoint of their bounds — they are
    /// excluded from the signature precisely because they carry no selected
    /// information, and they do not vary within a lineage anyway.
    pub fn species_morph(&self, index: usize) -> Vec<f32> {
        let all = self.inner.species.all();
        let Some(s) = all.get(index) else { return Vec::new() };

        let bounds = crate::genome::Traits::BOUNDS;
        let mut raw = [0f64; 9];
        for (t, slot) in raw.iter_mut().enumerate() {
            let (lo, hi) = bounds[t];
            *slot = (lo + hi) / 2.0;
        }
        for (i, &t) in crate::species::SIGNATURE_DIMS.iter().enumerate() {
            let (lo, hi) = bounds[t];
            raw[t] = lo + s.centroid[i] * (hi - lo);
        }

        let traits = crate::genome::Traits {
            vision: raw[0], speed: raw[1], metabolism: raw[2],
            energy_capacity: raw[3], mutation_rate: raw[4],
            reproduction_cost: raw[5], attack: raw[6], defense: raw[7],
            aggression: raw[8],
        };
        let m = MorphParams::from_traits(&traits);
        vec![m.pointiness, m.elongation, m.bulk, m.ornament, m.eye_size, m.pulse_rate, m.belly]
    }

    /// Species names, parallel to `species_list()` rows.
    ///
    /// A separate call because strings cannot ride a `Float32Array` — the one
    /// place the flat-buffer convention cannot stretch. Each call allocates a JS
    /// string per species, so call it when the roster changes or on the panel's
    /// timer, never per frame.
    pub fn species_names(&self) -> Vec<JsValue> {
        self.inner
            .species
            .all()
            .iter()
            .map(|s| JsValue::from_str(&s.name.full()))
            .collect()
    }

    /// Turn behavioural (brain-weight) clustering on or off.
    ///
    /// Off by default. It is by a wide margin the most expensive work in the
    /// simulation, and nothing needs it unless the archetype view is open — so
    /// the renderer enables it on toggle and disables it on dismiss. Visitors
    /// who never open the panel pay nothing.
    pub fn set_brain_clustering(&mut self, on: bool) {
        self.inner.brain_clusters.set_enabled(on);
    }

    /// Whether behavioural clustering is currently running.
    pub fn brain_clustering_enabled(&self) -> bool {
        self.inner.brain_clusters.enabled()
    }

    // ── Tunables ──────────────────────────────────────────────────────────────
    // Three dials, live. Values are clamped in `World`, so a slider that sends
    // something out of range is corrected rather than trusted; read the getter
    // back to see what stuck.

    pub fn set_food_regen_scale(&mut self, v: f32) {
        self.inner.set_food_regen_scale(v as f64);
    }
    pub fn food_regen_scale(&self) -> f32 {
        self.inner.tunables().food_regen_scale as f32
    }

    pub fn set_hunt_aggression_threshold(&mut self, v: f32) {
        self.inner.set_hunt_aggression_threshold(v as f64);
    }
    pub fn hunt_aggression_threshold(&self) -> f32 {
        self.inner.tunables().hunt_aggression_threshold as f32
    }

    pub fn set_cluster_k(&mut self, k: u32) {
        self.inner.set_cluster_k(k as usize);
    }
    pub fn cluster_k(&self) -> u32 {
        self.inner.tunables().cluster_k as u32
    }

    /// Whether any dial has moved this run. Once true it stays true: the run is
    /// no longer reproducible from (grid, population, seed) alone, and the UI
    /// says so next to the seed.
    pub fn tunables_modified(&self) -> bool {
        self.inner.tunables().modified
    }

    /// Renderer interpolation factor in [0, 1). Blend prev_pos and pos by this.
    pub fn get_alpha(&self) -> f32 {
        (self.accumulator / TICK_MS).clamp(0.0, 1.0)
    }

    /// Advance simulation by `n` steps (legacy / headless path).
    pub fn step_n(&mut self, n: u32) {
        for _ in 0..n {
            if self.inner.agent_count() == 0 { break; }
            self.inner.step();
        }
        self.last_cluster_step = self.inner.step_count
            - (self.inner.step_count % 50);
    }

    /// Return full state as a flat Float32Array. See layout constants above.
    ///
    /// Takes `&mut self` because it drains the pending-death queue: each death is
    /// reported exactly once, to whichever frame happens to observe it. A frame
    /// spanning several sim steps therefore sees every death in between, not just
    /// the last tick's.
    pub fn get_state(&mut self) -> Vec<f32> {
        let deaths = std::mem::take(&mut self.inner.last_deaths);
        let w = &self.inner;
        let n = w.agent_count();
        let gs = w.grid_size;
        let tile_count = gs * gs;

        let total_len = HEADER_LEN
            + n * AGENT_STRIDE
            + tile_count * TILE_STRIDE
            + deaths.len() * DEATH_STRIDE;
        let mut buf = vec![0f32; total_len];

        let stats = w.get_stats();
        buf[H_AGENT_COUNT] = n as f32;
        buf[H_GRID_SIZE] = gs as f32;
        buf[H_STEP] = w.step_count as f32;
        buf[H_TOTAL_FOOD] = stats.total_food as f32;
        buf[H_AVG_ENERGY] = stats.avg_energy as f32;
        buf[H_ALPHA] = self.get_alpha();
        buf[H_DEATH_COUNT] = deaths.len() as f32;

        let cluster = &w.cluster;
        let agent_base = HEADER_LEN;
        for i in 0..n {
            let off = agent_base + i * AGENT_STRIDE;
            buf[off + A_X] = w.pos_x[i];
            buf[off + A_Y] = w.pos_y[i];
            buf[off + A_PREV_X] = w.prev_x[i];
            buf[off + A_PREV_Y] = w.prev_y[i];
            buf[off + A_ENERGY_NORM] = (w.energy[i] / 100.0).clamp(0.0, 1.0) as f32;
            buf[off + A_VEL_X] = w.vel_x[i];
            buf[off + A_VEL_Y] = w.vel_y[i];
            buf[off + A_GENOME_CLUSTER] = cluster.genome_cluster_ids.get(i).copied().unwrap_or(0) as f32;
            buf[off + A_BRAIN_CLUSTER] =
                self.inner.brain_clusters.labels.get(i).copied().unwrap_or(0) as f32;
            buf[off + A_AGE_NORM] = (w.age[i] as f64 / w.death_age[i] as f64).clamp(0.0, 1.0) as f32;
            buf[off + A_ID] = w.ids[i] as f32;

            let morph = MorphParams::from_traits(&w.genome[i].traits);
            buf[off + A_MORPH_POINTINESS] = morph.pointiness;
            buf[off + A_MORPH_ELONGATION] = morph.elongation;
            buf[off + A_MORPH_BULK] = morph.bulk;
            buf[off + A_MORPH_ORNAMENT] = morph.ornament;
            buf[off + A_MORPH_EYE_SIZE] = morph.eye_size;
            buf[off + A_MORPH_PULSE_RATE] = morph.pulse_rate;
            buf[off + A_MORPH_BELLY] = morph.belly;
            buf[off + A_SPECIES] = w.species_ids.get(i).copied().unwrap_or(0) as f32;
        }

        let tile_base = agent_base + n * AGENT_STRIDE;
        for (ti, tile) in w.tiles.iter().enumerate() {
            let off = tile_base + ti * TILE_STRIDE;
            buf[off + T_FOOD] = tile.food_units as f32;
            buf[off + T_FERTILITY] = tile.fertility as f32;
            buf[off + T_MOVE_SPEED] = tile.movement_speed as f32;
        }

        let death_base = tile_base + tile_count * TILE_STRIDE;
        for (di, d) in deaths.iter().enumerate() {
            let off = death_base + di * DEATH_STRIDE;
            buf[off] = d.id as f32;
            buf[off + 1] = d.x;
            buf[off + 2] = d.y;
            buf[off + 3] = d.cause as f32;
        }

        buf
    }

    /// Composite makeup of one genome family (k-means cluster).
    ///
    /// Layout: [member_count | 9 mean traits | 9 trait spreads (sd)
    ///          | 4 mean per-layer brain weight magnitudes
    ///          | 4 per-layer spreads] = 27 floats. Empty if the family has
    /// no live members.
    ///
    /// Brain weights are summarised per layer rather than exposed raw: 488
    /// numbers say nothing legible, whereas mean |w| per layer shows which
    /// stage of the network a family has invested in, and the spread shows
    /// how genetically converged that family actually is.
    pub fn cluster_composite(&self, cluster: u32) -> Vec<f32> {
        let w = &self.inner;
        let members: Vec<usize> = w.cluster.genome_cluster_ids.iter()
            .enumerate()
            .filter(|(_, &c)| c as u32 == cluster)
            .map(|(i, _)| i)
            .filter(|&i| i < w.genome.len())
            .collect();
        if members.is_empty() { return Vec::new(); }

        let n = members.len() as f32;
        let mut out = Vec::with_capacity(27);
        out.push(n);

        let traits_of = |i: usize| {
            let t = &w.genome[i].traits;
            [t.vision, t.speed, t.metabolism, t.energy_capacity, t.mutation_rate,
             t.reproduction_cost, t.attack, t.defense, t.aggression]
        };

        let mut means = [0f32; 9];
        for &i in &members {
            let t = traits_of(i);
            for d in 0..9 { means[d] += t[d] as f32; }
        }
        for m in means.iter_mut() { *m /= n; }

        let mut sds = [0f32; 9];
        for &i in &members {
            let t = traits_of(i);
            for d in 0..9 { sds[d] += (t[d] as f32 - means[d]).powi(2); }
        }
        for (d, sd) in sds.iter_mut().enumerate() { *sd = (*sd / n).sqrt(); let _ = d; }

        out.extend_from_slice(&means);
        out.extend_from_slice(&sds);

        // Layer weight spans in the flat 488-float buffer (see brain.rs).
        const LAYERS: [(usize, usize); 4] = [(0, 60), (72, 216), (228, 372), (384, 480)];
        let mut lmeans = [0f32; 4];
        let mut per_agent = vec![[0f32; 4]; members.len()];
        for (mi, &i) in members.iter().enumerate() {
            let bw = &w.genome[i].brain_weights;
            for (li, &(a, b)) in LAYERS.iter().enumerate() {
                let mag: f32 = bw[a..b].iter().map(|x| x.abs()).sum::<f32>() / (b - a) as f32;
                per_agent[mi][li] = mag;
                lmeans[li] += mag;
            }
        }
        for m in lmeans.iter_mut() { *m /= n; }

        let mut lsds = [0f32; 4];
        for row in &per_agent {
            for li in 0..4 { lsds[li] += (row[li] - lmeans[li]).powi(2); }
        }
        for sd in lsds.iter_mut() { *sd = (*sd / n).sqrt(); }

        out.extend_from_slice(&lmeans);
        out.extend_from_slice(&lsds);
        out
    }

    /// Add food to the tile under world position (cx, cy). Clamped to MAX_FOOD_PER_TILE.
    pub fn inject_food(&mut self, cx: f32, cy: f32, amount: u32) {
        self.inner.inject_food(cx, cy, amount);
    }

    /// Disturb food, fertility, and agent velocities within `radius` world units of (cx, cy).
    /// `intensity` in [0, 1]: 1.0 = maximum disruption.
    pub fn stir(&mut self, cx: f32, cy: f32, radius: f32, intensity: f32) {
        self.inner.stir(cx, cy, radius, intensity);
    }

    /// Spawn `count` agents near world position (cx, cy).
    pub fn pour_agents(&mut self, cx: f32, cy: f32, count: usize) {
        self.inner.pour_agents(cx, cy, count);
    }

    // ── God mode ──────────────────────────────────────────────────────────────

    /// Comet: kill everything within `radius` world units of (cx, cy).
    /// Returns the body count.
    pub fn smite_radius(&mut self, cx: f32, cy: f32, radius: f32) -> u32 {
        self.inner.smite_radius(cx, cy, radius)
    }

    /// Sweep: kill everything in the world-space column [x0, x1).
    pub fn smite_band(&mut self, x0: f32, x1: f32) -> u32 {
        self.inner.smite_band(x0, x1)
    }

    /// Empty the pond.
    pub fn smite_all(&mut self) -> u32 {
        self.inner.smite_all()
    }

    /// Immortality: suppress every natural death. Smites still land.
    pub fn set_immortal(&mut self, on: bool) {
        self.inner.immortal = on;
    }

    pub fn is_immortal(&self) -> bool {
        self.inner.immortal
    }

    /// Toggle only the automatic triangle ecology. Player god-mode summons are
    /// independent and continue their current hunt.
    pub fn set_automatic_predators(&mut self, on: bool) {
        self.inner.set_automatic_predators(on);
    }

    pub fn automatic_predators_enabled(&self) -> bool {
        self.inner.automatic_predators_enabled
    }

    /// Summon an ultra predator: immortal, unkillable, eats until only
    /// `PREDATOR_MANUAL_FRAC` of the current population is left, then leaves.
    /// Returns its agent id, or `u32::MAX` if the pack is already at its cap.
    /// Summon the octagon — a hit-and-run hunter that culls to a fifth of the
    /// pond and leaves. Player power: no automatic rule ever fields this tier.
    pub fn summon_octagon(&mut self) -> u32 {
        let target = (self.inner.prey_count() as f64
            * crate::world::PREDATOR_MANUAL_FRAC).round() as usize;
        self.inner
            .summon_predator_pack_tier(target, false, crate::world::PREDATOR_MANUAL_TIER)
            .unwrap_or(0)
    }

    /// Summon the rectangle — the sweep. Same contract as the octagon, more
    /// final: it kills everything its edges touch as it rotates.
    pub fn summon_rectangle(&mut self) -> u32 {
        let target = (self.inner.prey_count() as f64
            * crate::world::PREDATOR_MANUAL_FRAC).round() as usize;
        self.inner
            .summon_predator_pack_tier(target, false, crate::world::PREDATOR_RECTANGLE_TIER)
            .unwrap_or(0)
    }

    /// Call off every player-summoned hunt: the octagon and the rectangle swim
    /// off, over their usual departure. Returns how many were dismissed. The
    /// automatic triangle ecology has its own switch and is untouched.
    pub fn dismiss_summoned_predators(&mut self) -> u32 {
        self.inner.dismiss_summoned_predators() as u32
    }

    /// Player-summoned hunters in the pond that have not been dismissed yet.
    pub fn summoned_predator_count(&self) -> u32 {
        self.inner.summoned_predator_count() as u32
    }

    pub fn summon_predator(&mut self) -> u32 {
        self.inner
            .summon_predator(crate::world::PREDATOR_MANUAL_FRAC, false)
            .unwrap_or(u32::MAX)
    }

    /// Every predator in the pond, as [id, leaving] pairs — same flat-buffer
    /// convention as `get_state()`. `leaving` is 1 while a predator is swimming
    /// off, so the renderer can fade it.
    /// Per predator: `[id, leaving, tier, angle, reach]`. The renderer draws a
    /// different shape per tier, and needs the sweep angle and reach to draw the
    /// top two at the size they actually kill at.
    pub fn predators_state(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.inner.predators.len() * PREDATOR_STATE_STRIDE);
        for p in &self.inner.predators {
            out.push(p.id as f32);
            out.push(if p.leaving.is_some() { 1.0 } else { 0.0 });
            out.push(p.tier as f32);
            out.push(p.angle);
            out.push(crate::world::tier_bite(p.tier));
        }
        out
    }

    pub fn predator_count(&self) -> u32 {
        self.inner.predators.len() as u32
    }

    /// True while any predator in the pond summoned itself at the cull
    /// threshold rather than being called by the player.
    pub fn predator_is_automatic(&self) -> bool {
        self.inner.predators.iter().any(|p| p.automatic)
    }

    /// Population the current cull is driving down to.
    pub fn predator_target(&self) -> u32 {
        self.inner.predators.first().map(|p| p.target_pop as u32).unwrap_or(0)
    }

    /// Combined kill count of every predator currently in the pond.
    pub fn predator_kills(&self) -> u32 {
        self.inner.predators.iter().map(|p| p.kills).sum()
    }

    /// Living population excluding predators.
    pub fn prey_count(&self) -> u32 {
        self.inner.prey_count() as u32
    }

    /// Sustainable population for this pond, and the band around it at which
    /// predators arrive and leave.
    pub fn pop_cap(&self) -> u32 { self.inner.pop_cap() as u32 }
    pub fn cull_trigger_pop(&self) -> u32 { self.inner.cull_trigger_pop() as u32 }

    pub fn step_count(&self) -> u32 { self.inner.step_count }
    pub fn agent_count(&self) -> usize { self.inner.agent_count() }
    pub fn grid_size(&self) -> usize { self.inner.grid_size }

    /// Brain/trait snapshot for one agent (inspector panel). Empty vec if the
    /// id is not alive. Layout: [5 inputs | 12 h0 | 12 h1 | 12 h2 | 8 logits
    /// | 8 sigmoid gates | energy_norm | age_norm | 9 traits] = 68 floats.
    pub fn inspect_agent(&self, id: u32) -> Vec<f32> {
        self.inner.inspect_agent(id).unwrap_or_default()
    }

    /// Population means of the 9 genome traits, in Traits field order.
    pub fn trait_means(&self) -> Vec<f32> {
        self.inner.trait_means().iter().map(|&v| v as f32).collect()
    }

    /// Rolling stat time-series as a flat Float32Array, oldest sample first.
    /// `len / stats_sample_stride()` samples; see `stats.rs` for field order.
    ///
    /// Deaths in each sample are counts for that interval alone, so the death
    /// panel plots waves rather than a monotone staircase. Cumulative totals for
    /// the summary table come from `death_totals()`.
    ///
    /// Cheap enough to call at 1 Hz; there is no reason to call it per frame,
    /// since samples only appear every `SAMPLE_INTERVAL` sim steps.
    pub fn stats_history(&self) -> Vec<f32> {
        self.inner.stats_history.to_flat()
    }

    /// Cumulative deaths per cause, indexed by the same codes the death records
    /// in `get_state()` use: [starvation, old_age, killed_in_combat, eaten_alive].
    pub fn death_totals(&self) -> Vec<f32> {
        self.inner.death_counts().iter().map(|&c| c as f32).collect()
    }

    /// Peak living population over the retained history window.
    pub fn peak_population(&self) -> u32 {
        self.inner.stats_history.peak_alive()
    }
}

/// Agents per tile the pond is assumed to sustain. The cull threshold is this
/// times the tile count, plus the hysteresis band.
#[wasm_bindgen]
pub fn predator_pop_per_tile() -> f32 { crate::world::PREDATOR_POP_PER_TILE as f32 }

#[wasm_bindgen]
pub fn stats_sample_stride() -> u32 { crate::stats::SAMPLE_STRIDE as u32 }
#[wasm_bindgen]
pub fn stats_sample_interval() -> u32 { crate::stats::SAMPLE_INTERVAL }

/// Trait bounds [lo, hi] × 9 in Traits field order (vision, speed, metabolism,
/// energy_capacity, mutation_rate, reproduction_cost, attack, defense,
/// aggression). Single source for JS bar normalization — mirrors
/// genome.rs::Traits::generate ranges.
#[wasm_bindgen]
pub fn trait_bounds() -> Vec<f32> {
    crate::genome::Traits::BOUNDS
        .iter()
        .flat_map(|&(lo, hi)| [lo as f32, hi as f32])
        .collect()
}

/// Default and bounds for each tunable, flat, three rows of
/// `[default, min, max]` in the order: food regen scale, hunt aggression
/// threshold, clustering k. Slider setup reads this so no range or default is
/// written down twice — the core owns them.
#[wasm_bindgen]
pub fn tunable_ranges() -> Vec<f32> {
    use crate::world::*;
    vec![
        DEFAULT_FOOD_REGEN_SCALE as f32,
        FOOD_REGEN_SCALE_RANGE.0 as f32,
        FOOD_REGEN_SCALE_RANGE.1 as f32,
        DEFAULT_HUNT_AGGRESSION_THRESHOLD as f32,
        HUNT_AGGRESSION_THRESHOLD_RANGE.0 as f32,
        HUNT_AGGRESSION_THRESHOLD_RANGE.1 as f32,
        DEFAULT_CLUSTER_K as f32,
        CLUSTER_K_RANGE.0 as f32,
        CLUSTER_K_RANGE.1 as f32,
    ]
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn js_console_error(s: String);
}

fn console_error_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        js_console_error(info.to_string());
    }));
}
