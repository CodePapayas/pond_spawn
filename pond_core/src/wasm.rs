use wasm_bindgen::prelude::*;

use crate::morphology::MorphParams;
use crate::spatial::SpatialHashGrid;
use crate::world::{World, MAX_SPEED, DT};

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

const HEADER_LEN: usize = 6;
const AGENT_STRIDE: usize = 18;
const TILE_STRIDE: usize = 3;

// Header field indices
const H_AGENT_COUNT: usize = 0;
const H_GRID_SIZE: usize = 1;
const H_STEP: usize = 2;
const H_TOTAL_FOOD: usize = 3;
const H_AVG_ENERGY: usize = 4;
const H_ALPHA: usize = 5;  // renderer interpolation factor [0,1)

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
    pub fn get_state(&self) -> Vec<f32> {
        let w = &self.inner;
        let n = w.agent_count();
        let gs = w.grid_size;
        let tile_count = gs * gs;

        let total_len = HEADER_LEN + n * AGENT_STRIDE + tile_count * TILE_STRIDE;
        let mut buf = vec![0f32; total_len];

        let stats = w.get_stats();
        buf[H_AGENT_COUNT] = n as f32;
        buf[H_GRID_SIZE] = gs as f32;
        buf[H_STEP] = w.step_count as f32;
        buf[H_TOTAL_FOOD] = stats.total_food as f32;
        buf[H_AVG_ENERGY] = stats.avg_energy as f32;
        buf[H_ALPHA] = self.get_alpha();

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
            buf[off + A_BRAIN_CLUSTER] = cluster.brain_cluster_ids.get(i).copied().unwrap_or(0) as f32;
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
        }

        let tile_base = agent_base + n * AGENT_STRIDE;
        for (ti, tile) in w.tiles.iter().enumerate() {
            let off = tile_base + ti * TILE_STRIDE;
            buf[off + T_FOOD] = tile.food_units as f32;
            buf[off + T_FERTILITY] = tile.fertility as f32;
            buf[off + T_MOVE_SPEED] = tile.movement_speed as f32;
        }

        buf
    }

    /// Add food to the tile under world position (cx, cy). Clamped to MAX_FOOD_PER_TILE.
    pub fn inject_food(&mut self, cx: f32, cy: f32, amount: u32) {
        let gs = self.inner.grid_size;
        let (tx, ty) = SpatialHashGrid::tile_of(cx, cy, gs);
        let tile = &mut self.inner.tiles[ty * gs + tx];
        tile.food_units = (tile.food_units + amount).min(3);
    }

    /// Disturb food, fertility, and agent velocities within `radius` world units of (cx, cy).
    /// `intensity` in [0, 1]: 1.0 = maximum disruption.
    pub fn stir(&mut self, cx: f32, cy: f32, radius: f32, intensity: f32) {
        let gs = self.inner.grid_size;
        let world_size = gs as f32;
        let r_cells = radius.ceil() as i32;
        let cx_tile = cx.floor() as i32;
        let cy_tile = cy.floor() as i32;
        let half = world_size * 0.5;

        // Disturb tiles within radius
        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx_tile + dx).rem_euclid(gs as i32) as usize;
                let ty = (cy_tile + dy).rem_euclid(gs as i32) as usize;
                let tile_cx = tx as f32 + 0.5;
                let tile_cy = ty as f32 + 0.5;
                let mut ddx = tile_cx - cx;
                let mut ddy = tile_cy - cy;
                if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
                if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }
                let dist = (ddx * ddx + ddy * ddy).sqrt();
                if dist > radius { continue; }

                let tile = &mut self.inner.tiles[ty * gs + tx];
                let drain = (tile.food_units as f32 * intensity) as u32;
                tile.food_units = tile.food_units.saturating_sub(drain);
                tile.fertility *= (1.0 - intensity * 0.5) as f64;
                tile.fertility = tile.fertility.max(0.0);
            }
        }

        // Apply velocity impulse to agents within radius (scatter outward from stir center)
        let impulse = intensity * 6.0;
        let n = self.inner.agent_count();
        for i in 0..n {
            let ax = self.inner.pos_x[i];
            let ay = self.inner.pos_y[i];
            let mut ddx = ax - cx;
            let mut ddy = ay - cy;
            if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
            if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }
            let dist = (ddx * ddx + ddy * ddy).sqrt();
            if dist > radius || dist < 0.001 { continue; }

            let falloff = (1.0 - dist / radius) * impulse;
            self.inner.vel_x[i] += ddx / dist * falloff;
            self.inner.vel_y[i] += ddy / dist * falloff;

            // Clamp to 2× max speed so stir can't fling agents unreasonably far
            let max = self.inner.genome[i].traits.speed as f32 * MAX_SPEED * 2.0;
            let cur = (self.inner.vel_x[i].powi(2) + self.inner.vel_y[i].powi(2)).sqrt();
            if cur > max && cur > 0.0 {
                self.inner.vel_x[i] = self.inner.vel_x[i] / cur * max;
                self.inner.vel_y[i] = self.inner.vel_y[i] / cur * max;
            }
        }
    }

    /// Spawn `count` agents near world position (cx, cy).
    pub fn pour_agents(&mut self, cx: f32, cy: f32, count: usize) {
        self.inner.pour_agents(cx, cy, count);
    }

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
}

/// Trait bounds [lo, hi] × 9 in Traits field order (vision, speed, metabolism,
/// energy_capacity, mutation_rate, reproduction_cost, attack, defense,
/// aggression). Single source for JS bar normalization — mirrors
/// genome.rs::Traits::generate ranges.
#[wasm_bindgen]
pub fn trait_bounds() -> Vec<f32> {
    vec![
        0.5, 1.05,   // vision
        0.5, 1.0,    // speed
        0.5, 1.05,   // metabolism
        0.95, 1.05,  // energy_capacity (locked)
        0.01, 0.25,  // mutation_rate (locked)
        0.75, 1.50,  // reproduction_cost
        0.5, 1.25,   // attack
        0.5, 1.07,   // defense
        0.0, 1.05,   // aggression
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
