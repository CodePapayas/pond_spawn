use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use crate::biome::{BiomeTile, MAX_FOOD_PER_TILE};
use crate::brain::{forward as brain_forward, sigmoid_outputs};
use crate::cluster::ClusterState;
use crate::genome::Genome;
use crate::memory::{AgentMemory, SUCCESS_SCALAR};
use crate::spatial::SpatialHashGrid;

// ── Steering output indices ───────────────────────────────────────────────────
const OUT_SEEK: usize = 0;       // force weight toward nearest food
const OUT_WANDER: usize = 1;     // random perturbation weight
const OUT_SEPARATE: usize = 2;   // repulsion from nearby agents
// OUT_FLEE = 3  (dormant — future threat system)
const OUT_EAT: usize = 4;        // discrete trigger gate
const OUT_REPRODUCE: usize = 5;  // discrete trigger gate
// OUT_ATTACK = 6  (dormant — routes through passive combat only)
const OUT_SLEEP: usize = 7;      // discrete trigger gate

// Memory record tags (match the output indices for meaningful tracking)
const TAG_EAT: u8 = OUT_EAT as u8;
const TAG_REPRODUCE: u8 = OUT_REPRODUCE as u8;
const TAG_SLEEP: u8 = OUT_SLEEP as u8;

// ── Physics constants ─────────────────────────────────────────────────────────
pub const DT: f32 = 1.0 / 20.0;        // 20 Hz fixed timestep (50 ms per tick)
pub const MAX_SPEED: f32 = 3.0;         // tiles/sec at speed_trait = 1.0
const MAX_FORCE: f32 = 8.0;             // steering acceleration magnitude cap
const WANDER_FORCE: f32 = 2.5;          // wander perturbation strength
const SEPARATION_RADIUS: f32 = 1.2;     // repulsion radius in tiles
const VISION_SCALE: f32 = 3.0;          // vision_trait × VISION_SCALE = radius in tiles
const MOVE_COST: f64 = 0.12;            // energy per tile traveled × metabolism

// ── Economy constants ─────────────────────────────────────────────────────────
const MATURITY_AGE: u32 = 100;
const CHILDHOOD_TICKS: u32 = 50;
const BIRTH_FAIL_CHANCE: f64 = 0.02;
const FAIL_COUNTS_CHANCE: f64 = 0.20;
const FOOD_ENERGY: f64 = 33.3;
const MAX_ENERGY_BASE: f64 = 100.0;
const BIRTH_ENERGY_TRANSFER: f64 = 0.40;

// ── Death ─────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CauseOfDeath {
    Starvation,
    OldAge,
    KilledInCombat,
    EatenAlive,
}

// ── Pending offspring ─────────────────────────────────────────────────────────
struct PendingAgent {
    genome: Genome,
    energy: f64,
    x: f32,
    y: f32,
    parent_defense: f64,
    parent_id: u32,
}

// ── Public stats ──────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct SimStats {
    pub step: u32,
    pub alive_agents: usize,
    pub total_food: u32,
    pub avg_energy: f64,
    pub median_lifespan: f64,
    pub deaths: HashMap<String, u32>,
}

// ── World ─────────────────────────────────────────────────────────────────────
pub struct World {
    pub grid_size: usize,
    pub step_count: u32,
    rng: ChaCha8Rng,
    death_range_pool: Vec<u32>,

    // Grid: flat, indexed y * grid_size + x (tile system unchanged)
    pub tiles: Vec<BiomeTile>,

    // SoA agent arrays — all same length, same index = same agent
    pub ids: Vec<u32>,
    pub energy: Vec<f64>,
    pub age: Vec<u32>,
    pub pos_x: Vec<f32>,           // continuous world x in [0, grid_size)
    pub pos_y: Vec<f32>,           // continuous world y in [0, grid_size)
    pub vel_x: Vec<f32>,           // velocity x in tiles/sec
    pub vel_y: Vec<f32>,           // velocity y in tiles/sec
    pub prev_x: Vec<f32>,          // position at previous tick (for renderer interpolation)
    pub prev_y: Vec<f32>,
    pub death_age: Vec<u32>,
    pub genome: Vec<Genome>,
    pub memory: Vec<AgentMemory>,
    parent_defense_bonus: Vec<f64>,
    parent_id: Vec<Option<u32>>,
    cause_of_death: Vec<Option<CauseOfDeath>>,
    offspring_count: Vec<u32>,
    max_offspring: Vec<u32>,
    last_reproduced_age: Vec<Option<u32>>,
    reproduction_cooldown: Vec<u32>,

    next_id: u32,
    pub lifespans: Vec<u32>,
    death_tally: HashMap<CauseOfDeath, u32>,
    spatial: SpatialHashGrid,
    pub cluster: ClusterState,

    // Pre-allocated scratch buffers — cleared and reused each step
    scratch_acting: Vec<usize>,
    scratch_dead: Vec<usize>,
    scratch_perceptions: Vec<[f32; 5]>,
    scratch_outputs: Vec<[f32; 8]>,  // sigmoid-gated brain outputs per acting agent
}

impl World {
    pub fn new(grid_size: usize, population: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let death_range_pool = create_death_range(&mut rng);
        let tiles = init_grid(grid_size, &mut rng);
        let spatial = SpatialHashGrid::new(grid_size);

        let mut world = Self {
            grid_size,
            step_count: 0,
            rng,
            death_range_pool,
            tiles,
            ids: Vec::new(),
            energy: Vec::new(),
            age: Vec::new(),
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            prev_x: Vec::new(),
            prev_y: Vec::new(),
            death_age: Vec::new(),
            genome: Vec::new(),
            memory: Vec::new(),
            parent_defense_bonus: Vec::new(),
            parent_id: Vec::new(),
            cause_of_death: Vec::new(),
            offspring_count: Vec::new(),
            max_offspring: Vec::new(),
            last_reproduced_age: Vec::new(),
            reproduction_cooldown: Vec::new(),
            next_id: 0,
            lifespans: Vec::new(),
            death_tally: HashMap::new(),
            spatial,
            cluster: ClusterState::new(),
            scratch_acting: Vec::new(),
            scratch_dead: Vec::new(),
            scratch_perceptions: Vec::new(),
            scratch_outputs: Vec::new(),
        };

        world.spawn_agents(population);
        world.spatial.rebuild(&world.pos_x, &world.pos_y);
        world
    }

    pub fn agent_count(&self) -> usize {
        self.ids.len()
    }

    pub fn get_stats(&self) -> SimStats {
        let n = self.ids.len();
        let total_food: u32 = self.tiles.iter().map(|t| t.food_units).sum();
        let avg_energy = if n > 0 {
            self.energy.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let median_lifespan = median(&self.lifespans);
        let mut deaths = HashMap::new();
        for (cause, &count) in &self.death_tally {
            deaths.insert(format!("{:?}", cause), count);
        }
        SimStats {
            step: self.step_count,
            alive_agents: n,
            total_food,
            avg_energy,
            median_lifespan,
            deaths,
        }
    }

    /// Spawn agents at random positions within `radius` tiles of (cx, cy).
    pub fn pour_agents(&mut self, cx: f32, cy: f32, count: usize) {
        let world_size = self.grid_size as f32;
        for _ in 0..count {
            let angle = self.rng.gen::<f32>() * TAU;
            let radius: f32 = self.rng.gen_range(0.3..2.0);
            let x = (cx + radius * angle.cos()).rem_euclid(world_size);
            let y = (cy + radius * angle.sin()).rem_euclid(world_size);
            let genome = Genome::generate(&mut self.rng);
            self.push_agent(x, y, 50.0, genome, 0.0, None);
        }
        self.spatial.rebuild(&self.pos_x, &self.pos_y);
    }

    // ── Step loop ─────────────────────────────────────────────────────────────

    pub fn step(&mut self) {
        self.step_count += 1;

        // Phase 1: food regen
        self.tick_food_regen();

        // Phase 2: age / passive metabolism drain / natural death
        self.scratch_dead.clear();
        self.tick_age_and_metabolism_scratch();

        // Rebuild spatial after natural deaths (dead agents still in arrays until reap)
        self.spatial.rebuild(&self.pos_x, &self.pos_y);

        // Phase 3: collect alive agents
        self.scratch_acting.clear();
        self.partition_agents_scratch();

        // Phase 4: perception → 5-input vector per acting agent
        let acting_len = self.scratch_acting.len();
        self.scratch_perceptions.resize(acting_len, [0f32; 5]);
        self.perceive_all();

        // Phase 5: brain forward → 8 sigmoid outputs per acting agent
        self.scratch_outputs.resize(acting_len, [0f32; 8]);
        self.steer_all();

        // Phase 6: integrate physics + fire discrete triggers
        let mut offspring: Vec<PendingAgent> = Vec::new();
        for slot in 0..acting_len {
            let idx = self.scratch_acting[slot];
            if self.cause_of_death[idx].is_some() { continue; }
            let perception = self.scratch_perceptions[slot];
            let outputs = self.scratch_outputs[slot];
            if let Some(child) = self.integrate_agent(idx, perception, outputs) {
                offspring.push(child);
            }
            if self.energy[idx] <= 0.0 && self.cause_of_death[idx].is_none() {
                self.cause_of_death[idx] = Some(CauseOfDeath::Starvation);
                self.lifespans.push(self.age[idx]);
                self.scratch_dead.push(idx);
            }
        }

        // Phase 7: passive combat per tile
        self.resolve_combat_spatial();

        // Phase 8: add offspring
        self.spawn_offspring(offspring);

        // Phase 9: reap dead
        let dead: Vec<usize> = self.scratch_dead.clone();
        self.reap_dead(dead);

        // Rebuild spatial for next step
        self.spatial.rebuild(&self.pos_x, &self.pos_y);

        // Phase 10: dual k-means clustering every 50 steps
        if self.step_count % 50 == 0 && !self.genome.is_empty() {
            self.cluster = ClusterState::run(&self.genome, 6, 24, self.step_count);
        }
    }

    // ── Phase implementations ─────────────────────────────────────────────────

    fn tick_food_regen(&mut self) {
        for tile in &mut self.tiles {
            if tile.food_units < MAX_FOOD_PER_TILE {
                let rate = tile.regen_rate();
                if self.rng.gen::<f64>() < rate {
                    tile.food_units += 1;
                }
            }
        }
    }

    fn tick_age_and_metabolism_scratch(&mut self) {
        let n = self.ids.len();
        for i in 0..n {
            if self.cause_of_death[i].is_some() { continue; }
            self.age[i] += 1;

            if self.age[i] >= self.death_age[i] {
                self.cause_of_death[i] = Some(CauseOfDeath::OldAge);
                self.lifespans.push(self.age[i]);
                self.scratch_dead.push(i);
                continue;
            }

            let metabolism = self.genome[i].traits.metabolism;
            self.energy[i] -= 0.1 * metabolism;

            if self.energy[i] <= 0.0 {
                self.cause_of_death[i] = Some(CauseOfDeath::Starvation);
                self.lifespans.push(self.age[i]);
                self.scratch_dead.push(i);
            }
        }
    }

    fn partition_agents_scratch(&mut self) {
        for i in 0..self.ids.len() {
            if self.cause_of_death[i].is_none() {
                self.scratch_acting.push(i);
            }
        }
    }

    fn perceive_all(&mut self) {
        for slot in 0..self.scratch_acting.len() {
            let idx = self.scratch_acting[slot];
            self.scratch_perceptions[slot] = self.perceive(idx);
        }
    }

    fn steer_all(&mut self) {
        for slot in 0..self.scratch_acting.len() {
            let idx = self.scratch_acting[slot];
            let weights = self.genome[idx].weights_array();
            let p = self.scratch_perceptions[slot];
            let logits = brain_forward(weights, p);
            self.scratch_outputs[slot] = sigmoid_outputs(logits);
        }
    }

    /// Build 5-input perception vector for one agent.
    fn perceive(&self, idx: usize) -> [f32; 5] {
        let px = self.pos_x[idx];
        let py = self.pos_y[idx];
        let vx = self.vel_x[idx];
        let vy = self.vel_y[idx];
        let vision = self.genome[idx].traits.vision as f32;
        let speed_trait = self.genome[idx].traits.speed as f32;
        let vision_radius = vision * VISION_SCALE;
        let world_size = self.grid_size as f32;

        // [0] energy
        let energy_norm = (self.energy[idx] / 100.0).clamp(0.0, 1.0) as f32;

        // [1,2] nearest food tile distance + angle relative to velocity
        let (food_dist_norm, food_angle_norm) =
            self.nearest_food_inputs(idx, px, py, vx, vy, vision_radius, world_size);

        // [3] agent density within separation radius
        let nearby = self.spatial.agents_near(px, py, SEPARATION_RADIUS + 0.5);
        let neighbor_count = nearby.iter()
            .filter(|&&i| i != idx && self.cause_of_death[i].is_none())
            .count();
        let agent_density_norm = (neighbor_count as f32 / 8.0).clamp(0.0, 1.0);

        // [4] current speed normalized to max
        let cur_speed = (vx * vx + vy * vy).sqrt();
        let max_speed = speed_trait * MAX_SPEED;
        let speed_norm = if max_speed > 0.0 { (cur_speed / max_speed).clamp(0.0, 1.0) } else { 0.0 };

        [energy_norm, food_dist_norm, food_angle_norm, agent_density_norm, speed_norm]
    }

    fn nearest_food_inputs(
        &self,
        _idx: usize,
        px: f32, py: f32,
        vx: f32, vy: f32,
        vision_radius: f32,
        world_size: f32,
    ) -> (f32, f32) {
        let gs = self.grid_size;
        let r_cells = vision_radius.ceil() as i32;
        let cx = px.floor() as i32;
        let cy = py.floor() as i32;

        let mut nearest_dist_sq = f32::MAX;
        let mut nearest_ddx = 0.0f32;
        let mut nearest_ddy = 0.0f32;
        let mut found = false;

        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx + dx).rem_euclid(gs as i32) as usize;
                let ty = (cy + dy).rem_euclid(gs as i32) as usize;
                if self.tiles[ty * gs + tx].food_units == 0 { continue; }

                // Food tile center, wrapped delta
                let fx = tx as f32 + 0.5;
                let fy = ty as f32 + 0.5;
                let mut ddx = fx - px;
                let mut ddy = fy - py;
                let half = world_size * 0.5;
                if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
                if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }

                let dist_sq = ddx * ddx + ddy * ddy;
                if dist_sq < nearest_dist_sq {
                    nearest_dist_sq = dist_sq;
                    nearest_ddx = ddx;
                    nearest_ddy = ddy;
                    found = true;
                }
            }
        }

        if !found {
            return (1.0, 0.0);
        }
        let dist = nearest_dist_sq.sqrt();
        if dist > vision_radius {
            return (1.0, 0.0);
        }

        let food_dist_norm = (dist / vision_radius).clamp(0.0, 1.0);

        let cur_speed = (vx * vx + vy * vy).sqrt();
        let food_angle_norm = if cur_speed > 0.01 {
            let food_angle = nearest_ddy.atan2(nearest_ddx);
            let vel_angle = vy.atan2(vx);
            let mut rel = food_angle - vel_angle;
            while rel > std::f32::consts::PI { rel -= TAU; }
            while rel < -std::f32::consts::PI { rel += TAU; }
            rel / std::f32::consts::PI
        } else {
            0.0
        };

        (food_dist_norm, food_angle_norm)
    }

    /// Apply steering forces, integrate position, fire discrete triggers.
    /// Returns Some(PendingAgent) if reproduction fires.
    fn integrate_agent(
        &mut self,
        idx: usize,
        perception: [f32; 5],
        outputs: [f32; 8],
    ) -> Option<PendingAgent> {
        let px = self.pos_x[idx];
        let py = self.pos_y[idx];
        let vx = self.vel_x[idx];
        let vy = self.vel_y[idx];
        let world_size = self.grid_size as f32;
        let speed_trait = self.genome[idx].traits.speed as f32;
        let metabolism = self.genome[idx].traits.metabolism;
        let vision = self.genome[idx].traits.vision as f32;
        let vision_radius = vision * VISION_SCALE;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;

        // Seek food (if food visible: perception[1] < 1.0)
        if perception[1] < 1.0 {
            if let Some((sdx, sdy)) = self.seek_food_dir(px, py, vision_radius, world_size) {
                fx += sdx * outputs[OUT_SEEK] * MAX_FORCE;
                fy += sdy * outputs[OUT_SEEK] * MAX_FORCE;
            }
        }

        // Wander — random perturbation
        let wander_angle = self.rng.gen::<f32>() * TAU;
        fx += wander_angle.cos() * outputs[OUT_WANDER] * WANDER_FORCE;
        fy += wander_angle.sin() * outputs[OUT_WANDER] * WANDER_FORCE;

        // Separation from nearby agents
        let (sx, sy) = self.separation_force(idx, px, py, world_size);
        fx += sx * outputs[OUT_SEPARATE] * MAX_FORCE;
        fy += sy * outputs[OUT_SEPARATE] * MAX_FORCE;

        // Velocity integration
        let max_speed = speed_trait * MAX_SPEED;
        let mut nvx = vx + fx * DT;
        let mut nvy = vy + fy * DT;
        let cur_speed = (nvx * nvx + nvy * nvy).sqrt();
        if cur_speed > max_speed && cur_speed > 0.0 {
            nvx = nvx / cur_speed * max_speed;
            nvy = nvy / cur_speed * max_speed;
        }

        // Save previous position for renderer interpolation
        self.prev_x[idx] = px;
        self.prev_y[idx] = py;

        // Integrate position with toroidal wrap
        let npx = (px + nvx * DT).rem_euclid(world_size);
        let npy = (py + nvy * DT).rem_euclid(world_size);

        // Incremental spatial update
        self.spatial.move_agent(idx, px, py, npx, npy);

        self.pos_x[idx] = npx;
        self.pos_y[idx] = npy;
        self.vel_x[idx] = nvx;
        self.vel_y[idx] = nvy;

        // Movement energy cost proportional to distance traveled
        let dist = (nvx * nvx + nvy * nvy).sqrt() * DT as f32;
        self.energy[idx] -= dist as f64 * metabolism * MOVE_COST;

        // Record dominant output index for memory ring buffer
        let dominant = outputs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u8)
            .unwrap_or(OUT_WANDER as u8);
        self.memory[idx].record_action(dominant);

        // Discrete triggers
        if outputs[OUT_EAT] > 0.5 {
            self.do_eat(idx);
        }
        if outputs[OUT_SLEEP] > 0.5 {
            // Give back half passive drain: net drain = 0.05 * metabolism instead of 0.1
            self.energy[idx] += 0.05 * metabolism;
            self.memory[idx].record_action(TAG_SLEEP);
        }
        if outputs[OUT_REPRODUCE] > 0.5 {
            return self.do_reproduce(idx);
        }

        None
    }

    /// Normalized direction vector toward nearest visible food tile. Returns None if no food visible.
    fn seek_food_dir(&self, px: f32, py: f32, vision_radius: f32, world_size: f32) -> Option<(f32, f32)> {
        let gs = self.grid_size;
        let r_cells = vision_radius.ceil() as i32;
        let cx = px.floor() as i32;
        let cy = py.floor() as i32;
        let half = world_size * 0.5;

        let mut nearest_dist_sq = f32::MAX;
        let mut result = None;

        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx + dx).rem_euclid(gs as i32) as usize;
                let ty = (cy + dy).rem_euclid(gs as i32) as usize;
                if self.tiles[ty * gs + tx].food_units == 0 { continue; }

                let fx = tx as f32 + 0.5;
                let fy = ty as f32 + 0.5;
                let mut ddx = fx - px;
                let mut ddy = fy - py;
                if ddx > half { ddx -= world_size; } else if ddx < -half { ddx += world_size; }
                if ddy > half { ddy -= world_size; } else if ddy < -half { ddy += world_size; }

                let dist_sq = ddx * ddx + ddy * ddy;
                if dist_sq < nearest_dist_sq && dist_sq.sqrt() <= vision_radius {
                    nearest_dist_sq = dist_sq;
                    let dist = dist_sq.sqrt().max(0.001);
                    result = Some((ddx / dist, ddy / dist));
                }
            }
        }
        result
    }

    /// Sum of repulsion vectors from all agents within SEPARATION_RADIUS.
    fn separation_force(&self, idx: usize, px: f32, py: f32, world_size: f32) -> (f32, f32) {
        let nearby = self.spatial.agents_near(px, py, SEPARATION_RADIUS + 0.5);
        let half = world_size * 0.5;
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        for other in nearby {
            if other == idx || self.cause_of_death[other].is_some() { continue; }
            let mut dx = px - self.pos_x[other];
            let mut dy = py - self.pos_y[other];
            if dx > half { dx -= world_size; } else if dx < -half { dx += world_size; }
            if dy > half { dy -= world_size; } else if dy < -half { dy += world_size; }
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < SEPARATION_RADIUS && dist > 0.001 {
                let strength = (SEPARATION_RADIUS - dist) / SEPARATION_RADIUS;
                fx += (dx / dist) * strength;
                fy += (dy / dist) * strength;
            }
        }
        (fx, fy)
    }

    fn do_eat(&mut self, idx: usize) {
        let (tx, ty) = SpatialHashGrid::tile_of(self.pos_x[idx], self.pos_y[idx], self.grid_size);
        let tile_idx = ty * self.grid_size + tx;
        if self.tiles[tile_idx].food_units == 0 { return; }
        let ec = self.genome[idx].traits.energy_capacity;
        let max_e = MAX_ENERGY_BASE * ec;
        let needed = max_e - self.energy[idx];
        if needed > 0.0 {
            let gained = FOOD_ENERGY.min(needed);
            self.energy[idx] += gained;
            self.tiles[tile_idx].food_units -= 1;
            let threshold = self.genome[idx].traits.metabolism * SUCCESS_SCALAR;
            if gained > threshold {
                self.memory[idx].record_success(TAG_EAT);
            }
        }
    }

    fn do_reproduce(&mut self, idx: usize) -> Option<PendingAgent> {
        if self.age[idx] < MATURITY_AGE { return None; }
        if self.energy[idx] < 40.0 { return None; }
        if self.offspring_count[idx] >= self.max_offspring[idx] { return None; }
        if let Some(last) = self.last_reproduced_age[idx] {
            if self.age[idx] - last < self.reproduction_cooldown[idx] { return None; }
        }

        let world_size = self.grid_size as f32;
        let repro_cost = self.genome[idx].traits.reproduction_cost;
        let cost = self.energy[idx] * 0.50 * repro_cost;
        self.energy[idx] -= cost;

        if self.rng.gen::<f64>() < BIRTH_FAIL_CHANCE {
            if self.rng.gen::<f64>() < FAIL_COUNTS_CHANCE {
                self.offspring_count[idx] += 1;
                self.last_reproduced_age[idx] = Some(self.age[idx]);
            }
            return None;
        }

        self.offspring_count[idx] += 1;
        self.last_reproduced_age[idx] = Some(self.age[idx]);

        // Spawn child nearby (within 2 tiles, random direction)
        let angle = self.rng.gen::<f32>() * TAU;
        let radius: f32 = self.rng.gen_range(0.5..2.0);
        let cx = (self.pos_x[idx] + radius * angle.cos()).rem_euclid(world_size);
        let cy = (self.pos_y[idx] + radius * angle.sin()).rem_euclid(world_size);

        let suppression = self.memory[idx].suppression(0.05);
        let child_genome = self.genome[idx].mutate(&mut self.rng, suppression);
        let parent_defense = self.genome[idx].traits.defense;
        let child_energy = cost * BIRTH_ENERGY_TRANSFER;

        self.memory[idx].record_action(TAG_REPRODUCE);

        Some(PendingAgent {
            genome: child_genome,
            energy: child_energy,
            x: cx,
            y: cy,
            parent_defense,
            parent_id: self.ids[idx],
        })
    }

    /// Passive combat resolved per tile — no HashMap alloc.
    fn resolve_combat_spatial(&mut self) {
        let gs = self.grid_size;
        for ty in 0..gs {
            for tx in 0..gs {
                let occupants: Vec<usize> = self.spatial.agents_at_tile(tx, ty)
                    .iter()
                    .copied()
                    .filter(|&i| self.cause_of_death[i].is_none())
                    .collect();
                if occupants.len() < 2 { continue; }

                for &attacker in &occupants {
                    if self.cause_of_death[attacker].is_some() { continue; }
                    if self.genome[attacker].traits.aggression < 0.80 { continue; }

                    let victim = occupants.iter()
                        .copied()
                        .find(|&i| i != attacker && self.cause_of_death[i].is_none());
                    let Some(victim) = victim else { continue };

                    let atk = self.genome[attacker].traits.attack;
                    let v_def = effective_defense(
                        self.genome[victim].traits.defense,
                        self.parent_defense_bonus[victim],
                        self.age[victim],
                    );
                    let metabolism = self.genome[attacker].traits.metabolism;
                    self.energy[attacker] -= 0.2 * metabolism;

                    if atk > v_def * 0.66 {
                        self.passive_eat(attacker, victim);
                    } else if atk > v_def * 0.33 {
                        if self.rng.gen::<f64>() >= 0.5 {
                            self.passive_eat(attacker, victim);
                        } else {
                            self.passive_eat(victim, attacker);
                        }
                    } else {
                        self.passive_eat(victim, attacker);
                    }
                }
            }
        }
    }

    fn passive_eat(&mut self, winner: usize, loser: usize) {
        let gained = self.energy[loser] * 0.125;
        let ec = self.genome[winner].traits.energy_capacity;
        self.energy[winner] = (self.energy[winner] + gained).min(MAX_ENERGY_BASE * ec);
        self.cause_of_death[loser] = Some(CauseOfDeath::KilledInCombat);
        self.energy[loser] = 0.0;
        self.lifespans.push(self.age[loser]);
        self.scratch_dead.push(loser);
    }

    fn spawn_offspring(&mut self, offspring: Vec<PendingAgent>) {
        for child in offspring {
            self.push_agent(child.x, child.y, child.energy, child.genome, child.parent_defense, Some(child.parent_id));
        }
    }

    /// Common path for adding any new agent (initial spawn, offspring, or pour_agents).
    fn push_agent(
        &mut self,
        x: f32,
        y: f32,
        energy: f64,
        genome: Genome,
        parent_defense: f64,
        parent_id: Option<u32>,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        let death_age = assign_death_age(&self.death_range_pool, &mut self.rng);
        let max_offspring = self.rng.gen_range(1u32..=10);
        let reproductive_window = death_age.saturating_sub(MATURITY_AGE).max(1);
        let cooldown = reproductive_window / max_offspring.max(1);

        // Random initial velocity — small fraction of max speed
        let angle = self.rng.gen::<f32>() * TAU;
        let speed_trait = genome.traits.speed as f32;
        let init_speed = self.rng.gen::<f32>() * speed_trait * MAX_SPEED * 0.3;

        self.ids.push(id);
        self.energy.push(energy);
        self.age.push(0);
        self.pos_x.push(x);
        self.pos_y.push(y);
        self.vel_x.push(angle.cos() * init_speed);
        self.vel_y.push(angle.sin() * init_speed);
        self.prev_x.push(x);
        self.prev_y.push(y);
        self.death_age.push(death_age);
        self.genome.push(genome);
        self.memory.push(AgentMemory::new());
        self.parent_defense_bonus.push(parent_defense);
        self.parent_id.push(parent_id);
        self.cause_of_death.push(None);
        self.offspring_count.push(0);
        self.max_offspring.push(max_offspring);
        self.last_reproduced_age.push(None);
        self.reproduction_cooldown.push(cooldown);
    }

    fn reap_dead(&mut self, mut dead: Vec<usize>) {
        dead.sort_unstable();
        dead.dedup();

        for &i in &dead {
            if let Some(cause) = &self.cause_of_death[i] {
                *self.death_tally.entry(cause.clone()).or_insert(0) += 1;
            }
        }

        dead.sort_unstable_by(|a, b| b.cmp(a));
        for &i in &dead {
            let last = self.ids.len() - 1;
            if i < last {
                self.ids.swap_remove(i);
                self.energy.swap_remove(i);
                self.age.swap_remove(i);
                self.pos_x.swap_remove(i);
                self.pos_y.swap_remove(i);
                self.vel_x.swap_remove(i);
                self.vel_y.swap_remove(i);
                self.prev_x.swap_remove(i);
                self.prev_y.swap_remove(i);
                self.death_age.swap_remove(i);
                self.genome.swap_remove(i);
                self.memory.swap_remove(i);
                self.parent_defense_bonus.swap_remove(i);
                self.parent_id.swap_remove(i);
                self.cause_of_death.swap_remove(i);
                self.offspring_count.swap_remove(i);
                self.max_offspring.swap_remove(i);
                self.last_reproduced_age.swap_remove(i);
                self.reproduction_cooldown.swap_remove(i);
            } else {
                self.ids.pop();
                self.energy.pop();
                self.age.pop();
                self.pos_x.pop();
                self.pos_y.pop();
                self.vel_x.pop();
                self.vel_y.pop();
                self.prev_x.pop();
                self.prev_y.pop();
                self.death_age.pop();
                self.genome.pop();
                self.memory.pop();
                self.parent_defense_bonus.pop();
                self.parent_id.pop();
                self.cause_of_death.pop();
                self.offspring_count.pop();
                self.max_offspring.pop();
                self.last_reproduced_age.pop();
                self.reproduction_cooldown.pop();
            }
        }
    }

    fn spawn_agents(&mut self, population: usize) {
        let world_size = self.grid_size as f32;
        for _ in 0..population {
            let x = self.rng.gen::<f32>() * world_size;
            let y = self.rng.gen::<f32>() * world_size;
            let genome = Genome::generate(&mut self.rng);
            self.push_agent(x, y, 100.0, genome, 0.0, None);
        }
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

fn effective_defense(defense: f64, parent_bonus: f64, age: u32) -> f64 {
    if age >= CHILDHOOD_TICKS || parent_bonus == 0.0 {
        return defense;
    }
    let ratio = age as f64 / CHILDHOOD_TICKS as f64;
    defense + parent_bonus * (1.0 - ratio)
}

fn init_grid(grid_size: usize, rng: &mut ChaCha8Rng) -> Vec<BiomeTile> {
    let n = grid_size * grid_size;
    let mut tiles: Vec<BiomeTile> = (0..n).map(|_| BiomeTile::generate(rng)).collect();
    assign_barren_tiles(&mut tiles, grid_size, rng);
    tiles
}

fn assign_barren_tiles(tiles: &mut Vec<BiomeTile>, grid_size: usize, rng: &mut ChaCha8Rng) {
    let total = grid_size * grid_size;
    let target_pct = rng.gen_range(0.35_f64..=0.45);
    let target = (total as f64 * target_pct) as usize;
    let num_seeds = (grid_size / 3).max(2).min(total);
    let gs = grid_size as i32;

    let mut all_idx: Vec<usize> = (0..total).collect();
    for i in 0..num_seeds {
        let j = rng.gen_range(i..total);
        all_idx.swap(i, j);
    }
    let seed_positions: Vec<(u16, u16)> = all_idx[..num_seeds]
        .iter()
        .map(|&i| ((i % grid_size) as u16, (i / grid_size) as u16))
        .collect();

    let mut barren: HashSet<(u16, u16)> = seed_positions.iter().copied().collect();
    let mut frontier: Vec<(u16, u16)> = seed_positions;
    let mut spread_prob = 0.55_f64;

    'grow: while barren.len() < target {
        if frontier.is_empty() {
            let remaining: Vec<(u16, u16)> = (0..total)
                .map(|i| ((i % grid_size) as u16, (i / grid_size) as u16))
                .filter(|p| !barren.contains(p))
                .collect();
            if remaining.is_empty() { break; }
            let seed = *remaining.choose(rng).unwrap();
            barren.insert(seed);
            frontier.push(seed);
        }
        let current: Vec<(u16, u16)> = frontier.drain(..).collect();
        for (fx, fy) in current {
            for &(dx, dy) in &[(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = (fx as i32 + dx).rem_euclid(gs) as u16;
                let ny = (fy as i32 + dy).rem_euclid(gs) as u16;
                if !barren.contains(&(nx, ny)) && rng.gen::<f64>() < spread_prob {
                    barren.insert((nx, ny));
                    frontier.push((nx, ny));
                    if barren.len() >= target { break 'grow; }
                }
            }
        }
        spread_prob = (spread_prob * 0.85).max(0.30);
    }

    for (x, y) in barren {
        tiles[y as usize * grid_size + x as usize].make_barren();
    }
}

fn create_death_range(rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut pool = Vec::with_capacity(200);
    for i in 0usize..200 {
        if i < 5 && rng.gen::<f64>() < 0.15 {
            pool.push(rng.gen_range(50u32..=150));
        } else if i > 15 && i < 20 && rng.gen::<f64>() < 0.05 {
            pool.push(rng.gen_range(200u32..=400));
        } else {
            pool.push(500 + (i as u32 + 500) / 4);
        }
    }
    pool
}

fn assign_death_age(pool: &[u32], rng: &mut ChaCha8Rng) -> u32 {
    let candidates: Vec<u32> = pool.iter().copied().filter(|&v| v > 0).collect();
    if candidates.is_empty() { return 750; }
    *candidates.choose(rng).unwrap()
}

fn median(v: &[u32]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n % 2 == 1 { s[n / 2] as f64 } else { (s[n / 2 - 1] + s[n / 2]) as f64 / 2.0 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn small_world() -> World {
        World::new(8, 30, 42)
    }

    #[test]
    fn world_initializes() {
        let w = small_world();
        assert!(w.agent_count() > 0);
        assert_eq!(w.tiles.len(), 64);
        assert_eq!(w.step_count, 0);
    }

    #[test]
    fn step_runs_without_panic() {
        let mut w = small_world();
        for _ in 0..10 {
            w.step();
            if w.agent_count() == 0 { break; }
        }
        assert!(w.step_count <= 10);
    }

    #[test]
    fn soa_arrays_same_length() {
        let mut w = small_world();
        w.step();
        let n = w.ids.len();
        assert_eq!(w.energy.len(), n);
        assert_eq!(w.age.len(), n);
        assert_eq!(w.pos_x.len(), n);
        assert_eq!(w.pos_y.len(), n);
        assert_eq!(w.vel_x.len(), n);
        assert_eq!(w.vel_y.len(), n);
        assert_eq!(w.prev_x.len(), n);
        assert_eq!(w.prev_y.len(), n);
        assert_eq!(w.genome.len(), n);
        assert_eq!(w.memory.len(), n);
        assert_eq!(w.cause_of_death.len(), n);
        assert_eq!(w.death_age.len(), n);
    }

    #[test]
    fn positions_in_bounds() {
        let mut w = small_world();
        let world_size = w.grid_size as f32;
        for _ in 0..20 {
            w.step();
        }
        for i in 0..w.agent_count() {
            assert!(w.pos_x[i] >= 0.0 && w.pos_x[i] < world_size, "pos_x[{}]={} out of bounds", i, w.pos_x[i]);
            assert!(w.pos_y[i] >= 0.0 && w.pos_y[i] < world_size, "pos_y[{}]={} out of bounds", i, w.pos_y[i]);
        }
    }

    #[test]
    fn food_regen_respects_max() {
        let mut w = small_world();
        for _ in 0..200 {
            w.tick_food_regen();
        }
        for tile in &w.tiles {
            assert!(tile.food_units <= MAX_FOOD_PER_TILE);
        }
    }

    #[test]
    fn energy_drains_each_step() {
        let mut w = World::new(6, 10, 99);
        let initial_energy: f64 = w.energy.iter().sum();
        w.step();
        let after_energy: f64 = w.energy.iter().sum();
        assert!(after_energy < initial_energy * 1.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let mut w1 = World::new(8, 20, 7);
        let mut w2 = World::new(8, 20, 7);
        for _ in 0..5 {
            w1.step();
            w2.step();
        }
        assert_eq!(w1.agent_count(), w2.agent_count());
        assert_eq!(w1.step_count, w2.step_count);
    }

    #[test]
    fn pour_agents_adds_agents() {
        let mut w = World::new(8, 5, 11);
        let before = w.agent_count();
        w.pour_agents(4.0, 4.0, 10);
        assert_eq!(w.agent_count(), before + 10);
    }
}
