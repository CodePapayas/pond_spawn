/// Tile-aligned spatial hash for continuous-space agent lookup.
///
/// Cell size = 1 world unit = 1 biome tile (world spans [0, grid_size) per axis).
/// Buckets are a flat Vec indexed by `ty * grid_size + tx`. Wraps toroidally.
use smallvec::SmallVec;

pub struct SpatialHashGrid {
    buckets: Vec<SmallVec<[usize; 4]>>,
    pub grid_size: usize,
}

impl SpatialHashGrid {
    pub fn new(grid_size: usize) -> Self {
        Self {
            buckets: vec![SmallVec::new(); grid_size * grid_size],
            grid_size,
        }
    }

    /// Tile indices for a continuous position. Clamps to [0, grid_size-1].
    #[inline]
    pub fn tile_of(x: f32, y: f32, grid_size: usize) -> (usize, usize) {
        let tx = (x.floor() as usize).min(grid_size.saturating_sub(1));
        let ty = (y.floor() as usize).min(grid_size.saturating_sub(1));
        (tx, ty)
    }

    #[inline]
    fn flat(&self, tx: usize, ty: usize) -> usize {
        ty * self.grid_size + tx
    }

    pub fn agents_at_tile(&self, tx: usize, ty: usize) -> &[usize] {
        &self.buckets[self.flat(tx, ty)]
    }

    pub fn count_at_tile(&self, tx: usize, ty: usize) -> usize {
        self.buckets[self.flat(tx, ty)].len()
    }

    /// Rebuild every bucket from scratch using current f32 positions.
    pub fn rebuild(&mut self, pos_x: &[f32], pos_y: &[f32]) {
        let gs = self.grid_size;
        for b in &mut self.buckets {
            b.clear();
        }
        for (idx, (&x, &y)) in pos_x.iter().zip(pos_y.iter()).enumerate() {
            let (tx, ty) = Self::tile_of(x, y, gs);
            self.buckets[ty * gs + tx].push(idx);
        }
    }

    /// All agent indices in every tile within ceil(radius) of (x, y), wrapping toroidally.
    /// Caller filters by actual Euclidean distance as needed.
    pub fn agents_near(&self, x: f32, y: f32, radius: f32) -> Vec<usize> {
        let gs = self.grid_size as i32;
        let r_cells = radius.ceil() as i32;
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;
        let mut out = Vec::new();
        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let tx = (cx + dx).rem_euclid(gs) as usize;
                let ty = (cy + dy).rem_euclid(gs) as usize;
                out.extend_from_slice(&self.buckets[ty * self.grid_size + tx]);
            }
        }
        out
    }

    /// Move one agent from its old tile to its new tile (O(bucket size)).
    pub fn move_agent(&mut self, idx: usize, old_x: f32, old_y: f32, new_x: f32, new_y: f32) {
        let gs = self.grid_size;
        let (otx, oty) = Self::tile_of(old_x, old_y, gs);
        let old_flat = self.flat(otx, oty);
        if let Some(pos) = self.buckets[old_flat].iter().position(|&i| i == idx) {
            self.buckets[old_flat].swap_remove(pos);
        }
        let (ntx, nty) = Self::tile_of(new_x, new_y, gs);
        let new_flat = nty * self.grid_size + ntx;
        self.buckets[new_flat].push(idx);
    }

    /// Remove one agent from its bucket.
    pub fn remove_agent(&mut self, idx: usize, x: f32, y: f32) {
        let (tx, ty) = Self::tile_of(x, y, self.grid_size);
        let flat = self.flat(tx, ty);
        if let Some(pos) = self.buckets[flat].iter().position(|&i| i == idx) {
            self.buckets[flat].swap_remove(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_of_basic() {
        assert_eq!(SpatialHashGrid::tile_of(1.7, 0.3, 8), (1, 0));
        assert_eq!(SpatialHashGrid::tile_of(0.0, 0.0, 8), (0, 0));
        assert_eq!(SpatialHashGrid::tile_of(7.99, 7.99, 8), (7, 7));
        assert_eq!(SpatialHashGrid::tile_of(8.5, 8.5, 8), (7, 7)); // clamped
    }

    #[test]
    fn rebuild_and_lookup() {
        let mut grid = SpatialHashGrid::new(4);
        let xs = vec![0.5f32, 1.5, 2.5];
        let ys = vec![0.5f32, 0.5, 0.5];
        grid.rebuild(&xs, &ys);
        assert_eq!(grid.agents_at_tile(0, 0), &[0]);
        assert_eq!(grid.agents_at_tile(1, 0), &[1]);
        assert_eq!(grid.agents_at_tile(2, 0), &[2]);
    }

    #[test]
    fn agents_near_covers_radius() {
        let mut grid = SpatialHashGrid::new(8);
        // Agent 0 at tile (4,4), agent 1 at tile (6,4) — 2 tiles apart
        let xs = vec![4.5f32, 6.5];
        let ys = vec![4.5f32, 4.5];
        grid.rebuild(&xs, &ys);
        let near = grid.agents_near(4.5, 4.5, 2.5);
        assert!(near.contains(&0));
        assert!(near.contains(&1));
    }

    #[test]
    fn move_agent_updates_bucket() {
        let mut grid = SpatialHashGrid::new(4);
        let xs = vec![0.5f32];
        let ys = vec![0.5f32];
        grid.rebuild(&xs, &ys);
        grid.move_agent(0, 0.5, 0.5, 2.5, 2.5);
        assert!(grid.agents_at_tile(0, 0).is_empty());
        assert_eq!(grid.agents_at_tile(2, 2), &[0]);
    }
}
