//! Rolling time-series of sim-wide statistics, sampled by `World::step()`.
//!
//! The renderer's HUD shows instantaneous scalars and the genome panel keeps its
//! own short trait history, but neither can answer "when did the population
//! crash" or "what killed them". This module keeps that record engine-side so
//! every consumer — web renderer, native renderer, headless CSV dump — reads the
//! same numbers from the same sampler.
//!
//! Deaths are stored per sample interval, not cumulatively. A cumulative tally
//! plots as a monotone staircase, which hides the thing worth seeing: the step
//! where a starvation wave hit. `World` keeps its cumulative tally for the
//! summary table and this module differences it at sample time.

/// Samples retained. At `SAMPLE_INTERVAL` = 10 this covers 6000 steps.
pub const HISTORY_LEN: usize = 600;
/// Steps between samples.
pub const SAMPLE_INTERVAL: u32 = 10;
/// Death causes tracked, matching `CauseOfDeath::code()`.
pub const CAUSE_COUNT: usize = 5;
/// Floats per sample in the flat export buffer. Must match `StatSample::write_to`.
pub const SAMPLE_STRIDE: usize = 12;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StatSample {
    pub step: u32,
    pub alive: u32,
    pub total_food: u32,
    pub avg_energy: f32,
    pub median_lifespan: f32,
    /// Age of the youngest and oldest living agent, in steps.
    pub min_age: u32,
    pub max_age: u32,
    /// Deaths during this interval only, indexed by `CauseOfDeath::code()`.
    pub deaths: [u32; CAUSE_COUNT],
}

impl StatSample {
    /// Append this sample to a flat f32 buffer. Layout is positional and mirrors
    /// the field order above; `SAMPLE_STRIDE` floats are written.
    pub fn write_to(&self, out: &mut Vec<f32>) {
        out.push(self.step as f32);
        out.push(self.alive as f32);
        out.push(self.total_food as f32);
        out.push(self.avg_energy);
        out.push(self.median_lifespan);
        out.push(self.min_age as f32);
        out.push(self.max_age as f32);
        for d in self.deaths {
            out.push(d as f32);
        }
    }
}

/// Fixed-capacity ring of samples. A run left going overnight must not grow
/// memory without bound, so old samples are overwritten rather than retained.
#[derive(Debug, Clone)]
pub struct StatHistory {
    buf: Vec<StatSample>,
    /// Index the next push writes to.
    head: usize,
    len: usize,
    /// Cumulative deaths per cause as of the last sample, for interval differencing.
    last_cumulative: [u32; CAUSE_COUNT],
}

impl Default for StatHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl StatHistory {
    pub fn new() -> Self {
        Self {
            buf: vec![StatSample::default(); HISTORY_LEN],
            head: 0,
            len: 0,
            last_cumulative: [0; CAUSE_COUNT],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Difference a cumulative per-cause tally against the previous sample,
    /// yielding deaths during this interval. Saturating because callers may
    /// reset a world without resetting the history.
    pub fn interval_deaths(&mut self, cumulative: [u32; CAUSE_COUNT]) -> [u32; CAUSE_COUNT] {
        let mut out = [0u32; CAUSE_COUNT];
        for i in 0..CAUSE_COUNT {
            out[i] = cumulative[i].saturating_sub(self.last_cumulative[i]);
        }
        self.last_cumulative = cumulative;
        out
    }

    pub fn push(&mut self, sample: StatSample) {
        self.buf[self.head] = sample;
        self.head = (self.head + 1) % HISTORY_LEN;
        if self.len < HISTORY_LEN {
            self.len += 1;
        }
    }

    /// Samples oldest first. Handles the wrap so consumers never see the ring.
    pub fn iter_chrono(&self) -> impl Iterator<Item = &StatSample> {
        let start = if self.len < HISTORY_LEN {
            0
        } else {
            self.head
        };
        (0..self.len).map(move |i| &self.buf[(start + i) % HISTORY_LEN])
    }

    /// Most recent sample, or `None` before the first one is taken.
    pub fn latest(&self) -> Option<&StatSample> {
        if self.len == 0 {
            return None;
        }
        Some(&self.buf[(self.head + HISTORY_LEN - 1) % HISTORY_LEN])
    }

    /// Flat f32 buffer, chronological, `len() * SAMPLE_STRIDE` long.
    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len * SAMPLE_STRIDE);
        for s in self.iter_chrono() {
            s.write_to(&mut out);
        }
        out
    }

    /// Peak living population over the retained window.
    pub fn peak_alive(&self) -> u32 {
        self.iter_chrono().map(|s| s.alive).max().unwrap_or(0)
    }

    /// CSV with a header row, for headless runs and cross-build diffing.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "step,alive,total_food,avg_energy,median_lifespan,min_age,max_age,\
             deaths_starvation,deaths_old_age,deaths_combat,deaths_eaten,deaths_smitten\n",
        );
        for s in self.iter_chrono() {
            out.push_str(&format!(
                "{},{},{},{:.4},{:.2},{},{},{},{},{},{},{}\n",
                s.step,
                s.alive,
                s.total_food,
                s.avg_energy,
                s.median_lifespan,
                s.min_age,
                s.max_age,
                s.deaths[0],
                s.deaths[1],
                s.deaths[2],
                s.deaths[3],
                s.deaths[4],
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(step: u32, alive: u32) -> StatSample {
        StatSample { step, alive, ..Default::default() }
    }

    #[test]
    fn empty_history_has_no_samples() {
        let h = StatHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.iter_chrono().count(), 0);
        assert!(h.latest().is_none());
        assert_eq!(h.to_flat().len(), 0);
    }

    #[test]
    fn chronological_before_wrap() {
        let mut h = StatHistory::new();
        for i in 0..5 {
            h.push(sample(i * 10, i));
        }
        let steps: Vec<u32> = h.iter_chrono().map(|s| s.step).collect();
        assert_eq!(steps, vec![0, 10, 20, 30, 40]);
        assert_eq!(h.latest().unwrap().step, 40);
    }

    #[test]
    fn chronological_across_wrap() {
        let mut h = StatHistory::new();
        // 1.5 rings' worth: the oldest retained sample is HISTORY_LEN back.
        let total = HISTORY_LEN + HISTORY_LEN / 2;
        for i in 0..total {
            h.push(sample(i as u32, i as u32));
        }
        assert_eq!(h.len(), HISTORY_LEN);
        let steps: Vec<u32> = h.iter_chrono().map(|s| s.step).collect();
        assert_eq!(steps.len(), HISTORY_LEN);
        assert_eq!(steps[0], (total - HISTORY_LEN) as u32);
        assert_eq!(*steps.last().unwrap(), (total - 1) as u32);
        // Strictly increasing — the wrap must not reorder.
        assert!(steps.windows(2).all(|w| w[1] == w[0] + 1));
        assert_eq!(h.latest().unwrap().step, (total - 1) as u32);
    }

    #[test]
    fn interval_deaths_difference_cumulative() {
        let mut h = StatHistory::new();
        assert_eq!(h.interval_deaths([3, 0, 1, 0, 0]), [3, 0, 1, 0, 0]);
        assert_eq!(h.interval_deaths([5, 2, 1, 0, 4]), [2, 2, 0, 0, 4]);
        assert_eq!(h.interval_deaths([5, 2, 1, 0, 4]), [0, 0, 0, 0, 0]);
    }

    #[test]
    fn interval_deaths_sum_to_cumulative() {
        let mut h = StatHistory::new();
        let tallies = [[1, 0, 0, 0, 0], [4, 1, 0, 0, 2], [4, 3, 2, 0, 2], [9, 3, 2, 5, 7]];
        let mut summed = [0u32; CAUSE_COUNT];
        for t in tallies {
            let interval = h.interval_deaths(t);
            for i in 0..CAUSE_COUNT {
                summed[i] += interval[i];
            }
        }
        assert_eq!(summed, tallies[tallies.len() - 1]);
    }

    #[test]
    fn flat_buffer_stride_matches_constant() {
        let mut h = StatHistory::new();
        for i in 0..7 {
            h.push(sample(i, i));
        }
        let flat = h.to_flat();
        assert_eq!(flat.len(), 7 * SAMPLE_STRIDE);
        // First field of each record is the step.
        for i in 0..7 {
            assert_eq!(flat[i * SAMPLE_STRIDE] as u32, i as u32);
        }
    }

    #[test]
    fn peak_alive_tracks_window_max() {
        let mut h = StatHistory::new();
        for (i, alive) in [4u32, 19, 7, 12].iter().enumerate() {
            h.push(sample(i as u32, *alive));
        }
        assert_eq!(h.peak_alive(), 19);
    }

    #[test]
    fn csv_has_header_and_one_row_per_sample() {
        let mut h = StatHistory::new();
        for i in 0..3 {
            h.push(sample(i, i));
        }
        let csv = h.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("step,alive"));
        assert_eq!(lines[0].split(',').count(), SAMPLE_STRIDE);
    }
}
