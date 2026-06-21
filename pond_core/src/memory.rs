const WINDOW: usize = 10;

/// Short-term memory per agent. Logs the last WINDOW actions and tracks successful
/// energy-gain events (EAT with energy_delta > metabolism * SUCCESS_SCALAR).
/// At reproduction, success_count suppresses effective_mutation_rate in the offspring.
/// Not inherited — offspring start with empty memory.
#[derive(Debug, Clone)]
pub struct AgentMemory {
    pub success_count: u32,
    ring: [u8; WINDOW],
    head: usize,
    len: usize,
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentMemory {
    pub fn new() -> Self {
        Self { success_count: 0, ring: [0u8; WINDOW], head: 0, len: 0 }
    }

    /// Push an action index into the ring buffer.
    pub fn record_action(&mut self, action: u8) {
        self.ring[self.head] = action;
        self.head = (self.head + 1) % WINDOW;
        if self.len < WINDOW { self.len += 1; }
    }

    /// Call after a successful EAT (energy_delta > metabolism * SUCCESS_SCALAR).
    pub fn record_success(&mut self, action: u8) {
        self.record_action(action);
        self.success_count += 1;
    }

    /// Returns the last `n` actions (oldest first), up to what's been recorded.
    pub fn recent_actions(&self, n: usize) -> Vec<u8> {
        let take = n.min(self.len);
        let mut out = Vec::with_capacity(take);
        for i in 0..take {
            let pos = (self.head + WINDOW - take + i) % WINDOW;
            out.push(self.ring[pos]);
        }
        out
    }

    /// Suppression factor: 1 / (1 + success_count * k). k=0.05 default.
    /// Returns value in (0, 1]. Higher success_count → lower mutation rate in offspring.
    pub fn suppression(&self, k: f32) -> f32 {
        1.0 / (1.0 + self.success_count as f32 * k)
    }
}

pub const SUCCESS_SCALAR: f64 = 1.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_memory_empty() {
        let m = AgentMemory::new();
        assert_eq!(m.success_count, 0);
        assert_eq!(m.recent_actions(10).len(), 0);
    }

    #[test]
    fn record_action_fills_ring() {
        let mut m = AgentMemory::new();
        for i in 0u8..12 {
            m.record_action(i);
        }
        assert_eq!(m.recent_actions(10).len(), 10);
        // Last 10 actions should be 2..=11
        let recent = m.recent_actions(10);
        assert_eq!(recent[9], 11);
        assert_eq!(recent[0], 2);
    }

    #[test]
    fn suppression_decreases_with_success() {
        let mut m = AgentMemory::new();
        let s0 = m.suppression(0.05);
        m.record_success(2);
        m.record_success(2);
        let s2 = m.suppression(0.05);
        assert!(s2 < s0);
        assert!((s0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn suppression_bounded() {
        let mut m = AgentMemory::new();
        for _ in 0..1000 {
            m.record_success(2);
        }
        let s = m.suppression(0.05);
        assert!(s > 0.0);
        assert!(s < 1.0);
    }
}
