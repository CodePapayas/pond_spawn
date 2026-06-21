pub mod biome;
pub mod brain;
pub mod cluster;
pub mod genome;
pub mod memory;
pub mod spatial;
pub mod world;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use biome::BiomeTile;
pub use brain::forward as brain_forward;
pub use cluster::ClusterState;
pub use genome::{Genome, Traits};
pub use memory::AgentMemory;
pub use spatial::SpatialHashGrid;
pub use world::{SimStats, World, DT, MAX_SPEED};
