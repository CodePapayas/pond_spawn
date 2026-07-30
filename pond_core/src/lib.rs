pub mod biome;
pub mod brain;
pub mod brain_cluster;
pub mod cluster;
pub mod genome;
pub mod memory;
pub mod morphology;
pub mod naming;
pub mod schema;
pub mod spatial;
pub mod species;
pub mod stats;
pub mod world;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use biome::BiomeTile;
pub use brain::forward as brain_forward;
pub use brain_cluster::BrainClusters;
pub use cluster::ClusterState;
pub use genome::{Genome, Traits};
pub use memory::AgentMemory;
pub use morphology::MorphParams;
pub use naming::Name;
pub use schema::SCHEMA_VERSION;
pub use spatial::SpatialHashGrid;
pub use species::{Species, SpeciesEvent, SpeciesRegistry};
pub use stats::{StatHistory, StatSample};
pub use world::{SimStats, World, DT, MAX_SPEED};
