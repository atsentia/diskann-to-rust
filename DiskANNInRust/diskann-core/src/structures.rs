//! Basic data structures for DiskANN

use crate::vectors::{Vector, VectorId};
use serde::{Deserialize, Serialize};

/// Graph node representing a vector in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Vector identifier
    pub id: VectorId,
    /// The actual vector data
    pub vector: Vector,
    /// Neighbors in the graph
    pub neighbors: Vec<VectorId>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: VectorId, vector: Vector) -> Self {
        Self {
            id,
            vector,
            neighbors: Vec::new(),
        }
    }
}