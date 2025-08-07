//! Index traits

use diskann_core::{DiskAnnResult, vectors::VectorId};

/// Trait for vector indices
pub trait Index<T> {
    /// Add a vector to the index
    fn add(&mut self, id: VectorId, vector: Vec<T>) -> DiskAnnResult<()>;
    
    /// Remove a vector from the index
    fn remove(&mut self, id: VectorId) -> DiskAnnResult<()>;
    
    /// Get the size of the index
    fn size(&self) -> usize;
}