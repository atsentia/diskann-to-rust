//! Search traits

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use diskann_core::{DiskAnnResult, vectors::VectorId};

/// Search result containing vector ID and distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector identifier
    pub id: VectorId,
    /// Distance to query vector
    pub distance: f32,
}

/// Trait for approximate nearest neighbor search
pub trait Search<T> {
    /// Search for k nearest neighbors
    fn search(&self, query: &[T], k: usize) -> DiskAnnResult<Vec<SearchResult>>;
}