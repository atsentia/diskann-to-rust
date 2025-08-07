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

/// Scratch buffer for zero-allocation search operations
#[derive(Debug)]
pub struct SearchBuffer {
    /// Candidate buffer for search operations
    pub candidates: Vec<VectorId>,
    /// Distance buffer for storing distances during search
    pub distances: Vec<f32>,
    /// Visited set buffer
    pub visited: Vec<bool>,
}

impl SearchBuffer {
    /// Create a new search buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            distances: Vec::with_capacity(capacity),
            visited: Vec::with_capacity(capacity),
        }
    }

    /// Clear all buffers for reuse
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.distances.clear();
        self.visited.clear();
    }

    /// Resize buffers to accommodate a specific number of nodes
    pub fn resize_for_nodes(&mut self, num_nodes: usize) {
        self.visited.resize(num_nodes, false);
        // Clear visited flags
        self.visited.fill(false);
    }
}

/// Trait for approximate nearest neighbor search
pub trait Search<T> {
    /// Search for k nearest neighbors
    fn search(&self, query: &[T], k: usize) -> DiskAnnResult<Vec<SearchResult>>;

    /// Search for k nearest neighbors with beam search
    fn search_with_beam(&self, query: &[T], k: usize, beam_width: usize) -> DiskAnnResult<Vec<SearchResult>> {
        // Default implementation falls back to regular search
        let _ = beam_width;
        self.search(query, k)
    }

    /// Zero-allocation search using provided scratch buffer
    fn search_with_buffer(
        &self,
        query: &[T],
        k: usize,
        beam_width: usize,
        buffer: &mut SearchBuffer,
    ) -> DiskAnnResult<Vec<SearchResult>> {
        // Default implementation ignores buffer and calls regular search
        let _ = buffer;
        self.search_with_beam(query, k, beam_width)
    }
}