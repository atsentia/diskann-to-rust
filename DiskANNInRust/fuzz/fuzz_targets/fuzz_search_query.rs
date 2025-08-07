#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use diskann_impl::VamanaIndex;
use diskann_traits::distance::EuclideanDistance;
use diskann_traits::index::Index;
use diskann_traits::search::Search;
use diskann_core::vectors::VectorId;

#[derive(Arbitrary, Debug)]
struct SearchQuery {
    // Random query vector
    query: Vec<f32>,
    // Search parameters
    k: u8,  // Number of results to find (1-255)
    beam_width: u8,  // Beam width for search (1-255)
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    // Index construction data
    vectors: Vec<Vec<f32>>,
    // Search queries to run
    queries: Vec<SearchQuery>,
}

fuzz_target!(|input: FuzzInput| {
    // Skip empty or too large inputs to avoid timeouts
    if input.vectors.is_empty() || input.vectors.len() > 1000 {
        return;
    }
    
    // Ensure all vectors have same dimension and reasonable size
    if let Some(first_vec) = input.vectors.first() {
        if first_vec.is_empty() || first_vec.len() > 1000 {
            return;
        }
        
        let dim = first_vec.len();
        let valid_vectors: Vec<_> = input.vectors.into_iter()
            .filter(|v| v.len() == dim)
            .take(100) // Limit to 100 vectors for performance
            .collect();
            
        if valid_vectors.len() < 2 {
            return;
        }
        
        // Create index
        let distance_fn = EuclideanDistance::default();
        let mut index = VamanaIndex::with_distance(distance_fn);
        
        // Insert vectors - this should not panic
        for (i, vector) in valid_vectors.iter().enumerate() {
            let vector_id = i as VectorId;
            if index.add(vector_id, vector.clone()).is_err() {
                // Add failed, skip this input
                return;
            }
        }
        
        // Run search queries
        for query in input.queries.iter().take(10) { // Limit queries
            if query.query.len() != dim {
                continue;
            }
            
            let k = std::cmp::max(1, query.k as usize);
            let k = std::cmp::min(k, valid_vectors.len());
            let beam_width = std::cmp::max(1, query.beam_width as usize);
            
            // Search should not panic, even with invalid parameters
            let _ = index.search(&query.query, k);
            let _ = index.search_with_beam(&query.query, k, beam_width);
        }
    }
});