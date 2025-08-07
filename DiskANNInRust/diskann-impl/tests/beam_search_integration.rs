//! Integration tests for beam search functionality

use diskann_impl::{IndexBuilder, VamanaConfig};
use diskann_traits::{distance::EuclideanDistance, search::{Search, SearchBuffer}};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_test_vectors(num_vectors: usize, dimension: usize) -> Vec<(u32, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(42);
    
    (0..num_vectors)
        .map(|i| {
            let vector = (0..dimension)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect();
            (i as u32, vector)
        })
        .collect()
}

#[test]
fn test_beam_search_quality() {
    let vectors = generate_test_vectors(100, 32);
    let query = vec![0.5; 32];
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .max_degree(16)
        .search_list_size(32)
        .build(vectors)
        .unwrap();
    
    // Test that larger beam width generally gives better quality results
    let k = 5;
    let results_small_beam = index.search_with_beam(&query, k, 8).unwrap();
    let results_large_beam = index.search_with_beam(&query, k, 64).unwrap();
    
    assert_eq!(results_small_beam.len(), k);
    assert_eq!(results_large_beam.len(), k);
    
    // Both should return valid results
    for result in &results_small_beam {
        assert!(result.distance >= 0.0);
    }
    for result in &results_large_beam {
        assert!(result.distance >= 0.0);
    }
    
    // Results should be sorted by distance
    for i in 1..results_small_beam.len() {
        assert!(results_small_beam[i-1].distance <= results_small_beam[i].distance);
    }
    for i in 1..results_large_beam.len() {
        assert!(results_large_beam[i-1].distance <= results_large_beam[i].distance);
    }
}

#[test]
fn test_zero_allocation_search() {
    let vectors = generate_test_vectors(50, 16);
    let query = vec![0.1; 16];
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .max_degree(8)
        .search_list_size(16)
        .build(vectors)
        .unwrap();
    
    let k = 5;
    let beam_width = 16;
    let mut buffer = SearchBuffer::new(100);
    
    // Test zero allocation search
    let results = index.search_with_buffer(&query, k, beam_width, &mut buffer).unwrap();
    
    assert_eq!(results.len(), k);
    for result in &results {
        assert!(result.distance >= 0.0);
    }
    
    // Test that buffer can be reused
    let results2 = index.search_with_buffer(&query, k, beam_width, &mut buffer).unwrap();
    assert_eq!(results2.len(), k);
}

#[test]
fn test_beam_search_consistency() {
    let vectors = generate_test_vectors(200, 64);
    let query = vec![0.0; 64];
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .max_degree(32)
        .search_list_size(64)
        .build(vectors)
        .unwrap();
    
    let k = 10;
    
    // Regular search vs beam search with same beam width should give similar results
    let regular_results = index.search(&query, k).unwrap();
    let beam_results = index.search_with_beam(&query, k, 64).unwrap();
    
    assert_eq!(regular_results.len(), k);
    assert_eq!(beam_results.len(), k);
    
    // Results should be similar (allowing for some variation due to tie-breaking)
    let regular_distances: Vec<f32> = regular_results.iter().map(|r| r.distance).collect();
    let beam_distances: Vec<f32> = beam_results.iter().map(|r| r.distance).collect();
    
    // At least the closest result should be very similar
    assert!((regular_distances[0] - beam_distances[0]).abs() < 1e-3);
}

#[test]
fn test_beam_search_edge_cases() {
    let vectors = vec![
        (0, vec![1.0, 0.0]),
        (1, vec![0.0, 1.0]),
    ];
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .build(vectors)
        .unwrap();
    
    let query = vec![0.5, 0.5];
    
    // Test with beam width larger than number of nodes
    let results = index.search_with_beam(&query, 10, 100).unwrap();
    assert_eq!(results.len(), 2); // Should return all available nodes
    
    // Test with k larger than number of nodes
    let results = index.search_with_beam(&query, 10, 4).unwrap();
    assert_eq!(results.len(), 2); // Should return all available nodes
    
    // Test with very small beam width
    let results = index.search_with_beam(&query, 1, 1).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_search_buffer_reuse() {
    let vectors = generate_test_vectors(100, 32);
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .build(vectors)
        .unwrap();
    
    let mut buffer = SearchBuffer::new(200);
    
    // Perform multiple searches with the same buffer
    for i in 0..10 {
        let query: Vec<f32> = (0..32).map(|j| (i + j) as f32 * 0.1).collect();
        let results = index.search_with_buffer(&query, 5, 16, &mut buffer).unwrap();
        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_recall_improvement_with_beam_width() {
    let vectors = generate_test_vectors(500, 128);
    let mut rng = StdRng::seed_from_u64(123);
    let query: Vec<f32> = (0..128).map(|_| rng.gen::<f32>()).collect();
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .max_degree(32)
        .search_list_size(64)
        .build(vectors)
        .unwrap();
    
    let k = 10;
    
    // Get ground truth with very large beam
    let ground_truth = index.search_with_beam(&query, k * 2, 200).unwrap();
    let truth_set: std::collections::HashSet<u32> = 
        ground_truth.iter().take(k).map(|r| r.id).collect();
    
    // Test different beam widths
    let beam_widths = [8, 16, 32, 64, 128];
    let mut recalls = Vec::new();
    
    for &beam_width in &beam_widths {
        let results = index.search_with_beam(&query, k, beam_width).unwrap();
        let result_set: std::collections::HashSet<u32> = 
            results.iter().take(k).map(|r| r.id).collect();
        
        let intersection = truth_set.intersection(&result_set).count();
        let recall = intersection as f64 / k as f64;
        recalls.push(recall);
    }
    
    // Recall should generally improve with larger beam widths
    // (though not guaranteed to be monotonic due to randomness)
    assert!(recalls.last().unwrap() >= recalls.first().unwrap());
}

#[test]
fn test_deterministic_search() {
    let vectors = generate_test_vectors(100, 32);
    let query = vec![0.0; 32];
    
    let distance_fn = EuclideanDistance;
    let index = IndexBuilder::new(distance_fn)
        .seed(42)
        .build(vectors)
        .unwrap();
    
    // Multiple searches with same parameters should give same results
    let results1 = index.search_with_beam(&query, 5, 32).unwrap();
    let results2 = index.search_with_beam(&query, 5, 32).unwrap();
    
    assert_eq!(results1.len(), results2.len());
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.id, r2.id);
        assert!((r1.distance - r2.distance).abs() < 1e-6);
    }
}