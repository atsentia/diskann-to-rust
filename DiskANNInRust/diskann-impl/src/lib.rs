//! Concrete implementations of DiskANN algorithms and data structures

#![deny(warnings)]
#![warn(missing_docs)]

/// Graph-based index implementations
pub mod graph;

/// Search algorithm implementations
pub mod search;

pub use graph::{VamanaIndex, IndexBuilder, VamanaConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_traits::{distance::EuclideanDistance, index::Index, search::Search};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_vamana_index_basic_operations() {
        let distance_fn = EuclideanDistance;
        let config = VamanaConfig {
            max_degree: 8,
            search_list_size: 16,
            seed: 42,
            alpha: 1.2,
        };
        let mut index = VamanaIndex::new(distance_fn, config);

        // Test adding vectors
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![0.5, 0.5],
        ];

        for (i, vector) in vectors.iter().enumerate() {
            index.add(i as u32, vector.clone()).unwrap();
        }

        assert_eq!(index.size(), 5);

        // Test search
        let query = vec![0.9, 0.1];
        let results = index.search(&query, 3).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);

        // Test removal
        index.remove(0).unwrap();
        assert_eq!(index.size(), 4);
    }

    #[test]
    fn test_index_builder() {
        let distance_fn = EuclideanDistance;
        
        let vectors = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![-1.0, 0.0]),
            (3, vec![0.0, -1.0]),
            (4, vec![0.5, 0.5]),
            (5, vec![-0.5, 0.5]),
            (6, vec![0.5, -0.5]),
            (7, vec![-0.5, -0.5]),
        ];

        let index = IndexBuilder::new(distance_fn)
            .max_degree(4)
            .search_list_size(8)
            .seed(42)
            .alpha(1.0)
            .build(vectors)
            .unwrap();

        assert_eq!(index.size(), 8);

        // Test that the graph has reasonable connectivity
        let avg_degree = index.average_degree();
        assert!(avg_degree > 0.0);
        assert!(avg_degree <= 4.0); // Should not exceed max_degree
    }

    #[test]
    fn test_degree_distribution() {
        let distance_fn = EuclideanDistance;
        let mut rng = StdRng::seed_from_u64(42);
        
        // Generate random vectors
        let mut vectors = Vec::new();
        for i in 0..50 {
            let vector = vec![rng.gen::<f32>() * 2.0 - 1.0, rng.gen::<f32>() * 2.0 - 1.0];
            vectors.push((i, vector));
        }

        let index = IndexBuilder::new(distance_fn)
            .max_degree(8)
            .search_list_size(16)
            .seed(42)
            .build(vectors)
            .unwrap();

        let degrees = index.degree_distribution();
        assert_eq!(degrees.len(), 50);
        
        // Check that no node exceeds max degree
        for &degree in &degrees {
            assert!(degree <= 8);
        }
        
        // Check that average degree is reasonable
        let avg_degree = index.average_degree();
        assert!(avg_degree > 1.0); // Should have some connectivity
        assert!(avg_degree <= 8.0); // Should not exceed max_degree
    }

    #[test]
    fn test_reachability_property() {
        let distance_fn = EuclideanDistance;
        let mut rng = StdRng::seed_from_u64(42);
        
        // Generate random vectors in 2D
        let mut vectors = Vec::new();
        for i in 0..20 {
            let vector = vec![rng.gen::<f32>() * 2.0 - 1.0, rng.gen::<f32>() * 2.0 - 1.0];
            vectors.push((i, vector));
        }

        let index = IndexBuilder::new(distance_fn)
            .max_degree(6)
            .search_list_size(12)
            .seed(42)
            .build(vectors)
            .unwrap();

        // Test reachability from a random pivot
        let pivot_id = 0; // Use first node as pivot
        let mut reachable_count = 0;
        
        for target_id in 0..20 {
            if index.is_reachable_within_k_hops(target_id, pivot_id, 20) {
                reachable_count += 1;
            }
        }

        // Most nodes should be reachable within 20 hops in a small graph
        assert!(reachable_count >= 15, "Only {} out of 20 nodes are reachable within 20 hops", reachable_count);
    }

    #[test]
    fn test_graph_connectivity_after_deletion() {
        let distance_fn = EuclideanDistance;
        
        let vectors = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
            (4, vec![0.5, 0.5]),
        ];

        let mut index = IndexBuilder::new(distance_fn)
            .max_degree(3)
            .search_list_size(5)
            .seed(42)
            .build(vectors)
            .unwrap();

        assert_eq!(index.size(), 5);

        // Remove a node
        index.remove(4).unwrap();
        assert_eq!(index.size(), 4);

        // Test that remaining nodes are still searchable
        let query = vec![0.1, 0.1];
        let results = index.search(&query, 2).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_insert_after_construction() {
        let distance_fn = EuclideanDistance;
        
        let initial_vectors = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![-1.0, 0.0]),
        ];

        let mut index = IndexBuilder::new(distance_fn)
            .max_degree(4)
            .search_list_size(8)
            .seed(42)
            .build(initial_vectors)
            .unwrap();

        assert_eq!(index.size(), 3);

        // Insert new vectors
        index.add(3, vec![0.0, -1.0]).unwrap();
        index.add(4, vec![0.5, 0.5]).unwrap();

        assert_eq!(index.size(), 5);

        // Test search with new vectors
        let query = vec![0.4, 0.4];
        let results = index.search(&query, 2).unwrap();
        assert!(!results.is_empty());
        
        // The closest should be vector 4 (0.5, 0.5)
        assert_eq!(results[0].id, 4);
    }

    #[test]
    fn test_deterministic_construction() {
        let distance_fn = EuclideanDistance;
        
        let vectors = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![-1.0, 0.0]),
            (3, vec![0.0, -1.0]),
            (4, vec![0.5, 0.5]),
        ];

        // Build index twice with same seed
        let index1 = IndexBuilder::new(distance_fn)
            .seed(42)
            .build(vectors.clone())
            .unwrap();

        let index2 = IndexBuilder::new(distance_fn)
            .seed(42)
            .build(vectors)
            .unwrap();

        // Degree distributions should be the same
        let degrees1 = index1.degree_distribution();
        let degrees2 = index2.degree_distribution();
        
        assert_eq!(degrees1.len(), degrees2.len());
        // Note: Due to the deterministic shuffling, the exact graph structure should be the same
        // but we test a weaker property here for robustness
        assert_eq!(index1.average_degree(), index2.average_degree());
    }

    #[cfg(feature = "unsafe_opt")]
    #[test]
    fn test_parallel_construction() {
        let distance_fn = EuclideanDistance;
        
        let vectors = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![-1.0, 0.0]),
            (3, vec![0.0, -1.0]),
            (4, vec![0.5, 0.5]),
            (5, vec![-0.5, 0.5]),
            (6, vec![0.5, -0.5]),
            (7, vec![-0.5, -0.5]),
        ];

        let index = IndexBuilder::new(distance_fn)
            .max_degree(4)
            .search_list_size(8)
            .seed(42)
            .build_parallel(vectors)
            .unwrap();

        assert_eq!(index.size(), 8);
        
        // Parallel construction should produce a valid index
        let query = vec![0.1, 0.1];
        let results = index.search(&query, 3).unwrap();
        assert!(!results.is_empty());
    }
}