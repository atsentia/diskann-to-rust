//! Loom concurrent testing for DiskANN data structures
//! These tests model concurrent operations to detect race conditions and deadlocks

#[cfg(loom)]
mod loom_tests {
    use loom::sync::{Arc, Mutex};
    use loom::thread;
    use std::collections::HashMap;
    use diskann_core::{vectors::VectorId, structures::GraphNode};
    use diskann_traits::distance::EuclideanDistance;

    /// A simplified concurrent-safe index for loom testing
    #[derive(Clone)]
    struct ConcurrentTestIndex {
        nodes: Arc<Mutex<HashMap<VectorId, GraphNode>>>,
        distance_fn: EuclideanDistance,
    }

    impl ConcurrentTestIndex {
        fn new() -> Self {
            Self {
                nodes: Arc::new(Mutex::new(HashMap::new())),
                distance_fn: EuclideanDistance::default(),
            }
        }

        fn insert(&self, id: VectorId, vector: Vec<f32>) -> Result<(), &'static str> {
            let mut nodes = self.nodes.lock().unwrap();
            
            if nodes.contains_key(&id) {
                return Err("Node already exists");
            }
            
            let node = GraphNode {
                id,
                vector,
                neighbors: Vec::new(),
            };
            
            nodes.insert(id, node);
            Ok(())
        }

        fn remove(&self, id: VectorId) -> Result<(), &'static str> {
            let mut nodes = self.nodes.lock().unwrap();
            
            if !nodes.contains_key(&id) {
                return Err("Node does not exist");
            }
            
            // Remove the node
            nodes.remove(&id);
            
            // Remove references to this node from all other nodes
            for node in nodes.values_mut() {
                node.neighbors.retain(|&neighbor_id| neighbor_id != id);
            }
            
            Ok(())
        }

        fn search(&self, _query: &[f32]) -> Vec<VectorId> {
            let nodes = self.nodes.lock().unwrap();
            
            // Simplified search: just return all node IDs
            nodes.keys().cloned().collect()
        }

        fn add_edge(&self, from: VectorId, to: VectorId) -> Result<(), &'static str> {
            let mut nodes = self.nodes.lock().unwrap();
            
            if !nodes.contains_key(&from) || !nodes.contains_key(&to) {
                return Err("One or both nodes do not exist");
            }
            
            if let Some(node) = nodes.get_mut(&from) {
                if !node.neighbors.contains(&to) {
                    node.neighbors.push(to);
                }
            }
            
            Ok(())
        }

        fn size(&self) -> usize {
            let nodes = self.nodes.lock().unwrap();
            nodes.len()
        }
    }

    #[test]
    fn test_concurrent_insert_remove() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                // Thread 1: Insert nodes 0, 2, 4
                for i in [0, 2, 4] {
                    let vector = vec![i as f32; 4];
                    let _ = index1.insert(i, vector);
                }
            });
            
            let handle2 = thread::spawn(move || {
                // Thread 2: Insert nodes 1, 3, 5
                for i in [1, 3, 5] {
                    let vector = vec![i as f32; 4];
                    let _ = index2.insert(i, vector);
                }
            });
            
            handle1.join().unwrap();
            handle2.join().unwrap();
            
            // All inserts should succeed without data races
            assert_eq!(index.size(), 6);
        });
    }

    #[test]
    fn test_concurrent_insert_search() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                // Thread 1: Insert nodes
                for i in 0..3 {
                    let vector = vec![i as f32; 4];
                    let _ = index1.insert(i, vector);
                }
            });
            
            let handle2 = thread::spawn(move || {
                // Thread 2: Search concurrently
                let query = vec![1.0; 4];
                let _results = index2.search(&query);
                // Search should not crash or deadlock
            });
            
            handle1.join().unwrap();
            handle2.join().unwrap();
            
            // Final state should be consistent
            assert!(index.size() <= 3); // At most 3 nodes inserted
        });
    }

    #[test]
    fn test_concurrent_remove_search() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            // Pre-populate with some nodes
            for i in 0..3 {
                let vector = vec![i as f32; 4];
                index.insert(i, vector).unwrap();
            }
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                // Thread 1: Remove nodes
                let _ = index1.remove(1);
            });
            
            let handle2 = thread::spawn(move || {
                // Thread 2: Search concurrently
                let query = vec![1.0; 4];
                let _results = index2.search(&query);
                // Search should handle concurrent removals gracefully
            });
            
            handle1.join().unwrap();
            handle2.join().unwrap();
            
            // Final state should be consistent
            assert!(index.size() >= 2); // At least 2 nodes remain
        });
    }

    #[test]
    fn test_concurrent_edge_operations() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            // Pre-populate with nodes
            for i in 0..3 {
                let vector = vec![i as f32; 4];
                index.insert(i, vector).unwrap();
            }
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                // Thread 1: Add edges
                let _ = index1.add_edge(0, 1);
                let _ = index1.add_edge(1, 2);
            });
            
            let handle2 = thread::spawn(move || {
                // Thread 2: Add different edges
                let _ = index2.add_edge(2, 0);
                let _ = index2.add_edge(0, 2);
            });
            
            handle1.join().unwrap();
            handle2.join().unwrap();
            
            // Graph should remain consistent
            assert_eq!(index.size(), 3);
        });
    }

    #[test]
    fn test_concurrent_remove_with_edges() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            // Pre-populate with nodes and edges
            for i in 0..3 {
                let vector = vec![i as f32; 4];
                index.insert(i, vector).unwrap();
            }
            index.add_edge(0, 1).unwrap();
            index.add_edge(1, 2).unwrap();
            index.add_edge(2, 0).unwrap();
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                // Thread 1: Remove a node with edges
                let _ = index1.remove(1);
            });
            
            let handle2 = thread::spawn(move || {
                // Thread 2: Try to add edge to the node being removed
                let _ = index2.add_edge(0, 1);
                let _ = index2.add_edge(2, 1);
            });
            
            handle1.join().unwrap();
            handle2.join().unwrap();
            
            // Graph should be consistent after concurrent remove/add edge operations
            let final_size = index.size();
            assert!(final_size >= 2 && final_size <= 3);
        });
    }

    #[test]
    fn test_insert_duplicate_concurrent() {
        loom::model(|| {
            let index = Arc::new(ConcurrentTestIndex::new());
            
            let index1 = Arc::clone(&index);
            let index2 = Arc::clone(&index);
            
            let handle1 = thread::spawn(move || {
                let vector = vec![1.0; 4];
                index1.insert(42, vector)
            });
            
            let handle2 = thread::spawn(move || {
                let vector = vec![2.0; 4];
                index2.insert(42, vector) // Same ID
            });
            
            let result1 = handle1.join().unwrap();
            let result2 = handle2.join().unwrap();
            
            // Exactly one should succeed, one should fail
            assert!(result1.is_ok() != result2.is_ok());
            assert_eq!(index.size(), 1);
        });
    }
}

#[cfg(not(loom))]
mod regular_tests {
    // Regular tests that run when loom is not enabled
    
    #[test]
    fn placeholder_concurrent_test() {
        // This is a placeholder test for when loom is not available
        // In a real scenario, you might use std::thread for basic concurrent testing
        assert!(true);
    }
}