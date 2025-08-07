//! Round-trip tests to verify bit-for-bit equality with C++ DiskANN format

use diskann_io::{IndexWriter, IndexLoader, FileIndexWriter, MmapIndexLoader};
use diskann_core::structures::GraphNode;
use diskann_core::vectors::Vector;
use tempfile::NamedTempFile;
use std::fs;

/// Test bit-for-bit equality of written and loaded vector data
#[test]
fn test_round_trip_bit_equality() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    // Create test vectors with varied floating point values
    let original_vectors = vec![
        vec![1.0, 2.5, -3.14159, 0.0],
        vec![f32::MAX, f32::MIN, f32::EPSILON, -f32::EPSILON],
        vec![std::f32::consts::PI, std::f32::consts::E, std::f32::consts::SQRT_2, std::f32::consts::LN_2],
        vec![42.0, -42.0, 0.5, -0.5],
    ];
    
    let temp_file = NamedTempFile::new().unwrap();
    
    // Write vectors
    let bytes_written = writer.write_vectors(temp_file.path(), &original_vectors).unwrap();
    
    // Load vectors back
    let loaded_vectors = loader.load_vectors(temp_file.path()).unwrap();
    
    // Verify exact equality
    assert_eq!(original_vectors.len(), loaded_vectors.len());
    for (i, (orig, loaded)) in original_vectors.iter().zip(loaded_vectors.iter()).enumerate() {
        assert_eq!(orig.len(), loaded.len(), "Vector {} dimension mismatch", i);
        for (j, (&orig_val, &loaded_val)) in orig.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(orig_val.to_bits(), loaded_val.to_bits(), 
                      "Bit-for-bit mismatch at vector[{}][{}]: {} != {}", 
                      i, j, orig_val, loaded_val);
        }
    }
    
    // Verify expected file size
    let expected_size = 8 + (4 * 4 * 4); // header + 4 vectors * 4 dims * 4 bytes/f32
    assert_eq!(bytes_written, expected_size);
    
    let actual_size = fs::metadata(temp_file.path()).unwrap().len() as usize;
    assert_eq!(actual_size, expected_size);
}

/// Test round-trip with graph nodes
#[test]
fn test_graph_round_trip_bit_equality() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    let original_nodes = vec![
        GraphNode::new(0, vec![1.0, 2.0, 3.0]),
        GraphNode::new(1, vec![4.0, 5.0, 6.0]),
        GraphNode::new(2, vec![7.0, 8.0, 9.0]),
    ];
    
    let temp_file = NamedTempFile::new().unwrap();
    
    // Write graph
    writer.write_graph(temp_file.path(), &original_nodes).unwrap();
    
    // Load graph back
    let loaded_nodes = loader.load_graph(temp_file.path()).unwrap();
    
    // Verify exact equality of vector data
    assert_eq!(original_nodes.len(), loaded_nodes.len());
    for (orig, loaded) in original_nodes.iter().zip(loaded_nodes.iter()) {
        assert_eq!(orig.vector.len(), loaded.vector.len());
        for (&orig_val, &loaded_val) in orig.vector.iter().zip(loaded.vector.iter()) {
            assert_eq!(orig_val.to_bits(), loaded_val.to_bits());
        }
    }
}

/// Test that memory-mapped and buffered I/O produce identical results  
#[test]
fn test_mmap_vs_buffered_consistency() {
    let writer = FileIndexWriter::new();
    let mmap_loader = MmapIndexLoader::new();
    let buffered_loader = MmapIndexLoader::buffered_only();
    
    let vectors = vec![
        vec![1.5, 2.5, 3.5],
        vec![-1.5, -2.5, -3.5],
    ];
    
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    // Load with both strategies
    let mmap_vectors = mmap_loader.load_vectors(temp_file.path()).unwrap();
    let buffered_vectors = buffered_loader.load_vectors(temp_file.path()).unwrap();
    
    // Should be identical
    assert_eq!(mmap_vectors.len(), buffered_vectors.len());
    for (mmap_vec, buffered_vec) in mmap_vectors.iter().zip(buffered_vectors.iter()) {
        assert_eq!(mmap_vec.len(), buffered_vec.len());
        for (&mmap_val, &buffered_val) in mmap_vec.iter().zip(buffered_vec.iter()) {
            assert_eq!(mmap_val.to_bits(), buffered_val.to_bits());
        }
    }
}

/// Test with edge case floating point values
#[test]
fn test_special_float_values() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    let special_vectors = vec![
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0],
        vec![-0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0],
    ];
    
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &special_vectors).unwrap();
    
    let loaded_vectors = loader.load_vectors(temp_file.path()).unwrap();
    
    assert_eq!(special_vectors.len(), loaded_vectors.len());
    for (orig, loaded) in special_vectors.iter().zip(loaded_vectors.iter()) {
        assert_eq!(orig.len(), loaded.len());
        for (&orig_val, &loaded_val) in orig.iter().zip(loaded.iter()) {
            // For NaN, check that both are NaN since NaN != NaN
            if orig_val.is_nan() {
                assert!(loaded_val.is_nan(), "Expected NaN, got {}", loaded_val);
            } else {
                assert_eq!(orig_val.to_bits(), loaded_val.to_bits());
            }
        }
    }
}

/// Test empty file handling
#[test]
fn test_empty_file_error() {
    let temp_file = NamedTempFile::new().unwrap();
    // File is empty by default
    
    let loader = MmapIndexLoader::new();
    let result = loader.load_vectors(temp_file.path());
    assert!(result.is_err());
}

/// Test file with invalid header
#[test]
fn test_invalid_header() {
    let temp_file = NamedTempFile::new().unwrap();
    // Write invalid header (negative dimensions)
    std::fs::write(temp_file.path(), &[2u8, 0, 0, 0, 255, 255, 255, 255]).unwrap();
    
    let loader = MmapIndexLoader::new();
    let result = loader.load_vectors(temp_file.path());
    assert!(result.is_err());
}

/// Benchmark test for large vector sets (disabled by default)
#[test]
#[ignore = "Large dataset test - run with --ignored"]
fn test_large_dataset_performance() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    // Create a large set of vectors
    let num_vectors = 10_000;
    let dimensions = 128;
    let vectors: Vec<Vector> = (0..num_vectors)
        .map(|i| {
            (0..dimensions)
                .map(|j| (i * dimensions + j) as f32 * 0.001)
                .collect()
        })
        .collect();
    
    let temp_file = NamedTempFile::new().unwrap();
    
    let start = std::time::Instant::now();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    let write_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let loaded_vectors = loader.load_vectors(temp_file.path()).unwrap();
    let load_time = start.elapsed();
    
    println!("Write time: {:?}", write_time);
    println!("Load time: {:?}", load_time);
    println!("Vectors: {}, Dimensions: {}", num_vectors, dimensions);
    
    // Verify consistency for first and last vectors
    assert_eq!(vectors[0], loaded_vectors[0]);
    assert_eq!(vectors[num_vectors - 1], loaded_vectors[num_vectors - 1]);
}

/// Test metadata extraction accuracy
#[test]
fn test_metadata_accuracy() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    let vectors = vec![
        vec![1.0; 64],  // 64-dimensional vector
        vec![2.0; 64],
        vec![3.0; 64],
    ];
    
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    let metadata = loader.get_metadata(temp_file.path()).unwrap();
    
    assert_eq!(metadata.num_vectors, 3);
    assert_eq!(metadata.dimension, 64);
    
    // File size should be header (8 bytes) + data (3 * 64 * 4 bytes)
    let expected_size = 8 + (3 * 64 * 4);
    assert_eq!(metadata.file_size, expected_size);
}

/// Test subset loading functionality
#[test]
fn test_subset_loading() {
    let writer = FileIndexWriter::new();
    let loader = MmapIndexLoader::new();
    
    let vectors: Vec<Vector> = (0..10)
        .map(|i| vec![i as f32, (i + 1) as f32])
        .collect();
    
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    // Load subset from index 3, count 4
    let subset = loader.load_vectors_subset(temp_file.path(), 3, 4).unwrap();
    
    assert_eq!(subset.len(), 4);
    assert_eq!(subset[0], vec![3.0, 4.0]);
    assert_eq!(subset[1], vec![4.0, 5.0]);
    assert_eq!(subset[2], vec![5.0, 6.0]);
    assert_eq!(subset[3], vec![6.0, 7.0]);
}