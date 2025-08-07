//! Property-based tests for file I/O operations
//! These tests verify round-trip encoding/decoding invariants

use proptest::prelude::*;
use diskann_io::format::{BinaryHeader, write_vectors_f32, read_vectors_f32};
use diskann_core::vectors::Vector;
use std::io::Cursor;

/// Generate valid binary headers
fn arb_header() -> impl Strategy<Value = BinaryHeader> {
    (1i32..10000, 1i32..1000).prop_map(|(points, dims)| {
        BinaryHeader {
            num_points: points,
            num_dimensions: dims,
        }
    })
}

/// Generate vectors with specific dimensions
fn arb_vectors_with_dims(num_vectors: usize, dims: usize) -> impl Strategy<Value = Vec<Vector>> {
    prop::collection::vec(
        prop::collection::vec(
            prop::num::f32::ANY.prop_filter("No NaN or infinite values", |x| x.is_finite()),
            dims..=dims
        ),
        num_vectors..=num_vectors
    )
}

/// Generate arbitrary valid vector datasets
fn arb_vector_dataset() -> impl Strategy<Value = Vec<Vector>> {
    (1usize..100, 1usize..128).prop_flat_map(|(num_vecs, dims)| {
        arb_vectors_with_dims(num_vecs, dims)
    })
}

proptest! {
    /// Test that binary header round-trip preserves data
    #[test]
    fn test_header_round_trip(header in arb_header()) {
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        
        // Write header
        prop_assert!(header.write_to(&mut cursor).is_ok(), "Header write should succeed");
        
        // Read it back
        cursor.set_position(0);
        let read_header = BinaryHeader::read_from(&mut cursor);
        prop_assert!(read_header.is_ok(), "Header read should succeed");
        
        let read_header = read_header.unwrap();
        prop_assert_eq!(header.num_points, read_header.num_points, "num_points should match");
        prop_assert_eq!(header.num_dimensions, read_header.num_dimensions, "num_dimensions should match");
    }

    /// Test that vector data round-trip preserves exact values
    #[test]
    fn test_vector_round_trip(vectors in arb_vector_dataset()) {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        
        // Write vectors
        let bytes_written = write_vectors_f32(&mut cursor, &vectors);
        prop_assert!(bytes_written.is_ok(), "Vector write should succeed");
        
        // Read them back
        cursor.set_position(0);
        let read_vectors = read_vectors_f32(&mut cursor);
        prop_assert!(read_vectors.is_ok(), "Vector read should succeed");
        
        let read_vectors = read_vectors.unwrap();
        
        // Verify exact match
        prop_assert_eq!(vectors.len(), read_vectors.len(), "Number of vectors should match");
        
        for (i, (original, read)) in vectors.iter().zip(read_vectors.iter()).enumerate() {
            prop_assert_eq!(original.len(), read.len(), "Vector {} dimension should match", i);
            
            for (j, (&orig_val, &read_val)) in original.iter().zip(read.iter()).enumerate() {
                prop_assert_eq!(orig_val, read_val, 
                    "Vector {}[{}] should match: {} vs {}", i, j, orig_val, read_val);
            }
        }
    }

    /// Test header validation properties
    #[test]
    fn test_header_validation(num_points in -1000i32..10000i32, num_dimensions in -1000i32..1000i32) {
        let header = BinaryHeader { num_points, num_dimensions };
        let result = header.validate();
        
        if num_points < 0 || num_dimensions <= 0 {
            prop_assert!(result.is_err(), "Invalid header should fail validation");
        } else {
            prop_assert!(result.is_ok(), "Valid header should pass validation");
        }
    }

    /// Test data size calculations
    #[test]
    fn test_data_size_calculations(header in arb_header()) {
        let expected_data_size = (header.num_points * header.num_dimensions) as usize * 4; // 4 bytes per f32
        let expected_total_size = 8 + expected_data_size; // 8 bytes for header
        
        prop_assert_eq!(header.data_size_f32(), expected_data_size, "Data size calculation should be correct");
        prop_assert_eq!(header.total_file_size_f32(), expected_total_size, "Total size calculation should be correct");
    }
}

#[cfg(test)]
mod simple_property_tests {
    use super::*;

    /// Test that file I/O handles empty vectors gracefully
    #[test]
    fn test_empty_vector_handling() {
        let empty_vectors: Vec<Vector> = vec![];
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        
        let result = write_vectors_f32(&mut cursor, &empty_vectors);
        assert!(result.is_err(), "Writing empty vectors should fail gracefully");
    }

    /// Test consistency of different vector dimensions
    #[test]
    fn test_dimension_consistency() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        
        // Mix in a vector with wrong dimension
        let wrong_dim_vector = vec![1.0f32; 3]; // Different dimension
        let mut mixed_vectors = vectors.clone();
        mixed_vectors.push(wrong_dim_vector);
        
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        
        let result = write_vectors_f32(&mut cursor, &mixed_vectors);
        assert!(result.is_err(), "Writing vectors with inconsistent dimensions should fail");
    }

    /// Test that special float values are handled properly
    #[test]
    fn test_special_float_values() {
        let dims = 4;
        let special_vectors = vec![
            vec![0.0f32; dims],           // All zeros
            vec![1.0f32; dims],           // All ones
            vec![-1.0f32; dims],          // All negative ones
            vec![f32::MIN_POSITIVE; dims], // Smallest positive
            vec![f32::MAX; dims],         // Largest finite
        ];
        
        for vectors in special_vectors {
            let dataset = vec![vectors];
            let mut buffer = Vec::new();
            let mut cursor = Cursor::new(&mut buffer);
            
            // Should handle special but finite values
            let write_result = write_vectors_f32(&mut cursor, &dataset);
            assert!(write_result.is_ok(), "Should handle special finite values");
            
            cursor.set_position(0);
            let read_result = read_vectors_f32(&mut cursor);
            assert!(read_result.is_ok(), "Should read back special finite values");
            
            let read_vectors = read_result.unwrap();
            assert_eq!(dataset, read_vectors, "Special values should round-trip exactly");
        }
    }

    /// Test binary format consistency with different endianness considerations
    #[test]
    fn test_binary_format_consistency() {
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        
        // Write data
        let mut buffer1 = Vec::new();
        let mut cursor1 = Cursor::new(&mut buffer1);
        let write_result1 = write_vectors_f32(&mut cursor1, &vectors);
        assert!(write_result1.is_ok(), "First write should succeed");
        
        // Write the same data again
        let mut buffer2 = Vec::new();
        let mut cursor2 = Cursor::new(&mut buffer2);
        let write_result2 = write_vectors_f32(&mut cursor2, &vectors);
        assert!(write_result2.is_ok(), "Second write should succeed");
        
        // Binary output should be identical
        assert_eq!(buffer1, buffer2, "Binary output should be deterministic");
    }
}