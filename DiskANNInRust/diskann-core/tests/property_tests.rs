//! Property-based tests for vector math operations
//! These tests verify mathematical invariants that should hold for any input

use proptest::prelude::*;
use diskann_core::math::{dot_product, l2_norm, l2_norm_squared, normalize};
use diskann_traits::distance::{EuclideanDistance, CosineDistance, Distance};

/// Generate vectors with reasonable size and values
fn arb_vector() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop::num::f32::ANY
            .prop_filter("Finite values only", |x| x.is_finite())
            .prop_map(|x| {
                if x.abs() < 1e-10 { 0.0 } // Clamp very small values to zero for numerical stability
                else if x.abs() > 1e6 { x.signum() * 1e6 } // Clamp large values
                else { x }
            }),
        1..=32 // Smaller dimensions for faster testing
    )
}

/// Generate pairs of vectors with matching dimensions
fn arb_vector_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    arb_vector().prop_flat_map(|v1| {
        let dim = v1.len();
        (Just(v1), prop::collection::vec(
            prop::num::f32::ANY
                .prop_filter("Finite values only", |x| x.is_finite())
                .prop_map(|x| {
                    if x.abs() < 1e-10 { 0.0 }
                    else if x.abs() > 1e6 { x.signum() * 1e6 }
                    else { x }
                }),
            dim..=dim
        ))
    })
}

proptest! {
    /// Test that normalization produces unit vectors
    #[test]
    fn test_normalize_produces_unit_vector(v in arb_vector()) {
        let normalized = normalize(&v);
        let norm = l2_norm(&normalized);
        
        // If original vector was non-zero, normalized should have unit length
        let original_norm = l2_norm(&v);
        if original_norm > 0.0 {
            // Should be close to 1.0, allowing for floating point precision
            prop_assert!((norm - 1.0).abs() < 1e-6, "Normalized vector should have unit length, got {}", norm);
        } else {
            // Zero vector should remain zero
            prop_assert_eq!(norm, 0.0, "Zero vector should remain zero after normalization");
        }
    }

    /// Test that normalization preserves direction (for non-zero vectors)
    #[test]
    fn test_normalize_preserves_direction(v in arb_vector()) {
        let original_norm = l2_norm(&v);
        if original_norm > 1e-30 { // Only test for vectors that aren't extremely close to zero
            let normalized = normalize(&v);
            // Check that the normalized vector is in the same direction
            let dot = dot_product(&v, &normalized);
            prop_assert!(dot > 0.0, "Normalized vector should be in same direction, got dot product {}", dot);
        }
    }

    /// Test L2 norm properties
    #[test]
    fn test_l2_norm_properties(v in arb_vector()) {
        let norm = l2_norm(&v);
        let norm_squared = l2_norm_squared(&v);
        
        // Non-negativity
        prop_assert!(norm >= 0.0, "L2 norm should be non-negative");
        prop_assert!(norm_squared >= 0.0, "L2 norm squared should be non-negative");
        
        // Consistency between norm and norm_squared - allow for floating point precision
        let norm_sq_calc = norm * norm;
        let precision_error = (norm_sq_calc - norm_squared).abs();
        let relative_error = if norm_squared > 0.0 { precision_error / norm_squared } else { precision_error };
        prop_assert!(relative_error < 1e-5, 
            "norm² should equal norm_squared within precision, got {} vs {} (relative error: {})", 
            norm_sq_calc, norm_squared, relative_error);
        
        // Zero iff vector is zero (considering floating point precision)
        let is_effectively_zero = v.iter().all(|&x| x.abs() < f32::EPSILON);
        let norm_is_zero = norm < f32::EPSILON;
        prop_assert_eq!(norm_is_zero, is_effectively_zero, 
            "Norm should be zero iff vector is effectively zero (considering precision)");
    }

    /// Test triangle inequality for Euclidean distance
    #[test]
    fn test_euclidean_triangle_inequality((v1, v2) in arb_vector_pair(), v3 in arb_vector()) {
        if v3.len() == v1.len() {
            let euclidean = EuclideanDistance::default();
            
            let d12 = euclidean.distance(&v1, &v2);
            let d13 = euclidean.distance(&v1, &v3);
            let d23 = euclidean.distance(&v2, &v3);
            
            // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            prop_assert!(d13 <= d12 + d23 + f32::EPSILON,
                "Triangle inequality violated: {} > {} + {}", d13, d12, d23);
            prop_assert!(d12 <= d13 + d23 + f32::EPSILON,
                "Triangle inequality violated: {} > {} + {}", d12, d13, d23);
            prop_assert!(d23 <= d12 + d13 + f32::EPSILON,
                "Triangle inequality violated: {} > {} + {}", d23, d12, d13);
        }
    }

    /// Test symmetry property for distance functions
    #[test]
    fn test_distance_symmetry((v1, v2) in arb_vector_pair()) {
        let euclidean = EuclideanDistance::default();
        let cosine = CosineDistance::default();
        
        let d12_euclidean = euclidean.distance(&v1, &v2);
        let d21_euclidean = euclidean.distance(&v2, &v1);
        prop_assert!((d12_euclidean - d21_euclidean).abs() < f32::EPSILON,
            "Euclidean distance should be symmetric");
        
        let d12_cosine = cosine.distance(&v1, &v2);
        let d21_cosine = cosine.distance(&v2, &v1);
        prop_assert!((d12_cosine - d21_cosine).abs() < f32::EPSILON,
            "Cosine distance should be symmetric");
    }

    /// Test identity property for distance functions
    #[test]
    fn test_distance_identity(v in arb_vector()) {
        let euclidean = EuclideanDistance::default();
        let cosine = CosineDistance::default();
        
        let d_euclidean = euclidean.distance(&v, &v);
        prop_assert_eq!(d_euclidean, 0.0, "Euclidean distance from vector to itself should be zero");
        
        let d_cosine = cosine.distance(&v, &v);
        // For cosine distance, zero vectors have distance 1.0, non-zero vectors have distance ~0.0
        let norm = l2_norm(&v);
        let expected_cosine = if norm < 1e-10 { 1.0 } else { 0.0 };
        let cosine_error = (d_cosine - expected_cosine).abs();
        prop_assert!(cosine_error < 1e-6, 
            "Cosine distance from vector to itself should be {} but got {} (error: {})", 
            expected_cosine, d_cosine, cosine_error);
    }

    /// Test dot product properties
    #[test]
    fn test_dot_product_properties((v1, v2) in arb_vector_pair()) {
        let dot12 = dot_product(&v1, &v2);
        let dot21 = dot_product(&v2, &v1);
        
        // Symmetry
        prop_assert!((dot12 - dot21).abs() < f32::EPSILON, "Dot product should be symmetric");
        
        // Relationship with norms
        let norm1 = l2_norm(&v1);
        let norm2 = l2_norm(&v2);
        
        if norm1 > 0.0 && norm2 > 0.0 {
            // Cauchy-Schwarz inequality: |⟨u,v⟩| ≤ ||u|| ||v||
            // Use relative tolerance for large values
            let tolerance = if norm1 * norm2 > 1e6 {
                (norm1 * norm2) * 1e-6 // 0.0001% relative tolerance for large values
            } else {
                1e-6
            };
            prop_assert!(dot12.abs() <= norm1 * norm2 + tolerance,
                "Cauchy-Schwarz inequality violated: {} > {} * {} (tolerance: {})", 
                dot12.abs(), norm1, norm2, tolerance);
        }
    }

    /// Test scaling properties
    #[test]
    fn test_scaling_properties(v in arb_vector(), scale in 0.1f32..10.0f32) {
        let scaled: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        
        let norm_original = l2_norm(&v);
        let norm_scaled = l2_norm(&scaled);
        
        // ||αv|| = |α| ||v|| - allow for floating point precision
        let expected = scale * norm_original;
        let precision_error = (norm_scaled - expected).abs();
        let relative_error = if expected > 0.0 { precision_error / expected } else { precision_error };
        prop_assert!(relative_error < 1e-5,
            "Scaling property violated: {} vs {} (relative error: {})", norm_scaled, expected, relative_error);
    }
}