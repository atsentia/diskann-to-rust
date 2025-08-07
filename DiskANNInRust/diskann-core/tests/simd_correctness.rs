//! Comprehensive correctness tests for SIMD vs scalar distance computations
//!
//! This test suite validates that SIMD implementations produce identical results
//! to scalar implementations across various vector sizes and input patterns.

#[cfg(feature = "simd")]
use diskann_core::simd::{
    l2_squared_distance_scalar, l2_squared_distance_dispatch,
    inner_product_distance_scalar, inner_product_distance_dispatch,
};

#[cfg(feature = "simd")]
use diskann_core::simd::{
    l2_squared_distance_simd, inner_product_distance_simd,
};

/// Generate test vectors with various patterns for exhaustive testing
#[cfg(feature = "simd")]
fn generate_test_vectors(len: usize) -> Vec<(Vec<f32>, Vec<f32>, &'static str)> {
    let mut test_cases = Vec::new();
    
    // Random-like patterns (using deterministic sequences for reproducibility)
    let random_a: Vec<f32> = (0..len).map(|i| {
        let x = (i as f32 * 0.31415926) % 1.0;
        x * 2.0 - 1.0  // Range [-1, 1]
    }).collect();
    let random_b: Vec<f32> = (0..len).map(|i| {
        let x = ((i as f32 + 100.0) * 0.27182818) % 1.0;
        x * 2.0 - 1.0  // Range [-1, 1]
    }).collect();
    test_cases.push((random_a, random_b, "random_pattern"));
    
    // Identical vectors (should give distance 0 for L2, and minimum distance for inner product)
    let identical = vec![0.5f32; len];
    test_cases.push((identical.clone(), identical, "identical_vectors"));
    
    // Opposite vectors
    let positive = vec![1.0f32; len];
    let negative = vec![-1.0f32; len];
    test_cases.push((positive, negative, "opposite_vectors"));
    
    // Zero vector vs non-zero
    let zero = vec![0.0f32; len];
    let ones = vec![1.0f32; len];
    test_cases.push((zero, ones, "zero_vs_ones"));
    
    // Alternating pattern
    let alternating_a: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let alternating_b: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
    test_cases.push((alternating_a, alternating_b, "alternating_pattern"));
    
    // Decreasing magnitude
    let decreasing_a: Vec<f32> = (0..len).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let decreasing_b: Vec<f32> = (0..len).map(|i| 2.0 / (i as f32 + 1.0)).collect();
    test_cases.push((decreasing_a, decreasing_b, "decreasing_magnitude"));
    
    // Large values (near f32 limits)
    let large_a = vec![1e6f32; len];
    let large_b = vec![1e6f32 + 1.0; len];
    test_cases.push((large_a, large_b, "large_values"));
    
    // Small values (near zero)
    let small_a = vec![1e-6f32; len];
    let small_b = vec![2e-6f32; len];
    test_cases.push((small_a, small_b, "small_values"));
    
    test_cases
}

#[cfg(feature = "simd")]
#[test]
fn test_l2_squared_distance_correctness_comprehensive() {
    // Use larger tolerance for accumulated floating point errors in SIMD
    const TOLERANCE: f32 = 1e-3;
    
    // Test vector lengths from 32 to 1024
    let test_lengths = [32, 64, 100, 128, 200, 256, 333, 512, 777, 1000, 1024];
    
    for &len in &test_lengths {
        let test_vectors = generate_test_vectors(len);
        
        for (a, b, pattern_name) in test_vectors {
            let scalar_result = l2_squared_distance_scalar(&a, &b);
            let dispatch_result = l2_squared_distance_dispatch(&a, &b);
            
            // Use relative tolerance for large values
            let tolerance = if scalar_result > 1e6 {
                scalar_result * 1e-5  // 0.001% relative tolerance for very large values
            } else {
                TOLERANCE
            };
            
            assert!(
                (scalar_result - dispatch_result).abs() < tolerance,
                "L2 squared distance mismatch for length {} pattern '{}': scalar={}, dispatch={}, tolerance={}",
                len, pattern_name, scalar_result, dispatch_result, tolerance
            );
            
            // Ensure the result is non-negative
            assert!(
                scalar_result >= 0.0 && dispatch_result >= 0.0,
                "L2 squared distance should be non-negative for length {} pattern '{}'",
                len, pattern_name
            );
            
            #[cfg(feature = "simd")]
            if len >= 8 {
                let simd_result = l2_squared_distance_simd(&a, &b);
                assert!(
                    (scalar_result - simd_result).abs() < tolerance,
                    "L2 squared distance SIMD mismatch for length {} pattern '{}': scalar={}, simd={}, tolerance={}",
                    len, pattern_name, scalar_result, simd_result, tolerance
                );
            }
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_inner_product_distance_correctness_comprehensive() {
    // Use larger tolerance for accumulated floating point errors in SIMD
    const TOLERANCE: f32 = 1e-3;
    
    // Test vector lengths from 32 to 1024
    let test_lengths = [32, 64, 100, 128, 200, 256, 333, 512, 777, 1000, 1024];
    
    for &len in &test_lengths {
        let test_vectors = generate_test_vectors(len);
        
        for (a, b, pattern_name) in test_vectors {
            let scalar_result = inner_product_distance_scalar(&a, &b);
            let dispatch_result = inner_product_distance_dispatch(&a, &b);
            
            // Use relative tolerance for large magnitude results
            let tolerance = if scalar_result.abs() > 1e6 {
                scalar_result.abs() * 1e-5  // 0.001% relative tolerance for very large values
            } else {
                TOLERANCE
            };
            
            assert!(
                (scalar_result - dispatch_result).abs() < tolerance,
                "Inner product distance mismatch for length {} pattern '{}': scalar={}, dispatch={}, tolerance={}",
                len, pattern_name, scalar_result, dispatch_result, tolerance
            );
            
            #[cfg(feature = "simd")]
            if len >= 8 {
                let simd_result = inner_product_distance_simd(&a, &b);
                assert!(
                    (scalar_result - simd_result).abs() < tolerance,
                    "Inner product distance SIMD mismatch for length {} pattern '{}': scalar={}, simd={}, tolerance={}",
                    len, pattern_name, scalar_result, simd_result, tolerance
                );
            }
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_vs_scalar_edge_cases() {
    const TOLERANCE: f32 = 1e-3;
    
    // Test with lengths that don't align to SIMD boundaries
    let odd_lengths = [31, 33, 65, 127, 129, 255, 257, 511, 513];
    
    for &len in &odd_lengths {
        // Test with simple incrementing pattern to make verification easier
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1) + 0.5).collect();
        
        let scalar_l2 = l2_squared_distance_scalar(&a, &b);
        let dispatch_l2 = l2_squared_distance_dispatch(&a, &b);
        
        let l2_tolerance = if scalar_l2 > 1e6 { scalar_l2 * 1e-4 } else { TOLERANCE };
        
        assert!(
            (scalar_l2 - dispatch_l2).abs() < l2_tolerance,
            "L2 distance mismatch for odd length {}: scalar={}, dispatch={}, tolerance={}",
            len, scalar_l2, dispatch_l2, l2_tolerance
        );
        
        let scalar_inner = inner_product_distance_scalar(&a, &b);
        let dispatch_inner = inner_product_distance_dispatch(&a, &b);
        
        let inner_tolerance = if scalar_inner.abs() > 1e3 { scalar_inner.abs() * 1e-4 } else { TOLERANCE };
        
        assert!(
            (scalar_inner - dispatch_inner).abs() < inner_tolerance,
            "Inner product distance mismatch for odd length {}: scalar={}, dispatch={}, tolerance={}",
            len, scalar_inner, dispatch_inner, inner_tolerance
        );
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_mathematical_properties() {
    let lengths = [64, 128, 256, 512];
    
    for &len in &lengths {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).cos()).collect();
        let c: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).tan().atan()).collect();
        
        // Test symmetry: d(a,b) = d(b,a)
        let d_ab = l2_squared_distance_dispatch(&a, &b);
        let d_ba = l2_squared_distance_dispatch(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-6, "L2 distance should be symmetric");
        
        let inner_ab = inner_product_distance_dispatch(&a, &b);
        let inner_ba = inner_product_distance_dispatch(&b, &a);
        assert!((inner_ab - inner_ba).abs() < 1e-6, "Inner product distance should be symmetric");
        
        // Test identity: d(a,a) = 0 for L2
        let d_aa = l2_squared_distance_dispatch(&a, &a);
        assert!(d_aa < 1e-6, "L2 distance to self should be zero");
        
        // Test non-negativity
        assert!(d_ab >= 0.0, "L2 distance should be non-negative");
        assert!(l2_squared_distance_dispatch(&a, &c) >= 0.0, "L2 distance should be non-negative");
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn test_avx2_specific_correctness() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip if AVX2 not available
    }
    
    const TOLERANCE: f32 = 1e-3;
    let lengths = [64, 128, 256, 512, 1024];
    
    for &len in &lengths {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.123).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.456).cos()).collect();
        
        let scalar_l2 = l2_squared_distance_scalar(&a, &b);
        let simd_l2 = l2_squared_distance_simd(&a, &b);
        let avx2_l2 = unsafe { diskann_core::simd::l2_squared_distance_avx2(&a, &b) };
        
        let l2_tolerance = if scalar_l2 > 1e6 { scalar_l2 * 1e-5 } else { TOLERANCE };
        
        assert!(
            (scalar_l2 - avx2_l2).abs() < l2_tolerance,
            "AVX2 L2 distance mismatch for length {}: scalar={}, avx2={}, tolerance={}",
            len, scalar_l2, avx2_l2, l2_tolerance
        );
        
        assert!(
            (simd_l2 - avx2_l2).abs() < l2_tolerance,
            "AVX2 vs portable SIMD L2 distance mismatch for length {}: simd={}, avx2={}, tolerance={}",
            len, simd_l2, avx2_l2, l2_tolerance
        );
        
        let scalar_inner = inner_product_distance_scalar(&a, &b);
        let simd_inner = inner_product_distance_simd(&a, &b);
        let avx2_inner = unsafe { diskann_core::simd::inner_product_distance_avx2(&a, &b) };
        
        let inner_tolerance = if scalar_inner.abs() > 1e6 { scalar_inner.abs() * 1e-5 } else { TOLERANCE };
        
        assert!(
            (scalar_inner - avx2_inner).abs() < inner_tolerance,
            "AVX2 inner product distance mismatch for length {}: scalar={}, avx2={}, tolerance={}",
            len, scalar_inner, avx2_inner, inner_tolerance
        );
        
        assert!(
            (simd_inner - avx2_inner).abs() < inner_tolerance,
            "AVX2 vs portable SIMD inner product distance mismatch for length {}: simd={}, avx2={}, tolerance={}",
            len, simd_inner, avx2_inner, inner_tolerance
        );
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_special_values() {
    const TOLERANCE: f32 = 1e-6;
    
    // Test with special floating point values
    let special_cases = vec![
        (vec![0.0f32, 0.0, 0.0, 0.0], vec![0.0f32, 0.0, 0.0, 0.0], "all_zeros"),
        (vec![1.0f32, 1.0, 1.0, 1.0], vec![1.0f32, 1.0, 1.0, 1.0], "all_ones"),
        (vec![f32::MIN_POSITIVE; 64], vec![f32::MIN_POSITIVE; 64], "min_positive"),
        (vec![-1.0f32, 1.0, -1.0, 1.0], vec![1.0f32, -1.0, 1.0, -1.0], "alternating_signs"),
    ];
    
    for (a, b, case_name) in special_cases {
        let scalar_l2 = l2_squared_distance_scalar(&a, &b);
        let dispatch_l2 = l2_squared_distance_dispatch(&a, &b);
        
        assert!(
            (scalar_l2 - dispatch_l2).abs() < TOLERANCE,
            "L2 distance mismatch for special case '{}': scalar={}, dispatch={}",
            case_name, scalar_l2, dispatch_l2
        );
        
        let scalar_inner = inner_product_distance_scalar(&a, &b);
        let dispatch_inner = inner_product_distance_dispatch(&a, &b);
        
        assert!(
            (scalar_inner - dispatch_inner).abs() < TOLERANCE,
            "Inner product distance mismatch for special case '{}': scalar={}, dispatch={}",
            case_name, scalar_inner, dispatch_inner
        );
    }
}