#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use diskann_core::math::{dot_product, l2_norm, l2_norm_squared, normalize};
use diskann_traits::distance::{EuclideanDistance, CosineDistance, InnerProductDistance, Distance};

#[derive(Arbitrary, Debug)]
struct VectorOperations {
    vec1: Vec<f32>,
    vec2: Vec<f32>,
    scalar: f32,
}

fuzz_target!(|input: VectorOperations| {
    // Skip empty vectors or extremely large ones
    if input.vec1.is_empty() || input.vec1.len() > 10000 {
        return;
    }
    
    // Test math operations with single vector
    let v1 = &input.vec1;
    
    // These should not panic for any input
    let _ = l2_norm(v1);
    let _ = l2_norm_squared(v1);
    
    // Test normalization - should handle zero vectors gracefully
    let mut normalized = v1.clone();
    let _ = normalize(&mut normalized);
    
    // Test with second vector if dimensions match
    if input.vec2.len() == input.vec1.len() && !input.vec2.is_empty() {
        let v2 = &input.vec2;
        
        // Test dot product
        let _ = dot_product(v1, v2);
        
        // Test distance functions - use default constructors
        let euclidean = EuclideanDistance::default();
        let cosine = CosineDistance::default();
        let inner_product = InnerProductDistance::default();
        
        // These should not panic for any valid vectors
        let _ = euclidean.distance(v1, v2);
        let _ = cosine.distance(v1, v2);
        let _ = inner_product.distance(v1, v2);
    }
    
    // Test operations with special float values
    if !input.scalar.is_nan() && input.scalar.is_finite() {
        let mut scaled = v1.clone();
        for x in &mut scaled {
            *x *= input.scalar;
        }
        
        // These should handle scaled vectors properly
        let _ = l2_norm(&scaled);
        let _ = normalize(&mut scaled);
    }
    
    // Test triangle inequality property for distance functions
    if input.vec1.len() == input.vec2.len() && input.vec1.len() >= 3 {
        let v1 = &input.vec1[..3]; // Use first 3 elements
        let v2 = &input.vec2[..3];
        let v3 = &[0.0f32; 3]; // Zero vector as third point
        
        let euclidean = EuclideanDistance::default();
        
        let d12 = euclidean.distance(v1, v2);
        let d13 = euclidean.distance(v1, v3);
        let d23 = euclidean.distance(v2, v3);
        
        // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        // This is a critical invariant that should always hold
        if d12.is_finite() && d13.is_finite() && d23.is_finite() {
            assert!(d13 <= d12 + d23 + f32::EPSILON, 
                "Triangle inequality violated: d({:?}, {:?}) = {} > {} + {} = {}",
                v1, v3, d13, d12, d23, d12 + d23);
        }
    }
});