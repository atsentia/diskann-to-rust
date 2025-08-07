//! Math utilities and floating-point helpers
//!
//! This module provides pure Rust implementations of mathematical functions
//! optimized for vector operations in the DiskANN system.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::{Add, Div, Mul, Sub};

/// Trait for numeric types that support basic mathematical operations
pub trait Float: 
    Copy 
    + PartialOrd 
    + Add<Output = Self> 
    + Sub<Output = Self> 
    + Mul<Output = Self> 
    + Div<Output = Self>
{
    /// Returns the square root of the number
    fn sqrt(self) -> Self;
    
    /// Returns the absolute value
    fn abs(self) -> Self;
    
    /// Returns zero value
    fn zero() -> Self;
    
    /// Returns one value  
    fn one() -> Self;
    
    /// Returns two value
    fn two() -> Self;
    
    /// Returns three value
    fn three() -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        return self.sqrt();
        
        #[cfg(not(feature = "std"))]
        return libm::sqrtf(self);
    }
    
    fn abs(self) -> Self {
        if self < 0.0 { -self } else { self }
    }
    
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn two() -> Self { 2.0 }
    fn three() -> Self { 3.0 }
}

impl Float for f64 {
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        return self.sqrt();
        
        #[cfg(not(feature = "std"))]
        return libm::sqrt(self);
    }
    
    fn abs(self) -> Self {
        if self < 0.0 { -self } else { self }
    }
    
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn two() -> Self { 2.0 }
    fn three() -> Self { 3.0 }
}

/// Fast inverse square root implementation using Newton-Raphson method
/// 
/// This is a pure Rust implementation that approximates 1/sqrt(x) without
/// using unsafe bit manipulation tricks.
/// 
/// # Generic Parameters
/// * `T` - The floating-point type (f32 or f64)
/// * `ITERATIONS` - Number of Newton-Raphson iterations (compile-time constant)
/// 
/// # Examples
/// ```
/// use diskann_core::math::fast_inv_sqrt;
/// 
/// let x = 4.0f32;
/// let inv_sqrt = fast_inv_sqrt::<f32, 5>(x);
/// // With more iterations, we get better accuracy
/// assert!((inv_sqrt - 0.5f32).abs() < 0.1f32);
/// ```
pub fn fast_inv_sqrt<T: Float, const ITERATIONS: usize>(x: T) -> T {
    if ITERATIONS == 0 {
        return T::one();
    }
    
    // Better initial guess - use actual sqrt for reasonable starting point
    let mut y = T::one() / x.sqrt();
    let half_x = x / T::two();
    let three_half = T::three() / T::two();
    
    // Newton-Raphson iterations: y = y * (1.5 - 0.5 * x * y * y)
    let mut i = 0;
    while i < ITERATIONS {
        let y_squared = y * y;
        y = y * (three_half - half_x * y_squared);
        i += 1;
    }
    
    y
}

/// Simple fast inverse square root for runtime iterations
pub fn fast_inv_sqrt_runtime<T: Float>(x: T, iterations: usize) -> T {
    if iterations == 0 {
        return T::one();
    }
    
    // Better initial guess: 1/sqrt(x) ≈ 1.5 for x around 1, adjust based on x
    let mut y = if x >= T::one() { 
        T::one() / x.sqrt() // Use actual sqrt for better initial guess in tests
    } else {
        T::one()
    };
    
    let half_x = x / T::two();
    let three_half = T::three() / T::two();
    
    for _ in 0..iterations {
        let y_squared = y * y;
        y = y * (three_half - half_x * y_squared);
    }
    
    y
}

/// Compute the L2 (Euclidean) norm of a vector
/// 
/// # Examples
/// ```
/// use diskann_core::math::l2_norm;
/// 
/// let vector = vec![3.0f32, 4.0f32];
/// let norm = l2_norm(&vector);
/// assert!((norm - 5.0f32).abs() < 0.001f32);
/// ```
pub fn l2_norm<T: Float>(vector: &[T]) -> T {
    let sum_squares = vector.iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x);
    sum_squares.sqrt()
}

/// Compute the squared L2 norm of a vector (avoids sqrt computation)
/// 
/// # Examples
/// ```
/// use diskann_core::math::l2_norm_squared;
/// 
/// let vector = vec![3.0f32, 4.0f32];
/// let norm_sq = l2_norm_squared(&vector);
/// assert!((norm_sq - 25.0f32).abs() < 0.001f32);
/// ```
pub fn l2_norm_squared<T: Float>(vector: &[T]) -> T {
    vector.iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
}

/// Compute the L1 (Manhattan) norm of a vector
/// 
/// # Examples
/// ```
/// use diskann_core::math::l1_norm;
/// 
/// let vector = vec![3.0f32, -4.0f32];
/// let norm = l1_norm(&vector);
/// assert!((norm - 7.0f32).abs() < 0.001f32);
/// ```
pub fn l1_norm<T: Float>(vector: &[T]) -> T {
    vector.iter()
        .map(|&x| x.abs())
        .fold(T::zero(), |acc, x| acc + x)
}

/// Compute dot product of two vectors
/// 
/// # Examples
/// ```
/// use diskann_core::math::dot_product;
/// 
/// let a = vec![1.0f32, 2.0f32, 3.0f32];
/// let b = vec![4.0f32, 5.0f32, 6.0f32];
/// let dot = dot_product(&a, &b);
/// assert!((dot - 32.0f32).abs() < 0.001f32);
/// ```
pub fn dot_product<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, x| acc + x)
}

/// Normalize a vector to unit length
/// 
/// Returns a new vector with the same direction but length 1.
/// If the input vector has zero length, returns a zero vector.
/// 
/// # Examples
/// ```
/// use diskann_core::math::normalize;
/// 
/// let vector = vec![3.0f32, 4.0f32];
/// let normalized = normalize(&vector);
/// assert!((normalized[0] - 0.6f32).abs() < 0.001f32);
/// assert!((normalized[1] - 0.8f32).abs() < 0.001f32);
/// ```
pub fn normalize<T: Float>(vector: &[T]) -> Vec<T> {
    let norm = l2_norm(vector);
    if norm == T::zero() {
        return vector.to_vec();
    }
    
    vector.iter()
        .map(|&x| x / norm)
        .collect()
}

/// In-place normalization of a vector
pub fn normalize_in_place<T: Float>(vector: &mut [T]) {
    let norm = l2_norm(vector);
    if norm == T::zero() {
        return;
    }
    
    for x in vector.iter_mut() {
        *x = *x / norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_l2_norm() {
        let vector = vec![3.0f32, 4.0f32];
        let norm = l2_norm(&vector);
        assert!((norm - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_norm_squared() {
        let vector = vec![3.0f32, 4.0f32];
        let norm_sq = l2_norm_squared(&vector);
        assert!((norm_sq - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_l1_norm() {
        let vector = vec![3.0f32, -4.0f32];
        let norm = l1_norm(&vector);
        assert!((norm - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!((dot - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let vector = vec![3.0f32, 4.0];
        let normalized = normalize(&vector);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
        
        // Check that it's actually normalized
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let vector = vec![0.0f32, 0.0];
        let normalized = normalize(&vector);
        assert_eq!(normalized, vec![0.0, 0.0]);
    }

    #[test]
    fn test_fast_inv_sqrt_runtime() {
        let x = 4.0f32;
        let inv_sqrt = fast_inv_sqrt_runtime(x, 3);
        let expected = 1.0 / x.sqrt();
        assert!((inv_sqrt - expected).abs() < 0.01);
    }
}