//! SIMD-accelerated distance computations using portable SIMD
//!
//! This module provides SIMD implementations for distance calculations
//! with runtime CPU feature detection and fallback to scalar implementations.

#[cfg(feature = "simd")]
use wide::f32x8;

/// Runtime dispatch for L2 (Euclidean) squared distance computation
/// 
/// Automatically selects the best available implementation based on CPU features:
/// - AVX2/AVX-512 SIMD implementation when available  
/// - Scalar fallback for compatibility
/// 
/// # Safety
/// Uses runtime CPU feature detection to ensure SIMD instructions are available
/// before calling SIMD implementations.
/// 
/// # Examples
/// ```
/// use diskann_core::simd::l2_squared_distance_dispatch;
/// 
/// let a = vec![1.0f32, 2.0, 3.0, 4.0];
/// let b = vec![5.0f32, 6.0, 7.0, 8.0]; 
/// let distance = l2_squared_distance_dispatch(&a, &b);
/// ```
pub fn l2_squared_distance_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    #[cfg(feature = "simd")]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We've just checked that AVX2 is available at runtime
                return unsafe { l2_squared_distance_avx2(a, b) };
            }
        }
        
        // Use portable SIMD if supported
        if a.len() >= 8 {
            return l2_squared_distance_simd(a, b);
        }
    }
    
    // Fallback to scalar implementation
    l2_squared_distance_scalar(a, b)
}

/// Runtime dispatch for inner product distance computation
/// 
/// Computes 1.0 - dot_product(a, b) with SIMD acceleration when available.
/// 
/// # Safety
/// Uses runtime CPU feature detection to ensure SIMD instructions are available.
/// 
/// # Examples
/// ```
/// use diskann_core::simd::inner_product_distance_dispatch;
/// 
/// let a = vec![1.0f32, 0.0, 0.0, 0.0];  // normalized vector
/// let b = vec![0.0f32, 1.0, 0.0, 0.0];  // normalized vector
/// let distance = inner_product_distance_dispatch(&a, &b);
/// assert!((distance - 1.0).abs() < 1e-6);
/// ```
pub fn inner_product_distance_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    #[cfg(feature = "simd")]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We've just checked that AVX2 is available at runtime
                return unsafe { inner_product_distance_avx2(a, b) };
            }
        }
        
        // Use portable SIMD if supported
        if a.len() >= 8 {
            return inner_product_distance_simd(a, b);
        }
    }
    
    // Fallback to scalar implementation
    inner_product_distance_scalar(a, b)
}

/// Scalar implementation of L2 squared distance
/// 
/// This is the fallback implementation that works on all architectures.
/// 
/// # Examples
/// ```
/// use diskann_core::simd::l2_squared_distance_scalar;
/// 
/// let a = vec![0.0f32, 0.0];
/// let b = vec![3.0f32, 4.0];
/// let dist_sq = l2_squared_distance_scalar(&a, &b);
/// assert!((dist_sq - 25.0).abs() < 1e-6);
/// ```
pub fn l2_squared_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Scalar implementation of inner product distance
/// 
/// Computes 1.0 - dot_product(a, b). This is the fallback implementation.
/// 
/// # Examples
/// ```
/// use diskann_core::simd::inner_product_distance_scalar;
/// 
/// let a = vec![1.0f32, 0.0];  // normalized
/// let b = vec![0.0f32, 1.0];  // normalized  
/// let distance = inner_product_distance_scalar(&a, &b);
/// assert!((distance - 1.0).abs() < 1e-6);
/// ```
pub fn inner_product_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
    }
    1.0 - dot_product
}

#[cfg(feature = "simd")]
/// SIMD implementation of L2 squared distance using portable SIMD
/// 
/// Uses 256-bit SIMD vectors (8 f32 elements) for acceleration.
/// Falls back to scalar processing for remaining elements.
/// 
/// # Safety
/// This function is safe as it uses the portable SIMD API which handles
/// alignment and availability checks internally.
/// 
/// # Examples
/// ```
/// # #[cfg(feature = "simd")]
/// # {
/// use diskann_core::simd::l2_squared_distance_simd;
/// 
/// let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
/// let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
/// let distance = l2_squared_distance_simd(&a, &b);
/// assert_eq!(distance, 16.0); // Each diff is 1.0, squared = 1.0, sum = 16.0
/// # }
/// ```
pub fn l2_squared_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len - (len % 8);
    let mut sum = f32x8::ZERO;
    
    // Process 8 elements at a time
    let mut i = 0;
    while i < simd_len {
        // SAFETY: We've ensured i + 8 <= simd_len <= len, so slice access is safe
        let chunk_a = f32x8::from([
            a[i], a[i+1], a[i+2], a[i+3], 
            a[i+4], a[i+5], a[i+6], a[i+7]
        ]);
        let chunk_b = f32x8::from([
            b[i], b[i+1], b[i+2], b[i+3], 
            b[i+4], b[i+5], b[i+6], b[i+7]
        ]);
        let diff = chunk_a - chunk_b;
        sum += diff * diff;
        i += 8;
    }
    
    // Sum the SIMD register
    let mut result = sum.to_array().iter().sum::<f32>();
    
    // Process remaining elements
    for j in simd_len..len {
        let diff = a[j] - b[j];
        result += diff * diff;
    }
    
    result
}

#[cfg(feature = "simd")]
/// SIMD implementation of inner product distance using portable SIMD
/// 
/// Computes 1.0 - dot_product(a, b) using 256-bit SIMD vectors.
/// 
/// # Safety
/// This function is safe as it uses the portable SIMD API.
/// 
/// # Examples  
/// ```
/// # #[cfg(feature = "simd")]
/// # {
/// use diskann_core::simd::inner_product_distance_simd;
/// 
/// let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let distance = inner_product_distance_simd(&a, &b);
/// assert!((distance - 1.0).abs() < 1e-6);
/// # }
/// ```
pub fn inner_product_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len - (len % 8);
    let mut sum = f32x8::ZERO;
    
    // Process 8 elements at a time
    let mut i = 0;
    while i < simd_len {
        // SAFETY: We've ensured i + 8 <= simd_len <= len, so slice access is safe
        let chunk_a = f32x8::from([
            a[i], a[i+1], a[i+2], a[i+3], 
            a[i+4], a[i+5], a[i+6], a[i+7]
        ]);
        let chunk_b = f32x8::from([
            b[i], b[i+1], b[i+2], b[i+3], 
            b[i+4], b[i+5], b[i+6], b[i+7]
        ]);
        sum += chunk_a * chunk_b;
        i += 8;
    }
    
    // Sum the SIMD register for dot product
    let mut dot_product = sum.to_array().iter().sum::<f32>();
    
    // Process remaining elements
    for j in simd_len..len {
        dot_product += a[j] * b[j];
    }
    
    1.0 - dot_product
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
/// AVX2-optimized L2 squared distance computation
/// 
/// # Safety
/// This function requires AVX2 support. Call only after verifying with
/// `is_x86_feature_detected!("avx2")`.
/// 
/// The function uses raw pointer access with unchecked bounds, so the caller
/// must ensure that both slices have the same length.
/// 
/// # Examples
/// ```
/// # #[cfg(all(feature = "simd", target_arch = "x86_64"))]
/// # {
/// use diskann_core::simd::l2_squared_distance_avx2;
/// 
/// if is_x86_feature_detected!("avx2") {
///     let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///     let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
///     let distance = unsafe { l2_squared_distance_avx2(&a, &b) };
///     assert_eq!(distance, 8.0); // Each diff is 1.0, squared = 1.0, sum = 8.0
/// }
/// # }
/// ```
#[target_feature(enable = "avx2")]
pub unsafe fn l2_squared_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    
    let len = a.len();
    let simd_len = len - (len % 8);
    let mut sum = _mm256_setzero_ps();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    let mut i = 0;
    while i < simd_len {
        // SAFETY: Caller ensures slices have same length and we stay within bounds
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq_diff = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq_diff);
        i += 8;
    }
    
    // Horizontal sum of the AVX2 register
    let sum_lo = _mm256_castps256_ps128(sum);
    let sum_hi = _mm256_extractf128_ps(sum, 1);
    let sum_128 = _mm_add_ps(sum_lo, sum_hi);
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
    let mut result = _mm_cvtss_f32(sum_32);
    
    // Process remaining elements
    for j in simd_len..len {
        let diff = *a_ptr.add(j) - *b_ptr.add(j);
        result += diff * diff;
    }
    
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
/// AVX2-optimized inner product distance computation
/// 
/// # Safety
/// This function requires AVX2 support. Call only after verifying with
/// `is_x86_feature_detected!("avx2")`.
/// 
/// The function uses raw pointer access, so the caller must ensure that
/// both slices have the same length.
/// 
/// # Examples
/// ```
/// # #[cfg(all(feature = "simd", target_arch = "x86_64"))]
/// # {
/// use diskann_core::simd::inner_product_distance_avx2;
/// 
/// if is_x86_feature_detected!("avx2") {
///     let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///     let b = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///     let distance = unsafe { inner_product_distance_avx2(&a, &b) };
///     assert!((distance - 0.0).abs() < 1e-6); // 1.0 - 1.0 = 0.0
/// }
/// # }
/// ```
#[target_feature(enable = "avx2")]
pub unsafe fn inner_product_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    
    let len = a.len();
    let simd_len = len - (len % 8);
    let mut sum = _mm256_setzero_ps();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    let mut i = 0;
    while i < simd_len {
        // SAFETY: Caller ensures slices have same length and we stay within bounds
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }
    
    // Horizontal sum of the AVX2 register
    let sum_lo = _mm256_castps256_ps128(sum);
    let sum_hi = _mm256_extractf128_ps(sum, 1);
    let sum_128 = _mm_add_ps(sum_lo, sum_hi);
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
    let mut dot_product = _mm_cvtss_f32(sum_32);
    
    // Process remaining elements
    for j in simd_len..len {
        dot_product += *a_ptr.add(j) * *b_ptr.add(j);
    }
    
    1.0 - dot_product
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_l2_squared_distance_scalar() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let distance = l2_squared_distance_scalar(&a, &b);
        assert!((distance - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_distance_scalar() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let distance = inner_product_distance_scalar(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);
        
        let parallel = inner_product_distance_scalar(&a, &a);
        assert!(parallel < 1e-6);
    }

    #[test]
    fn test_l2_squared_distance_dispatch() {
        let a = vec![0.0f32, 0.0, 1.0, 2.0];
        let b = vec![3.0f32, 4.0, 1.0, 2.0];
        let distance = l2_squared_distance_dispatch(&a, &b);
        assert!((distance - 25.0).abs() < 1e-6); // 3^2 + 4^2 + 0^2 + 0^2 = 25
    }

    #[test]
    fn test_inner_product_distance_dispatch() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let distance = inner_product_distance_dispatch(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_l2_squared_distance_simd() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let distance = l2_squared_distance_simd(&a, &b);
        assert!((distance - 16.0).abs() < 1e-6); // Each diff is 1.0, squared = 1.0, sum = 16.0
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_inner_product_distance_simd() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let distance = inner_product_distance_simd(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_implementations() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            
            let l2_distance = unsafe { l2_squared_distance_avx2(&a, &b) };
            assert!((l2_distance - 8.0).abs() < 1e-6);
            
            let inner_distance = unsafe { inner_product_distance_avx2(&a, &a) };
            let expected_dot = a.iter().map(|x| x * x).sum::<f32>();
            let expected_distance = 1.0 - expected_dot;
            assert!((inner_distance - expected_distance).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_scalar_equivalence() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (i as f32 * 0.1) + 0.5).collect();
        
        let scalar_l2 = l2_squared_distance_scalar(&a, &b);
        let dispatch_l2 = l2_squared_distance_dispatch(&a, &b);
        assert!((scalar_l2 - dispatch_l2).abs() < 1e-6);
        
        let scalar_inner = inner_product_distance_scalar(&a, &b);
        let dispatch_inner = inner_product_distance_dispatch(&a, &b);
        assert!((scalar_inner - dispatch_inner).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        // Empty vectors
        let empty: Vec<f32> = vec![];
        assert_eq!(l2_squared_distance_dispatch(&empty, &empty), 0.0);
        assert_eq!(inner_product_distance_dispatch(&empty, &empty), 1.0);
        
        // Mismatched lengths
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(l2_squared_distance_dispatch(&a, &b), f32::INFINITY);
        assert_eq!(inner_product_distance_dispatch(&a, &b), f32::INFINITY);
    }
}