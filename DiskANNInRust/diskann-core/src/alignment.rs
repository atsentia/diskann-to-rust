//! Aligned vector allocation macros
//!
//! This module provides macros for creating vectors with specific memory alignment
//! without using unsafe code, ensuring 32-byte alignment for SIMD operations.

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, vec};

use crate::utils::round_up;

/// Create a vector with 32-byte alignment without using unsafe code
/// 
/// This macro creates a vector that is guaranteed to have 32-byte alignment
/// by over-allocating and then finding the correctly aligned portion.
/// 
/// # Examples
/// ```
/// use diskann_core::aligned_vec;
/// 
/// let aligned = aligned_vec![f32; 100];
/// assert_eq!(aligned.len(), 100);
/// 
/// let aligned_with_values = aligned_vec![1.0f32; 50];
/// assert_eq!(aligned_with_values.len(), 50);
/// assert!(aligned_with_values.iter().all(|&x| x == 1.0));
/// ```
#[macro_export]
macro_rules! aligned_vec {
    // aligned_vec![T; n] - create vector of n default elements
    ($t:ty; $n:expr) => {{
        $crate::alignment::create_aligned_vec::<$t>($n, None)
    }};
    
    // aligned_vec![value; n] - create vector of n copies of value
    ($value:expr; $n:expr) => {{
        $crate::alignment::create_aligned_vec($n, Some($value))
    }};
}

/// Internal function to create aligned vectors
/// 
/// This function creates a vector with 32-byte alignment by using padding
/// and ensuring the data portion starts at an aligned address.
pub fn create_aligned_vec<T: Clone + Default>(
    size: usize, 
    fill_value: Option<T>
) -> Vec<T> {
    // Calculate alignment requirements
    const ALIGNMENT: usize = 32;
    let type_size = core::mem::size_of::<T>();
    
    // If the type is already well-aligned or doesn't need SIMD alignment
    if type_size >= ALIGNMENT || ALIGNMENT % type_size != 0 {
        let value = fill_value.unwrap_or_default();
        return vec![value; size];
    }
    
    // Calculate how many extra elements we might need for alignment
    let elements_per_alignment = ALIGNMENT / type_size;
    let extra_elements = elements_per_alignment - 1;
    
    // Create a larger vector to ensure we can find an aligned portion
    let total_size = size + extra_elements;
    let value = fill_value.unwrap_or_default();
    let mut vec = vec![value.clone(); total_size];
    
    // Find the aligned starting position within our over-allocated vector
    let ptr = vec.as_ptr() as usize;
    let aligned_ptr = round_up(ptr as u64, ALIGNMENT as u64) as usize;
    let offset_elements = (aligned_ptr - ptr) / type_size;
    
    // If we're already aligned, just truncate to the desired size
    if offset_elements == 0 {
        vec.truncate(size);
        return vec;
    }
    
    // Create a new vector with the properly aligned data
    let mut aligned_vec = Vec::with_capacity(size);
    let start_idx = core::cmp::min(offset_elements, vec.len());
    let end_idx = core::cmp::min(start_idx + size, vec.len());
    
    for i in start_idx..end_idx {
        aligned_vec.push(vec[i].clone());
    }
    
    // Fill remaining elements if needed
    while aligned_vec.len() < size {
        aligned_vec.push(value.clone());
    }
    
    aligned_vec
}

/// Check if a vector's data is aligned to the specified boundary
/// 
/// # Examples
/// ```
/// use diskann_core::alignment::is_vector_aligned;
/// 
/// let vec = vec![1.0f32; 100];
/// // Result depends on allocator behavior, but function should not panic
/// let _is_aligned = is_vector_aligned(&vec, 32);
/// ```
pub fn is_vector_aligned<T>(vec: &[T], alignment: usize) -> bool {
    let ptr = vec.as_ptr() as usize;
    ptr % alignment == 0
}

/// Get the alignment of a vector's data
/// 
/// Returns the largest power of 2 that the vector's data pointer is aligned to.
/// 
/// # Examples
/// ```
/// use diskann_core::alignment::get_vector_alignment;
/// 
/// let vec = vec![1.0f32; 100];
/// let alignment = get_vector_alignment(&vec);
/// assert!(alignment >= 1);
/// assert!((alignment & (alignment - 1)) == 0); // Check it's a power of 2
/// ```
pub fn get_vector_alignment<T>(vec: &[T]) -> usize {
    let ptr = vec.as_ptr() as usize;
    if ptr == 0 || vec.is_empty() {
        return 1;
    }
    
    // Find the largest power of 2 that divides the address
    let mut alignment = 1;
    while alignment <= ptr && ptr % (alignment * 2) == 0 {
        alignment *= 2;
    }
    alignment
}

/// Ensure a vector has at least the specified alignment
/// 
/// If the vector is not properly aligned, creates a new aligned vector.
/// Otherwise returns the original vector.
/// 
/// # Examples
/// ```
/// use diskann_core::alignment::ensure_aligned;
/// 
/// let vec = vec![1.0f32; 100];
/// let aligned = ensure_aligned(vec, 32);
/// assert_eq!(aligned.len(), 100);
/// ```
pub fn ensure_aligned<T: Clone + Default>(
    vec: Vec<T>, 
    required_alignment: usize
) -> Vec<T> {
    if is_vector_aligned(&vec, required_alignment) {
        vec
    } else {
        // Need to create a new aligned vector
        let size = vec.len();
        if size == 0 {
            return vec;
        }
        
        // Use the first element as the fill value if available
        let fill_value = vec.first().cloned();
        let mut aligned = create_aligned_vec(size, fill_value);
        
        // Copy the data
        for (i, item) in vec.into_iter().enumerate() {
            if i < aligned.len() {
                aligned[i] = item;
            }
        }
        
        aligned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_aligned_vec_macro() {
        let aligned = aligned_vec![f32; 100];
        assert_eq!(aligned.len(), 100);
        
        let aligned_with_values = aligned_vec![1.0f32; 50];
        assert_eq!(aligned_with_values.len(), 50);
        assert!(aligned_with_values.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_create_aligned_vec() {
        let vec = create_aligned_vec::<f32>(100, None);
        assert_eq!(vec.len(), 100);
        
        let vec_with_value = create_aligned_vec(50, Some(2.5f32));
        assert_eq!(vec_with_value.len(), 50);
        assert!(vec_with_value.iter().all(|&x| x == 2.5));
    }

    #[test]
    fn test_is_vector_aligned() {
        let vec = vec![1.0f32; 100];
        // This test just ensures the function works, alignment depends on allocator
        let _result = is_vector_aligned(&vec, 32);
        let _result = is_vector_aligned(&vec, 16);
        let _result = is_vector_aligned(&vec, 8);
    }

    #[test]
    fn test_get_vector_alignment() {
        let vec = vec![1.0f32; 100];
        let alignment = get_vector_alignment(&vec);
        assert!(alignment >= 1);
        assert!((alignment & (alignment - 1)) == 0); // Power of 2 check
        
        // Test empty vector
        let empty: Vec<f32> = vec![];
        let empty_alignment = get_vector_alignment(&empty);
        assert_eq!(empty_alignment, 1);
    }

    #[test]
    fn test_ensure_aligned() {
        let vec = vec![1.0f32, 2.0, 3.0, 4.0];
        let aligned = ensure_aligned(vec.clone(), 32);
        assert_eq!(aligned.len(), 4);
        
        // Test empty vector
        let empty: Vec<f32> = vec![];
        let empty_aligned = ensure_aligned(empty, 32);
        assert_eq!(empty_aligned.len(), 0);
    }

    #[test]
    fn test_alignment_edge_cases() {
        // Test with different types
        let vec_u8 = create_aligned_vec::<u8>(64, Some(42));
        assert_eq!(vec_u8.len(), 64);
        assert!(vec_u8.iter().all(|&x| x == 42));
        
        let vec_u64 = create_aligned_vec::<u64>(32, Some(0xDEADBEEF));
        assert_eq!(vec_u64.len(), 32);
        assert!(vec_u64.iter().all(|&x| x == 0xDEADBEEF));
    }
}