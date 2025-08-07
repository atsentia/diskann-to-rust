//! Bit manipulation and alignment utilities
//!
//! This module provides utilities for memory alignment, bit manipulation,
//! and low-level operations needed by the DiskANN system.

/// Round up X to the nearest multiple of Y
/// 
/// # Examples
/// ```
/// use diskann_core::utils::round_up;
/// 
/// assert_eq!(round_up(15, 8), 16);
/// assert_eq!(round_up(16, 8), 16);
/// assert_eq!(round_up(17, 8), 24);
/// ```
pub const fn round_up(x: u64, y: u64) -> u64 {
    ((x / y) + if x % y != 0 { 1 } else { 0 }) * y
}

/// Round down X to the nearest multiple of Y
/// 
/// # Examples
/// ```
/// use diskann_core::utils::round_down;
/// 
/// assert_eq!(round_down(15, 8), 8);
/// assert_eq!(round_down(16, 8), 16);
/// assert_eq!(round_down(17, 8), 16);
/// ```
pub const fn round_down(x: u64, y: u64) -> u64 {
    (x / y) * y
}

/// Divide and round up
/// 
/// # Examples
/// ```
/// use diskann_core::utils::div_round_up;
/// 
/// assert_eq!(div_round_up(15, 8), 2);
/// assert_eq!(div_round_up(16, 8), 2);
/// assert_eq!(div_round_up(17, 8), 3);
/// ```
pub const fn div_round_up(x: u64, y: u64) -> u64 {
    (x / y) + if x % y != 0 { 1 } else { 0 }
}

/// Check if X is aligned to Y bytes
/// 
/// # Examples
/// ```
/// use diskann_core::utils::is_aligned;
/// 
/// assert!(is_aligned(16, 8));
/// assert!(is_aligned(24, 8));
/// assert!(!is_aligned(15, 8));
/// ```
pub const fn is_aligned(x: u64, y: u64) -> bool {
    x % y == 0
}

/// Check if X is aligned to 512 bytes
/// 
/// # Examples
/// ```
/// use diskann_core::utils::is_512_aligned;
/// 
/// assert!(is_512_aligned(512));
/// assert!(is_512_aligned(1024));
/// assert!(!is_512_aligned(256));
/// ```
pub const fn is_512_aligned(x: u64) -> bool {
    is_aligned(x, 512)
}

/// Check if X is aligned to 4096 bytes (page size)
/// 
/// # Examples
/// ```
/// use diskann_core::utils::is_4096_aligned;
/// 
/// assert!(is_4096_aligned(4096));
/// assert!(is_4096_aligned(8192));
/// assert!(!is_4096_aligned(2048));
/// ```
pub const fn is_4096_aligned(x: u64) -> bool {
    is_aligned(x, 4096)
}

/// Check if X is aligned to 32 bytes (SIMD alignment)
/// 
/// # Examples
/// ```
/// use diskann_core::utils::is_32_aligned;
/// 
/// assert!(is_32_aligned(32));
/// assert!(is_32_aligned(64));
/// assert!(!is_32_aligned(16));
/// ```
pub const fn is_32_aligned(x: u64) -> bool {
    is_aligned(x, 32)
}

/// Metadata size constant (4KB for unified files)
pub const METADATA_SIZE: usize = 4096;

/// Buffer size for cached I/O operations
pub const BUFFER_SIZE_FOR_CACHED_IO: usize = 1024 * 1048576; // 1GB

/// Calculate the next power of 2 greater than or equal to x
/// 
/// # Examples
/// ```
/// use diskann_core::utils::next_power_of_2;
/// 
/// assert_eq!(next_power_of_2(5), 8);
/// assert_eq!(next_power_of_2(8), 8);
/// assert_eq!(next_power_of_2(9), 16);
/// ```
pub const fn next_power_of_2(mut x: u64) -> u64 {
    if x == 0 {
        return 1;
    }
    
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x + 1
}

/// Check if a number is a power of 2
/// 
/// # Examples
/// ```
/// use diskann_core::utils::is_power_of_2;
/// 
/// assert!(is_power_of_2(8));
/// assert!(is_power_of_2(16));
/// assert!(!is_power_of_2(10));
/// assert!(!is_power_of_2(0));
/// ```
pub const fn is_power_of_2(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Count the number of set bits in an integer (population count)
/// 
/// # Examples
/// ```
/// use diskann_core::utils::popcount;
/// 
/// assert_eq!(popcount(0b1010), 2);
/// assert_eq!(popcount(0b1111), 4);
/// assert_eq!(popcount(0), 0);
/// ```
pub const fn popcount(mut x: u64) -> u32 {
    let mut count = 0;
    while x != 0 {
        count += 1;
        x &= x - 1; // Remove the lowest set bit
    }
    count
}

/// Find the position of the least significant set bit (0-indexed)
/// Returns None if x is 0
/// 
/// # Examples
/// ```
/// use diskann_core::utils::trailing_zeros;
/// 
/// assert_eq!(trailing_zeros(8), Some(3));  // 0b1000
/// assert_eq!(trailing_zeros(12), Some(2)); // 0b1100
/// assert_eq!(trailing_zeros(0), None);
/// ```
pub const fn trailing_zeros(x: u64) -> Option<u32> {
    if x == 0 {
        return None;
    }
    
    let mut count = 0;
    let mut val = x;
    while val & 1 == 0 {
        count += 1;
        val >>= 1;
    }
    Some(count)
}

/// Find the position of the most significant set bit (0-indexed from right)
/// Returns None if x is 0
/// 
/// # Examples
/// ```
/// use diskann_core::utils::leading_zeros;
/// 
/// assert_eq!(leading_zeros(8), Some(60));  // 0b1000 in 64-bit
/// assert_eq!(leading_zeros(0), None);
/// ```
pub const fn leading_zeros(mut x: u64) -> Option<u32> {
    if x == 0 {
        return None;
    }
    
    let mut count = 0;
    while x & (1u64 << 63) == 0 {
        count += 1;
        x <<= 1;
    }
    Some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up() {
        assert_eq!(round_up(15, 8), 16);
        assert_eq!(round_up(16, 8), 16);
        assert_eq!(round_up(17, 8), 24);
        assert_eq!(round_up(0, 8), 0);
    }

    #[test]
    fn test_round_down() {
        assert_eq!(round_down(15, 8), 8);
        assert_eq!(round_down(16, 8), 16);
        assert_eq!(round_down(17, 8), 16);
        assert_eq!(round_down(7, 8), 0);
    }

    #[test]
    fn test_div_round_up() {
        assert_eq!(div_round_up(15, 8), 2);
        assert_eq!(div_round_up(16, 8), 2);
        assert_eq!(div_round_up(17, 8), 3);
        assert_eq!(div_round_up(0, 8), 0);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(16, 8));
        assert!(is_aligned(24, 8));
        assert!(!is_aligned(15, 8));
        assert!(is_aligned(0, 8));
    }

    #[test]
    fn test_alignment_specific() {
        assert!(is_512_aligned(512));
        assert!(is_512_aligned(1024));
        assert!(!is_512_aligned(256));

        assert!(is_4096_aligned(4096));
        assert!(is_4096_aligned(8192));
        assert!(!is_4096_aligned(2048));

        assert!(is_32_aligned(32));
        assert!(is_32_aligned(64));
        assert!(!is_32_aligned(16));
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
    }

    #[test]
    fn test_is_power_of_2() {
        assert!(!is_power_of_2(0));
        assert!(is_power_of_2(1));
        assert!(is_power_of_2(2));
        assert!(!is_power_of_2(3));
        assert!(is_power_of_2(4));
        assert!(!is_power_of_2(5));
        assert!(!is_power_of_2(6));
        assert!(!is_power_of_2(7));
        assert!(is_power_of_2(8));
    }

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(0b1), 1);
        assert_eq!(popcount(0b10), 1);
        assert_eq!(popcount(0b11), 2);
        assert_eq!(popcount(0b1010), 2);
        assert_eq!(popcount(0b1111), 4);
        assert_eq!(popcount(u64::MAX), 64);
    }

    #[test]
    fn test_trailing_zeros() {
        assert_eq!(trailing_zeros(0), None);
        assert_eq!(trailing_zeros(1), Some(0));
        assert_eq!(trailing_zeros(2), Some(1));
        assert_eq!(trailing_zeros(4), Some(2));
        assert_eq!(trailing_zeros(8), Some(3));
        assert_eq!(trailing_zeros(12), Some(2)); // 0b1100
    }

    #[test]
    fn test_leading_zeros() {
        assert_eq!(leading_zeros(0), None);
        assert_eq!(leading_zeros(1), Some(63));
        assert_eq!(leading_zeros(2), Some(62));
        assert_eq!(leading_zeros(4), Some(61));
        assert_eq!(leading_zeros(8), Some(60));
    }
}