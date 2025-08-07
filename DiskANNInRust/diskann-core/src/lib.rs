//! # DiskANN Core Library
//! 
//! This crate provides the fundamental building blocks for the DiskANN vector search implementation,
//! including SIMD-optimized distance functions, memory-aligned data structures, and core algorithms.
//! 
//! ## Key Features
//! 
//! - **SIMD-optimized distance calculations** for high performance
//! - **Memory-aligned data structures** for cache efficiency  
//! - **Zero-cost abstractions** with compile-time optimizations
//! - **Comprehensive testing** with property-based validation
//! 
//! ## Basic Usage
//! 
//! ```rust
//! use diskann_core::math::{euclidean_distance, normalize};
//! 
//! // Calculate distance between two vectors
//! let a = [1.0, 2.0, 3.0];
//! let b = [4.0, 5.0, 6.0];
//! let distance = euclidean_distance(&a, &b);
//! 
//! // Normalize a vector to unit length
//! let mut vector = [3.0, 4.0, 0.0];
//! normalize(&mut vector);
//! assert!((vector[0] * vector[0] + vector[1] * vector[1] - 1.0).abs() < 1e-6);
//! ```
//! 
//! ## SIMD Optimization Example
//! 
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use diskann_core::math::euclidean_distance;
//! 
//! // Standard usage automatically selects SIMD when available
//! let query = vec![1.0; 128];
//! let candidate = vec![2.0; 128];
//! 
//! let distance = euclidean_distance(&query, &candidate);
//! println!("Distance: {}", distance);
//! # Ok(())
//! # }
//! ```
//!
//! This crate provides the fundamental building blocks for the DiskANN
//! approximate nearest neighbor search system.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![deny(warnings)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Core error types for the DiskANN system
pub mod error;

/// Vector and distance computation utilities
pub mod vectors;

/// Basic data structures
pub mod structures;

/// Math utilities and floating-point helpers
pub mod math;

/// Bit manipulation and alignment utilities
pub mod utils;

/// Aligned vector allocation macros
pub mod alignment;

/// SIMD-accelerated distance computations
#[cfg(feature = "simd")]
pub mod simd;

pub use error::*;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}