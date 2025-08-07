//! Core data structures and algorithms for DiskANN vector search
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