//! Core data structures and algorithms for DiskANN vector search
//!
//! This crate provides the fundamental building blocks for the DiskANN
//! approximate nearest neighbor search system.

#![deny(warnings)]
#![warn(missing_docs)]

/// Core error types for the DiskANN system
pub mod error;

/// Vector and distance computation utilities
pub mod vectors;

/// Basic data structures
pub mod structures;

pub use error::*;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}