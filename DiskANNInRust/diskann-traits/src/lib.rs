//! Core traits and interfaces for DiskANN components

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Distance computation traits
pub mod distance;

/// Index traits
pub mod index;

/// Search traits
pub mod search;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}