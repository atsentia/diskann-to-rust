//! Core traits and interfaces for DiskANN components

#![deny(warnings)]
#![warn(missing_docs)]

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