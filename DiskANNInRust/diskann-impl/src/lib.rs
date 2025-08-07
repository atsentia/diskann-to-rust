//! Concrete implementations of DiskANN algorithms and data structures

#![deny(warnings)]
#![warn(missing_docs)]

/// Graph-based index implementations
pub mod graph;

/// Search algorithm implementations
pub mod search;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}