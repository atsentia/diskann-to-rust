//! I/O operations and file format handling for DiskANN

#![deny(warnings)]
#![warn(missing_docs)]

/// File I/O operations
pub mod file;

/// Index serialization and deserialization  
pub mod serialization;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}