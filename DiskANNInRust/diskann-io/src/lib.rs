//! I/O operations and file format handling for DiskANN

#![deny(warnings)]
#![warn(missing_docs)]

/// File I/O operations
pub mod file;

/// Index serialization and deserialization  
pub mod serialization;

/// Binary format definitions compatible with C++ DiskANN
pub mod format;

/// Memory-mapped I/O with safety wrappers
pub mod mmap;

/// Index writer abstractions
pub mod writer;

/// Index loader abstractions with mmap support
pub mod loader;

/// Disk-based index for memory-efficient storage
pub mod disk_index;

// Re-export main types for convenience
pub use writer::{IndexWriter, FileIndexWriter, IndexWriterBuilder};
pub use loader::{IndexLoader, MmapIndexLoader, IndexLoaderBuilder, IndexMetadata};
pub use format::{BinaryHeader, write_vectors_f32, read_vectors_f32, write_graph_nodes, read_graph_nodes};
pub use serialization::{serialize_index, deserialize_index};
pub use mmap::{MappingStrategy, platform_mmap_info};
pub use disk_index::{DiskIndex, DiskIndexBuilder, DiskIndexConfig};

#[cfg(feature = "mmap")]
pub use mmap::SafeMmap;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}