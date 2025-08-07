//! Index loader abstractions with memory-mapped I/O support

use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use anyhow::{Result, Context};
use diskann_core::structures::GraphNode;
use diskann_core::vectors::{Vector, VectorId};
use crate::mmap::{platform_mmap_info};
use crate::format::{BinaryHeader, read_vectors_f32};

/// Trait for loading indices from persistent storage
pub trait IndexLoader {
    /// Load vector data from storage
    fn load_vectors<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>>;
    
    /// Load graph nodes (vectors + adjacency lists) from storage
    fn load_graph<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GraphNode>>;
    
    /// Load only a subset of vectors for testing or partial loading
    fn load_vectors_subset<P: AsRef<Path>>(&self, path: P, start: usize, count: usize) -> Result<Vec<Vector>>;
    
    /// Get file metadata without loading the full data
    fn get_metadata<P: AsRef<Path>>(&self, path: P) -> Result<IndexMetadata>;
}

/// Metadata about an index file
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// Number of vectors in the file
    pub num_vectors: usize,
    /// Dimension of each vector
    pub dimension: usize,
    /// Total file size in bytes
    pub file_size: usize,
    /// Strategy being used (mmap or buffered)
    pub strategy: String,
}

/// Memory-mapped index loader with fallback to buffered I/O
pub struct MmapIndexLoader {
    /// Whether to prefer memory mapping when available
    prefer_mmap: bool,
}

impl MmapIndexLoader {
    /// Create a new mmap loader that prefers memory mapping
    pub fn new() -> Self {
        Self {
            prefer_mmap: true,
        }
    }
    
    /// Create a loader that always uses buffered I/O
    pub fn buffered_only() -> Self {
        Self {
            prefer_mmap: false,
        }
    }
    
    /// Load vectors using the optimal strategy for the platform
    fn load_vectors_with_strategy<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>> {
        #[cfg(feature = "mmap")]
        if self.prefer_mmap {
            return self.load_vectors_mmap(path.as_ref());
        }
        
        self.load_vectors_buffered(path.as_ref())
    }
    
    /// Load vectors using memory mapping (if available)
    #[cfg(feature = "mmap")]
    #[allow(dead_code)]
    fn load_vectors_mmap<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>> {
        use crate::mmap::MappingStrategy;
        
        let mapping = MappingStrategy::new(path.as_ref())?;
        
        // Read header first
        let header_bytes = mapping.slice(0, std::mem::size_of::<BinaryHeader>())?;
        let header = unsafe { 
            *(header_bytes.as_ptr() as *const BinaryHeader)
        };
        header.validate()?;
        
        let num_points = header.num_points as usize;
        let num_dimensions = header.num_dimensions as usize;
        
        // Read vector data using zero-copy when possible
        let data_offset = std::mem::size_of::<BinaryHeader>();
        let f32_data = mapping.typed_slice::<f32>(data_offset, num_points * num_dimensions)?;
        
        // Convert to Vector format
        let mut vectors = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let start_idx = i * num_dimensions;
            let end_idx = start_idx + num_dimensions;
            vectors.push(f32_data[start_idx..end_idx].to_vec());
        }
        
        tracing::info!("Loaded {} vectors using {} strategy", 
                      vectors.len(), mapping.strategy_info());
        
        Ok(vectors)
    }
    
    /// Fallback for platforms without mmap
    #[cfg(not(feature = "mmap"))]
    #[allow(dead_code)]
    fn load_vectors_mmap<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>> {
        tracing::warn!("Memory mapping not available, falling back to buffered I/O");
        self.load_vectors_buffered(path)
    }
    
    /// Load vectors using buffered I/O
    fn load_vectors_buffered<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;
        
        let mut reader = BufReader::with_capacity(64 * 1024 * 1024, file); // 64MB buffer
        
        let vectors = read_vectors_f32(&mut reader)
            .context("Failed to read vector data")?;
        
        tracing::info!("Loaded {} vectors using buffered I/O", vectors.len());
        
        Ok(vectors)
    }
}

impl Default for MmapIndexLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexLoader for MmapIndexLoader {
    fn load_vectors<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vector>> {
        self.load_vectors_with_strategy(path)
    }
    
    fn load_graph<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GraphNode>> {
        // For now, just load as vectors and convert to graph nodes
        // TODO: Add proper graph adjacency list loading
        let vectors = self.load_vectors(path)?;
        
        let nodes = vectors
            .into_iter()
            .enumerate()
            .map(|(i, vector)| GraphNode::new(i as VectorId, vector))
            .collect();
            
        Ok(nodes)
    }
    
    fn load_vectors_subset<P: AsRef<Path>>(&self, path: P, start: usize, count: usize) -> Result<Vec<Vector>> {
        // For simplicity, load all vectors then take subset
        // TODO: Optimize to only load required portion
        let all_vectors = self.load_vectors(path)?;
        
        if start >= all_vectors.len() {
            anyhow::bail!("Start index {} exceeds vector count {}", start, all_vectors.len());
        }
        
        let end = std::cmp::min(start + count, all_vectors.len());
        Ok(all_vectors[start..end].to_vec())
    }
    
    fn get_metadata<P: AsRef<Path>>(&self, path: P) -> Result<IndexMetadata> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;
        
        let mut reader = BufReader::new(file);
        let header = BinaryHeader::read_from(&mut reader)
            .context("Failed to read file header")?;
        
        header.validate()?;
        
        let file_size = std::fs::metadata(path.as_ref())
            .context("Failed to get file metadata")?
            .len() as usize;
        
        Ok(IndexMetadata {
            num_vectors: header.num_points as usize,
            dimension: header.num_dimensions as usize,
            file_size,
            strategy: if self.prefer_mmap { 
                format!("mmap-preferred ({})", platform_mmap_info())
            } else { 
                "buffered-only".to_string() 
            },
        })
    }
}

/// Builder for creating index loaders with different configurations
pub struct IndexLoaderBuilder {
    prefer_mmap: bool,
}

impl IndexLoaderBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            prefer_mmap: true,
        }
    }
    
    /// Set whether to prefer memory mapping when available
    pub fn prefer_mmap(mut self, prefer: bool) -> Self {
        self.prefer_mmap = prefer;
        self
    }
    
    /// Build a memory-mapped index loader
    pub fn build_mmap_loader(self) -> MmapIndexLoader {
        if self.prefer_mmap {
            MmapIndexLoader::new()
        } else {
            MmapIndexLoader::buffered_only()
        }
    }
}

impl Default for IndexLoaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{IndexWriter, FileIndexWriter};
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_mmap_loader_round_trip() {
        let writer = FileIndexWriter::new();
        let loader = MmapIndexLoader::new();
        
        let original_vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let temp_file = NamedTempFile::new().unwrap();
        
        // Write vectors
        writer.write_vectors(temp_file.path(), &original_vectors).unwrap();
        
        // Load vectors back
        let loaded_vectors = loader.load_vectors(temp_file.path()).unwrap();
        
        assert_eq!(original_vectors.len(), loaded_vectors.len());
        for (orig, loaded) in original_vectors.iter().zip(loaded_vectors.iter()) {
            assert_eq!(orig.len(), loaded.len());
            for (&orig_val, &loaded_val) in orig.iter().zip(loaded.iter()) {
                assert!((orig_val - loaded_val).abs() < f32::EPSILON);
            }
        }
    }
    
    #[test]
    fn test_metadata_extraction() {
        let writer = FileIndexWriter::new();
        let loader = MmapIndexLoader::new();
        
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_vectors(temp_file.path(), &vectors).unwrap();
        
        let metadata = loader.get_metadata(temp_file.path()).unwrap();
        
        assert_eq!(metadata.num_vectors, 2);
        assert_eq!(metadata.dimension, 4);
        assert_eq!(metadata.file_size, 8 + (2 * 4 * 4)); // header + data
    }
    
    #[test]
    fn test_vectors_subset() {
        let writer = FileIndexWriter::new();
        let loader = MmapIndexLoader::new();
        
        let vectors = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_vectors(temp_file.path(), &vectors).unwrap();
        
        let subset = loader.load_vectors_subset(temp_file.path(), 1, 2).unwrap();
        
        assert_eq!(subset.len(), 2);
        assert_eq!(subset[0], vec![3.0, 4.0]);
        assert_eq!(subset[1], vec![5.0, 6.0]);
    }
    
    #[test]
    fn test_buffered_only_loader() {
        let loader = MmapIndexLoader::buffered_only();
        assert!(!loader.prefer_mmap);
    }
    
    #[test]
    fn test_builder_pattern() {
        let loader = IndexLoaderBuilder::new()
            .prefer_mmap(false)
            .build_mmap_loader();
        
        assert!(!loader.prefer_mmap);
    }
}