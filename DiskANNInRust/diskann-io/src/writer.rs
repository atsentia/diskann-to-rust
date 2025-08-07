//! Index writer abstractions for persistent storage

use std::path::Path;
use std::fs::File;
use std::io::{BufWriter, Write};
use anyhow::{Result, Context};
use diskann_core::structures::GraphNode;
use diskann_core::vectors::Vector;
use crate::format::{write_vectors_f32, write_graph_nodes};

/// Trait for writing indices to persistent storage
pub trait IndexWriter {
    /// Write vector data to storage
    fn write_vectors<P: AsRef<Path>>(&self, path: P, vectors: &[Vector]) -> Result<usize>;
    
    /// Write graph nodes (vectors + adjacency lists) to storage  
    fn write_graph<P: AsRef<Path>>(&self, path: P, nodes: &[GraphNode]) -> Result<usize>;
    
    /// Write with custom buffer size for performance tuning
    fn write_vectors_buffered<P: AsRef<Path>>(&self, path: P, vectors: &[Vector], buffer_size: usize) -> Result<usize>;
}

/// Standard file-based index writer
pub struct FileIndexWriter {
    /// Default buffer size for writes
    buffer_size: usize,
}

impl FileIndexWriter {
    /// Create a new file index writer with default buffer size
    pub fn new() -> Self {
        Self {
            buffer_size: 64 * 1024 * 1024, // 64MB default buffer
        }
    }
    
    /// Create a new file index writer with custom buffer size
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        Self { buffer_size }
    }
}

impl Default for FileIndexWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexWriter for FileIndexWriter {
    fn write_vectors<P: AsRef<Path>>(&self, path: P, vectors: &[Vector]) -> Result<usize> {
        self.write_vectors_buffered(path, vectors, self.buffer_size)
    }
    
    fn write_graph<P: AsRef<Path>>(&self, path: P, nodes: &[GraphNode]) -> Result<usize> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {}", path.as_ref().display()))?;
        
        let mut writer = BufWriter::with_capacity(self.buffer_size, file);
        
        let bytes_written = write_graph_nodes(&mut writer, nodes)
            .context("Failed to write graph nodes")?;
        
        writer.flush()
            .context("Failed to flush writer")?;
        
        tracing::info!("Wrote {} graph nodes to {}, {} bytes total", 
                      nodes.len(), path.as_ref().display(), bytes_written);
        
        Ok(bytes_written)
    }
    
    fn write_vectors_buffered<P: AsRef<Path>>(&self, path: P, vectors: &[Vector], buffer_size: usize) -> Result<usize> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot write empty vector array");
        }
        
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {}", path.as_ref().display()))?;
        
        let mut writer = BufWriter::with_capacity(buffer_size, file);
        
        let bytes_written = write_vectors_f32(&mut writer, vectors)
            .context("Failed to write vector data")?;
        
        writer.flush()
            .context("Failed to flush writer")?;
        
        tracing::info!("Wrote {} vectors to {}, {} bytes total", 
                      vectors.len(), path.as_ref().display(), bytes_written);
        
        Ok(bytes_written)
    }
}

/// Builder for creating index writers with different configurations
pub struct IndexWriterBuilder {
    buffer_size: usize,
}

impl IndexWriterBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            buffer_size: 64 * 1024 * 1024,
        }
    }
    
    /// Set the buffer size for writes
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    /// Build a file-based index writer
    pub fn build_file_writer(self) -> FileIndexWriter {
        FileIndexWriter::with_buffer_size(self.buffer_size)
    }
}

impl Default for IndexWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_file_writer_vectors() {
        let writer = FileIndexWriter::new();
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let temp_file = NamedTempFile::new().unwrap();
        let bytes_written = writer.write_vectors(temp_file.path(), &vectors).unwrap();
        
        // Header (8 bytes) + data (2 * 3 * 4 = 24 bytes) = 32 bytes total
        assert_eq!(bytes_written, 32);
        
        // Verify file exists and has correct size
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert_eq!(metadata.len(), 32);
    }
    
    #[test]
    fn test_file_writer_graph() {
        let writer = FileIndexWriter::new();
        let nodes = vec![
            GraphNode::new(0, vec![1.0, 2.0]),
            GraphNode::new(1, vec![3.0, 4.0]),
        ];
        
        let temp_file = NamedTempFile::new().unwrap();
        let bytes_written = writer.write_graph(temp_file.path(), &nodes).unwrap();
        
        // Header (8 bytes) + data (2 * 2 * 4 = 16 bytes) = 24 bytes total
        assert_eq!(bytes_written, 24);
    }
    
    #[test]
    fn test_builder_pattern() {
        let writer = IndexWriterBuilder::new()
            .buffer_size(128 * 1024)
            .build_file_writer();
        
        assert_eq!(writer.buffer_size, 128 * 1024);
    }
    
    #[test]
    fn test_empty_vectors_error() {
        let writer = FileIndexWriter::new();
        let temp_file = NamedTempFile::new().unwrap();
        
        let result = writer.write_vectors(temp_file.path(), &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty vector array"));
    }
}