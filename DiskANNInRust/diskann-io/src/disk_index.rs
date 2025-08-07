//! Disk-based index implementation for memory-efficient search
//!
//! This module provides functionality for building and searching indices
//! that are stored on disk rather than in memory, enabling search on
//! datasets larger than available RAM.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

/// Configuration for disk index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIndexConfig {
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Dimension of each vector
    pub dimension: usize,
    /// Maximum degree of graph nodes
    pub max_degree: usize,
    /// Search list size used during construction
    pub search_list_size: usize,
    /// Alpha parameter for pruning
    pub alpha: f32,
    /// Version of the index format
    pub index_version: String,
}

/// Disk-based index for memory-efficient storage and search
pub struct DiskIndex {
    config: DiskIndexConfig,
    graph_path: PathBuf,
    vectors_path: PathBuf,
    metadata_path: Option<PathBuf>,
}

impl DiskIndex {
    /// Create a new disk index
    pub fn new(index_dir: &Path) -> Result<Self> {
        let config_path = index_dir.join("config.json");
        let graph_path = index_dir.join("graph.bin");
        let vectors_path = index_dir.join("vectors.bin");
        let metadata_path = {
            let path = index_dir.join("metadata.bin");
            if path.exists() { Some(path) } else { None }
        };

        // Load configuration
        let config_file = File::open(&config_path)
            .context("Failed to open config file")?;
        let config: DiskIndexConfig = serde_json::from_reader(config_file)
            .context("Failed to parse config")?;

        Ok(Self {
            config,
            graph_path,
            vectors_path,
            metadata_path,
        })
    }

    /// Get index configuration
    pub fn config(&self) -> &DiskIndexConfig {
        &self.config
    }

    /// Load a specific vector from disk
    pub fn load_vector(&self, id: usize) -> Result<Vec<f32>> {
        if id >= self.config.num_vectors {
            anyhow::bail!("Vector ID {} out of range", id);
        }

        let mut file = File::open(&self.vectors_path)?;
        
        // Skip header (num_vectors, dimension)
        file.seek(SeekFrom::Start(8))?;
        
        // Seek to vector position
        let vector_size = self.config.dimension * 4; // f32 = 4 bytes
        let offset = 8 + (id * vector_size) as u64;
        file.seek(SeekFrom::Start(offset))?;
        
        // Read vector
        let mut buffer = vec![0u8; vector_size];
        file.read_exact(&mut buffer)?;
        
        // Convert to f32
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            })
            .collect();
        
        Ok(vector)
    }

    /// Load neighbors for a specific node from the graph
    pub fn load_neighbors(&self, id: usize) -> Result<Vec<u32>> {
        let file = File::open(&self.graph_path)?;
        let mut reader = BufReader::new(file);
        
        // Read graph header
        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;
        let num_nodes = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let max_degree = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        
        if id >= num_nodes as usize {
            anyhow::bail!("Node ID {} out of range", id);
        }
        
        // Seek to node's neighbor list
        // Each node stores: degree (u32) + neighbors (max_degree * u32)
        let node_size = 4 + (max_degree as usize * 4);
        let offset = 8 + (id * node_size) as u64;
        
        let mut file = reader.into_inner();
        file.seek(SeekFrom::Start(offset))?;
        
        // Read degree
        let mut degree_bytes = [0u8; 4];
        file.read_exact(&mut degree_bytes)?;
        let degree = u32::from_le_bytes(degree_bytes) as usize;
        
        // Read neighbors
        let mut neighbors = Vec::with_capacity(degree);
        for _ in 0..degree {
            let mut neighbor_bytes = [0u8; 4];
            file.read_exact(&mut neighbor_bytes)?;
            neighbors.push(u32::from_le_bytes(neighbor_bytes));
        }
        
        Ok(neighbors)
    }

    /// Get total index size in bytes
    pub fn size_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        
        if self.graph_path.exists() {
            total += self.graph_path.metadata()?.len();
        }
        if self.vectors_path.exists() {
            total += self.vectors_path.metadata()?.len();
        }
        if let Some(ref metadata_path) = self.metadata_path {
            if metadata_path.exists() {
                total += metadata_path.metadata()?.len();
            }
        }
        
        Ok(total)
    }
}

/// Builder for creating disk indices
pub struct DiskIndexBuilder {
    index_dir: PathBuf,
    max_degree: usize,
    search_list_size: usize,
    alpha: f32,
}

impl DiskIndexBuilder {
    /// Create a new disk index builder
    pub fn new(index_dir: impl Into<PathBuf>) -> Self {
        Self {
            index_dir: index_dir.into(),
            max_degree: 32,
            search_list_size: 64,
            alpha: 1.2,
        }
    }

    /// Set maximum degree
    pub fn max_degree(mut self, max_degree: usize) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set search list size
    pub fn search_list_size(mut self, search_list_size: usize) -> Self {
        self.search_list_size = search_list_size;
        self
    }

    /// Set alpha parameter
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Build disk index from vectors and graph
    pub fn build(
        self,
        vectors: &[Vec<f32>],
        graph: &[Vec<u32>],
    ) -> Result<DiskIndex> {
        // Create index directory
        std::fs::create_dir_all(&self.index_dir)?;

        let num_vectors = vectors.len();
        let dimension = vectors.first()
            .ok_or_else(|| anyhow::anyhow!("No vectors provided"))?
            .len();

        // Write configuration
        let config = DiskIndexConfig {
            num_vectors,
            dimension,
            max_degree: self.max_degree,
            search_list_size: self.search_list_size,
            alpha: self.alpha,
            index_version: "0.1.0".to_string(),
        };

        let config_path = self.index_dir.join("config.json");
        let config_file = File::create(&config_path)?;
        serde_json::to_writer_pretty(config_file, &config)?;

        // Write vectors
        let vectors_path = self.index_dir.join("vectors.bin");
        let vectors_file = File::create(&vectors_path)?;
        let mut writer = BufWriter::new(vectors_file);
        
        // Write header
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;
        writer.write_all(&(dimension as u32).to_le_bytes())?;
        
        // Write vectors
        for vector in vectors {
            for &val in vector {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        writer.flush()?;

        // Write graph
        let graph_path = self.index_dir.join("graph.bin");
        let graph_file = File::create(&graph_path)?;
        let mut writer = BufWriter::new(graph_file);
        
        // Write graph header
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;
        writer.write_all(&(self.max_degree as u32).to_le_bytes())?;
        
        // Write adjacency lists
        for neighbors in graph {
            let degree = neighbors.len().min(self.max_degree);
            writer.write_all(&(degree as u32).to_le_bytes())?;
            
            // Write neighbors
            for &neighbor in &neighbors[..degree] {
                writer.write_all(&neighbor.to_le_bytes())?;
            }
            
            // Pad with zeros if needed
            for _ in degree..self.max_degree {
                writer.write_all(&0u32.to_le_bytes())?;
            }
        }
        writer.flush()?;

        // Return disk index
        DiskIndex::new(&self.index_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_disk_index_build_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let index_dir = temp_dir.path();

        // Create sample data
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let graph = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];

        // Build index
        let builder = DiskIndexBuilder::new(index_dir)
            .max_degree(4)
            .search_list_size(10);
        
        let index = builder.build(&vectors, &graph).unwrap();

        // Test configuration
        assert_eq!(index.config().num_vectors, 3);
        assert_eq!(index.config().dimension, 3);

        // Test vector loading
        let vector0 = index.load_vector(0).unwrap();
        assert_eq!(vector0, vec![1.0, 0.0, 0.0]);

        // Test neighbor loading
        let neighbors0 = index.load_neighbors(0).unwrap();
        assert_eq!(neighbors0, vec![1, 2]);

        // Test size
        let size = index.size_bytes().unwrap();
        assert!(size > 0);
    }
}