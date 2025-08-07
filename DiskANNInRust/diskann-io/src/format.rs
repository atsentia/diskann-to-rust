//! Binary format definitions for DiskANN index files
//! 
//! This module implements binary compatibility with the C++ DiskANN format:
//! - 4 bytes: number of points (i32)
//! - 4 bytes: number of dimensions (i32)  
//! - data: points * dimensions * sizeof(T) bytes

use std::io::{Read, Write, Result as IoResult};
use anyhow::{Result, Context};
use diskann_core::vectors::{Vector, VectorId};
use diskann_core::structures::GraphNode;

/// Binary file header for DiskANN format
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryHeader {
    /// Number of points (vectors) in the file
    pub num_points: i32,
    /// Number of dimensions per vector
    pub num_dimensions: i32,
}

impl BinaryHeader {
    /// Create a new binary header
    pub fn new(num_points: usize, num_dimensions: usize) -> Self {
        Self {
            num_points: num_points as i32,
            num_dimensions: num_dimensions as i32,
        }
    }
    
    /// Read header from reader
    pub fn read_from<R: Read>(reader: &mut R) -> IoResult<Self> {
        let mut header = Self { num_points: 0, num_dimensions: 0 };
        reader.read_exact(unsafe { 
            std::slice::from_raw_parts_mut(
                &mut header as *mut Self as *mut u8,
                std::mem::size_of::<Self>()
            )
        })?;
        Ok(header)
    }
    
    /// Write header to writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> IoResult<()> {
        writer.write_all(unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>()
            )
        })
    }
    
    /// Get the expected data size in bytes for f32 vectors
    pub fn data_size_f32(&self) -> usize {
        (self.num_points * self.num_dimensions) as usize * std::mem::size_of::<f32>()
    }
    
    /// Get the total file size including header
    pub fn total_file_size_f32(&self) -> usize {
        std::mem::size_of::<Self>() + self.data_size_f32()
    }
    
    /// Validate header values
    pub fn validate(&self) -> Result<()> {
        if self.num_points < 0 {
            anyhow::bail!("Invalid num_points: {}", self.num_points);
        }
        if self.num_dimensions <= 0 {
            anyhow::bail!("Invalid num_dimensions: {}", self.num_dimensions);
        }
        Ok(())
    }
}

/// Write vector data in DiskANN binary format
pub fn write_vectors_f32<W: Write>(
    writer: &mut W,
    vectors: &[Vector],
) -> Result<usize> {
    if vectors.is_empty() {
        anyhow::bail!("Cannot write empty vector array");
    }
    
    let num_points = vectors.len();
    let num_dimensions = vectors[0].len();
    
    // Validate all vectors have same dimension
    for (i, vector) in vectors.iter().enumerate() {
        if vector.len() != num_dimensions {
            anyhow::bail!("Vector {} has {} dimensions, expected {}", i, vector.len(), num_dimensions);
        }
    }
    
    let header = BinaryHeader::new(num_points, num_dimensions);
    header.validate()?;
    
    // Write header
    header.write_to(writer)
        .context("Failed to write binary header")?;
    
    // Write vector data
    for vector in vectors {
        for &value in vector {
            writer.write_all(&value.to_le_bytes())
                .context("Failed to write vector data")?;
        }
    }
    
    Ok(header.total_file_size_f32())
}

/// Read vector data from DiskANN binary format
pub fn read_vectors_f32<R: Read>(reader: &mut R) -> Result<Vec<Vector>> {
    let header = BinaryHeader::read_from(reader)
        .context("Failed to read binary header")?;
    
    header.validate()?;
    
    let num_points = header.num_points as usize;
    let num_dimensions = header.num_dimensions as usize;
    
    let mut vectors = Vec::with_capacity(num_points);
    
    for _ in 0..num_points {
        let mut vector = Vector::with_capacity(num_dimensions);
        for _ in 0..num_dimensions {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)
                .context("Failed to read vector component")?;
            vector.push(f32::from_le_bytes(bytes));
        }
        vectors.push(vector);
    }
    
    Ok(vectors)
}

/// Write graph nodes to binary format (vector data + adjacency lists)
pub fn write_graph_nodes<W: Write>(
    writer: &mut W,
    nodes: &[GraphNode],
) -> Result<usize> {
    if nodes.is_empty() {
        anyhow::bail!("Cannot write empty graph nodes array");
    }
    
    // Extract vectors from nodes
    let vectors: Vec<Vector> = nodes.iter().map(|node| node.vector.clone()).collect();
    
    // Write vector data first
    let bytes_written = write_vectors_f32(writer, &vectors)?;
    
    // TODO: Add adjacency list writing when graph format is defined
    
    Ok(bytes_written)
}

/// Read graph nodes from binary format
pub fn read_graph_nodes<R: Read>(reader: &mut R) -> Result<Vec<GraphNode>> {
    let vectors = read_vectors_f32(reader)?;
    
    // Convert vectors to graph nodes
    let nodes = vectors
        .into_iter()
        .enumerate()
        .map(|(i, vector)| {
            GraphNode::new(i as VectorId, vector)
        })
        .collect();
        
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_binary_header_round_trip() {
        let header = BinaryHeader::new(100, 128);
        let mut buffer = Vec::new();
        header.write_to(&mut buffer).unwrap();
        
        let mut cursor = Cursor::new(buffer);
        let read_header = BinaryHeader::read_from(&mut cursor).unwrap();
        
        assert_eq!(header.num_points, read_header.num_points);
        assert_eq!(header.num_dimensions, read_header.num_dimensions);
    }
    
    #[test]
    fn test_vectors_round_trip() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let mut buffer = Vec::new();
        let bytes_written = write_vectors_f32(&mut buffer, &vectors).unwrap();
        
        let mut cursor = Cursor::new(buffer);
        let read_vectors = read_vectors_f32(&mut cursor).unwrap();
        
        assert_eq!(vectors.len(), read_vectors.len());
        for (orig, read) in vectors.iter().zip(read_vectors.iter()) {
            assert_eq!(orig.len(), read.len());
            for (&orig_val, &read_val) in orig.iter().zip(read.iter()) {
                assert!((orig_val - read_val).abs() < f32::EPSILON);
            }
        }
        
        // Check bytes written calculation
        let expected_bytes = 8 + (3 * 3 * 4); // header + data
        assert_eq!(bytes_written, expected_bytes);
    }
    
    #[test]
    fn test_graph_nodes_round_trip() {
        let nodes = vec![
            GraphNode::new(0, vec![1.0, 2.0]),
            GraphNode::new(1, vec![3.0, 4.0]),
        ];
        
        let mut buffer = Vec::new();
        write_graph_nodes(&mut buffer, &nodes).unwrap();
        
        let mut cursor = Cursor::new(buffer);
        let read_nodes = read_graph_nodes(&mut cursor).unwrap();
        
        assert_eq!(nodes.len(), read_nodes.len());
        for (orig, read) in nodes.iter().zip(read_nodes.iter()) {
            assert_eq!(orig.vector.len(), read.vector.len());
            for (&orig_val, &read_val) in orig.vector.iter().zip(read.vector.iter()) {
                assert!((orig_val - read_val).abs() < f32::EPSILON);
            }
        }
    }
}