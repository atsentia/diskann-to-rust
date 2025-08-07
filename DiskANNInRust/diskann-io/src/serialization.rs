//! Index serialization and deserialization

use diskann_core::structures::GraphNode;
use anyhow::{Result, Context};
use std::io::{Read, Write};
use crate::format::{write_vectors_f32, read_vectors_f32};

/// Serialize index to writer (simple vector-only format for now)
pub fn serialize_index<W: Write>(writer: &mut W, nodes: &[GraphNode]) -> Result<()> {
    if nodes.is_empty() {
        anyhow::bail!("Cannot serialize empty index");
    }
    
    // Extract vectors from nodes
    let vectors: Vec<_> = nodes.iter().map(|node| node.vector.clone()).collect();
    
    // Write vectors using diskann-io format
    write_vectors_f32(writer, &vectors)
        .context("Failed to serialize index vectors")?;
    
    Ok(())
}

/// Deserialize index from reader (simple vector-only format for now)
pub fn deserialize_index<R: Read>(reader: &mut R) -> Result<Vec<GraphNode>> {
    // Read vectors using diskann-io format
    let vectors = read_vectors_f32(reader)
        .context("Failed to deserialize index vectors")?;
    
    // Convert vectors to graph nodes
    let nodes = vectors
        .into_iter()
        .enumerate()
        .map(|(i, vector)| {
            GraphNode::new(i as u32, vector)
        })
        .collect();
        
    Ok(nodes)
}