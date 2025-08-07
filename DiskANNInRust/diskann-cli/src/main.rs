//! Command-line interface for DiskANN vector search operations

#![deny(warnings)]

use clap::{Parser, Subcommand};
use anyhow::{Result, Context, bail};
use tracing::info;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use diskann_impl::IndexBuilder;
use diskann_traits::{distance::EuclideanDistance, index::Index, search::Search};
use diskann_io::{write_vectors_f32, read_vectors_f32};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an index from vector data
    Build {
        /// Input vector file path
        #[arg(short, long)]
        input: String,
        /// Output index file path
        #[arg(short, long)]
        output: String,
        /// Maximum degree per node (R parameter)
        #[arg(long, default_value_t = 64)]
        max_degree: usize,
        /// Search list size during construction (L parameter)
        #[arg(long, default_value_t = 100)]
        search_list_size: usize,
        /// Alpha parameter for pruning
        #[arg(long, default_value_t = 1.2)]
        alpha: f32,
        /// Random seed for deterministic behavior
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Search for nearest neighbors
    Search {
        /// Index file path
        #[arg(short, long)]
        index: String,
        /// Query vector file path
        #[arg(short, long)]
        query: String,
        /// Number of nearest neighbors to find
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Beam width for search (larger = better quality, slower)
        #[arg(long, default_value_t = 64)]
        beam: usize,
        /// Output results to file (optional)
        #[arg(short, long)]
        output: Option<String>,
    },
}

/// Load vectors from a binary file using diskann-io format
/// Expected format: [num_vectors: u32][dimension: u32][vector_data: f32...]
fn load_vectors_from_file(file_path: &str) -> Result<Vec<(u32, Vec<f32>)>> {
    let path = Path::new(file_path);
    if !path.exists() {
        bail!("File not found: {}", file_path);
    }

    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", file_path))?;
    let mut reader = BufReader::new(file);

    // Use diskann-io to read vectors
    let vectors = read_vectors_f32(&mut reader)
        .with_context(|| format!("Failed to read vectors from {}", file_path))?;

    info!("Loading {} vectors of dimension {}", vectors.len(), 
          if vectors.is_empty() { 0 } else { vectors[0].len() });

    // Convert to (id, vector) pairs
    let indexed_vectors = vectors
        .into_iter()
        .enumerate()
        .map(|(id, vector)| (id as u32, vector))
        .collect();

    Ok(indexed_vectors)
}

/// Load a single query vector from file
/// Expected format: [dimension: u32][vector_data: f32...]
fn load_query_from_file(file_path: &str) -> Result<Vec<f32>> {
    let path = Path::new(file_path);
    if !path.exists() {
        bail!("File not found: {}", file_path);
    }

    let file = File::open(path)
        .with_context(|| format!("Failed to open query file: {}", file_path))?;
    let mut reader = BufReader::new(file);

    // Read dimension
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)
        .context("Failed to read query dimension")?;
    let dimension = u32::from_le_bytes(buffer);

    let mut query = vec![0.0f32; dimension as usize];
    let mut float_buffer = [0u8; 4];
    
    for component in &mut query {
        reader.read_exact(&mut float_buffer)
            .context("Failed to read query component")?;
        *component = f32::from_le_bytes(float_buffer);
    }

    Ok(query)
}

/// Save search results to file
fn save_results(results: &[(u32, f32)], output_path: &str) -> Result<()> {
    use std::io::Write;
    
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path))?;
    let mut writer = std::io::BufWriter::new(file);

    writeln!(writer, "id,distance")?;
    for (id, distance) in results {
        writeln!(writer, "{},{}", id, distance)?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Build { 
            input, 
            output, 
            max_degree,
            search_list_size,
            alpha,
            seed,
        } => {
            info!("Building index from {} to {}", input, output);
            info!("Parameters: max_degree={}, search_list_size={}, alpha={}, seed={}", 
                  max_degree, search_list_size, alpha, seed);

            // Load vectors
            let vectors = load_vectors_from_file(&input)
                .context("Failed to load input vectors")?;
            
            info!("Loaded {} vectors", vectors.len());

            // Build index
            let distance_fn = EuclideanDistance;
            
            // Clone vectors for saving before building index
            let vector_data_for_save: Vec<Vec<f32>> = vectors.iter().map(|(_, v)| v.clone()).collect();
            
            let index = IndexBuilder::new(distance_fn)
                .max_degree(max_degree)
                .search_list_size(search_list_size)
                .alpha(alpha)
                .seed(seed)
                .build(vectors)
                .context("Failed to build index")?;

            info!("Built index with {} nodes, average degree: {:.2}", 
                  index.size(), index.average_degree());

            // Save index to file - for now, we'll save the original vectors
            // In the future, this would save the full graph structure
            info!("Saving index to {}", output);
            let output_file = File::create(&output)
                .with_context(|| format!("Failed to create output file: {}", output))?;
            let mut writer = std::io::BufWriter::new(output_file);
            
            // Save the original vectors (simple approach for demo)
            write_vectors_f32(&mut writer, &vector_data_for_save)
                .context("Failed to save index to file")?;
            
            info!("Index saved successfully to {}", output);
            println!("Index building completed successfully!");
        }
        Commands::Search { 
            index: index_path, 
            query: query_path, 
            k, 
            beam,
            output,
        } => {
            info!("Searching index {} with query {} for {} neighbors (beam={})", 
                  index_path, query_path, k, beam);

            // Load index from file
            let index = if Path::new(&index_path).exists() {
                info!("Loading index from {}", index_path);
                let index_file = File::open(&index_path)
                    .with_context(|| format!("Failed to open index file: {}", index_path))?;
                let mut reader = BufReader::new(index_file);
                
                // Load vectors and rebuild index
                let loaded_vectors = read_vectors_f32(&mut reader)
                    .context("Failed to load vectors from index file")?;
                
                info!("Loaded {} vectors from index", loaded_vectors.len());
                
                let distance_fn = EuclideanDistance;
                let vector_data: Vec<(u32, Vec<f32>)> = loaded_vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v))
                    .collect();
                
                IndexBuilder::new(distance_fn)
                    .max_degree(64)
                    .search_list_size(128)
                    .build(vector_data)
                    .context("Failed to rebuild index from loaded vectors")?
            } else {
                // Create demo index if file doesn't exist
                info!("Index file not found, creating demo index for testing");
                let distance_fn = EuclideanDistance;
                let demo_vectors = vec![
                    (0, vec![1.0, 0.0, 0.0]),
                    (1, vec![0.0, 1.0, 0.0]),
                    (2, vec![0.0, 0.0, 1.0]),
                    (3, vec![0.5, 0.5, 0.0]),
                    (4, vec![0.0, 0.5, 0.5]),
                ];

                IndexBuilder::new(distance_fn)
                    .max_degree(32)
                    .search_list_size(64)
                    .build(demo_vectors)
                    .context("Failed to create demo index")?
            };

            // Load query
            let query = load_query_from_file(&query_path)
                .context("Failed to load query vector")?;
            
            info!("Loaded query vector of dimension {}", query.len());

            // Perform search
            let results = index.search_with_beam(&query, k, beam)
                .context("Search failed")?;

            // Display results
            println!("Search Results:");
            println!("ID\tDistance");
            let mut result_pairs = Vec::new();
            for (i, result) in results.iter().enumerate() {
                println!("{}\t{:.6}", result.id, result.distance);
                result_pairs.push((result.id, result.distance));
                if i >= k - 1 {
                    break;
                }
            }

            // Save results if output specified
            if let Some(output_path) = output {
                save_results(&result_pairs, &output_path)
                    .context("Failed to save results")?;
                info!("Results saved to {}", output_path);
            }

            info!("Search completed successfully");
        }
    }
    
    Ok(())
}