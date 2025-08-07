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

/// Load vectors from a binary file
/// Expected format: [num_vectors: u32][dimension: u32][vector_data: f32...]
fn load_vectors_from_file(file_path: &str) -> Result<Vec<(u32, Vec<f32>)>> {
    let path = Path::new(file_path);
    if !path.exists() {
        bail!("File not found: {}", file_path);
    }

    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", file_path))?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)
        .context("Failed to read number of vectors")?;
    let num_vectors = u32::from_le_bytes(buffer);

    reader.read_exact(&mut buffer)
        .context("Failed to read vector dimension")?;
    let dimension = u32::from_le_bytes(buffer);

    info!("Loading {} vectors of dimension {}", num_vectors, dimension);

    let mut vectors = Vec::with_capacity(num_vectors as usize);
    
    for id in 0..num_vectors {
        let mut vector = vec![0.0f32; dimension as usize];
        let mut float_buffer = [0u8; 4];
        
        for component in &mut vector {
            reader.read_exact(&mut float_buffer)
                .with_context(|| format!("Failed to read vector component for vector {}", id))?;
            *component = f32::from_le_bytes(float_buffer);
        }
        
        vectors.push((id, vector));
    }

    Ok(vectors)
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
            let index = IndexBuilder::new(distance_fn)
                .max_degree(max_degree)
                .search_list_size(search_list_size)
                .alpha(alpha)
                .seed(seed)
                .build(vectors)
                .context("Failed to build index")?;

            info!("Built index with {} nodes, average degree: {:.2}", 
                  index.size(), index.average_degree());

            // TODO: Save index to file using diskann-io
            // For now, just log success
            info!("Index built successfully (save to file not yet implemented)");
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

            // TODO: Load index from file
            // For now, create a demo index
            let distance_fn = EuclideanDistance;
            let demo_vectors = vec![
                (0, vec![1.0, 0.0, 0.0]),
                (1, vec![0.0, 1.0, 0.0]),
                (2, vec![0.0, 0.0, 1.0]),
                (3, vec![0.5, 0.5, 0.0]),
                (4, vec![0.0, 0.5, 0.5]),
            ];

            let index = IndexBuilder::new(distance_fn)
                .max_degree(32)
                .search_list_size(64)
                .build(demo_vectors)
                .context("Failed to create demo index")?;

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