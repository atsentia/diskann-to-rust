//! DiskANN Pure Rust Demo - Complete A-Z Example
//!
//! This demo shows:
//! 1. Loading a text dataset (STSB)
//! 2. Generating embeddings in Rust
//! 3. Building a Vamana index
//! 4. Performing semantic search
//! 5. Displaying results with timing

mod dataset;
mod embeddings;
mod model;
mod demo;

use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;
use colored::*;

#[derive(Parser)]
#[command(name = "diskann-demo")]
#[command(about = "DiskANN Pure Rust Demo - Semantic Search", long_about = None)]
struct Cli {
    /// Cache directory for dataset and embeddings
    #[arg(short, long, default_value = ".diskann_cache")]
    cache_dir: PathBuf,
    
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the complete demo (download, embed, index, search)
    Full,
    
    /// Interactive search mode
    Interactive,
    
    /// Just prepare the dataset
    Prepare,
    
    /// Clean cache directory
    Clean,
}

fn main() -> Result<()> {
    // Set up colored output for Windows
    #[cfg(windows)]
    colored::control::set_virtual_terminal(true).ok();
    
    let cli = Cli::parse();
    
    // Ensure cache directory exists
    std::fs::create_dir_all(&cli.cache_dir)?;
    
    match cli.command {
        Some(Commands::Full) | None => {
            // Default to full demo
            demo::run_full_demo(&cli.cache_dir)?;
        }
        Some(Commands::Interactive) => {
            demo::run_interactive_mode(&cli.cache_dir)?;
        }
        Some(Commands::Prepare) => {
            println!("{}", "Preparing dataset...".bold());
            let dataset = dataset::STSBDataset::download_and_load(&cli.cache_dir)?;
            
            println!("{}", "Generating embeddings...".bold());
            let embeddings_path = cli.cache_dir.join("embeddings.bin");
            
            if !embeddings_path.exists() {
                // Try real model first
                let embeddings = match model::RealEmbeddingModel::new(&cli.cache_dir) {
                    Ok(mut model) => {
                        println!("✓ Using real all-MiniLM-L6-v2 model");
                        model.embed_texts(&dataset.sentences)?
                    }
                    Err(e) => {
                        println!("⚠ Could not load ONNX model: {}", e);
                        println!("  Falling back to simple embeddings...");
                        let generator = embeddings::EmbeddingsGenerator::new(384);
                        generator.embed_texts(&dataset.sentences)?
                    }
                };
                
                // Save embeddings
                use diskann_io::write_vectors_f32;
                let file = std::fs::File::create(&embeddings_path)?;
                let mut writer = std::io::BufWriter::new(file);
                write_vectors_f32(&mut writer, &embeddings)?;
                
                println!("✓ Embeddings saved to {:?}", embeddings_path);
            } else {
                println!("✓ Embeddings already exist");
            }
        }
        Some(Commands::Clean) => {
            println!("Cleaning cache directory...");
            if cli.cache_dir.exists() {
                std::fs::remove_dir_all(&cli.cache_dir)?;
                println!("✓ Cache cleaned");
            } else {
                println!("Cache directory doesn't exist");
            }
        }
    }
    
    Ok(())
}