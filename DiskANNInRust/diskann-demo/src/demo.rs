//! Complete demo flow orchestration

use anyhow::{Result, Context};
use colored::*;
use std::path::Path;
use std::time::Instant;
use std::fs;
use std::io::Write;

use diskann_impl::{VamanaIndex, IndexBuilder, VamanaConfig};
use diskann_traits::{distance::{EuclideanDistance, CosineDistance}, index::Index, search::Search};
use diskann_io::{write_vectors_f32, read_vectors_f32};

use crate::dataset::STSBDataset;
use crate::embeddings::EmbeddingsGenerator;
use crate::model::RealEmbeddingModel;

/// Run the complete demo
pub fn run_full_demo(cache_dir: &Path) -> Result<()> {
    println!("{}", "=".repeat(60).blue());
    println!("{}", "DiskANN Pure Rust Demo - STSB Semantic Search".bold().green());
    println!("{}", "=".repeat(60).blue());
    println!();
    
    // Step 1: Load dataset
    println!("{}", "Step 1: Loading STSB Dataset".bold());
    let dataset = STSBDataset::download_and_load(cache_dir)?;
    println!();
    
    // Step 2: Generate embeddings
    println!("{}", "Step 2: Generating Embeddings".bold());
    let embeddings_path = cache_dir.join("embeddings.bin");
    let embeddings = if embeddings_path.exists() {
        println!("✓ Using cached embeddings");
        load_embeddings(&embeddings_path)?
    } else {
        // Try to use real model first, fall back to simple embeddings
        let embeddings = match RealEmbeddingModel::new(cache_dir) {
            Ok(mut model) => {
                println!("✓ Using real all-MiniLM-L6-v2 model");
                model.embed_texts(&dataset.sentences)?
            }
            Err(e) => {
                println!("⚠ Could not load ONNX model: {}", e);
                println!("  Falling back to simple embeddings...");
                let generator = EmbeddingsGenerator::new(384);
                generator.embed_texts(&dataset.sentences)?
            }
        };
        save_embeddings(&embeddings, &embeddings_path)?;
        embeddings
    };
    println!();
    
    // Step 3: Build index
    println!("{}", "Step 3: Building Vamana Index".bold());
    let start = Instant::now();
    
    let index = build_index(embeddings.clone())?;
    
    let build_time = start.elapsed();
    println!("✓ Index built in {:.2?}", build_time);
    println!("  - Nodes: {}", index.size());
    println!("  - Average degree: {:.2}", index.average_degree());
    println!();
    
    // Step 4: Run benchmark queries
    println!("{}", "Step 4: Running Benchmark Queries".bold());
    run_benchmark_queries(&index, &dataset, cache_dir)?;
    
    // Step 5: Show statistics
    println!();
    println!("{}", "=".repeat(60).blue());
    println!("{}", "Demo Statistics".bold().green());
    println!("{}", "=".repeat(60).blue());
    println!("Dataset size: {} sentences", dataset.sentences.len());
    println!("Embedding dimensions: 384");
    println!("Index build time: {:.2?}", build_time);
    println!("Index parameters:");
    println!("  - Max degree (R): 32");
    println!("  - Search list size (L): 64");
    println!("  - Alpha: 1.2");
    
    Ok(())
}

/// Build Vamana index from embeddings
fn build_index(embeddings: Vec<Vec<f32>>) -> Result<VamanaIndex<CosineDistance>> {
    let distance_fn = CosineDistance;
    let config = VamanaConfig {
        max_degree: 32,
        search_list_size: 64,
        alpha: 1.2,
        seed: 42,
    };
    
    // Convert to indexed vectors
    let vectors: Vec<(u32, Vec<f32>)> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    
    let index = IndexBuilder::new(distance_fn)
        .max_degree(config.max_degree)
        .search_list_size(config.search_list_size)
        .alpha(config.alpha)
        .seed(config.seed)
        .build(vectors)
        .context("Failed to build index")?;
    
    Ok(index)
}

/// Run benchmark queries
fn run_benchmark_queries(
    index: &VamanaIndex<CosineDistance>,
    dataset: &STSBDataset,
    cache_dir: &Path,
) -> Result<()> {
    // Try to use real model, fall back to simple
    let queries = dataset.get_sample_queries();
    let query_embeddings = match RealEmbeddingModel::new(cache_dir) {
        Ok(mut model) => {
            println!("Using real model for queries");
            let mut embeddings = Vec::new();
            for q in &queries {
                embeddings.push(model.embed_text(q)?);
            }
            embeddings
        }
        Err(_) => {
            let generator = EmbeddingsGenerator::new(384);
            queries.iter().map(|q| generator.embed_text(q)).collect()
        }
    };
    
    println!();
    for (i, (query, query_embedding)) in queries.iter().zip(query_embeddings.iter()).enumerate() {
        println!("{}", format!("Query {}: \"{}\"", i + 1, query).yellow());
        
        // Search
        let start = Instant::now();
        let results = index.search(&query_embedding, 5)?;
        let search_time = start.elapsed();
        
        println!("Search time: {:.2?}", search_time);
        println!("Results:");
        
        for (rank, result) in results.iter().enumerate() {
            let sentence = &dataset.sentences[result.id as usize];
            // Cosine distance is 1 - cosine_similarity, so similarity = 1 - distance
            let similarity = 1.0 - result.distance;
            
            println!("  {}. [Similarity: {:.3}] {}", 
                     rank + 1, 
                     similarity,
                     if sentence.len() > 60 {
                         format!("{}...", &sentence[..60])
                     } else {
                         sentence.clone()
                     });
        }
        println!();
    }
    
    Ok(())
}

/// Save embeddings to file
fn save_embeddings(embeddings: &[Vec<f32>], path: &Path) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_vectors_f32(&mut writer, embeddings)?;
    Ok(())
}

/// Load embeddings from file
fn load_embeddings(path: &Path) -> Result<Vec<Vec<f32>>> {
    let file = fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    read_vectors_f32(&mut reader)
}

/// Run interactive search mode
pub fn run_interactive_mode(cache_dir: &Path) -> Result<()> {
    // Load dataset and embeddings
    let dataset = STSBDataset::download_and_load(cache_dir)?;
    let embeddings_path = cache_dir.join("embeddings.bin");
    
    let embeddings = if embeddings_path.exists() {
        load_embeddings(&embeddings_path)?
    } else {
        let generator = EmbeddingsGenerator::new(384);
        let embeddings = generator.embed_texts(&dataset.sentences)?;
        save_embeddings(&embeddings, &embeddings_path)?;
        embeddings
    };
    
    // Build index
    println!("Building index...");
    let index = build_index(embeddings)?;
    
    // Try to use real model, fall back to simple
    // Since we need mutable access for the real model, we'll use simple embeddings for interactive mode
    println!("⚠ Using simple embeddings for interactive queries");
    let generator = EmbeddingsGenerator::new(384);
    let embed_fn = move |text: &str| generator.embed_text(text);
    
    println!("{}", "=".repeat(60).blue());
    println!("{}", "Interactive Search Mode".bold().green());
    println!("{}", "Type 'quit' to exit".italic());
    println!("{}", "=".repeat(60).blue());
    
    loop {
        print!("\n{} ", "Query:".cyan());
        std::io::stdout().flush()?;
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let query = input.trim();
        
        if query.eq_ignore_ascii_case("quit") || query.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        
        if query.is_empty() {
            continue;
        }
        
        // Generate embedding and search
        let query_embedding = embed_fn(query);
        let start = Instant::now();
        let results = index.search(&query_embedding, 5)?;
        let search_time = start.elapsed();
        
        println!("\n{} ({:.2?})", "Results:".green(), search_time);
        for (rank, result) in results.iter().enumerate() {
            let sentence = &dataset.sentences[result.id as usize];
            // Cosine distance is 1 - cosine_similarity, so similarity = 1 - distance
            let similarity = 1.0 - result.distance;
            
            println!("  {}. [{:.3}] {}", 
                     rank + 1, 
                     similarity,
                     sentence);
        }
    }
    
    Ok(())
}