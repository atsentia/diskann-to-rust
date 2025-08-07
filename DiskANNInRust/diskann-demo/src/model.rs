//! Real embedding model using ONNX Runtime
//!
//! Downloads and uses sentence-transformers/all-MiniLM-L6-v2 for actual embeddings

use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ndarray::{Array2, Axis};
use tokenizers::{Tokenizer, PaddingParams, PaddingStrategy, TruncationParams};

/// Model files we need to download
const MODEL_FILES: &[(&str, &str)] = &[
    ("model.onnx", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"),
    ("tokenizer.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"),
    ("tokenizer_config.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"),
];

/// Real embedding model using ONNX
pub struct RealEmbeddingModel {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl RealEmbeddingModel {
    /// Create a new embedding model, downloading if necessary
    pub fn new(cache_dir: &Path) -> Result<Self> {
        println!("Initializing real embedding model (all-MiniLM-L6-v2)...");
        
        // Ensure cache directory exists
        let model_dir = cache_dir.join("all-MiniLM-L6-v2");
        fs::create_dir_all(&model_dir)?;
        
        // Download model files if needed
        println!("Checking for model files...");
        Self::download_model_files(&model_dir)?;
        
        // Load ONNX model
        let model_path = model_dir.join("model.onnx");
        println!("Loading ONNX model from {:?}...", model_path);
        
        let session = Session::builder()
            .expect("Failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("Failed to set optimization level")
            .with_intra_threads(4)
            .expect("Failed to set intra threads")
            .commit_from_file(&model_path)?;
        
        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        println!("Loading tokenizer from {:?}...", tokenizer_path);
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Configure tokenizer padding and truncation
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));
        
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: 256,
            ..Default::default()
        }));
        
        Ok(Self {
            session,
            tokenizer,
            dimension: 384, // all-MiniLM-L6-v2 outputs 384D embeddings
        })
    }
    
    /// Download model files if they don't exist
    fn download_model_files(model_dir: &Path) -> Result<()> {
        for (filename, url) in MODEL_FILES {
            let file_path = model_dir.join(filename);
            
            if !file_path.exists() {
                println!("Downloading {}...", filename);
                Self::download_file(url, &file_path)?;
            } else {
                println!("✓ {} already exists", filename);
            }
        }
        
        Ok(())
    }
    
    /// Download a file from URL
    fn download_file(url: &str, path: &Path) -> Result<()> {
        use std::io::Read;
        
        println!("  Downloading from: {}", url);
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} Downloading... {bytes} bytes ({elapsed})")
                .unwrap()
        );
        
        // Download file with progress updates
        let response = ureq::get(url)
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .context("Failed to download file")?;
        
        let mut bytes = Vec::new();
        let mut reader = response.into_reader();
        let mut buffer = [0; 8192];
        
        loop {
            match reader.read(&mut buffer)? {
                0 => break,
                n => {
                    bytes.extend_from_slice(&buffer[..n]);
                    pb.set_position(bytes.len() as u64);
                    pb.tick();
                }
            }
        }
        
        pb.finish_with_message(format!("✓ Downloaded {} MB", bytes.len() / 1_000_000));
        
        // Save to file
        fs::write(path, bytes)?;
        
        Ok(())
    }
    
    /// Generate embeddings for multiple texts
    pub fn embed_texts(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        let pb = ProgressBar::new(texts.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Generating embeddings")
                .unwrap()
                .progress_chars("=>-")
        );
        
        let mut all_embeddings = Vec::new();
        
        // Process in batches for efficiency
        let batch_size = 32;
        for chunk in texts.chunks(batch_size) {
            let embeddings = self.embed_batch(chunk)?;
            all_embeddings.extend(embeddings);
            pb.inc(chunk.len() as u64);
        }
        
        pb.finish_with_message("✓ Embeddings generated");
        Ok(all_embeddings)
    }
    
    /// Generate embeddings for a batch of texts
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Tokenize texts
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        // Extract input tensors
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let batch_size = texts.len();
        
        // Create input arrays
        let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));
        
        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            
            for (j, &id) in ids.iter().enumerate() {
                input_ids[[i, j]] = id as i64;
            }
            for (j, &m) in mask.iter().enumerate() {
                attention_mask[[i, j]] = m as i64;
            }
            for (j, &t) in type_ids.iter().enumerate() {
                token_type_ids[[i, j]] = t as i64;
            }
        }
        
        // Run inference - convert to proper format for ORT v2
        use ort::{inputs, value::Tensor};
        
        // Create ORT tensors from arrays
        let input_ids_tensor = Tensor::from_array((
            vec![batch_size, max_len],
            input_ids.as_slice().unwrap().to_vec()
        ))?;
        let attention_mask_tensor = Tensor::from_array((
            vec![batch_size, max_len],
            attention_mask.as_slice().unwrap().to_vec()
        ))?;
        let token_type_ids_tensor = Tensor::from_array((
            vec![batch_size, max_len],
            token_type_ids.as_slice().unwrap().to_vec()
        ))?;
        
        let outputs = self.session.run(inputs![
            input_ids_tensor,
            attention_mask_tensor,
            token_type_ids_tensor
        ])?;
        
        // Extract sentence embeddings (mean pooling)
        let output = outputs.get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("Missing last_hidden_state output"))?;
        let (_shape, raw_data) = output.try_extract_tensor::<f32>()?;
        
        // Reshape to 3D tensor
        let token_embeddings = ndarray::ArrayView3::from_shape(
            (batch_size, max_len, self.dimension),
            raw_data
        )?;
        
        // Convert to owned array to avoid borrow conflicts
        let token_embeddings = token_embeddings.to_owned();
        
        // Now we can drop outputs and work with the owned data
        drop(outputs);
        
        let mut embeddings = Vec::new();
        for i in 0..batch_size {
            let token_emb_slice = token_embeddings.slice(ndarray::s![i, .., ..]);
            let attention_mask_slice = attention_mask.slice(ndarray::s![i, ..]).mapv(|x| x as f32);
            let embedding = self.mean_pooling(
                token_emb_slice,
                attention_mask_slice
            );
            
            // L2 normalize
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = if norm > 0.0 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            };
            
            embeddings.push(normalized);
        }
        
        Ok(embeddings)
    }
    
    /// Mean pooling over token embeddings
    fn mean_pooling(&self, token_embeddings: ndarray::ArrayView2<f32>, attention_mask: ndarray::Array1<f32>) -> Vec<f32> {
        let expanded_mask = attention_mask.insert_axis(Axis(1));
        let masked_embeddings = &token_embeddings * &expanded_mask;
        
        let sum_embeddings = masked_embeddings.sum_axis(Axis(0));
        let sum_mask = expanded_mask.sum_axis(Axis(0)).mapv(|x| if x > 0.0 { x } else { 1.0 });
        
        (sum_embeddings / sum_mask).to_vec()
    }
    
    /// Generate embedding for a single text
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        Ok(embeddings.into_iter().next().unwrap())
    }
    
    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}