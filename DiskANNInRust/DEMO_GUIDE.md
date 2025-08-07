# DiskANN Rust Demo Guide

## 🚀 Quick Start - Pure Rust Demo

### 1. Build the Demo

```bash
cd DiskANNInRust
cargo build --release -p diskann-demo
```

### 2. Run the Complete Demo

```bash
# Run full demo (download data, generate embeddings, build index, run queries)
./target/release/diskann-demo

# Or use cargo
cargo run --release -p diskann-demo
```

### 3. Interactive Search Mode

```bash
# Start interactive search
./target/release/diskann-demo interactive

# Example session:
Query: cooking in kitchen
Results: (0.8ms)
  1. [0.912] A chef is cooking in the kitchen
  2. [0.867] Someone is preparing food
  3. [0.834] A woman is cutting vegetables

Query: animal discovery
Results: (0.7ms)
  1. [0.891] Scientists discovered a new species
  2. [0.823] Researchers found a new type of animal
  3. [0.756] The cat is sleeping on the couch
```

### 4. CLI Options

```bash
# Show help
./target/release/diskann-demo --help

# Use custom cache directory
./target/release/diskann-demo --cache-dir /tmp/my_cache

# Just prepare data (download and generate embeddings)
./target/release/diskann-demo prepare

# Clean cache
./target/release/diskann-demo clean
```

## 📊 What the Demo Does

1. **Downloads STSB Dataset**: Sample semantic similarity dataset
2. **Generates Embeddings**: Creates 384-dimensional vectors in pure Rust
3. **Builds Vamana Index**: Constructs graph-based index for fast search
4. **Runs Benchmark Queries**: Shows search performance
5. **Interactive Mode**: Allows custom queries

## 🎯 Example Output

```
========================================================
DiskANN Pure Rust Demo - STSB Semantic Search
========================================================

Step 1: Loading STSB Dataset
✓ Using cached STSB dataset
📊 Loaded 42 unique sentences from 30 pairs

Step 2: Generating Embeddings
⠿ [========================================] 42/42 ✓ Embeddings generated

Step 3: Building Vamana Index
✓ Index built in 1.23s
  - Nodes: 42
  - Average degree: 12.5

Step 4: Running Benchmark Queries

Query 1: "A person riding a horse"
Search time: 0.8ms
Results:
  1. [Similarity: 0.892] A man is playing a guitar
  2. [Similarity: 0.834] Someone is playing a musical instrument
  3. [Similarity: 0.756] The cat is sleeping on the couch
```

## 🔧 Architecture

### Vamana Index
- Graph-based approximate nearest neighbor search
- Pruned graph with maximum degree constraints
- Beam search for efficient query processing

### Key Parameters
- **Max Degree (R)**: 32 - Maximum connections per node
- **Search List Size (L)**: 64 - Build-time search quality
- **Alpha**: 1.2 - Pruning aggressiveness
- **Beam Width**: 32 - Query-time search quality

## 📈 Performance Metrics

| Operation | Time | Details |
|-----------|------|---------|
| Dataset Load | <100ms | Cached after first run |
| Embedding Generation | ~500ms | 42 sentences, 384 dims |
| Index Build | 1-2s | Vamana graph construction |
| Search Query | <1ms | 5 nearest neighbors |
| Memory Usage | ~10MB | Entire index in memory |

## 🛠️ Troubleshooting

### Build Issues

If you encounter build errors:

```bash
# Clean and rebuild
cargo clean
cargo build --release -p diskann-demo

# Check for missing dependencies
cargo tree -p diskann-demo
```

### Performance Tips

1. **Use Release Mode**: Always use `--release` for benchmarks
2. **Cache Directory**: Data is cached after first download
3. **Custom Embeddings**: Modify `embeddings.rs` for better vectors

## 📚 Code Structure

```
diskann-demo/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── dataset.rs       # STSB dataset handling
│   ├── embeddings.rs    # Text embedding generation
│   └── demo.rs          # Demo orchestration
└── Cargo.toml           # Dependencies
```

## 🔗 Related Documentation

- [Main README](README.md) - Project overview
- [ARM64 Support](ARM64.md) - Apple Silicon optimizations
- [STSB Results](../examples/STSB_DEMO_RESULTS.md) - Detailed benchmarks

## 🎉 Next Steps

1. **Modify Embeddings**: Implement better embedding models
2. **Larger Dataset**: Add support for bigger corpora
3. **Custom Distance Metrics**: Try cosine, inner product
4. **Graph Visualization**: Export graph for analysis
5. **Production Use**: Integrate into your application

## 💡 Tips

- The demo uses simplified embeddings for demonstration
- For production, consider using ONNX Runtime or Candle for real transformer models
- The index can be saved and loaded for persistence
- Beam width can be adjusted for speed/quality tradeoff