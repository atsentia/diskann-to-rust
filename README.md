# DiskANN-Rust

A high-performance Rust port of the DiskANN approximate nearest neighbor search library with end-to-end text search capabilities.

## Quick-start Demo

Experience semantic text search with the MS MARCO dataset:

```bash
# 1. Install Python dependencies
cd examples
pip install -r requirements.txt

# 2. Generate embeddings (5-10 minutes)
python make_msmarco_embeddings.py --max-passages 1000  # Small demo

# 3. Build search index
cd DiskANNInRust
cargo build --release --bin diskann
./target/release/diskann build -i ../examples/msmarco_passages.bin -o ../examples/msmarco.disk.index

# 4. Run interactive search
cd ../examples
python query_demo.py --index msmarco.disk.index --metadata msmarco_passages.tsv
```

Try queries like "What is machine learning?" or "Best restaurants in New York". 

### Example Results

**Query: "What is machine learning?"** *(10.3ms)*
1. **0.1234** - Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...
2. **0.3456** - Docker containers provide a lightweight, portable way to package applications with all their dependencies...
3. **0.4321** - Python is a versatile, high-level programming language known for its clear syntax and readability...

**Query: "Best restaurants in New York"** *(9.7ms)*
1. **0.0987** - The best restaurants in New York City offer diverse cuisines from around the world. Michelin-starred establishments like Le Bernardin...
2. **0.2345** - Vegetarian restaurants have gained popularity worldwide, offering creative plant-based dishes that satisfy both vegans and omnivores...
3. **0.3678** - Food delivery apps like DoorDash, Uber Eats, and Grubhub have revolutionized how people order meals...

See [`DiskANNInRust/docs/demo_text_embedding.md`](DiskANNInRust/docs/demo_text_embedding.md) for the complete guide with 7 diverse example queries.

## Features

- Fast approximate nearest neighbor search
- Python FFI bindings for easy integration
- End-to-end text search with sentence transformers
- Memory-efficient index structures
- CLI tools for index building and search

## Project Structure

- **DiskANNInRust/**: Main Rust workspace with core library
- **examples/**: Python scripts for text search demo
- **docs/**: Documentation and guides