# Getting Started

This guide will help you get up and running with DiskANN-Rust quickly.

## Prerequisites

- **Rust**: Version 1.80 or later
- **Operating System**: Linux, macOS, or Windows
- **Architecture**: x86_64 or ARM64 (aarch64)

## Installation

### From crates.io (Recommended)

Add DiskANN-Rust to your `Cargo.toml`:

```toml
[dependencies]
diskann-impl = "0.1.0"

# For enhanced performance (optional)
diskann-impl = { version = "0.1.0", features = ["simd"] }
```

### From Source

Clone the repository and build:

```bash
git clone https://github.com/atsentia/diskann-to-rust.git
cd diskann-to-rust/DiskANNInRust
cargo build --release
```

## Quick Start Example

Here's a minimal example to get you started:

```rust
use diskann_impl::{VamanaIndex, IndexBuilder, VamanaConfig};
use diskann_traits::distance::EuclideanDistance;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Prepare your data
    let vectors = vec![
        (0, vec![1.0, 2.0, 3.0]),
        (1, vec![4.0, 5.0, 6.0]),
        (2, vec![7.0, 8.0, 9.0]),
        (3, vec![2.0, 3.0, 4.0]),
        (4, vec![5.0, 6.0, 7.0]),
    ];

    // 2. Configure the index
    let config = VamanaConfig {
        max_degree: 32,        // Maximum connections per node
        search_list_size: 64,  // Construction search quality
        alpha: 1.2,            // Pruning parameter
        seed: 42,              // Reproducible builds
    };

    // 3. Build the index
    let index = IndexBuilder::new(EuclideanDistance)
        .max_degree(config.max_degree)
        .search_list_size(config.search_list_size)
        .alpha(config.alpha)
        .seed(config.seed)
        .build(vectors)?;

    // 4. Search for nearest neighbors
    let query = vec![3.0, 4.0, 5.0];
    let k = 3; // Find 3 nearest neighbors
    let results = index.search(&query, k)?;

    // 5. Use the results
    for result in results {
        println!("ID: {}, Distance: {:.3}", result.id, result.distance);
    }

    Ok(())
}
```

## Performance-Optimized Example

For maximum performance, enable SIMD and unsafe optimizations:

```rust
use diskann_impl::{VamanaIndex, IndexBuilder, VamanaConfig};
use diskann_traits::distance::EuclideanDistance;

// Enable high-performance features
// Add to Cargo.toml: features = ["simd", "unsafe_opt"]

fn high_performance_search() -> Result<(), Box<dyn std::error::Error>> {
    // Large dataset example
    let vectors: Vec<(u32, Vec<f32>)> = (0..10000)
        .map(|i| (i, vec![i as f32; 128])) // 128-dimensional vectors
        .collect();

    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 128,
        alpha: 1.2,
        seed: 42,
    };

    let index = IndexBuilder::new(EuclideanDistance)
        .max_degree(config.max_degree)
        .search_list_size(config.search_list_size)
        .alpha(config.alpha)
        .seed(config.seed)
        .build(vectors)?;

    // High-quality search with beam width
    let query = vec![5000.0; 128];
    let results = index.search_with_beam(&query, 10, 64)?;

    println!("Found {} results", results.len());
    Ok(())
}
```

## Command Line Usage

DiskANN-Rust includes a CLI that matches the original DiskANN interface:

```bash
# Build an index
cargo run --bin diskann -- build \
    --input vectors.bin \
    --output index.ann \
    --max-degree 32 \
    --search-list-size 64

# Search the index
cargo run --bin diskann -- search \
    --index index.ann \
    --query query.bin \
    --k 10 \
    --beam-width 64 \
    --output results.csv
```

## Configuration Guidelines

### For Interactive Applications (Low Latency)
```rust
VamanaConfig {
    max_degree: 32,
    search_list_size: 64,
    alpha: 1.2,
    seed: 42,
}

// Search with small beam width
let results = index.search_with_beam(&query, k, 16)?;
```

### For Batch Processing (High Throughput)
```rust
VamanaConfig {
    max_degree: 64,
    search_list_size: 128,
    alpha: 1.2,
    seed: 42,
}

// Search with larger beam width for better quality
let results = index.search_with_beam(&query, k, 128)?;
```

### For Memory-Constrained Environments
```rust
VamanaConfig {
    max_degree: 16,
    search_list_size: 32,
    alpha: 1.0,
    seed: 42,
}
```

## Feature Flags

Choose the right features for your use case:

```toml
[dependencies]
# Basic usage
diskann-impl = "0.1.0"

# High performance (recommended for production)
diskann-impl = { version = "0.1.0", features = ["simd"] }

# Maximum performance (requires careful validation)
diskann-impl = { version = "0.1.0", features = ["simd", "unsafe_opt"] }

# Memory-optimized builds
diskann-impl = { version = "0.1.0", features = ["jemalloc"] }

# Development and debugging
diskann-impl = { version = "0.1.0", features = ["tracing", "metrics"] }
```

## Common Patterns

### Zero-Allocation Search

For high-throughput scenarios, reuse search buffers:

```rust
use diskann_traits::search::SearchBuffer;

let mut buffer = SearchBuffer::new(1000);
loop {
    let query = get_next_query();
    let results = index.search_with_buffer(&query, k, beam_width, &mut buffer)?;
    process_results(results);
    // buffer is reused, no allocations
}
```

### Batch Processing

Process multiple queries efficiently:

```rust
let queries = load_queries()?;
let results: Vec<_> = queries
    .par_iter()  // Parallel processing with rayon
    .map(|query| index.search(query, k))
    .collect::<Result<Vec<_>, _>>()?;
```

### Error Handling

```rust
use diskann_core::DiskAnnError;

match index.search(&query, k) {
    Ok(results) => println!("Found {} results", results.len()),
    Err(DiskAnnError::DimensionMismatch { expected, actual }) => {
        eprintln!("Query dimension {} doesn't match index dimension {}", actual, expected);
    }
    Err(e) => eprintln!("Search failed: {}", e),
}
```

## Next Steps

- **Learn the API**: Read the [Programming Interface](./api.md) guide
- **Understand Architecture**: See [Architecture Overview](../architecture/overview.md)
- **Tune Performance**: Check out [Performance Tuning](../features/performance.md)
- **See Examples**: Browse the [Examples](../examples/build-index.md) section

## Getting Help

- 📖 **Documentation**: This book and [API docs](https://docs.rs/diskann-impl)
- 🐛 **Issues**: [GitHub Issues](https://github.com/atsentia/diskann-to-rust/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/atsentia/diskann-to-rust/discussions)
- 📧 **Email**: Contact the maintainers for enterprise support