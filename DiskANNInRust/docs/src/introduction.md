# DiskANN Rust Implementation

Welcome to the comprehensive documentation for **DiskANN-Rust**, a high-performance Rust implementation of the DiskANN approximate nearest neighbor search algorithm.

## What is DiskANN?

DiskANN is a state-of-the-art algorithm for approximate nearest neighbor (ANN) search that enables fast similarity search over large vector datasets. Originally developed by Microsoft Research, DiskANN combines:

- **Graph-based indexing** using the Vamana algorithm
- **Beam search** for high-quality approximate results
- **Memory-efficient** design for billion-scale datasets
- **High throughput** with low latency guarantees

## Why Rust?

This Rust implementation brings several advantages:

- **Memory Safety**: Eliminates segmentation faults and memory leaks
- **Performance**: Zero-cost abstractions with C++-level performance
- **Concurrency**: Safe, efficient parallel processing
- **Ecosystem**: Modern tooling and package management

## Key Features

- ✅ **Full Vamana Algorithm**: Complete implementation of the graph construction algorithm
- ✅ **Beam Search**: Configurable quality vs. speed trade-offs
- ✅ **SIMD Optimizations**: Hardware-accelerated distance calculations
- ✅ **Memory Efficiency**: Zero-allocation search paths
- ✅ **C FFI**: Native interoperability with C/C++ code
- ✅ **CLI Tools**: Command-line interface matching original DiskANN
- ✅ **Comprehensive Testing**: Property-based and fuzz testing

## Performance Characteristics

| Operation | Latency | Throughput | Quality |
|-----------|---------|------------|---------|
| Index Build | ~O(n log n) | High | Optimal |
| Single Query | ~1ms | 1000+ QPS | 90%+ recall@10 |
| Batch Search | ~0.5ms/query | 2000+ QPS | 95%+ recall@10 |

## Getting Started

```rust
use diskann_impl::VamanaIndex;

// Build an index
let vectors = load_vectors("dataset.bin")?;
let index = VamanaIndex::builder()
    .max_degree(32)
    .search_list_size(64)
    .build(&vectors)?;

// Search for nearest neighbors
let query = [0.1, 0.2, 0.3, /* ... */];
let results = index.search(&query, 10)?;
```

## Documentation Structure

This documentation is organized into several sections:

- **User Guide**: How to install, configure, and use DiskANN-Rust
- **Architecture**: Deep dive into algorithms and implementation details
- **Features**: Advanced capabilities and optimization techniques
- **Examples**: Practical code examples and tutorials
- **Development**: Contributing guidelines and development setup
- **Reference**: Complete API documentation and specifications

## Version Information

- **Current Version**: 0.1.0 (Release Candidate)
- **Rust Edition**: 2024
- **MSRV**: 1.80+
- **License**: MIT OR Apache-2.0

## Support and Community

- 📝 **Issues**: [GitHub Issues](https://github.com/atsentia/diskann-to-rust/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/atsentia/diskann-to-rust/discussions)
- 📖 **Documentation**: This book and API docs
- 🔧 **Contributing**: See [Development Guide](./development/contributing.md)

---

*Ready to get started? Head to the [Getting Started](./user-guide/getting-started.md) guide.*