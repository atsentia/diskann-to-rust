# DiskANN-Rust

A high-performance Rust port of the DiskANN approximate nearest neighbor search library.

## Project Structure

This workspace contains the following crates:

- **diskann-core**: Core data structures and algorithms
- **diskann-io**: I/O operations and file format handling  
- **diskann-traits**: Core traits and interfaces
- **diskann-impl**: Concrete implementations of algorithms
- **diskann-cli**: Command-line interface

## Features

- `simd`: Enable SIMD optimizations for distance computations
- `unsafe_opt`: Enable unsafe optimizations for performance
- `wasm`: WebAssembly support (experimental)

## Quick Start

### Building

```bash
# Build all crates
cargo build

# Build with SIMD optimizations
cargo build --features simd

# Build CLI in release mode
cargo build --release --bin diskann
```

### Testing

```bash
# Run all tests
cargo test

# Test specific features
cargo test --features simd
cargo test --features unsafe_opt
cargo test --no-default-features
```

### CLI Usage

```bash
# Build an index
cargo run --bin diskann -- build -i vectors.bin -o index.bin

# Search for nearest neighbors
cargo run --bin diskann -- search -i index.bin -q query.bin -k 10
```

## Development

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run performance baseline script
./scripts/perf_baseline.sh
```

### Code Quality

The project uses `#![deny(warnings)]` and is configured for:

- Security audits with `cargo audit`
- License and dependency checking with `cargo deny`
- Code coverage with `cargo tarpaulin`
- Benchmarking with Criterion

### CI/CD

The project includes comprehensive CI that tests on:

- **Platforms**: Linux, macOS, Windows  
- **Rust versions**: Stable 1.80+, Nightly
- **Features**: All feature combinations

## Project Status

This is Task 1 of the DiskANN-Rust roadmap - workspace bootstrap and CI setup.

✅ **Completed:**
- Cargo workspace with six-layer decomposition
- Feature flags (simd, wasm, unsafe_opt)
- CI pipeline with security audits
- Coverage and benchmark scaffolding
- CLI interface placeholder

🚧 **Next Steps:**
- Implement core distance computations
- Add vector I/O operations
- Build graph-based index structures
- Implement search algorithms

## License

MIT OR Apache-2.0
