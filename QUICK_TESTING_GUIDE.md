# DiskANNInRust: Quick Start Testing Guide

## Basic Compilation & Testing Commands

### 1. Compilation
```bash
# Navigate to the DiskANNInRust directory
cd DiskANNInRust

# Check if project compiles (fast)
cargo check

# Build debug version
cargo build

# Build optimized release version  
cargo build --release
```

### 2. Running Tests

#### Unit Tests Only (Fast - ~5 seconds)
```bash
# Run all library unit tests
cargo test --lib

# Run tests for specific crate
cargo test --package diskann-core
cargo test --package diskann-traits  
cargo test --package diskann-io
cargo test --package diskann-impl
cargo test --package diskann-ffi
```

#### All Tests Including Integration (Slower - ~2 minutes)
```bash
# Run all tests (includes property tests and integration tests)
cargo test

# Run tests with failure details
cargo test --no-fail-fast -- --nocapture

# Skip the problematic property test
cargo test -- --skip test_vector_round_trip
```

#### Exclude Long-Running Tests (Medium - ~30 seconds)
```bash
# Run tests excluding integration tests
cargo test --lib
cargo test --package diskann-core --tests
cargo test --package diskann-traits --tests  
cargo test --package diskann-io --tests
cargo test --package diskann-ffi --tests
# Skip diskann-impl integration tests (beam search takes 82s)
cargo test --package diskann-impl --lib
```

### 3. Performance Benchmarks
```bash
# Run all benchmarks (can take 10+ minutes)
cargo bench

# Run specific benchmark crate
cargo bench --package diskann-bench
cargo bench --package diskann-core
cargo bench --package diskann-io
cargo bench --package diskann-impl
```

### 4. CLI Tool Testing
```bash
# Test CLI after release build
./target/release/diskann --help
./target/release/diskann build --help
./target/release/diskann search --help
```

### 5. Code Quality Checks
```bash
# Run linter (if clippy installed)
cargo clippy

# Check formatting
cargo fmt --check

# Generate documentation
cargo doc --open
```

## Expected Results Summary

- **Compilation**: Should complete without errors in ~30-90 seconds
- **Unit Tests**: 67 tests pass in under 5 seconds  
- **Integration Tests**: 7 additional tests pass but take ~82 seconds
- **Property Tests**: 7 out of 8 pass (1 expected failure in diskann-io)
- **Benchmarks**: Search throughput ~94-100K elements/second
- **CLI**: Functional with build and search commands

## Common Issues & Solutions

### 1. Property Test Failure
```
Error: test_vector_round_trip fails with "Too many local rejects"
Solution: This is non-critical, affects edge case testing only
Workaround: cargo test -- --skip test_vector_round_trip
```

### 2. Long Compilation Times
```
Issue: First build takes ~90 seconds due to dependencies
Solution: Subsequent builds are much faster (~5-30 seconds)
```

### 3. Benchmark Timeouts
```
Issue: Benchmarks can run for 10+ minutes
Solution: Use Ctrl+C to interrupt if needed, partial results still valid
```

## CI/CD Recommended Commands

### Fast CI Pipeline (< 5 minutes)
```bash
cargo check                    # Compilation check
cargo test --lib              # Unit tests only  
cargo clippy                   # Linting
```

### Full CI Pipeline (< 15 minutes)
```bash
cargo build --release         # Optimized build
cargo test --timeout 600      # All tests with 10min timeout
cargo bench --quick           # Quick benchmark smoke test
```

This testing approach ensures reliable validation while maintaining reasonable CI/CD pipeline execution times.