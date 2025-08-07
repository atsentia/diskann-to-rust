# Contributing to DiskANN-to-Rust

Welcome to the DiskANN-to-Rust project! This document outlines the contribution guidelines, safety guarantees, and invariants that make this codebase robust and reliable.

## Safety and Validation Framework

### Core Invariants and Guarantees

The DiskANN-to-Rust implementation maintains several critical invariants that ensure correctness and safety:

#### 1. Mathematical Invariants

**Distance Function Properties:**
- **Symmetry**: `d(a, b) == d(b, a)` for all distance functions
- **Non-negativity**: `d(a, b) >= 0` for all vectors a, b
- **Identity**: `d(a, a) == 0` for all vectors a
- **Triangle Inequality**: `d(a, c) <= d(a, b) + d(b, c)` for metric distance functions (Euclidean, Manhattan)

**Vector Operations:**
- **Normalization**: `||normalize(v)|| == 1` for non-zero vectors, `normalize(0) == 0`
- **Scaling**: `||α·v|| == |α|·||v||` for scalar α and vector v
- **Cauchy-Schwarz**: `|⟨u,v⟩| <= ||u||·||v||` for dot product
- **L2 Norm Consistency**: `||v||² == l2_norm_squared(v)`

#### 2. Graph Structure Invariants

**Vamana Index Properties:**
- **Connectivity**: All nodes remain reachable from the start node
- **Degree Bounds**: `node.neighbors.len() <= max_degree` for all nodes
- **Bidirectional Consistency**: If node A references B, the graph remains valid after B's removal
- **No Self-Loops**: `node.neighbors` never contains `node.id`
- **Unique Neighbors**: No duplicate neighbor IDs in adjacency lists

#### 3. File Format Invariants

**Binary Format Guarantees:**
- **Header Validation**: `num_points >= 0` and `num_dimensions > 0`
- **Data Size Consistency**: File size equals `8 + num_points * num_dimensions * 4` bytes
- **Endianness**: Little-endian format for cross-platform compatibility
- **Round-trip Fidelity**: `read(write(data)) == data` for all valid data

#### 4. Concurrent Access Safety

**Thread Safety Guarantees:**
- **Data Race Freedom**: No concurrent mutable access to shared data
- **Deadlock Prevention**: Consistent lock ordering in concurrent operations
- **Memory Safety**: No use-after-free or double-free errors
- **Atomic Operations**: Search operations are atomic with respect to index modifications

## Validation and Testing Framework

### 1. Fuzzing Infrastructure

Our fuzzing setup uses `cargo fuzz` to discover edge cases and security vulnerabilities:

**Fuzz Targets:**
- `fuzz_file_parser`: Tests binary file format parsing with malformed inputs
- `fuzz_search_query`: Tests search operations with random vectors and parameters
- `fuzz_vector_ops`: Tests mathematical operations for floating-point edge cases

**Running Fuzz Tests:**
```bash
cd DiskANNInRust
cargo +nightly fuzz run fuzz_file_parser
cargo +nightly fuzz run fuzz_search_query  
cargo +nightly fuzz run fuzz_vector_ops
```

**Interpreting Results:**
- **No crashes**: All invariants hold under random inputs
- **Crashes found**: Violations detected - must be addressed before merge
- **Timeout**: Infinite loops or performance issues detected

### 2. Property-Based Testing

We use `proptest` to verify mathematical properties across large input spaces:

**Property Test Categories:**
- **Mathematical Properties**: Triangle inequality, normalization, scaling laws
- **Round-trip Properties**: File I/O, serialization consistency  
- **Edge Case Handling**: Zero vectors, extreme values, empty inputs

**Running Property Tests:**
```bash
cargo test property_tests
```

### 3. Concurrent Testing with Loom

`loom` tests model concurrent execution to find race conditions and deadlocks:

**Concurrent Scenarios Tested:**
- Simultaneous insert/remove operations
- Concurrent search during index modification
- Edge addition/removal under contention
- Memory consistency across threads

**Running Loom Tests:**
```bash
RUSTFLAGS="--cfg loom" cargo test --test loom_tests
```

### 4. Memory Safety with Miri

Miri detects undefined behavior and memory safety violations:

**Running Miri Tests:**
```bash
cargo +nightly miri test
```

**Miri Checks:**
- Use-after-free detection
- Buffer overflow protection
- Uninitialized memory access
- Invalid pointer arithmetic

## Contribution Guidelines

### 1. Code Quality Standards

**Before Submitting a PR:**
1. **Run all tests**: `cargo test --all-features`
2. **Check formatting**: `cargo fmt --check`
3. **Run linting**: `cargo clippy -- -D warnings`
4. **Verify no regressions**: Existing tests must pass
5. **Add tests**: New functionality requires corresponding tests

### 2. Adding New Features

**For New Algorithms:**
- Add property tests verifying mathematical correctness
- Include fuzz targets for new parsing/processing logic
- Document performance characteristics and complexity
- Verify thread safety if applicable

**For New Distance Functions:**
- Verify metric properties (if applicable)
- Test symmetry, non-negativity, identity properties
- Add to fuzz test coverage
- Benchmark against existing implementations

**For New File Formats:**
- Implement round-trip property tests
- Add fuzz target for format parsing
- Document binary layout and endianness
- Ensure compatibility with existing tools

### 3. Performance Considerations

**Optimization Guidelines:**
- Profile before optimizing: Use `cargo bench` for baselines
- SIMD usage: Enable `simd` feature flag when beneficial
- Memory allocation: Prefer reusing buffers in hot paths
- Unsafe code: Requires careful review and documentation

**Performance Regression Prevention:**
- New features must not regress existing benchmarks by >5%
- Critical paths require micro-benchmarks
- Memory usage should be bounded and predictable

### 4. Safety-First Development

**Memory Safety:**
- Minimize `unsafe` code usage
- All `unsafe` blocks require safety comments
- Use tools: AddressSanitizer, Valgrind, Miri
- Prefer safe abstractions over raw pointers

**Numerical Stability:**
- Handle floating-point edge cases (NaN, infinity, denormals)
- Use appropriate epsilon comparisons
- Consider overflow/underflow in accumulation
- Test with extreme input values

## CI/CD and Automation

### Nightly Validation

Our nightly CI runs comprehensive validation:

1. **30+ minute fuzz sessions** across all targets
2. **Loom concurrent testing** with thousands of schedules
3. **Miri validation** under strict mode
4. **Coverage analysis** with 90% target
5. **Performance regression detection**

### Release Criteria

Before any release:
- [ ] Two consecutive nightly fuzz runs with zero crashes
- [ ] Coverage ≥ 90% on all core modules
- [ ] All Miri tests pass under strict mode
- [ ] Performance benchmarks within 5% of baseline
- [ ] All property tests pass with large iteration counts

## Debugging and Troubleshooting

### Common Issues

**Fuzz Crashes:**
1. Reproduce locally: `cargo fuzz run target_name artifacts/crash-file`
2. Debug with GDB: Set breakpoints in relevant code paths
3. Add property tests covering the failure case
4. Fix root cause, verify with extended fuzz run

**Property Test Failures:**
1. Examine the minimal failing case from proptest output
2. Add unit test for the specific scenario
3. Verify mathematical assumptions and edge case handling
4. Consider floating-point precision issues

**Loom Test Failures:**
1. Review concurrent access patterns
2. Ensure proper synchronization primitives
3. Check for missing atomic operations
4. Verify lock ordering consistency

**Miri Violations:**
1. Focus on `unsafe` code blocks first
2. Check array bounds and pointer arithmetic
3. Verify lifetime management
4. Review memory allocation/deallocation

### Useful Debug Commands

```bash
# Reproduce a specific fuzz crash
cargo fuzz run target_name path/to/crash-file

# Run property tests with more iterations
PROPTEST_CASES=10000 cargo test property_tests

# Enable detailed loom debugging
LOOM_LOG=trace RUSTFLAGS="--cfg loom" cargo test loom_tests

# Run Miri with maximum strictness
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri test
```

## Architecture and Design Principles

### Modular Design

- **diskann-core**: Mathematical primitives and data structures
- **diskann-traits**: Abstract interfaces and algorithms
- **diskann-impl**: Concrete algorithm implementations  
- **diskann-io**: File format handling and I/O operations
- **diskann-cli**: Command-line interface
- **diskann-ffi**: C FFI bindings for external integration

### Error Handling

- Use `Result<T, E>` for all fallible operations
- Provide meaningful error messages with context
- Fail fast on invariant violations
- Log errors at appropriate levels

### Documentation

- Document all public APIs with examples
- Include complexity analysis for algorithms
- Explain safety invariants for `unsafe` code
- Provide usage examples in module documentation

## Getting Help

- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for design questions
- **Code Review**: All PRs require review from maintainers
- **Documentation**: Check inline docs and this contributing guide

Thank you for contributing to DiskANN-to-Rust! Your efforts help build a safer, faster, and more reliable vector search library.