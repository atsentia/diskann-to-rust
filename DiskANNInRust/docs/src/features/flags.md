# Feature Flags

DiskANN-Rust uses Cargo feature flags to enable optional functionality and optimizations. This allows you to customize the build for your specific use case while keeping the binary size minimal.

## Core Features

### `default`
Enables the standard feature set for most use cases:
- Basic distance functions
- Standard memory allocators
- Safe implementations

```toml
[dependencies]
diskann-impl = "0.1.0"  # Uses default features
```

### `simd`
**Status**: ✅ Available  
**Performance Impact**: +15-30% for distance calculations

Enables SIMD optimizations for distance computations:
- AVX2 instructions on x86_64
- NEON instructions on ARM64
- Fallback to scalar for unsupported architectures

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["simd"] }
```

```rust
// Example with SIMD optimization
use diskann_core::distances::euclidean_simd;

let distance = euclidean_simd(&query, &candidate)?;
```

### `unsafe_opt`
**Status**: 🔬 Experimental  
**Performance Impact**: +10-20% for hot paths  
**Safety**: Requires careful review

Enables unsafe optimizations that provide significant performance improvements:
- Manual prefetching for graph traversal
- Unchecked array access in hot loops
- Pointer arithmetic optimizations

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["unsafe_opt"] }
```

**⚠️ Warning**: Only enable `unsafe_opt` if:
1. You need ≥10% performance improvement
2. You've validated correctness in your use case
3. You understand the safety implications

## Memory Management Features

### `jemalloc`
**Performance Impact**: +5-15% for large datasets

Uses jemalloc as the global allocator for better memory performance:

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["jemalloc"] }
```

### `mimalloc`
**Performance Impact**: +5-10% for allocation-heavy workloads

Alternative allocator optimized for multi-threaded performance:

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["mimalloc"] }
```

## I/O Features

### `mmap`
**Default**: Enabled  
Memory-mapped file I/O for large datasets:

```rust
use diskann_io::MmapVectorLoader;

let loader = MmapVectorLoader::new("large_dataset.bin")?;
let vectors = loader.load_subset(0..1000000)?;
```

### `async-io`
**Status**: 🔬 Experimental  
Asynchronous I/O operations:

```toml
[dependencies]
diskann-io = { version = "0.1.0", features = ["async-io"] }
```

## Integration Features

### `ffi`
**Default**: Enabled  
C FFI layer for interoperability:

```c
#include "diskann.h"

DiskAnnIndexHandle handle = diskann_create_index();
// ... use index
diskann_destroy_index(handle);
```

### `python`
**Status**: 🚧 Planned  
Python bindings via PyO3:

```python
import diskann_py

index = diskann_py.VamanaIndex.build(vectors)
results = index.search(query, k=10)
```

## Debugging Features

### `tracing`
**Default**: Enabled in dev builds  
Structured logging and tracing:

```rust
use tracing::{info, debug};

debug!("Starting beam search with width {}", beam_width);
let results = index.search_with_beam(query, k, beam_width)?;
info!("Search completed in {:?}", elapsed);
```

### `metrics`
Performance metrics collection:

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["metrics"] }
```

```rust
use diskann_impl::metrics::SearchMetrics;

let metrics = SearchMetrics::new();
let results = index.search_instrumented(query, k, &metrics)?;
println!("Search latency: {:?}", metrics.latency());
```

## Build Configuration

### Optimized Release Build

For maximum performance in production:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"

[dependencies]
diskann-impl = { 
    version = "0.1.0", 
    features = ["simd", "jemalloc"] 
}
```

### Development Build

For development with debugging support:

```toml
[dependencies]
diskann-impl = { 
    version = "0.1.0", 
    features = ["tracing", "metrics"] 
}
```

### Minimal Build

For size-constrained environments:

```toml
[dependencies]
diskann-impl = { 
    version = "0.1.0", 
    default-features = false,
    features = ["basic"] 
}
```

## Platform-Specific Features

### x86_64 Optimizations
```toml
[target.'cfg(target_arch = "x86_64")'.dependencies]
diskann-impl = { version = "0.1.0", features = ["avx2", "simd"] }
```

### ARM64 Optimizations
```toml
[target.'cfg(target_arch = "aarch64")'.dependencies]
diskann-impl = { version = "0.1.0", features = ["neon", "simd"] }
```

## Feature Compatibility Matrix

| Feature | `simd` | `unsafe_opt` | `jemalloc` | `mimalloc` |
|---------|--------|--------------|------------|------------|
| `simd` | ✅ | ✅ | ✅ | ✅ |
| `unsafe_opt` | ✅ | ✅ | ✅ | ✅ |
| `jemalloc` | ✅ | ✅ | ✅ | ❌ |
| `mimalloc` | ✅ | ✅ | ❌ | ✅ |

## Performance Impact Summary

| Feature | Build Time | Binary Size | Runtime Performance |
|---------|------------|-------------|-------------------|
| `default` | Baseline | Baseline | Baseline |
| `simd` | +10% | +50KB | +15-30% |
| `unsafe_opt` | +5% | +20KB | +10-20% |
| `jemalloc` | +20% | +500KB | +5-15% |
| `tracing` | +15% | +200KB | -2-5% |

---

**Next**: Learn about [Performance Tuning](./performance.md) techniques.