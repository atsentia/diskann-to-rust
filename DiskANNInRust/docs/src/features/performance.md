# Performance Tuning

This guide covers advanced performance optimization techniques for DiskANN-Rust, including profiling, unsafe optimizations, and system-level tuning.

## Performance Analysis Tools

### Built-in Profiling

DiskANN-Rust includes comprehensive performance analysis tools:

```bash
# Run flamegraph analysis to identify hot paths
./scripts/flamegraph_analysis.sh

# Generate performance reports
cargo run --release --bin diskann -- benchmark \
    --dataset vectors.bin \
    --output perf_report.json
```

### Using cargo-flamegraph

For detailed profiling with flamegraphs:

```bash
# Install flamegraph
cargo install flamegraph

# Profile search operations
cargo flamegraph --release --bench search_benchmarks \
    --output search_flamegraph.svg

# Profile index construction  
cargo flamegraph --release --bench construction_benchmarks \
    --output build_flamegraph.svg
```

### Memory Profiling

```bash
# Install memory profiler
cargo install cargo-bloat

# Analyze binary size
cargo bloat --release --crates

# Profile memory usage during execution
valgrind --tool=massif target/release/diskann
```

## Performance Optimization Levels

### Level 1: Compiler Optimizations

Basic release profile optimization:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

**Expected improvement**: 20-30% over debug builds

### Level 2: SIMD Optimizations

Enable SIMD for distance calculations:

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["simd"] }
```

```rust
// Automatically uses AVX2 on x86_64 or NEON on ARM64
use diskann_impl::VamanaIndex;

let index = VamanaIndex::new(distance_fn, config);
// Distance calculations now use SIMD when available
```

**Expected improvement**: 15-30% for distance-heavy workloads

### Level 3: Unsafe Optimizations (≥10% improvement required)

⚠️ **Warning**: Only use unsafe optimizations after careful validation

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["simd", "unsafe_opt"] }
```

```rust
#[cfg(feature = "unsafe_opt")]
use diskann_impl::optimized::*;

// Example: High-performance distance calculation
#[cfg(all(feature = "unsafe_opt", target_arch = "x86_64"))]
fn fast_distance(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() % 8 == 0 {
        // Safety: We've verified AVX2 support and alignment
        unsafe { euclidean_distance_avx2_unsafe(a, b) }
    } else {
        // Fallback to safe implementation
        diskann_core::math::euclidean_distance(a, b)
    }
}
```

**Expected improvement**: 10-25% for critical paths

### Level 4: Memory Allocator Optimization

Use high-performance allocators:

```toml
[dependencies]
diskann-impl = { version = "0.1.0", features = ["jemalloc"] }

# Alternative: mimalloc for Windows
diskann-impl = { version = "0.1.0", features = ["mimalloc"] }
```

**Expected improvement**: 5-15% for allocation-heavy workloads

## Specific Optimization Techniques

### 1. Distance Function Optimization

The most critical hot path in vector search:

```rust
// Standard implementation
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// Optimized with manual loop unrolling (unsafe_opt feature)
#[cfg(feature = "unsafe_opt")]
fn euclidean_distance_optimized(a: &[f32], b: &[f32]) -> f32 {
    // Safety: Bounds checking is caller's responsibility
    unsafe { diskann_impl::optimized::euclidean_distance_unrolled_unsafe(a, b) }
}

// SIMD-optimized (x86_64 with AVX2)
#[cfg(all(target_arch = "x86_64", feature = "unsafe_opt"))]
fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() % 8 == 0 {
        unsafe { diskann_impl::optimized::euclidean_distance_avx2_unsafe(a, b) }
    } else {
        euclidean_distance_optimized(a, b)
    }
}
```

### 2. Memory Layout Optimization

Align data structures for SIMD operations:

```rust
use std::alloc::{alloc, dealloc, Layout};

// Align vectors to 32-byte boundaries for AVX2
#[repr(align(32))]
struct AlignedVector {
    data: Vec<f32>,
}

impl AlignedVector {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size * 4, 32).unwrap();
        let ptr = unsafe { alloc(layout) as *mut f32 };
        let data = unsafe { Vec::from_raw_parts(ptr, size, size) };
        Self { data }
    }
}
```

### 3. Graph Traversal Optimization

Optimize memory access patterns:

```rust
#[cfg(feature = "unsafe_opt")]
fn beam_search_optimized(
    graph: &VamanaGraph,
    query: &[f32],
    beam_width: usize,
) -> Vec<Candidate> {
    // Prefetch neighbor data to reduce cache misses
    let neighbor_ptrs: Vec<*const f32> = candidates
        .iter()
        .flat_map(|c| graph.get_neighbors(c.id))
        .map(|id| graph.get_vector(id).as_ptr())
        .collect();
    
    // Safety: Pointers are valid within graph lifetime
    unsafe {
        diskann_impl::optimized::prefetch_neighbors(&neighbor_ptrs, 2);
    }
    
    // Continue with normal beam search...
}
```

### 4. Parallel Processing Optimization

Optimize for multi-core systems:

```rust
use rayon::prelude::*;

// Parallel index construction
fn build_index_parallel(vectors: Vec<(u32, Vec<f32>)>) -> VamanaIndex {
    let chunks: Vec<_> = vectors.chunks(1000).collect();
    
    let partial_graphs: Vec<_> = chunks
        .par_iter()
        .map(|chunk| build_partial_graph(chunk))
        .collect();
    
    merge_graphs(partial_graphs)
}

// Parallel batch search
fn search_batch_parallel(
    index: &VamanaIndex,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<SearchResult>> {
    queries
        .par_iter()
        .map(|query| index.search(query, k))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}
```

## Platform-Specific Optimizations

### x86_64 Optimizations

```toml
[target.'cfg(target_arch = "x86_64")'.dependencies]
diskann-impl = { version = "0.1.0", features = ["avx2", "fma"] }
```

```rust
#[cfg(target_arch = "x86_64")]
mod x86_optimizations {
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        // Use FMA (Fused Multiply-Add) for better accuracy and performance
        let mut sum = _mm256_setzero_ps();
        
        for chunk in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = _mm256_loadu_ps(chunk.0.as_ptr());
            let vb = _mm256_loadu_ps(chunk.1.as_ptr());
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        
        // Horizontal sum
        let sum_low = _mm256_castps256_ps128(sum);
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_quad = _mm_add_ps(sum_low, sum_high);
        let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
        let sum_single = _mm_hadd_ps(sum_dual, sum_dual);
        
        _mm_cvtss_f32(sum_single)
    }
}
```

### ARM64 Optimizations

```toml
[target.'cfg(target_arch = "aarch64")'.dependencies]
diskann-impl = { version = "0.1.0", features = ["neon"] }
```

```rust
#[cfg(target_arch = "aarch64")]
mod arm_optimizations {
    use std::arch::aarch64::*;
    
    #[target_feature(enable = "neon")]
    unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        
        for chunk in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let va = vld1q_f32(chunk.0.as_ptr());
            let vb = vld1q_f32(chunk.1.as_ptr());
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }
        
        // Horizontal sum
        let sum2 = vaddq_f32(sum, vrev64q_f32(sum));
        let sum1 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
        vget_lane_f32(sum1, 0).sqrt()
    }
}
```

## Performance Monitoring

### Runtime Performance Metrics

```rust
use std::time::Instant;

struct PerformanceMetrics {
    search_latency: Vec<Duration>,
    throughput: f64,
    cache_hit_rate: f64,
}

impl PerformanceMetrics {
    fn measure_search(&mut self, index: &VamanaIndex, query: &[f32], k: usize) {
        let start = Instant::now();
        let _results = index.search(query, k).unwrap();
        self.search_latency.push(start.elapsed());
    }
    
    fn p95_latency(&self) -> Duration {
        let mut sorted = self.search_latency.clone();
        sorted.sort();
        sorted[(sorted.len() as f32 * 0.95) as usize]
    }
    
    fn qps(&self) -> f64 {
        let total_time: Duration = self.search_latency.iter().sum();
        self.search_latency.len() as f64 / total_time.as_secs_f64()
    }
}
```

### Memory Usage Monitoring

```rust
#[cfg(feature = "metrics")]
use metrics::{counter, histogram, gauge};

fn monitor_memory_usage() {
    // Track memory allocation patterns
    gauge!("diskann.memory.heap_size").set(get_heap_size() as f64);
    gauge!("diskann.memory.graph_size").set(get_graph_memory_usage() as f64);
    
    // Track cache performance
    counter!("diskann.cache.hits").increment(1);
    counter!("diskann.cache.misses").increment(1);
    
    // Track operation performance
    histogram!("diskann.search.latency").record(search_duration.as_millis() as f64);
}
```

## Benchmarking Best Practices

### Proper Benchmark Setup

```rust
use criterion::{black_box, Criterion};

fn benchmark_search(c: &mut Criterion) {
    let index = setup_large_index();
    let queries = generate_realistic_queries();
    
    let mut group = c.benchmark_group("search_performance");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("baseline", |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &queries[query_idx % queries.len()];
            query_idx += 1;
            black_box(index.search(black_box(query), black_box(10)))
        });
    });
    
    #[cfg(feature = "unsafe_opt")]
    group.bench_function("optimized", |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &queries[query_idx % queries.len()];
            query_idx += 1;
            black_box(index.search_optimized(black_box(query), black_box(10)))
        });
    });
}
```

### Performance Regression Testing

```bash
#!/bin/bash
# Add to CI/CD pipeline

# Run benchmarks and compare with baseline
cargo bench --bench search_benchmarks -- --save-baseline current
cargo bench --bench search_benchmarks -- --baseline current

# Fail if performance regresses by more than 5%
if [[ $? -ne 0 ]]; then
    echo "Performance regression detected!"
    exit 1
fi
```

## Troubleshooting Performance Issues

### Common Performance Problems

1. **High latency**: Check beam width and distance function efficiency
2. **Low throughput**: Enable parallel processing and SIMD
3. **Memory pressure**: Use jemalloc and optimize data structures
4. **Cache misses**: Improve memory layout and prefetching

### Debugging Tools

```bash
# Profile with perf (Linux)
perf record -g cargo run --release --bin diskann
perf report

# Memory profiling with Valgrind
valgrind --tool=callgrind target/release/diskann
kcachegrind callgrind.out.*

# CPU profiling with Intel VTune (x86_64)
vtune -collect hotspots target/release/diskann
```

## Performance Targets

### Production Performance Goals

| Metric | Target | Notes |
|--------|--------|-------|
| Single Query Latency | <1ms | 90th percentile, k=10 |
| Batch Throughput | >1000 QPS | Concurrent queries |
| Memory Usage | <20 bytes/vector | Excluding vector storage |
| Index Build Time | O(n log n) | Parallel construction |
| Recall@10 | >90% | Quality vs speed trade-off |

### Performance Validation

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_search_latency_target() {
        let index = build_test_index(1000, 128);
        let query = vec![0.0; 128];
        
        let start = Instant::now();
        let _results = index.search(&query, 10).unwrap();
        let latency = start.elapsed();
        
        assert!(latency < Duration::from_millis(1), 
                "Search latency {} exceeded 1ms target", latency.as_millis());
    }
    
    #[test]
    fn test_throughput_target() {
        let index = build_test_index(10000, 128);
        let queries = generate_queries(1000, 128);
        
        let start = Instant::now();
        for query in &queries {
            let _results = index.search(query, 10).unwrap();
        }
        let duration = start.elapsed();
        
        let qps = queries.len() as f64 / duration.as_secs_f64();
        assert!(qps > 1000.0, "Throughput {} QPS below 1000 QPS target", qps);
    }
}
```

---

**Next**: Learn about [Memory Management](./memory.md) optimization techniques.