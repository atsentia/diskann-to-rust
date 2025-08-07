# Architecture Overview

DiskANN-Rust is designed as a modular, high-performance system with clear separation of concerns. This chapter provides an overview of the system architecture and design principles.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DiskANN-Rust System                     │
├─────────────────────────────────────────────────────────────┤
│  diskann-cli     │  diskann-ffi     │  Application Layer   │
├─────────────────────────────────────────────────────────────┤
│             diskann-impl (Vamana Algorithm)                │
├─────────────────────────────────────────────────────────────┤
│  diskann-traits  │  diskann-io      │  diskann-bench      │
├─────────────────────────────────────────────────────────────┤
│                    diskann-core                            │
│          (SIMD, Math, Memory, Data Structures)             │
└─────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. **Zero-Cost Abstractions**
Every abstraction layer is designed to compile away, ensuring no runtime overhead:

```rust
// High-level API
let results = index.search(&query, k)?;

// Compiles to the same optimized code as:
// Manual beam search implementation
```

### 2. **Memory Safety**
All unsafe code is:
- Clearly marked and documented
- Justified with performance measurements (≥10% improvement)
- Isolated in small, auditable functions
- Tested with property-based testing

### 3. **Modular Design**
Each crate has a single responsibility:
- **diskann-core**: Fundamental data structures and algorithms
- **diskann-traits**: Interface definitions
- **diskann-impl**: Algorithm implementations
- **diskann-io**: File format handling
- **diskann-cli**: Command-line interface
- **diskann-ffi**: C interoperability

### 4. **Performance First**
Design decisions prioritize performance:
- SIMD-optimized distance calculations
- Memory-aligned data structures
- Cache-friendly graph layouts
- Zero-allocation search paths

## Data Flow Architecture

### Index Construction Flow

```
Vector Data → Parse & Validate → Memory Layout → Graph Construction → Index
     ↓              ↓                 ↓              ↓            ↓
diskann-io    diskann-core    diskann-core   diskann-impl   Serialized
```

### Search Flow

```
Query Vector → Distance Calc → Beam Search → Result Ranking → Results
     ↓             ↓             ↓             ↓           ↓
diskann-core  diskann-core  diskann-impl  diskann-impl  Application
```

## Memory Layout Design

### Vector Storage
```rust
#[repr(C, align(32))]  // SIMD-aligned
pub struct VectorData {
    pub dimensions: u32,
    pub data: Vec<f32>,  // Aligned for AVX operations
}
```

### Graph Representation
```rust
pub struct VamanaGraph {
    pub nodes: Vec<GraphNode>,     // Dense node storage
    pub edges: Vec<EdgeList>,      // Adjacency lists
    pub entry_point: VectorId,     // Single entry point
}
```

## Threading Model

### Construction: Parallel by Default
- Vector insertion: Parallel rayon iterators
- Edge pruning: Work-stealing thread pool
- Distance calculations: SIMD + threading

### Search: Lock-Free
- Read-only access to immutable graph
- Thread-local search buffers
- No synchronization overhead

```rust
// Thread-safe search
impl VamanaIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Candidate> {
        // No locks needed - immutable after construction
        self.beam_search(query, k, DEFAULT_BEAM_WIDTH)
    }
}
```

## Error Handling Strategy

### Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum DiskAnnError {
    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Memory alignment error")]
    Alignment,
}
```

### Error Propagation
- Library functions return `Result<T, DiskAnnError>`
- FFI layer converts to error codes
- CLI provides user-friendly error messages

## Performance Characteristics

### Time Complexity
| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| Index Build | O(n log n) | O(n log n) | O(n²) |
| Single Search | O(log n) | O(log n) | O(n) |
| Batch Search | O(b log n) | O(b log n) | O(bn) |

### Space Complexity
| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Vector Storage | O(nd) | n vectors, d dimensions |
| Graph Structure | O(nR) | R = max degree |
| Search Buffers | O(k + beam_width) | Per thread |

## Configuration Architecture

### Compile-Time Configuration
```rust
// Feature flags determine capabilities
#[cfg(feature = "simd")]
use crate::simd::distance_avx2;

#[cfg(not(feature = "simd"))]
use crate::scalar::distance_scalar;
```

### Runtime Configuration
```rust
pub struct VamanaConfig {
    pub max_degree: usize,         // Graph connectivity
    pub search_list_size: usize,   // Construction parameter
    pub beam_width: usize,         // Search quality/speed
    pub distance_function: DistanceFn,
}
```

## Integration Points

### C FFI Layer
```c
// Opaque handle for safety
typedef struct DiskAnnIndex* DiskAnnIndexHandle;

// Error-code based API
DiskAnnError diskann_search(
    DiskAnnIndexHandle handle,
    const float* query,
    uint32_t k,
    SearchResult* results
);
```

### Python Integration (Planned)
```python
# PyO3-based bindings
import diskann

index = diskann.VamanaIndex.build(vectors)
results = index.search(query, k=10)
```

## Extensibility Design

### Custom Distance Functions
```rust
pub trait DistanceFunction: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    
    #[cfg(feature = "simd")]
    fn distance_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance(a, b)  // Default to scalar
    }
}
```

### Custom Graph Algorithms
```rust
pub trait GraphConstructor {
    fn build_graph(&self, vectors: &[Vector]) -> Result<Graph, BuildError>;
}

impl GraphConstructor for VamanaBuilder {
    // Implementation details...
}
```

## Quality Assurance Architecture

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Property Tests**: Algorithm correctness
- **Fuzz Tests**: Input validation and robustness
- **Benchmark Tests**: Performance regression detection

### Monitoring Points
```rust
// Instrumentation points for performance monitoring
#[cfg(feature = "metrics")]
let _span = tracing::info_span!("beam_search").entered();

let candidates = self.explore_neighbors(query, beam_width)?;
```

---

**Next**: Learn about the [Core Components](./components.md) in detail.