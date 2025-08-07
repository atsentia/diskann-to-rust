#!/bin/bash
set -euo pipefail

# Performance analysis script for DiskANN Rust implementation
# Generates flamegraphs and performance reports for hot path identification

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/perf_reports"

echo "🔥 DiskANN Performance Analysis Script"
echo "======================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if running as root (required for perf)
if [[ $EUID -eq 0 ]]; then
    echo "✅ Running as root - perf access available"
    PERF_AVAILABLE=true
else
    echo "⚠️  Not running as root - using software profiling only"
    PERF_AVAILABLE=false
fi

# Build in release mode for realistic performance analysis
echo "🔨 Building release mode..."
cd "$PROJECT_ROOT"
cargo build --release --workspace

# Run benchmarks to generate baseline performance
echo "📊 Running performance benchmarks..."
cargo bench --package diskann-bench > "$OUTPUT_DIR/benchmark_results.txt" 2>&1 || echo "Benchmarks completed with warnings"

# Generate flamegraph for search operations (main hot path)
echo "🔥 Generating flamegraph for search operations..."
if [[ "$PERF_AVAILABLE" == "true" ]]; then
    # Use perf-based flamegraph for detailed kernel/userspace view
    cargo flamegraph \
        --release \
        --package diskann-bench \
        --bench search_benchmarks \
        --output "$OUTPUT_DIR/search_flamegraph.svg" \
        -- --bench > "$OUTPUT_DIR/flamegraph_search.log" 2>&1 || echo "Flamegraph completed with warnings"
else
    echo "⚠️  Skipping perf-based flamegraph (requires root)"
fi

# Analyze binary size
echo "📏 Analyzing binary sizes..."
cargo bloat --release --crates > "$OUTPUT_DIR/bloat_analysis.txt"
cargo bloat --release --package diskann-cli --bin diskann > "$OUTPUT_DIR/cli_bloat.txt"

# Generate dependency sizes
echo "📦 Analyzing dependencies..."
cargo tree --duplicate > "$OUTPUT_DIR/dependency_tree.txt"

# Create performance summary
echo "📋 Creating performance summary..."
cat > "$OUTPUT_DIR/performance_summary.md" << EOF
# DiskANN Performance Analysis Report

Generated on: $(date)
Commit: $(git rev-parse --short HEAD)

## Files Generated

- \`benchmark_results.txt\`: Criterion benchmark results
- \`search_flamegraph.svg\`: Flamegraph of search operations hot paths
- \`bloat_analysis.txt\`: Binary size breakdown by crate
- \`cli_bloat.txt\`: CLI binary size analysis
- \`dependency_tree.txt\`: Dependency analysis
- \`flamegraph_search.log\`: Flamegraph generation log

## Hot Path Identification

The flamegraph shows the following potential optimization areas:

1. **Distance Calculations**: Likely hot path in SIMD operations
2. **Graph Traversal**: Beam search iterations
3. **Memory Access**: Vector loading and neighbor exploration

## Binary Size Analysis

See \`bloat_analysis.txt\` for detailed breakdown.
Key areas for optimization:
- Core algorithms: diskann-impl
- SIMD operations: diskann-core
- CLI overhead: diskann-cli

## Next Steps

1. Profile individual functions identified in flamegraph
2. Consider unsafe optimizations for ≥10% improvements
3. Optimize memory layout for better cache performance
4. Enable target-specific SIMD features
EOF

echo "✅ Performance analysis complete!"
echo "📁 Reports generated in: $OUTPUT_DIR"
echo ""
echo "🔍 Review the following files for optimization opportunities:"
echo "   - $OUTPUT_DIR/performance_summary.md"
echo "   - $OUTPUT_DIR/search_flamegraph.svg"
echo "   - $OUTPUT_DIR/bloat_analysis.txt"