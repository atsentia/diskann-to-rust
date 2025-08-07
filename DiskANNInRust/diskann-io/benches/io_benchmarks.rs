//! Benchmarks for I/O performance comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use diskann_io::{IndexWriter, IndexLoader, FileIndexWriter, MmapIndexLoader, IndexWriterBuilder};
use diskann_core::vectors::Vector;
use tempfile::NamedTempFile;
use std::time::Duration;

fn create_test_vectors(num_vectors: usize, dimensions: usize) -> Vec<Vector> {
    (0..num_vectors)
        .map(|i| {
            (0..dimensions)
                .map(|j| (i * dimensions + j) as f32 * 0.001)
                .collect()
        })
        .collect()
}

fn bench_write_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_performance");
    
    // Test different buffer sizes
    let buffer_sizes = vec![1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024];
    let vectors = create_test_vectors(1000, 128);
    
    for buffer_size in buffer_sizes {
        group.bench_with_input(
            BenchmarkId::new("buffered_write", buffer_size),
            &buffer_size,
            |b, &buffer_size| {
                let writer = IndexWriterBuilder::new()
                    .buffer_size(buffer_size)
                    .build_file_writer();
                
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    writer.write_vectors(temp_file.path(), black_box(&vectors)).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_read_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_performance");
    
    // Create test data of different sizes
    let test_cases = vec![
        (100, 32),     // Small: 100 vectors, 32 dimensions
        (1000, 128),   // Medium: 1000 vectors, 128 dimensions
        (5000, 256),   // Large: 5000 vectors, 256 dimensions
    ];
    
    for (num_vectors, dimensions) in test_cases {
        let vectors = create_test_vectors(num_vectors, dimensions);
        let writer = FileIndexWriter::new();
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_vectors(temp_file.path(), &vectors).unwrap();
        
        // Benchmark memory-mapped loading
        group.bench_with_input(
            BenchmarkId::new("mmap_read", format!("{}x{}", num_vectors, dimensions)),
            &temp_file,
            |b, temp_file| {
                let loader = MmapIndexLoader::new();
                b.iter(|| {
                    black_box(loader.load_vectors(temp_file.path()).unwrap());
                });
            },
        );
        
        // Benchmark buffered loading
        group.bench_with_input(
            BenchmarkId::new("buffered_read", format!("{}x{}", num_vectors, dimensions)),
            &temp_file,
            |b, temp_file| {
                let loader = MmapIndexLoader::buffered_only();
                b.iter(|| {
                    black_box(loader.load_vectors(temp_file.path()).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_round_trip_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip_performance");
    group.measurement_time(Duration::from_secs(30));
    
    let vectors = create_test_vectors(2000, 128);
    
    group.bench_function("full_round_trip", |b| {
        b.iter(|| {
            let temp_file = NamedTempFile::new().unwrap();
            let writer = FileIndexWriter::new();
            let loader = MmapIndexLoader::new();
            
            // Write
            writer.write_vectors(temp_file.path(), black_box(&vectors)).unwrap();
            
            // Read back
            let loaded = loader.load_vectors(temp_file.path()).unwrap();
            black_box(loaded);
        });
    });
    
    group.finish();
}

fn bench_metadata_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_performance");
    
    let vectors = create_test_vectors(10000, 256);
    let writer = FileIndexWriter::new();
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    group.bench_function("metadata_extraction", |b| {
        let loader = MmapIndexLoader::new();
        b.iter(|| {
            black_box(loader.get_metadata(temp_file.path()).unwrap());
        });
    });
    
    group.finish();
}

fn bench_subset_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("subset_loading");
    
    let vectors = create_test_vectors(10000, 128);
    let writer = FileIndexWriter::new();
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    let subset_sizes = vec![10, 100, 1000];
    
    for subset_size in subset_sizes {
        group.bench_with_input(
            BenchmarkId::new("subset_load", subset_size),
            &subset_size,
            |b, &subset_size| {
                let loader = MmapIndexLoader::new();
                b.iter(|| {
                    black_box(loader.load_vectors_subset(temp_file.path(), 0, subset_size).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_alignment_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_performance");
    
    // Test loading with different alignment scenarios
    let vectors = create_test_vectors(1000, 128);
    let writer = FileIndexWriter::new();
    let temp_file = NamedTempFile::new().unwrap();
    writer.write_vectors(temp_file.path(), &vectors).unwrap();
    
    group.bench_function("aligned_access", |b| {
        let loader = MmapIndexLoader::new();
        b.iter(|| {
            // Load entire file - should be well-aligned
            black_box(loader.load_vectors(temp_file.path()).unwrap());
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_write_performance,
    bench_read_performance,
    bench_round_trip_performance,
    bench_metadata_extraction,
    bench_subset_loading,
    bench_alignment_performance
);

criterion_main!(benches);