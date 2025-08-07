use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use diskann_impl::{IndexBuilder, VamanaConfig};
use diskann_traits::{distance::EuclideanDistance, search::{Search, SearchBuffer}};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

fn generate_test_data(num_vectors: usize, dimension: usize, seed: u64) -> Vec<(u32, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    
    (0..num_vectors)
        .map(|i| {
            let vector = (0..dimension)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0) // Range [-1, 1]
                .collect();
            (i as u32, vector)
        })
        .collect()
}

fn generate_queries(num_queries: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    
    (0..num_queries)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect()
}

fn build_index(vectors: Vec<(u32, Vec<f32>)>, config: VamanaConfig) -> impl Search<f32> {
    IndexBuilder::new(EuclideanDistance)
        .max_degree(config.max_degree)
        .search_list_size(config.search_list_size)
        .alpha(config.alpha)
        .seed(config.seed)
        .build(vectors)
        .expect("Failed to build index")
}

fn bench_search_throughput(c: &mut Criterion) {
    let num_vectors = 1000;
    let dimension = 128;
    let num_queries = 100;
    let k = 10;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let queries = generate_queries(num_queries, dimension, 42);
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    
    let mut group = c.benchmark_group("search_throughput");
    group.throughput(Throughput::Elements(num_queries as u64));
    
    // Benchmark different beam widths
    for beam_width in [16, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("beam_search", beam_width),
            &beam_width,
            |b, &beam_width| {
                let mut query_idx = 0;
                b.iter(|| {
                    let query = &queries[query_idx % queries.len()];
                    query_idx += 1;
                    black_box(index.search_with_beam(black_box(query), k, beam_width).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_search_latency(c: &mut Criterion) {
    let num_vectors = 10000;
    let dimension = 256;
    let k = 10;
    let beam_width = 64;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let query = generate_queries(1, dimension, 42).into_iter().next().unwrap();
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    
    c.bench_function("search_latency_p95", |b| {
        b.iter(|| {
            black_box(index.search_with_beam(black_box(&query), k, beam_width).unwrap())
        });
    });
}

fn bench_zero_allocation_search(c: &mut Criterion) {
    let num_vectors = 1000;
    let dimension = 128;
    let k = 10;
    let beam_width = 64;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let query = generate_queries(1, dimension, 42).into_iter().next().unwrap();
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    let mut buffer = SearchBuffer::new(1000);
    
    c.bench_function("zero_allocation_search", |b| {
        b.iter(|| {
            black_box(index.search_with_buffer(
                black_box(&query), 
                k, 
                beam_width, 
                black_box(&mut buffer)
            ).unwrap())
        });
    });
}

fn bench_index_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_construction");
    
    for num_vectors in [100, 500, 1000, 2000] {
        let dimension = 128;
        let vectors = generate_test_data(num_vectors, dimension, 42);
        
        group.throughput(Throughput::Elements(num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::new("vamana_construction", num_vectors),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let config = VamanaConfig {
                        max_degree: 64,
                        search_list_size: 100,
                        alpha: 1.2,
                        seed: 42,
                    };
                    
                    black_box(IndexBuilder::new(EuclideanDistance)
                        .max_degree(config.max_degree)
                        .search_list_size(config.search_list_size)
                        .alpha(config.alpha)
                        .seed(config.seed)
                        .build(black_box(vectors.clone()))
                        .unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_recall_quality(c: &mut Criterion) {
    let num_vectors = 5000;
    let dimension = 128;
    let num_queries = 50;
    let k = 10;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let queries = generate_queries(num_queries, dimension, 42);
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    
    c.bench_function("recall_at_10", |b| {
        b.iter(|| {
            let mut total_recall = 0.0;
            for query in &queries {
                // For this benchmark, we'll compare different beam widths
                let baseline_results = index.search_with_beam(query, k * 2, 128).unwrap();
                let test_results = index.search_with_beam(query, k, 64).unwrap();
                
                // Calculate recall (simplified)
                let baseline_ids: std::collections::HashSet<_> = 
                    baseline_results.iter().take(k).map(|r| r.id).collect();
                let test_ids: std::collections::HashSet<_> = 
                    test_results.iter().take(k).map(|r| r.id).collect();
                
                let intersection = baseline_ids.intersection(&test_ids).count();
                let recall = intersection as f64 / k as f64;
                total_recall += recall;
            }
            black_box(total_recall / queries.len() as f64)
        });
    });
}

/// QPS (Queries Per Second) measurement
fn measure_qps() -> f64 {
    let num_vectors = 1000;
    let dimension = 128;
    let num_queries = 1000;
    let k = 10;
    let beam_width = 64;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let queries = generate_queries(num_queries, dimension, 42);
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    
    let start = Instant::now();
    for query in &queries {
        let _ = index.search_with_beam(query, k, beam_width).unwrap();
    }
    let duration = start.elapsed();
    
    num_queries as f64 / duration.as_secs_f64()
}

/// P95 latency measurement
fn measure_p95_latency() -> Duration {
    let num_vectors = 1000;
    let dimension = 128;
    let num_queries = 1000;
    let k = 10;
    let beam_width = 64;
    
    let vectors = generate_test_data(num_vectors, dimension, 42);
    let queries = generate_queries(num_queries, dimension, 42);
    
    let config = VamanaConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        seed: 42,
    };
    
    let index = build_index(vectors, config);
    
    let mut latencies = Vec::new();
    for query in &queries {
        let start = Instant::now();
        let _ = index.search_with_beam(query, k, beam_width).unwrap();
        latencies.push(start.elapsed());
    }
    
    latencies.sort();
    latencies[((latencies.len() as f64) * 0.95) as usize]
}

fn bench_performance_metrics(c: &mut Criterion) {
    c.bench_function("qps_measurement", |b| {
        b.iter(|| {
            black_box(measure_qps())
        });
    });
    
    c.bench_function("p95_latency_measurement", |b| {
        b.iter(|| {
            black_box(measure_p95_latency())
        });
    });
}

criterion_group!(
    benches,
    bench_search_throughput,
    bench_search_latency,
    bench_zero_allocation_search,
    bench_index_construction,
    bench_recall_quality,
    bench_performance_metrics
);
criterion_main!(benches);