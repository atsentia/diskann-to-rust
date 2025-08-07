use criterion::{criterion_group, criterion_main, black_box, Criterion};
use diskann_core::{
    math::{l2_norm, l2_norm_squared, l1_norm, dot_product, normalize, fast_inv_sqrt_runtime},
    utils::{round_up, round_down, is_aligned, next_power_of_2, popcount},
    aligned_vec,
};
use diskann_traits::distance::{Distance, EuclideanDistance, ManhattanDistance, CosineDistance};

fn benchmark_math_functions(c: &mut Criterion) {
    let vector_small = vec![1.0f32; 128];
    let vector_large = vec![1.0f32; 1024];
    let vector_a = (0..512).map(|i| i as f32 * 0.1).collect::<Vec<f32>>();
    let vector_b = (0..512).map(|i| (i as f32 * 0.1) + 0.5).collect::<Vec<f32>>();

    c.bench_function("l2_norm_128", |b| {
        b.iter(|| l2_norm(black_box(&vector_small)))
    });
    
    c.bench_function("l2_norm_1024", |b| {
        b.iter(|| l2_norm(black_box(&vector_large)))
    });
    
    c.bench_function("l2_norm_squared_512", |b| {
        b.iter(|| l2_norm_squared(black_box(&vector_a)))
    });
    
    c.bench_function("l1_norm_512", |b| {
        b.iter(|| l1_norm(black_box(&vector_a)))
    });
    
    c.bench_function("dot_product_512", |b| {
        b.iter(|| dot_product(black_box(&vector_a), black_box(&vector_b)))
    });
    
    c.bench_function("normalize_512", |b| {
        b.iter(|| normalize(black_box(&vector_a)))
    });
    
    c.bench_function("fast_inv_sqrt_3_iter", |b| {
        b.iter(|| fast_inv_sqrt_runtime(black_box(4.0f32), black_box(3)))
    });
}

fn benchmark_distance_functions(c: &mut Criterion) {
    let euclidean = EuclideanDistance;
    let manhattan = ManhattanDistance;
    let cosine = CosineDistance;
    
    let vector_a = (0..512).map(|i| i as f32 * 0.1).collect::<Vec<f32>>();
    let vector_b = (0..512).map(|i| (i as f32 * 0.1) + 0.5).collect::<Vec<f32>>();
    
    c.bench_function("euclidean_distance_512", |b| {
        b.iter(|| euclidean.distance(black_box(&vector_a), black_box(&vector_b)))
    });
    
    c.bench_function("manhattan_distance_512", |b| {
        b.iter(|| manhattan.distance(black_box(&vector_a), black_box(&vector_b)))
    });
    
    c.bench_function("cosine_distance_512", |b| {
        b.iter(|| cosine.distance(black_box(&vector_a), black_box(&vector_b)))
    });
}

fn benchmark_utils_functions(c: &mut Criterion) {
    let values = [15u64, 127, 255, 1023, 4095, 16383, 65535];
    
    c.bench_function("round_up", |b| {
        b.iter(|| {
            for &val in &values {
                black_box(round_up(black_box(val), black_box(8)));
            }
        })
    });
    
    c.bench_function("round_down", |b| {
        b.iter(|| {
            for &val in &values {
                black_box(round_down(black_box(val), black_box(8)));
            }
        })
    });
    
    c.bench_function("is_aligned", |b| {
        b.iter(|| {
            for &val in &values {
                black_box(is_aligned(black_box(val), black_box(8)));
            }
        })
    });
    
    c.bench_function("next_power_of_2", |b| {
        b.iter(|| {
            for &val in &values {
                black_box(next_power_of_2(black_box(val)));
            }
        })
    });
    
    c.bench_function("popcount", |b| {
        b.iter(|| {
            for &val in &values {
                black_box(popcount(black_box(val)));
            }
        })
    });
}

fn benchmark_aligned_allocation(c: &mut Criterion) {
    c.bench_function("aligned_vec_f32_128", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = aligned_vec![f32; black_box(128)];
        })
    });
    
    c.bench_function("aligned_vec_f32_1024", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = aligned_vec![f32; black_box(1024)];
        })
    });
    
    c.bench_function("regular_vec_f32_1024", |b| {
        b.iter(|| {
            let _vec = vec![black_box(0.0f32); black_box(1024)];
        })
    });
}

criterion_group!(
    benches, 
    benchmark_math_functions,
    benchmark_distance_functions,
    benchmark_utils_functions,
    benchmark_aligned_allocation
);
criterion_main!(benches);