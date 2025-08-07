use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_placeholder(_c: &mut Criterion) {
    // TODO: Add actual I/O benchmarks when functionality is implemented
}

criterion_group!(benches, benchmark_placeholder);
criterion_main!(benches);