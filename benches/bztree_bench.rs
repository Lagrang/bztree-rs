use bztree::BzTree;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let mut tree = BzTree::with_node_size(*size as u16);
                let mut key = 0;
                b.iter(|| {
                    let guard = crossbeam_epoch::pin();
                    tree.insert(key, key, &guard);
                    key += 1;
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("upsert");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let mut tree = BzTree::with_node_size(*size as u16);
                let mut key = 0;
                b.iter(|| {
                    let guard = crossbeam_epoch::pin();
                    tree.upsert(key, key, &guard);
                    key += 1;
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("delete");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let mut tree = BzTree::with_node_size(*size as u16);
                b.iter_custom(|iters| {
                    for i in 0..iters {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i, i, &guard);
                    }
                    let start = Instant::now();
                    for i in 0..iters {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i, &guard).unwrap();
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("get");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let mut tree = BzTree::with_node_size(*size as u16);
                b.iter_custom(|iters| {
                    for i in 0..iters {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i, i, &guard);
                    }
                    let guard = unsafe { crossbeam_epoch::unprotected() };
                    let start = Instant::now();
                    for i in 0..iters {
                        tree.get(&i, &guard).unwrap();
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("scan");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let mut tree = BzTree::with_node_size(*size as u16);
                b.iter_custom(|iters| {
                    for i in 0..iters {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i, i, &guard);
                    }
                    let guard = unsafe { crossbeam_epoch::unprotected() };
                    let start = Instant::now();
                    for i in 0..iters {
                        tree.range(i..i + (node_size * 5), &guard).count();
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
