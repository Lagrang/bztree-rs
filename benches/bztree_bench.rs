use bztree::BzTree;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use crossbeam_utils::thread;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::time::Instant;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_insert");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let tree = BzTree::with_node_size(*size as u16);
                let tree = &tree;
                b.iter_custom(|iters| {
                    let mut keys: Vec<u64> = (0..iters).collect();
                    keys.shuffle(&mut thread_rng());

                    let cpus = num_cpus::get();
                    let per_thread_kv = keys.len() / cpus + 1;

                    let mut keys_per_thread = Vec::new();
                    for tid in 0..cpus {
                        let start = std::cmp::min(tid * per_thread_kv, keys.len());
                        let end = std::cmp::min((tid + 1) * per_thread_kv, keys.len());
                        keys_per_thread.push(&keys[start..end]);
                    }
                    let keys_per_thread = &keys_per_thread;

                    let start = thread::scope(|scope| {
                        for tid in 0..cpus {
                            scope.spawn(move |_| {
                                for i in keys_per_thread[tid] {
                                    let guard = crossbeam_epoch::pin();
                                    tree.insert(*i, *i, &guard);
                                }
                            });
                        }
                        Instant::now()
                    })
                    .unwrap();

                    start.elapsed()
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("concurrent_upsert");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let tree = BzTree::with_node_size(*size as u16);
                let tree = &tree;
                b.iter_custom(|iters| {
                    let mut keys: Vec<u64> = (0..iters).collect();
                    keys.shuffle(&mut thread_rng());

                    let cpus = num_cpus::get();
                    let per_thread_kv = keys.len() / cpus + 1;

                    let mut keys_per_thread = Vec::new();
                    for tid in 0..cpus {
                        let start = std::cmp::min(tid * per_thread_kv, keys.len());
                        let end = std::cmp::min((tid + 1) * per_thread_kv, keys.len());
                        keys_per_thread.push(&keys[start..end]);
                    }
                    let keys_per_thread = &keys_per_thread;

                    let start = thread::scope(|scope| {
                        for tid in 0..cpus {
                            scope.spawn(move |_| {
                                for i in keys_per_thread[tid] {
                                    let guard = crossbeam_epoch::pin();
                                    tree.upsert(*i, *i, &guard);
                                }
                            });
                        }
                        Instant::now()
                    })
                    .unwrap();

                    start.elapsed()
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("concurrent_delete");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let tree = &BzTree::with_node_size(*size as u16);
                b.iter_custom(|iters| {
                    let mut keys = Vec::new();
                    for i in 0..iters as u64 {
                        let guard = crossbeam_epoch::pin();
                        tree.upsert(i, i, &guard);
                        keys.push(i);
                    }

                    keys.shuffle(&mut thread_rng());
                    let cpus = num_cpus::get();
                    let per_thread_kv = keys.len() / cpus + 1;
                    let mut keys_per_thread = Vec::new();
                    for tid in 0..cpus {
                        let start = std::cmp::min(tid * per_thread_kv, keys.len());
                        let end = std::cmp::min((tid + 1) * per_thread_kv, keys.len());
                        keys_per_thread.push(&keys[start..end]);
                    }
                    let keys_per_thread = &keys_per_thread;

                    let start = thread::scope(|scope| {
                        (0..cpus).for_each(|tid| {
                            scope.spawn(move |_| {
                                for i in keys_per_thread[tid] {
                                    let guard = crossbeam_epoch::pin();
                                    tree.delete(i, &guard);
                                }
                            });
                        });
                        Instant::now()
                    })
                    .unwrap();
                    start.elapsed()
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("insert");
    for node_size in (40..=100).step_by(20) {
        group.bench_with_input(
            BenchmarkId::new("node_size", node_size),
            &node_size,
            |b, size| {
                let tree = BzTree::with_node_size(*size as u16);
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
                let tree = BzTree::with_node_size(*size as u16);
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
                let tree = BzTree::with_node_size(*size as u16);
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
                let tree = BzTree::with_node_size(*size as u16);
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
                let tree = BzTree::with_node_size(*size as u16);
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
