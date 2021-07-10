use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Instant;

use rand::prelude::*;
use rand::seq::SliceRandom;

use crate::history_verifier::{History, Ops};
use bztree::BzTree;
use crossbeam_utils::thread;
use std::fmt::Debug;
use std::hash::Hash;

mod history_verifier;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn test<F, TreeSupplier, K, V>(tree_creator: TreeSupplier, test: F)
where
    F: Fn(&BzTree<K, V>, usize, usize),
    K: Clone + Ord + Hash + Debug,
    V: Send + Sync,
    TreeSupplier: Fn(usize) -> BzTree<K, V>,
{
    let cpus = num_cpus::get();
    let per_thread_changes = 5000;
    for mult in 1..=3 {
        let threads = cpus * mult;
        let node_size = thread_rng().gen_range(50..100);
        println!("Node size: {}", node_size);
        let bz_tree = tree_creator(node_size);
        test(&bz_tree, threads, per_thread_changes);
    }
}

#[test]
fn insert_of_non_overlaping_keys_and_search() {
    test(
        |size| BzTree::<String, usize>::with_node_size(size as u16),
        |tree, threads, shard_size: usize| {
            thread::scope(|scope| {
                for id in 0..threads {
                    scope.spawn(move |_| {
                        let thread_id = id;
                        let base = shard_size * thread_id;
                        let mut keys: Vec<usize> = (base..base + shard_size).collect();
                        keys.shuffle(&mut thread_rng());
                        for i in keys {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            let value = thread_rng().gen::<usize>();
                            if thread_rng().gen_bool(0.5) {
                                tree.insert(key.clone(), value, &guard);
                            } else {
                                tree.upsert(key.clone(), value, &guard);
                            }

                            let found_val = tree
                                .get(&key, &guard)
                                .unwrap_or_else(|| panic!("{:?} not found", &key));
                            assert_eq!(found_val, &value);
                        }
                    });
                }
            })
            .unwrap();
        },
    );
}

#[test]
fn upsert_of_overlaping_keys() {
    test(
        |size| BzTree::<String, usize>::with_node_size(size as u16),
        |tree, threads, thread_changes_count| {
            let mut per_thread_elem_set = Vec::new();
            for _ in 0..threads {
                let mut elems = Vec::with_capacity(thread_changes_count);
                (0..thread_changes_count).for_each(|i| elems.push(i));
                elems.shuffle(&mut thread_rng());
                per_thread_elem_set.push(elems);
            }

            let history = thread::scope(|scope| {
                let mut handles = Vec::new();
                for elems in &per_thread_elem_set {
                    handles.push(scope.spawn(move |_| {
                        let mut ops = Ops::new();
                        for i in elems {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            let value = thread_rng().gen::<usize>();
                            let start = Instant::now();
                            tree.upsert(key.clone(), value, &guard);
                            ops.insert(key, value, start);
                        }
                        ops
                    }));
                }

                let ops = handles
                    .drain(..)
                    .map(|h| h.join().unwrap())
                    .fold(Ops::new(), |ops1, ops2| ops1.merge(ops2));

                History::based_on(ops)
            })
            .unwrap();

            let guard = crossbeam_epoch::pin();
            history.run_check(|key| tree.get(key, &guard));
        },
    );
}

#[test]
fn add_and_delete() {
    let max_val = 50;
    test(
        |size| BzTree::<String, usize>::with_node_size(size as u16),
        |tree, threads, thread_changes| {
            let mut per_thread_elem_set = Vec::with_capacity(threads);
            for _ in 0..threads {
                // each thread upsert and delete small range of keys
                // to increase contention on same values between threads
                let mut indexes = Vec::with_capacity(thread_changes);
                (0..thread_changes).for_each(|_| indexes.push(thread_rng().gen_range(1..max_val)));
                per_thread_elem_set.push(indexes);
            }

            let history = thread::scope(|scope| {
                let mut handles = Vec::new();
                for elems in &per_thread_elem_set {
                    handles.push(scope.spawn(move |_| {
                        let mut ops = Ops::new();
                        for i in elems {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            let value = thread_rng().gen_range(1..100000);
                            let start = Instant::now();
                            if thread_rng().gen_bool(0.4) {
                                tree.upsert(key.clone(), value, &guard);
                                ops.insert(key, value, start);
                            } else if thread_rng().gen_bool(0.3)
                                && tree.delete(&key, &crossbeam_epoch::pin()).is_some()
                            {
                                ops.delete(key.clone(), start);
                            } else if tree.insert(key.clone(), value, &crossbeam_epoch::pin()) {
                                ops.insert(key.clone(), value, start);
                            }
                        }
                        ops
                    }));
                }

                let ops = handles
                    .drain(..)
                    .map(|h| h.join().unwrap())
                    .fold(Ops::new(), |ops1, ops2| ops1.merge(ops2));

                History::based_on(ops)
            })
            .unwrap();

            let guard = crossbeam_epoch::pin();
            history.run_check(|key| tree.get(key, &guard));
        },
    );
}

#[test]
fn key_search() {
    test(
        |size| BzTree::<String, usize>::with_node_size(size as u16),
        |tree, threads, changes| {
            thread::scope(|scope| {
                let mut keys: Vec<usize> = (0..changes * threads).collect();
                keys.shuffle(&mut thread_rng());
                for i in keys {
                    let guard = crossbeam_epoch::pin();
                    let key = i.to_string();
                    let value = i;
                    tree.insert(key, value, &guard);
                }

                for thread_id in 0..threads {
                    scope.spawn(move |_| {
                        let base = changes * thread_id;
                        for i in base..base + changes {
                            let key = i.to_string();
                            let guard = crossbeam_epoch::pin();
                            let found_val = tree
                                .get(&key, &guard)
                                .unwrap_or_else(|| panic!("{:?} not found", &key));
                            assert_eq!(*found_val, i);
                        }
                    });
                }
            })
            .unwrap();
        },
    );
}

#[test]
fn overlapped_inserts_and_deletes() {
    test(
        |size| BzTree::<String, String>::with_node_size(size as u16),
        |tree, threads, changes| {
            let min = 0;
            let max = (threads - 1) * changes + changes;
            let mid = max / 2;
            for thread_id in 0..threads {
                let guard = crossbeam_epoch::pin();
                let mut keys: Vec<usize> =
                    (thread_id * changes..thread_id * changes + changes).collect();
                keys.shuffle(&mut thread_rng());
                for i in keys {
                    tree.insert(i.to_string(), i.to_string(), &guard);
                }
            }

            thread::scope(|scope| {
                scope.spawn(move |_| {
                    for i in min..max {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }

                    for i in min..max {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });

                scope.spawn(move |_| {
                    for i in (min..max).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }

                    for i in (min..max).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });

                scope.spawn(move |_| {
                    for i in min..mid {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }
                    for i in min..mid {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });

                scope.spawn(move |_| {
                    for i in (mid..max).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }
                    for i in (mid..max).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });

                scope.spawn(move |_| {
                    for i in (min..mid).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }
                    for i in (min..mid).rev() {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });

                scope.spawn(move |_| {
                    for i in mid..max {
                        let guard = crossbeam_epoch::pin();
                        tree.insert(i.to_string(), i.to_string(), &guard);
                    }
                    for i in mid..max {
                        let guard = crossbeam_epoch::pin();
                        tree.delete(&i.to_string(), &guard);
                    }
                });
            })
            .unwrap();

            assert!(tree.iter(&crossbeam_epoch::pin()).next().is_none());
        },
    );
}

#[test]
fn scan() {
    struct ThreadState {
        last_written_value: AtomicI64,
        val_range: Range<i64>,
    }

    test(
        |size| BzTree::<i64, i64>::with_node_size(size as u16),
        |tree, threads, changes| {
            let mut thread_checkpoint = HashMap::new();
            for thread_id in 0..threads {
                thread_checkpoint.insert(
                    thread_id,
                    ThreadState {
                        last_written_value: AtomicI64::new(-1),
                        val_range: ((thread_id * changes) as i64
                            ..(thread_id * changes + changes) as i64),
                    },
                );
            }

            thread::scope(|scope| {
                for thread_id in 0..threads {
                    let state = thread_checkpoint.get(&thread_id).unwrap();
                    scope.spawn(move |_| {
                        for i in state.val_range.clone() {
                            let guard = crossbeam_epoch::pin();
                            let key = i;
                            if thread_rng().gen_bool(0.3) {
                                tree.insert(key, key, &guard);
                            } else {
                                tree.upsert(key, key, &guard);
                            }
                            state.last_written_value.store(key, Ordering::Release);
                        }
                    });
                }

                // spawn same count of 'monitoring' threads which
                // check are all values written by insertion
                // thread visible to scanner thread
                for thread_id in 0..threads {
                    let state = thread_checkpoint.get(&thread_id).unwrap();

                    // check forward scan
                    scope.spawn(move |_| loop {
                        let guard = crossbeam_epoch::pin();
                        let last_written_val = state.last_written_value.load(Ordering::Acquire);
                        let scanned: Vec<i64> = tree
                            .range(state.val_range.clone(), &guard)
                            .map(|(_, v)| *v)
                            .collect();
                        let expected: Vec<i64> =
                            (state.val_range.start..=last_written_val).collect();
                        assert!(
                            scanned.starts_with(&expected),
                            "scanned: {:?}, expected: {:?}; Range: {:?}---{:?}",
                            scanned,
                            expected,
                            state.val_range,
                            (state.val_range.start..=last_written_val)
                        );

                        if state.val_range.end == last_written_val + 1 {
                            break;
                        }
                    });

                    // check reversed scan
                    scope.spawn(move |_| loop {
                        let guard = crossbeam_epoch::pin();
                        let last_written_val = state.last_written_value.load(Ordering::Acquire);
                        let scanned: Vec<i64> = tree
                            .range(state.val_range.clone(), &guard)
                            .rev()
                            .map(|(_, v)| *v)
                            .collect();
                        let expected: Vec<i64> =
                            (state.val_range.start..=last_written_val).rev().collect();
                        assert!(
                            scanned.ends_with(&expected),
                            "Reversed= scanned: {:?}, expected: {:?}; Range: {:?}---{:?}",
                            scanned,
                            expected,
                            state.val_range,
                            (state.val_range.start..=last_written_val)
                        );

                        if state.val_range.end == last_written_val + 1 {
                            break;
                        }
                    });

                    // check iter()
                    scope.spawn(move |_| loop {
                        let guard = crossbeam_epoch::pin();
                        let last_written_val = state.last_written_value.load(Ordering::Acquire);

                        let scanned: Vec<i64> = tree
                            .iter(&guard)
                            .filter_map(|(_, v)| {
                                if state.val_range.contains(v) {
                                    Some(*v)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let expected: Vec<i64> =
                            (state.val_range.start..=last_written_val).collect();
                        assert!(
                            scanned.starts_with(&expected),
                            "scanned: {:?}, expected: {:?}",
                            scanned,
                            expected,
                        );

                        if state.val_range.end == last_written_val + 1 {
                            break;
                        }
                    });
                }
            })
            .unwrap();
        },
    );
}

#[test]
fn scan_with_deletes() {
    struct ThreadState {
        last_removed_value: AtomicUsize,
        val_range: Range<usize>,
    }

    test(
        |size| BzTree::<usize, usize>::with_node_size(size as u16),
        |tree, threads, changes| {
            let mut thread_checkpoint = HashMap::new();
            for thread_id in 0..threads {
                let state = ThreadState {
                    last_removed_value: AtomicUsize::new(thread_id * changes + changes),
                    val_range: ((thread_id * changes)..(thread_id * changes + changes)),
                };
                for i in state.val_range.clone() {
                    let guard = crossbeam_epoch::pin();
                    tree.insert(i, i, &guard);
                }
                thread_checkpoint.insert(thread_id, state);
            }

            thread::scope(|scope| {
                // spawn same count of 'monitoring' threads which
                // check that all values deleted not visible to scanner thread
                for thread_id in 0..threads {
                    let state = thread_checkpoint.get(&thread_id).unwrap();

                    scope.spawn(move |_| {
                        for i in state.val_range.clone().rev() {
                            let guard = crossbeam_epoch::pin();
                            tree.delete(&i, &guard).unwrap();
                            state.last_removed_value.store(i, Ordering::Release);
                        }
                    });

                    // check forward scan
                    scope.spawn(move |_| loop {
                        let guard = crossbeam_epoch::pin();
                        let last_removed_val = state.last_removed_value.load(Ordering::Acquire);
                        assert!(tree
                            .range(state.val_range.clone(), &guard)
                            .filter(|(_, v)| {
                                (last_removed_val..state.val_range.end).contains(&v)
                            })
                            .next()
                            .is_none());

                        if state.val_range.start == last_removed_val {
                            break;
                        }
                    });

                    // check iter()
                    scope.spawn(move |_| loop {
                        let guard = crossbeam_epoch::pin();
                        let last_removed_val = state.last_removed_value.load(Ordering::Acquire);
                        assert!(tree
                            .iter(&guard)
                            .filter(|(_, v)| {
                                (last_removed_val..state.val_range.end).contains(&v)
                            })
                            .next()
                            .is_none());

                        if state.val_range.start == last_removed_val {
                            break;
                        }
                    });
                }
            })
            .unwrap();
        },
    );
}

#[test]
fn compute_with_value_update() {
    test(
        |size| BzTree::<usize, usize>::with_node_size(size as u16),
        |tree, threads, iters| {
            for i in 0..threads * 3 {
                assert!(tree.insert(i, 0, &crossbeam_epoch::pin()));
            }

            let history = thread::scope(|scope| {
                let mut handles = Vec::new();
                for _ in 0..threads {
                    handles.push(scope.spawn(|_| {
                        let mut ops = Ops::new();
                        for _ in 0..iters {
                            let key = thread_rng().gen_range(0..threads * 3);
                            assert!(
                                tree.compute(
                                    &key,
                                    |(_, v)| {
                                        ops.insert(key, v + 1, Instant::now());
                                        Some(v + 1)
                                    },
                                    &crossbeam_epoch::pin()
                                ),
                                "{:?} not found",
                                key
                            );
                        }
                        ops
                    }));
                }
                let ops = handles
                    .drain(..)
                    .map(|h| h.join().unwrap())
                    .fold(Ops::new(), |ops1, ops2| ops1.merge(ops2));
                History::based_on(ops)
            })
            .unwrap();

            let guard = crossbeam_epoch::pin();
            history.run_check(|key| tree.get(key, &guard));
        },
    );
}

#[test]
fn compute_with_value_delete() {
    test(
        |size| BzTree::<usize, usize>::with_node_size(size as u16),
        |tree, threads, iters| {
            let history = thread::scope(|scope| {
                let mut handles = Vec::new();
                for _ in 0..threads {
                    handles.push(scope.spawn(|_| {
                        let mut ops = Ops::new();
                        for _ in 0..iters {
                            let key = thread_rng().gen_range(0..threads * 3);
                            if thread_rng().gen_bool(0.5) {
                                let mut new_val: usize = 0;
                                if tree.compute(
                                    &key,
                                    |(_, v)| {
                                        new_val = v + 1;
                                        Some(v + 1)
                                    },
                                    &crossbeam_epoch::pin(),
                                ) {
                                    ops.insert(key, new_val, Instant::now());
                                } else {
                                    tree.insert(key, 0, &crossbeam_epoch::pin());
                                    ops.insert(key, 0, Instant::now());
                                }
                            } else {
                                if tree.compute(&key, |(_, _)| None, &crossbeam_epoch::pin()) {
                                    ops.delete(key, Instant::now());
                                } else {
                                    tree.insert(key, 0, &crossbeam_epoch::pin());
                                    ops.insert(key, 0, Instant::now());
                                }
                            }
                        }
                        ops
                    }));
                }
                let ops = handles
                    .drain(..)
                    .map(|h| h.join().unwrap())
                    .fold(Ops::new(), |ops1, ops2| ops1.merge(ops2));
                History::based_on(ops)
            })
            .unwrap();

            let guard = crossbeam_epoch::pin();
            history.run_check(|key| tree.get(key, &guard));
        },
    );
}
#[test]
fn liveness() {
    test(
        |size| BzTree::<String, usize>::with_node_size(size as u16),
        |tree, threads, shard_size: usize| {
            thread::scope(|scope| {
                for id in 0..threads {
                    scope.spawn(move |_| {
                        let thread_id = id;
                        let base = shard_size * thread_id;
                        let mut keys: Vec<usize> = (base..base + shard_size).collect();
                        keys.shuffle(&mut thread_rng());
                        for i in keys {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            let value = thread_rng().gen::<usize>();
                            if thread_rng().gen_bool(0.5) {
                                tree.insert(key.clone(), value, &guard);
                            } else {
                                tree.upsert(key.clone(), value, &guard);
                            }
                        }
                    });

                    scope.spawn(move |_| {
                        let thread_id = id;
                        let base = shard_size * thread_id;
                        let mut keys: Vec<usize> = (base..base + shard_size).collect();
                        keys.shuffle(&mut thread_rng());
                        for i in keys {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            tree.delete(&key, &guard);
                        }
                    });

                    scope.spawn(move |_| {
                        let thread_id = id;
                        let base = shard_size * thread_id;
                        let mut keys: Vec<usize> = (base..base + shard_size).collect();
                        keys.shuffle(&mut thread_rng());
                        for i in keys {
                            let guard = crossbeam_epoch::pin();
                            let key = i.to_string();
                            tree.get(&key, &guard);
                        }
                    });

                    scope.spawn(move |_| {
                        let thread_id = id;
                        let base = shard_size * thread_id;
                        let keys: Vec<usize> = (base..base + shard_size).collect();
                        for _ in 0..keys.len() {
                            let guard = crossbeam_epoch::pin();
                            let i1 = thread_rng().gen_range(0..keys.len());
                            let i2 = thread_rng().gen_range(0..keys.len());
                            let start = keys.get(std::cmp::min(i1, i2)).unwrap().to_string();
                            let end = keys.get(std::cmp::max(i1, i2)).unwrap().to_string();
                            tree.range(start..end, &guard);
                        }
                    });
                }
            })
            .unwrap();
        },
    );
}
