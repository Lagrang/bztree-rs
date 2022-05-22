use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::option::Option::Some;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::prelude::*;
use rand::{thread_rng, Rng};

use bztree::BzTree;
use history_verifier::{History, Ops};

mod history_verifier;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn test_insertion<K, V>(mut elements: Vec<(K, V)>, node_size: usize, use_upsert: bool)
where
    K: Clone + Ord + Hash + Debug,
    V: Send + Sync + Clone + Ord + Debug,
{
    let tree = BzTree::with_node_size(node_size as u16);
    for (key, val) in &elements {
        let guard = crossbeam_epoch::pin();
        if use_upsert {
            tree.upsert(key.clone(), val.clone(), &guard);
        } else {
            tree.insert(key.clone(), val.clone(), &guard);
        }
    }
    elements.sort_by_key(|(k, _)| k.clone());

    let guard = crossbeam_epoch::pin();
    let scanned: Vec<(&K, &V)> = tree.iter(&guard).collect();
    assert_eq!(scanned.len(), elements.len());
    for (i, (k, v)) in scanned.iter().enumerate() {
        assert_eq!(*k, &elements[i].0);
        assert_eq!(*v, &elements[i].1);
    }

    drop(tree);
}

fn fill(tree: &BzTree<Key<usize>, String>, size: usize) -> Vec<(Key<usize>, String)> {
    let mut expected_items: Vec<(Key<usize>, String)> = Vec::new();
    for key in 0..size {
        let key = Key::new(key);
        let val = thread_rng().gen::<usize>().to_string();
        let guard = crossbeam_epoch::pin();
        tree.insert(key.clone(), val.clone(), &guard);
        if thread_rng().gen_bool(0.3) {
            let val = thread_rng().gen::<usize>().to_string();
            tree.upsert(key.clone(), val.clone(), &guard);
            expected_items.push((key, val));
        } else {
            expected_items.push((key, val));
        }
    }
    expected_items
}

//TODO: add test which will fill the tree and then pop all elements
//TODO: add test which will fill the tree and then remove all elements
//TODO: add test which will mix all operations

#[test]
fn sequential_key_insert() {
    let node_size: usize = thread_rng().gen_range(50..100);
    println!("Node size: {:?}", node_size);
    let levels: u32 = 3;
    let tree_size = node_size.pow(levels) as usize;
    let mut elems = Vec::new();
    for i in 0..tree_size {
        elems.push((
            Key::new(i.to_string()),
            thread_rng().gen::<usize>().to_string(),
        ));
    }
    test_insertion(elems, node_size, false);
}

#[test]
fn random_key_insert() {
    let node_size: usize = thread_rng().gen_range(50..100);
    println!("Node size: {:?}", node_size);
    let levels: u32 = 3;
    let tree_size = node_size.pow(levels) as usize;
    let mut elems = Vec::new();
    for i in 0..tree_size {
        elems.push((
            Key::new(i.to_string()),
            thread_rng().gen::<usize>().to_string(),
        ));
    }
    elems.shuffle(&mut thread_rng());
    test_insertion(elems, node_size, false);
}

/// Test create minimum size node to check split of all types
/// of nodes(root, leaf and interim)
#[test]
fn min_sized_node_inserts() {
    for node_size in 2..5usize {
        let levels: u32 = 3;
        let tree_size = node_size.pow(levels) as usize;
        let mut elems = Vec::new();
        for i in 0..tree_size {
            elems.push((
                Key::new(i.to_string()),
                thread_rng().gen::<usize>().to_string(),
            ));
        }
        test_insertion(elems, node_size, false);
    }
}

#[test]
fn upsert() {
    let node_size: usize = thread_rng().gen_range(50..100);
    println!("Node size: {:?}", node_size);

    let levels: u32 = 3;
    let tree_size = node_size.pow(levels) as usize;
    let mut items = HashMap::new();
    for i in 0..tree_size {
        items.insert(
            Key::new((i % node_size).to_string()),
            thread_rng().gen::<usize>().to_string(),
        );
    }
    test_insertion(items.drain().collect(), node_size, true);
}

/// This test case reveals bug in split implementation
/// which is not respect case when node split try to insert
/// into parent node 2 nodes one of which have same key as
/// other element in parent node.
#[test]
fn upsert_same_element() {
    let node_size: usize = thread_rng().gen_range(50..100);
    println!("Node size: {:?}", node_size);
    let levels = 3;
    let tree_size = node_size.pow(levels);
    let tree_items: Vec<usize> = std::iter::repeat(1).take(tree_size).collect();
    let tree = BzTree::with_node_size(node_size as u16);
    for i in &tree_items {
        let guard = crossbeam_epoch::pin();
        tree.upsert(*i, *i, &guard);
    }

    let guard = crossbeam_epoch::pin();
    let vec: Vec<usize> = tree.iter(&guard).map(|(_, v)| *v).collect();
    assert_eq!(vec.len(), 1);
    assert_eq!(vec[0], 1);
}

#[test]
fn combined_inserts_and_upserts() {
    let node_size: usize = thread_rng().gen_range(50..100);
    println!("Node size: {:?}", node_size);
    let mut expected: Vec<(Key<usize>, String)> = Vec::with_capacity(node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    let tree_levels = 3;

    for i in 0..node_size.pow(tree_levels) {
        let guard = crossbeam_epoch::pin();
        tree.insert(Key::new(i), i.to_string(), &guard);
        tree.upsert(Key::new(i), (i + 1).to_string(), &guard);
        expected.push((Key::new(i), (i + 1).to_string()));
    }

    let guard = crossbeam_epoch::pin();
    let tree_elems: Vec<(&Key<usize>, &String)> = tree.iter(&guard).collect();
    assert_eq!(tree_elems.len(), expected.len());
    for (k, v) in tree_elems {
        let val: usize = *k.borrow();
        assert_eq!((val + 1).to_string(), *v);
    }
}

#[test]
fn deletes_starting_from_tree_start() {
    let node_size: usize = thread_rng().gen_range(2..100);
    let tree_levels = 3;
    let size = node_size.pow(tree_levels);
    println!("Node size: {:?}", node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    for _ in 0..2 {
        let expected_items = fill(&tree, size);
        for (key, value) in &expected_items {
            let guard = &crossbeam_epoch::pin();
            let key: &usize = key.borrow();
            assert_eq!(
                tree.delete(key, guard)
                    .expect("Inserted element cannot be deleted because not found"),
                value
            );
        }
        let guard = &crossbeam_epoch::pin();
        assert!(tree.iter(guard).next().is_none(), "Tree not empty");
    }
}

#[test]
fn deletes_starting_from_tree_end() {
    let node_size: usize = thread_rng().gen_range(2..100);
    let tree_levels = 3;
    let size = node_size.pow(tree_levels);
    println!("Node size: {:?}", node_size);
    let mut tree = BzTree::with_node_size(node_size as u16);
    for _ in 0..2 {
        let expected_items = fill(&mut tree, size);
        for (key, value) in expected_items.iter().rev() {
            let key: &usize = key.borrow();
            let guard = &crossbeam_epoch::pin();
            assert_eq!(
                tree.delete(key, guard)
                    .expect("Inserted element cannot be deleted because not found"),
                value
            );
        }
        let guard = &crossbeam_epoch::pin();
        assert!(tree.iter(guard).next().is_none(), "Tree not empty");
    }
}

#[test]
fn deletes_at_random_positions() {
    let node_size: usize = thread_rng().gen_range(2..100);
    println!("Node size: {:?}", node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    let size = node_size.pow(3);
    for _ in 0..2 {
        let mut expected_items = fill(&tree, size);
        expected_items.shuffle(&mut thread_rng());
        for (key, value) in &expected_items {
            let key: &usize = key.borrow();
            let guard = &crossbeam_epoch::pin();
            assert_eq!(
                tree.delete(key, guard)
                    .expect("Inserted element cannot be deleted because not found"),
                value
            );
        }
        let guard = &crossbeam_epoch::pin();
        assert!(tree.iter(guard).next().is_none(), "Tree not empty");
    }
}

#[test]
fn range() {
    let node_size: usize = thread_rng().gen_range(2..30);
    println!("Node size: {:?}", node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    let tree_levels = 3;

    let mut expected_keys: Vec<Key<usize>> = Vec::new();
    let mut expected_items: HashMap<Key<usize>, String> = HashMap::new();
    for key in 0..node_size.pow(tree_levels) {
        let guard = &crossbeam_epoch::pin();
        let key = Key::new(key);
        let val = thread_rng().gen::<usize>();
        tree.insert(key.clone(), val.to_string(), guard);
        expected_items.insert(key.clone(), val.to_string());
        expected_keys.push(key.clone());
        if thread_rng().gen_bool(0.5) {
            let i: usize = *key.borrow();
            let k: usize = thread_rng().gen_range(0..i + 1);
            let key = Key::new(k);
            let new_val = thread_rng().gen::<usize>();
            tree.upsert(key.clone(), new_val.to_string(), guard);
            expected_items.insert(key, new_val.to_string());
        }
        if thread_rng().gen_bool(0.35) {
            let key = expected_keys.choose(&mut thread_rng()).unwrap();
            let key: &usize = key.borrow();
            if tree.delete(key, guard).is_some() {
                expected_items.remove(key);
            }
        }

        // check range scan after insert/delete
        if thread_rng().gen_bool(0.05) {
            let i1 = thread_rng().gen_range(0..expected_keys.len());
            let i2 = thread_rng().gen_range(0..expected_keys.len());
            let start_key = &expected_keys[std::cmp::min(i1, i2)];
            let end_key = &expected_keys[std::cmp::max(i1, i2)];
            let kvs: HashMap<&Key<usize>, &String> =
                tree.range(start_key..end_key, &guard).collect();

            let mut expected_size = 0;
            for k in &expected_keys[std::cmp::min(i1, i2)..std::cmp::max(i1, i2)] {
                let key: &usize = k.borrow();
                // key can be removed, skip it
                if let Some(val) = expected_items.get(key) {
                    expected_size += 1;
                    assert_eq!(&val, kvs.get(k).unwrap());
                }
            }
            assert_eq!(kvs.len(), expected_size);
        } else if thread_rng().gen_bool(0.05) {
            let i = thread_rng().gen_range(0..expected_keys.len());
            let start_key = &expected_keys[i];
            let kvs: HashMap<&Key<usize>, &String> = tree.range(start_key.., &guard).collect();

            let mut expected_size = 0;
            for k in &expected_keys[i..] {
                let key: &usize = k.borrow();
                if let Some(val) = expected_items.get(key) {
                    expected_size += 1;
                    assert_eq!(&val, kvs.get(k).unwrap());
                }
            }
            assert_eq!(kvs.len(), expected_size);
        } else if thread_rng().gen_bool(0.05) {
            let i = thread_rng().gen_range(0..expected_keys.len());
            let end_key = &expected_keys[i];
            let kvs: HashMap<&Key<usize>, &String> = tree.range(..=end_key, &guard).collect();
            let mut expected_size = 0;
            for k in &expected_keys[..=i] {
                let key: &usize = k.borrow();
                if let Some(val) = expected_items.get(key) {
                    expected_size += 1;
                    assert_eq!(&val, kvs.get(k).unwrap());
                }
            }
            assert_eq!(kvs.len(), expected_size);
        }
    }
}

#[test]
fn iter() {
    let node_size: usize = thread_rng().gen_range(2..30);
    println!("Node size: {:?}", node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    let tree_levels = 3;

    let mut expected_keys: Vec<usize> = Vec::new();
    let mut expected_items: HashMap<Key<usize>, String> = HashMap::new();
    let mut keys: Vec<usize> = (0..node_size.pow(tree_levels)).collect();
    keys.shuffle(&mut thread_rng());
    for key in &keys {
        expected_keys.push(*key);
        let key = Key::new(*key);
        let val = thread_rng().gen::<usize>();
        let guard = crossbeam_epoch::pin();
        if tree.insert(key.clone(), val.to_string(), &guard) {
            expected_items.insert(key.clone(), val.to_string());
        }

        if thread_rng().gen_bool(0.45) {
            let key = Key::new(*keys.choose(&mut thread_rng()).unwrap());
            let new_val = thread_rng().gen::<usize>();
            tree.upsert(key.clone(), new_val.to_string(), &guard);
            expected_items.insert(key, new_val.to_string());
        }
        if thread_rng().gen_bool(0.35) {
            let key = expected_keys.choose(&mut thread_rng()).unwrap();
            if tree.delete(key, &guard).is_some() {
                expected_items.remove(key);
            }
        }

        if thread_rng().gen_bool(0.05) {
            let kvs: HashMap<&Key<usize>, &String> = tree.iter(&guard).collect();
            assert_eq!(kvs.len(), expected_items.len());
            for (k, v) in expected_items.iter() {
                assert_eq!(&v, kvs.get(k).unwrap());
            }
        }
    }

    let guard = crossbeam_epoch::pin();
    let kvs: HashMap<&Key<usize>, &String> = tree.iter(&guard).collect();
    assert_eq!(kvs.len(), expected_items.len());
    for (k, v) in expected_items.iter() {
        assert_eq!(&v, kvs.get(k).unwrap());
    }
}

#[test]
fn mixed_scan() {
    let node_size: usize = thread_rng().gen_range(2..100);
    println!("Node size: {:?}", node_size);
    let tree = BzTree::with_node_size(node_size as u16);
    let mut expected_keys = Vec::new();
    let mut keys: Vec<usize> = (0..node_size.pow(3)).collect();
    keys.shuffle(&mut thread_rng());
    for key in keys {
        let guard = crossbeam_epoch::pin();
        tree.upsert(key, key, &guard);
        expected_keys.push(key);
    }

    let guard = crossbeam_epoch::pin();
    let mut iter = tree.iter(&guard).peekable();
    let mut next_expected_forward = iter.next().unwrap().0 + 1;
    let mut next_expected_reversed = iter.next_back().unwrap().0 - 1;
    let mut iterated = 2;
    while iter.peek().is_some() {
        if thread_rng().gen_bool(0.5) {
            if let Some(kv) = iter.next() {
                assert_eq!(*kv.0, next_expected_forward);
                next_expected_forward += 1;
                iterated += 1;
            }
        } else {
            if let Some(kv) = iter.next_back() {
                assert_eq!(*kv.0, next_expected_reversed);
                next_expected_reversed -= 1;
                iterated += 1;
            }
        }
    }

    assert_eq!(iterated, expected_keys.len());
}

#[test]
fn all_operations_combinations() {
    let size = 1000;
    let mut last_state = Ops::new();
    let tree: BzTree<Key<u64>, String> = BzTree::with_node_size(size);
    let mut greatest_key = Key::new(0);
    for _ in 0..=size {
        // try to create same keys to check upserts
        let key = Key::new(thread_rng().gen_range(0..size as u64 / 3));
        let value = thread_rng().gen::<u64>().to_string();
        // check modification operations
        let key_val: &u64 = key.borrow();
        let guard = crossbeam_epoch::pin();
        if thread_rng().gen_bool(0.65) {
            if tree.insert(key.clone(), value.clone(), &guard) {
                last_state.insert(key.clone(), value, Instant::now());
                greatest_key = greatest_key.max(key);
            } else {
                tree.upsert(key.clone(), value.clone(), &guard).unwrap();
                last_state.insert(key.clone(), value.clone(), Instant::now());
                greatest_key = greatest_key.max(key);
            }
        } else if tree.delete(key_val, &guard).is_some() {
            last_state.delete(key, Instant::now());
        }

        // check get/scan
        if thread_rng().gen_bool(0.5) {
            History::from(&last_state).run_check(|key| {
                let key_val: &u64 = key.borrow();
                tree.get(key_val, &guard)
            });
        } else {
            check_scanners(&tree, &last_state, greatest_key.clone());
        }
    }
}

#[test]
fn conditional_op_combinations() {
    let size = 1000;
    let mut last_state = Ops::new();
    let tree: BzTree<Key<u64>, u64> = BzTree::with_node_size(size);
    for i in 0..=size {
        let key = Key::new(i as u64);
        let value = thread_rng().gen::<u64>();
        let guard = crossbeam_epoch::pin();
        if tree.insert(key.clone(), value, &guard) {
            last_state.insert(key.clone(), value, Instant::now());
        } else {
            tree.upsert(key.clone(), value, &guard).unwrap();
            last_state.insert(key.clone(), value, Instant::now());
        }
    }

    let mut keys: Vec<Key<u64>> = tree
        .iter(unsafe { crossbeam_epoch::unprotected() })
        .map(|(k, _)| k.clone())
        .collect();
    keys.append(&mut keys.clone());
    keys.append(&mut keys.clone());
    keys.shuffle(&mut thread_rng());

    for key in keys {
        let guard = crossbeam_epoch::pin();
        if thread_rng().gen_bool(0.5) {
            tree.compute(
                key.borrow() as &u64,
                |(k, v)| {
                    last_state.insert(k.clone(), *v + 1, Instant::now());
                    Some(*v + 1)
                },
                &guard,
            );
        } else {
            tree.compute(
                key.borrow() as &u64,
                |(k, _)| {
                    last_state.delete(k.clone(), Instant::now());
                    None
                },
                &guard,
            );
        }
    }

    History::from(&last_state).run_check(|key| {
        let key_val: &u64 = key.borrow();
        tree.get(key_val, unsafe { crossbeam_epoch::unprotected() })
    });
}

#[test]
fn check_kv_drop() {
    let ref_cnt = AtomicUsize::new(0);

    let tree: BzTree<String, Droppable> = BzTree::with_node_size(70);
    let mut vec: Vec<usize> = (0..2000).collect();
    vec.shuffle(&mut thread_rng());
    let guard = unsafe { crossbeam_epoch::unprotected() };
    for i in &vec {
        if thread_rng().gen_bool(0.5) {
            tree.insert(i.to_string(), Droppable::new(&ref_cnt), &guard);
        } else {
            tree.upsert(i.to_string(), Droppable::new(&ref_cnt), &guard);
        }
    }

    vec.shuffle(&mut thread_rng());
    for i in vec {
        tree.delete(&i.to_string(), &guard);
    }

    assert_eq!(ref_cnt.load(Ordering::Relaxed), 0);
}

fn check_scanners(
    tree: &BzTree<Key<u64>, String>,
    last_state: &Ops<Key<u64>, String>,
    greatest_key: Key<u64>,
) {
    let guard = crossbeam_epoch::pin();
    let prob = 0.35;
    if thread_rng().gen_bool(prob) {
        History::from(&last_state).run_scanner_check(|| (.., false, Box::new(tree.iter(&guard))));
    } else if thread_rng().gen_bool(prob) {
        // right edge check
        History::from(&last_state).run_scanner_check(|| {
            let range = ..greatest_key.clone();
            (range.clone(), false, Box::new(tree.range(range, &guard)))
        });
        History::from(&last_state).run_scanner_check(|| {
            let range = ..=greatest_key.clone();
            (range.clone(), false, Box::new(tree.range(range, &guard)))
        });
        // right edge check with custom value
        let sign: i64 = if thread_rng().gen_bool(prob) {
            1
        } else if thread_rng().gen_bool(prob) {
            -1
        } else {
            0
        };
        let key_val: u64 = *greatest_key.borrow();
        let edge = if key_val == 0 {
            Key::new(0)
        } else {
            Key::new((key_val as i64 + sign) as u64)
        };
        History::from(&last_state).run_scanner_check(|| {
            let range = ..=edge.clone();
            (range.clone(), false, Box::new(tree.range(range, &guard)))
        });
        History::from(&last_state).run_scanner_check(|| {
            let range = ..edge.clone();
            (range.clone(), false, Box::new(tree.range(range, &guard)))
        });
    } else {
        // left edge check with custom value
        let key_val: u64 = *greatest_key.borrow();
        let edge = thread_rng().gen_range(0..key_val + 1);
        History::from(&last_state).run_scanner_check(|| {
            let range = Key::new(edge)..;
            (range.clone(), false, Box::new(tree.range(range, &guard)))
        });
        // left+right edge check with custom value
        let left_edge = thread_rng().gen_range(0..key_val + 1);
        let right_edge = thread_rng().gen_range(left_edge as u64..key_val + 1);
        if thread_rng().gen_bool(prob) {
            History::from(&last_state).run_scanner_check(|| {
                let range = Key::new(left_edge)..Key::new(right_edge);
                (range.clone(), false, Box::new(tree.range(range, &guard)))
            });
        } else {
            History::from(&last_state).run_scanner_check(|| {
                let range = Key::new(left_edge)..=Key::new(right_edge);
                (range.clone(), false, Box::new(tree.range(range, &guard)))
            });
        }
    }
}

#[derive(Debug)]
struct Droppable {
    val: usize,
    ref_cnt: *mut AtomicUsize,
}

unsafe impl Send for Droppable {}
unsafe impl Sync for Droppable {}

impl Droppable {
    fn new(ref_cnt: &AtomicUsize) -> Self {
        let val = ref_cnt.fetch_add(1, Ordering::AcqRel);
        Self {
            val: val + 1,
            ref_cnt: ref_cnt as *const AtomicUsize as *mut AtomicUsize,
        }
    }
}

impl Clone for Droppable {
    fn clone(&self) -> Self {
        unsafe { (*self.ref_cnt).fetch_add(1, Ordering::AcqRel) };
        Self {
            val: self.val,
            ref_cnt: self.ref_cnt,
        }
    }
}

impl Ord for Droppable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

impl PartialOrd for Droppable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.val.cmp(&other.val))
    }
}

impl Eq for Droppable {}
impl PartialEq for Droppable {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl Hash for Droppable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state)
    }
}

impl Drop for Droppable {
    fn drop(&mut self) {
        unsafe {
            (*self.ref_cnt).fetch_sub(1, Ordering::AcqRel);
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
struct Key<K> {
    inner: Box<K>,
}

impl<K> Borrow<K> for Key<K> {
    fn borrow(&self) -> &K {
        self.inner.borrow()
    }
}

impl<K> Key<K> {
    fn new(val: K) -> Self {
        Key {
            inner: Box::new(val),
        }
    }
}

impl<K: Debug> Display for Key<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?}", self.inner)
    }
}
