[![Crate](https://img.shields.io/crates/v/bztree.svg)](https://crates.io/crates/bztree)
[![API](https://docs.rs/bztree/badge.svg)](https://docs.rs/bztree)

# BzTree
BzTree(concurrent B-tree) implementation for Rust based on paper
[BzTree: A High-Performance Latch-free Range Index for Non-Volatile Memory](http://www.vldb.org/pvldb/vol11/p553-arulraj.pdf).  
Current implementation doesn't support non-volatile memory and supposed
to be used only as in-memory(not persistent) data structure.  
BzTree uses [MwCAS](https://crates.io/crates/mwcas) crate to get access
to multi-word CAS.

## Usage
```rust
/// use bztree::BzTree;
///
/// let mut tree = BzTree::with_node_size(2);
/// let guard = crossbeam_epoch::pin();
///
/// let key1 = "key_1".to_string();
/// assert!(tree.insert(key1.clone(), 1, &crossbeam_epoch::pin()));
/// assert!(!tree.insert(key1.clone(), 5, &crossbeam_epoch::pin()));
/// tree.upsert(key1.clone(), 10, &crossbeam_epoch::pin());
///
/// assert!(matches!(tree.delete(&key1, &guard), Some(&10)));
///
/// let key2 = "key_2".to_string();
/// tree.insert(key2.clone(), 2, &crossbeam_epoch::pin());
/// assert!(tree.compute(&key2, |(_, v)| Some(v + 1), &guard));
/// assert!(matches!(tree.get(&key2, &guard), Some(&3)));
///
/// assert!(tree.compute(&key2, |(_, v)| {
///     if *v == 3 {
///         None
///     } else {
///         Some(v + 1)
///     }
/// }, &guard));
/// assert!(matches!(tree.get(&key2, &guard), None));
```
