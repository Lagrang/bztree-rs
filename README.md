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
let mut tree = BzTree::with_node_size(2);
tree.insert("key_1".to_string(), "1", &crossbeam_epoch::pin());
```
