//! # BzTree
//! Concurrent B-tree implementation based on paper
//! [BzTree: A High-Performance Latch-free Range Index for Non-Volatile Memory](http://www.vldb.org/pvldb/vol11/p553-arulraj.pdf).
//! Current implementation doesn't support non-volatile memory and supposed to be used only as
//! in-memory(not persistent) data structure.
//! BzTree uses [MwCAS](https://crates.io/crates/mwcas) crate to get access to multi-word CAS.
//!
//! # Examples
//! ```
//! use bztree::BzTree;
//!
//! let mut tree = BzTree::with_node_size(2);
//! let guard = crossbeam_epoch::pin();
//!
//! let key1 = "key_1".to_string();
//! assert!(tree.insert(key1.clone(), 1, &crossbeam_epoch::pin()));
//! assert!(!tree.insert(key1.clone(), 5, &crossbeam_epoch::pin()));
//! tree.upsert(key1.clone(), 10, &crossbeam_epoch::pin());
//!
//! assert!(matches!(tree.delete(&key1, &guard), Some(&10)));
//!
//! let key2 = "key_2".to_string();
//! tree.insert(key2.clone(), 2, &crossbeam_epoch::pin());
//! assert!(tree.compute(&key2, |(_, v)| Some(v + 1), &guard));
//! assert!(matches!(tree.get(&key2, &guard), Some(&3)));
//!
//! assert!(tree.compute(&key2, |(_, v)| {
//!     if *v == 3 {
//!         None
//!     } else {
//!         Some(v + 1)
//!     }
//! }, &guard));
//! assert!(matches!(tree.get(&key2, &guard), None));
//! ```

extern crate core;

mod node;
mod scanner;

use crate::node::{DeleteError, InsertError, MergeMode, Node, NodeEdge, SplitMode};
use crate::scanner::Scanner;
use crate::NodePointer::{Interim, Leaf};
use crossbeam_epoch::Guard;
use mwcas::{HeapPointer, MwCas};
use node::status_word::StatusWord;
use std::borrow::Borrow;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::option::Option::Some;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// BzTree data structure.
///
/// # Key-value trait bounds
///
/// Keys should implement [Ord] and [Clone], values should be [Clone], [Send] and [Sync].
///
/// Because of nature of concurrent trees, node split/merge operations are optimistic and can fail:
/// implementation creates copy of nodes which is a part of split/merge process
/// and try to replace existing node by new one. At one point in time, 2 nodes can
/// points to the same key-value pair. To reduce count of unsafe code inside tree implementation,
/// key and values should implement [Clone].
///
/// # Visibility of changes
/// Changes made by `insert` and `delete` operations will be immediately visible through
/// `get`/`scan`
/// operations if `insert`/`delete` happens before `get`/`scan`.
/// `Scan/range` operation may see tree updates which made concurrently with scanner progress, e.g.
/// scanner doesn't provide snapshot view of tree.
///
/// # Memory reclamation
/// BzTree uses [crossbeam_epoch::Guard] memory reclamation to free memory of removed/replaced
/// key-values.
/// Because of internal representation of tree nodes, drop of some removed keys can be delayed
/// until node split/merge. This limitation caused by 'sorted' space inside tree node which uses
/// binary search to locate keys inside. Current implementation of binary search should have an
/// access to removed/replaced keys during search.
///
/// # Heap allocation
/// BzTree allocates memory for nodes on heap. Key and values in leaf nodes(e.g., nodes which
/// stores actual user data) stored directly in node memory without additional heap allocations.
/// Links to other nodes, stored in interim nodes, allocated on heap.
///
/// # Thread safety
/// BzTree can be safely shared between threads, e.g. it implements [Send] ans [Sync].
pub struct BzTree<K: Ord, V> {
    root: NodeLink<K, V>,
    node_size: usize,
}

type NodeLink<K, V> = HeapPointer<NodePointer<K, V>>;

unsafe impl<K: Ord, V> Send for BzTree<K, V> {}

unsafe impl<K: Ord, V> Sync for BzTree<K, V> {}

/// Leaf node of tree actually store KV pairs.
type LeafNode<K, V> = Node<K, V>;
/// Interim node store links to other nodes(leaf and interim).
/// Each cell in interim node points to tree node which contain
/// keys less or equal to cell key.
/// Interim node always contain special guard cell which represent
/// 'positive infinite' key. This cell used to store link to node
/// which keys greater than any other key in current interim node.
type InterimNode<K, V> = Node<Key<K>, NodeLink<K, V>>;

impl<K, V> BzTree<K, V>
where
    K: Clone + Ord,
    V: Clone + Send + Sync,
{
    /// Create new tree with default node size.
    pub fn new() -> BzTree<K, V> {
        Self::with_node_size(60)
    }

    /// Create new tree with passed node size.
    pub fn with_node_size(node_size: u16) -> BzTree<K, V> {
        assert!(node_size > 1, "Max node elements should be > 1");
        let root = Node::with_capacity(node_size);
        BzTree {
            root: HeapPointer::new(NodePointer::new_leaf(root)),
            node_size: node_size as usize,
        }
    }

    /// Insert key-value pair to tree if no elements with same key already exists.
    /// # Return
    /// Returns true if key-value pair successfully inserted, otherwise false if key already in
    /// tree.
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// let key = "key".to_string();
    /// assert!(tree.insert(key.clone(), 1, &guard));
    /// assert!(!tree.insert(key.clone(), 5, &guard));
    /// ```
    pub fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let mut value: V = value;
        let search_key = Key::new(key.clone());
        loop {
            let node = self.find_leaf_for_key(&search_key, true, guard).unwrap();
            match node.insert(key.clone(), value, guard) {
                Ok(_) => {
                    return true;
                }
                Err(InsertError::Split(val)) => {
                    value = val;
                    // try to find path to overflowed node
                    let leaf_ptr = node as *const LeafNode<K, V>;
                    let path = self.find_path_to_key(&search_key, false, guard);
                    if let Some(path) = path {
                        if let Leaf(found_leaf) = path.node_pointer {
                            // if overflowed node is not split/merged by other thread during insert
                            if ptr::eq(leaf_ptr, found_leaf.deref()) {
                                self.split_leaf(path, guard);
                            }
                        }
                    }
                }
                Err(InsertError::DuplicateKey) => {
                    return false;
                }
                Err(InsertError::Retry(val)) | Err(InsertError::NodeFrozen(val)) => {
                    value = val;
                }
            }
        }
    }

    /// Insert key-value pair if not already exists, otherwise replace value of existing element.
    /// # Return
    /// Returns value previously associated with same key or `None`.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// let key = "key".to_string();
    /// assert!(matches!(tree.upsert(key.clone(), 10, &guard), None));
    /// assert!(matches!(tree.upsert(key.clone(), 11, &guard), Some(10)));
    /// ```
    pub fn upsert<'g>(&'g self, key: K, value: V, guard: &'g Guard) -> Option<&'g V> {
        let mut value: V = value;
        let search_key = Key::new(key.clone());
        loop {
            // use raw pointer to overcome borrowing rules in loop
            // which borrows value even on return statement
            let node = self.find_leaf_for_key(&search_key, true, guard).unwrap();
            match node.upsert(key.clone(), value, guard) {
                Ok(prev_val) => {
                    return prev_val;
                }
                Err(InsertError::Split(val)) => {
                    value = val;
                    // try to find path to overflowed node
                    let path = self.find_path_to_key(&search_key, false, guard);
                    if let Some(path) = path {
                        if let Leaf(found_leaf) = path.node_pointer {
                            // if overflowed node is not split/merged by other thread
                            if ptr::eq(node, found_leaf.deref()) {
                                self.split_leaf(path, guard);
                            }
                        }
                    }
                }
                Err(InsertError::DuplicateKey) => {
                    panic!("Duplicate key error reported on upsert")
                }
                Err(InsertError::Retry(val)) | Err(InsertError::NodeFrozen(val)) => {
                    value = val;
                }
            };
        }
    }

    /// Delete value associated with key.
    /// # Return
    /// Returns removed value if key found in tree.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// let key = "key".to_string();
    /// assert!(tree.insert(key.clone(), 10, &guard));
    /// assert!(matches!(tree.delete(&key, &guard), Some(&10)));
    /// ```
    pub fn delete<'g, Q>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Ord + Clone,
    {
        let search_key = Key::new(key.clone());
        loop {
            // use raw pointer to overcome borrowing rules in loop
            // which borrows value even on return statement
            if let Some(node) = self.find_leaf_for_key(&search_key, false, guard) {
                let len = node.estimated_len(guard);
                match node.delete(key.borrow(), guard) {
                    Ok(val) => {
                        if self.should_merge(len - 1) {
                            self.merge_recursive(&search_key, guard);
                        }
                        return Some(val);
                    }
                    Err(DeleteError::KeyNotFound) => return None,
                    Err(DeleteError::Retry) => {}
                }
            } else {
                return None;
            }
        }
    }

    /// Get value associated with key.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// let key = "key".to_string();
    /// assert!(tree.insert(key.clone(), 10, &guard));
    /// assert!(matches!(tree.get(&key, &guard), Some(&10)));
    /// ```
    pub fn get<'g, Q>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Clone + Ord,
    {
        let search_key = Key::new(key.clone());
        self.find_leaf_for_key(&search_key, false, guard)
            .and_then(|node| node.get(key, guard).map(|(_, val, _, _)| val))
    }

    /// Create tree range scanner which will return values whose keys is in passed range.
    ///
    /// Visibility of changes made in tree during scan, described in [BzTree] doc.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// let mut scanner = tree.range("key1".to_string()..="key2".to_string(), &guard);
    /// assert!(matches!(scanner.next(), Some((_, &1))));
    /// assert!(matches!(scanner.next(), Some((_, &2))));
    /// assert!(matches!(scanner.next(), None));
    /// ```
    pub fn range<'g>(
        &'g self,
        // TODO: think how we can accept Q instead of K in range
        key_range: impl RangeBounds<K> + 'g + Clone,
        guard: &'g Guard,
    ) -> impl DoubleEndedIterator<Item = (&'g K, &'g V)> + 'g {
        return match self.root.read(guard) {
            Leaf(root) => Scanner::from_leaf_root(root, key_range, guard),
            Interim(root) => Scanner::from_non_leaf_root(root, key_range, guard),
        };
    }

    /// Create iterator through all key-values of tree.
    ///
    /// Iterator based on tree range scanner and have same changes visibility guarantees.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// let mut scanner = tree.iter(&guard);
    /// assert!(matches!(scanner.next(), Some((_, &1))));
    /// assert!(matches!(scanner.next(), Some((_, &2))));
    /// assert!(matches!(scanner.next(), Some((_, &3))));
    /// assert!(matches!(scanner.next(), None));
    /// ```
    pub fn iter<'g>(&'g self, guard: &'g Guard) -> impl DoubleEndedIterator<Item = (&'g K, &'g V)> {
        self.range(.., guard)
    }

    /// Return first element of tree according to key ordering.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// assert!(matches!(tree.first(&guard), Some((_, &1))));
    /// ```
    #[inline]
    pub fn first<'g>(&'g self, guard: &'g Guard) -> Option<(&'g K, &'g V)> {
        self.iter(guard).next()
    }

    /// Return last element of tree according to key ordering.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// assert!(matches!(tree.last(&guard), Some((_, &3))));
    /// ```
    #[inline]
    pub fn last<'g>(&'g self, guard: &'g Guard) -> Option<(&'g K, &'g V)> {
        self.iter(guard).rev().next()
    }

    /// Remove and return first element of tree according to key ordering.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// assert!(matches!(tree.pop_first(&guard), Some((_, &1))));
    /// assert!(matches!(tree.pop_first(&guard), Some((_, &2))));
    /// assert!(matches!(tree.pop_first(&guard), Some((_, &3))));
    /// assert!(matches!(tree.pop_first(&guard), None));
    /// ```
    pub fn pop_first<'g>(&'g self, guard: &'g Guard) -> Option<(K, &'g V)> {
        self.pop(NodeEdge::Left, guard)
    }

    /// Remove and return last element of tree according to key ordering.
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// tree.insert("key1".to_string(), 1, &guard);
    /// tree.insert("key2".to_string(), 2, &guard);
    /// tree.insert("key3".to_string(), 3, &guard);
    ///
    /// assert!(matches!(tree.pop_last(&guard), Some((_, &3))));
    /// assert!(matches!(tree.pop_last(&guard), Some((_, &2))));
    /// assert!(matches!(tree.pop_last(&guard), Some((_, &1))));
    /// assert!(matches!(tree.pop_last(&guard), None));
    /// ```
    pub fn pop_last<'g>(&'g self, guard: &'g Guard) -> Option<(K, &'g V)> {
        self.pop(NodeEdge::Right, guard)
    }

    fn pop<'g>(&'g self, pop_from: NodeEdge, guard: &'g Guard) -> Option<(K, &'g V)> {
        loop {
            let (edge_node, edge_key) = self.find_edge_node(pop_from, guard);
            let node_status = edge_node.status_word().read(guard);
            if node_status.is_frozen() {
                // found node already removed from tree, restart
                continue;
            }

            if let Some(edge_entry) = edge_node.edge_kv(pop_from, guard) {
                // It's enough to only validate status word of leaf node to ensure that we pop
                // current max element. While pop in progress, other threads can change tree
                // shape by merge and split.
                // If node which currently contains min/max element is merged/split, we will
                // detect such change by checking node's status word.
                // Merging of parent node also can proceed without additional validation
                // because merge of parent do not change ordering of leaf elements.
                // Split of parent node have 2 cases:
                // 1. split caused by overflow of other leaf node(e.g., not current min/max node).
                // In this case min or max node still contain min/max value and can be used for
                // pop operation.
                // 2. split caused by overflow of current min/max node.
                // This case covered by status word validation.
                let len = edge_node.estimated_len(guard);
                match edge_node.conditional_delete(node_status, edge_entry.location, guard) {
                    Ok(_) => {
                        if self.should_merge(len - 1) {
                            if let Some(k) = edge_key {
                                self.merge_recursive(k, guard);
                            }
                        }
                        return Some((edge_entry.key.clone(), edge_entry.value));
                    }
                    Err(DeleteError::Retry) | Err(DeleteError::KeyNotFound) => {}
                }
            } else {
                if self.root.read(guard).points_to_leaf(edge_node) {
                    // current max/min node is empty and it is the root of tree
                    return None;
                }
                // node with max possible key is empty,
                // we should help to merge such node and try again
                if let Some(k) = edge_key {
                    self.merge_recursive(k, guard);
                }
            }
        }
    }

    /// Update or delete element with passed key using conditional logic.
    ///
    /// Function `F` accepts current value of key and based on it, produces new value.
    /// If `F` returns `Some(V)` then current value will be updated. Otherwise(`None`) key will
    /// be removed from tree.
    ///
    /// Function `F` can be called several times in case of concurrent modification of tree.
    /// Because of this behaviour, function code should not modify global application state or
    /// such code should be carefully designed(understanding consequences of repeated function
    /// calls for same key).
    ///
    /// # Examples
    /// ```
    /// use bztree::BzTree;
    ///
    /// let tree = BzTree::new();
    /// let guard = crossbeam_epoch::pin();
    ///
    /// let key = "key".to_string();
    /// tree.insert(key.clone(), 2, &guard);
    ///
    /// assert!(tree.compute(&key, |(_, v)| Some(v + 1), &guard));
    /// assert!(matches!(tree.get(&key, &guard), Some(&3)));
    ///
    /// assert!(tree.compute(&key, |(_, v)| {
    ///  if *v == 3 {
    ///      None
    ///  } else {
    ///      Some(v + 1)
    ///  }
    /// }, &guard));
    /// assert!(matches!(tree.get(&key, &guard), None));
    /// ```
    pub fn compute<'g, Q, F>(&'g self, key: &Q, mut new_val: F, guard: &'g Guard) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + Clone,
        F: FnMut((&K, &V)) -> Option<V>,
    {
        let search_key = Key::new(key.clone());
        loop {
            let node = self.find_leaf_for_key(&search_key, false, guard);
            if node.is_none() {
                return false;
            }

            let node = node.unwrap();
            if let Some((found_key, val, status_word, value_index)) = node.get(key, guard) {
                // compute new value for key
                if let Some(new_val) = new_val((found_key, val)) {
                    match node.conditional_upsert(found_key.clone(), new_val, status_word, guard) {
                        Ok(_) => {
                            return true;
                        }
                        Err(InsertError::Split(_)) => {
                            // try to find path to overflowed node
                            if let Some(path) = self.find_path_to_key(&search_key, false, guard) {
                                if let Leaf(found_leaf) = path.node_pointer {
                                    // if overflowed node is not split/merged by other thread
                                    if ptr::eq(node, found_leaf.deref()) {
                                        self.split_leaf(path, guard);
                                    }
                                }
                            }
                        }
                        Err(InsertError::DuplicateKey) => {
                            panic!("Duplicate key error reported on upsert")
                        }
                        Err(InsertError::Retry(_)) | Err(InsertError::NodeFrozen(_)) => {}
                    };
                } else {
                    // no new value exists for key, caller want to remove KV
                    let len = node.estimated_len(guard);
                    match node.conditional_delete(status_word, value_index, guard) {
                        Ok(_) => {
                            if self.should_merge(len - 1) {
                                self.merge_recursive(&search_key, guard);
                            }
                            return true;
                        }
                        Err(DeleteError::Retry) | Err(DeleteError::KeyNotFound) => {}
                    }
                }
            } else {
                // TODO: add possibility to insert KV if not exists
                return false;
            }
        }
    }

    #[inline(always)]
    fn should_merge(&self, node_size: usize) -> bool {
        node_size <= self.node_size / 3
    }

    fn merge_recursive<Q>(&self, key: &Key<Q>, guard: &Guard)
    where
        K: Borrow<Q>,
        Q: Ord + Clone,
    {
        loop {
            if let Some(path) = self.find_path_to_key(key, false, guard) {
                match self.merge_node(path, guard) {
                    MergeResult::Completed => break,
                    MergeResult::Retry => {}
                    MergeResult::RecursiveMerge(path) => {
                        // repeat until we try recursively merge parent nodes(until we find root
                        // or some parent cannot be merged).
                        let mut merge_path = path;
                        while let MergeResult::RecursiveMerge(path) =
                            self.merge_node(merge_path, guard)
                        {
                            merge_path = path;
                        }
                    }
                }
            }
        }
    }

    fn merge_node<'g>(
        &'g self,
        mut path: TraversePath<'g, K, V>,
        guard: &'g Guard,
    ) -> MergeResult<'g, K, V> {
        // obtain direct parent of merging node
        let parent = path.parents.pop();
        if parent.is_none() {
            return self.merge_root(path.node_pointer, guard);
        }

        // freeze underflow node to ensure that no one can modify it
        if !path.node_pointer.try_froze() {
            return MergeResult::Retry;
        }

        match path.node_pointer {
            Leaf(node) => {
                // merge is not required anymore, node received new KVs
                // while we tried to merge it
                if !self.should_merge(node.estimated_len(guard)) {
                    path.node_pointer.try_unfroze();
                    return MergeResult::Completed;
                }
            }
            Interim(_) => {
                // this is explicit request to merge interim node,
                // always proceed with merge
            }
        };

        // merging of underflow node with sibling will return 'new parent' node.
        // This 'new parent' doesn't contain underutilized node and should be
        // installed in grandparent node(by replacing 'current parent' node pointer with pointer
        // to 'new parent' node).
        let parent = parent.unwrap();
        let parent_status = parent.node().status_word().read(guard);
        if parent_status.is_frozen() {
            path.node_pointer.try_unfroze();
            return MergeResult::Retry;
        }

        // if no siblings in parent node
        if parent.node().estimated_len(guard) <= 1 {
            path.node_pointer.try_unfroze();
            return if path.node_pointer.len(guard) == 0 {
                // underflow node is empty and parent doesn't contain any other nodes,
                // we can remove empty parent node from the tree
                self.remove_empty_parent(
                    TraversePath {
                        parents: path.parents,
                        node_pointer: parent.node_pointer,
                        cas_pointer: parent.cas_pointer,
                    },
                    guard,
                )
            } else {
                // underflow node cannot be merged with siblings, try to merge parent with it
                // sibling on their level. This give a chance later to merge current underflow node
                // with sibling node from another parent.
                MergeResult::RecursiveMerge(TraversePath {
                    cas_pointer: parent.cas_pointer,
                    node_pointer: parent.node_pointer,
                    parents: path.parents,
                })
            };
        }

        let mut mwcas = MwCas::new();
        // merge underflow node with one of its siblings
        let merge_res =
            self.merge_with_sibling(path.node_pointer, &parent, parent_status, &mut mwcas, guard);
        let new_parent = match merge_res {
            Ok(new_parent) => new_parent,
            Err(e) => {
                path.node_pointer.try_unfroze();
                return e;
            }
        };

        // check that parent node is also underflow and should be merged on it's own level
        let new_parent_should_merge = self.should_merge(new_parent.estimated_len(guard));

        mwcas.compare_exchange(
            parent.node().status_word(),
            parent_status,
            parent_status.froze(),
        );
        mwcas.compare_exchange(
            parent.cas_pointer,
            parent.node_pointer,
            NodePointer::new_interim(new_parent),
        );
        // if parent node has grandparent node, then check that grandparent not frozen.
        // if parent node has no grandparent node, then parent is current root, we can simply CAS
        // on root pointer.
        if let Some(grand_parent) = path.parents.last() {
            let status_word = grand_parent.node().status_word().read(guard);
            if status_word.is_frozen() {
                // grandparent merged/split in progress, rollback changes and retry
                // when parent node will be moved to new grandparent
                // or this grandparent will be unfrozen
                path.node_pointer.try_unfroze();
                return MergeResult::Retry;
            }
            // check that grandparent is not frozen and increase it version
            // (because we make in-place update of link to parent node)
            mwcas.compare_exchange(
                grand_parent.node().status_word(),
                status_word,
                status_word.inc_version(),
            );
        }

        // merge prepared, try to install new nodes instead of underutilized
        if mwcas.exec(guard) {
            // we successfully merge node, now check is it parent become underutilized.
            // we try to merge parent only after original node merged, this will provide
            // bottom-up merge until we reach root.
            if new_parent_should_merge {
                let new_parent_ptr = parent.cas_pointer.read(guard);
                return MergeResult::RecursiveMerge(TraversePath {
                    cas_pointer: parent.cas_pointer,
                    node_pointer: new_parent_ptr,
                    parents: path.parents,
                });
            }
            MergeResult::Completed
        } else {
            path.node_pointer.try_unfroze();
            MergeResult::Retry
        }
    }

    /// Method will remove parent node from grandparent.
    /// Method suppose that underutilized parent contains only 1 link to child node and this
    /// child is empty.
    /// We create copy of grandparent node without link to such underutilized parent and install
    /// new grandparent inside tree.
    fn remove_empty_parent(
        &self,
        mut empty_node_path: TraversePath<K, V>,
        guard: &Guard,
    ) -> MergeResult<K, V> {
        let mut mwcas = MwCas::new();
        let node_status = empty_node_path.node_pointer.status_word().read(guard);
        if node_status.is_frozen() {
            return MergeResult::Retry;
        }
        mwcas.compare_exchange(
            empty_node_path.node_pointer.status_word(),
            node_status,
            node_status.froze(),
        );

        if let Some(parent_handle) = empty_node_path.parents.pop() {
            let parent_node = parent_handle.node_pointer.to_interim_node();
            let parent_status = parent_node.status_word().read(guard);
            if parent_status.is_frozen() {
                return MergeResult::Retry;
            }
            // Parent node contains 1 node which should be removed because it empty.
            // We should recursively process this parent node because logically it empty. Then we
            // found some grandparent node which is not empty, we remove all empty nodes from it.
            // This will release all empty nodes found below it.
            if parent_node.estimated_len(guard) == 1 {
                return self.remove_empty_parent(
                    TraversePath {
                        parents: empty_node_path.parents,
                        cas_pointer: parent_handle.cas_pointer,
                        node_pointer: parent_handle.node_pointer,
                    },
                    guard,
                );
            }

            mwcas.compare_exchange(
                parent_node.status_word(),
                parent_status,
                parent_status.froze(),
            );

            // if parent node has grandparent node, then check that grandparent not frozen.
            if let Some(grand_parent) = empty_node_path.parents.last() {
                let status_word = grand_parent.node().status_word().read(guard);
                if status_word.is_frozen() {
                    // grandparent merged/split in progress, rollback changes and retry
                    // when parent node will be moved to new grandparent
                    // or this grandparent will be unfrozen
                    return MergeResult::Retry;
                }

                mwcas.compare_exchange(
                    grand_parent.node().status_word(),
                    status_word,
                    status_word.inc_version(),
                );
            }

            // create copy of parent node without link to empty child
            let new_node = InterimNode::new_readonly(
                parent_node
                    .iter(guard)
                    .filter_map(|(k, v)| {
                        if k != &parent_handle.child_key {
                            Some((k.clone(), v.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            );

            mwcas.compare_exchange(
                parent_handle.cas_pointer,
                parent_handle.node_pointer,
                NodePointer::new_interim(new_node),
            );
        } else {
            // Parent node is root and it contains only 1 link to empty child node, replace root
            // node by empty leaf node. Parent node already frozen at this moment, simply replace
            // old parent by new root node.
            mwcas.compare_exchange(
                empty_node_path.cas_pointer,
                empty_node_path.node_pointer,
                NodePointer::new_leaf(LeafNode::with_capacity(self.node_size as u16)),
            );
        }

        if mwcas.exec(guard) {
            MergeResult::Completed
        } else {
            MergeResult::Retry
        }
    }

    fn merge_root(&self, root_ptr: &NodePointer<K, V>, guard: &Guard) -> MergeResult<K, V> {
        let mut cur_root = root_ptr;
        loop {
            if let Interim(root) = cur_root {
                let root_status = root.status_word().read(guard);
                // if root contains only 1 node, move this node up to root level
                if !root_status.is_frozen() && root.estimated_len(guard) == 1 {
                    let (_, child_ptr) = root.iter(guard).next().unwrap();
                    let child_node_pointer = child_ptr.read(guard);
                    let child_status = child_node_pointer.status_word().read(guard);
                    let mut mwcas = MwCas::new();
                    mwcas.compare_exchange(root.status_word(), root_status, root_status.froze());
                    mwcas.compare_exchange(
                        child_node_pointer.status_word(),
                        child_status,
                        child_status.clone(),
                    );
                    mwcas.compare_exchange(&self.root, cur_root, child_node_pointer.clone());
                    if !child_status.is_frozen() && mwcas.exec(guard) {
                        cur_root = self.root.read(guard);
                        continue;
                    }
                }
            }
            break;
        }
        MergeResult::Completed
    }

    /// Merge node with one of it's sibling if possible.
    fn merge_with_sibling<'g>(
        &self,
        node: &'g NodePointer<K, V>,
        parent: &'g Parent<'g, K, V>,
        parent_status: &StatusWord,
        mwcas: &mut MwCas<'g>,
        guard: &'g Guard,
    ) -> Result<InterimNode<K, V>, MergeResult<K, V>> {
        let mut siblings: Vec<(&Key<K>, &NodePointer<K, V>)> = Vec::with_capacity(2);
        let node_key = &parent.child_key;
        let sibling_array = parent.node().get_siblings(node_key, guard);
        if parent.node().status_word().read(guard) != parent_status {
            // parent changed, found siblings can be invalid, retry
            return Err(MergeResult::Retry);
        }

        for (key, sibling) in sibling_array.iter().flatten() {
            siblings.push((key, sibling.read(guard)));
        }

        let mut merged_sibling_key: Option<&Key<K>> = None;
        let mut merged: Option<(&Key<K>, NodePointer<K, V>)> = None;
        for (sibling_key, sibling) in &siblings {
            let sibling_status = sibling.status_word().read(guard);
            if sibling_status.is_frozen() {
                // sibling is frozen by merge in other thread, retry merge again because parent
                // already changed and current merge will fail.
                return Err(MergeResult::Retry);
            }

            // try merge node with sibling
            match node {
                Leaf(node) => match sibling {
                    Leaf(other) => match node.merge_with_leaf(other, self.node_size, guard) {
                        MergeMode::NewNode(merged_node) => {
                            let node_key = std::cmp::max(&parent.child_key, sibling_key);
                            let node_ptr = NodePointer::new_leaf(merged_node);
                            merged = Some((node_key, node_ptr));
                            merged_sibling_key = Some(*sibling_key);
                            mwcas.compare_exchange(
                                sibling.status_word(),
                                sibling_status,
                                sibling_status.froze(),
                            );
                            break;
                        }
                        MergeMode::MergeFailed => {}
                    },
                    Interim(_) => {
                        panic!("Can't merge leaf with interim node")
                    }
                },
                Interim(node) => match sibling {
                    Interim(other) => {
                        match node.merge_with_interim(
                            other,
                            node.estimated_len(guard) + other.estimated_len(guard),
                            guard,
                        ) {
                            MergeMode::NewNode(merged_node) => {
                                let node_key = std::cmp::max(&parent.child_key, sibling_key);
                                let node_ptr = NodePointer::new_interim(merged_node);
                                merged = Some((node_key, node_ptr));
                                merged_sibling_key = Some(*sibling_key);
                                mwcas.compare_exchange(
                                    sibling.status_word(),
                                    sibling_status,
                                    sibling_status.froze(),
                                );
                                break;
                            }
                            MergeMode::MergeFailed => {}
                        }
                    }
                    Leaf(_) => panic!("Can't merge interim node with leaf node"),
                },
            }
        }

        if merged.is_none() {
            // no siblings have enough space to be merged
            return Err(MergeResult::Completed);
        }

        let (merged_key, merged_node_ptr) = merged.unwrap();
        let merged_sibling_key = merged_sibling_key.unwrap();
        // create new parent node with merged node and without empty/merged siblings.
        let mut buffer = Vec::with_capacity(parent.node().estimated_len(guard) - 1);
        let merged_node_key = &parent.child_key;
        for (key, val) in parent.node().iter(guard) {
            if key == merged_node_key {
                // replace underutilized node by merged one
                buffer.push((
                    (*merged_key).clone(),
                    HeapPointer::new(merged_node_ptr.clone()),
                ));
            } else if key != merged_sibling_key {
                // remove merged sibling
                buffer.push((key.clone(), HeapPointer::new(val.read(guard).clone())));
            }
        }
        Ok(InterimNode::new_readonly(buffer))
    }

    fn split_leaf(&self, path_to_leaf: TraversePath<K, V>, guard: &Guard) {
        let mut path = path_to_leaf;
        // try_split can return parent which should also be split
        // because it will overflow when we split passed leaf node
        while let Some(split_node) = self.try_split(path, guard) {
            path = split_node;
        }
    }

    /// Try split passed node. If node requires split of it's parents, method return new split
    /// context with parent node as split node.
    fn try_split<'g>(
        &self,
        mut path: TraversePath<'g, K, V>,
        guard: &'g Guard,
    ) -> Option<TraversePath<'g, K, V>> {
        let parent = path.parents.pop();
        if parent.is_none() {
            self.split_root(path.node_pointer, guard);
            return None;
        }

        if !path.node_pointer.try_froze() {
            // someone already try to split/merge node
            return None;
        }

        // freeze node to ensure that no one can modify it during split
        let parent = parent.unwrap();
        let parent_status = parent.node().status_word().read(guard);
        if parent_status.is_frozen() {
            path.node_pointer.try_unfroze();
            return None;
        }

        // node split results in new parent with replaced overflowed node by two new nodes
        match self.try_split_node(path.node_pointer, &parent.child_key, parent.node(), guard) {
            SplitResult::Split(new_parent) => {
                // node was split: create new parent which links to 2 new children which replace
                // original overflow node
                let mut mwcas = MwCas::new();
                mwcas.compare_exchange(
                    parent.node().status_word(),
                    parent_status,
                    parent_status.froze(),
                );
                if let Some(grand_parent) = path.parents.last() {
                    let status = grand_parent.node().status_word().read(guard);
                    if !status.is_frozen() {
                        // check that grandparent is not frozen and increase it version
                        // (because we make in-place update of link to parent node)
                        mwcas.compare_exchange(
                            grand_parent.node().status_word(),
                            status,
                            status.inc_version(),
                        );
                    } else {
                        // retry split again
                        path.node_pointer.try_unfroze();
                        return None;
                    }
                }

                // update link in grandparent node to new parent node
                // or parent is a root node which should be replaced by new one
                mwcas.compare_exchange(parent.cas_pointer, parent.node_pointer, new_parent);
                if !mwcas.exec(guard) {
                    path.node_pointer.try_unfroze();
                }
                None
            }
            SplitResult::Compacted(compacted_node) => {
                // node overflow caused by too many updates => replace overflow node
                // with new compacted node which can accept more elements
                let mut mwcas = MwCas::new();
                mwcas.compare_exchange(
                    parent.node().status_word(),
                    parent_status,
                    parent_status.inc_version(),
                );
                mwcas.compare_exchange(path.cas_pointer, path.node_pointer, compacted_node);
                if !mwcas.exec(guard) {
                    path.node_pointer.try_unfroze();
                }
                None
            }
            SplitResult::ParentOverflow => {
                // parent node is full and should be split before we can insert new child
                path.node_pointer.try_unfroze();
                Some(TraversePath {
                    cas_pointer: parent.cas_pointer,
                    node_pointer: parent.node_pointer,
                    parents: path.parents,
                })
            }
        }
    }

    fn split_root(&self, root: &NodePointer<K, V>, guard: &Guard) {
        if !root.try_froze() {
            return;
        }

        let new_root = match root {
            Leaf(cur_root) => {
                match cur_root.split_leaf(guard) {
                    SplitMode::Split(left, right) => {
                        // greatest key of left node moved to parent as split point
                        let left_key = left
                            .last_kv(guard)
                            .expect("Left node must have at least 1 element after split")
                            .0
                            .clone();
                        debug_assert!(
                            right.exact_len(guard) > 0,
                            "Right node must have at least 1 element after split"
                        );
                        // keys between (..left_key] in left
                        // keys between (left_key..+Inf] in right
                        let sorted_elements = vec![
                            (
                                Key::new(left_key),
                                HeapPointer::new(NodePointer::new_leaf(left)),
                            ),
                            (
                                Key::pos_infinite(),
                                HeapPointer::new(NodePointer::new_leaf(right)),
                            ),
                        ];
                        NodePointer::new_interim(InterimNode::new_readonly(sorted_elements))
                    }
                    SplitMode::Compact(compacted_root) => NodePointer::new_leaf(compacted_root),
                }
            }
            Interim(cur_root) => {
                match cur_root.split_interim(guard) {
                    SplitMode::Split(left, right) => {
                        let left_key = left
                            .last_kv(guard)
                            .expect("Left node must have at least 1 element after split")
                            .0
                            .clone();
                        let right_key = right
                            .last_kv(guard)
                            .expect("Right node must have at least 1 element after split")
                            .0
                            .clone();
                        // keys between (..left_key] in left
                        // keys between (left_key..right_key] in right
                        let sorted_elements = vec![
                            (left_key, HeapPointer::new(NodePointer::new_interim(left))),
                            (right_key, HeapPointer::new(NodePointer::new_interim(right))),
                        ];
                        NodePointer::new_interim(InterimNode::new_readonly(sorted_elements))
                    }
                    SplitMode::Compact(compacted_root) => NodePointer::new_interim(compacted_root),
                }
            }
        };

        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(&self.root, root, new_root);
        let res = mwcas.exec(guard);
        debug_assert!(res);
    }

    fn try_split_node(
        &self,
        node: &NodePointer<K, V>,
        underflow_node_key: &Key<K>,
        parent_node: &InterimNode<K, V>,
        guard: &Guard,
    ) -> SplitResult<K, V> {
        /// Create new parent node with 2 nodes created by split of other node.
        #[inline(always)]
        fn new_parent<K, V>(
            parent_node: &InterimNode<K, V>,
            underflow_node_key: &Key<K>,
            left_key: Key<K>,
            left_child: NodePointer<K, V>,
            right_child: NodePointer<K, V>,
            max_node_size: usize,
            guard: &Guard,
        ) -> Option<InterimNode<K, V>>
        where
            K: Clone + Ord,
        {
            let node_len = parent_node.estimated_len(guard);
            if node_len == max_node_size {
                // parent node overflow, should be split
                return None;
            }

            let mut sorted_elems = Vec::with_capacity(node_len + 1);
            for (key, val) in parent_node.iter(guard) {
                // overflowed node found inside parent, replace it by 2 new nodes
                if underflow_node_key == key {
                    sorted_elems.push((left_key.clone(), HeapPointer::new(left_child.clone())));
                    sorted_elems.push((
                        underflow_node_key.clone(),
                        HeapPointer::new(right_child.clone()),
                    ));
                } else {
                    sorted_elems.push((key.clone(), val.clone()));
                }
            }

            Some(InterimNode::new_readonly(sorted_elems))
        }

        match node {
            Leaf(leaf) => match leaf.split_leaf(guard) {
                SplitMode::Split(left, right) => {
                    let left_key = left
                        .last_kv(guard)
                        .expect("Left node must have at least 1 element after split")
                        .0
                        .clone();
                    if let Some(new_parent) = new_parent(
                        parent_node,
                        underflow_node_key,
                        Key::new(left_key),
                        NodePointer::new_leaf(left),
                        NodePointer::new_leaf(right),
                        self.node_size,
                        guard,
                    ) {
                        SplitResult::Split(NodePointer::new_interim(new_parent))
                    } else {
                        SplitResult::ParentOverflow
                    }
                }
                SplitMode::Compact(compacted_node) => {
                    SplitResult::Compacted(NodePointer::new_leaf(compacted_node))
                }
            },
            Interim(interim) => match interim.split_interim(guard) {
                SplitMode::Split(left, right) => {
                    let left_key = left
                        .last_kv(guard)
                        .expect("Left node must have at least 1 element after split")
                        .0
                        .clone();
                    if let Some(new_parent) = new_parent(
                        parent_node,
                        underflow_node_key,
                        left_key,
                        NodePointer::new_interim(left),
                        NodePointer::new_interim(right),
                        self.node_size,
                        guard,
                    ) {
                        SplitResult::Split(NodePointer::new_interim(new_parent))
                    } else {
                        SplitResult::ParentOverflow
                    }
                }
                SplitMode::Compact(compacted_node) => {
                    SplitResult::Compacted(NodePointer::new_interim(compacted_node))
                }
            },
        }
    }

    /// Find leaf node(with traversal path from root) which can contain passed key.
    /// If no node exists which can contain passed key, it can be created using
    /// `create_pos_inf_node_if_not_exists` flag.
    fn find_path_to_key<'g, Q>(
        &'g self,
        search_key: &Key<Q>,
        create_pos_inf_node_if_not_exists: bool,
        guard: &'g Guard,
    ) -> Option<TraversePath<'g, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let mut parents = Vec::new();
        let mut next_node: &NodeLink<K, V> = &self.root;
        loop {
            next_node = match next_node.read(guard) {
                node_pointer @ Interim(node) => {
                    if let Some((child_node_key, child_node_ptr)) = node.closest(search_key, guard)
                    {
                        // found node which can contain the key
                        parents.push(Parent {
                            cas_pointer: next_node,
                            node_pointer,
                            child_key: child_node_key.clone(),
                        });
                        child_node_ptr
                    } else if create_pos_inf_node_if_not_exists {
                        // No place found for the key, we need to grow the tree.
                        // Greatest element of interim node will serve as +Inf node to
                        // accommodate new KV.
                        self.insert_pos_inf_node(
                            TraversePath {
                                cas_pointer: next_node,
                                node_pointer,
                                parents,
                            },
                            guard,
                        );
                        // tree is changed, restart from root
                        parents = Vec::new();
                        &self.root
                    } else {
                        return None;
                    }
                }
                leaf_pointer @ Leaf(_) => {
                    return Some(TraversePath {
                        cas_pointer: next_node,
                        node_pointer: leaf_pointer,
                        parents,
                    });
                }
            }
        }
    }

    /// When interim node doesn't contains +Inf node and client tries to insert key which is
    /// greater than any other key in the tree, we should update interim's node which
    /// contains link to leaf with greatest element. This update will replace key of greatest
    /// element in interim node by +Inf key. This updated interim can now accept new keys which
    /// are greater than other keys of the tree.
    fn insert_pos_inf_node<'g>(&'g self, mut path: TraversePath<K, V>, guard: &'g Guard) -> bool {
        let parent = Parent {
            child_key: Key::pos_infinite(), // key is ignored in this context
            node_pointer: path.node_pointer,
            cas_pointer: path.cas_pointer,
        };
        path.parents.push(parent);

        // update key of greatest node to +Inf on all levels of tree
        let mut mwcas = MwCas::new();
        for interim in path.parents {
            let node = interim.node_pointer.to_interim_node();
            let status_word = node.status_word().read(guard);
            if status_word.is_frozen() {
                return false;
            }
            mwcas.compare_exchange(
                interim.node_pointer.status_word(),
                status_word,
                status_word.inc_version(),
            );

            let mut elems: Vec<(Key<K>, NodeLink<K, V>)> = node
                .iter(guard)
                .map(|(key, val)| (key.clone(), val.clone()))
                .collect();
            elems.last_mut().unwrap().0 = Key::pos_infinite();
            let new_interim = InterimNode::new_readonly(elems);
            // update link in parent node to new interim node
            mwcas.compare_exchange(
                interim.cas_pointer,
                interim.node_pointer,
                NodePointer::new_interim(new_interim),
            );
        }

        mwcas.exec(guard)
    }

    /// Find leaf node which can contain passed key
    fn find_leaf_for_key<'g, Q>(
        &'g self,
        search_key: &Key<Q>,
        create_if_not_exists: bool,
        guard: &'g Guard,
    ) -> Option<&'g LeafNode<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + Clone,
    {
        self.find_path_to_key(search_key, create_if_not_exists, guard)
            .map(|path| path.node_pointer.to_leaf_node())
    }

    /// Return leaf node which contains max known key of the tree. Also return key in parent node
    /// which points to this leaf(or `None` if this is root node).
    fn find_edge_node<'g, Q>(
        &'g self,
        node_edge: NodeEdge,
        guard: &'g Guard,
    ) -> (&'g LeafNode<K, V>, Option<&'g Key<K>>)
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let mut next_node: (&NodeLink<K, V>, Option<&Key<K>>) = (&self.root, None);
        loop {
            next_node = match next_node.0.read(guard) {
                Interim(node) => {
                    let (key, child_node_ptr) = node
                        .edge_kv(node_edge, guard)
                        .map(|e| (e.key, e.value))
                        .expect("Tree never contains empty interim nodes");

                    (child_node_ptr, Some(key))
                }
                Leaf(node) => {
                    return (node, next_node.1);
                }
            }
        }
    }
}

impl<K, V> Default for BzTree<K, V>
where
    K: Clone + Ord,
    V: Clone + Send + Sync,
{
    fn default() -> Self {
        BzTree::new()
    }
}

enum MergeResult<'g, K: Ord, V> {
    Completed,
    RecursiveMerge(TraversePath<'g, K, V>),
    Retry,
}

enum SplitResult<K: Ord, V> {
    Split(NodePointer<K, V>),
    Compacted(NodePointer<K, V>),
    ParentOverflow,
}

struct TraversePath<'g, K: Ord, V> {
    /// Node pointer of this node inside parent(used by MwCAS).
    cas_pointer: &'g NodeLink<K, V>,
    /// Pointer to found node inside tree(read from CAS pointer during traversal)
    node_pointer: &'g NodePointer<K, V>,
    /// Chain of parents including root node(starts from root, vector end is most closest parent)
    parents: Vec<Parent<'g, K, V>>,
}

struct ArcLeafNode<K: Ord, V> {
    ref_cnt: *mut AtomicUsize,
    node: *mut LeafNode<K, V>,
}

impl<K: Ord, V> ArcLeafNode<K, V> {
    fn new(leaf: LeafNode<K, V>) -> Self {
        ArcLeafNode {
            ref_cnt: Box::into_raw(Box::new(AtomicUsize::new(1))),
            node: Box::into_raw(Box::new(leaf)),
        }
    }
}

impl<K: Ord, V> Deref for ArcLeafNode<K, V> {
    type Target = LeafNode<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.node }
    }
}

impl<K: Ord, V> DerefMut for ArcLeafNode<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.node }
    }
}

impl<K: Ord, V> Clone for ArcLeafNode<K, V> {
    fn clone(&self) -> Self {
        unsafe {
            let prev = (*self.ref_cnt).fetch_add(1, Ordering::AcqRel);
            debug_assert!(prev > 0);
        }
        ArcLeafNode {
            ref_cnt: self.ref_cnt,
            node: self.node,
        }
    }
}

impl<K: Ord, V> Drop for ArcLeafNode<K, V> {
    fn drop(&mut self) {
        unsafe {
            let prev = (*self.ref_cnt).fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                drop(Box::from_raw(self.ref_cnt));
                drop(Box::from_raw(self.node));
            }
        }
    }
}

unsafe impl<K: Ord, V> Send for ArcLeafNode<K, V> {}

unsafe impl<K: Ord, V> Sync for ArcLeafNode<K, V> {}

struct ArcInterimNode<K: Ord, V> {
    ref_cnt: *mut AtomicUsize,
    node: *mut InterimNode<K, V>,
}

impl<K: Ord, V> ArcInterimNode<K, V> {
    fn new(interim: InterimNode<K, V>) -> Self {
        ArcInterimNode {
            ref_cnt: Box::into_raw(Box::new(AtomicUsize::new(1))),
            node: Box::into_raw(Box::new(interim)),
        }
    }
}

impl<K: Ord, V> Deref for ArcInterimNode<K, V> {
    type Target = InterimNode<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.node }
    }
}

impl<K: Ord, V> DerefMut for ArcInterimNode<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.node }
    }
}

impl<K: Ord, V> Clone for ArcInterimNode<K, V> {
    fn clone(&self) -> Self {
        unsafe {
            let prev = (*self.ref_cnt).fetch_add(1, Ordering::AcqRel);
            debug_assert!(prev > 0);
        }
        ArcInterimNode {
            ref_cnt: self.ref_cnt,
            node: self.node,
        }
    }
}

impl<K: Ord, V> Drop for ArcInterimNode<K, V> {
    fn drop(&mut self) {
        unsafe {
            let prev = (*self.ref_cnt).fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                drop(Box::from_raw(self.ref_cnt));
                drop(Box::from_raw(self.node));
            }
        }
    }
}

unsafe impl<K: Ord, V> Send for ArcInterimNode<K, V> {}

unsafe impl<K: Ord, V> Sync for ArcInterimNode<K, V> {}

enum NodePointer<K: Ord, V> {
    Leaf(ArcLeafNode<K, V>),
    Interim(ArcInterimNode<K, V>),
}

impl<K: Ord, V> Clone for NodePointer<K, V> {
    fn clone(&self) -> Self {
        match self {
            Leaf(node) => Leaf(node.clone()),
            Interim(node) => Interim(node.clone()),
        }
    }
}

unsafe impl<K: Ord, V> Send for NodePointer<K, V> {}

unsafe impl<K: Ord, V> Sync for NodePointer<K, V> {}

impl<K: Ord, V> From<ArcInterimNode<K, V>> for NodePointer<K, V> {
    fn from(node: ArcInterimNode<K, V>) -> Self {
        Interim(node)
    }
}

impl<K: Ord, V> NodePointer<K, V> {
    #[inline]
    fn new_leaf(node: LeafNode<K, V>) -> NodePointer<K, V> {
        let leaf_node = ArcLeafNode::new(node);
        Leaf(leaf_node)
    }

    #[inline]
    fn new_interim(node: InterimNode<K, V>) -> NodePointer<K, V> {
        let interim_node = ArcInterimNode::new(node);
        Interim(interim_node)
    }

    #[inline]
    fn points_to_leaf(&self, node_ptr: &LeafNode<K, V>) -> bool {
        match self {
            Interim(_) => false,
            Leaf(node) => ptr::eq(node.node, node_ptr),
        }
    }

    #[inline]
    fn to_interim_node(&self) -> &InterimNode<K, V> {
        match self {
            Interim(node) => node,
            Leaf(_) => panic!("Pointer points to leaf node"),
        }
    }

    #[inline]
    fn to_leaf_node(&self) -> &LeafNode<K, V> {
        match self {
            Interim(_) => panic!("Pointer points to interim node"),
            Leaf(node) => node,
        }
    }

    #[inline]
    fn len(&self, guard: &Guard) -> usize
    where
        K: Ord,
    {
        match self {
            Leaf(node) => node.exact_len(guard),
            Interim(node) => node.estimated_len(guard),
        }
    }

    #[inline]
    fn try_froze(&self) -> bool {
        let guard = crossbeam_epoch::pin();
        match self {
            Leaf(node) => node.try_froze(&guard),
            Interim(node) => node.try_froze(&guard),
        }
    }

    #[inline]
    fn try_unfroze(&self) -> bool {
        let guard = crossbeam_epoch::pin();
        match self {
            Leaf(node) => node.try_unfroze(&guard),
            Interim(node) => node.try_unfroze(&guard),
        }
    }

    #[inline]
    fn status_word(&self) -> &HeapPointer<StatusWord> {
        match self {
            Leaf(node) => node.status_word(),
            Interim(node) => node.status_word(),
        }
    }
}

/// Special wrapper for tree keys which can hold special 'empty key'.
/// Empty key means 'positive infinity' key, which always great than any other key.
/// Such key used as guard element in node and get access to keys greater than
/// any other keys in node.
#[derive(Clone, Debug)]
#[repr(transparent)]
struct Key<K> {
    key: Option<K>,
}

impl<K> Key<K> {
    #[inline(always)]
    fn new(key: K) -> Self {
        Key { key: Some(key) }
    }

    #[inline(always)]
    fn pos_infinite() -> Self {
        Key { key: None }
    }
}

impl<K: Ord> Ord for Key<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match &self.key {
            Some(key) => match &other.key {
                Some(other_key) => key.cmp(other_key),
                None => std::cmp::Ordering::Less,
            },
            None => match other.key {
                Some(_) => std::cmp::Ordering::Greater,
                None => std::cmp::Ordering::Equal,
            },
        }
    }
}

/// Impl used by [`Node`] to find siblings or closest node to some other node.
impl<K: Borrow<Q>, Q: Ord> PartialOrd<Key<Q>> for Key<K> {
    fn partial_cmp(&self, other: &Key<Q>) -> Option<std::cmp::Ordering> {
        match &self.key {
            Some(key) => match &other.key {
                Some(other_key) => Some(key.borrow().cmp(other_key)),
                None => Some(std::cmp::Ordering::Less),
            },
            None => match other.key {
                Some(_) => Some(std::cmp::Ordering::Greater),
                None => Some(std::cmp::Ordering::Equal),
            },
        }
    }
}

impl<K: Eq> Eq for Key<K> {}

impl<K: Borrow<Q>, Q: Eq> PartialEq<Key<Q>> for Key<K> {
    fn eq(&self, other: &Key<Q>) -> bool {
        match &self.key {
            Some(key) => matches!(&other.key, Some(other_key) if key.borrow() == other_key),
            None => other.key.is_none(),
        }
    }
}

struct Parent<'a, K: Ord, V> {
    /// Node pointer of this parent inside grandparent(used by MwCAS).
    cas_pointer: &'a NodeLink<K, V>,
    /// Parent node pointer inside grandparent at moment of tree traversal(actual value read from
    /// cas_pointer at some moment in time).
    node_pointer: &'a NodePointer<K, V>,
    /// Key in parent node which points to leaf/non-leaf child.
    child_key: Key<K>,
}

impl<'a, K: Ord, V> Parent<'a, K, V> {
    fn node(&self) -> &InterimNode<K, V> {
        self.node_pointer.to_interim_node()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;
    use std::fmt::{Debug, Display, Formatter};

    use crate::BzTree;

    #[test]
    fn insert() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new("1"), "1", &crossbeam_epoch::pin());
        tree.insert(Key::new("2"), "2", &crossbeam_epoch::pin());
        tree.insert(Key::new("3"), "3", &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        assert!(matches!(tree.get(&"1", &guard), Some(&"1")));
        assert!(matches!(tree.get(&"2", &guard), Some(&"2")));
        assert!(matches!(tree.get(&"3", &guard), Some(&"3")));
    }

    #[test]
    fn upsert() {
        let tree = BzTree::with_node_size(2);
        tree.upsert(Key::new("1"), "1", &crossbeam_epoch::pin());
        tree.upsert(Key::new("2"), "2", &crossbeam_epoch::pin());
        tree.upsert(Key::new("3"), "3", &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        assert!(matches!(tree.get(&"1", &guard), Some(&"1")));
        assert!(matches!(tree.get(&"2", &guard), Some(&"2")));
        assert!(matches!(tree.get(&"3", &guard), Some(&"3")));
    }

    #[test]
    fn delete() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new("1"), "1", &crossbeam_epoch::pin());
        tree.insert(Key::new("2"), "2", &crossbeam_epoch::pin());
        tree.insert(Key::new("3"), "3", &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        assert!(matches!(tree.delete(&"1", &guard), Some(&"1")));
        assert!(matches!(tree.delete(&"2", &guard), Some(&"2")));
        assert!(matches!(tree.delete(&"3", &guard), Some(&"3")));

        assert!(tree.iter(&guard).next().is_none());

        assert!(tree.get(&"1", &guard).is_none());
        assert!(tree.get(&"2", &guard).is_none());
        assert!(tree.get(&"3", &guard).is_none());
    }

    #[test]
    fn forward_scan() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new("1"), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new("2"), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new("3"), String::from("3"), &crossbeam_epoch::pin());
        tree.insert(Key::new("4"), String::from("4"), &crossbeam_epoch::pin());
        tree.insert(Key::new("5"), String::from("5"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        assert!(tree.range(Key::new("6").., &guard).next().is_none());

        let mut iter = tree.range(Key::new("2").., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("3").., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("1").., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("3")..=Key::new("6"), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("2")..Key::new("4"), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..=Key::new("5"), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..Key::new("6"), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn reversed_scan() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new("1"), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new("2"), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new("3"), String::from("3"), &crossbeam_epoch::pin());
        tree.insert(Key::new("4"), String::from("4"), &crossbeam_epoch::pin());
        tree.insert(Key::new("5"), String::from("5"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        let mut iter = tree.range(Key::new("1").., &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("3").., &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..Key::new("3"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..=Key::new("3"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("2")..Key::new("4"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("2")..=Key::new("4"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn reversed_scan_with_deletes() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new("1"), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new("2"), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new("3"), String::from("3"), &crossbeam_epoch::pin());
        tree.insert(Key::new("4"), String::from("4"), &crossbeam_epoch::pin());
        tree.insert(Key::new("5"), String::from("5"), &crossbeam_epoch::pin());
        tree.insert(Key::new("6"), String::from("6"), &crossbeam_epoch::pin());
        tree.insert(Key::new("7"), String::from("7"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        tree.delete(&"1", &guard).unwrap();
        tree.delete(&"2", &guard).unwrap();
        tree.delete(&"4", &guard).unwrap();
        tree.delete(&"6", &guard).unwrap();
        tree.delete(&"7", &guard).unwrap();

        let mut iter = tree.range(Key::new("0").., &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("2").., &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new("3").., &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..Key::new("3"), &guard).rev();
        assert!(iter.next().is_none());

        let mut iter = tree.range(..=Key::new("3"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..=Key::new("5"), &guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn mixed_scan() {
        let tree = BzTree::with_node_size(3);
        tree.insert(
            Key::new(String::from("1")),
            String::from("1"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("2")),
            String::from("2"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("3")),
            String::from("3"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("4")),
            String::from("4"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("5")),
            String::from("5"),
            &crossbeam_epoch::pin(),
        );

        let guard = crossbeam_epoch::pin();
        let mut iter = tree.range((Key::new(String::from("1"))).., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next_back(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next_back(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn mixed_scan_on_root_node() {
        // size of tree node is greater than count of elements,
        // e.g. all elements placed in leaf root node
        let tree = BzTree::with_node_size(30);
        tree.insert(
            Key::new(String::from("1")),
            String::from("1"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("2")),
            String::from("2"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("3")),
            String::from("3"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("4")),
            String::from("4"),
            &crossbeam_epoch::pin(),
        );
        tree.insert(
            Key::new(String::from("5")),
            String::from("5"),
            &crossbeam_epoch::pin(),
        );

        let guard = crossbeam_epoch::pin();
        let mut iter = tree.range((Key::new(String::from("1"))).., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next_back(), Some((_, val)) if val == &String::from("5")));
        assert!(matches!(iter.next_back(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn scan_after_delete() {
        let tree = BzTree::with_node_size(3);
        tree.insert(Key::new(1), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new(2), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new(3), String::from("3"), &crossbeam_epoch::pin());
        tree.insert(Key::new(4), String::from("4"), &crossbeam_epoch::pin());
        tree.insert(Key::new(5), String::from("5"), &crossbeam_epoch::pin());
        tree.insert(Key::new(6), String::from("6"), &crossbeam_epoch::pin());
        tree.insert(Key::new(7), String::from("7"), &crossbeam_epoch::pin());
        tree.insert(Key::new(8), String::from("8"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        tree.delete(&1, &guard).unwrap();
        tree.delete(&2, &guard).unwrap();
        tree.delete(&5, &guard).unwrap();
        tree.delete(&7, &guard).unwrap();
        tree.delete(&8, &guard).unwrap();
        let mut iter = tree.range(Key::new(1).., &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("6")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new(3)..=Key::new(6), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("6")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(Key::new(2)..Key::new(4), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..=Key::new(6), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("6")));
        assert!(iter.next().is_none());

        let mut iter = tree.range(..Key::new(7), &guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("4")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("6")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn scan_outside_of_existing_key_range() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        tree.insert(Key::new(2), String::from("2"), &guard);
        tree.insert(Key::new(3), String::from("3"), &guard);
        tree.insert(Key::new(4), String::from("4"), &guard);
        tree.insert(Key::new(5), String::from("5"), &guard);
        tree.insert(Key::new(6), String::from("6"), &guard);
        tree.insert(Key::new(7), String::from("7"), &guard);
        tree.insert(Key::new(8), String::from("8"), &guard);
        tree.insert(Key::new(9), String::from("9"), &guard);
        tree.insert(Key::new(10), String::from("10"), &guard);
        tree.insert(Key::new(11), String::from("11"), &guard);
        tree.insert(Key::new(12), String::from("12"), &guard);
        tree.insert(Key::new(13), String::from("13"), &guard);

        assert_eq!(tree.range(Key::new(0)..Key::new(1), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(0)..=Key::new(1), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(0)..Key::new(2), &guard).count(), 0);
        assert_eq!(tree.range(..Key::new(2), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(14).., &guard).count(), 0);
        assert_eq!(tree.range(Key::new(14)..Key::new(16), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(14)..=Key::new(16), &guard).count(), 0);

        assert_eq!(tree.range(Key::new(5)..Key::new(5), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(6)..Key::new(2), &guard).count(), 0);
        assert_eq!(tree.range(Key::new(14)..Key::new(10), &guard).count(), 0);
    }

    #[test]
    fn test_iter() {
        let tree = BzTree::with_node_size(2);
        tree.insert(Key::new(1), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new(2), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new(3), String::from("3"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        let mut iter = tree.iter(&guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_iter_on_root_node() {
        // size of tree node is greater than count of elements,
        // e.g. all elements placed in leaf root node
        let tree = BzTree::with_node_size(20);
        tree.insert(Key::new(1), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new(2), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new(3), String::from("3"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();
        let mut iter = tree.iter(&guard);
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_rev_iter_on_root_node() {
        // size of tree node is greater than count of elements,
        // e.g. all elements placed in leaf root node
        let tree = BzTree::with_node_size(20);
        tree.insert(Key::new(1), String::from("1"), &crossbeam_epoch::pin());
        tree.insert(Key::new(2), String::from("2"), &crossbeam_epoch::pin());
        tree.insert(Key::new(3), String::from("3"), &crossbeam_epoch::pin());

        let guard = crossbeam_epoch::pin();

        let mut iter = tree.iter(&guard).rev();
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("3")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("2")));
        assert!(matches!(iter.next(), Some((_, val)) if val == &String::from("1")));
        assert!(iter.next().is_none());
    }

    #[test]
    fn first() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        tree.insert(Key::new("1"), "1", &guard);
        tree.insert(Key::new("2"), "2", &guard);
        tree.insert(Key::new("3"), "3", &guard);
        tree.insert(Key::new("4"), "4", &guard);
        tree.insert(Key::new("5"), "5", &guard);
        tree.insert(Key::new("6"), "6", &guard);
        tree.insert(Key::new("7"), "7", &guard);

        assert!(matches!(tree.first(&guard), Some((k, _)) if k == &Key::new("1")));

        tree.delete(&"1", &guard);
        tree.delete(&"2", &guard);
        tree.delete(&"3", &guard);

        assert!(matches!(tree.first(&guard), Some((k, _)) if k == &Key::new("4")));
    }

    #[test]
    fn last() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        tree.insert(Key::new("1"), "1", &guard);
        tree.insert(Key::new("2"), "2", &guard);
        tree.insert(Key::new("3"), "3", &guard);
        tree.insert(Key::new("4"), "4", &guard);
        tree.insert(Key::new("5"), "5", &guard);
        tree.insert(Key::new("6"), "6", &guard);
        tree.insert(Key::new("7"), "7", &guard);

        assert!(matches!(tree.last(&guard), Some((k, _)) if k == &Key::new("7")));

        tree.delete(&"5", &guard);
        tree.delete(&"6", &guard);
        tree.delete(&"7", &guard);

        assert!(matches!(tree.last(&guard), Some((k, _)) if k == &Key::new("4")));
    }

    #[test]
    fn pop_first() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        let count = 55;
        for i in 0..count {
            tree.insert(Key::new(i), i, &guard);
        }

        for i in 0..count {
            println!("{i}");
            assert!(matches!(tree.pop_first(&guard), Some((k, _)) if k == Key::new(i)));
        }

        assert!(tree.iter(&guard).next().is_none());
    }

    #[test]
    fn pop_last() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        let count = 55;
        for i in 0..count {
            tree.insert(Key::new(i), i, &guard);
        }

        for i in (0..count).rev() {
            assert!(matches!(tree.pop_last(&guard), Some((k, _)) if k == Key::new(i)));
        }

        assert!(tree.iter(&guard).next().is_none());
    }

    #[test]
    fn conditional_insert() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        let count = 55;
        for i in 0..count {
            tree.insert(Key::new(i), i, &guard);
        }

        for i in 0..count {
            assert!(tree.compute(&i, |(_, v)| Some(v + 1), &guard));
        }

        for i in 0..count {
            assert_eq!(*tree.get(&i, &guard).unwrap(), i + 1);
        }

        assert!(!tree.compute(&(count + 1), |(_, v)| Some(v + 1), &guard));
    }

    #[test]
    fn conditional_remove() {
        let tree = BzTree::with_node_size(2);
        let guard = crossbeam_epoch::pin();
        let count = 55;
        for i in 0..count {
            tree.insert(Key::new(i), i, &guard);
        }

        for i in 0..count {
            assert!(tree.compute(&i, |(_, _)| None, &guard));
        }

        for i in 0..count {
            assert!(matches!(tree.get(&i, &guard), None));
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
}
