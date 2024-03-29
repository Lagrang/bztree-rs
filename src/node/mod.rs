mod metadata;
pub mod scanner;
pub mod status_word;

use crate::node::scanner::NodeScanner;
use crossbeam_epoch::Guard;
use metadata::Metadata;
use mwcas::{HeapPointer, MwCas, U64Pointer};
use status_word::StatusWord;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::mem::MaybeUninit;
use std::ops::RangeBounds;
use std::option::Option::Some;
use std::sync::atomic;
use std::{mem, ptr};

/// BzTree node.
pub struct Node<K: Ord, V> {
    status_word: HeapPointer<StatusWord>,
    sorted_len: usize,
    data_block: Vec<Entry<K, V>>,
    readonly: bool,
}

macro_rules! deref_mut {
    ($node: expr) => {
        unsafe { &mut *($node as *const Node<K, V> as *mut Node<K, V>) }
    };
}

impl<K: Ord, V> Node<K, V> {
    /// Create empty node which can hold up to `max_elements`.
    pub fn with_capacity(max_elements: u16) -> Self
    where
        K: Clone + Ord,
    {
        Self::init_with_capacity(Vec::new(), max_elements)
    }

    pub fn new_readonly(mut sorted_elements: Vec<(K, V)>) -> Self
    where
        K: Clone + Ord,
        V: Clone + Send + Sync,
    {
        if sorted_elements.len() > u16::MAX as usize {
            panic!(
                "Node max size {}, but passed vector length {}",
                u16::MAX,
                sorted_elements.len()
            );
        }

        let capacity = sorted_elements.len();
        let mut kv_block = Vec::with_capacity(capacity);
        sorted_elements.drain(..).for_each(|(k, v)| {
            kv_block.push(Entry {
                key: MaybeUninit::new(k),
                value: MaybeUninit::new(v),
                metadata: Metadata::visible().into(),
            });
        });

        Node {
            data_block: kv_block,
            status_word: HeapPointer::new(StatusWord::with_records(capacity as u16)),
            sorted_len: capacity,
            readonly: true,
        }
    }

    fn init_with_capacity(mut sorted_elements: Vec<(K, V)>, capacity: u16) -> Self {
        let capacity = capacity as usize;
        let sorted_len = sorted_elements.len();
        if sorted_len > capacity {
            panic!(
                "Sorted KV vector(size={:?}) is greater than node capacity {:?}",
                sorted_len, capacity
            );
        }

        let mut kv_block = Vec::with_capacity(capacity);
        let remains = capacity - sorted_len;
        sorted_elements.drain(..).for_each(|(k, v)| {
            kv_block.push(Entry {
                key: MaybeUninit::new(k),
                value: MaybeUninit::new(v),
                metadata: Metadata::visible().into(),
            });
        });

        (0..remains).for_each(|_| {
            kv_block.push(Entry {
                key: MaybeUninit::uninit(),
                value: MaybeUninit::uninit(),
                metadata: Metadata::not_used().into(),
            });
        });

        Node {
            data_block: kv_block,
            status_word: HeapPointer::new(StatusWord::with_records(sorted_len as u16)),
            sorted_len,
            readonly: false,
        }
    }

    pub fn insert<'g>(&'g self, key: K, value: V, guard: &'g Guard) -> Result<(), InsertError<V>>
    where
        K: Ord,
        V: Send + Sync,
    {
        let cur_status = self.status_word.read(guard);
        match self.insert_phase_one(key, value, false, cur_status, guard) {
            Ok(reserved_entry) => self
                .insert_phase_two(reserved_entry, false, guard)
                .map(|_| ()),
            Err(e) => Err(e),
        }
    }

    /// Upsert value associated with passed key.
    /// ## Return:
    /// Previous value associated with key, if it exists.
    pub fn upsert<'g>(
        &'g self,
        key: K,
        value: V,
        guard: &'g Guard,
    ) -> Result<Option<&'g V>, InsertError<V>>
    where
        K: Ord,
        V: Send + Sync,
    {
        let cur_status = self.status_word.read(guard);
        self.conditional_upsert(key, value, cur_status, guard)
    }

    /// Upsert value associated with passed key if node wasn't changed(status word is same).
    /// ## Return:
    /// Previous value associated with key, if it exists.
    pub fn conditional_upsert<'g>(
        &'g self,
        key: K,
        value: V,
        cur_status: &StatusWord,
        guard: &'g Guard,
    ) -> Result<Option<&'g V>, InsertError<V>>
    where
        K: Ord,
        V: Send + Sync,
    {
        debug_assert!(!self.readonly);
        match self.insert_phase_one(key, value, true, cur_status, guard) {
            Ok(reserved_entry) => self.insert_phase_two(reserved_entry, true, guard),
            Err(e) => Err(e),
        }
    }

    /// Remove value associated with passed key.
    /// ## Return:
    /// Value associated with removed key
    pub fn delete<'g, Q>(&'g self, key: &Q, guard: &'g Guard) -> Result<&'g V, DeleteError>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        debug_assert!(!self.readonly);
        let status_word = self.status_word.read(guard);
        if status_word.is_frozen() {
            return Err(DeleteError::Retry);
        }

        let index = self
            .get_internal(key, status_word, true, guard)
            .map(|(_, _, kv_index)| kv_index)
            .map_err(|_| DeleteError::KeyNotFound)?;

        let entry = &mut deref_mut!(self).data_block[index];
        let cur_metadata: Metadata = entry.metadata.read(guard).into();
        if !cur_metadata.is_visible() {
            return Err(DeleteError::Retry);
        }

        let new_status = status_word.delete_entry();
        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(&self.status_word, status_word, new_status);
        mwcas.compare_exchange_u64(
            &entry.metadata,
            cur_metadata.into(),
            Metadata::deleted().into(),
        );
        if mwcas.exec(guard) {
            // only removed value should be dropped
            // key remains live because internally used during search operations
            unsafe {
                entry.defer_value_drop(guard);
                Ok(entry.value_mut())
            }
        } else {
            // other thread change state of node, retry
            Err(DeleteError::Retry)
        }
    }

    /// Remove value associated with passed key if node wasn't changed(status word is same).
    /// ## Return:
    /// Value associated with removed key
    pub fn conditional_delete<'g, Q>(
        &'g self,
        status_word: &StatusWord,
        kv_index: usize,
        guard: &'g Guard,
    ) -> Result<&'g V, DeleteError>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        debug_assert!(!self.readonly);
        if status_word.is_frozen() {
            return Err(DeleteError::Retry);
        }

        let new_status = status_word.delete_entry();
        let entry = &mut deref_mut!(self).data_block[kv_index];
        let cur_metadata: Metadata = entry.metadata.read(guard).into();
        if cur_metadata.is_deleted() {
            return Err(DeleteError::KeyNotFound);
        }
        if !cur_metadata.is_visible() {
            return Err(DeleteError::Retry);
        }

        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(&self.status_word, status_word, new_status);
        mwcas.compare_exchange_u64(
            &entry.metadata,
            cur_metadata.into(),
            Metadata::deleted().into(),
        );
        if mwcas.exec(guard) {
            // only removed value should be dropped
            // key remains live because internally used during search operations
            unsafe {
                entry.defer_value_drop(guard);
                Ok(entry.value_mut())
            }
        } else {
            // other thread change state of node, retry
            Err(DeleteError::Retry)
        }
    }

    pub fn get<'g, Q>(
        &'g self,
        key: &Q,
        guard: &'g Guard,
    ) -> Option<(&'g K, &'g V, &'g StatusWord, usize)>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let status_word = self.status_word.read(guard);
        self.get_internal(key, status_word, true, guard)
            .map_or_else(
                |_| None,
                |(key, val, val_index)| Some((key, val, status_word, val_index)),
            )
    }

    pub fn conditional_get<'g, Q>(
        &'g self,
        key: &Q,
        status_word: &StatusWord,
        guard: &'g Guard,
    ) -> Option<(&'g K, &'g V, usize)>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.get_internal(key, status_word, true, guard)
            .map_or_else(
                |_| None,
                |(key, val, val_index)| Some((key, val, val_index)),
            )
    }

    /// Get value which key is equal or greater that to passed key.  
    ///
    /// **Warning**: this method can be called only for **read-only** nodes, it ignores
    /// any updates made after node creation.
    pub fn closest<'g, Q>(&'g self, key: &Q, _: &'g Guard) -> Option<(&'g K, &'g mut V)>
    where
        K: PartialOrd<Q>,
        V: Clone,
    {
        debug_assert!(self.readonly);
        if self.sorted_len == 0 {
            return None;
        }

        let index = self.data_block[0..self.sorted_len]
            .binary_search_by(|entry| {
                // sorted block doesn't contain reserved entries, so it can be ignored here.
                // sorted block can contain deleted entries, but metadata of removed entries
                // still points to valid keys, so it's safe to compare them here.
                unsafe { entry.key() }
                    .borrow()
                    .partial_cmp(key)
                    .expect("Q type must always be comparable with K")
            })
            .map_or_else(
                |closest_pos| {
                    // find position which points to element which is greater that passed key
                    closest_pos
                },
                |index| {
                    // find exact match for key
                    index
                },
            );

        if index < self.sorted_len {
            let entry = &mut deref_mut!(self).data_block[index];
            let value = unsafe { &mut *entry.value.as_mut_ptr() };
            let key = unsafe { entry.key() };
            Some((key, value))
        } else {
            // passed key is greater than any element in block
            None
        }
    }

    /// Get left and right siblings for entry identified by key.  
    ///
    /// **Warning**: this method can be called only for **read-only** nodes, it ignores
    /// any updates made after node creation.
    pub fn get_siblings<'g, Q>(&'g self, key: &Q, _: &'g Guard) -> [Option<(&'g K, &'g V)>; 2]
    where
        K: PartialOrd<Q>,
        V: Clone,
    {
        debug_assert!(self.readonly);
        if self.sorted_len == 0 {
            return [None, None];
        }

        let index = self.data_block[0..self.sorted_len]
            .binary_search_by(|entry| {
                // sorted block doesn't contain reserved entries, so it can be ignored here.
                // sorted block can contain deleted entries, but metadata of removed entries
                // still points to valid keys, so it's safe to compare them here.
                unsafe { entry.key() }
                    .borrow()
                    .partial_cmp(key)
                    .expect("Q type must always be comparable with K")
            })
            .map_or_else(|_| panic!("Key not found!"), |index| index);

        let left = if index > 0 {
            let kv = &self.data_block[index - 1];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        };
        let right = if index < self.sorted_len - 1 {
            let kv = &self.data_block[index + 1];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        };
        [left, right]
    }

    pub fn range<'g, Q, Range>(&'g self, key_range: Range, guard: &'g Guard) -> NodeScanner<K, V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + 'g,
        Range: RangeBounds<Q> + 'g,
    {
        NodeScanner::new(self.status_word.read(guard), self, key_range, guard)
    }

    pub fn iter<'g>(&'g self, guard: &'g Guard) -> NodeScanner<K, V>
    where
        K: Ord,
    {
        self.range(.., guard)
    }

    /// Find KV which placed on some of node's edge(e.g. min KV or max KV).
    pub fn edge_kv<'g>(&'g self, edge: NodeEdge, guard: &'g Guard) -> Option<EntryWithLoc<'g, K, V>>
    where
        K: Ord,
    {
        let status_word = self.status_word().read(guard);
        self.conditional_edge_kv(edge, status_word, guard)
    }

    /// Find KV which placed on some of node's edge(e.g. min KV or max KV) at some point of time
    /// (represented by status word).
    pub fn conditional_edge_kv<'g>(
        &'g self,
        edge: NodeEdge,
        status_word: &StatusWord,
        guard: &'g Guard,
    ) -> Option<EntryWithLoc<'g, K, V>>
    where
        K: Ord,
    {
        let expected_ord = match edge {
            NodeEdge::Left => Ordering::Less,
            NodeEdge::Right => Ordering::Greater,
        };

        let mut unsorted_edge: Option<EntryWithLoc<K, V>> = None;
        // do we have some KVs in unsorted part?
        if self.sorted_len < status_word.reserved_records() as usize {
            for index in self.sorted_len..status_word.reserved_records() as usize {
                let entry = &self.data_block[index];
                let metadata: Metadata = entry.metadata.read(guard).into();
                if metadata.is_visible() {
                    // read the key only after we ensured that entry is valid, otherwise key can
                    // point to uninitialized memory
                    let key = unsafe { entry.key() };
                    unsorted_edge = unsorted_edge
                        .map(|edge_kv| {
                            debug_assert!(key != edge_kv.key);
                            if key.cmp(edge_kv.key) == expected_ord {
                                EntryWithLoc {
                                    location: index,
                                    key,
                                    value: unsafe { entry.value() },
                                }
                            } else {
                                edge_kv
                            }
                        })
                        .or_else(|| {
                            Some(EntryWithLoc {
                                location: index,
                                key,
                                value: unsafe { entry.value() },
                            })
                        });
                }
            }
        }

        let sorted_edge = match edge {
            NodeEdge::Left => self.data_block[..self.sorted_len]
                .iter()
                .enumerate()
                .filter_map(|(i, entry)| {
                    let metadata: Metadata = entry.metadata.read(guard).into();
                    if metadata.is_visible() {
                        Some(EntryWithLoc {
                            key: unsafe { entry.key() },
                            value: unsafe { entry.value() },
                            location: i,
                        })
                    } else {
                        None
                    }
                })
                .next(),
            NodeEdge::Right => self.data_block[..self.sorted_len]
                .iter()
                .enumerate()
                .rev()
                .filter_map(|(i, entry)| {
                    let metadata: Metadata = entry.metadata.read(guard).into();
                    if metadata.is_visible() {
                        Some(EntryWithLoc {
                            key: unsafe { entry.key() },
                            value: unsafe { entry.value() },
                            location: i,
                        })
                    } else {
                        None
                    }
                })
                .next(),
        };

        if sorted_edge.is_none() {
            unsorted_edge
        } else if unsorted_edge.is_none() {
            sorted_edge
        } else {
            let sorted_edge = sorted_edge.unwrap();
            let sorted_key = sorted_edge.key;
            let unsorted_edge = unsorted_edge.unwrap();
            let unsorted_key = unsorted_edge.key;
            debug_assert!(sorted_key != unsorted_key);
            if sorted_key.cmp(unsorted_key) == expected_ord {
                Some(sorted_edge)
            } else {
                Some(unsorted_edge)
            }
        }
    }

    pub fn last_kv<'g>(&'g self, guard: &'g Guard) -> Option<(&'g K, &'g V)>
    where
        K: Ord,
    {
        return self
            .edge_kv(NodeEdge::Right, guard)
            .map(|kv| (kv.key, kv.value));
    }

    pub fn split_leaf(&self, guard: &Guard) -> SplitMode<K, V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        debug_assert!(
            self.status_word.read(guard).is_frozen(),
            "Node must be frozen before split"
        );

        let mut kvs: Vec<(K, V)> = self.clone_content(guard);
        let capacity = self.capacity() as u16;
        if kvs.len() < capacity as usize {
            // node contains too many updates/deletes which cause split
            // instead of split we return compacted node
            return SplitMode::Compact(Self::init_with_capacity(kvs, capacity));
        }

        let split_point = kvs.len() / 2;
        let left = Self::init_with_capacity(kvs.drain(..split_point).collect(), capacity);
        let right = Self::init_with_capacity(kvs, capacity);

        SplitMode::Split(left, right)
    }

    pub fn split_interim(&self, guard: &Guard) -> SplitMode<K, V>
    where
        K: Clone + Ord,
        V: Clone + Send + Sync,
    {
        debug_assert!(
            self.status_word.read(guard).is_frozen(),
            "Node must be frozen before split"
        );

        let mut kvs: Vec<(K, V)> = self.clone_content(guard);
        let split_point = kvs.len() / 2;
        let left = Self::new_readonly(kvs.drain(..split_point).collect());
        let right = Self::new_readonly(kvs);

        SplitMode::Split(left, right)
    }

    pub fn merge_with_leaf(
        &self,
        other: &Self,
        merged_node_capacity: usize,
        guard: &Guard,
    ) -> MergeMode<K, V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        debug_assert!(
            self.status_word.read(guard).is_frozen(),
            "Node must be frozen before merge"
        );

        let mut p1 = self
            .iter(guard)
            .map(|(k, v)| ((*k).clone(), (*v).clone()))
            .peekable();
        let mut p2 = other
            .iter(guard)
            .map(|(k, v)| ((*k).clone(), (*v).clone()))
            .peekable();
        let mut sorted_kvs = Vec::with_capacity(merged_node_capacity);
        while let Some((k1, _)) = p1.peek() {
            if let Some((k2, _)) = p2.peek() {
                match k1.cmp(k2) {
                    Ordering::Less => {
                        sorted_kvs.push(p1.next().unwrap());
                    }
                    Ordering::Greater => {
                        sorted_kvs.push(p2.next().unwrap());
                    }
                    Ordering::Equal => panic!("Nodes of Btree can't have common keys"),
                }
            } else {
                sorted_kvs.push(p1.next().unwrap());
            }

            if sorted_kvs.len() > merged_node_capacity {
                return MergeMode::MergeFailed;
            }
        }
        p2.for_each(|kv| sorted_kvs.push(kv));

        if sorted_kvs.len() > merged_node_capacity {
            return MergeMode::MergeFailed;
        }

        MergeMode::NewNode(Self::init_with_capacity(
            sorted_kvs,
            merged_node_capacity as u16,
        ))
    }

    pub fn merge_with_interim(
        &self,
        other: &Self,
        merged_node_capacity: usize,
        guard: &Guard,
    ) -> MergeMode<K, V>
    where
        K: Clone + Ord,
        V: Clone + Send + Sync,
    {
        debug_assert!(
            self.status_word.read(guard).is_frozen(),
            "Node must be frozen before merge"
        );

        let len = self.estimated_len(guard);
        let other_len = other.estimated_len(guard);
        if len + other_len > merged_node_capacity {
            return MergeMode::MergeFailed;
        }

        let mut p1 = self
            .iter(guard)
            .map(|(k, v)| ((*k).clone(), (*v).clone()))
            .peekable();
        let mut p2 = other
            .iter(guard)
            .map(|(k, v)| ((*k).clone(), (*v).clone()))
            .peekable();
        let mut sorted_kvs = Vec::with_capacity(len + other_len);
        while let Some((k1, _)) = p1.peek() {
            if let Some((k2, _)) = p2.peek() {
                match k1.cmp(k2) {
                    Ordering::Less => {
                        sorted_kvs.push(p1.next().unwrap());
                    }
                    Ordering::Greater => {
                        sorted_kvs.push(p2.next().unwrap());
                    }
                    Ordering::Equal => panic!("Nodes of Btree can't have common keys"),
                }
            } else {
                sorted_kvs.push(p1.next().unwrap());
            }
        }
        p2.for_each(|kv| sorted_kvs.push(kv));

        MergeMode::NewNode(Self::new_readonly(sorted_kvs))
    }

    /// Clone all KVs of this node.
    pub fn clone_content<'g>(&'g self, guard: &'g Guard) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.iter(guard)
            .map(|(key, val)| (key.clone(), val.clone()))
            .collect()
    }

    /// Clone KVs of this node which pass the filter.
    pub fn clone_with_filter<'g, F>(&'g self, filter: F, guard: &'g Guard) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
        F: FnMut(&(&K, &V)) -> bool,
    {
        self.iter(guard)
            .filter(filter)
            .map(|(key, val)| (key.clone(), val.clone()))
            .collect()
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.data_block.len()
    }

    #[inline(always)]
    pub fn remaining_capacity(&self, guard: &Guard) -> usize {
        self.capacity() - self.status_word.read(guard).reserved_records() as usize
    }

    /// Estimated length of node. If no updates/deleted were applied to node,
    /// method return exact length. Read-only nodes should use this method
    /// to obtain exact length.
    #[inline(always)]
    pub fn estimated_len(&self, guard: &Guard) -> usize {
        let status_word: &StatusWord = self.status_word.read(guard);
        (status_word.reserved_records() - status_word.deleted_records()) as usize
    }

    #[inline(always)]
    pub fn exact_len(&self, guard: &Guard) -> usize
    where
        K: Ord,
    {
        self.iter(guard).count()
    }

    #[inline(always)]
    pub fn status_word(&self) -> &HeapPointer<StatusWord> {
        &self.status_word
    }

    #[inline]
    pub fn try_froze(&self, guard: &Guard) -> bool {
        let cur_status = self.status_word.read(guard);
        if cur_status.is_frozen() {
            return false;
        }
        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(&self.status_word, cur_status, cur_status.froze());
        mwcas.exec(guard)
    }

    #[inline]
    pub fn try_unfroze(&self, guard: &Guard) -> bool {
        let cur_status = self.status_word.read(guard);
        if !cur_status.is_frozen() {
            return false;
        }
        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(&self.status_word, cur_status, cur_status.unfroze());
        mwcas.exec(guard)
    }

    fn get_internal<'g, Q>(
        &'g self,
        key: &Q,
        status_word: &StatusWord,
        await_reserved_entries: bool,
        guard: &'g Guard,
    ) -> Result<(&'g K, &'g V, usize), SearchError>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        if status_word.reserved_records() as usize > self.sorted_len {
            // scan unsorted part first because it contain most recent values
            for index in (self.sorted_len..status_word.reserved_records() as usize).rev() {
                let entry = &self.data_block[index];
                let metadata = loop {
                    let metadata: Metadata = entry.metadata.read(guard).into();
                    if !metadata.is_reserved() {
                        break metadata;
                    }
                    if !await_reserved_entries {
                        return Err(SearchError::ReservedEntryFound);
                    }
                    // someone try to add entry at same time, continue check same entry
                    // until reserved entry will become valid
                };

                if metadata.is_visible() {
                    // read the key only after we ensured that entry is valid, otherwise key can
                    // point to uninitialized memory
                    let entry_key = unsafe { entry.key() };
                    if entry_key.borrow() == key {
                        return Ok((entry_key, unsafe { entry.value() }, index));
                    }
                } else if metadata.is_deleted() {
                    let entry_key = unsafe { entry.key() };
                    if entry_key.borrow() == key {
                        // most recent state of key indicates that it was deleted
                        return Err(SearchError::KeyNotFound);
                    }
                }
            }
        }

        if self.sorted_len == 0 {
            return Err(SearchError::KeyNotFound);
        }

        self.data_block[0..self.sorted_len]
            .binary_search_by(|entry| {
                // sorted block doesn't contain reserved entries, so it can be ignored here.
                // sorted block can contain deleted entries, but metadata of removed entries
                // still points to valid keys, so it's safe to compare them here.
                unsafe { entry.key() }.borrow().cmp(key)
            })
            .map_or_else(
                |_| Err(SearchError::KeyNotFound),
                |index| {
                    let entry = &self.data_block[index];
                    let metadata: Metadata = entry.metadata.read(guard).into();
                    if metadata.is_visible() {
                        Ok((unsafe { entry.key() }, unsafe { entry.value() }, index))
                    } else {
                        Err(SearchError::KeyNotFound)
                    }
                },
            )
    }

    fn insert_phase_one(
        &self,
        key: K,
        value: V,
        is_upsert: bool,
        cur_status: &StatusWord,
        guard: &Guard,
    ) -> Result<ReservedEntry<K, V>, InsertError<V>>
    where
        K: Ord,
    {
        // Phase 1: reserve space in node for new KV
        if cur_status.is_frozen() {
            return Err(InsertError::NodeFrozen(value));
        }

        if self.remaining_capacity(guard) == 0 {
            return Err(InsertError::Split(value));
        }

        // Scan unsorted data space in node to check if some other
        // reservation started before ours, but not yet completed.
        // Such reservations can be for the same key, we should await their completion.
        match self.get_internal(&key, cur_status, false, guard) {
            Ok((_, _, index)) => {
                // found same key which is actual value(not reserved)
                if is_upsert {
                    self.try_reserve_entry(key, value, cur_status, false, guard)
                        .map(|mut reserved| {
                            reserved.overwritten_entry = Some(index);
                            reserved
                        })
                } else {
                    Err(InsertError::DuplicateKey)
                }
            }
            Err(SearchError::KeyNotFound) => {
                self.try_reserve_entry(key, value, cur_status, false, guard)
            }
            Err(SearchError::ReservedEntryFound) => {
                // We must rescan unsorted space again right before
                // completion of KV insert to ensure that
                // this reservation is not for same key.
                self.try_reserve_entry(key, value, cur_status, true, guard)
            }
        }
    }

    fn insert_phase_two<'g>(
        &'g self,
        mut new_entry: ReservedEntry<K, V>,
        is_upsert: bool,
        guard: &'g Guard,
    ) -> Result<Option<&'g V>, InsertError<V>>
    where
        K: Ord,
    {
        // Phase 2: complete KV write into reserved space
        if new_entry.await_reserved_entries {
            // reserved entry which is not committed found:
            // await until all space reservations will be completed
            // and we can ensure that no duplicate key in node
            if let Ok((_, _, index)) =
                self.get_internal(&new_entry.key, &new_entry.prev_status_word, true, guard)
            {
                if is_upsert {
                    new_entry.overwritten_entry = Some(index);
                } else {
                    self.clear_reserved_entry(new_entry.index, guard);
                    return Err(InsertError::DuplicateKey);
                }
            }
        }

        // Complete write of KV into reserved entry.
        let index = new_entry.index;
        let reserved_metadata: Metadata = self.data_block[index].metadata.read(guard).into();
        debug_assert!(reserved_metadata.is_reserved());

        let self_mut = deref_mut!(self);
        unsafe {
            // we should write KV entry before we will make it visible
            self_mut.data_block[index]
                .key
                .as_mut_ptr()
                .write_volatile(new_entry.key);
            self_mut.data_block[index]
                .value
                .as_mut_ptr()
                .write_volatile(new_entry.value);
            // ensure that all written data can be seen by other threads
            atomic::fence(atomic::Ordering::SeqCst);
        }

        loop {
            let current_status: &StatusWord = self.status_word.read(guard);
            if current_status.is_frozen() {
                // no one seen this KV yet, move out it from node
                self.clear_reserved_entry(index, guard);
                unsafe {
                    mem::replace(&mut self_mut.data_block[index].key, MaybeUninit::uninit())
                        .assume_init();
                }
                let value = unsafe {
                    mem::replace(&mut self_mut.data_block[index].value, MaybeUninit::uninit())
                        .assume_init()
                };
                return Err(InsertError::NodeFrozen(value));
            }

            let mut mwcas = MwCas::new();
            mwcas.compare_exchange_u64(
                &self.data_block[index].metadata,
                reserved_metadata.into(),
                Metadata::visible().into(),
            );

            let mut new_status = current_status.clone();
            // mark existing KV as 'deleted' if it will be overwritten by upsert operation
            if let Some(existing_entry_index) = new_entry.overwritten_entry {
                let existing_entry = &mut self_mut.data_block[existing_entry_index];
                let metadata: Metadata = existing_entry.metadata.read(guard).into();
                // No need to check that metadata is still visible here.
                // If someone already removed this existing entry, this will be catch by status
                // word CAS.
                mwcas.compare_exchange_u64(
                    &existing_entry.metadata,
                    metadata.into(),
                    Metadata::deleted().into(),
                );
                new_status = new_status.delete_entry();
            }

            // ensure that node is not frozen during MWCAS
            // or status word changed by other insertion/deletion/split/merge.
            mwcas.compare_exchange(&self.status_word, current_status, new_status);

            if mwcas.exec(guard) {
                return if let Some(existing_entry_index) = new_entry.overwritten_entry {
                    // drop value replaced by upsert
                    let existing_entry = &mut self_mut.data_block[existing_entry_index];
                    unsafe {
                        existing_entry.defer_value_drop(guard);
                        Ok(Some(existing_entry.value()))
                    }
                } else {
                    Ok(None)
                };
            }
        }
    }

    /// Method try to reserved space for new KV entry.
    fn try_reserve_entry(
        &self,
        key: K,
        value: V,
        current_status: &StatusWord,
        await_reserved_entries: bool,
        guard: &Guard,
    ) -> Result<ReservedEntry<K, V>, InsertError<V>> {
        let next_entry_index = current_status.reserved_records() as usize;
        let metadata = &self.data_block[next_entry_index].metadata;
        let mut mwcas = MwCas::new();
        mwcas.compare_exchange(
            &self.status_word,
            current_status,
            current_status.reserve_entry(),
        );
        mwcas.compare_exchange_u64(metadata, metadata.read(guard), Metadata::reserved().into());
        if mwcas.exec(guard) {
            Ok(ReservedEntry {
                key,
                value,
                index: next_entry_index,
                overwritten_entry: None, // will be filled by caller if needed
                prev_status_word: current_status.clone(),
                await_reserved_entries,
            })
        } else {
            // other thread change state of node, retry
            Err(InsertError::Retry(value))
        }
    }

    /// Mark metadata entry as unused(will be skipped by search operations)
    fn clear_reserved_entry(&self, index: usize, guard: &Guard) {
        let reserved_metadata: Metadata = self.data_block[index].metadata.read(guard).into();
        debug_assert!(reserved_metadata.is_reserved());
        let mut mwcas = MwCas::new();
        mwcas.compare_exchange_u64(
            &self.data_block[index].metadata,
            reserved_metadata.into(),
            Metadata::not_used().into(),
        );
        let res = mwcas.exec(guard);
        debug_assert!(res);
    }
}

impl<K: Ord, V> Drop for Node<K, V> {
    fn drop(&mut self) {
        // BzTree drops node only when no threads can access it:
        // node removed from tree structure and all threads which
        // perform tree scan already completed.
        let guard = unsafe { crossbeam_epoch::unprotected() };
        for entry in self.data_block.drain(..).rev() {
            let metadata: Metadata = entry.metadata.read(guard).into();
            unsafe {
                if metadata.is_visible() || metadata.is_deleted() {
                    drop(entry.key.assume_init());
                }
                if metadata.is_visible() {
                    drop(entry.value.assume_init());
                }
            }
        }
    }
}

unsafe impl<K: Ord, V> Sync for Node<K, V> {}
unsafe impl<K: Ord, V> Send for Node<K, V> {}

impl<K, V> Display for Node<K, V>
where
    K: Clone + Ord + Display,
    V: Display + Send + Sync,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let guard = crossbeam_epoch::pin();
        writeln!(
            f,
            "length: {}, total capacity/remains: {}/{}, status word: {}",
            self.exact_len(&guard),
            self.capacity(),
            self.remaining_capacity(&guard),
            self.status_word.read(&guard)
        )
        .unwrap();

        for (k, v) in self.iter(&crossbeam_epoch::pin()) {
            writeln!(f, "key: {} | value: {}", k, v).unwrap();
        }
        Ok(())
    }
}

pub struct EntryWithLoc<'a, K, V> {
    pub key: &'a K,
    pub value: &'a V,
    pub location: usize,
}

struct Entry<K, V> {
    metadata: U64Pointer,
    key: MaybeUninit<K>,
    // Manually drop because value can be references by several nodes at time(during split/merge)
    value: MaybeUninit<V>,
}

impl<K, V> Entry<K, V> {
    /// Read the key part of entry.
    /// *Warning*: before reading the key reference, caller should ensure that entry is valid.
    /// Otherwise, key can points to uninitialized memory.
    #[inline(always)]
    unsafe fn key(&self) -> &K {
        &*self.key.as_ptr()
    }

    /// Read the value part of entry.
    /// *Warning*: before reading the value reference, caller should ensure that entry is valid.
    /// Otherwise, value can points to uninitialized memory.
    #[inline(always)]
    unsafe fn value(&self) -> &V {
        &*self.value.as_ptr()
    }

    /// Read the value part of entry.
    /// *Warning*: before reading the value reference, caller should ensure that entry is valid.
    /// Otherwise, value can points to uninitialized memory.
    #[inline(always)]
    unsafe fn value_mut(&mut self) -> &mut V {
        &mut *self.value.as_mut_ptr()
    }

    /// Defer drop of value inside this [`KeyValue`] after upsert/delete.
    /// Key part of KV dropped at same time with drop of node(key part still used
    /// by node even after KV removal).
    #[inline(always)]
    unsafe fn defer_value_drop(&mut self, guard: &Guard) {
        // Move out value from KeyValue structure into heap and later drop it:
        //
        // Suppose we have 2 concurrent operations on same node:
        // - upsert or remove some key
        // - merged or split of node
        // Upsert/remove operations return reference to replaced/removed value.
        // Memory used by merged/split node must be deallocated strictly after releasing of
        // removed/replaced value.
        // For instance, one thread(T1) execute upsert for some key. In response, it gets
        // previous value fo same key. At this moment, tree schedule drop of replaced value(by
        // using code of this method). T1 continues with some other actions and
        // `guard` not dropped yet.
        // At the same time, another thread(T2) execute delete which try to remove key from same
        // node. Suppose that this remove cause node to become underutilized and consequently
        // this node was merged. After merge completed, T2 drops `guard` and this cause deallocation
        // of merged node.
        // Meanwhile, T1 also complete it execution and drops it's `guard`. This will cause a drop
        // of value which was replaced by upsert. This drop cause a process crash because node which
        // holds a value already deallocated and node memory released to system.
        // Epoch based reclamation can't ensure proper ordering of memory deallocation.
        // Each thread has it's own 'cleanup' queue and there is no ordering between
        // queue items of different threads. That's why replaced/removed values should be moved
        // out from node.

        // safety: value can be safely copied because this method called only
        // when node declares this KV as not accessible anymore.
        let moved_val = ptr::read(&self.value);
        let heap_val = Box::new(moved_val.assume_init());
        guard.defer_unchecked(move || drop(heap_val))
    }
}

pub enum MergeMode<K: Ord, V> {
    /// Merge completed successfully. Returns new node which contains
    /// elements of both source nodes.
    NewNode(Node<K, V>),
    /// Can't merge nodes because not enough space
    MergeFailed,
}

pub enum SplitMode<K: Ord, V> {
    /// Overflow node was split into 2 nodes
    Split(Node<K, V>, Node<K, V>),
    /// Node overflow happened because of too many updates
    /// which eat all space inside node. Split of such node
    /// produce compacted node which is snapshot of latest
    /// values of each key inside it(e.g, we cleanup space
    /// consumed by updates which is not visible anymore
    /// because of more new updates to same key).
    Compact(Node<K, V>),
}

pub struct ReservedEntry<K, V> {
    key: K,
    value: V,
    /// Reserved metadata entry index in unsorted space
    index: usize,
    /// Index of entry inside KV block which value will be overwritten by upsert
    overwritten_entry: Option<usize>,
    /// Status word before reservation phase starts
    prev_status_word: StatusWord,
    /// Possible key duplicate, check unsorted space entry in node before complete insert
    await_reserved_entries: bool,
}

/// Insertion error type which return value which cannot be
/// inserted into node(this is used for insertion retries)
#[derive(Debug, Copy, Clone)]
pub enum InsertError<V> {
    Split(V),
    Retry(V),
    NodeFrozen(V),
    /// Duplicate key found at index inside KV block
    DuplicateKey,
}

#[derive(Debug, Copy, Clone)]
pub enum DeleteError {
    KeyNotFound,
    Retry,
}

#[derive(Debug, Copy, Clone)]
pub enum SearchError {
    ReservedEntryFound,
    KeyNotFound,
}

#[derive(Debug, Copy, Clone)]
pub enum NodeEdge {
    Left,
    Right,
}

#[cfg(test)]
mod tests {

    use crate::node::{DeleteError, InsertError, MergeMode};
    use crate::node::{Node, SplitMode};
    use rand::{thread_rng, Rng};
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::hash::Hash;
    use std::ops::RangeBounds;

    fn create_str_node(max_elements: u16) -> Node<String, u64> {
        Node::with_capacity(max_elements)
    }

    fn create_wrapped_u16_node(max_elements: u16) -> Node<UInt16Buf, u16> {
        Node::with_capacity(max_elements)
    }

    #[test]
    fn insert_and_search() {
        let elements = 500;
        let node = create_str_node(elements);
        for i in 0..elements {
            let guard = crossbeam_epoch::pin();
            let key = i.to_string();
            let value = thread_rng().gen::<u64>();
            node.insert(key.clone(), value, &guard).unwrap();

            let guard = crossbeam_epoch::pin();
            let res = node.get(&key, &guard);
            let (_, found_val, _, _) = res.unwrap_or_else(|| panic!("{:?} not found", &key));
            assert_eq!(found_val, &value);
        }
    }

    #[test]
    fn upsert_and_search() {
        let elements = 500;
        let node = create_str_node(elements);
        for i in 0..elements {
            let guard = crossbeam_epoch::pin();
            let key = i.to_string();
            let value = thread_rng().gen::<u64>();
            node.upsert(key.clone(), value, &guard).unwrap();

            let guard = crossbeam_epoch::pin();
            let res = node.get(&key, &guard);
            let (_, found_val, _, _) = res.unwrap_or_else(|| panic!("{:?} not found", &key));
            assert_eq!(found_val, &value);
        }
    }

    #[test]
    fn insert_existing_key() {
        let guard = crossbeam_epoch::pin();
        let node = create_str_node(2);
        let (key, value) = rand_kv();
        node.insert(key.clone(), value, &guard).unwrap();
        let result = node.insert(key, value, &guard);
        assert!(matches!(result, Err(InsertError::DuplicateKey)));
    }

    #[test]
    fn update_key() {
        let max_elements = 500;
        let node = create_str_node(max_elements * 2); // upsert create new KV
        for i in 0..max_elements {
            let guard = crossbeam_epoch::pin();
            let key = i.to_string();
            let value = thread_rng().gen::<u64>();
            node.insert(key.clone(), value, &guard).unwrap();

            let new_val = thread_rng().gen::<u64>();
            node.upsert(key.clone(), new_val, &guard).unwrap();

            let guard = crossbeam_epoch::pin();
            let res = node.get(&key, &guard);
            let (_, found_val, _, _) = res.unwrap_or_else(|| panic!("{:?} not found", &key));
            assert_eq!(found_val, &new_val);
        }
    }

    #[test]
    fn delete_keys() {
        let elements = 128u16;
        let node = create_wrapped_u16_node(elements * 2);

        for i in 0..elements {
            let guard = crossbeam_epoch::pin();
            let (key, value) = (UInt16Buf(i), i);
            node.insert(key, value, &guard)
                .expect("No space left for inserts!");
        }

        for i in 0..elements {
            let key = UInt16Buf::new(i as u16);

            let guard = crossbeam_epoch::pin();
            assert_eq!(
                *node.delete(&key, &guard).expect("Key cannot be removed!"),
                i
            );

            let res = node.get(&key, &guard);
            assert!(matches!(res, None));
        }
    }

    #[test]
    fn delete_non_existing_keys() {
        let max_elements = 500;
        let node = create_wrapped_u16_node(max_elements);

        for i in 0..max_elements {
            let guard = crossbeam_epoch::pin();
            let (key, value) = (UInt16Buf(i), i + 1);
            node.insert(key, value, &guard).unwrap();
        }

        let key = UInt16Buf(max_elements + 1);
        let guard = crossbeam_epoch::pin();
        let res = node.delete(&key, &guard);
        assert!(matches!(res, Err(DeleteError::KeyNotFound)));
    }

    #[test]
    fn split_node() {
        for _ in 0..100 {
            let max_elements = thread_rng().gen_range(1..100);
            let node = create_str_node(max_elements);
            let mut elem_count = 0;
            let guard = crossbeam_epoch::pin();
            for _ in 0..max_elements {
                let (key, value) = rand_kv();
                if thread_rng().gen_bool(0.5) {
                    node.insert(key, value, &guard).unwrap();
                } else {
                    node.upsert(key, value, &guard).unwrap();
                }
                elem_count += 1;
            }

            node.try_froze(&guard);

            match node.split_leaf(&crossbeam_epoch::pin()) {
                SplitMode::Split(left, right) => {
                    let merged: Vec<(&String, &u64)> =
                        left.iter(&guard).chain(right.iter(&guard)).collect();
                    for (i, (k, v)) in node.iter(&guard).enumerate() {
                        assert_eq!(merged[i].0, k);
                        assert_eq!(merged[i].1, v);
                    }
                    assert_eq!(merged.len(), elem_count);
                }
                SplitMode::Compact(_) => panic!("Node must be split, not compacted"),
            }
        }
    }

    #[test]
    fn compact_node() {
        for _ in 0..100 {
            let elements = thread_rng().gen_range(1..100);
            let node = create_str_node(elements * 2);
            let guard = crossbeam_epoch::pin();
            for i in 0..elements {
                let key = i.to_string();
                let value = i as u64;
                node.insert(key, value, &guard).unwrap();
            }

            let kvs: Vec<(String, u64)> = node.iter(&guard).map(|(k, v)| (k.clone(), *v)).collect();
            for (k, v) in kvs {
                node.upsert(k, v + 1, &guard).unwrap();
            }

            node.try_froze(&guard);

            match node.split_leaf(&crossbeam_epoch::pin()) {
                SplitMode::Split(_, _) => {
                    panic!("Node must compacted, not split");
                }
                SplitMode::Compact(compacted) => {
                    assert_eq!(compacted.exact_len(&guard), elements as usize)
                }
            }
        }
    }

    #[test]
    fn split_empty_node() {
        let node = create_str_node(0);

        let guard = &crossbeam_epoch::pin();
        node.try_froze(guard);
        node.split_leaf(guard);
    }

    #[test]
    fn merge_nodes() {
        for _ in 0..100 {
            let mut set = HashSet::new();
            for _ in 0..thread_rng().gen_range(0..100) {
                set.insert((thread_rng().gen::<u32>(), thread_rng().gen::<u32>()));
            }

            let mut expected_elems: Vec<(&u32, &u32)> = Vec::new();
            for (k, v) in set.iter() {
                expected_elems.push((k, v));
            }

            expected_elems.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
            let split_point = if expected_elems.is_empty() {
                0
            } else {
                thread_rng().gen_range(0..expected_elems.len())
            };
            let (vec1, vec2) = expected_elems.split_at(split_point);
            let node1 = Node::new_readonly(vec1.iter().map(|(k, v)| (**k, **v)).collect());
            let node2 = Node::new_readonly(vec2.iter().map(|(k, v)| (**k, **v)).collect());

            let guard = crossbeam_epoch::pin();
            node1.try_froze(&guard);
            node2.try_froze(&guard);

            let merged_node = if let MergeMode::NewNode(node) = node1.merge_with_leaf(
                &node2,
                node1.estimated_len(&guard) + node2.estimated_len(&guard),
                &guard,
            ) {
                node
            } else {
                panic!("Unexpected merge mode");
            };
            let res_vec: Vec<(&u32, &u32)> = merged_node.iter(&guard).collect();
            assert_eq!(merged_node.exact_len(&guard), res_vec.len());
            assert_eq!(expected_elems.len(), res_vec.len());
            for i in 0..expected_elems.len() {
                assert_eq!(expected_elems[i], res_vec[i]);
            }
        }
    }

    #[test]
    fn try_merge_when_no_space_left() {
        for _ in 0..thread_rng().gen_range(100..300) {
            let mut set = HashSet::new();
            for _ in 0..thread_rng().gen_range(1..200) {
                set.insert((thread_rng().gen::<u32>(), thread_rng().gen::<u32>()));
            }
            let mut vec: Vec<(u32, u32)> = Vec::new();
            for (k, v) in set.iter() {
                vec.push((*k, *v));
            }

            vec.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
            let (vec1, vec2) = vec.split_at(vec.len() / 2);
            let node1: Node<u32, u32> = Node::new_readonly(vec1.to_vec());
            let node2 = Node::new_readonly(vec2.to_vec());

            let guard = crossbeam_epoch::pin();
            node1.try_froze(&guard);
            node2.try_froze(&guard);

            let cap = thread_rng()
                .gen_range(0..node1.estimated_len(&guard) + node2.estimated_len(&guard));
            assert!(matches!(
                node1.merge_with_leaf(&node2, cap, &guard),
                MergeMode::MergeFailed
            ));
        }
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn scan_on_unsorted_node() {
        let node = Node::with_capacity(150);

        let min = 1;
        let max = 10;
        let guard = crossbeam_epoch::pin();
        for i in (min..max).rev() {
            let (key, value) = (i, i);
            node.insert(key, value + 1, &guard)
                .expect("No space left for inserts!");
            node.delete(&key, &guard).unwrap();
            node.upsert(key, value, &guard)
                .expect("No space left for upsert!");
        }

        // generate keys which should be omitted by scan
        for i in max..15 {
            let (key, value) = (i, i);
            node.insert(key, value, &guard)
                .expect("No space left for inserts!");
            node.delete(&key, &guard).unwrap();
        }

        let tree_elements: Vec<(&i32, &i32)> = node.iter(&guard).collect();
        let expected_vec: Vec<(i32, i32)> = (min..max).map(|i| (i, i)).collect();
        assert_eq!(tree_elements.len(), expected_vec.len());
        // test order of scanned elements
        for (idx, kv) in expected_vec.iter().enumerate() {
            assert_eq!(tree_elements[idx].0, &kv.0);
            assert_eq!(tree_elements[idx].1, &kv.1);
        }

        check_scan(&node, 0..=1, 1);
        check_scan(&node, 1..=3, 3);
        check_scan(&node, 4..=4, 1);
        check_scan(&node, 5..=9, 5);
        check_scan(&node, 10..=15, 0);

        check_scan(&node, 0..1, 0);
        check_scan(&node, 1..3, 2);
        check_scan(&node, 5..9, 4);
        check_scan(&node, 10..15, 0);

        check_scan(&node, ..=1, 1);
        check_scan(&node, ..=3, 3);
        check_scan(&node, ..=4, 4);
        check_scan(&node, ..=9, 9);
        check_scan(&node, ..=15, 9);

        let scanner: Vec<(&i32, &i32)> = node.range(15..=10, &guard).collect();
        assert!(scanner.is_empty());
        let scanner: Vec<(&i32, &i32)> = node.range(25.., &guard).collect();
        assert!(scanner.is_empty());
        let scanner: Vec<(&i32, &i32)> = node.range(..0, &guard).collect();
        assert!(scanner.is_empty());
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn scan_on_sorted_node() {
        let mut sorted = Vec::new();
        let min = 1;
        let max = 10;
        for i in min..max {
            sorted.push((i, i));
        }

        let max_elements = (sorted.len() * 5) as u16;
        let elems: Vec<(i32, i32)> = sorted.iter().map(|(k, v)| (*k, *v)).collect();
        let node = Node::init_with_capacity(elems, max_elements);

        let guard = crossbeam_epoch::pin();
        node.insert(max, max + 1, &guard)
            .expect("No space left for inserts!");
        node.delete(&max, &guard).unwrap();
        node.upsert(max, max, &guard)
            .expect("No space left for inserts!");

        // generate keys which should be omitted by scan
        for i in max + 1..15 {
            let guard = crossbeam_epoch::pin();
            let (key, value) = (i, i);
            node.insert(key, value, &guard)
                .expect("No space left for inserts!");
            node.delete(&key, &guard).unwrap();
        }

        let tree_elements: Vec<(&i32, &i32)> = node.range(.., &guard).collect();
        let expected_vec: Vec<(i32, i32)> = (min..=max).map(|i| (i, i)).collect();
        assert_eq!(tree_elements.len(), expected_vec.len());
        // test order of scanned elements
        for (idx, kv) in expected_vec.iter().enumerate() {
            assert_eq!(tree_elements[idx].0, &kv.0);
            assert_eq!(tree_elements[idx].1, &kv.1);
        }

        check_scan(&node, 0..=1, 1);
        check_scan(&node, 1..=3, 3);
        check_scan(&node, 4..=4, 1);
        check_scan(&node, 5..=9, 5);
        check_scan(&node, 11..=15, 0);

        check_scan(&node, 0..1, 0);
        check_scan(&node, 1..3, 2);
        check_scan(&node, 5..9, 4);
        check_scan(&node, 11..15, 0);

        check_scan(&node, ..=1, 1);
        check_scan(&node, ..=3, 3);
        check_scan(&node, ..=4, 4);
        check_scan(&node, ..=9, 9);
        check_scan(&node, ..=15, 10);

        let scanner: Vec<(&i32, &i32)> = node.range(15..=10, &guard).collect();
        assert!(scanner.is_empty());
        let scanner: Vec<(&i32, &i32)> = node.range(25.., &guard).collect();
        assert!(scanner.is_empty());
        let scanner: Vec<(&i32, &i32)> = node.range(..0, &guard).collect();
        assert!(scanner.is_empty());
    }

    #[test]
    fn range() {
        let guard = crossbeam_epoch::pin();
        let vec: Vec<(u32, u32)> = vec![(1, 1), (2, 2), (3, 3)];
        let max_elements = (vec.len() * 4) as u16;
        let elems: Vec<(u32, u32)> = vec.iter().map(|(k, v)| (*k, *v)).collect();
        let node = Node::init_with_capacity(elems, max_elements);
        node.insert(4, 4, &guard).unwrap();
        node.insert(5, 5, &guard).unwrap();
        node.insert(6, 6, &guard).unwrap();
        node.insert(7, 7, &guard).unwrap();

        let actual: Vec<u32> = node.range(1.., &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (1..=7).collect();
        assert_eq!(expected, actual);

        let actual: Vec<u32> = node.range(3.., &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (3..=7).collect();
        assert_eq!(expected, actual);

        let actual: Vec<u32> = node.range(..4, &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (1..=3).collect();
        assert_eq!(expected, actual);

        let actual: Vec<u32> = node.range(..=6, &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (1..=6).collect();
        assert_eq!(expected, actual);

        let actual: Vec<u32> = node.range(2..=5, &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (2..=5).collect();
        assert_eq!(expected, actual);

        let actual: Vec<u32> = node.range(2..5, &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (2..5).collect();
        assert_eq!(expected, actual);

        node.delete(&1, &guard).unwrap();
        let actual: Vec<u32> = node.range(0.., &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (2..=7).collect();
        assert_eq!(expected, actual);

        node.delete(&5, &guard).unwrap();
        let actual: Vec<u32> = node.range(..=5, &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (2..5).collect();
        assert_eq!(expected, actual);

        node.delete(&7, &guard).unwrap();
        let actual: Vec<u32> = node.range(5.., &guard).map(|(k, _)| *k).collect();
        let expected: Vec<u32> = (6..7).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn iter() {
        let guard = crossbeam_epoch::pin();
        let vec: Vec<(u32, u32)> = vec![(1, 1), (2, 2), (3, 3)];
        let max_elements = (vec.len() * 4) as u16;
        let elems: Vec<(u32, u32)> = vec.iter().map(|(k, v)| (*k, *v)).collect();
        let node = Node::init_with_capacity(elems, max_elements);

        node.insert(4, 4, &guard).unwrap();
        node.insert(5, 6, &guard).unwrap();
        node.upsert(5, 5, &guard).unwrap();
        node.insert(6, 6, &guard).unwrap();
        node.delete(&6, &guard).unwrap();
        node.upsert(1, 2, &guard).unwrap();
        node.upsert(1, 0, &guard).unwrap();
        node.delete(&2, &guard).unwrap();

        let expected_vec = vec![(1, 0), (3, 3), (4, 4), (5, 5)];
        let mut expected = expected_vec.iter();
        let mut iter = node.iter(&guard);
        assert_eq!((&iter).len(), (&expected).len());
        let mut i = 0;
        while let Some((k, v)) = (&mut iter).next() {
            let nxt = expected.next().unwrap();
            assert_eq!(*k, nxt.0);
            assert_eq!(*v, nxt.1);
            i += 1;
            assert_eq!(iter.len(), expected_vec.len() - i);
        }
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn iter_reverted() {
        let guard = crossbeam_epoch::pin();
        let vec: Vec<(u32, u32)> = vec![(1, 1), (2, 2), (3, 3)];
        let max_elements = (vec.len() * 4) as u16;
        let elems: Vec<(u32, u32)> = vec.iter().map(|(k, v)| (*k, *v)).collect();
        let node = Node::init_with_capacity(elems, max_elements);

        node.insert(4, 4, &guard).unwrap();
        node.insert(5, 5, &guard).unwrap();
        node.insert(6, 6, &guard).unwrap();
        node.delete(&6, &guard).unwrap();
        node.upsert(1, 0, &guard).unwrap();
        node.delete(&2, &guard).unwrap();
        let mut expected_vec: Vec<(u32, u32)> = vec![(1, 0), (3, 3), (4, 4), (5, 5)];
        expected_vec.reverse();

        let mut expected = expected_vec.iter();
        let mut iter = node.iter(&guard).rev();
        assert_eq!((&iter).len(), (&expected).len());
        let mut i = 0;
        while let Some((k, v)) = (&mut iter).next() {
            let nxt = expected.next().unwrap();
            assert_eq!(*k, nxt.0);
            assert_eq!(*v, nxt.1);
            i += 1;
            assert_eq!(iter.len(), expected_vec.len() - i);
        }
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn iter_on_empty_sorted_space() {
        let guard = crossbeam_epoch::pin();
        let node: Node<u32, u32> = Node::with_capacity(10);
        node.insert(1, 1, &guard).unwrap();
        node.insert(2, 2, &guard).unwrap();
        node.insert(3, 3, &guard).unwrap();
        node.delete(&3, &guard).unwrap();
        node.insert(3, 3, &guard).unwrap();
        node.insert(4, 4, &guard).unwrap();
        node.insert(5, 6, &guard).unwrap();
        node.upsert(5, 5, &guard).unwrap();
        node.insert(6, 6, &guard).unwrap();
        node.upsert(1, 2, &guard).unwrap();
        node.delete(&6, &guard).unwrap();
        node.upsert(1, 0, &guard).unwrap();
        node.delete(&2, &guard).unwrap();

        let expected: Vec<(u32, u32)> = vec![(1, 0), (3, 3), (4, 4), (5, 5)];
        let mut count = 0;
        for (i, (k, v)) in node.iter(&guard).enumerate() {
            assert_eq!(*k, expected[i].0);
            assert_eq!(*v, expected[i].1);
            count += 1;
        }
        assert_eq!(count, expected.len());
    }

    #[test]
    fn iter_on_empty_unsorted_space() {
        let guard = crossbeam_epoch::pin();
        let expected: Vec<(u32, u32)> = vec![(1, 0), (3, 3), (4, 4), (5, 5)];
        let node: Node<u32, u32> = Node::new_readonly(expected.clone());

        let mut count = 0;
        for (i, (k, v)) in node.iter(&guard).enumerate() {
            assert_eq!(*k, expected[i].0);
            assert_eq!(*v, expected[i].1);
            count += 1;
        }
        assert_eq!(count, expected.len());
    }

    #[test]
    fn peek_next_back() {
        let guard = crossbeam_epoch::pin();
        let vec: Vec<(u32, u32)> = vec![(1, 1), (2, 2), (3, 3)]
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        let node = Node::init_with_capacity(vec, 4);

        node.insert(4, 4, &guard).unwrap();

        let mut iter = node.iter(&guard);
        assert!(matches!(iter.peek_next_back(), Some((k, _)) if *k == 4));
        iter.next_back();
        assert!(matches!(iter.peek_next_back(), Some((k, _)) if *k == 3));
        iter.next_back();
        assert!(matches!(iter.peek_next_back(), Some((k, _)) if *k == 2));
        iter.next_back();
        assert!(matches!(iter.peek_next_back(), Some((k, _)) if *k == 1));
        iter.next_back();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn max() {
        let guard = crossbeam_epoch::pin();
        let sorted: Vec<(u32, u32)> = vec![(1, 1), (3, 3), (4, 4), (5, 5)];
        let node: Node<u32, u32> =
            Node::init_with_capacity(sorted.clone(), (sorted.len() * 2) as u16);

        assert!(matches!(node.last_kv(&guard), Some((k, _)) if *k == 5 ));

        node.insert(6, 6, &guard).unwrap();
        assert!(matches!(node.last_kv(&guard), Some((k, _)) if *k == 6 ));

        node.delete(&6, &guard).unwrap();
        assert!(matches!(node.last_kv(&guard), Some((k, _)) if *k == 5 ));
        node.delete(&5, &guard).unwrap();
        assert!(matches!(node.last_kv(&guard), Some((k, _)) if *k == 4 ));
    }

    fn rand_kv() -> (String, u64) {
        let key = thread_rng().gen::<u64>();
        let value = thread_rng().gen::<u64>();
        (key.to_string(), value)
    }

    fn check_scan<R, T>(node: &Node<T, T>, range: R, expected_size: usize)
    where
        R: RangeBounds<T> + Clone,
        T: PartialOrd<T> + Ord + Debug,
    {
        let guard = crossbeam_epoch::pin();
        let scanner: Vec<(&T, &T)> = node.range(range.clone(), &guard).collect();
        assert_eq!(scanner.len(), expected_size);
        for (k, v) in scanner {
            assert!(range.clone().contains(k));
            assert_eq!(k, v);
        }
    }

    #[cfg(test)]
    mod metadata_tests {
        use crate::node::metadata::Metadata;

        #[test]
        fn create_reserved_metadata() {
            let metadata = Metadata::reserved();
            assert!(!metadata.is_deleted());
            assert!(!metadata.is_visible());
            assert!(metadata.is_reserved());
        }

        #[test]
        fn create_visible_metadata() {
            let metadata = Metadata::visible();
            assert!(!metadata.is_deleted());
            assert!(!metadata.is_reserved());
            assert!(metadata.is_visible());
        }

        #[test]
        fn create_deleted_metadata() {
            let metadata = Metadata::deleted();
            assert!(metadata.is_deleted());
            assert!(!metadata.is_reserved());
            assert!(!metadata.is_visible());
        }
    }

    #[derive(Clone, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
    struct UInt16Buf(u16);

    impl UInt16Buf {
        fn new(val: u16) -> UInt16Buf {
            UInt16Buf(val)
        }
    }
}
