use crate::node::metadata::Metadata;
use crate::{Node, StatusWord};
use crossbeam_epoch::Guard;
use std::borrow::Borrow;
use std::collections::{BTreeSet, Bound};
use std::ops::RangeBounds;

pub struct NodeScanner<'a, K: Ord, V> {
    node: &'a Node<K, V>,
    // BzTree node can allocate maximum u16::MAX records,
    // use u16 instead of usize to reduce size of scanner
    kv_indexes: Vec<u16>,
    fwd_idx: u16,
    rev_idx: u16,
}

impl<'a, K, V> NodeScanner<'a, K, V>
where
    K: Ord,
{
    pub fn new<Q>(
        status_word: &StatusWord,
        node: &'a Node<K, V>,
        key_range: impl RangeBounds<Q>,
        guard: &'a Guard,
    ) -> Self
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let mut kvs: Vec<u16> = Vec::with_capacity(status_word.reserved_records() as usize);

        if node.readonly || status_word.reserved_records() as usize == node.sorted_len {
            if node.sorted_len > 0 {
                let sorted_block = &node.data_block[0..node.sorted_len];
                let start_idx = match key_range.start_bound() {
                    Bound::Excluded(key) => sorted_block
                        .binary_search_by(|entry| unsafe { entry.key() }.borrow().cmp(key))
                        .map_or_else(|index| index, |index| index + 1),
                    Bound::Included(key) => sorted_block
                        .binary_search_by(|entry| unsafe { entry.key() }.borrow().cmp(key))
                        .map_or_else(|index| index, |index| index),
                    Bound::Unbounded => 0,
                };

                let end_idx = match key_range.end_bound() {
                    Bound::Excluded(key) => sorted_block
                        .binary_search_by(|entry| unsafe { entry.key() }.borrow().cmp(key))
                        .map_or_else(|index| index, |index| index),
                    Bound::Included(key) => sorted_block
                        .binary_search_by(|entry| unsafe { entry.key() }.borrow().cmp(key))
                        .map_or_else(|index| index, |index| index + 1),
                    Bound::Unbounded => node.sorted_len,
                };

                if !node.readonly {
                    (start_idx..end_idx).for_each(|i| {
                        let metadata: Metadata = node.data_block[i].metadata.read(guard).into();
                        if metadata.is_visible() {
                            kvs.push(i as u16);
                        }
                    });
                } else {
                    (start_idx..end_idx).for_each(|i| kvs.push(i as u16));
                }
            }
        } else {
            let mut scanned_keys = BTreeSet::new();
            // use reversed iterator because unsorted part at end of KV block has most recent values
            for i in (0..status_word.reserved_records() as usize).rev() {
                let entry = &node.data_block[i];
                let metadata = loop {
                    let metadata: Metadata = entry.metadata.read(guard).into();
                    if !metadata.is_reserved() {
                        break metadata;
                    }
                    // someone try to add entry at same time, continue check same entry
                    // until reserved entry will become valid
                };

                // if is first time when we see this key, we return it
                // otherwise, most recent version(even deletion) of key
                // already seen by scanner and older versions must be ignored.
                if metadata.visible_or_deleted() {
                    let key = unsafe { node.data_block[i].key() };
                    if key_range.contains(key.borrow())
                        && scanned_keys.insert(key)
                        && metadata.is_visible()
                    {
                        kvs.push(i as u16);
                    }
                }
            }

            kvs.sort_by(|index1, index2| {
                let key1 = unsafe { node.data_block[*index1 as usize].key() };
                let key2 = unsafe { node.data_block[*index2 as usize].key() };
                key1.cmp(key2)
            });
        }

        if kvs.is_empty() {
            return NodeScanner {
                // move 'forward' index in front of 'reversed' to simulate empty scanner
                fwd_idx: 1,
                rev_idx: 0,
                kv_indexes: kvs,
                node,
            };
        }

        NodeScanner {
            fwd_idx: 0,
            rev_idx: (kvs.len() - 1) as u16,
            kv_indexes: kvs,
            node,
        }
    }

    pub fn peek_next(&mut self) -> Option<(&'a K, &'a V)> {
        if self.rev_idx >= self.fwd_idx {
            let index = self.kv_indexes[self.fwd_idx as usize] as usize;
            let kv = &self.node.data_block[index];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        }
    }

    pub fn peek_next_back(&mut self) -> Option<(&'a K, &'a V)> {
        if self.rev_idx >= self.fwd_idx {
            let index = self.kv_indexes[self.rev_idx as usize] as usize;
            let kv = &self.node.data_block[index];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        }
    }
}

impl<'a, K: Ord, V> Iterator for NodeScanner<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rev_idx >= self.fwd_idx {
            let index = self.kv_indexes[self.fwd_idx as usize] as usize;
            self.fwd_idx += 1;
            let kv = &self.node.data_block[index];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = if self.rev_idx < self.fwd_idx {
            0
        } else {
            (self.rev_idx - self.fwd_idx + 1) as usize
        };
        (size, Some(size))
    }
}

impl<'a, K: Ord, V> DoubleEndedIterator for NodeScanner<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.rev_idx >= self.fwd_idx {
            let index = self.kv_indexes[self.rev_idx as usize] as usize;
            if self.rev_idx > 0 {
                self.rev_idx -= 1;
            } else {
                // reversed iteration reaches 0 element,
                // to break iteration, we move 'forward' index in front of 'reversed'
                // because we can't set -1 to rev_idx because it unsigned.
                self.fwd_idx = self.rev_idx + 1;
            }
            let kv = &self.node.data_block[index];
            Some(unsafe { (kv.key(), kv.value()) })
        } else {
            None
        }
    }
}

impl<'a, K: Ord, V> ExactSizeIterator for NodeScanner<'a, K, V> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}
