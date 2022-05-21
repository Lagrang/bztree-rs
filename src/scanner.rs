use crate::node::NodeScanner;
use crate::{InterimNode, Key, LeafNode, NodePointer};
use crossbeam_epoch::Guard;
use mwcas::HeapPointer;
use std::borrow::Borrow;
use std::ops::{Bound, RangeBounds};
use std::option::Option::Some;

pub struct Scanner<'g, K: Ord, V, Range> {
    forward: Iter<'g, K, V>,
    reversed: Iter<'g, K, V>,
    range: Range,
    guard: &'g Guard,
}

struct Iter<'g, K: Ord, V> {
    interims: Vec<NodeScanner<'g, Key<K>, HeapPointer<NodePointer<K, V>>>>,
    iter: Option<NodeScanner<'g, K, V>>,
    last_key: Option<&'g K>,
}

impl<'g, K: Ord, V> Iter<'g, K, V> {
    fn clear(&mut self) {
        self.iter = None;
        self.interims.clear();
    }
}

impl<'g, K: Ord, V, Range> Scanner<'g, K, V, Range> {
    pub fn from_leaf_root<Q>(root: &'g LeafNode<K, V>, key_range: Range, guard: &'g Guard) -> Self
    where
        Range: RangeBounds<Q> + Clone + 'g,
        K: Clone + Borrow<Q> + Ord,
        V: Send + Sync,
        Q: Ord + 'g,
    {
        Scanner {
            forward: Iter {
                interims: Vec::new(),
                iter: Some(root.range(key_range.clone(), guard)),
                last_key: None,
            },
            reversed: Iter {
                interims: Vec::new(),
                iter: Some(root.range(key_range.clone(), guard)),
                last_key: None,
            },
            range: key_range,
            guard,
        }
    }

    pub(crate) fn from_non_leaf_root(
        root: &'g InterimNode<K, V>,
        key_range: Range,
        guard: &'g Guard,
    ) -> Self
    where
        Range: RangeBounds<K> + Clone + 'g,
        K: Ord + Clone,
        V: Send + Sync,
    {
        // interim root node always contains +Inf key which contains nodes with keys greater than
        // any other in tree. We should always scan node corresponds to +Inf key.
        let wide_range = KeyRange::with_unbounded_end(key_range.clone());
        Scanner {
            forward: Iter {
                interims: vec![root.range(wide_range.clone(), guard)],
                iter: None,
                last_key: None,
            },
            reversed: Iter {
                interims: vec![root.range(wide_range, guard)],
                iter: None,
                last_key: None,
            },
            range: key_range,
            guard,
        }
    }

    fn next_leaf(&mut self) -> Option<NodeScanner<'g, K, V>>
    where
        Range: RangeBounds<K> + Clone + 'g,
        K: Ord + Clone,
    {
        let range_check = KeyRange::from(self.range.clone());
        while let Some(nodes) = self.forward.interims.last_mut() {
            if let Some((key, node_ptr)) = nodes.next() {
                if !range_check.contains(key) {
                    // current interim node corresponds to data range
                    // which is already partially outside of requested range.
                    // We should stop scan of current interim and remaining nodes at
                    // upper levels.
                    self.forward.interims.clear();
                }

                match node_ptr.read(self.guard) {
                    NodePointer::Interim(node) => {
                        // interim node should scan all nodes including node those key
                        // is greater or equal to actual range end, to cover +Inf node.
                        // This is why we use `with_unbounded_end`.
                        let range = KeyRange::with_unbounded_end(self.range.clone());
                        let scanner = node.range(range, self.guard);
                        self.forward.interims.push(scanner);
                    }
                    NodePointer::Leaf(node) => {
                        let scanner = node.range(self.range.clone(), self.guard);
                        if scanner.len() == 0 {
                            // some nodes can be temporary empty before they merged with siblings
                            continue;
                        }
                        return Some(scanner);
                    }
                }
            } else {
                // no more child nodes to scan inside interim node, go to next interim node
                self.forward.interims.pop();
            }
        }
        None
    }

    fn next_leaf_rev(&mut self) -> Option<NodeScanner<'g, K, V>>
    where
        Range: RangeBounds<K> + Clone + 'g,
        K: Ord + Clone,
    {
        while let Some(nodes) = self.reversed.interims.last_mut() {
            if let Some((key, node_ptr)) = nodes.next_back() {
                // Check is node outside of required key range.
                // Because initial scanner range covers +Inf node, we can try to skip nodes
                // which surely doesn't contain required keys.
                //
                // Here we compute range of keys which current node contains.
                // Range will start(exclusively) from end key of previous node and end
                // (inclusively) with key of current node.
                // If computed range doesn't contain 'end bound' of requested key range, then
                // this node doesn't contain required keys and can be skipped.
                if let Some((left_bound, _)) = nodes.peek_next_back() {
                    let required_range = KeyRange::from(self.range.clone());
                    let node_range = (Bound::Excluded(left_bound), Bound::Included(key));
                    if !Self::intersects(&node_range, &required_range) {
                        continue;
                    }
                }

                match node_ptr.read(self.guard) {
                    NodePointer::Interim(node) => {
                        // interim node should scan all nodes including node those key
                        // is greater or equal to range end(to cover +Inf node).
                        // This is why we use `with_unbounded_end`.
                        let range = KeyRange::with_unbounded_end(self.range.clone());
                        let scanner = node.range(range, self.guard);
                        self.reversed.interims.push(scanner);
                    }
                    NodePointer::Leaf(node) => {
                        let scanner = node.range(self.range.clone(), self.guard);
                        if scanner.len() == 0 {
                            // some nodes can be temporary empty before they merged with siblings
                            continue;
                        }
                        return Some(scanner);
                    }
                }
            } else {
                // no more child nodes to scan inside interim node, go to next interim node
                self.reversed.interims.pop();
            }
        }
        None
    }

    #[inline(always)]
    fn intersects(range: &(Bound<&Key<K>>, Bound<&Key<K>>), intersects_with: &KeyRange<K>) -> bool
    where
        K: Ord,
    {
        let res = match range.start_bound() {
            Bound::Excluded(key) | Bound::Included(key) => intersects_with.contains(key),
            Bound::Unbounded => match range.end_bound() {
                Bound::Unbounded => true,
                Bound::Excluded(key) | Bound::Included(key) => intersects_with.contains(key),
            },
        };

        if res {
            return true;
        }

        match range.end_bound() {
            Bound::Excluded(key) | Bound::Included(key) => intersects_with.contains(key),
            Bound::Unbounded => true,
        }
    }
}

impl<'g, K, V, Range> Iterator for Scanner<'g, K, V, Range>
where
    Range: RangeBounds<K> + Clone + 'g,
    K: Ord + Clone,
    V: Send + Sync,
{
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(iter) = &mut self.forward.iter {
            let next_kv = iter.next();
            if let Some((k, _)) = next_kv {
                // check is reversed iterator already reach same key
                // if yes, stop iteration
                if let Some(rev_key) = self.reversed.last_key {
                    if k >= rev_key {
                        self.forward.clear();
                        return None;
                    }
                }
                self.forward.last_key = Some(k);
                return next_kv;
            } else {
                self.forward.iter = None;
            }
        }

        // current node iterator exhausted => go to next leaf node
        if let Some(mut next_leaf) = self.next_leaf() {
            let next_kv = next_leaf.next();
            self.forward.iter = Some(next_leaf);
            if let Some((k, _)) = next_kv {
                if let Some(rev_key) = self.reversed.last_key {
                    if k >= rev_key {
                        self.forward.clear();
                        return None;
                    }
                }
                self.forward.last_key = Some(k);
                return next_kv;
            }
        }

        None
    }
}

impl<'g, K, V, Range> DoubleEndedIterator for Scanner<'g, K, V, Range>
where
    Range: RangeBounds<K> + Clone + 'g,
    K: Ord + Clone,
    V: Send + Sync,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(iter) = &mut self.reversed.iter {
            let next_kv = iter.next_back();
            if let Some((k, _)) = next_kv {
                if let Some(fwd_key) = self.forward.last_key {
                    if k <= fwd_key {
                        self.reversed.clear();
                        return None;
                    }
                }
                self.reversed.last_key = Some(k);
                return next_kv;
            } else {
                self.reversed.iter = None;
            }
        }

        if let Some(mut next_leaf) = self.next_leaf_rev() {
            let next_kv = next_leaf.next_back();
            self.reversed.iter = Some(next_leaf);
            if let Some((k, _)) = next_kv {
                if let Some(fwd_key) = self.forward.last_key {
                    if k <= fwd_key {
                        self.reversed.clear();
                        return None;
                    }
                }
                self.reversed.last_key = Some(k);
                return next_kv;
            }
        }

        None
    }
}

#[derive(Clone)]
struct KeyRange<K> {
    start_bound: Bound<Key<K>>,
    end_bound: Bound<Key<K>>,
}

impl<K> KeyRange<K>
where
    K: Clone,
{
    fn from<Range>(range: Range) -> Self
    where
        Range: RangeBounds<K>,
    {
        KeyRange {
            start_bound: match range.start_bound() {
                Bound::Included(key) => Bound::Included(Key::new(key.clone())),
                Bound::Excluded(key) => Bound::Excluded(Key::new(key.clone())),
                Bound::Unbounded => Bound::Unbounded,
            },
            end_bound: match range.end_bound() {
                Bound::Included(key) => Bound::Included(Key::new(key.clone())),
                Bound::Excluded(key) => Bound::Excluded(Key::new(key.clone())),
                Bound::Unbounded => Bound::Unbounded,
            },
        }
    }

    fn with_unbounded_end<Range>(range: Range) -> Self
    where
        Range: RangeBounds<K>,
    {
        KeyRange {
            start_bound: match range.start_bound() {
                Bound::Included(key) => Bound::Included(Key::new(key.clone())),
                Bound::Excluded(key) => Bound::Excluded(Key::new(key.clone())),
                Bound::Unbounded => Bound::Unbounded,
            },
            end_bound: Bound::Unbounded,
        }
    }
}

impl<K> RangeBounds<Key<K>> for KeyRange<K> {
    fn start_bound(&self) -> Bound<&Key<K>> {
        match &self.start_bound {
            Bound::Included(key) => Bound::Included(key),
            Bound::Excluded(key) => Bound::Excluded(key),
            Bound::Unbounded => Bound::Unbounded,
        }
    }

    fn end_bound(&self) -> Bound<&Key<K>> {
        match &self.end_bound {
            Bound::Included(key) => Bound::Included(key),
            Bound::Excluded(key) => Bound::Excluded(key),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}
