use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::RangeBounds;
use std::time::Instant;

#[derive(Copy, Clone, Debug)]
struct Op<V> {
    value: Option<V>,
    start_time: Instant,
    end_time: Instant,
}

#[derive(Clone)]
pub struct Ops<K, V> {
    key_map: HashMap<K, Vec<Op<V>>>,
}

pub struct History<K, V> {
    candidate_map: HashMap<K, Vec<Option<V>>>,
    full_history: HashMap<K, Vec<Op<V>>>,
}

impl<K, V> History<K, V>
where
    K: Eq + Hash + Clone + Ord + Debug,
    V: Eq + Debug + Clone,
{
    pub fn from(ops: &Ops<K, V>) -> Self {
        Self::based_on(ops.clone())
    }

    pub fn based_on(mut ops: Ops<K, V>) -> Self {
        let mut candidate_map = HashMap::new();
        for (key, ops) in ops.key_map.iter_mut() {
            ops.sort_by(|op1, op2| op1.end_time.cmp(&op2.end_time));
            let latest_ended_op = ops.last().unwrap();
            let range = latest_ended_op.start_time..=latest_ended_op.end_time;
            for op in ops {
                if range.contains(&op.end_time) {
                    candidate_map
                        .entry(key.clone())
                        .or_insert_with(Vec::new)
                        .push(op.value.clone());
                }
            }
        }

        // sort for scanner checks
        for ops in ops.key_map.values_mut() {
            ops.sort_unstable_by(|op1, op2| op1.start_time.cmp(&op2.start_time));
        }

        History {
            candidate_map,
            full_history: ops.key_map,
        }
    }

    pub fn run_check<'h: 'g, 'g, F>(&'h self, observed_val_by_key: F)
    where
        F: Fn(&'g K) -> Option<&'g V>,
    {
        for (key, possible_values) in &self.candidate_map {
            let val = observed_val_by_key(key);
            possible_values
                .iter()
                .filter(|pval| match pval {
                    Some(possible_val) => {
                        matches!(val, Some(actual_val) if possible_val == actual_val)
                    }
                    None => {
                        matches!(val, None)
                    }
                })
                .last()
                .unwrap_or_else(|| {
                    panic!(
                        "Key has value {:?} which is not expected by execution history: {:?}. \
                        Full history of changes(sorted by time) for key '{:?}': {:?}",
                        val,
                        possible_values,
                        key,
                        self.full_history.get(key)
                    )
                });
        }
    }

    pub fn run_scanner_check<'h, F, Range>(&'h self, scanner: F)
    where
        Range: RangeBounds<K> + 'h,
        F: Fn() -> (Range, bool, Box<dyn Iterator<Item = (&'h K, &'h V)> + 'h>),
    {
        let (range, is_reversed, scanner) = scanner();
        let mut kvs = BTreeMap::new();
        self.full_history.iter().for_each(|(k, v)| {
            if range.contains(&k) {
                if let Some(v) = v.iter().last() {
                    if let Some(v) = &v.value {
                        kvs.insert(k, v);
                    }
                }
            }
        });

        let merged: Box<dyn Iterator<Item = (&&K, &&V)>> = if is_reversed {
            Box::new(kvs.iter().rev())
        } else {
            Box::new(kvs.iter())
        };

        let expected: Vec<(&&K, &&V)> = merged.collect();
        let scanned: Vec<(&K, &V)> = scanner.collect();
        expected
            .iter()
            .zip(scanned.iter())
            .all(|((k1, v1), (k2, v2))| *k1 == k2 && *v1 == v2);
        assert_eq!(expected.len(), scanned.len());
    }
}

impl<K, V> Ops<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    pub fn new() -> Self {
        Ops {
            key_map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V, start: Instant) {
        self.key_map.entry(key).or_insert_with(Vec::new).push(Op {
            value: Some(value),
            start_time: start,
            end_time: Instant::now(),
        });
    }

    pub fn delete(&mut self, key: K, start: Instant) {
        self.key_map.entry(key).or_insert_with(Vec::new).push(Op {
            value: None,
            start_time: start,
            end_time: Instant::now(),
        });
    }

    pub fn merge(mut self, other: Self) -> Self {
        for (key, mut op) in other.key_map {
            self.key_map
                .entry(key.clone())
                .or_insert_with(Vec::new)
                .append(&mut op);
        }
        self
    }
}
