// Part of the archive DE port; see the header of archivede.rs and NOTICE.fronts.
//! Duplicate-preserving Pareto archive with optional typed payloads.
use crate::indicators::dominates;
use std::{cmp::Ordering, collections::BTreeMap};

#[derive(Clone)]
struct Member<Y, P> {
    y: Y,
    payload: P,
    sum: f64,
    serial: u64,
}

/// An unbounded strict-minimization archive for any number of objectives.
/// Equal nondominated observations coexist, each with its own payload. For one
/// payload per objective pair and an exact hypervolume use
/// [`ParetoArchive2D`](crate::ParetoArchive2D).
///
/// `Y` can be a fixed-size array, an owned vector, or a borrowed slice.
/// A two-objective archive keeps an x-ordered skyline; higher dimensions use
/// rounded-sum ordering to restrict dominance checks. Python: `fcmaes_rust.ParetoArchiveND`
/// and fast-cma-es's `fcmaes.pareto.ParetoArchiveND`.
pub struct ParetoArchiveND<Y, P = ()> {
    width: usize,
    line: bool,
    members: Vec<Member<Y, P>>,
    // Keep payloads out of the hot higher-dimensional dominance scans.
    rows: Vec<(f64, Y)>,
    payloads: Vec<P>,
    blocks: Option<super::blocks::Blocks<Y, P>>,
    skyline: BTreeMap<u64, Vec<Member<Y, P>>>,
    witness: Option<Y>,
    cache: [f64; 2],
    best_sum: f64,
    next: u64,
    len: usize,
}

impl<Y: AsRef<[f64]> + Clone, P> ParetoArchiveND<Y, P> {
    /// Construct an empty archive for the given objective count.
    ///
    /// # Panics
    /// Panics if the objective count is zero.
    pub fn new(width: usize) -> Self {
        assert!(width > 0, "objective count must be positive");
        Self {
            width,
            line: width == 2,
            members: Vec::new(),
            rows: Vec::new(),
            payloads: Vec::new(),
            blocks: None,
            skyline: BTreeMap::new(),
            witness: None,
            cache: [f64::INFINITY; 2],
            best_sum: f64::INFINITY,
            next: 0,
            len: 0,
        }
    }
    /// Number of retained observations, including duplicates.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether no observations are retained.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Discard all observations.
    pub fn clear(&mut self) {
        *self = Self::new(self.width);
    }
    /// Insert an observation. The payload factory runs only upon admission.
    ///
    /// # Panics
    /// Panics for a non-finite objective or a row of the wrong width, or if the factory panics.
    pub fn insert_with(&mut self, y: Y, payload: impl FnOnce() -> P) -> bool {
        let v = y.as_ref();
        assert!(
            v.len() == self.width && v.iter().all(|x| x.is_finite()),
            "expected a finite objective row of the archive width"
        );
        if v.len() == 2 && self.cache[0] < v[0] && self.cache[1] < v[1] {
            return false;
        }
        if v.len() != 2
            && self
                .witness
                .as_ref()
                .is_some_and(|p| dominates(p.as_ref(), v))
        {
            return false;
        }
        let sum = v.iter().sum::<f64>();
        if v.len() == 3 {
            if self.blocks.is_none() && self.rows.len() >= 256 {
                self.blocks = Some(super::blocks::Blocks::from_sorted(
                    std::mem::take(&mut self.rows),
                    std::mem::take(&mut self.payloads),
                ));
            }
            if let Some(blocks) = &mut self.blocks {
                return blocks.insert(sum, y, payload, &mut self.witness, &mut self.len);
            }
        }
        let serial = self.next;
        if v.len() == 2 {
            let (x, b) = (v[0], v[1]);
            if self.line {
                let (big, small) = if x.abs() >= b.abs() { (x, b) } else { (b, x) };
                if sum.is_finite()
                    && sum - big == small
                    && self.members.first().is_none_or(|p| p.sum == sum)
                {
                    self.members.push(Member {
                        y,
                        payload: payload(),
                        sum,
                        serial,
                    });
                    self.record_2d(x, b, sum);
                    return true;
                }
                for member in self.members.drain(..) {
                    self.skyline
                        .entry(crate::pareto::coordinate_key(member.y.as_ref()[0]))
                        .or_default()
                        .push(member);
                }
                self.line = false;
            }
            let key = crate::pareto::coordinate_key(x);
            if let Some((&k, members)) = self.skyline.range(..=key).next_back() {
                let other = members[0].y.as_ref()[1];
                if other < b || (other == b && k != key) {
                    return false;
                }
                if other == b {
                    let entry = Member {
                        y,
                        payload: payload(),
                        sum,
                        serial,
                    };
                    self.skyline
                        .get_mut(&key)
                        .expect("existing skyline key")
                        .push(entry);
                    self.record_2d(x, b, sum);
                    return true;
                }
            }
            let entry = Member {
                y,
                payload: payload(),
                sum,
                serial,
            };
            while let Some((&k, _)) = self
                .skyline
                .range(key..)
                .next()
                .filter(|(_, m)| m[0].y.as_ref()[1] >= b)
            {
                self.len -= self.skyline.remove(&k).expect("existing skyline key").len();
            }
            self.skyline.insert(key, vec![entry]);
            self.record_2d(x, b, sum);
        } else {
            let end = self.rows.partition_point(|p| p.0 <= sum);
            if let Some((_, p)) = self.rows[..end]
                .iter()
                .find(|(_, p)| dominates(p.as_ref(), v))
            {
                self.witness = Some(p.clone());
                return false;
            }
            let payload = payload();
            let first = self.rows.partition_point(|p| p.0 < sum);
            let mut at = end;
            if let Some(offset) = self.rows[first..]
                .iter()
                .position(|(_, p)| dominates(v, p.as_ref()))
            {
                let first_removed = first + offset;
                let mut out = first_removed;
                at = if first_removed < end { end - 1 } else { end };
                for i in first_removed + 1..self.rows.len() {
                    if dominates(v, self.rows[i].1.as_ref()) {
                        if i < end {
                            at -= 1;
                        }
                    } else {
                        self.rows.swap(out, i);
                        self.payloads.swap(out, i);
                        out += 1;
                    }
                }
                self.rows.truncate(out);
                self.payloads.truncate(out);
            }
            self.rows.insert(at, (sum, y));
            self.payloads.insert(at, payload);
            self.len = self.rows.len();
        }
        true
    }
    fn record_2d(&mut self, x: f64, b: f64, sum: f64) {
        self.next += 1;
        self.len += 1;
        if sum < self.best_sum {
            self.best_sum = sum;
            self.cache = [x, b];
        }
    }
    /// Retained objective and payload references in unspecified storage order.
    pub fn entries(&self) -> impl Iterator<Item = (&Y, &P)> {
        self.members
            .iter()
            .chain(self.skyline.values().flatten())
            .map(|p| (&p.y, &p.payload))
            .chain(
                self.rows
                    .iter()
                    .zip(&self.payloads)
                    .map(|((_, y), p)| (y, p)),
            )
            .chain(self.blocks.iter().flat_map(|b| b.entries()))
    }
    /// Snapshot ordered by rounded objective sum and arrival within equal sums.
    pub fn ordered_entries(&self) -> Vec<(&Y, &P)> {
        if let Some(b) = &self.blocks {
            return b.entries().collect();
        }
        if self.width != 2 {
            return self
                .rows
                .iter()
                .zip(&self.payloads)
                .map(|((_, y), p)| (y, p))
                .collect();
        }
        if self.skyline.is_empty() {
            return self.members.iter().map(|p| (&p.y, &p.payload)).collect();
        }
        let mut rows: Vec<_> = self.skyline.values().flatten().collect();
        rows.sort_unstable_by(|a, b| {
            a.sum
                .partial_cmp(&b.sum)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.serial.cmp(&b.serial))
        });
        rows.into_iter().map(|p| (&p.y, &p.payload)).collect()
    }
}
impl<Y: AsRef<[f64]> + Clone> ParetoArchiveND<Y, ()> {
    /// Insert objectives without a payload. Returns the admission decision.
    ///
    /// # Panics
    /// Panics for invalid objective rows, as in [`Self::insert_with`].
    pub fn add(&mut self, y: Y) -> bool {
        self.insert_with(y, || ())
    }
}
