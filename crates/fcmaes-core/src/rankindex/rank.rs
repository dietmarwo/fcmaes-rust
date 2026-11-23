use std::cmp::Ordering;

pub(crate) fn cmp(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).expect("coordinate must not be NaN")
}

pub(super) fn validate(points: &[f64], d: usize) -> Result<usize, &'static str> {
    if d == 0 || !points.len().is_multiple_of(d) {
        return Err("invalid row-major point array");
    }
    let n = points.len() / d;
    if n > u32::MAX as usize {
        return Err("too many points");
    }
    if !points.iter().all(|x| x.is_finite()) {
        return Err("stored coordinates must be finite");
    }
    Ok(n)
}

// Match the positional ranks of equal coordinates in the supplied GCC 14 build.
// std::sort leaves these ties unspecified across C++ libraries. Its median-of-three
// partitions define their practical order. Insertion sort finishes the short runs.
fn axis_order(points: &[f64], d: usize, axis: usize) -> Vec<u32> {
    let mut ids: Vec<_> = (0..(points.len() / d) as u32).collect();
    if ids.is_empty() {
        return ids;
    }
    let value = |id: u32| points[id as usize * d + axis];
    fn sift<T: Fn(u32) -> f64>(a: &mut [u32], mut hole: usize, item: u32, value: &T) {
        let top = hole;
        while 2 * hole + 1 < a.len() {
            let l = 2 * hole + 1;
            let r = l + 1;
            let child = if r < a.len() && value(a[r]) >= value(a[l]) {
                r
            } else {
                l
            };
            a[hole] = a[child];
            hole = child;
        }
        while hole > top {
            let parent = (hole - 1) / 2;
            if value(a[parent]) >= value(item) {
                break;
            }
            a[hole] = a[parent];
            hole = parent;
        }
        a[hole] = item;
    }
    fn partition<T: Fn(u32) -> f64>(mut a: &mut [u32], mut depth: u32, v: &T) {
        while a.len() > 16 {
            if depth == 0 {
                for i in (0..a.len() / 2).rev() {
                    sift(a, i, a[i], v);
                }
                for end in (1..a.len()).rev() {
                    a.swap(0, end);
                    let item = a[0];
                    sift(&mut a[..end], 0, item, v);
                }
                return;
            }
            depth -= 1;
            let (i, j, k) = (1, a.len() / 2, a.len() - 1);
            let (x, y, z) = (v(a[i]), v(a[j]), v(a[k]));
            let mid = if x < y {
                if y < z {
                    j
                } else if x < z {
                    k
                } else {
                    i
                }
            } else if x < z {
                i
            } else if y < z {
                k
            } else {
                j
            };
            a.swap(0, mid);
            let pivot = v(a[0]);
            let (mut lo, mut hi) = (1, a.len());
            loop {
                while v(a[lo]) < pivot {
                    lo += 1;
                }
                hi -= 1;
                while pivot < v(a[hi]) {
                    hi -= 1;
                }
                if lo >= hi {
                    break;
                }
                a.swap(lo, hi);
                lo += 1;
            }
            let (left, right) = a.split_at_mut(lo);
            partition(right, depth, v);
            a = left;
        }
    }
    let depth = 2 * ids.len().ilog2();
    partition(&mut ids, depth, &value);
    for i in 1..ids.len() {
        let id = ids[i];
        let x = value(id);
        let mut j = i;
        while j > 0 && x < value(ids[j - 1]) {
            ids[j] = ids[j - 1];
            j -= 1;
        }
        ids[j] = id;
    }
    ids
}

pub(crate) struct Ranks {
    pub columns: Vec<Vec<f64>>,
    pub orders: Vec<Vec<u32>>,
    pub by_id: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Span {
    pub lo: u32,
    pub width: u32,
}

impl Span {
    pub fn new(col: &[f64], low: f64, high: f64) -> Option<Self> {
        let a = col.partition_point(|&x| x < low);
        let b = col.partition_point(|&x| x <= high);
        if a >= b || low.is_nan() || high.is_nan() {
            None
        } else {
            Some(Self {
                lo: a as u32,
                width: (b - a - 1) as u32,
            })
        }
    }
    pub fn contains(self, rank: u32) -> bool {
        rank.wrapping_sub(self.lo) <= self.width
    }
}

// Cache each comparison key locally. Use the reference order for tied columns.
fn ordered_bits(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}

struct Column {
    coords: Vec<f64>,
    ids: Vec<u32>,
    by_id: Vec<u32>,
}
fn make_column(n: usize, entries: impl IntoIterator<Item = (f64, u32)>) -> Column {
    let mut by_id = vec![0; n];
    let mut ids = Vec::with_capacity(n);
    let mut coords = Vec::with_capacity(n);
    for (r, (x, id)) in entries.into_iter().enumerate() {
        by_id[id as usize] = r as u32;
        ids.push(id);
        coords.push(x);
    }
    Column { coords, ids, by_id }
}
fn sorted_column(points: &[f64], d: usize, j: usize, n: usize) -> Column {
    let reference_order = || {
        make_column(
            n,
            axis_order(points, d, j)
                .into_iter()
                .map(|id| (points[id as usize * d + j], id)),
        )
    };
    if n < 256 {
        return reference_order();
    }
    let mut sample = std::array::from_fn::<_, 64, _>(|i| points[i * d + j]);
    sample.sort_unstable_by(|a, b| cmp(*a, *b));
    if sample.windows(2).any(|w| w[0] == w[1]) {
        return reference_order();
    }
    let mut entries: Vec<_> = (0..n).map(|i| (points[i * d + j], i as u32)).collect();
    let mut scratch = vec![(0., 0u32); n];
    for shift in [0, 11, 22, 33, 44, 55] {
        let bucket = |x: f64| ((ordered_bits(x) >> shift) & 2047) as usize;
        let mut counts = [0usize; 2048];
        for &(x, _) in &entries {
            counts[bucket(x)] += 1;
        }
        if counts.contains(&n) {
            continue;
        }
        let mut offset = 0;
        for count in &mut counts {
            let k = *count;
            *count = offset;
            offset += k;
        }
        for &entry in &entries {
            let pos = &mut counts[bucket(entry.0)];
            scratch[*pos] = entry;
            *pos += 1;
        }
        std::mem::swap(&mut entries, &mut scratch);
    }
    if entries.windows(2).any(|w| w[0].0 == w[1].0) {
        return reference_order();
    }
    make_column(n, entries)
}

pub(super) fn cached_ranks(points: &[f64], d: usize, n: usize) -> Ranks {
    let mut out = Ranks {
        columns: Vec::with_capacity(d),
        orders: Vec::with_capacity(d),
        by_id: Vec::with_capacity(d),
    };
    for j in 0..d {
        let c = sorted_column(points, d, j, n);
        out.columns.push(c.coords);
        out.orders.push(c.ids);
        out.by_id.push(c.by_id);
    }
    out
}
