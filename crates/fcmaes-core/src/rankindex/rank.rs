// Part of the rank index port; see the header of rankindex.rs and NOTICE.fronts.
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

// Monotone integer key of a finite coordinate. Both zeros share one key, so the
// stable radix passes keep equal coordinates in row order.
fn ordered_bits(x: f64) -> u64 {
    let bits = if x == 0. { 0 } else { x.to_bits() };
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
// Rows sorted by coordinate; equal coordinates keep ascending row order.
// Positional ranks, Borda output order and grid cells are therefore the same
// on every platform and in the C++ indexes of fast-cma-es.
fn sorted_column(points: &[f64], d: usize, j: usize, n: usize) -> Column {
    let mut entries: Vec<_> = (0..n).map(|i| (points[i * d + j], i as u32)).collect();
    if n < 256 {
        // A stable sort on the radix key gives the same order as the passes below.
        entries.sort_by_key(|&(x, _)| ordered_bits(x));
        return make_column(n, entries);
    }
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
