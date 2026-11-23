// Part of the rank index port; see the header of rankindex.rs and NOTICE.fronts.
use super::rank::{Span, cached_ranks};

/// Matching input rows and traversal diagnostics for one closed box.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct GridResult {
    /// Input row IDs in index traversal order.
    pub ids: Vec<u32>,
    /// Number of theoretical grid cells intersecting the rank box.
    pub touched_cells: u128,
    /// Occupied cells intersecting the rank box.
    pub nonempty_cells: u64,
    /// Occupied cells contained entirely in the rank box.
    pub full_cells: u64,
    /// Rows individually inspected inside partially covered cells.
    pub point_tests: u64,
}

#[derive(Clone, Copy, Default)]
struct Axis {
    span: Span,
    first: u32,
    last: u32,
    left_full: bool,
    right_full: bool,
}
impl Axis {
    fn full(&self, c: u32) -> bool {
        (c != self.first || self.left_full) && (c != self.last || self.right_full)
    }
}

/// Sparse uniform grid in rank space, with original IDs ordered by cell and then row.
pub struct GridIndex {
    columns: Vec<Vec<f64>>,
    keys: Vec<u64>,
    starts: Vec<usize>,
    ids: Vec<u32>,
    rows: Vec<u32>,
    direct: Vec<u32>,
    resolution: u32,
}

impl GridIndex {
    /// Build from row-major points. The positive nominal query volume sets the resolution.
    ///
    /// # Panics
    /// Panics on invalid dimensions, non-finite coordinates, an invalid volume, more than
    /// i32::MAX rows, or a nonempty grid whose cell key cannot fit in 64 bits.
    pub fn new(points: &[f64], d: usize, volume: f64) -> Self {
        Self::try_new(points, d, volume).expect("invalid rank grid input")
    }
    /// Build a grid while checking the matrix and cell-key domain.
    ///
    /// # Errors
    /// Returns an error on invalid coordinates, dimensions, volume, or unrepresentable keys.
    pub fn try_new(points: &[f64], d: usize, volume: f64) -> Result<Self, &'static str> {
        let n = super::rank::validate(points, d)?;
        if n > i32::MAX as usize {
            return Err("grid supports at most i32::MAX rows");
        }
        if !volume.is_finite() || volume <= 0. {
            return Err("query volume must be finite and positive");
        }
        let g = volume.powf(-1. / d as f64).ceil().max(2.);
        if g > u32::MAX as f64 {
            return Err("resolution exceeds 32 bits");
        }
        let g = g as u32;
        let maxkey = if n == 0 {
            0
        } else {
            (0..d)
                .try_fold(0u64, |k, _| {
                    k.checked_mul(g as u64)?.checked_add((g - 1) as u64)
                })
                .ok_or("grid cell key exceeds 64 bits")?
        };
        let ranks = cached_ranks(points, d, n);
        let mut out = Self {
            columns: ranks.columns,
            keys: Vec::new(),
            starts: Vec::new(),
            ids: vec![0; n],
            rows: Vec::with_capacity(points.len()),
            direct: Vec::new(),
            resolution: g,
        };
        if n == 0 {
            return Ok(out);
        }
        let bins: Vec<_> = (0..n as u64).map(|r| r * g as u64 / n as u64).collect();
        let mut keys = vec![0u64; n];
        for axis in &ranks.by_id {
            for (key, &r) in keys.iter_mut().zip(axis) {
                *key = *key * g as u64 + bins[r as usize];
            }
        }
        if maxkey < 262144 && maxkey < 8 * n as u64 + 64 {
            let mut counts = vec![0usize; maxkey as usize + 1];
            for &k in &keys {
                counts[k as usize] += 1;
            }
            let mut pos = 0usize;
            for (k, count) in counts.iter_mut().enumerate() {
                out.direct.push(out.keys.len() as u32);
                let size = *count;
                *count = pos;
                if size > 0 {
                    out.keys.push(k as u64);
                    out.starts.push(pos);
                    pos += size;
                }
            }
            for (id, &k) in keys.iter().enumerate() {
                let offset = &mut counts[k as usize];
                out.ids[*offset] = id as u32;
                *offset += 1;
            }
        } else {
            out.ids = (0..n as u32).collect();
            out.ids.sort_unstable_by_key(|&id| (keys[id as usize], id));
            for (i, &id) in out.ids.iter().enumerate() {
                let key = keys[id as usize];
                if out.keys.last() != Some(&key) {
                    out.keys.push(key);
                    out.starts.push(i);
                }
            }
        }
        out.starts.push(n);
        for &id in &out.ids {
            for axis in &ranks.by_id {
                out.rows.push(axis[id as usize]);
            }
        }
        Ok(out)
    }
    /// Number of cells per axis.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }
    /// Number of nonempty cells.
    pub fn occupied_cells(&self) -> usize {
        self.keys.len()
    }

    /// Dimension of the indexed points.
    pub fn dimension(&self) -> usize {
        self.columns.len()
    }

    /// Report input rows in a closed box. A NaN or inverted bound produces no hits.
    ///
    /// # Panics
    /// Panics if either endpoint has a different dimension from the index.
    pub fn query(&self, low: &[f64], high: &[f64]) -> GridResult {
        let d = self.columns.len();
        assert_eq!(low.len(), d);
        assert_eq!(high.len(), d);
        let mut out = GridResult::default();
        let n = self.ids.len() as u64;
        let g = self.resolution as u64;
        let mut axes = Vec::with_capacity(d);
        let mut entire = true;
        let mut touched = 1u128;
        for (j, col) in self.columns.iter().enumerate() {
            let Some(s) = Span::new(col, low[j], high[j]) else {
                return out;
            };
            let hi = s.lo + s.width;
            let first = u64::from(s.lo) * g / n;
            let last = u64::from(hi) * g / n;
            entire &= u64::from(s.width) + 1 == n;
            touched *= u128::from(last - first + 1);
            axes.push(Axis {
                span: s,
                first: first as u32,
                last: last as u32,
                left_full: (first * n).div_ceil(g) >= u64::from(s.lo),
                right_full: ((last + 1) * n - 1) / g <= u64::from(hi),
            });
        }
        out.touched_cells = touched;
        if entire {
            out.ids = self.ids.clone();
            out.full_cells = self.keys.len() as u64;
            out.nonempty_cells = out.full_cells;
        } else if touched / u128::from(axes[d - 1].last - axes[d - 1].first + 1)
            > 4 * self.keys.len() as u128 + 128
        {
            // Large representable domains can have exponentially many empty prefixes.
            for (i, &key) in self.keys.iter().enumerate() {
                let mut k = key;
                let mut full = true;
                let mut intersects = true;
                for a in axes.iter().rev() {
                    let c = (k % g) as u32;
                    k /= g;
                    if c < a.first || c > a.last {
                        intersects = false;
                        break;
                    }
                    full &= a.full(c);
                }
                if intersects {
                    self.cell(i, full, &axes, &mut out);
                }
            }
        } else {
            self.visit(0, 0, true, &axes, &mut out);
        }
        out
    }

    fn lower(&self, k: u64) -> usize {
        if self.direct.is_empty() {
            self.keys.partition_point(|&x| x < k)
        } else {
            self.direct[k as usize] as usize
        }
    }

    fn visit(&self, j: usize, prefix: u64, full: bool, axes: &[Axis], out: &mut GridResult) {
        let a = axes[j];
        let base = prefix * self.resolution as u64;
        if j + 1 < axes.len() {
            for c in a.first..=a.last {
                self.visit(j + 1, base + c as u64, full && a.full(c), axes, out);
            }
        } else {
            let mut i = self.lower(base + u64::from(a.first));
            let end = base + u64::from(a.last);
            while i < self.keys.len() && self.keys[i] <= end {
                self.cell(i, full && a.full((self.keys[i] - base) as u32), axes, out);
                i += 1;
            }
        }
    }

    fn cell(&self, i: usize, full: bool, axes: &[Axis], out: &mut GridResult) {
        let (begin, end) = (self.starts[i], self.starts[i + 1]);
        out.nonempty_cells += 1;
        if full {
            out.full_cells += 1;
            out.ids.extend_from_slice(&self.ids[begin..end]);
        } else {
            out.point_tests += (end - begin) as u64;
            let d = axes.len();
            let rows = &self.rows[begin * d..end * d];
            fn scan<const D: usize>(rows: &[u32], ids: &[u32], axes: &[Axis], hits: &mut Vec<u32>) {
                for (row, &id) in rows.chunks_exact(D).zip(ids) {
                    if (0..D).all(|j| axes[j].span.contains(row[j])) {
                        hits.push(id);
                    }
                }
            }
            match d {
                3 => scan::<3>(rows, &self.ids[begin..end], axes, &mut out.ids),
                5 => scan::<5>(rows, &self.ids[begin..end], axes, &mut out.ids),
                _ => {
                    for (row, &id) in rows.chunks_exact(d).zip(&self.ids[begin..end]) {
                        if row.iter().zip(axes).all(|(&r, a)| a.span.contains(r)) {
                            out.ids.push(id);
                        }
                    }
                }
            }
        }
    }
}
