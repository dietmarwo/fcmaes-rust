use super::rank::{Span, cached_ranks};

/// Matching input rows and traversal diagnostics for one closed box.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct BordaResult {
    /// Input row IDs in index traversal order.
    pub ids: Vec<u32>,
    /// Rows in the shortest marginal rank interval.
    pub axis_candidates: u64,
    /// Rows in that interval whose Borda score lies in the query band.
    pub score_candidates: u64,
}

/// Immutable index over positional ranks on each axis. Cross-ranks take O(n d²) space.
pub struct BordaIndex {
    guides: Vec<Guide>,
    columns: Vec<Vec<f64>>,
    ids: Vec<Vec<u32>>,
    scores: Vec<Vec<u64>>,
    layout: CrossRanks,
}

enum CrossRanks {
    Narrow(Vec<Vec<Vec<u16>>>),
    Wide(Vec<Vec<Vec<u32>>>),
}
fn cross_ranks<R>(ranks: &super::rank::Ranks, convert: impl Fn(u32) -> R) -> Vec<Vec<Vec<R>>> {
    ranks
        .orders
        .iter()
        .enumerate()
        .map(|(axis, ids)| {
            ranks
                .by_id
                .iter()
                .enumerate()
                .map(|(j, r)| {
                    if j == axis {
                        Vec::new()
                    } else {
                        ids.iter().map(|&id| convert(r[id as usize])).collect()
                    }
                })
                .collect()
        })
        .collect()
}

impl BordaIndex {
    /// Build from row-major finite coordinates with positive dimension.
    ///
    /// # Panics
    /// Panics on a malformed matrix, a non-finite coordinate, or more than u32::MAX rows.
    pub fn new(points: &[f64], d: usize) -> Self {
        Self::try_new(points, d).expect("invalid Borda index input")
    }
    /// Build an index and validate the row-major matrix.
    ///
    /// # Errors
    /// Returns an error on invalid dimensions or non-finite coordinates.
    pub fn try_new(points: &[f64], d: usize) -> Result<Self, &'static str> {
        let n = super::rank::validate(points, d)?;
        let ranks = cached_ranks(points, d, n);
        let mut sums = vec![0u64; n];
        for col in &ranks.by_id {
            for (sum, &r) in sums.iter_mut().zip(col) {
                *sum += u64::from(r);
            }
        }
        let scores = ranks
            .orders
            .iter()
            .map(|order| order.iter().map(|&id| sums[id as usize]).collect())
            .collect();
        let layout = if n <= 65536 {
            CrossRanks::Narrow(cross_ranks(&ranks, |r| r as u16))
        } else {
            CrossRanks::Wide(cross_ranks(&ranks, |r| r))
        };
        Ok(Self {
            guides: ranks.columns.iter().map(|v| Guide::new(v)).collect(),
            columns: ranks.columns,
            ids: ranks.orders,
            scores,
            layout,
        })
    }

    /// Dimension of the indexed points.
    pub fn dimension(&self) -> usize {
        self.columns.len()
    }

    /// Report input rows in a closed box. A NaN or inverted bound produces no hits.
    ///
    /// # Panics
    /// Panics if either endpoint has a different dimension from the index.
    pub fn query(&self, low: &[f64], high: &[f64]) -> BordaResult {
        let d = self.columns.len();
        assert_eq!(low.len(), d);
        assert_eq!(high.len(), d);
        let mut out = BordaResult::default();
        let mut spans = Vec::with_capacity(d);
        let (mut axis, mut lower, mut upper) = (0, 0u64, 0u64);
        for (j, col) in self.columns.iter().enumerate() {
            let Some(s) = self.guides[j].span(col, low[j], high[j]) else {
                return out;
            };
            spans.push(s);
            if s.width < spans[axis].width {
                axis = j;
            }
            lower += u64::from(s.lo);
            upper += u64::from(s.lo + s.width);
        }
        let span = spans[axis];
        let begin = span.lo as usize;
        let end = begin + span.width as usize + 1;
        out.axis_candidates = (end - begin) as u64;
        let scores = &self.scores[axis][begin..end];
        out.score_candidates = scores
            .iter()
            .map(|&s| u64::from(s.wrapping_sub(lower) <= upper - lower))
            .sum();
        let ids = &self.ids[axis];
        if spans
            .iter()
            .enumerate()
            .all(|(j, s)| j == axis || s.width as usize + 1 == ids.len())
        {
            out.ids.extend_from_slice(&ids[begin..end]);
            return out;
        }
        match &self.layout {
            CrossRanks::Wide(cross) => filter(
                &cross[axis],
                &spans,
                axis,
                ids,
                begin..end,
                &mut out.ids,
                Span::contains,
            ),
            CrossRanks::Narrow(cross) => filter(
                &cross[axis],
                &spans,
                axis,
                ids,
                begin..end,
                &mut out.ids,
                |s, r| r.wrapping_sub(s.lo as u16) <= s.width as u16,
            ),
        }

        out
    }
}

// Both rank widths share the same block and hit emission logic.
fn filter<R: Copy>(
    layout: &[Vec<R>],
    spans: &[Span],
    axis: usize,
    ids: &[u32],
    range: std::ops::Range<usize>,
    out: &mut Vec<u32>,
    contains: impl Fn(Span, R) -> bool,
) {
    let end = range.end;
    for p in range.step_by(64) {
        let count = 64.min(end - p);
        let mut hits = [1u8; 64];
        for (j, &s) in spans.iter().enumerate() {
            if axis == j || s.width as usize + 1 == ids.len() {
                continue;
            }
            for (h, &r) in hits[..count].iter_mut().zip(&layout[j][p..p + count]) {
                *h &= u8::from(contains(s, r));
            }
        }
        for (base, chunk) in hits[..count].chunks(8).enumerate() {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            let mut bits = u64::from_le_bytes(bytes);
            while bits != 0 {
                let i = bits.trailing_zeros() as usize / 8;
                bits &= bits - 1;
                out.push(ids[p + base * 8 + i]);
            }
        }
    }
}

struct Guide {
    min: f64,
    scale: f64,
    offsets: Vec<u32>,
}
impl Guide {
    fn bucket(&self, x: f64) -> usize {
        (((x - self.min) * self.scale) as usize).min(255)
    }
    fn new(col: &[f64]) -> Self {
        let min = col.first().copied().unwrap_or(0.);
        let width = col.last().copied().unwrap_or(0.) - min;
        let mut g = Self {
            min,
            scale: 256. / width,
            offsets: Vec::new(),
        };
        if width > 0. && g.scale.is_finite() && g.scale > 0. {
            g.offsets = vec![0; 257];
            for &x in col {
                let j = g.bucket(x);
                g.offsets[j + 1] += 1;
            }
            for j in 0..256 {
                g.offsets[j + 1] += g.offsets[j];
            }
        }
        g
    }
    fn span(&self, col: &[f64], low: f64, high: f64) -> Option<Span> {
        if self.offsets.is_empty() {
            return Span::new(col, low, high);
        }
        if low.is_nan() || high.is_nan() {
            return None;
        }
        let pos = |value: f64, inclusive: bool| {
            let k = self.bucket(value);
            let start = self.offsets[k] as usize;
            start
                + col[start..self.offsets[k + 1] as usize]
                    .partition_point(|&x| if inclusive { x <= value } else { x < value })
        };
        let a = pos(low, false);
        let b = pos(high, true);
        (a < b).then(|| Span {
            lo: a as u32,
            width: (b - a - 1) as u32,
        })
    }
}
