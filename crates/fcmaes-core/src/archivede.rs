//! Archive-gated DE/rand/1/bin for bounded, continuous, minimized objectives.
//! The archive keeps every strictly nondominated evaluation, including duplicates.
//! The breeding population uses objective champions and greedy rank-space spread.
//! Adapted from Aleksey Vaneev's `archive_de.cpp` (BiteOpt).
#![doc = include_str!("../FRONT_TOOLS.md")]
mod archive;
mod blocks;
mod mt;
pub use archive::ParetoArchive;
use mt::Mt;

/// A decision or objective row. Arrays avoid per-row allocation for known widths.
pub trait Row: AsRef<[f64]> + AsMut<[f64]> + Clone {
    /// Construct a zero row, or return `None` if a fixed width does not match.
    fn zeros(width: usize) -> Option<Self>;
}
impl<const N: usize> Row for [f64; N] {
    fn zeros(width: usize) -> Option<Self> {
        (width == N).then_some([0.; N])
    }
}
impl Row for Vec<f64> {
    fn zeros(width: usize) -> Option<Self> {
        Some(vec![0.; width])
    }
}

/// Controls the population, evaluation cap, random stream, and survivor policy.
#[derive(Clone, Copy, Debug)]
pub struct ArchiveDeParams {
    /// Population size, at least four and greater than the objective count.
    pub population: usize,
    /// Hard cap including initialization and a possible partial final generation.
    pub max_evaluations: u64,
    /// MT19937 seed. The two-word uniform draw reproduces the source driver.
    pub seed: u32,
    /// Admit offspring to the survivor pool only if they enter the archive.
    pub gate: bool,
    /// Use champions and rank-space spread. False selects by stable rank-sum truncation.
    pub tiered: bool,
}
impl Default for ArchiveDeParams {
    fn default() -> Self {
        Self {
            population: 100,
            max_evaluations: 20_100,
            seed: 100,
            gate: true,
            tiered: true,
        }
    }
}

/// Synchronous archive-gated DE with real-valued box bounds. Arrays and `Vec<f64>`
/// use the same implementation; choose the latter for runtime dimensions.
pub struct ArchiveDe<X: Row, Y: Row> {
    lower: X,
    upper: X,
    unit_box: bool,
    width: usize,
    params: ArchiveDeParams,
    rng: Mt,
    parents: Vec<X>,
    values: Vec<Y>,
    pending: Vec<X>,
    asked: Vec<X>,
    evaluations: u64,
    archive: ParetoArchive<Y, X>,
}

fn select<Y: Row>(pool: &[Y], np: usize, tiered: bool) -> Vec<usize> {
    let n = pool.len();
    let width = pool[0].as_ref().len();
    let mut scores = vec![0usize; n];
    let mut ranks = vec![Y::zeros(width).expect("validated objective width"); n];
    let mut order: Vec<_> = (0..n).collect();
    let mut used = vec![false; n];
    let mut keep = Vec::with_capacity(np);
    for j in 0..width {
        for (i, id) in order.iter_mut().enumerate() {
            *id = i;
        }
        order.sort_by(|&a, &b| {
            pool[a].as_ref()[j]
                .partial_cmp(&pool[b].as_ref()[j])
                .unwrap()
        });
        if tiered {
            let champion = *order
                .iter()
                .find(|&&i| !used[i])
                .expect("more candidates than objectives");
            used[champion] = true;
            keep.push(champion);
        }
        for (r, &id) in order.iter().enumerate() {
            scores[id] += r;
            ranks[id].as_mut()[j] = r as f64;
        }
    }
    if width <= 8 {
        let mut offsets = vec![0usize; width * (n - 1) + 1];
        for &s in &scores {
            offsets[s] += 1;
        }
        let mut next = 0;
        for c in &mut offsets {
            let size = *c;
            *c = next;
            next += size;
        }
        for (id, &s) in scores.iter().enumerate() {
            order[offsets[s]] = id;
            offsets[s] += 1;
        }
    } else {
        for (i, id) in order.iter_mut().enumerate() {
            *id = i;
        }
        order.sort_by_key(|&i| scores[i]);
    }
    if !tiered {
        order.truncate(np);
        return order;
    }
    let mut eligible = order[..6 * n / 10].iter().copied().filter(|&i| !used[i]);
    if let Some(mut seed) = eligible.next() {
        let mut active: Vec<_> = eligible
            .map(|i| (i, ranks[i].clone(), f64::INFINITY))
            .collect();
        keep.push(seed);
        used[seed] = true;
        let squared = width as f64 * (n as f64) * (n as f64) < 4503599627370496.;
        for _ in 1..3 * np / 5 {
            if active.is_empty() || keep.len() == np {
                break;
            }
            let center = ranks[seed].as_ref();
            let mut best = 0;
            let mut farthest = -1.;
            for (k, (_, rank, min_dist)) in active.iter_mut().enumerate() {
                let distance = rank
                    .as_ref()
                    .iter()
                    .zip(center)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>();
                *min_dist = min_dist.min(if squared { distance } else { distance.sqrt() });
                if *min_dist > farthest {
                    best = k;
                    farthest = *min_dist;
                }
            }
            seed = active.remove(best).0;
            keep.push(seed);
            used[seed] = true;
        }
    }
    for id in order {
        if keep.len() == np {
            break;
        }
        if !used[id] {
            keep.push(id);
        }
    }
    keep
}

impl<X: Row, Y: Row> ArchiveDe<X, Y> {
    /// Construct a state with finite ordered bounds. Equal bounds fix a coordinate.
    ///
    /// # Errors
    /// Returns an error for invalid widths, bounds, or population size.
    pub fn new(
        lower: X,
        upper: X,
        nobj: usize,
        params: ArchiveDeParams,
    ) -> Result<Self, &'static str> {
        let (lo, hi) = (lower.as_ref(), upper.as_ref());
        if lo.is_empty()
            || lo.len() != hi.len()
            || lo.len() > u32::MAX as usize
            || lo
                .iter()
                .zip(hi)
                .any(|(&a, &b)| !a.is_finite() || !b.is_finite() || a > b)
        {
            return Err("bounds must be matching nonempty finite ordered vectors");
        }
        if nobj == 0
            || params.population < 4
            || params.population <= nobj
            || params.population > u32::MAX as usize
            || Y::zeros(nobj).is_none()
        {
            return Err(
                "require a matching positive objective width and a population >= 4 and > nobj",
            );
        }
        let unit_box = lo.iter().zip(hi).all(|(&a, &b)| a == 0. && b == 1.);
        Ok(Self {
            lower,
            upper,
            unit_box,
            width: nobj,
            params,
            rng: Mt::new(params.seed),
            parents: vec![],
            values: vec![],
            pending: vec![],
            asked: vec![],
            evaluations: 0,
            archive: ParetoArchive::new(nobj),
        })
    }
    fn decode(&self, u: &X) -> X {
        if self.unit_box {
            return u.clone();
        }
        let mut x = u.clone();
        for ((v, &lo), &hi) in x
            .as_mut()
            .iter_mut()
            .zip(self.lower.as_ref())
            .zip(self.upper.as_ref())
        {
            let width = hi - lo;
            let t = *v;
            *v = if t == 1. {
                hi
            } else if width.is_finite() {
                (lo + t * width).clamp(lo, hi)
            } else {
                (1. - t) * lo + t * hi
            };
        }
        x
    }
    /// Return the pending real-coordinate batch in evaluation order. Repeated calls
    /// are idempotent. An empty slice signals budget exhaustion.
    pub fn ask(&mut self) -> &[X] {
        if !self.pending.is_empty() || self.evaluations == self.params.max_evaluations {
            return if self.unit_box {
                &self.pending
            } else {
                &self.asked
            };
        }
        let np = self.params.population;
        let n = (np as u64).min(self.params.max_evaluations - self.evaluations) as usize;
        let dim = self.lower.as_ref().len();
        for i in 0..n {
            let mut x = self.lower.clone();
            if self.parents.is_empty() {
                for v in x.as_mut() {
                    *v = self.rng.unit();
                }
            } else {
                let mut draw = || self.rng.next() as usize % np;
                let mut a = draw();
                while a == i {
                    a = draw();
                }
                let mut b = draw();
                while b == i || b == a {
                    b = draw();
                }
                let mut c = draw();
                while c == i || c == a || c == b {
                    c = draw();
                }
                let forced = self.rng.next() as usize % dim;
                let (pa, pb, pc, pi) = (
                    self.parents[a].as_ref(),
                    self.parents[b].as_ref(),
                    self.parents[c].as_ref(),
                    self.parents[i].as_ref(),
                );
                for (j, v) in x.as_mut().iter_mut().enumerate() {
                    *v = if self.rng.unit() < 0.3 || j == forced {
                        (pa[j] + 0.5 * (pb[j] - pc[j])).clamp(0., 1.)
                    } else {
                        pi[j]
                    };
                }
            }
            if !self.unit_box {
                self.asked.push(self.decode(&x));
            }
            self.pending.push(x);
        }
        if self.unit_box {
            &self.pending
        } else {
            &self.asked
        }
    }
    /// Submit the objective rows of the entire pending batch, in order.
    ///
    /// # Errors
    /// A missing batch, incorrect shape, or non-finite objective leaves the pending batch intact.
    pub fn tell(&mut self, values: &[Y]) -> Result<(), &'static str> {
        if self.pending.is_empty() {
            return Err("tell requires an outstanding ask");
        }
        if values.len() != self.pending.len()
            || values.iter().any(|y| {
                y.as_ref().len() != self.width || !y.as_ref().iter().all(|x| x.is_finite())
            })
        {
            return Err("expected a finite objective row for each pending decision");
        }
        let initial = self.parents.is_empty();
        let mut physical = self.asked.drain(..);
        for (unit, y) in self.pending.drain(..).zip(values) {
            let x = physical.next();
            let admitted = self
                .archive
                .insert_with(y.clone(), || x.unwrap_or_else(|| unit.clone()));
            if initial || !self.params.gate || admitted {
                self.parents.push(unit);
                self.values.push(y.clone());
            }
        }
        if !initial {
            let keep = select(&self.values, self.params.population, self.params.tiered);
            self.parents = keep.iter().map(|&i| self.parents[i].clone()).collect();
            self.values = keep.iter().map(|&i| self.values[i].clone()).collect();
        }
        self.evaluations += values.len() as u64;
        Ok(())
    }
    /// Total number of submitted objective evaluations.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }
    /// Decision dimension.
    pub fn dimension(&self) -> usize {
        self.lower.as_ref().len()
    }
    /// Objective dimension.
    pub fn objectives(&self) -> usize {
        self.width
    }
    /// All nondominated evaluated objectives, with the corresponding real decisions as payloads.
    pub fn archive(&self) -> &ParetoArchive<Y, X> {
        &self.archive
    }
    /// Return the evaluated breeding population as (real decision, objectives) pairs.
    pub fn population(&self) -> Vec<(X, Y)> {
        self.parents
            .iter()
            .zip(&self.values)
            .map(|(u, y)| (self.decode(u), y.clone()))
            .collect()
    }
    /// Evaluate one row at a time until the budget is exhausted.
    ///
    /// # Errors
    /// Propagates invalid objective rows through [`Self::tell`].
    pub fn minimize(&mut self, mut objective: impl FnMut(&X) -> Y) -> Result<(), &'static str> {
        loop {
            let xs = self.ask();
            if xs.is_empty() {
                return Ok(());
            }
            let ys = xs.iter().map(&mut objective).collect::<Vec<_>>();
            self.tell(&ys)?;
        }
    }
}
