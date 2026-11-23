// Part of the archive DE port; see the header of archivede.rs and NOTICE.fronts.
//! Sum-ordered blocks for large three-objective fronts. Component minima and
//! maxima let a whole block fail a dominance test without inspecting its rows.

const CAP: usize = 96;
struct Block<Y, P> {
    rows: Vec<(f64, Y, P)>,
    min: [f64; 3],
    max: [f64; 3],
}
impl<Y: AsRef<[f64]>, P> Block<Y, P> {
    fn new(rows: Vec<(f64, Y, P)>) -> Self {
        let mut b = Self {
            rows,
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        };
        b.refresh();
        b
    }
    fn refresh(&mut self) {
        self.min = [f64::INFINITY; 3];
        self.max = [f64::NEG_INFINITY; 3];
        for (_, y, _) in &self.rows {
            for (j, &v) in y.as_ref().iter().enumerate() {
                self.min[j] = self.min[j].min(v);
                self.max[j] = self.max[j].max(v);
            }
        }
    }
}
pub(super) struct Blocks<Y, P> {
    blocks: Vec<Block<Y, P>>,
}
impl<Y: AsRef<[f64]> + Clone, P> Blocks<Y, P> {
    pub(super) fn from_sorted(rows: Vec<(f64, Y)>, payloads: Vec<P>) -> Self {
        let mut it = rows.into_iter().zip(payloads).map(|((s, y), p)| (s, y, p));
        let mut blocks = Vec::new();
        while it.len() != 0 {
            blocks.push(Block::new(it.by_ref().take(CAP / 2).collect()));
        }
        Self { blocks }
    }
    pub(super) fn entries(&self) -> impl Iterator<Item = (&Y, &P)> {
        self.blocks
            .iter()
            .flat_map(|b| b.rows.iter().map(|(_, y, p)| (y, p)))
    }
    pub(super) fn insert(
        &mut self,
        sum: f64,
        y: Y,
        payload: impl FnOnce() -> P,
        witness: &mut Option<Y>,
        len: &mut usize,
    ) -> bool {
        let v = y.as_ref();
        for b in self.blocks.iter().take_while(|b| b.rows[0].0 <= sum) {
            if (0..3).any(|j| b.min[j] > v[j]) {
                continue;
            }
            if let Some((_, p, _)) = b
                .rows
                .iter()
                .take_while(|r| r.0 <= sum)
                .find(|(_, p, _)| crate::indicators::dominates(p.as_ref(), v))
            {
                *witness = Some(p.clone());
                return false;
            }
        }
        let payload = payload();
        let start = self
            .blocks
            .partition_point(|b| b.rows.last().unwrap().0 < sum);
        for b in &mut self.blocks[start..] {
            if (0..3).any(|j| b.max[j] < v[j]) {
                continue;
            }
            let before = b.rows.len();
            b.rows
                .retain(|(s, p, _)| *s < sum || !crate::indicators::dominates(v, p.as_ref()));
            if b.rows.len() != before {
                *len -= before - b.rows.len();
                if !b.rows.is_empty() {
                    b.refresh();
                }
            }
        }
        self.blocks.retain(|b| !b.rows.is_empty());
        if self.blocks.is_empty() {
            self.blocks.push(Block::new(vec![(sum, y, payload)]));
        } else {
            let slot = self
                .blocks
                .partition_point(|b| b.rows.last().unwrap().0 <= sum)
                .min(self.blocks.len() - 1);
            let b = &mut self.blocks[slot];
            let at = b.rows.partition_point(|r| r.0 <= sum);
            for (j, &x) in v.iter().enumerate() {
                b.min[j] = b.min[j].min(x);
                b.max[j] = b.max[j].max(x);
            }
            b.rows.insert(at, (sum, y, payload));
            if b.rows.len() > CAP {
                let right = Block::new(b.rows.split_off(CAP / 2));
                b.refresh();
                self.blocks.insert(slot + 1, right);
            }
        }
        *len += 1;
        true
    }
}
