use super::Entry;

const CAPACITY: usize = 96;
type Position = (usize, usize);

/// Sorted short pages. New left extremes accumulate in reverse order, making both
/// extremes cheap while entries() still walks the entire front in increasing x.
#[derive(Clone)]
pub(super) struct Pages<P> {
    blocks: Vec<Vec<Entry<P>>>,
    ends: Vec<f64>,
    left: Vec<Entry<P>>,
    hint: Option<Position>,
    len: usize,
}
impl<P> Pages<P> {
    pub(super) fn new() -> Self {
        Self {
            blocks: vec![],
            ends: vec![],
            left: vec![],
            hint: None,
            len: 0,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &Entry<P>> {
        self.left.iter().rev().chain(self.blocks.iter().flatten())
    }
    pub(super) fn first(&self) -> Option<&Entry<P>> {
        self.left.last().or_else(|| self.blocks.first()?.first())
    }
    pub(super) fn last(&self) -> Option<&Entry<P>> {
        self.blocks.last()?.last()
    }
    pub(super) fn replace_last(&mut self, entry: Entry<P>) {
        let last = self
            .blocks
            .last_mut()
            .and_then(|page| page.last_mut())
            .unwrap();
        debug_assert_eq!(entry.point[0], last.point[0]);
        *last = entry;
    }
    pub(super) fn prepend(&mut self, e: Entry<P>) {
        self.left.push(e);
        self.len += 1;
    }
    pub(super) fn append(&mut self, e: Entry<P>) {
        let x = e.point[0];
        if self.blocks.last().is_none_or(|page| page.len() >= CAPACITY) {
            self.blocks.push(Vec::new());
            self.ends.push(x);
        }
        let b = self.blocks.len() - 1;
        self.hint = Some((b, self.blocks[b].len()));
        self.blocks[b].push(e);
        self.ends[b] = x;
        self.len += 1;
    }
    pub(super) fn get(&self, x: f64) -> Option<&Entry<P>> {
        if self.left.first().is_some_and(|e| x <= e.point[0]) {
            let i = self.left.partition_point(|e| e.point[0] > x);
            return self.left.get(i).filter(|e| e.point[0] == x);
        }
        self.at(self.locate(x)).filter(|e| e.point[0] == x)
    }
    pub(super) fn flush_left(&mut self) {
        if self.left.is_empty() {
            return;
        }
        self.hint = None;
        if self.left.len() + self.blocks[0].len() <= CAPACITY {
            self.blocks[0].splice(0..0, self.left.drain(..).rev());
        } else {
            let mut prefix = Vec::with_capacity(self.left.len().div_ceil(CAPACITY));
            let mut iter = self.left.drain(..).rev();
            while iter.len() > 0 {
                prefix.push(iter.by_ref().take(CAPACITY).collect::<Vec<_>>());
            }
            self.ends
                .splice(0..0, prefix.iter().map(|b| b.last().unwrap().point[0]));
            self.blocks.splice(0..0, prefix);
        }
    }
    pub(super) fn locate(&self, x: f64) -> Position {
        if let Some((block, i)) = self.hint {
            let p = &self.blocks[block];
            if i + 1 < p.len() && p[i].point[0] < x && x <= p[i + 1].point[0] {
                return (block, i + 1);
            }
        }
        let b = self.ends.partition_point(|&end| end < x);
        let i = self
            .blocks
            .get(b)
            .map_or(0, |p| p.partition_point(|e| e.point[0] < x));
        (b, i)
    }
    pub(super) fn at(&self, (b, i): Position) -> Option<&Entry<P>> {
        self.blocks.get(b)?.get(i)
    }
    pub(super) fn before(&self, (b, i): Position) -> Option<&Entry<P>> {
        if i > 0 {
            self.at((b, i - 1))
        } else {
            self.blocks.get(b.checked_sub(1)?)?.last()
        }
    }
    pub(super) fn next(&self, (b, i): Position) -> Position {
        if i + 1 == self.blocks[b].len() {
            (b + 1, 0)
        } else {
            (b, i + 1)
        }
    }
    /// Replace a half-open interval, dropping whole dominated pages together.
    pub(super) fn replace(
        &mut self,
        (b, i): Position,
        (end, j): Position,
        removed: usize,
        e: Entry<P>,
    ) {
        if b == self.blocks.len() {
            self.append(e);
            return;
        }
        if b == end {
            self.blocks[b].splice(i..j, std::iter::once(e));
        } else {
            self.blocks[b].truncate(i);
            self.blocks[b].push(e);
            if j > 0 {
                self.blocks[end].drain(..j);
            }
            self.blocks.drain(b + 1..end);
            self.ends.drain(b + 1..end);
        }
        self.ends[b] = self.blocks[b].last().unwrap().point[0];
        self.len = self.len - removed + 1;
        self.hint = Some((b, i));
        if self.blocks[b].len() > CAPACITY {
            let mid = CAPACITY / 2;
            let right = self.blocks[b].split_off(mid);
            self.ends[b] = self.blocks[b].last().unwrap().point[0];
            self.ends.insert(b + 1, right.last().unwrap().point[0]);
            self.blocks.insert(b + 1, right);
            if i >= mid {
                self.hint = Some((b + 1, i - mid));
            }
        }
    }
}
