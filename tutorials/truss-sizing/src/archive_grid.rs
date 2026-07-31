//! Deterministic regular-grid factorization and niche mapping.

/// Two-dimensional grid derived from one archive capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveGrid {
    /// Horizontal cells.
    pub columns: usize,
    /// Vertical cells.
    pub rows: usize,
}

impl ArchiveGrid {
    /// Factor a capacity into the closest rectangular shape.
    #[must_use]
    pub fn from_capacity(capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return None;
        }
        let mut rows = (capacity as f64).sqrt().floor() as usize;
        while rows > 1 && !capacity.is_multiple_of(rows) {
            rows -= 1;
        }
        Some(Self {
            columns: capacity / rows,
            rows,
        })
    }

    /// Total cell count.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.columns * self.rows
    }

    /// Map bounded descriptors to an archive-native row-major niche.
    #[must_use]
    pub fn niche(self, descriptors: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> Option<usize> {
        if descriptors
            .iter()
            .chain(lower.iter())
            .chain(upper.iter())
            .any(|value| !value.is_finite())
            || lower[0] >= upper[0]
            || lower[1] >= upper[1]
        {
            return None;
        }
        let x = (((descriptors[0] - lower[0]) / (upper[0] - lower[0])).clamp(0.0, 1.0)
            * self.columns as f64)
            .floor() as usize;
        let y = (((descriptors[1] - lower[1]) / (upper[1] - lower[1])).clamp(0.0, 1.0)
            * self.rows as f64)
            .floor() as usize;
        Some(x.min(self.columns - 1) + self.columns * y.min(self.rows - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_capacity_has_one_canonical_shape() {
        let grid = ArchiveGrid::from_capacity(120).unwrap();
        assert_eq!(
            grid,
            ArchiveGrid {
                columns: 12,
                rows: 10
            }
        );
        assert_eq!(grid.capacity(), 120);
        assert_eq!(grid.niche([1.0, 1.0], [0.0, 0.0], [1.0, 1.0]), Some(119));
    }
}
