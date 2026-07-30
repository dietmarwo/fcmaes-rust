//! Exact two-dimensional regular-grid mapping used by `fcmaes-core`.

/// Row-major layout of an archive created with zero CVT samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveGrid {
    capacity: usize,
    rows: usize,
    base_columns: usize,
    extra_columns: usize,
}

impl ArchiveGrid {
    /// Reproduce the regular-grid factorization used by `fcmaes-core`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "archive capacity must be positive");
        let rows = (capacity as f64).sqrt().floor().max(1.0) as usize;
        Self {
            capacity,
            rows,
            base_columns: capacity / rows,
            extra_columns: capacity % rows,
        }
    }

    /// Number of archive niches.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of columns in each archive row.
    #[must_use]
    pub fn row_lengths(&self) -> Vec<usize> {
        (0..self.rows).map(|row| self.columns(row)).collect()
    }

    /// Rectangular `[columns, rows]`, if every row has the same length.
    #[must_use]
    pub fn rectangular_shape(&self) -> Option<[usize; 2]> {
        (self.extra_columns == 0).then_some([self.base_columns, self.rows])
    }

    /// Map a bounded descriptor pair to its archive-native niche.
    #[must_use]
    pub fn niche(&self, value: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> Option<usize> {
        if value.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let normalized_y =
            ((value[1] - lower[1]) / (upper[1] - lower[1])).clamp(0.0, 1.0 - f64::EPSILON);
        let row = (normalized_y * self.rows as f64) as usize;
        let columns = self.columns(row);
        let normalized_x =
            ((value[0] - lower[0]) / (upper[0] - lower[0])).clamp(0.0, 1.0 - f64::EPSILON);
        let column = (normalized_x * columns as f64) as usize;
        Some(row * self.base_columns + row.min(self.extra_columns) + column)
    }

    fn columns(&self, row: usize) -> usize {
        self.base_columns + usize::from(row < self.extra_columns)
    }
}

#[cfg(test)]
mod tests {
    use fcmaes_core::{Archive, Rng};

    use super::*;

    #[test]
    fn pilot_layout_matches_the_real_archive() {
        for capacity in [40, 100] {
            let lower = [0.15, 0.08];
            let upper = [0.35, 0.23];
            let layout = ArchiveGrid::new(capacity);
            let mut rng = Rng::new(42);
            let archive = Archive::try_new(28, &lower, &upper, capacity, 0, &mut rng).unwrap();
            for y in 0..101 {
                for x in 0..101 {
                    let descriptor = [
                        lower[0] + (upper[0] - lower[0]) * x as f64 / 100.0,
                        lower[1] + (upper[1] - lower[1]) * y as f64 / 100.0,
                    ];
                    assert_eq!(
                        layout.niche(descriptor, lower, upper),
                        Some(archive.index_of_niche(&descriptor))
                    );
                }
            }
        }
        assert_eq!(ArchiveGrid::new(100).rectangular_shape(), Some([10, 10]));
        assert_eq!(ArchiveGrid::new(40).row_lengths(), vec![7, 7, 7, 7, 6, 6]);
    }
}
