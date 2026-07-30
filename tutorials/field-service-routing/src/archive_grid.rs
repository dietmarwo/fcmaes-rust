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

    /// Map a descriptor pair to its archive-native niche.
    #[must_use]
    pub fn niche(&self, value: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> Option<usize> {
        self.position(value, lower, upper)
            .map(|(column, row)| self.offset(row) + column)
    }

    /// Return archive-native column and row coordinates.
    #[must_use]
    pub fn coordinates(
        &self,
        value: [f64; 2],
        lower: [f64; 2],
        upper: [f64; 2],
    ) -> Option<[usize; 2]> {
        self.position(value, lower, upper)
            .map(|(column, row)| [column, row])
    }

    fn position(
        &self,
        value: [f64; 2],
        lower: [f64; 2],
        upper: [f64; 2],
    ) -> Option<(usize, usize)> {
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
        Some((column, row))
    }

    fn columns(&self, row: usize) -> usize {
        self.base_columns + usize::from(row < self.extra_columns)
    }

    fn offset(&self, row: usize) -> usize {
        row * self.base_columns + row.min(self.extra_columns)
    }
}

#[cfg(test)]
mod tests {
    use fcmaes_core::{Archive, Rng};

    use super::*;

    #[test]
    fn pilot_layout_matches_the_real_archive() {
        for capacity in [60, 120] {
            let lower = [3.5, 0.0];
            let upper = [8.5, 1.0];
            let layout = ArchiveGrid::new(capacity);
            let mut rng = Rng::new(42);
            let archive = Archive::try_new(104, &lower, &upper, capacity, 0, &mut rng).unwrap();
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
        assert_eq!(
            ArchiveGrid::new(60).row_lengths(),
            vec![9, 9, 9, 9, 8, 8, 8]
        );
        assert_eq!(ArchiveGrid::new(120).row_lengths(), vec![12; 10]);
    }
}
