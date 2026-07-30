//! Exact inverse of the regular two-dimensional grid used by `fcmaes-core`.
//!
//! `Archive::try_new(..., samples_per_niche = 0, ...)` factorizes a requested
//! capacity into `floor(sqrt(capacity))` rows. Rows differ by at most one
//! column when the capacity is not rectangular. Keeping this mapping in one
//! tutorial module prevents diagnostics and exported coordinates from
//! inventing a different tessellation.

/// Row-major two-dimensional archive layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveGrid {
    capacity: usize,
    rows: usize,
    base_columns: usize,
    extra_columns: usize,
}

impl ArchiveGrid {
    /// Reproduce the `fcmaes-core` regular-grid factorization.
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

    /// Total niche count.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Columns in each row, including the ragged prefix when present.
    #[must_use]
    pub fn row_lengths(&self) -> Vec<usize> {
        (0..self.rows).map(|row| self.columns_in_row(row)).collect()
    }

    /// `[columns, rows]` for a rectangular layout, otherwise `None`.
    #[must_use]
    pub fn rectangular_shape(&self) -> Option<[usize; 2]> {
        (self.extra_columns == 0).then_some([self.base_columns, self.rows])
    }

    /// Convert an archive niche id to `(column, row)`.
    #[must_use]
    pub fn coordinate(&self, niche: usize) -> Option<(usize, usize)> {
        if niche >= self.capacity {
            return None;
        }
        let wide_row_size = self.base_columns + 1;
        let wide_prefix = self.extra_columns * wide_row_size;
        if niche < wide_prefix {
            Some((niche % wide_row_size, niche / wide_row_size))
        } else {
            let offset = niche - wide_prefix;
            Some((
                offset % self.base_columns,
                self.extra_columns + offset / self.base_columns,
            ))
        }
    }

    /// Map a bounded descriptor pair to the same niche as `fcmaes-core`.
    #[must_use]
    pub fn niche(&self, descriptors: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> Option<usize> {
        if descriptors
            .iter()
            .zip(lower.iter().zip(upper))
            .any(|(value, (lo, hi))| !value.is_finite() || *value < *lo || *value > hi)
        {
            return None;
        }
        let normalized_y =
            ((descriptors[1] - lower[1]) / (upper[1] - lower[1])).clamp(0.0, 1.0 - f64::EPSILON);
        let row = (normalized_y * self.rows as f64) as usize;
        let columns = self.columns_in_row(row);
        let normalized_x =
            ((descriptors[0] - lower[0]) / (upper[0] - lower[0])).clamp(0.0, 1.0 - f64::EPSILON);
        let column = (normalized_x * columns as f64) as usize;
        Some(row * self.base_columns + row.min(self.extra_columns) + column)
    }

    fn columns_in_row(&self, row: usize) -> usize {
        self.base_columns + usize::from(row < self.extra_columns)
    }
}

#[cfg(test)]
mod tests {
    use fcmaes_core::{Archive, Rng};

    use super::*;

    #[test]
    fn publication_and_smoke_layouts_match_fcmaes_archive_indices() {
        for capacity in [60, 120] {
            let layout = ArchiveGrid::new(capacity);
            let lower = [-52.0, 6.0];
            let upper = [52.0, 14.0];
            let mut rng = Rng::new(42);
            let archive = Archive::try_new(32, &lower, &upper, capacity, 0, &mut rng).unwrap();
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
        assert_eq!(ArchiveGrid::new(120).rectangular_shape(), Some([12, 10]));
        assert_eq!(
            ArchiveGrid::new(60).row_lengths(),
            vec![9, 9, 9, 9, 8, 8, 8]
        );
    }

    #[test]
    fn every_niche_has_a_unique_inverse_coordinate() {
        for capacity in [30, 60, 120] {
            let layout = ArchiveGrid::new(capacity);
            let coordinates = (0..capacity)
                .map(|niche| layout.coordinate(niche).unwrap())
                .collect::<std::collections::HashSet<_>>();
            assert_eq!(coordinates.len(), capacity);
        }
    }
}
