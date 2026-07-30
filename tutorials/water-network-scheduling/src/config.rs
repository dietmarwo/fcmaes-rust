//! Frozen reproducibility budgets.

/// Workload preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Fast CI and local validation.
    Smoke,
    /// Checked-in publication protocol.
    Publication,
}

/// Protocol budgets.
#[derive(Clone, Copy, Debug)]
pub struct Protocol {
    pub so_evaluations: u64,
    pub so_retries: usize,
    pub pilot_samples: usize,
    pub qd_evaluations: usize,
    pub qd_capacity: usize,
    pub mo_evaluations: usize,
    pub mo_population: usize,
    pub parallel_benchmark_candidates: usize,
}

impl Preset {
    /// Return the frozen protocol for this preset.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                so_evaluations: 150,
                so_retries: 2,
                pilot_samples: 120,
                qd_evaluations: 240,
                qd_capacity: 40,
                mo_evaluations: 240,
                mo_population: 40,
                parallel_benchmark_candidates: 100,
            },
            Self::Publication => Protocol {
                so_evaluations: 4_000,
                so_retries: 8,
                pilot_samples: 1_500,
                qd_evaluations: 6_000,
                qd_capacity: 100,
                mo_evaluations: 6_000,
                mo_population: 100,
                parallel_benchmark_candidates: 10_000,
            },
        }
    }
}
