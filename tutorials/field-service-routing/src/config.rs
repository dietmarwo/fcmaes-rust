//! Frozen campaign budgets.

/// CLI workload preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Small deterministic CI run.
    Smoke,
    /// Checked-in documentation run.
    Publication,
}

/// Reproducible optimization budgets.
#[derive(Clone, Copy, Debug)]
pub struct Protocol {
    /// Candidate calls per scalar arm.
    pub so_evaluations: u64,
    /// Retry count per scalar arm.
    pub so_retries: usize,
    /// Descriptor-pilot samples.
    pub pilot_samples: usize,
    /// MAP-Elites calls.
    pub qd_evaluations: usize,
    /// MAP-Elites archive capacity.
    pub qd_capacity: usize,
    /// MAP-Elites batch size.
    pub qd_chunk_size: usize,
    /// MODE calls.
    pub mo_evaluations: usize,
    /// MODE population.
    pub mo_population: usize,
    /// Frozen local-search move budget per instance.
    pub baseline_moves: usize,
}

impl Preset {
    /// Return the workload for this preset.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                so_evaluations: 600,
                so_retries: 3,
                pilot_samples: 240,
                qd_evaluations: 640,
                qd_capacity: 60,
                qd_chunk_size: 32,
                mo_evaluations: 640,
                mo_population: 64,
                baseline_moves: 5_000,
            },
            Self::Publication => Protocol {
                so_evaluations: 10_000,
                so_retries: 10,
                pilot_samples: 2_000,
                qd_evaluations: 16_000,
                qd_capacity: 120,
                qd_chunk_size: 64,
                mo_evaluations: 12_000,
                mo_population: 120,
                baseline_moves: 100_000,
            },
        }
    }
}
