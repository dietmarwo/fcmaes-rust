//! Shared tutorial configuration.

/// Value of lost electrical load in currency units per kWh.
pub const ELECTRICITY_VOLL: f64 = 10.0;
/// Curtailment has no artificial economic penalty, preserving capacity monotonicity.
pub const CURTAILMENT_COST: f64 = 0.0;

/// CLI workload preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Small end-to-end CI workload.
    Smoke,
    /// Checked-in tutorial workload.
    Publication,
}

/// Frozen optimizer and horizon settings.
#[derive(Clone, Copy, Debug)]
pub struct Protocol {
    /// Representative days in one main dispatch solve.
    pub representative_days: usize,
    /// Requested scalar evaluations per arm.
    pub so_evaluations: u64,
    /// Scalar retry count.
    pub so_retries: usize,
    /// Structured pilot samples per seed.
    pub pilot_samples: usize,
    /// QD candidate budget.
    pub qd_evaluations: usize,
    /// QD archive capacity.
    pub qd_capacity: usize,
    /// QD chunk size.
    pub qd_chunk_size: usize,
    /// MODE candidate budget.
    pub mo_evaluations: usize,
    /// MODE population.
    pub mo_population: usize,
    /// Coarse annual BiteOpt budget.
    pub annual_evaluations: u64,
}

impl Preset {
    /// Frozen settings for this preset.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                representative_days: 4,
                so_evaluations: 48,
                so_retries: 2,
                pilot_samples: 24,
                qd_evaluations: 96,
                qd_capacity: 60,
                qd_chunk_size: 16,
                mo_evaluations: 128,
                mo_population: 32,
                annual_evaluations: 8,
            },
            Self::Publication => Protocol {
                representative_days: 12,
                so_evaluations: 160,
                so_retries: 4,
                pilot_samples: 80,
                qd_evaluations: 256,
                qd_capacity: 120,
                qd_chunk_size: 32,
                mo_evaluations: 384,
                mo_population: 64,
                annual_evaluations: 32,
            },
        }
    }

    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Publication => "publication",
        }
    }
}
