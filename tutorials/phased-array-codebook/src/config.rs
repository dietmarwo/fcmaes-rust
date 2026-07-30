//! Shared physical and campaign configuration.

use serde::{Deserialize, Serialize};

/// Hardware quantization used by every optimization arm.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    /// Phase-shifter bit count.
    pub phase_bits: u32,
    /// Attenuator bit count.
    pub attenuator_bits: u32,
    /// Attenuator step in decibels.
    pub attenuator_step_db: f64,
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            phase_bits: 6,
            attenuator_bits: 5,
            attenuator_step_db: 0.5,
        }
    }
}

/// CLI workload preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Small deterministic CI workload.
    Smoke,
    /// Checked-in documentation workload.
    Publication,
}

/// Equal-budget settings used by all result arms.
#[derive(Clone, Copy, Debug)]
pub struct Protocol {
    /// Linear-cut samples.
    pub cut_points: usize,
    /// Per-SO-arm evaluations.
    pub so_evaluations: u64,
    /// Parallel retries per SO arm.
    pub so_retries: usize,
    /// MODE candidate budget.
    pub mo_evaluations: usize,
    /// MODE population size.
    pub mo_population: usize,
    /// MAP-Elites candidate budget.
    pub qd_evaluations: usize,
    /// MAP-Elites archive capacity.
    pub qd_capacity: usize,
    /// MAP-Elites batch size.
    pub qd_chunk_size: usize,
    /// Descriptor-pilot candidates per seed.
    pub pilot_samples: usize,
}

impl Preset {
    /// Return the frozen protocol for this preset.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                cut_points: 721,
                so_evaluations: 600,
                so_retries: 3,
                mo_evaluations: 512,
                mo_population: 64,
                qd_evaluations: 512,
                qd_capacity: 60,
                qd_chunk_size: 32,
                pilot_samples: 180,
            },
            Self::Publication => Protocol {
                cut_points: 4_001,
                so_evaluations: 8_000,
                so_retries: 8,
                mo_evaluations: 8_192,
                mo_population: 128,
                qd_evaluations: 12_000,
                qd_capacity: 120,
                qd_chunk_size: 64,
                pilot_samples: 720,
            },
        }
    }
}
