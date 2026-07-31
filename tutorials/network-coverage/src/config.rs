//! Frozen smoke and publication budgets.

/// CLI reproducibility preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Fast local and CI checks.
    Smoke,
    /// Frozen evidence campaign.
    Publication,
}

impl Preset {
    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Publication => "publication",
        }
    }

    /// Resolve the preset's budgets.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                so_evaluations: 128,
                so_retries: 1,
                mo_evaluations: 128,
                mo_sensitivity_evaluations: 0,
                mo_population: 16,
                throughput_samples: 128,
            },
            Self::Publication => Protocol {
                so_evaluations: 8_192,
                so_retries: 8,
                mo_evaluations: 8_192,
                mo_sensitivity_evaluations: 200_000,
                mo_population: 64,
                throughput_samples: 2_048,
            },
        }
    }
}

/// Effective run budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Protocol {
    /// Candidate calls requested for each scalar arm.
    pub so_evaluations: u64,
    /// Independent scalar retries.
    pub so_retries: usize,
    /// Candidate calls requested from MODE.
    pub mo_evaluations: usize,
    /// Additive high-budget MODE check; zero disables it.
    pub mo_sensitivity_evaluations: usize,
    /// MODE population size.
    pub mo_population: usize,
    /// Candidates in each throughput measurement.
    pub throughput_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_adds_a_strictly_larger_budget_sensitivity_run() {
        let smoke = Preset::Smoke.protocol();
        let publication = Preset::Publication.protocol();
        assert_eq!(smoke.mo_sensitivity_evaluations, 0);
        assert!(publication.mo_sensitivity_evaluations > publication.mo_evaluations);
    }
}
