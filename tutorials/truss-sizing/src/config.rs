//! Frozen smoke and publication budgets.

/// Reproducibility preset selected by the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// Fast local and CI execution.
    Smoke,
    /// Frozen tutorial evidence campaign.
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

    /// Resolve all default campaign budgets.
    #[must_use]
    pub const fn protocol(self) -> Protocol {
        match self {
            Self::Smoke => Protocol {
                so_evaluations: 64,
                so_retries: 1,
                pilot_per_arm: 8,
                mo_evaluations: 16,
                mo_population: 8,
            },
            Self::Publication => Protocol {
                so_evaluations: 2_048,
                so_retries: 8,
                pilot_per_arm: 128,
                mo_evaluations: 256,
                mo_population: 32,
            },
        }
    }
}

/// Effective campaign budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Protocol {
    /// Scalar calls requested per optimizer.
    pub so_evaluations: u64,
    /// Scalar retries per optimizer.
    pub so_retries: usize,
    /// Structured descriptor observations attempted per arm.
    pub pilot_per_arm: usize,
    /// MODE candidate-call budget.
    pub mo_evaluations: usize,
    /// MODE population size.
    pub mo_population: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_is_strictly_larger_than_smoke() {
        let smoke = Preset::Smoke.protocol();
        let publication = Preset::Publication.protocol();
        assert!(publication.so_evaluations > smoke.so_evaluations);
        assert!(publication.pilot_per_arm > smoke.pilot_per_arm);
        assert!(publication.mo_evaluations > smoke.mo_evaluations);
    }
}
