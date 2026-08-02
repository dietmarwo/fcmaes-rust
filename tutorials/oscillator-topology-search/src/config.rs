//! Frozen experiment presets and common-random-number streams.

use serde::{Deserialize, Serialize};

/// Samples retained after burn-in.
pub const SAMPLE_COUNT: usize = 128;
/// Sampling interval in model time units.
pub const SAMPLE_INTERVAL: f64 = 1.0;
/// Burn-in time omitted from spectral scoring.
pub const BURN_IN: f64 = 64.0;
/// Initial copy number for all three gene products.
pub const INITIAL_COPIES: isize = 10;
/// Fixed half-saturation count from the Python precedent.
pub const HILL_K: f64 = 20.0;
/// Physical guard used to reject runaway simulations.
pub const MAX_MOLECULES: f64 = 100_000.0;
/// Expected period used by the scalar inner objective.
pub const TARGET_PERIOD: f64 = 24.0;

/// Named common-random-number streams used during optimization.
pub const TRAINING_SEEDS: [u64; 8] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
];

/// Disjoint streams reserved for validation.
pub const VALIDATION_SEEDS: [u64; 8] = [
    0xa458_fea3_f493_3d7e,
    0x0d95_748f_728e_b658,
    0x718b_cd58_8215_4aee,
    0x7b54_a41d_c25a_59b5,
    0x9c30_d539_2af2_6013,
    0xc5d1_b023_2860_85f0,
    0xca41_7918_b8db_38ef,
    0x8e79_dcb0_603a_180e,
];

/// Physical cores available to the process, capped by its logical CPU quota.
pub fn physical_workers() -> usize {
    num_cpus::get_physical().max(1).min(num_cpus::get().max(1))
}

/// Reproducible experiment size.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    Smoke,
    Publication,
}

impl Preset {
    /// Parse a command-line value.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "publication" => Some(Self::Publication),
            _ => None,
        }
    }

    /// Artifact-directory label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Smoke => "ci-smoke",
            Self::Publication => "publication",
        }
    }
}

/// Fixed budgets for one run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Protocol {
    /// Preset default; a larger resumed campaign target is recorded separately.
    #[serde(rename = "preset_candidates_per_arm", alias = "candidates_per_arm")]
    pub candidates_per_arm: usize,
    /// Independent BiteOpt restarts assigned to each proposed topology.
    pub inner_retries: usize,
    /// Worker threads for the independent inner restarts; zero uses available CPUs.
    pub workers: i32,
    /// Evaluation budget for each inner restart.
    pub inner_evaluations: u64,
    pub training_replications: usize,
    pub validation_replications: usize,
    pub bite_depth: i32,
}

impl Protocol {
    /// Protocol associated with a preset.
    pub fn for_preset(preset: Preset) -> Self {
        let physical_workers = physical_workers();
        match preset {
            Preset::Smoke => Self {
                candidates_per_arm: 3,
                inner_retries: physical_workers,
                workers: physical_workers as i32,
                inner_evaluations: 48,
                training_replications: 1,
                validation_replications: 2,
                bite_depth: 2,
            },
            Preset::Publication => Self {
                candidates_per_arm: 20,
                inner_retries: physical_workers,
                workers: physical_workers as i32,
                inner_evaluations: 480,
                training_replications: 2,
                validation_replications: 5,
                bite_depth: 6,
            },
        }
    }

    /// Total requested optimizer evaluations assigned to one topology.
    pub fn requested_evaluations_per_topology(self) -> u64 {
        self.inner_evaluations
            .saturating_mul(self.inner_retries as u64)
    }

    /// Effective number of concurrent inner restart workers.
    pub fn resolved_workers(self) -> usize {
        if self.inner_retries <= 1 {
            return 1;
        }
        let requested = if self.workers <= 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            self.workers as usize
        };
        requested.max(1).min(self.inner_retries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sets_are_disjoint() {
        assert!(
            TRAINING_SEEDS
                .iter()
                .all(|seed| !VALIDATION_SEEDS.contains(seed))
        );
    }

    #[test]
    fn inner_worker_budget_is_bounded_by_retries() {
        let mut protocol = Protocol::for_preset(Preset::Smoke);
        protocol.inner_retries = 4;
        protocol.workers = 32;
        protocol.inner_evaluations = 25;
        assert_eq!(protocol.resolved_workers(), 4);
        assert_eq!(protocol.requested_evaluations_per_topology(), 100);
    }

    #[test]
    fn presets_default_to_physical_cores() {
        let physical = physical_workers();
        for preset in [Preset::Smoke, Preset::Publication] {
            let protocol = Protocol::for_preset(preset);
            assert_eq!(protocol.inner_retries, physical);
            assert_eq!(protocol.workers, physical as i32);
            assert_eq!(protocol.resolved_workers(), physical);
        }
    }

    #[test]
    fn serialized_protocol_labels_the_candidate_count_as_a_preset_default() {
        let protocol = Protocol::for_preset(Preset::Smoke);
        let mut value = serde_json::to_value(protocol).unwrap();
        assert_eq!(
            value["preset_candidates_per_arm"],
            serde_json::Value::from(protocol.candidates_per_arm)
        );
        assert!(value.get("candidates_per_arm").is_none());

        let object = value.as_object_mut().unwrap();
        let count = object.remove("preset_candidates_per_arm").unwrap();
        object.insert("candidates_per_arm".to_owned(), count);
        assert_eq!(serde_json::from_value::<Protocol>(value).unwrap(), protocol);
    }
}
