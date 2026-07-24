use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use fcmaes_core::parallel_batch;
use serde::{Deserialize, Serialize};

use crate::data::DatasetHashes;
use crate::objective::{Evaluator, ValidationEvaluation};
use crate::space::ForestConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalArm {
    pub name: String,
    pub source_run: String,
    pub source_manifest_hash: String,
    pub config: ForestConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalStudyPlan {
    pub schema_version: u32,
    pub frozen: bool,
    pub data_hashes: DatasetHashes,
    pub final_model_seeds: Vec<u64>,
    pub arms: Vec<FinalArm>,
}

impl FinalStudyPlan {
    pub fn validate(&self, expected_hashes: &DatasetHashes) -> Result<(), &'static str> {
        if self.schema_version != 1 {
            return Err("unsupported final-study schema version");
        }
        if !self.frozen {
            return Err("final-study plan must be frozen before test evaluation");
        }
        if &self.data_hashes != expected_hashes {
            return Err("final-study data hashes do not match generated data");
        }
        if self.final_model_seeds.is_empty() {
            return Err("final-study plan requires at least one model seed");
        }
        if self.arms.is_empty() {
            return Err("final-study plan requires at least one arm");
        }
        if self.arms.iter().any(|arm| {
            arm.name.trim().is_empty()
                || arm.source_run.trim().is_empty()
                || arm.source_manifest_hash.trim().is_empty()
        }) {
            return Err("every final-study arm needs a name, source run, and manifest hash");
        }
        let unique_names: HashSet<&str> = self.arms.iter().map(|arm| arm.name.as_str()).collect();
        if unique_names.len() != self.arms.len() {
            return Err("final-study arm names must be unique");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalArmResult {
    pub name: String,
    pub source_run: String,
    pub config: ForestConfig,
    pub test: ValidationEvaluation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalStudyResult {
    pub data_hashes: DatasetHashes,
    pub final_model_seeds: Vec<u64>,
    pub arms: Vec<FinalArmResult>,
}

pub fn finalize_study(
    evaluator: Arc<Evaluator>,
    plan: &FinalStudyPlan,
    workers: usize,
) -> Result<FinalStudyResult, Box<dyn Error>> {
    let hashes = evaluator.dataset.hashes();
    plan.validate(&hashes)?;
    for arm in &plan.arms {
        let actual_hash = source_manifest_hash(Path::new(&arm.source_run))?;
        if actual_hash != arm.source_manifest_hash {
            return Err(format!(
                "source manifest hash mismatch for frozen arm '{}'",
                arm.name
            )
            .into());
        }
    }
    let arms = parallel_batch(&plan.arms, workers as i32, |arm| FinalArmResult {
        name: arm.name.clone(),
        source_run: arm.source_run.clone(),
        config: arm.config.clone(),
        test: evaluator.evaluate_final(&arm.config, &plan.final_model_seeds),
    });
    if arms.iter().any(|arm| arm.test.metrics.is_none()) {
        return Err("at least one frozen arm failed final-test evaluation".into());
    }
    Ok(FinalStudyResult {
        data_hashes: hashes,
        final_model_seeds: plan.final_model_seeds.clone(),
        arms,
    })
}

pub fn source_manifest_hash(source_run: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(source_run.join("run.json"))?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!("{hash:016x}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Dataset, Preset};

    #[test]
    fn finalization_requires_a_frozen_matching_plan() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Arc::new(Evaluator::new(Arc::clone(&dataset), 0.1, 42));
        let source_run =
            std::env::temp_dir().join(format!("fcmaes-hpo-finalize-{}", std::process::id()));
        fs::create_dir_all(&source_run).unwrap();
        fs::write(source_run.join("run.json"), b"{\"frozen\":true}\n").unwrap();
        let mut plan = FinalStudyPlan {
            schema_version: 1,
            frozen: false,
            data_hashes: dataset.hashes(),
            final_model_seeds: vec![201],
            arms: vec![FinalArm {
                name: "default".to_string(),
                source_run: source_run.display().to_string(),
                source_manifest_hash: source_manifest_hash(&source_run).unwrap(),
                config: ForestConfig {
                    n_trees: 8,
                    max_depth: 4,
                    ..ForestConfig::default_config()
                },
            }],
        };
        assert!(finalize_study(Arc::clone(&evaluator), &plan, 1).is_err());
        plan.frozen = true;
        let result = finalize_study(evaluator, &plan, 1).unwrap();
        assert_eq!(result.arms.len(), 1);
        assert!(result.arms[0].test.metrics.is_some());
        fs::write(source_run.join("run.json"), b"{\"tampered\":true}\n").unwrap();
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Arc::new(Evaluator::new(dataset, 0.1, 42));
        assert!(finalize_study(evaluator, &plan, 1).is_err());
        fs::remove_dir_all(source_run).unwrap();
    }
}
