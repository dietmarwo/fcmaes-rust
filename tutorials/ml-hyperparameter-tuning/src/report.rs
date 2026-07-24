use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::thread;

use serde_json::json;

use crate::benchmark::{BenchmarkOptions, BenchmarkOutcome};
use crate::data::Dataset;
use crate::metrics::Metrics;
use crate::objective::LOG_LOSS_CEILING;
use crate::optimize::{
    BaselineOptions, BaselineOutcome, MultiOptions, MultiOutcome, QD_DESCRIPTOR_LOWER,
    QD_DESCRIPTOR_UPPER, QdOptions, QdOutcome, ScalarOptions, ScalarOutcome, SelectedCandidate,
};
use crate::protocol::{FinalStudyPlan, FinalStudyResult};
use crate::space::{DECISION_NAMES, ForestConfig};

pub fn effective_workers(workers: usize) -> usize {
    if workers == 0 {
        thread::available_parallelism().map_or(1, usize::from)
    } else {
        workers
    }
}

fn write_selected(directory: &Path, selected: &SelectedCandidate) -> Result<(), Box<dyn Error>> {
    fs::write(
        directory.join("selected.json"),
        serde_json::to_string_pretty(selected)? + "\n",
    )?;
    Ok(())
}

fn dataset_metadata(dataset: &Dataset) -> serde_json::Value {
    json!({
        "generator_version": 1,
        "feature_count": crate::data::FEATURE_COUNT,
        "tuning_rows": dataset.tuning.len(),
        "selection_rows": dataset.selection.len(),
        "test_rows": dataset.test.len(),
        "tuning_folds": dataset.folds.len(),
        "data_seed": dataset.config.data_seed,
        "tuning_prevalence": dataset.tuning.prevalence(),
        "selection_prevalence": dataset.selection.prevalence(),
        "test_prevalence": dataset.test.prevalence(),
        "hashes": dataset.hashes(),
        "bayes_reference": dataset.bayes,
    })
}

fn software_metadata() -> serde_json::Value {
    json!({
        "tutorial": env!("CARGO_PKG_VERSION"),
        "fcmaes-core": "0.1.2",
        "smartcore": "0.5.3",
    })
}

fn objective_protocol(
    tuning_model_seed: u64,
    min_recall: f64,
    structural_cost_limit: u64,
) -> serde_json::Value {
    json!({
        "tuning_model_seed": tuning_model_seed,
        "common_random_numbers": true,
        "minority_recall_floor": min_recall,
        "structural_cost_limit": structural_cost_limit,
        "infeasible_fitness_base": LOG_LOSS_CEILING + 1.0,
        "probability_clip": 1.0e-6,
        "parallelism_owner": "fcmaes candidate workers",
        "inner_model_workers": 1,
    })
}

fn candidate_csv(
    trace: &[crate::objective::CandidateEvaluation],
) -> Result<String, std::fmt::Error> {
    let mut csv = String::from(
        "candidate_id,feasible,scalar_fitness,log_loss,brier,pr_auc,roc_auc,ece,recall,precision,predicted_positive_rate,false_positives,false_negatives,mean_model_bytes,mean_structural_cost,estimated_structural_cost,recall_violation,structural_violation,model_fits,trees_fitted,elapsed_seconds,failure",
    );
    for name in DECISION_NAMES {
        write!(csv, ",decision_{name}")?;
    }
    csv.push('\n');
    for (index, evaluation) in trace.iter().enumerate() {
        let metrics = evaluation.metrics.unwrap_or_default();
        write!(
            csv,
            "{index},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            u8::from(evaluation.feasible()),
            evaluation.scalar_fitness,
            metric_or_nan(evaluation.metrics, |value| value.log_loss),
            metric_or_nan(evaluation.metrics, |value| value.brier),
            metric_or_nan(evaluation.metrics, |value| value.pr_auc),
            metric_or_nan(evaluation.metrics, |value| value.roc_auc),
            metric_or_nan(evaluation.metrics, |value| value.ece),
            metric_or_nan(evaluation.metrics, |value| value.recall),
            metric_or_nan(evaluation.metrics, |value| value.precision),
            metric_or_nan(evaluation.metrics, |value| value.predicted_positive_rate),
            metrics.false_positives,
            metrics.false_negatives,
            evaluation.mean_model_bytes,
            evaluation.mean_structural_cost,
            evaluation.estimated_structural_cost,
            evaluation.recall_violation,
            evaluation.structural_violation,
            evaluation.model_fits,
            evaluation.trees_fitted,
            evaluation.elapsed_seconds,
            evaluation
                .failure
                .as_ref()
                .map_or_else(String::new, |failure| format!("{failure:?}")),
        )?;
        if let Some(config) = &evaluation.config {
            for value in config.as_decisions() {
                write!(csv, ",{value}")?;
            }
        } else {
            for _ in DECISION_NAMES {
                csv.push_str(",NaN");
            }
        }
        csv.push('\n');
    }
    Ok(csv)
}

fn metric_or_nan(metrics: Option<Metrics>, field: impl FnOnce(Metrics) -> f64) -> f64 {
    metrics.map_or(f64::NAN, field)
}

pub fn write_scalar_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &ScalarOutcome,
    options: &ScalarOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    fs::write(
        directory.join("candidates.csv"),
        candidate_csv(&outcome.trace)?,
    )?;
    write_selected(directory, &outcome.selected)?;
    let mut convergence = String::from("evaluations,elapsed_seconds,best_quality\n");
    for improvement in &outcome.improvements {
        writeln!(
            convergence,
            "{},{},{}",
            improvement.evaluations, improvement.elapsed_seconds, -improvement.value
        )?;
    }
    if outcome.improvements.is_empty() {
        writeln!(
            convergence,
            "{},{},{}",
            outcome.evaluations,
            outcome.elapsed.as_secs_f64(),
            -outcome.selected.tuning.scalar_fitness
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": "scalar",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.retries * options.evaluations_per_retry as usize,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "model_fits": outcome.model_fits + outcome.selection_model_fits,
        "tuning_model_fits": outcome.model_fits,
        "selection_model_fits": outcome.selection_model_fits,
        "trees_fitted": outcome.trees_fitted + outcome.selection_trees_fitted,
        "tuning_trees_fitted": outcome.trees_fitted,
        "selection_trees_fitted": outcome.selection_trees_fitted,
        "software": software_metadata(),
        "objective_protocol": objective_protocol(
            outcome.tuning_model_seed,
            outcome.min_recall,
            outcome.structural_cost_limit,
        ),
        "duplicate_configurations": outcome.duplicate_configurations,
        "dataset": dataset_metadata(dataset),
        "optimizer": {
            "algorithm": "BiteOpt parallel retry",
            "retries": options.retries,
            "evaluations_per_retry": options.evaluations_per_retry,
            "completed_retries": outcome.completed_retries,
            "depth": options.depth,
            "shortlist": options.shortlist,
            "selection_seeds": options.selection_seeds,
        },
        "selected": outcome.selected,
        "objectives": [],
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": {
            "candidates": "candidates.csv",
            "selected": "selected.json",
            "convergence": "convergence.csv",
        },
    });
    write_manifest(directory, &manifest)
}

pub fn write_baseline_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &BaselineOutcome,
    options: &BaselineOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    fs::write(
        directory.join("candidates.csv"),
        candidate_csv(&outcome.trace)?,
    )?;
    write_selected(directory, &outcome.selected)?;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": format!("baseline-{}", outcome.method.name()),
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "model_fits": outcome.model_fits + outcome.selection_model_fits,
        "tuning_model_fits": outcome.model_fits,
        "selection_model_fits": outcome.selection_model_fits,
        "trees_fitted": outcome.trees_fitted + outcome.selection_trees_fitted,
        "tuning_trees_fitted": outcome.trees_fitted,
        "selection_trees_fitted": outcome.selection_trees_fitted,
        "software": software_metadata(),
        "objective_protocol": objective_protocol(
            outcome.tuning_model_seed,
            outcome.min_recall,
            outcome.structural_cost_limit,
        ),
        "duplicate_configurations": outcome.duplicate_configurations,
        "dataset": dataset_metadata(dataset),
        "optimizer": {
            "algorithm": outcome.method.name(),
            "shortlist": options.shortlist,
            "selection_seeds": options.selection_seeds,
        },
        "selected": outcome.selected,
        "objectives": [],
        "descriptors": [],
        "artifacts": {
            "candidates": "candidates.csv",
            "selected": "selected.json",
        },
    });
    write_manifest(directory, &manifest)
}

pub fn write_multi_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &MultiOutcome,
    options: &MultiOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_negative_pr_auc,objective_brier,objective_model_bytes,objective_inference_work,constraint_recall,constraint_structural,selection_log_loss,selection_log_loss_sdev",
    );
    for name in DECISION_NAMES {
        write!(pareto, ",decision_{name}")?;
    }
    pareto.push('\n');
    for (index, point) in outcome.pareto.iter().enumerate() {
        let tuning = point
            .tuning
            .metrics
            .expect("Pareto point has tuning metrics");
        write!(
            pareto,
            "{index},1,{},{},{},{},{},{},{},{},{}",
            u8::from(point.selected),
            -tuning.pr_auc,
            tuning.brier,
            point.tuning.mean_model_bytes,
            point.tuning.mean_structural_cost,
            point.tuning.recall_violation,
            point.tuning.structural_violation,
            point.selection.score(),
            point.selection.log_loss_sdev,
        )?;
        for value in point
            .tuning
            .config
            .as_ref()
            .expect("Pareto point has config")
            .as_decisions()
        {
            write!(pareto, ",{value}")?;
        }
        pareto.push('\n');
    }
    fs::write(directory.join("pareto.csv"), pareto)?;
    let mut convergence =
        String::from("evaluations,elapsed_seconds,best_quality,feasible_population\n");
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.best_quality,
            sample.feasible_population
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": "mo",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "model_fits": outcome.model_fits + outcome.selection_model_fits,
        "tuning_model_fits": outcome.model_fits,
        "selection_model_fits": outcome.selection_model_fits,
        "trees_fitted": outcome.trees_fitted + outcome.selection_trees_fitted,
        "tuning_trees_fitted": outcome.trees_fitted,
        "selection_trees_fitted": outcome.selection_trees_fitted,
        "software": software_metadata(),
        "objective_protocol": objective_protocol(
            outcome.tuning_model_seed,
            outcome.min_recall,
            outcome.structural_cost_limit,
        ),
        "dataset": dataset_metadata(dataset),
        "optimizer": {
            "algorithm": "constrained MODE",
            "popsize": options.popsize,
            "selection_seeds": options.selection_seeds,
        },
        "objectives": [
            {"column": "objective_negative_pr_auc", "label": "Negative PR-AUC"},
            {"column": "objective_brier", "label": "Brier score"},
            {"column": "objective_model_bytes", "label": "Serialized model size", "unit": "bytes"},
            {"column": "objective_inference_work", "label": "Tree-depth work proxy"},
        ],
        "descriptors": [],
        "convergence_metrics": ["best_quality", "feasible_population"],
        "artifacts": {
            "pareto": "pareto.csv",
            "convergence": "convergence.csv",
        },
    });
    write_manifest(directory, &manifest)
}

pub fn write_qd_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &QdOutcome,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,descriptor_predicted_positive_rate_train,descriptor_error_ratio_train,descriptor_predicted_positive_rate_validation,descriptor_error_ratio_validation,visit_count,retained_niche",
    );
    for name in DECISION_NAMES {
        write!(archive, ",decision_{name}")?;
    }
    archive.push('\n');
    for point in &outcome.elites {
        write!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.tuning.scalar_fitness,
            point.selection.score(),
            point.descriptors_training[0],
            point.descriptors_training[1],
            point.descriptors_selection[0],
            point.descriptors_selection[1],
            point.visit_count,
            u8::from(point.retained_niche),
        )?;
        for value in point
            .tuning
            .config
            .as_ref()
            .expect("QD point has config")
            .as_decisions()
        {
            write!(archive, ",{value}")?;
        }
        archive.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive)?;
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let side = (outcome.capacity as f64).sqrt() as usize;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": "qd",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "validation_elapsed_seconds": outcome.validation_elapsed.as_secs_f64(),
        "model_fits": outcome.model_fits + outcome.selection_model_fits,
        "tuning_model_fits": outcome.model_fits,
        "selection_model_fits": outcome.selection_model_fits,
        "trees_fitted": outcome.trees_fitted + outcome.selection_trees_fitted,
        "tuning_trees_fitted": outcome.trees_fitted,
        "selection_trees_fitted": outcome.selection_trees_fitted,
        "software": software_metadata(),
        "objective_protocol": objective_protocol(
            outcome.tuning_model_seed,
            outcome.min_recall,
            outcome.structural_cost_limit,
        ),
        "dataset": dataset_metadata(dataset),
        "qd_decision": outcome.decision,
        "descriptors": [
            {
                "column": "descriptor_predicted_positive_rate",
                "label": "Predicted positive rate",
                "bounds": [QD_DESCRIPTOR_LOWER[0], QD_DESCRIPTOR_UPPER[0]]
            },
            {
                "column": "descriptor_error_ratio",
                "label": "log10 false-positive / false-negative ratio",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            },
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_validation_column": "quality_validation",
            "quality_label": "Cross-validated log-loss (lower is better)",
            "occupied": outcome.occupied,
            "distinct_configurations": outcome.distinct_configurations,
            "retained_niches": outcome.retained_niches,
            "invalid_evaluations": outcome.invalid_evaluations,
            "clipped_descriptors": outcome.clipped_descriptors,
            "selection_seeds": options.selection_seeds,
        },
        "convergence_metrics": ["coverage", "qd_score", "best_quality", "invalid_fraction"],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv",
        },
    });
    write_manifest(directory, &manifest)
}

pub fn write_final_artifacts(
    directory: &Path,
    dataset: &Dataset,
    plan: &FinalStudyPlan,
    result: &FinalStudyResult,
    workers: usize,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    fs::write(
        directory.join("study-plan.json"),
        serde_json::to_string_pretty(plan)? + "\n",
    )?;
    let mut csv = String::from(
        "arm,source_run,log_loss,log_loss_sdev,brier,pr_auc,roc_auc,ece,recall,precision,predicted_positive_rate,mean_model_bytes,mean_structural_cost,model_fits,trees_fitted,elapsed_seconds\n",
    );
    for arm in &result.arms {
        let metrics = arm.test.metrics.expect("final arm has metrics");
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            arm.name,
            arm.source_run,
            metrics.log_loss,
            arm.test.log_loss_sdev,
            metrics.brier,
            metrics.pr_auc,
            metrics.roc_auc,
            metrics.ece,
            metrics.recall,
            metrics.precision,
            metrics.predicted_positive_rate,
            arm.test.mean_model_bytes,
            arm.test.mean_structural_cost,
            arm.test.model_fits,
            arm.test.trees_fitted,
            arm.test.elapsed_seconds,
        )?;
    }
    fs::write(directory.join("final_metrics.csv"), csv)?;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": "final-test",
        "command": command,
        "seed": dataset.config.data_seed,
        "workers": effective_workers(workers),
        "requested_evaluations": result.arms.len(),
        "actual_evaluations": result.arms.len(),
        "elapsed_seconds": result.arms.iter().map(|arm| arm.test.elapsed_seconds).sum::<f64>(),
        "software": software_metadata(),
        "dataset": dataset_metadata(dataset),
        "final_model_seeds": result.final_model_seeds,
        "objectives": [],
        "descriptors": [],
        "artifacts": {
            "study_plan": "study-plan.json",
            "final_metrics": "final_metrics.csv",
        },
    });
    write_manifest(directory, &manifest)
}

pub fn write_benchmark_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &BenchmarkOutcome,
    options: &BenchmarkOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut latency =
        String::from("candidate_id,model_bytes,structural_cost,fit_seconds,microseconds_per_row");
    for name in DECISION_NAMES {
        write!(latency, ",decision_{name}")?;
    }
    latency.push('\n');
    for sample in &outcome.latency {
        write!(
            latency,
            "{},{},{},{},{}",
            sample.candidate_id,
            sample.model_bytes,
            sample.structural_cost,
            sample.fit_seconds,
            sample.microseconds_per_row
        )?;
        for value in sample.config.as_decisions() {
            write!(latency, ",{value}")?;
        }
        latency.push('\n');
    }
    fs::write(directory.join("latency.csv"), latency)?;
    let mut scaling =
        String::from("workers,candidates,wall_seconds,candidates_per_second,peak_rss_kib\n");
    for sample in &outcome.scaling {
        writeln!(
            scaling,
            "{},{},{},{},{}",
            sample.workers,
            sample.candidates,
            sample.wall_seconds,
            sample.candidates_per_second,
            sample
                .peak_rss_kib
                .map_or_else(String::new, |value| value.to_string())
        )?;
    }
    fs::write(directory.join("parallel_scaling.csv"), scaling)?;
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "ml-hyperparameter-tuning",
        "formulation": "benchmark",
        "command": command,
        "seed": options.seed,
        "workers": options.maximum_workers,
        "requested_evaluations": options.candidates,
        "actual_evaluations": outcome.latency.len(),
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "software": software_metadata(),
        "dataset": dataset_metadata(dataset),
        "benchmark": {
            "prediction_repetitions": options.prediction_repetitions,
            "latency_isolated": true,
            "parallelism_owner": "fcmaes candidate pool",
            "inner_model_workers": 1,
        },
        "objectives": [],
        "descriptors": [],
        "artifacts": {
            "latency": "latency.csv",
            "parallel_scaling": "parallel_scaling.csv",
        },
    });
    write_manifest(directory, &manifest)
}

fn write_manifest(directory: &Path, manifest: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(manifest)? + "\n",
    )?;
    Ok(())
}

pub fn write_study_plan(path: &Path, plan: &FinalStudyPlan) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(plan)? + "\n")?;
    Ok(())
}

pub fn peak_rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

pub fn config_summary(config: &ForestConfig) -> String {
    format!(
        "trees={} depth={} leaf={} split={} rows={:.3} features={:.3} positive_weight={:.3} criterion={:?}",
        config.n_trees,
        config.max_depth,
        config.min_samples_leaf,
        config.min_samples_split,
        config.row_sample_fraction,
        config.feature_fraction,
        config.positive_sampling_weight,
        config.criterion,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Dataset, Preset};
    use crate::objective::Evaluator;
    use crate::optimize::{BaselineMethod, optimize_baseline};
    use std::sync::Arc;

    #[test]
    fn baseline_writer_emits_a_valid_manifest() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Arc::new(Evaluator::new(Arc::clone(&dataset), 0.1, 42));
        let options = BaselineOptions {
            method: BaselineMethod::Default,
            evaluations: 1,
            workers: 1,
            seed: 42,
            shortlist: 1,
            selection_seeds: vec![101],
        };
        let outcome = optimize_baseline(evaluator, &options).unwrap();
        let directory = std::env::temp_dir().join(format!(
            "fcmaes-hpo-report-{}-{}",
            std::process::id(),
            crate::data::stream_seed(42, 9)
        ));
        write_baseline_artifacts(&directory, &dataset, &outcome, &options, "test").unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(directory.join("run.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema_version"], 1);
        assert!(directory.join("candidates.csv").is_file());
        fs::remove_dir_all(directory).unwrap();
    }
}
