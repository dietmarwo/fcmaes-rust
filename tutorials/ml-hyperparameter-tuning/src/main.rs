use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ml_hyperparameter_tuning::benchmark::{BenchmarkOptions, run_benchmark};
use ml_hyperparameter_tuning::data::{DataConfig, Dataset, Preset};
use ml_hyperparameter_tuning::objective::Evaluator;
use ml_hyperparameter_tuning::optimize::{
    BaselineMethod, BaselineOptions, BaselineOutcome, MultiOptions, QdOptions, ScalarOptions,
    optimize_baseline, optimize_multi, optimize_qd, optimize_scalar,
};
use ml_hyperparameter_tuning::protocol::{
    FinalArm, FinalArmExclusion, FinalStudyPlan, finalize_study, source_manifest_hash,
};
use ml_hyperparameter_tuning::report::{
    config_summary, peak_rss_kib, revalidate_qd_artifacts, write_baseline_artifacts,
    write_baseline_failure_artifact, write_benchmark_artifacts, write_final_artifacts,
    write_multi_artifacts, write_qd_artifacts, write_scalar_artifacts, write_study_plan,
};
use ml_hyperparameter_tuning::space::default_coordinates;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Evaluate,
    Scalar,
    Multi,
    Qd,
    RevalidateQd,
    Baselines,
    BudgetSweep,
    Benchmark,
    Finalize,
    All,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    preset: Preset,
    workers: usize,
    seed: u64,
    min_recall: f64,
    shortlist: usize,
    evaluations_per_retry: u64,
    retries: usize,
    bite_depth: i32,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    baseline_evaluations: usize,
    benchmark_candidates: usize,
    prediction_repetitions: usize,
    output: PathBuf,
    write_output: bool,
    final_plan: Option<PathBuf>,
}

impl Args {
    fn for_preset(preset: Preset) -> Self {
        match preset {
            Preset::Smoke => Self {
                mode: RunMode::Evaluate,
                preset,
                workers: 4,
                seed: 42,
                min_recall: 0.10,
                shortlist: 4,
                evaluations_per_retry: 8,
                retries: 4,
                bite_depth: 4,
                mo_evaluations: 32,
                popsize: 8,
                qd_evaluations: 32,
                qd_capacity: 16,
                qd_chunk_size: 8,
                baseline_evaluations: 32,
                benchmark_candidates: 6,
                prediction_repetitions: 10,
                output: PathBuf::from("results/quick"),
                write_output: true,
                final_plan: None,
            },
            Preset::Publication => Self {
                mode: RunMode::Evaluate,
                preset,
                workers: 24,
                seed: 42,
                min_recall: 0.25,
                shortlist: 20,
                evaluations_per_retry: 512,
                retries: 24,
                bite_depth: 6,
                mo_evaluations: 16_384,
                popsize: 256,
                qd_evaluations: 16_384,
                qd_capacity: 400,
                qd_chunk_size: 256,
                baseline_evaluations: 2_048,
                benchmark_candidates: 24,
                prediction_repetitions: 100,
                output: PathBuf::from("results/publication"),
                write_output: true,
                final_plan: None,
            },
        }
    }
}

fn usage() {
    println!(
        "Validation-aware ML hyperparameter optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --preset NAME                 smoke or publication (smoke)\n\
         --mode NAME                   evaluate, scalar, mo, qd, revalidate-qd, baselines,\n\
                                       budget-sweep, benchmark, finalize, or all\n\
         --workers N                   Candidate-evaluation workers (preset)\n\
         --seed N                      Optimizer seed (42)\n\
         --min-recall X                Minority-recall feasibility floor (preset)\n\
         --shortlist N                 Unique configurations re-ranked on selection data\n\
         --evaluations-per-retry N     BiteOpt evaluations in every retry\n\
         --retries N                   Independent BiteOpt retries\n\
         --bite-depth N                BiteOpt depth in 1..=36\n\
         --mo-evaluations N            Requested MODE evaluations\n\
         --popsize N                   MODE population size\n\
         --qd-evaluations N            Requested MAP-Elites evaluations\n\
         --qd-capacity N               Square archive capacity\n\
         --qd-chunk-size N             Even QD evaluation batch size\n\
         --baseline-evaluations N      Random/LHS candidate calls\n\
         --benchmark-candidates N      Configurations for isolated benchmarks\n\
         --prediction-repetitions N    Timed prediction repetitions\n\
         --final-plan FILE             Frozen study plan for finalize mode\n\
         --output DIR                  Artifact directory (preset)\n\
         --no-output                   Print results without writing artifacts\n\
         -h, --help                    Show this help\n\
         \n\
         SmartCore tree fits are serial. fcmaes owns candidate-level\n\
         parallelism; no nested ML worker pool is started."
    );
}

fn value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut preset = Preset::Smoke;
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == "--preset" {
            let preset_name = raw.get(index + 1).ok_or("missing value for --preset")?;
            preset = match preset_name.as_str() {
                "smoke" => Preset::Smoke,
                "publication" => Preset::Publication,
                other => return Err(format!("unknown preset: {other}").into()),
            };
            index += 2;
        } else {
            index += 1;
        }
    }
    let mut parsed = Args::for_preset(preset);
    let mut args = raw.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--preset" => {
                let _ = value(&mut args, "--preset")?;
            }
            "--mode" => {
                parsed.mode = match value(&mut args, "--mode")?.as_str() {
                    "evaluate" => RunMode::Evaluate,
                    "scalar" => RunMode::Scalar,
                    "mo" | "multi" => RunMode::Multi,
                    "qd" => RunMode::Qd,
                    "revalidate-qd" => RunMode::RevalidateQd,
                    "baselines" | "baseline" => RunMode::Baselines,
                    "budget-sweep" => RunMode::BudgetSweep,
                    "benchmark" => RunMode::Benchmark,
                    "finalize" => RunMode::Finalize,
                    "all" => RunMode::All,
                    other => return Err(format!("unknown mode: {other}").into()),
                }
            }
            "--workers" => parsed.workers = value(&mut args, "--workers")?.parse()?,
            "--seed" => parsed.seed = value(&mut args, "--seed")?.parse()?,
            "--min-recall" => parsed.min_recall = value(&mut args, "--min-recall")?.parse()?,
            "--shortlist" => parsed.shortlist = value(&mut args, "--shortlist")?.parse()?,
            "--evaluations-per-retry" => {
                parsed.evaluations_per_retry =
                    value(&mut args, "--evaluations-per-retry")?.parse()?
            }
            "--retries" => parsed.retries = value(&mut args, "--retries")?.parse()?,
            "--bite-depth" => parsed.bite_depth = value(&mut args, "--bite-depth")?.parse()?,
            "--mo-evaluations" => {
                parsed.mo_evaluations = value(&mut args, "--mo-evaluations")?.parse()?
            }
            "--popsize" => parsed.popsize = value(&mut args, "--popsize")?.parse()?,
            "--qd-evaluations" => {
                parsed.qd_evaluations = value(&mut args, "--qd-evaluations")?.parse()?
            }
            "--qd-capacity" => parsed.qd_capacity = value(&mut args, "--qd-capacity")?.parse()?,
            "--qd-chunk-size" => {
                parsed.qd_chunk_size = value(&mut args, "--qd-chunk-size")?.parse()?
            }
            "--baseline-evaluations" => {
                parsed.baseline_evaluations = value(&mut args, "--baseline-evaluations")?.parse()?
            }
            "--benchmark-candidates" => {
                parsed.benchmark_candidates = value(&mut args, "--benchmark-candidates")?.parse()?
            }
            "--prediction-repetitions" => {
                parsed.prediction_repetitions =
                    value(&mut args, "--prediction-repetitions")?.parse()?
            }
            "--final-plan" => parsed.final_plan = Some(value(&mut args, "--final-plan")?.into()),
            "--output" => parsed.output = value(&mut args, "--output")?.into(),
            "--no-output" => parsed.write_output = false,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    validate_args(&parsed)?;
    Ok(Some(parsed))
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.workers == 0 {
        return Err("--workers must be positive".into());
    }
    if !(0.0..=1.0).contains(&args.min_recall) {
        return Err("--min-recall must lie in [0, 1]".into());
    }
    if args.shortlist == 0
        || args.evaluations_per_retry == 0
        || args.retries == 0
        || args.mo_evaluations == 0
        || args.qd_evaluations == 0
        || args.baseline_evaluations == 0
        || args.benchmark_candidates < 2
        || args.prediction_repetitions == 0
    {
        return Err("all budgets, retries, and shortlist size must be positive".into());
    }
    if !(1..=36).contains(&args.bite_depth) {
        return Err("--bite-depth must lie in 1..=36".into());
    }
    if args.popsize < 4 {
        return Err("--popsize must be at least four".into());
    }
    if args.qd_chunk_size < 2 || !args.qd_chunk_size.is_multiple_of(2) {
        return Err("--qd-chunk-size must be even and at least two".into());
    }
    let side = (args.qd_capacity as f64).sqrt() as usize;
    if side < 2 || side * side != args.qd_capacity {
        return Err("--qd-capacity must be a perfect square of at least four".into());
    }
    if args.mode == RunMode::Finalize && args.final_plan.is_none() {
        return Err("--final-plan is required in finalize mode".into());
    }
    if args.mode == RunMode::RevalidateQd && !args.write_output {
        return Err(
            "revalidate-qd rewrites QD artifacts and cannot be used with --no-output".into(),
        );
    }
    Ok(())
}

fn selection_seeds(args: &Args) -> Vec<u64> {
    let count = if args.preset == Preset::Smoke { 1 } else { 5 };
    (0..count)
        .map(|index| ml_hyperparameter_tuning::data::stream_seed(args.seed, 100 + index as u64))
        .collect()
}

fn command() -> String {
    std::env::args().collect::<Vec<_>>().join(" ")
}

fn scalar_options(args: &Args) -> ScalarOptions {
    ScalarOptions {
        evaluations_per_retry: args.evaluations_per_retry,
        retries: args.retries,
        workers: args.workers,
        depth: args.bite_depth,
        seed: args.seed,
        shortlist: args.shortlist,
        selection_seeds: selection_seeds(args),
    }
}

fn print_selected(label: &str, selected: &ml_hyperparameter_tuning::optimize::SelectedCandidate) {
    let config = selected
        .tuning
        .config
        .as_ref()
        .expect("selected candidate has config");
    let tuning = selected
        .tuning
        .metrics
        .expect("selected candidate has metrics");
    let selection = selected
        .selection
        .metrics
        .expect("selected candidate has selection metrics");
    println!(
        "{label} tuning_log_loss={:.8} selection_log_loss={:.8} selection_sdev={:.8} selection_pr_auc={:.8} selection_recall={:.6}",
        tuning.log_loss,
        selection.log_loss,
        selected.selection.log_loss_sdev,
        selection.pr_auc,
        selection.recall,
    );
    println!("{label}_CONFIG {}", config_summary(config));
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(args.preset))?);
    let evaluator = Arc::new(Evaluator::new(
        Arc::clone(&dataset),
        args.min_recall,
        ml_hyperparameter_tuning::data::stream_seed(args.seed, 50),
    ));
    println!(
        "DATA preset={} tuning={} selection={} test={} features={} prevalence={:.6} bayes_log_loss={:.8}±{:.8}",
        args.preset.name(),
        dataset.tuning.len(),
        dataset.selection.len(),
        dataset.test.len(),
        ml_hyperparameter_tuning::data::FEATURE_COUNT,
        dataset.tuning.prevalence(),
        dataset.bayes.log_loss,
        dataset.bayes.log_loss_standard_error,
    );

    match args.mode {
        RunMode::Evaluate => {
            let evaluation = evaluator.evaluate(&default_coordinates());
            println!(
                "EVALUATE feasible={} fitness={:.8} model_fits={} trees={} seconds={:.6}",
                evaluation.feasible(),
                evaluation.scalar_fitness,
                evaluation.model_fits,
                evaluation.trees_fitted,
                evaluation.elapsed_seconds,
            );
            if let Some(config) = &evaluation.config {
                println!("EVALUATE_CONFIG {}", config_summary(config));
            }
        }
        RunMode::Scalar => {
            let options = scalar_options(&args);
            let outcome = optimize_scalar(Arc::clone(&evaluator), &options)?;
            print_scalar(&outcome);
            if args.write_output {
                write_scalar_artifacts(&args.output, &dataset, &outcome, &options, &command())?;
            }
        }
        RunMode::Multi => {
            let options = multi_options(&args);
            let outcome = optimize_multi(Arc::clone(&evaluator), &options)?;
            print_multi(&outcome);
            if args.write_output {
                write_multi_artifacts(&args.output, &dataset, &outcome, &options, &command())?;
            }
        }
        RunMode::Qd => {
            let options = qd_options(&args);
            let outcome = optimize_qd(Arc::clone(&evaluator), &options)?;
            print_qd(&outcome);
            if args.write_output {
                write_qd_artifacts(&args.output, &dataset, &outcome, &options, &command())?;
            }
        }
        RunMode::RevalidateQd => {
            let options = qd_options(&args);
            let outcome = revalidate_qd_artifacts(
                &args.output,
                &dataset,
                Arc::clone(&evaluator),
                &options,
                &command(),
            )?;
            println!(
                "QD_REVALIDATED occupied={} retained={} selection_fits={} selection_trees={} seconds={:.6}",
                outcome.occupied,
                outcome.retained_niches,
                outcome.selection_model_fits,
                outcome.selection_trees_fitted,
                outcome.elapsed.as_secs_f64(),
            );
            println!("QD_DECISION {}", outcome.decision);
        }
        RunMode::Baselines => {
            let _ = run_baselines(&args, Arc::clone(&evaluator), &dataset, &args.output)?;
        }
        RunMode::BudgetSweep => {
            run_budget_sweep(&args, Arc::clone(&evaluator), &dataset)?;
        }
        RunMode::Benchmark => {
            let options = BenchmarkOptions {
                candidates: args.benchmark_candidates,
                maximum_workers: args.workers,
                prediction_repetitions: args.prediction_repetitions,
                seed: args.seed,
            };
            let outcome = run_benchmark(Arc::clone(&evaluator), &options)?;
            println!(
                "BENCHMARK latency_samples={} scaling_samples={} seconds={:.6}",
                outcome.latency.len(),
                outcome.scaling.len(),
                outcome.elapsed.as_secs_f64()
            );
            if args.write_output {
                write_benchmark_artifacts(&args.output, &dataset, &outcome, &options, &command())?;
            }
        }
        RunMode::Finalize => {
            let plan_path = args
                .final_plan
                .as_ref()
                .expect("validated final-plan argument");
            let plan: FinalStudyPlan = serde_json::from_str(&fs::read_to_string(plan_path)?)?;
            let result = finalize_study(evaluator, &plan, args.workers)?;
            for arm in &result.arms {
                let metrics = arm.test.metrics.expect("finalized arm has metrics");
                println!(
                    "FINAL arm={} log_loss={:.8} sdev={:.8} pr_auc={:.8} recall={:.6}",
                    arm.name,
                    metrics.log_loss,
                    arm.test.log_loss_sdev,
                    metrics.pr_auc,
                    metrics.recall
                );
            }
            if args.write_output {
                write_final_artifacts(
                    &args.output,
                    &dataset,
                    &plan,
                    &result,
                    args.workers,
                    &command(),
                )?;
            }
        }
        RunMode::All => {
            run_all(&args, evaluator, &dataset)?;
        }
    }
    if let Some(rss) = peak_rss_kib() {
        println!("PEAK_RSS_KIB {rss}");
    }
    Ok(())
}

fn print_scalar(outcome: &ml_hyperparameter_tuning::optimize::ScalarOutcome) {
    println!(
        "SCALAR evaluations={} retries={} tuning_fits={} selection_fits={} tuning_trees={} selection_trees={} duplicates={} seconds={:.6}",
        outcome.evaluations,
        outcome.completed_retries,
        outcome.model_fits,
        outcome.selection_model_fits,
        outcome.trees_fitted,
        outcome.selection_trees_fitted,
        outcome.duplicate_configurations,
        outcome.elapsed.as_secs_f64(),
    );
    print_selected("SCALAR_SELECTED", &outcome.selected);
}

fn multi_options(args: &Args) -> MultiOptions {
    MultiOptions {
        evaluations: args.mo_evaluations,
        popsize: args.popsize,
        workers: args.workers,
        seed: args.seed,
        selection_seeds: selection_seeds(args),
    }
}

fn print_multi(outcome: &ml_hyperparameter_tuning::optimize::MultiOutcome) {
    println!(
        "MODE evaluations={} generations={} pareto={} tuning_fits={} selection_fits={} tuning_trees={} selection_trees={} seconds={:.6}",
        outcome.evaluations,
        outcome.generations,
        outcome.pareto.len(),
        outcome.model_fits,
        outcome.selection_model_fits,
        outcome.trees_fitted,
        outcome.selection_trees_fitted,
        outcome.elapsed.as_secs_f64(),
    );
    let selected = ml_hyperparameter_tuning::optimize::SelectedCandidate {
        tuning: outcome.representative.tuning.clone(),
        selection: outcome.representative.selection.clone(),
    };
    print_selected("MODE_SELECTED", &selected);
}

fn qd_options(args: &Args) -> QdOptions {
    let selection_seeds = selection_seeds(args);
    let apply_publication_criteria = args.preset == Preset::Publication
        && args.workers == 24
        && args.qd_evaluations == 16_384
        && args.qd_capacity == 400
        && args.qd_chunk_size == 256
        && (args.min_recall - 0.25).abs() < f64::EPSILON
        && selection_seeds.len() == 5;
    QdOptions {
        evaluations: args.qd_evaluations,
        capacity: args.qd_capacity,
        chunk_size: args.qd_chunk_size,
        workers: args.workers,
        seed: args.seed,
        selection_seeds,
        apply_publication_criteria,
    }
}

fn print_qd(outcome: &ml_hyperparameter_tuning::optimize::QdOutcome) {
    println!(
        "QD evaluations={} occupied={} capacity={} coverage={:.6} retained={} distinct_configs={} tuning_fits={} selection_fits={} tuning_trees={} selection_trees={} seconds={:.6}",
        outcome.evaluations,
        outcome.occupied,
        outcome.capacity,
        outcome.occupied as f64 / outcome.capacity as f64,
        outcome.retained_niches,
        outcome.distinct_configurations,
        outcome.model_fits,
        outcome.selection_model_fits,
        outcome.trees_fitted,
        outcome.selection_trees_fitted,
        outcome.elapsed.as_secs_f64(),
    );
    println!("QD_DECISION {}", outcome.decision);
}

fn baseline_options(args: &Args, method: BaselineMethod, evaluations: usize) -> BaselineOptions {
    BaselineOptions {
        method,
        evaluations,
        workers: args.workers,
        seed: args.seed,
        shortlist: args.shortlist,
        selection_seeds: selection_seeds(args),
    }
}

struct BaselineStage {
    outcomes: Vec<BaselineOutcome>,
    excluded_arms: Vec<FinalArmExclusion>,
}

fn run_baselines(
    args: &Args,
    evaluator: Arc<Evaluator>,
    dataset: &Dataset,
    output: &Path,
) -> Result<BaselineStage, Box<dyn Error>> {
    let mut outcomes = Vec::new();
    let mut excluded_arms = Vec::new();
    for method in [
        BaselineMethod::Default,
        BaselineMethod::Random,
        BaselineMethod::LatinHypercube,
    ] {
        let evaluations = if method == BaselineMethod::Default {
            1
        } else {
            args.baseline_evaluations
        };
        let options = baseline_options(args, method, evaluations);
        // Arms fail independently. The single-candidate default arm is
        // infeasible on the publication data, and aborting the whole stage for
        // it would discard the random and Latin-hypercube arms that did
        // succeed. A skipped arm gets both a manifest and a study-plan
        // exclusion, rather than disappearing into terminal output.
        let outcome = match optimize_baseline(Arc::clone(&evaluator), &options) {
            Ok(outcome) => outcome,
            Err(error) => {
                let reason = error.to_string();
                let source_run = output.join(method.name());
                println!("BASELINE_SKIPPED method={} reason={reason}", method.name());
                if args.write_output {
                    write_baseline_failure_artifact(
                        &source_run,
                        dataset,
                        &options,
                        &command(),
                        &reason,
                        evaluator.as_ref(),
                    )?;
                }
                excluded_arms.push(FinalArmExclusion {
                    name: format!("baseline-{}", method.name()),
                    source_run: source_run.display().to_string(),
                    reason,
                });
                continue;
            }
        };
        println!(
            "BASELINE method={} evaluations={} tuning_fits={} selection_fits={} tuning_trees={} selection_trees={} duplicates={} seconds={:.6}",
            method.name(),
            outcome.evaluations,
            outcome.model_fits,
            outcome.selection_model_fits,
            outcome.trees_fitted,
            outcome.selection_trees_fitted,
            outcome.duplicate_configurations,
            outcome.elapsed.as_secs_f64(),
        );
        print_selected(
            &format!("BASELINE_{}_SELECTED", method.name().to_uppercase()),
            &outcome.selected,
        );
        if args.write_output {
            write_baseline_artifacts(
                &output.join(method.name()),
                dataset,
                &outcome,
                &options,
                &command(),
            )?;
        }
        outcomes.push(outcome);
    }
    if outcomes.is_empty() {
        return Err("no baseline arm produced a feasible configuration".into());
    }
    Ok(BaselineStage {
        outcomes,
        excluded_arms,
    })
}

fn run_all(
    args: &Args,
    evaluator: Arc<Evaluator>,
    dataset: &Dataset,
) -> Result<(), Box<dyn Error>> {
    let scalar_options = scalar_options(args);
    let scalar = optimize_scalar(Arc::clone(&evaluator), &scalar_options)?;
    print_scalar(&scalar);
    let multi_options = multi_options(args);
    let multi = optimize_multi(Arc::clone(&evaluator), &multi_options)?;
    print_multi(&multi);
    let qd_options = qd_options(args);
    let qd = optimize_qd(Arc::clone(&evaluator), &qd_options)?;
    print_qd(&qd);
    let baselines = run_baselines(
        args,
        Arc::clone(&evaluator),
        dataset,
        &args.output.join("baselines"),
    )?;
    if args.write_output {
        write_scalar_artifacts(
            &args.output.join("scalar"),
            dataset,
            &scalar,
            &scalar_options,
            &command(),
        )?;
        write_multi_artifacts(
            &args.output.join("mo"),
            dataset,
            &multi,
            &multi_options,
            &command(),
        )?;
        write_qd_artifacts(
            &args.output.join("qd"),
            dataset,
            &qd,
            &qd_options,
            &command(),
        )?;
        let mut arms = vec![
            final_arm(
                "fcmaes-scalar",
                args.output.join("scalar"),
                scalar
                    .selected
                    .tuning
                    .config
                    .clone()
                    .expect("selected scalar config"),
            )?,
            final_arm(
                "fcmaes-mode-representative",
                args.output.join("mo"),
                multi
                    .representative
                    .tuning
                    .config
                    .clone()
                    .expect("selected MODE config"),
            )?,
            final_arm(
                "fcmaes-qd-representative",
                args.output.join("qd"),
                qd.representative
                    .tuning
                    .config
                    .clone()
                    .expect("selected QD config"),
            )?,
        ];
        for outcome in baselines.outcomes {
            arms.push(final_arm(
                &format!("baseline-{}", outcome.method.name()),
                args.output.join("baselines").join(outcome.method.name()),
                outcome
                    .selected
                    .tuning
                    .config
                    .expect("selected baseline config"),
            )?);
        }
        let plan = FinalStudyPlan {
            schema_version: 1,
            frozen: false,
            data_hashes: dataset.hashes(),
            final_model_seeds: (0..if args.preset == Preset::Smoke { 1 } else { 5 })
                .map(|index| {
                    ml_hyperparameter_tuning::data::stream_seed(args.seed, 200 + index as u64)
                })
                .collect(),
            arms,
            excluded_arms: baselines.excluded_arms,
        };
        write_study_plan(&args.output.join("study-plan.json"), &plan)?;
        println!(
            "STUDY_PLAN {} (review it, set frozen=true, then run --mode finalize)",
            args.output.join("study-plan.json").display()
        );
    }
    Ok(())
}

fn final_arm(
    name: &str,
    source_run: PathBuf,
    config: ml_hyperparameter_tuning::space::ForestConfig,
) -> Result<FinalArm, Box<dyn Error>> {
    Ok(FinalArm {
        name: name.to_string(),
        source_manifest_hash: source_manifest_hash(&source_run)?,
        source_run: source_run.display().to_string(),
        config,
    })
}

fn run_budget_sweep(
    args: &Args,
    evaluator: Arc<Evaluator>,
    dataset: &Dataset,
) -> Result<(), Box<dyn Error>> {
    let budgets: Vec<usize> = if args.preset == Preset::Smoke {
        vec![8, 16, 32]
    } else {
        vec![256, 2_048, 16_384]
    };
    let directory = &args.output;
    if args.write_output {
        fs::create_dir_all(directory)?;
    }
    let mut summary = String::from(
        "comparison,budget,target_wall_seconds,method,actual_evaluations,wall_seconds,model_fits,tuning_model_fits,selection_model_fits,trees_fitted,tuning_trees_fitted,selection_trees_fitted,tuning_log_loss,selection_log_loss,selection_log_loss_sdev\n",
    );
    let mut total_evaluations = 0usize;
    let mut total_elapsed = 0.0;
    for budget in budgets {
        let retries = args.retries.min(budget).max(1);
        let evaluations_per_retry = (budget / retries).max(1) as u64;
        let scalar_options = ScalarOptions {
            evaluations_per_retry,
            retries,
            workers: args.workers,
            depth: args.bite_depth,
            seed: args.seed,
            shortlist: args.shortlist,
            selection_seeds: selection_seeds(args),
        };
        let scalar = optimize_scalar(Arc::clone(&evaluator), &scalar_options)?;
        append_study_row(
            &mut summary,
            "equal-calls",
            budget,
            None,
            "fcmaes-biteopt",
            scalar.evaluations as usize,
            scalar.elapsed.as_secs_f64(),
            scalar.model_fits,
            scalar.selection_model_fits,
            scalar.trees_fitted,
            scalar.selection_trees_fitted,
            &scalar.selected,
        )?;
        append_study_row(
            &mut summary,
            "calibrated-wall",
            budget,
            Some(scalar.elapsed.as_secs_f64()),
            "fcmaes-biteopt",
            scalar.evaluations as usize,
            scalar.elapsed.as_secs_f64(),
            scalar.model_fits,
            scalar.selection_model_fits,
            scalar.trees_fitted,
            scalar.selection_trees_fitted,
            &scalar.selected,
        )?;
        total_evaluations += scalar.evaluations as usize;
        total_elapsed += scalar.elapsed.as_secs_f64();
        if args.write_output {
            write_scalar_artifacts(
                &directory.join(format!("{budget}-fcmaes")),
                dataset,
                &scalar,
                &scalar_options,
                &command(),
            )?;
        }
        for method in [BaselineMethod::Random, BaselineMethod::LatinHypercube] {
            let options = baseline_options(args, method, scalar.evaluations as usize);
            let outcome = optimize_baseline(Arc::clone(&evaluator), &options)?;
            append_study_row(
                &mut summary,
                "equal-calls",
                budget,
                None,
                method.name(),
                outcome.evaluations,
                outcome.elapsed.as_secs_f64(),
                outcome.model_fits,
                outcome.selection_model_fits,
                outcome.trees_fitted,
                outcome.selection_trees_fitted,
                &outcome.selected,
            )?;
            total_evaluations += outcome.evaluations;
            total_elapsed += outcome.elapsed.as_secs_f64();
            if args.write_output {
                write_baseline_artifacts(
                    &directory.join(format!("{budget}-{}", method.name())),
                    dataset,
                    &outcome,
                    &options,
                    &command(),
                )?;
            }

            // The equal-call run is the timing pilot. Scale the second-stage
            // call count once, then report both target and achieved wall time;
            // do not claim that calibration makes the times identical.
            let pilot_seconds = outcome.elapsed.as_secs_f64().max(1.0e-9);
            let scaled = (outcome.evaluations as f64 * scalar.elapsed.as_secs_f64() / pilot_seconds)
                .round()
                .max(1.0) as usize;
            if scaled > 10_000_000 {
                return Err(format!(
                    "calibrated wall-time budget of {scaled} evaluations is unsafe; \
                     inspect the timing pilot"
                )
                .into());
            }
            let wall_options = baseline_options(args, method, scaled);
            let wall_outcome = optimize_baseline(Arc::clone(&evaluator), &wall_options)?;
            append_study_row(
                &mut summary,
                "calibrated-wall",
                budget,
                Some(scalar.elapsed.as_secs_f64()),
                method.name(),
                wall_outcome.evaluations,
                wall_outcome.elapsed.as_secs_f64(),
                wall_outcome.model_fits,
                wall_outcome.selection_model_fits,
                wall_outcome.trees_fitted,
                wall_outcome.selection_trees_fitted,
                &wall_outcome.selected,
            )?;
            total_evaluations += wall_outcome.evaluations;
            total_elapsed += wall_outcome.elapsed.as_secs_f64();
            if args.write_output {
                write_baseline_artifacts(
                    &directory.join(format!("{budget}-{}-wall", method.name())),
                    dataset,
                    &wall_outcome,
                    &wall_options,
                    &command(),
                )?;
            }
        }
    }
    print!("{summary}");
    if args.write_output {
        fs::write(directory.join("budget_summary.csv"), summary)?;
        let manifest = serde_json::json!({
            "schema_version": 1,
            "tutorial": "ml-hyperparameter-tuning",
            "formulation": "budget-sweep",
            "command": command(),
            "seed": args.seed,
            "workers": args.workers,
            "requested_evaluations": 0,
            "actual_evaluations": total_evaluations,
            "elapsed_seconds": total_elapsed,
            "software": {
                "tutorial": env!("CARGO_PKG_VERSION"),
                "fcmaes-core": fcmaes_core::CORE_VERSION,
                "smartcore": "0.5.3",
            },
            "objective_protocol": {
                "tuning_model_seed": evaluator.tuning_model_seed,
                "common_random_numbers": true,
                "minority_recall_floor": evaluator.min_recall,
                "structural_cost_limit": evaluator.forest.structural_cost_limit,
                "parallelism_owner": "fcmaes candidate workers",
                "inner_model_workers": 1,
            },
            "objectives": [],
            "descriptors": [],
            "table_rows_duplicate_reference_arm": true,
            "comparison_protocols": [
                "equal candidate calls",
                "one-pilot calibrated wall time"
            ],
            "artifacts": {"budget_summary": "budget_summary.csv"}
        });
        fs::write(
            directory.join("run.json"),
            serde_json::to_string_pretty(&manifest)? + "\n",
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_study_row(
    output: &mut String,
    comparison: &str,
    budget: usize,
    target_wall_seconds: Option<f64>,
    method: &str,
    evaluations: usize,
    wall_seconds: f64,
    tuning_model_fits: usize,
    selection_model_fits: usize,
    tuning_trees_fitted: usize,
    selection_trees_fitted: usize,
    selected: &ml_hyperparameter_tuning::optimize::SelectedCandidate,
) -> Result<(), std::fmt::Error> {
    let target_wall_seconds =
        target_wall_seconds.map_or_else(String::new, |value| value.to_string());
    let model_fits = tuning_model_fits + selection_model_fits;
    let trees_fitted = tuning_trees_fitted + selection_trees_fitted;
    writeln!(
        output,
        "{comparison},{budget},{target_wall_seconds},{method},{evaluations},{wall_seconds},{model_fits},{tuning_model_fits},{selection_model_fits},{trees_fitted},{tuning_trees_fitted},{selection_trees_fitted},{},{},{}",
        selected
            .tuning
            .metrics
            .expect("selected tuning metrics")
            .log_loss,
        selected
            .selection
            .metrics
            .expect("selected selection metrics")
            .log_loss,
        selected.selection.log_loss_sdev,
    )
}
