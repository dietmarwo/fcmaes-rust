use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, RetryBounds,
    RetryConfig, RetryImprovement, RetryRunResult, Rng, map_elites_batch_with_progress,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use serde::{Deserialize, Serialize};

use crate::objective::{CandidateEvaluation, Evaluator, ValidationEvaluation};
use crate::space::{DIMENSION, ForestConfig, LOWER_BOUNDS, UPPER_BOUNDS, default_coordinates};

const OBJECTIVES: usize = 4;
const CONSTRAINTS: usize = 2;
/// Behavior-space bounds for `[precision, sharpness]`, frozen from a recorded
/// range pilot rather than guessed: 1,280 uniform-random and Latin-hypercube
/// publication candidates produced 271 feasible designs spanning precision
/// 0.2654–0.4648 and sharpness 0.1210–0.3987. The frozen rectangle adds a
/// small margin so MAP-Elites can push past the randomly sampled extremes
/// without most of either axis being structurally unreachable.
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [0.24, 0.10];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [0.52, 0.45];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedCandidate {
    pub tuning: CandidateEvaluation,
    pub selection: ValidationEvaluation,
}

#[derive(Clone, Debug)]
pub struct ScalarOptions {
    pub evaluations_per_retry: u64,
    pub retries: usize,
    pub workers: usize,
    pub depth: i32,
    pub seed: u64,
    pub shortlist: usize,
    pub selection_seeds: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct ScalarOutcome {
    pub optimizer_best_x: Vec<f64>,
    pub selected: SelectedCandidate,
    pub shortlist: Vec<SelectedCandidate>,
    pub trace: Vec<CandidateEvaluation>,
    pub evaluations: u64,
    pub completed_retries: usize,
    pub duplicate_configurations: usize,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub selection_model_fits: usize,
    pub selection_trees_fitted: usize,
    pub tuning_model_seed: u64,
    pub min_recall: f64,
    pub structural_cost_limit: u64,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    evaluator: Arc<Evaluator>,
    options: &ScalarOptions,
) -> Result<ScalarOutcome, Box<dyn Error>> {
    if options.evaluations_per_retry == 0 || options.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    if !(1..=36).contains(&options.depth) {
        return Err("BiteOpt depth must lie in 1..=36".into());
    }
    let trace = Arc::new(Mutex::new(Vec::new()));
    let objective = {
        let evaluator = Arc::clone(&evaluator);
        let trace = Arc::clone(&trace);
        move |values: &[f64]| {
            let evaluation = evaluator.evaluate(values);
            let fitness = evaluation.scalar_fitness;
            trace
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(evaluation);
            fitness
        }
    };
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let config = RetryConfig {
        num_retries: options.retries,
        workers: options.workers,
        capacity: options.retries.min(500),
        max_evaluations: options.evaluations_per_retry,
        seed: options.seed,
        statistic_num: 1_000,
        ..Default::default()
    };
    let started = Instant::now();
    let result = retry(&objective, &bounds, &config, |objective, context| {
        let mut rng = Rng::new(context.seed);
        let guess: Vec<f64> = (0..DIMENSION).map(|_| rng.uniform01()).collect();
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(&guess),
            &BiteParams {
                max_evaluations: context.max_evaluations,
                seed: rng.next_u64(),
                runid: context.run_id as i64,
                ..Default::default()
            },
            options.depth,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("BiteOpt retry returned no finite hyperparameter configuration".into());
    }
    drop(objective);
    let trace = Arc::try_unwrap(trace)
        .map_err(|_| "scalar trace is still shared")?
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let shortlist_evaluation = select_shortlist(
        &trace,
        &evaluator,
        options.shortlist,
        &options.selection_seeds,
        options.workers,
    )?;
    let shortlist = shortlist_evaluation.candidates;
    let selected = shortlist
        .first()
        .cloned()
        .ok_or("no feasible scalar configuration survived selection")?;
    let duplicate_configurations = duplicate_count(&trace);
    let model_fits = trace.iter().map(|evaluation| evaluation.model_fits).sum();
    let trees_fitted = trace.iter().map(|evaluation| evaluation.trees_fitted).sum();
    let selection_model_fits = shortlist_evaluation.model_fits;
    let selection_trees_fitted = shortlist_evaluation.trees_fitted;
    Ok(ScalarOutcome {
        optimizer_best_x: result.x,
        selected,
        shortlist,
        trace,
        evaluations: result.evaluations,
        completed_retries: result.runs,
        duplicate_configurations,
        model_fits,
        trees_fitted,
        selection_model_fits,
        selection_trees_fitted,
        tuning_model_seed: evaluator.tuning_model_seed,
        min_recall: evaluator.min_recall,
        structural_cost_limit: evaluator.forest.structural_cost_limit,
        elapsed: started.elapsed(),
        improvements: result.improvements,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineMethod {
    Random,
    LatinHypercube,
    Default,
}

impl BaselineMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::LatinHypercube => "lhs",
            Self::Default => "default",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BaselineOptions {
    pub method: BaselineMethod,
    pub evaluations: usize,
    pub workers: usize,
    pub seed: u64,
    pub shortlist: usize,
    pub selection_seeds: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct BaselineOutcome {
    pub method: BaselineMethod,
    pub selected: SelectedCandidate,
    pub shortlist: Vec<SelectedCandidate>,
    pub trace: Vec<CandidateEvaluation>,
    pub evaluations: usize,
    pub duplicate_configurations: usize,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub selection_model_fits: usize,
    pub selection_trees_fitted: usize,
    pub tuning_model_seed: u64,
    pub min_recall: f64,
    pub structural_cost_limit: u64,
    pub elapsed: Duration,
}

pub fn optimize_baseline(
    evaluator: Arc<Evaluator>,
    options: &BaselineOptions,
) -> Result<BaselineOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("baseline evaluations must be positive".into());
    }
    let candidates = baseline_candidates(options.method, options.evaluations, options.seed);
    let started = Instant::now();
    let trace = parallel_batch(&candidates, options.workers as i32, |values| {
        evaluator.evaluate(values)
    });
    let shortlist_evaluation = select_shortlist(
        &trace,
        &evaluator,
        options.shortlist,
        &options.selection_seeds,
        options.workers,
    )?;
    let shortlist = shortlist_evaluation.candidates;
    let selected = shortlist
        .first()
        .cloned()
        .ok_or("no feasible baseline configuration survived selection")?;
    Ok(BaselineOutcome {
        method: options.method,
        selected,
        evaluations: trace.len(),
        duplicate_configurations: duplicate_count(&trace),
        model_fits: trace.iter().map(|evaluation| evaluation.model_fits).sum(),
        trees_fitted: trace.iter().map(|evaluation| evaluation.trees_fitted).sum(),
        selection_model_fits: shortlist_evaluation.model_fits,
        selection_trees_fitted: shortlist_evaluation.trees_fitted,
        tuning_model_seed: evaluator.tuning_model_seed,
        min_recall: evaluator.min_recall,
        structural_cost_limit: evaluator.forest.structural_cost_limit,
        elapsed: started.elapsed(),
        shortlist,
        trace,
    })
}

fn baseline_candidates(method: BaselineMethod, evaluations: usize, seed: u64) -> Vec<Vec<f64>> {
    match method {
        BaselineMethod::Default => vec![default_coordinates().to_vec()],
        BaselineMethod::Random => {
            let mut rng = Rng::new(seed);
            (0..evaluations)
                .map(|_| (0..DIMENSION).map(|_| rng.uniform01()).collect())
                .collect()
        }
        BaselineMethod::LatinHypercube => latin_hypercube(evaluations, seed),
    }
}

fn latin_hypercube(samples: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut result: Vec<Vec<f64>> = (0..samples)
        .map(|_| Vec::with_capacity(DIMENSION))
        .collect();
    for _ in 0..DIMENSION {
        let mut strata: Vec<usize> = (0..samples).collect();
        shuffle(&mut strata, &mut rng);
        for (row, stratum) in result.iter_mut().zip(strata) {
            row.push((stratum as f64 + rng.uniform01()) / samples.max(1) as f64);
        }
    }
    result
}

fn shuffle(values: &mut [usize], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let swap = rng.int_below((index + 1) as i64) as usize;
        values.swap(index, swap);
    }
}

fn duplicate_count(trace: &[CandidateEvaluation]) -> usize {
    let mut unique = HashSet::new();
    trace
        .iter()
        .filter_map(|evaluation| evaluation.config.as_ref())
        .filter(|config| !unique.insert(config.canonical_key()))
        .count()
}

struct ShortlistEvaluation {
    candidates: Vec<SelectedCandidate>,
    model_fits: usize,
    trees_fitted: usize,
}

fn select_shortlist(
    trace: &[CandidateEvaluation],
    evaluator: &Evaluator,
    limit: usize,
    selection_seeds: &[u64],
    workers: usize,
) -> Result<ShortlistEvaluation, Box<dyn Error>> {
    let mut sorted: Vec<CandidateEvaluation> = trace
        .iter()
        .filter(|evaluation| evaluation.feasible())
        .cloned()
        .collect();
    sorted.sort_by(|left, right| {
        left.scalar_fitness
            .total_cmp(&right.scalar_fitness)
            .then_with(|| {
                left.mean_structural_cost
                    .total_cmp(&right.mean_structural_cost)
            })
    });
    let mut unique = HashSet::new();
    sorted.retain(|evaluation| {
        evaluation
            .config
            .as_ref()
            .is_some_and(|config| unique.insert(config.canonical_key()))
    });
    sorted.truncate(limit.max(1));
    let mut selected = parallel_batch(&sorted, workers as i32, |tuning| {
        let config = tuning
            .config
            .as_ref()
            .expect("shortlisted evaluation has a configuration");
        SelectedCandidate {
            tuning: tuning.clone(),
            selection: evaluator.evaluate_selection(config, selection_seeds),
        }
    });
    let model_fits = selected
        .iter()
        .map(|candidate| candidate.selection.model_fits)
        .sum();
    let trees_fitted = selected
        .iter()
        .map(|candidate| candidate.selection.trees_fitted)
        .sum();
    selected.retain(|candidate| {
        candidate
            .selection
            .metrics
            .is_some_and(|metrics| metrics.recall >= evaluator.min_recall)
    });
    selected.sort_by(|left, right| {
        left.selection
            .score()
            .total_cmp(&right.selection.score())
            .then_with(|| {
                left.selection
                    .mean_structural_cost
                    .total_cmp(&right.selection.mean_structural_cost)
            })
    });
    Ok(ShortlistEvaluation {
        candidates: selected,
        model_fits,
        trees_fitted,
    })
}

#[derive(Clone, Debug)]
pub struct MultiOptions {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: usize,
    pub seed: u64,
    pub selection_seeds: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub best_quality: f64,
    pub feasible_population: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub tuning: CandidateEvaluation,
    pub selection: ValidationEvaluation,
    pub selected: bool,
}

#[derive(Clone, Debug)]
pub struct MultiOutcome {
    pub pareto: Vec<ParetoPoint>,
    pub representative: ParetoPoint,
    pub evaluations: usize,
    pub generations: usize,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub selection_model_fits: usize,
    pub selection_trees_fitted: usize,
    pub tuning_model_seed: u64,
    pub min_recall: f64,
    pub structural_cost_limit: u64,
    pub elapsed: Duration,
    pub convergence: Vec<MoProgress>,
}

pub fn optimize_multi(
    evaluator: Arc<Evaluator>,
    options: &MultiOptions,
) -> Result<MultiOutcome, Box<dyn Error>> {
    if options.evaluations == 0 || options.popsize < 4 {
        return Err("MODE needs positive evaluations and popsize >= 4".into());
    }
    let generations = options.evaluations.div_ceil(options.popsize);
    let actual_evaluations = generations * options.popsize;
    let fitness = Fitness::bounded(
        DIMENSION,
        OBJECTIVES + CONSTRAINTS,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
    );
    let parameters = ModeParams {
        popsize: options.popsize as i32,
        nsga_update: true,
        seed: options.seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, OBJECTIVES, CONSTRAINTS, None, &parameters)?;
    let started = Instant::now();
    let mut convergence = Vec::with_capacity(generations);
    let model_fits = AtomicUsize::new(0);
    let trees_fitted = AtomicUsize::new(0);
    let mut evaluated_by_point = HashMap::new();
    for generation in 0..generations {
        let candidates = mode.ask();
        let evaluations = parallel_batch(&candidates, options.workers as i32, |values| {
            evaluator.evaluate(values)
        });
        model_fits.fetch_add(
            evaluations
                .iter()
                .map(|evaluation| evaluation.model_fits)
                .sum(),
            Ordering::Relaxed,
        );
        trees_fitted.fetch_add(
            evaluations
                .iter()
                .map(|evaluation| evaluation.trees_fitted)
                .sum(),
            Ordering::Relaxed,
        );
        let feasible_population = evaluations
            .iter()
            .filter(|evaluation| evaluation.feasible())
            .count();
        let best_quality = evaluations
            .iter()
            .filter(|evaluation| evaluation.feasible())
            .filter_map(|evaluation| evaluation.metrics)
            .map(|metrics| -metrics.log_loss)
            .fold(f64::NEG_INFINITY, f64::max);
        let values: Vec<Vec<f64>> = evaluations
            .iter()
            .map(CandidateEvaluation::mode_values)
            .collect();
        for (candidate, evaluation) in candidates.iter().zip(&evaluations) {
            evaluated_by_point.insert(point_key(candidate), evaluation.clone());
        }
        mode.try_tell(&values)?;
        convergence.push(MoProgress {
            evaluations: (generation + 1) * options.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality,
            feasible_population,
        });
    }
    let population = mode.population();
    let evaluations: Vec<CandidateEvaluation> = population
        .iter()
        .map(|values| {
            evaluated_by_point
                .get(&point_key(values))
                .cloned()
                .ok_or("MODE final population contains an unevaluated point")
        })
        .collect::<Result<_, _>>()?;
    let feasible: Vec<(usize, Vec<f64>)> = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.feasible())
        .map(|(index, evaluation)| (index, evaluation.mode_values()[..OBJECTIVES].to_vec()))
        .collect();
    if feasible.is_empty() {
        return Err("MODE returned no feasible hyperparameter configuration".into());
    }
    let objective_rows: Vec<Vec<f64>> = feasible.iter().map(|(_, row)| row.clone()).collect();
    let local_indices = pareto_indices(&objective_rows, OBJECTIVES)?;
    let training_points: Vec<CandidateEvaluation> = local_indices
        .iter()
        .map(|&local| evaluations[feasible[local].0].clone())
        .collect();
    let validated = parallel_batch(&training_points, options.workers as i32, |tuning| {
        let selection = evaluator.evaluate_selection(
            tuning
                .config
                .as_ref()
                .expect("feasible evaluation has a configuration"),
            &options.selection_seeds,
        );
        ParetoPoint {
            tuning: tuning.clone(),
            selection,
            selected: false,
        }
    });
    let selection_model_fits = validated
        .iter()
        .map(|point| point.selection.model_fits)
        .sum();
    let selection_trees_fitted = validated
        .iter()
        .map(|point| point.selection.trees_fitted)
        .sum();
    let mut pareto: Vec<ParetoPoint> = validated
        .into_iter()
        .filter(|point| {
            point
                .selection
                .metrics
                .is_some_and(|metrics| metrics.recall >= evaluator.min_recall)
        })
        .collect();
    let selection_rows: Vec<Vec<f64>> = pareto
        .iter()
        .map(|point| {
            let metrics = point
                .selection
                .metrics
                .expect("selection-feasible Pareto point has metrics");
            vec![
                -metrics.pr_auc,
                metrics.brier,
                point.selection.mean_model_bytes,
                point.selection.mean_structural_cost,
            ]
        })
        .collect();
    let selection_pareto_indices = pareto_indices(&selection_rows, OBJECTIVES)?;
    pareto = selection_pareto_indices
        .into_iter()
        .map(|index| pareto[index].clone())
        .collect();
    pareto.sort_by(|left, right| {
        left.selection
            .score()
            .total_cmp(&right.selection.score())
            .then_with(|| {
                left.selection
                    .mean_structural_cost
                    .total_cmp(&right.selection.mean_structural_cost)
            })
    });
    if pareto.is_empty() {
        return Err("MODE Pareto configurations all failed selection evaluation".into());
    }
    pareto[0].selected = true;
    let representative = pareto[0].clone();
    Ok(MultiOutcome {
        pareto,
        representative,
        evaluations: actual_evaluations,
        generations,
        model_fits: model_fits.load(Ordering::Relaxed),
        trees_fitted: trees_fitted.load(Ordering::Relaxed),
        selection_model_fits,
        selection_trees_fitted,
        tuning_model_seed: evaluator.tuning_model_seed,
        min_recall: evaluator.min_recall,
        structural_cost_limit: evaluator.forest.structural_cost_limit,
        elapsed: started.elapsed(),
        convergence,
    })
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: usize,
    pub seed: u64,
    pub selection_seeds: Vec<u64>,
    /// Apply the frozen publication acceptance criteria. Callers must set this
    /// only for the exact pre-registered publication protocol; archive capacity
    /// alone is not evidence level.
    pub apply_publication_criteria: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QdPoint {
    pub niche_id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub tuning: CandidateEvaluation,
    pub selection: ValidationEvaluation,
    pub descriptors_training: [f64; 2],
    pub descriptors_selection: [f64; 2],
    pub retained_niche: bool,
    pub visit_count: u64,
}

#[derive(Clone, Debug)]
pub struct QdOutcome {
    pub elites: Vec<QdPoint>,
    pub representative: QdPoint,
    pub evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub distinct_configurations: usize,
    pub retained_niches: usize,
    pub decision: String,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub selection_model_fits: usize,
    pub selection_trees_fitted: usize,
    pub tuning_model_seed: u64,
    pub min_recall: f64,
    pub structural_cost_limit: u64,
    pub elapsed: Duration,
    pub validation_elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

struct HpoQdBatch {
    evaluator: Arc<Evaluator>,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
    model_fits: Arc<AtomicUsize>,
    trees_fitted: Arc<AtomicUsize>,
    trace: Arc<Mutex<HashMap<Vec<u64>, CandidateEvaluation>>>,
}

impl QdBatchFitness for HpoQdBatch {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers as i32, |values| {
            self.evaluator.evaluate(values)
        });
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        self.model_fits.fetch_add(
            evaluated
                .iter()
                .map(|evaluation| evaluation.model_fits)
                .sum(),
            Ordering::Relaxed,
        );
        self.trees_fitted.fetch_add(
            evaluated
                .iter()
                .map(|evaluation| evaluation.trees_fitted)
                .sum(),
            Ordering::Relaxed,
        );
        {
            let mut trace = self
                .trace
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (values, evaluation) in xs.iter().zip(&evaluated) {
                trace.insert(point_key(values), evaluation.clone());
            }
        }
        evaluated
            .into_iter()
            .map(|evaluation| {
                let (quality, mut descriptors) = evaluation.qd_values();
                if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                } else if descriptors
                    .iter()
                    .zip(QD_DESCRIPTOR_LOWER.iter().zip(QD_DESCRIPTOR_UPPER))
                    .any(|(&value, (&lower, upper))| value < lower || value > upper)
                {
                    self.clipped.fetch_add(1, Ordering::Relaxed);
                    for (value, (&lower, &upper)) in descriptors
                        .iter_mut()
                        .zip(QD_DESCRIPTOR_LOWER.iter().zip(&QD_DESCRIPTOR_UPPER))
                    {
                        *value = value.clamp(lower, upper);
                    }
                }
                (quality, descriptors.to_vec())
            })
            .collect()
    }
}

pub fn optimize_qd(
    evaluator: Arc<Evaluator>,
    options: &QdOptions,
) -> Result<QdOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("QD evaluations must be positive".into());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be even and at least two".into());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let generations = options.evaluations.div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size;
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER_BOUNDS, &UPPER_BOUNDS, &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let model_fits = Arc::new(AtomicUsize::new(0));
    let trees_fitted = Arc::new(AtomicUsize::new(0));
    let trace = Arc::new(Mutex::new(HashMap::new()));
    let mut batch = HpoQdBatch {
        evaluator: Arc::clone(&evaluator),
        workers: options.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        clipped: Arc::clone(&clipped),
        model_fits: Arc::clone(&model_fits),
        trees_fitted: Arc::clone(&trees_fitted),
        trace: Arc::clone(&trace),
    };
    let parameters = MapElitesParams {
        generations,
        chunk_size: options.chunk_size,
        use_sbx: false,
        ..Default::default()
    };
    let started = Instant::now();
    let mut convergence = Vec::with_capacity(generations);
    map_elites_batch_with_progress(
        &mut archive,
        &mut batch,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
        &parameters,
        &mut rng,
        &mut |_, archive| {
            let count = evaluations.load(Ordering::Relaxed);
            convergence.push(QdProgress {
                evaluations: count,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / count.max(1) as f64,
            });
        },
    )?;
    let elapsed = started.elapsed();
    let occupied_indices: Vec<usize> = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .collect();
    let elite_evaluations: Vec<(usize, CandidateEvaluation)> = {
        let trace = trace
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        occupied_indices
            .iter()
            .map(|&index| {
                let x = &archive.xs()[index];
                trace
                    .get(&point_key(x))
                    .cloned()
                    .map(|evaluation| (index, evaluation))
                    .ok_or("MAP-Elites archive contains an unevaluated point")
            })
            .collect::<Result<_, _>>()?
    };
    let validation_started = Instant::now();
    let validated = parallel_batch(
        &elite_evaluations,
        options.workers as i32,
        |(index, tuning)| {
            let config = tuning
                .config
                .as_ref()
                .expect("archive elite decodes to a configuration");
            let selection = evaluator.evaluate_selection(config, &options.selection_seeds);
            (*index, tuning.clone(), selection)
        },
    );
    let validation_elapsed = validation_started.elapsed();
    let selection_model_fits = validated
        .iter()
        .map(|(_, _, selection)| selection.model_fits)
        .sum();
    let selection_trees_fitted = validated
        .iter()
        .map(|(_, _, selection)| selection.trees_fitted)
        .sum();
    let mut elites = Vec::with_capacity(validated.len());
    for (niche_id, tuning, selection) in validated {
        let descriptors_training = [
            archive.descriptors()[niche_id][0],
            archive.descriptors()[niche_id][1],
        ];
        let descriptors_selection = selection
            .mean_single_forest_qd_descriptors
            .unwrap_or([f64::NAN; 2]);
        let retained_niche = selection
            .metrics
            .is_some_and(|metrics| metrics.recall >= evaluator.min_recall)
            && qd_niche_index(descriptors_selection, side) == Some(niche_id);
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            tuning,
            selection,
            descriptors_training,
            descriptors_selection,
            retained_niche,
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| {
        left.tuning
            .scalar_fitness
            .total_cmp(&right.tuning.scalar_fitness)
    });
    let representative = elites
        .iter()
        .filter(|point| {
            point
                .selection
                .metrics
                .is_some_and(|metrics| metrics.recall >= evaluator.min_recall)
        })
        .min_by(|left, right| {
            left.selection
                .score()
                .total_cmp(&right.selection.score())
                .then_with(|| {
                    left.selection
                        .mean_structural_cost
                        .total_cmp(&right.selection.mean_structural_cost)
                })
        })
        .cloned()
        .ok_or("MAP-Elites returned no selection-feasible hyperparameter configuration")?;
    let distinct_configurations = elites
        .iter()
        .filter_map(|point| point.tuning.config.as_ref())
        .map(ForestConfig::canonical_key)
        .collect::<HashSet<_>>()
        .len();
    let retained_niches = elites.iter().filter(|point| point.retained_niche).count();
    let coverage = archive.occupied() as f64 / archive.capacity() as f64;
    let retention = retained_niches as f64 / archive.occupied().max(1) as f64;
    let decision = qd_decision(
        options.apply_publication_criteria,
        coverage,
        distinct_configurations,
        retention,
    );
    Ok(QdOutcome {
        representative,
        evaluations: actual_evaluations,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        distinct_configurations,
        retained_niches,
        decision,
        model_fits: model_fits.load(Ordering::Relaxed),
        trees_fitted: trees_fitted.load(Ordering::Relaxed),
        selection_model_fits,
        selection_trees_fitted,
        tuning_model_seed: evaluator.tuning_model_seed,
        min_recall: evaluator.min_recall,
        structural_cost_limit: evaluator.forest.structural_cost_limit,
        elapsed,
        validation_elapsed,
        convergence,
        elites,
    })
}

pub fn qd_decision(
    apply_publication_criteria: bool,
    coverage: f64,
    distinct_configurations: usize,
    retention: f64,
) -> String {
    if !apply_publication_criteria {
        "exploratory: publication acceptance criteria not applied".to_string()
    } else if coverage >= 0.4 && distinct_configurations >= 50 && retention >= 0.5 {
        "accepted: coverage, configuration diversity, and niche retention passed".to_string()
    } else {
        "rejected: at least one pre-registered QD criterion failed".to_string()
    }
}

fn point_key(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub fn qd_niche_index(descriptors: [f64; 2], side: usize) -> Option<usize> {
    if descriptors.iter().any(|value| !value.is_finite())
        || descriptors[0] < QD_DESCRIPTOR_LOWER[0]
        || descriptors[0] > QD_DESCRIPTOR_UPPER[0]
        || descriptors[1] < QD_DESCRIPTOR_LOWER[1]
        || descriptors[1] > QD_DESCRIPTOR_UPPER[1]
    {
        return None;
    }
    let x = ((descriptors[0] - QD_DESCRIPTOR_LOWER[0])
        / (QD_DESCRIPTOR_UPPER[0] - QD_DESCRIPTOR_LOWER[0]))
        .clamp(0.0, 1.0 - f64::EPSILON);
    let y = ((descriptors[1] - QD_DESCRIPTOR_LOWER[1])
        / (QD_DESCRIPTOR_UPPER[1] - QD_DESCRIPTOR_LOWER[1]))
        .clamp(0.0, 1.0 - f64::EPSILON);
    Some((y * side as f64) as usize * side + (x * side as f64) as usize)
}

pub fn configurations_by_key(evaluations: &[CandidateEvaluation]) -> HashMap<String, ForestConfig> {
    evaluations
        .iter()
        .filter_map(|evaluation| evaluation.config.clone())
        .map(|config| (config.canonical_key(), config))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Dataset, Preset};
    use crate::metrics::Metrics;

    fn evaluator() -> Arc<Evaluator> {
        Arc::new(Evaluator::new(
            Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap()),
            0.1,
            42,
        ))
    }

    #[test]
    fn latin_hypercube_uses_every_stratum() {
        let sample = latin_hypercube(16, 42);
        for dimension in 0..DIMENSION {
            let mut strata: Vec<usize> = sample
                .iter()
                .map(|row| (row[dimension] * 16.0) as usize)
                .collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..16).collect::<Vec<_>>());
        }
    }

    #[test]
    fn tiny_baseline_selects_a_configuration() {
        let result = optimize_baseline(
            evaluator(),
            &BaselineOptions {
                method: BaselineMethod::Random,
                evaluations: 4,
                workers: 2,
                seed: 42,
                shortlist: 2,
                selection_seeds: vec![101],
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 4);
        assert!(result.selected.selection.metrics.is_some());
    }

    #[test]
    fn tiny_mode_counts_only_requested_tuning_evaluations() {
        let evaluator = Arc::new(Evaluator::new(
            Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap()),
            0.0,
            42,
        ));
        let result = optimize_multi(
            evaluator,
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 2,
                seed: 42,
                selection_seeds: vec![101],
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 8);
        assert_eq!(result.model_fits, 8 * 5);
        assert!(result.selection_model_fits > 0);
    }

    #[test]
    fn tiny_qd_reuses_recorded_elite_evaluations() {
        let evaluator = Arc::new(Evaluator::new(
            Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap()),
            0.0,
            42,
        ));
        let result = optimize_qd(
            evaluator,
            &QdOptions {
                evaluations: 8,
                capacity: 4,
                chunk_size: 4,
                workers: 2,
                seed: 42,
                selection_seeds: vec![101],
                apply_publication_criteria: false,
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 8);
        assert_eq!(result.model_fits, 8 * 5);
        assert_eq!(result.selection_model_fits, result.occupied);
        assert!(result.decision.starts_with("exploratory:"));
    }

    #[test]
    fn qd_niche_mapping_checks_bounds() {
        assert_eq!(qd_niche_index(QD_DESCRIPTOR_LOWER, 20), Some(0));
        assert_eq!(qd_niche_index(QD_DESCRIPTOR_UPPER, 20), Some(399));
        assert_eq!(
            qd_niche_index([QD_DESCRIPTOR_UPPER[0] + 0.01, QD_DESCRIPTOR_LOWER[1]], 20),
            None
        );
        assert_eq!(
            qd_niche_index([QD_DESCRIPTOR_LOWER[0] - 0.01, QD_DESCRIPTOR_LOWER[1]], 20),
            None
        );
    }

    #[test]
    fn qd_publication_decision_requires_an_explicit_protocol_flag() {
        assert!(qd_decision(false, 0.8, 200, 0.8).starts_with("exploratory:"));
        assert!(qd_decision(true, 0.8, 200, 0.8).starts_with("accepted:"));
        assert!(qd_decision(true, 0.8, 200, 0.1).starts_with("rejected:"));
    }

    /// The rejected descriptor pair collapsed onto a ribbon because both axes
    /// moved together. Guard the replacement against silently regressing to a
    /// confusion-count axis.
    #[test]
    fn qd_descriptors_are_precision_and_sharpness() {
        let metrics = Metrics::calculate(&[0, 0, 1, 1], &[0.2, 0.4, 0.6, 0.9]).unwrap();
        assert_eq!(
            metrics.qd_descriptors(),
            [metrics.precision, metrics.sharpness]
        );
        // A hedging forest and a decisive one can share an operating point yet
        // must land in different sharpness niches.
        let hedging = Metrics::calculate(&[0, 0, 1, 1], &[0.45, 0.48, 0.52, 0.55]).unwrap();
        let decisive = Metrics::calculate(&[0, 0, 1, 1], &[0.02, 0.10, 0.90, 0.98]).unwrap();
        assert_eq!(hedging.precision, decisive.precision);
        assert!(decisive.sharpness > hedging.sharpness * 5.0);
    }

    #[test]
    fn independent_seed_derivation_changes_streams() {
        assert_ne!(
            crate::data::stream_seed(42, 0),
            crate::data::stream_seed(42, 1)
        );
    }
}
