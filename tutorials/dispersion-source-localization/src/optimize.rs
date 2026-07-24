use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    AdvancedRetryConfig, Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams,
    QdBatchFitness, RetryBounds, RetryConfig, RetryImprovement, RetryRunResult, Rng,
    advanced_retry, map_elites_batch_with_progress, optimize_bite, parallel_batch, pareto_indices,
};

use crate::model::{
    DIMENSION, Dataset, Design, Metrics, OBJECTIVES, QD_DESCRIPTOR_LOWER, QD_DESCRIPTOR_UPPER,
    evaluate_training, evaluate_validation, lower_bounds, multi_objective, qd_objective,
    scalar_objective, upper_bounds,
};

#[derive(Clone, Debug)]
pub struct ScalarOptions {
    pub evaluations_per_retry: u64,
    pub retries: usize,
    pub workers: usize,
    pub depth: i32,
    pub max_eval_fac: f64,
    pub seed: u64,
}

impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            evaluations_per_retry: 750,
            retries: 16,
            workers: 0,
            depth: 6,
            max_eval_fac: 4.0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarOutcome {
    pub design: Design,
    pub training: Metrics,
    pub validation: Metrics,
    pub evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    dataset: &Dataset,
    options: &ScalarOptions,
) -> Result<ScalarOutcome, Box<dyn Error>> {
    if options.evaluations_per_retry == 0 || options.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    if !(1..=36).contains(&options.depth) {
        return Err("BiteOpt depth must lie in 1..=36".into());
    }
    if !options.max_eval_fac.is_finite() || options.max_eval_fac < 1.0 {
        return Err("advanced-retry maximum evaluation factor must be at least one".into());
    }
    let bounds = RetryBounds::new(lower_bounds().to_vec(), upper_bounds().to_vec())?;
    let objective = |values: &[f64]| scalar_objective(values, dataset);
    let retry = RetryConfig {
        num_retries: options.retries,
        workers: options.workers,
        capacity: options.retries.min(500),
        max_evaluations: options.evaluations_per_retry,
        seed: options.seed,
        statistic_num: 1_000,
        ..Default::default()
    };
    let config = AdvancedRetryConfig {
        retry,
        check_interval: (options.retries / 4).max(1),
        max_eval_fac: options.max_eval_fac,
        crossover_probability: 0.55,
        diversity_threshold: 0.12,
    };
    let started = Instant::now();
    let result = advanced_retry(&objective, &bounds, &config, |objective, context| {
        let mut rng = Rng::new(context.seed);
        let random_guess: Vec<f64> = context
            .bounds
            .lower()
            .iter()
            .zip(context.bounds.upper())
            .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
            .collect();
        let guess = context.guess.as_deref().unwrap_or(&random_guess);
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(guess),
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
        return Err("coordinated BiteOpt retry returned no finite source hypothesis".into());
    }
    let design = Design::from_slice(&result.x)?;
    let training = evaluate_training(design.values(), dataset)?;
    let validation = evaluate_validation(design.values(), dataset)?;
    Ok(ScalarOutcome {
        design,
        training,
        validation,
        evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        improvements: result.improvements,
    })
}

#[derive(Clone, Debug)]
pub struct MultiOptions {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for MultiOptions {
    fn default() -> Self {
        Self {
            evaluations: 20_000,
            popsize: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub design: Design,
    pub objectives: [f64; OBJECTIVES],
}

#[derive(Clone, Copy, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
pub struct MultiOutcome {
    pub pareto: Vec<ParetoPoint>,
    pub representative: ParetoPoint,
    pub training: Metrics,
    pub validation: Metrics,
    pub evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub convergence: Vec<MoProgress>,
    pub quality: f64,
}

fn balanced_objectives(values: &[f64; OBJECTIVES]) -> f64 {
    values[0] + 0.35 * values[1] + 0.01 * values[2]
}

pub fn optimize_multi(
    dataset: &Dataset,
    options: &MultiOptions,
) -> Result<MultiOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("MODE evaluations must be positive".into());
    }
    if options.popsize < 4 {
        return Err("MODE population size must be at least four".into());
    }
    if options.popsize > i32::MAX as usize {
        return Err("MODE population size is too large".into());
    }
    let generations = options.evaluations.div_ceil(options.popsize);
    let evaluations = generations * options.popsize;
    let lower = lower_bounds();
    let upper = upper_bounds();
    let fitness = Fitness::bounded(DIMENSION, OBJECTIVES, &lower, &upper);
    let parameters = ModeParams {
        popsize: options.popsize as i32,
        nsga_update: true,
        seed: options.seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, OBJECTIVES, 0, None, &parameters)?;
    let mut convergence = Vec::with_capacity(generations);
    let mut best_balanced = f64::INFINITY;
    let started = Instant::now();
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, options.workers as i32, |values| {
            multi_objective(values, dataset)
        });
        for values in &ys {
            best_balanced =
                best_balanced.min(balanced_objectives(&[values[0], values[1], values[2]]));
        }
        mode.tell(&ys);
        convergence.push(MoProgress {
            evaluations: (generation + 1) * options.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality: -best_balanced,
        });
    }
    let population = mode.population();
    let values = parallel_batch(&population, options.workers as i32, |candidate| {
        multi_objective(candidate, dataset)
    });
    let indices = pareto_indices(&values, OBJECTIVES)?;
    let mut pareto = Vec::with_capacity(indices.len());
    for index in indices {
        pareto.push(ParetoPoint {
            design: Design::from_slice(&population[index])?,
            objectives: [values[index][0], values[index][1], values[index][2]],
        });
    }
    if pareto.is_empty() {
        return Err("MODE returned an empty source-localization Pareto front".into());
    }
    pareto.sort_by(|left, right| {
        balanced_objectives(&left.objectives).total_cmp(&balanced_objectives(&right.objectives))
    });
    let representative = pareto[0].clone();
    let quality = -balanced_objectives(&representative.objectives);
    let training = evaluate_training(representative.design.values(), dataset)?;
    let validation = evaluate_validation(representative.design.values(), dataset)?;
    Ok(MultiOutcome {
        pareto,
        representative,
        training,
        validation,
        evaluations,
        generations,
        elapsed: started.elapsed(),
        convergence,
        quality,
    })
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for QdOptions {
    fn default() -> Self {
        Self {
            evaluations: 20_000,
            capacity: 400,
            chunk_size: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdPoint {
    pub niche_id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub design: Design,
    pub quality_train: f64,
    pub quality_validation: f64,
    pub descriptors: [f64; 2],
    pub visit_count: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct QdOutcome {
    pub elites: Vec<QdPoint>,
    pub representative: QdPoint,
    pub training: Metrics,
    pub validation: Metrics,
    pub evaluations: usize,
    pub validation_evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub elapsed: Duration,
    pub validation_elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

struct DispersionQdBatch<'a> {
    dataset: &'a Dataset,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for DispersionQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers as i32, |values| {
            qd_objective(values, self.dataset)
        });
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        let mut output = Vec::with_capacity(evaluated.len());
        for (quality, descriptors) in evaluated {
            if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
                self.invalid.fetch_add(1, Ordering::Relaxed);
            } else if descriptors
                .iter()
                .zip(QD_DESCRIPTOR_LOWER.iter().zip(QD_DESCRIPTOR_UPPER))
                .any(|(&value, (&lower, upper))| value < lower || value > upper)
            {
                self.clipped.fetch_add(1, Ordering::Relaxed);
            }
            output.push((quality, descriptors.to_vec()));
        }
        output
    }
}

pub fn optimize_qd(dataset: &Dataset, options: &QdOptions) -> Result<QdOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("QD evaluations must be positive".into());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".into());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let generations = options.evaluations.div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size;
    let lower = lower_bounds();
    let upper = upper_bounds();
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&lower, &upper, &mut rng);

    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = DispersionQdBatch {
        dataset,
        workers: options.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        clipped: Arc::clone(&clipped),
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
        &lower,
        &upper,
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
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);

    let occupied_indices: Vec<usize> = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .collect();
    let validation_started = Instant::now();
    let validation_metrics = parallel_batch(
        &occupied_indices
            .iter()
            .map(|&index| archive.xs()[index].clone())
            .collect::<Vec<_>>(),
        options.workers as i32,
        |values| evaluate_validation(values, dataset),
    );
    let validation_elapsed = validation_started.elapsed();
    let mut elites = Vec::with_capacity(occupied_indices.len());
    for (&niche_id, validation) in occupied_indices.iter().zip(validation_metrics) {
        let quality_validation = validation.map_or(f64::INFINITY, |metrics| metrics.scalar_score);
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            design: Design::from_slice(&archive.xs()[niche_id])?,
            quality_train: archive.ys()[niche_id],
            quality_validation,
            descriptors: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| left.quality_train.total_cmp(&right.quality_train));
    let representative = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites did not find a valid source hypothesis")?;
    let training = evaluate_training(representative.design.values(), dataset)?;
    let validation = evaluate_validation(representative.design.values(), dataset)?;
    Ok(QdOutcome {
        representative,
        training,
        validation,
        evaluations: actual_evaluations,
        validation_evaluations: occupied_indices.len(),
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        elapsed,
        validation_elapsed,
        convergence,
        elites,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_optimizers_return_finite_results() {
        let dataset = Dataset::synthetic();
        let scalar = optimize_scalar(
            &dataset,
            &ScalarOptions {
                evaluations_per_retry: 12,
                retries: 2,
                workers: 2,
                depth: 1,
                max_eval_fac: 1.0,
                seed: 7,
            },
        )
        .unwrap();
        assert!(scalar.training.scalar_score.is_finite());

        let multi = optimize_multi(
            &dataset,
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 2,
                seed: 8,
            },
        )
        .unwrap();
        assert_eq!(multi.evaluations, 8);
        assert!(!multi.pareto.is_empty());

        let qd = optimize_qd(
            &dataset,
            &QdOptions {
                evaluations: 16,
                capacity: 16,
                chunk_size: 4,
                workers: 2,
                seed: 9,
            },
        )
        .unwrap();
        assert_eq!(qd.evaluations, 16);
        assert!(qd.occupied > 0);
    }

    #[test]
    fn optimizer_options_are_validated() {
        let dataset = Dataset::synthetic();
        assert!(
            optimize_scalar(
                &dataset,
                &ScalarOptions {
                    evaluations_per_retry: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_multi(
                &dataset,
                &MultiOptions {
                    popsize: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_qd(
                &dataset,
                &QdOptions {
                    capacity: 15,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
