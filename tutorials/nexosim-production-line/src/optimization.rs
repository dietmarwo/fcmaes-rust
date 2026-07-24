//! MODE, MAP-Elites, and the explicit outer-versus-inner comparison.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::time::Instant;

use fcmaes_core::{
    Archive, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, Rng,
    map_elites_batch_with_progress, parallel_batch, pareto_indices,
};

use crate::model::{DIM, Design, INTEGERS, LOWER, Metrics, OBJECTIVES, UPPER, simulate};

pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [0.0, 0.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [100.0, 200.0];
pub const VALIDATION_SEED_XOR: u64 = 0x6A09_E667_F3BC_C909;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelStrategy {
    /// Parallel MODE candidate/replication evaluations; single-thread NeXosim.
    Outer,
    /// Serial MODE candidate evaluations; multithreaded NeXosim.
    Inner,
}

impl ParallelStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Outer => "outer-fcmaes",
            Self::Inner => "inner-nexosim",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "outer" | "outer-fcmaes" => Ok(Self::Outer),
            "inner" | "inner-nexosim" => Ok(Self::Inner),
            _ => Err("strategy must be outer, inner, or both".to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    pub evaluations: usize,
    pub popsize: usize,
    pub replications: usize,
    pub workers: usize,
    pub horizon_minutes: f64,
    pub seed: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            evaluations: 512,
            popsize: 32,
            replications: 4,
            workers: 0,
            horizon_minutes: 240.0,
            seed: 42,
        }
    }
}

impl OptimizationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.evaluations == 0 || self.replications == 0 {
            return Err("evaluations and replications must be positive".to_string());
        }
        if self.popsize < 4 {
            return Err("MODE population size must be at least four".to_string());
        }
        if self.popsize > i32::MAX as usize {
            return Err("MODE population size is too large".to_string());
        }
        if self.workers > usize::BITS as usize {
            return Err(format!(
                "worker count must not exceed NeXosim's {}-thread limit",
                usize::BITS
            ));
        }
        if !self.horizon_minutes.is_finite() || self.horizon_minutes <= 0.0 {
            return Err("horizon must be finite and positive".to_string());
        }
        Ok(())
    }

    pub fn resolved_workers(&self) -> usize {
        if self.workers == 0 {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
                .min(usize::BITS as usize)
        } else {
            self.workers
        }
    }
}

#[derive(Clone, Debug)]
pub struct FrontMember {
    pub design: Design,
    pub objectives: [f64; OBJECTIVES],
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub strategy: ParallelStrategy,
    pub evaluations: usize,
    pub simulation_replications: usize,
    pub wall_seconds: f64,
    pub front: Vec<FrontMember>,
    pub balanced_score: f64,
    pub convergence: Vec<MoProgress>,
}

#[derive(Clone, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub best_quality: f64,
}

/// Average stochastic replications with common random numbers.
pub fn evaluate_design(
    x: &[f64],
    config: &OptimizationConfig,
    nexosim_threads: usize,
) -> Result<[f64; OBJECTIVES], String> {
    Ok(
        evaluate_metrics(x, config, nexosim_threads, config.seed, config.replications)?
            .objectives(),
    )
}

/// Evaluate and average a named replication set.
pub fn evaluate_metrics(
    x: &[f64],
    config: &OptimizationConfig,
    nexosim_threads: usize,
    seed_root: u64,
    replications: usize,
) -> Result<Metrics, String> {
    if replications == 0 {
        return Err("replications must be positive".to_string());
    }
    let design = Design::decode(x)?;
    let mut values = Vec::with_capacity(replications);
    for replication in 0..replications {
        let seed = replication_seed(seed_root, replication);
        values.push(simulate(
            design,
            seed,
            config.horizon_minutes,
            nexosim_threads,
        )?);
    }
    Ok(mean_metrics(&values))
}

fn replication_seed(root: u64, replication: usize) -> u64 {
    root.wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ 0xD1B5_4A32_D192_ED03
}

pub fn optimize(
    strategy: ParallelStrategy,
    config: &OptimizationConfig,
    progress: bool,
) -> Result<OptimizationResult, String> {
    config.validate()?;
    let workers = config.resolved_workers();
    let (evaluation_workers, nexosim_threads) = match strategy {
        ParallelStrategy::Outer => (workers, 1),
        ParallelStrategy::Inner => (1, workers),
    };
    let fitness = Fitness::bounded(DIM, OBJECTIVES, &LOWER, &UPPER);
    let params = ModeParams {
        popsize: config.popsize as i32,
        nsga_update: true,
        seed: config.seed ^ 0xA076_1D64_78BD_642F,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, OBJECTIVES, 0, Some(INTEGERS.to_vec()), &params)
        .map_err(str::to_string)?;
    let generations = config.evaluations.div_ceil(config.popsize);
    let started = Instant::now();
    let mut evaluations = 0;
    let mut best_balanced_score = f64::NEG_INFINITY;
    let mut convergence = Vec::with_capacity(generations);
    for generation in 0..generations {
        let xs = mode.ask();
        let evaluated = parallel_batch(&xs, evaluation_workers as i32, |x| {
            evaluate_design(x, config, nexosim_threads)
        });
        let ys = evaluated.into_iter().collect::<Result<Vec<_>, _>>()?;
        for values in &ys {
            best_balanced_score = best_balanced_score.max(balanced_score(*values));
        }
        let ys = ys.into_iter().map(Vec::from).collect::<Vec<_>>();
        evaluations += ys.len();
        mode.tell(&ys);
        convergence.push(MoProgress {
            evaluations,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality: best_balanced_score,
        });
        if progress {
            eprintln!(
                "strategy={} generation={}/{} evaluations={} replications={} elapsed={:.3}s",
                strategy.name(),
                generation + 1,
                generations,
                evaluations,
                evaluations * config.replications,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let result = mode.result();
    let indices = pareto_indices(&result.y, OBJECTIVES).map_err(str::to_string)?;
    let mut front = indices
        .into_iter()
        .map(|index| {
            let values = &result.y[index];
            FrontMember {
                design: Design::decode(&result.x[index]).expect("MODE returned valid dimension"),
                objectives: [values[0], values[1], values[2], values[3]],
            }
        })
        .collect::<Vec<_>>();
    front.sort_by(|left, right| left.objectives[0].total_cmp(&right.objectives[0]));
    let balanced_score = front
        .iter()
        .map(|member| balanced_score(member.objectives))
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(OptimizationResult {
        strategy,
        evaluations,
        simulation_replications: evaluations * config.replications,
        wall_seconds: started.elapsed().as_secs_f64(),
        front,
        balanced_score,
        convergence,
    })
}

fn balanced_score(objectives: [f64; OBJECTIVES]) -> f64 {
    let throughput = -objectives[0];
    throughput
        / ((1.0 + objectives[1]) * (1.0 + 0.10 * objectives[2]) * (1.0 + 0.05 * objectives[3]))
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub validation_replications: usize,
    pub seed: u64,
}

impl Default for QdOptions {
    fn default() -> Self {
        Self {
            evaluations: 4_096,
            capacity: 400,
            chunk_size: 128,
            validation_replications: 8,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdPoint {
    pub niche_id: usize,
    pub validation_niche_id: Option<usize>,
    pub grid_x: usize,
    pub grid_y: usize,
    pub design: Design,
    pub quality_train: f64,
    pub quality_validation: f64,
    pub training: Metrics,
    pub validation: Metrics,
    pub visit_count: u64,
}

#[derive(Clone, Debug)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct QdResult {
    pub elites: Vec<QdPoint>,
    pub representative: QdPoint,
    pub evaluations: usize,
    pub validation_evaluations: usize,
    pub simulation_replications: usize,
    pub validation_replications: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub validation_same_niche_fraction: f64,
    pub elapsed: Duration,
    pub validation_elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

fn qd_quality(metrics: Metrics) -> f64 {
    let departures = (metrics.shipped + metrics.scrapped + metrics.overflowed).max(1) as f64;
    let loss_fraction = (metrics.scrapped + metrics.overflowed) as f64 / departures;
    metrics.mean_lead_time + 2.0 * metrics.cost_rate + 5.0 * loss_fraction
}

pub fn qd_objective(x: &[f64], config: &OptimizationConfig) -> Result<(f64, [f64; 2]), String> {
    let metrics = evaluate_metrics(x, config, 1, config.seed, config.replications)?;
    if metrics.shipped == 0
        || !metrics.throughput_per_hour.is_finite()
        || !metrics.mean_wip.is_finite()
    {
        return Ok((f64::INFINITY, [f64::INFINITY; 2]));
    }
    Ok((
        qd_quality(metrics),
        [metrics.throughput_per_hour, metrics.mean_wip],
    ))
}

struct ProductionQdBatch<'a> {
    config: &'a OptimizationConfig,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for ProductionQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers as i32, |x| qd_objective(x, self.config));
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        let mut output = Vec::with_capacity(evaluated.len());
        for evaluation in evaluated {
            let (quality, descriptors) = evaluation.unwrap_or((f64::INFINITY, [f64::INFINITY; 2]));
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

pub fn optimize_qd(config: &OptimizationConfig, options: &QdOptions) -> Result<QdResult, String> {
    config.validate()?;
    if options.evaluations == 0 || options.validation_replications == 0 {
        return Err("QD and validation evaluations must be positive".to_string());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".to_string());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".to_string());
    }
    let generations = options.evaluations.div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size;
    let workers = config.resolved_workers();
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIM,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )
    .map_err(str::to_string)?;
    archive.seed_uniform(&LOWER, &UPPER, &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = ProductionQdBatch {
        config,
        workers,
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
        &LOWER,
        &UPPER,
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
    )
    .map_err(str::to_string)?;
    let elapsed = started.elapsed();
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);

    let occupied_indices = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .collect::<Vec<_>>();
    let candidates = occupied_indices
        .iter()
        .map(|&index| archive.xs()[index].clone())
        .collect::<Vec<_>>();
    let training_metrics = parallel_batch(&candidates, workers as i32, |x| {
        evaluate_metrics(x, config, 1, config.seed, config.replications)
    });
    let validation_started = Instant::now();
    let validation_metrics = parallel_batch(&candidates, workers as i32, |x| {
        evaluate_metrics(
            x,
            config,
            1,
            config.seed ^ VALIDATION_SEED_XOR,
            options.validation_replications,
        )
    });
    let validation_elapsed = validation_started.elapsed();

    let mut same_niche = 0usize;
    let mut finite_validation = 0usize;
    let mut elites = Vec::with_capacity(occupied_indices.len());
    for (((&niche_id, x), training), validation) in occupied_indices
        .iter()
        .zip(&candidates)
        .zip(training_metrics)
        .zip(validation_metrics)
    {
        let training = training?;
        let (validation, quality_validation, validation_niche_id) = match validation {
            Ok(metrics)
                if metrics.shipped > 0
                    && metrics.throughput_per_hour.is_finite()
                    && metrics.mean_wip.is_finite() =>
            {
                let descriptors = [metrics.throughput_per_hour, metrics.mean_wip];
                let validation_niche = archive.index_of_niche(&descriptors);
                finite_validation += 1;
                if validation_niche == niche_id {
                    same_niche += 1;
                }
                (metrics, qd_quality(metrics), Some(validation_niche))
            }
            _ => (Metrics::default(), f64::INFINITY, None),
        };
        elites.push(QdPoint {
            niche_id,
            validation_niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            design: Design::decode(x)?,
            quality_train: archive.ys()[niche_id],
            quality_validation,
            training,
            validation,
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| left.quality_train.total_cmp(&right.quality_train));
    let representative = elites
        .first()
        .cloned()
        .ok_or_else(|| "MAP-Elites found no valid production policy".to_string())?;
    Ok(QdResult {
        representative,
        evaluations: actual_evaluations,
        validation_evaluations: occupied_indices.len(),
        simulation_replications: actual_evaluations * config.replications,
        validation_replications: occupied_indices.len() * options.validation_replications,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        validation_same_niche_fraction: same_niche as f64 / finite_validation.max(1) as f64,
        elapsed,
        validation_elapsed,
        convergence,
        elites,
    })
}

fn write_design_header(output: &mut String) {
    for name in [
        "buffer_capacity",
        "speed_a",
        "speed_b",
        "maintenance_threshold_a",
        "maintenance_threshold_b",
        "rework_probability",
        "dispatch_priority",
        "staff_a",
        "staff_b",
    ] {
        let _ = write!(output, ",decision_{name}");
    }
}

pub fn write_mode_artifacts(
    directory: &Path,
    result: &OptimizationResult,
    config: &OptimizationConfig,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_negative_throughput,objective_mean_lead_time,objective_mean_wip,objective_cost_rate",
    );
    write_design_header(&mut pareto);
    pareto.push('\n');
    for (index, member) in result.front.iter().enumerate() {
        let selected = balanced_score(member.objectives) == result.balanced_score;
        let _ = write!(
            pareto,
            "{index},1,{},{},{},{},{}",
            u8::from(selected),
            member.objectives[0],
            member.objectives[1],
            member.objectives[2],
            member.objectives[3],
        );
        for value in member.design.as_vector() {
            let _ = write!(pareto, ",{value}");
        }
        pareto.push('\n');
    }
    fs::write(directory.join("pareto.csv"), pareto)?;

    let mut convergence = String::from("evaluations,elapsed_seconds,best_quality\n");
    for sample in &result.convergence {
        let _ = writeln!(
            convergence,
            "{},{},{}",
            sample.evaluations, sample.elapsed_seconds, sample.best_quality
        );
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "nexosim-production-line",
        "formulation": "mo",
        "strategy": result.strategy.name(),
        "command": command,
        "seed": config.seed ^ 0xA076_1D64_78BD_642F,
        "simulation_seed_root": config.seed,
        "workers": config.resolved_workers(),
        "requested_evaluations": config.evaluations,
        "actual_evaluations": result.evaluations,
        "elapsed_seconds": result.wall_seconds,
        "simulation": {
            "replications": config.replications,
            "horizon_minutes": config.horizon_minutes
        },
        "objectives": [
            {
                "column": "objective_negative_throughput",
                "label": "Throughput",
                "unit": "orders/hour",
                "display_sign": -1
            },
            {
                "column": "objective_mean_lead_time",
                "label": "Mean lead time",
                "unit": "minutes"
            },
            {
                "column": "objective_mean_wip",
                "label": "Mean WIP",
                "unit": "orders"
            },
            {
                "column": "objective_cost_rate",
                "label": "Energy/staff cost rate"
            }
        ],
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": {
            "pareto": "pareto.csv",
            "convergence": "convergence.csv"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

pub fn write_qd_artifacts(
    directory: &Path,
    result: &QdResult,
    config: &OptimizationConfig,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,descriptor_throughput_per_hour_train,descriptor_mean_wip_train,descriptor_throughput_per_hour_validation,descriptor_mean_wip_validation,validation_niche_id,same_niche,visit_count",
    );
    write_design_header(&mut archive);
    archive.push('\n');
    for point in &result.elites {
        let _ = write!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality_train,
            point.quality_validation,
            point.training.throughput_per_hour,
            point.training.mean_wip,
            point.validation.throughput_per_hour,
            point.validation.mean_wip,
            point
                .validation_niche_id
                .map_or(-1_i64, |value| value as i64),
            u8::from(point.validation_niche_id == Some(point.niche_id)),
            point.visit_count,
        );
        for value in point.design.as_vector() {
            let _ = write!(archive, ",{value}");
        }
        archive.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive)?;

    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &result.convergence {
        let _ = writeln!(
            convergence,
            "{},{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction,
        );
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let side = (result.capacity as f64).sqrt() as usize;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "nexosim-production-line",
        "formulation": "qd",
        "strategy": "outer-fcmaes",
        "command": command,
        "root_seed": config.seed,
        "seed": options.seed,
        "simulation_seed_root": config.seed,
        "validation_seed_root": config.seed ^ VALIDATION_SEED_XOR,
        "workers": config.resolved_workers(),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": result.evaluations,
        "elapsed_seconds": result.elapsed.as_secs_f64(),
        "validation_elapsed_seconds": result.validation_elapsed.as_secs_f64(),
        "simulation": {
            "replications": config.replications,
            "validation_replications": options.validation_replications,
            "horizon_minutes": config.horizon_minutes
        },
        "descriptors": [
            {
                "column": "descriptor_throughput_per_hour",
                "label": "Achieved throughput",
                "unit": "orders/hour",
                "bounds": [QD_DESCRIPTOR_LOWER[0], QD_DESCRIPTOR_UPPER[0]]
            },
            {
                "column": "descriptor_mean_wip",
                "label": "Mean WIP",
                "unit": "orders",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            }
        ],
        "qd": {
            "capacity": result.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_validation_column": "quality_validation",
            "quality_label": "Lead-time/cost/loss quality (lower is better)",
            "occupied": result.occupied,
            "coverage": result.occupied as f64 / result.capacity as f64,
            "qd_score": result.qd_score,
            "best_quality": result.representative.quality_train,
            "invalid_evaluations": result.invalid_evaluations,
            "clipped_descriptors": result.clipped_descriptors,
            "validation_evaluations": result.validation_evaluations,
            "validation_same_niche_fraction": result.validation_same_niche_fraction
        },
        "convergence_metrics": [
            "coverage", "qd_score", "best_quality", "invalid_fraction"
        ],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

/// Aggregate raw metrics, used by tests and possible custom drivers.
pub fn mean_metrics(values: &[Metrics]) -> Metrics {
    if values.is_empty() {
        return Metrics::default();
    }
    let count = values.len() as f64;
    Metrics {
        arrivals: values.iter().map(|value| value.arrivals).sum::<usize>() / values.len(),
        shipped: values.iter().map(|value| value.shipped).sum::<usize>() / values.len(),
        scrapped: values.iter().map(|value| value.scrapped).sum::<usize>() / values.len(),
        overflowed: values.iter().map(|value| value.overflowed).sum::<usize>() / values.len(),
        throughput_per_hour: values
            .iter()
            .map(|value| value.throughput_per_hour)
            .sum::<f64>()
            / count,
        mean_lead_time: values.iter().map(|value| value.mean_lead_time).sum::<f64>() / count,
        mean_wip: values.iter().map(|value| value.mean_wip).sum::<f64>() / count,
        energy: values.iter().map(|value| value.energy).sum::<f64>() / count,
        cost_rate: values.iter().map(|value| value.cost_rate).sum::<f64>() / count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> OptimizationConfig {
        OptimizationConfig {
            evaluations: 8,
            popsize: 4,
            replications: 1,
            workers: 2,
            horizon_minutes: 15.0,
            seed: 3,
        }
    }

    #[test]
    fn objective_is_finite_and_repeatable() {
        let config = tiny_config();
        let x = Design::default().as_vector();
        let first = evaluate_design(&x, &config, 1).unwrap();
        let second = evaluate_design(&x, &config, 1).unwrap();
        assert_eq!(first, second);
        assert!(first.into_iter().all(f64::is_finite));
    }

    #[test]
    fn both_parallel_strategies_complete() {
        let config = tiny_config();
        let outer = optimize(ParallelStrategy::Outer, &config, false).unwrap();
        let inner = optimize(ParallelStrategy::Inner, &config, false).unwrap();
        assert_eq!(outer.evaluations, 8);
        assert_eq!(outer.simulation_replications, 8);
        assert!(!outer.front.is_empty());
        assert!(outer.balanced_score.is_finite());
        assert_eq!(outer.evaluations, inner.evaluations);
        assert_eq!(outer.simulation_replications, inner.simulation_replications);
        assert_eq!(outer.balanced_score, inner.balanced_score);
        assert_eq!(outer.front.len(), inner.front.len());
        for (left, right) in outer.front.iter().zip(&inner.front) {
            assert_eq!(left.design, right.design);
            assert_eq!(left.objectives, right.objectives);
        }
    }

    #[test]
    fn invalid_config_is_rejected() {
        let mut config = tiny_config();
        config.popsize = 3;
        assert!(config.validate().is_err());
        config.popsize = 4;
        config.replications = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn tiny_qd_run_uses_outer_parallelism_and_holdout_seeds() {
        let config = tiny_config();
        let result = optimize_qd(
            &config,
            &QdOptions {
                evaluations: 32,
                capacity: 16,
                chunk_size: 8,
                validation_replications: 2,
                seed: 19,
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 32);
        assert_eq!(result.simulation_replications, 32);
        assert_eq!(result.occupied, result.elites.len());
        assert_eq!(result.validation_evaluations, result.occupied);
        assert!(result.representative.quality_train.is_finite());
        assert!(result.validation_same_niche_fraction.is_finite());
    }
}
