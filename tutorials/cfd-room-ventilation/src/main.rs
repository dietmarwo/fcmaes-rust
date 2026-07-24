use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use cfd_room_ventilation::{
    DIMENSION, Design, Evaluation, LOWER_BOUNDS, RoomConfig, RoomProblem, UPPER_BOUNDS,
};
use fcmaes_core::{
    Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams, RetryBounds, RetryConfig,
    RetryRunResult, Rng, map_elites_batch_with_progress, optimize_bite, parallel_batch,
    pareto_indices, retry,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Evaluate,
    Single,
    Multi,
    Qd,
    Both,
    All,
}

impl RunMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "evaluate" | "baseline" => Ok(Self::Evaluate),
            "single" | "so" => Ok(Self::Single),
            "multi" | "mo" | "mode" => Ok(Self::Multi),
            "qd" | "map-elites" | "map_elites" => Ok(Self::Qd),
            "both" => Ok(Self::Both),
            "all" => Ok(Self::All),
            _ => Err("--mode must be evaluate, single, multi, qd, both, or all".to_owned()),
        }
    }

    fn includes_single(self) -> bool {
        matches!(self, Self::Single | Self::Both | Self::All)
    }

    fn includes_multi(self) -> bool {
        matches!(self, Self::Multi | Self::Both | Self::All)
    }

    fn includes_qd(self) -> bool {
        matches!(self, Self::Qd | Self::All)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Evaluate => "evaluate",
            Self::Single => "single",
            Self::Multi => "multi",
            Self::Qd => "qd",
            Self::Both => "both",
            Self::All => "all",
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    mode: RunMode,
    workers: usize,
    retries: usize,
    evaluations: u64,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    nx: usize,
    ny: usize,
    flow_steps: usize,
    scalar_steps: usize,
    seed: u64,
    csv: Option<PathBuf>,
    output: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Both,
            workers: 4,
            retries: 4,
            evaluations: 200,
            mo_evaluations: 512,
            popsize: 64,
            qd_evaluations: 512,
            qd_capacity: 100,
            qd_chunk_size: 64,
            nx: 40,
            ny: 24,
            flow_steps: 500,
            scalar_steps: 300,
            seed: 42,
            csv: None,
            output: None,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::from_args(env::args().skip(1))
    }

    fn from_args(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut parsed = Self::default();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--mode" => parsed.mode = RunMode::parse(&next_value(&mut arguments, "--mode")?)?,
                "--workers" => parsed.workers = parse_value(&mut arguments, "--workers")?,
                "--retries" => parsed.retries = parse_value(&mut arguments, "--retries")?,
                "--evaluations" => {
                    parsed.evaluations = parse_value(&mut arguments, "--evaluations")?
                }
                "--mo-evaluations" => {
                    parsed.mo_evaluations = parse_value(&mut arguments, "--mo-evaluations")?
                }
                "--popsize" => parsed.popsize = parse_value(&mut arguments, "--popsize")?,
                "--qd-evaluations" => {
                    parsed.qd_evaluations = parse_value(&mut arguments, "--qd-evaluations")?
                }
                "--qd-capacity" => {
                    parsed.qd_capacity = parse_value(&mut arguments, "--qd-capacity")?
                }
                "--qd-chunk-size" => {
                    parsed.qd_chunk_size = parse_value(&mut arguments, "--qd-chunk-size")?
                }
                "--nx" => parsed.nx = parse_value(&mut arguments, "--nx")?,
                "--ny" => parsed.ny = parse_value(&mut arguments, "--ny")?,
                "--flow-steps" => parsed.flow_steps = parse_value(&mut arguments, "--flow-steps")?,
                "--scalar-steps" => {
                    parsed.scalar_steps = parse_value(&mut arguments, "--scalar-steps")?
                }
                "--seed" => parsed.seed = parse_value(&mut arguments, "--seed")?,
                "--csv" => parsed.csv = Some(next_value(&mut arguments, "--csv")?.into()),
                "--output" => parsed.output = Some(next_value(&mut arguments, "--output")?.into()),
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.retries == 0
            || self.evaluations == 0
            || self.mo_evaluations == 0
            || self.qd_evaluations == 0
        {
            return Err("retry and evaluation counts must be positive".to_owned());
        }
        if self.popsize < 4 {
            return Err("--popsize must be at least four".to_owned());
        }
        if self.qd_chunk_size < 2 || !self.qd_chunk_size.is_multiple_of(2) {
            return Err("--qd-chunk-size must be an even number of at least two".to_owned());
        }
        let qd_side = (self.qd_capacity as f64).sqrt() as usize;
        if qd_side < 2 || qd_side * qd_side != self.qd_capacity {
            return Err("--qd-capacity must be a perfect square of at least four".to_owned());
        }
        RoomConfig {
            nx: self.nx,
            ny: self.ny,
            flow_steps: self.flow_steps,
            scalar_steps: self.scalar_steps,
            ..Default::default()
        }
        .validate()
        .map_err(str::to_owned)
    }
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value after {option}"))
}

fn parse_value<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    next_value(arguments, option)?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "Experimental room-ventilation CFD optimization\n\
         \nUsage: cargo run --release -- [OPTIONS]\n\
         \n  --mode NAME             evaluate, single, multi, qd, both, or all (both)\n\
         \n  --workers N             Retry/MODE workers; 0 uses available CPUs (4)\n\
         \n  --retries N             Independent BiteOpt retries (4)\n\
         \n  --evaluations N         Evaluations per BiteOpt retry (200)\n\
         \n  --mo-evaluations N      Requested MODE evaluations (512)\n\
         \n  --popsize N             MODE population size (64)\n\
         \n  --qd-evaluations N      Requested MAP-Elites evaluations (512)\n\
         \n  --qd-capacity N         Square 2-D archive capacity (100)\n\
         \n  --qd-chunk-size N       MAP-Elites evaluation batch (64)\n\
         \n  --nx N                  CFD cells in x (40)\n\
         \n  --ny N                  CFD cells in y (24)\n\
         \n  --flow-steps N          Maximum flow iterations (500)\n\
         \n  --scalar-steps N        Pollutant transport horizon (300)\n\
         \n  --seed N                Optimizer root seed (42)\n\
         \n  --csv PATH              Write final selected CFD fields as CSV\n\
         \n  --output DIR            Write Pareto/archive and convergence CSV files"
    );
}

fn print_evaluation(prefix: &str, design: Design, evaluation: &Evaluation) {
    println!(
        "{prefix} valid={} feasible={} scalar={:.9} exposure={:.9} max_receptor={:.9} fan_power={:.9} final_mass_fraction={:.9} clearance={:.9} flow_rate={:.6} low_velocity_fraction={:.6} mass_imbalance={:.6} pressure_drop={:.6e} flow_residual={:.6e} flow_iterations={} sources={} worst_exposure_source={:?}",
        evaluation.valid,
        evaluation.feasible(),
        evaluation.scalar_objective(),
        evaluation.exposure,
        evaluation.maximum_receptor,
        evaluation.fan_power,
        evaluation.final_mass_fraction,
        evaluation.clearance_time,
        evaluation.flow_rate_m2_s,
        evaluation.low_velocity_fraction,
        evaluation.mass_imbalance,
        evaluation.pressure_drop_lattice,
        evaluation.flow_residual,
        evaluation.flow_iterations,
        evaluation.source_count,
        evaluation.worst_exposure_source
    );
    println!("{prefix}_DESIGN {:?}", design.as_array());
    println!("{prefix}_CONSTRAINTS {:?}", evaluation.constraints());
}

fn run_single(problem: &RoomProblem, args: &Args) -> Result<(Design, Evaluation), Box<dyn Error>> {
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let config = RetryConfig {
        num_retries: args.retries,
        workers: args.workers,
        capacity: args.retries.min(500),
        max_evaluations: args.evaluations,
        seed: args.seed ^ 0xA076_1D64_78BD_642F,
        ..Default::default()
    };
    let objective = |x: &[f64]| problem.evaluate(x).scalar_objective();
    let started = Instant::now();
    let result = retry(&objective, &bounds, &config, |objective, context| {
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
            1,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("BiteOpt retry returned no finite candidate".into());
    }
    let design = Design::decode(&result.x).ok_or("BiteOpt returned a malformed design")?;
    let evaluation = problem.evaluate_design(design);
    println!(
        "SO_RUN evaluations={} retries={} seconds={:.6} evaluations_per_second={:.2}",
        result.evaluations,
        result.runs,
        started.elapsed().as_secs_f64(),
        result.evaluations as f64 / started.elapsed().as_secs_f64().max(1.0e-9)
    );
    print_evaluation("SO", design, &evaluation);
    Ok((design, evaluation))
}

#[derive(Clone, Debug)]
struct MultiPoint {
    design: Design,
    evaluation: Evaluation,
}

#[derive(Clone, Copy, Debug)]
struct ProgressSample {
    evaluations: usize,
    elapsed_seconds: f64,
    best_quality: f64,
    feasible_fraction: f64,
}

#[derive(Clone, Debug)]
struct MultiOutcome {
    selected: MultiPoint,
    pareto: Vec<MultiPoint>,
    evaluations: usize,
    generations: usize,
    elapsed: Duration,
    convergence: Vec<ProgressSample>,
}

fn run_multi(problem: &RoomProblem, args: &Args) -> Result<MultiOutcome, Box<dyn Error>> {
    let fitness = Fitness::bounded(DIMENSION, 8, &LOWER_BOUNDS, &UPPER_BOUNDS);
    let params = ModeParams {
        popsize: args.popsize as i32,
        nsga_update: true,
        seed: args.seed ^ 0xE703_7ED1_A0B4_28DB,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, 4, 4, None, &params)?;
    let generations = args.mo_evaluations.div_ceil(args.popsize);
    let started = Instant::now();
    let mut convergence = Vec::with_capacity(generations);
    let mut best_quality = f64::INFINITY;
    for generation in 0..generations {
        let xs = mode.ask();
        let batch = parallel_batch(&xs, args.workers as i32, |x| problem.evaluate(x));
        let mut feasible = 0usize;
        for evaluation in &batch {
            if evaluation.feasible() {
                feasible += 1;
                best_quality = best_quality.min(evaluation.scalar_objective());
            }
        }
        let ys: Vec<Vec<f64>> = batch.iter().map(Evaluation::mode_values).collect();
        mode.tell(&ys);
        convergence.push(ProgressSample {
            evaluations: (generation + 1) * args.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality,
            feasible_fraction: feasible as f64 / batch.len().max(1) as f64,
        });
    }
    let elapsed = started.elapsed();
    let population = mode.population();
    let evaluations = parallel_batch(&population, args.workers as i32, |candidate| {
        problem.evaluate(candidate)
    });
    let feasible: Vec<usize> = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.feasible())
        .map(|(index, _)| index)
        .collect();
    let values: Vec<Vec<f64>> = feasible
        .iter()
        .map(|&index| evaluations[index].objectives().to_vec())
        .collect();
    let local_front = pareto_indices(&values, 4)?;
    let mut front: Vec<usize> = local_front.iter().map(|&local| feasible[local]).collect();
    if front.is_empty() {
        return Err("MODE returned no feasible Pareto point".into());
    }
    front.sort_by(|&left, &right| {
        evaluations[left]
            .scalar_objective()
            .total_cmp(&evaluations[right].scalar_objective())
    });
    let selected = front[0];
    println!(
        "MO_RUN pareto={} feasible={} evaluations={} generations={} seconds={:.6} evaluations_per_second={:.2}",
        front.len(),
        feasible.len(),
        generations * args.popsize,
        generations,
        elapsed.as_secs_f64(),
        (generations * args.popsize) as f64 / elapsed.as_secs_f64().max(1.0e-9)
    );
    for (rank, &index) in front.iter().take(12).enumerate() {
        let evaluation = &evaluations[index];
        println!(
            "MO_POINT rank={} exposure={:.9} max_receptor={:.9} fan_power={:.9} final_mass_fraction={:.9} scalar={:.9} design={:?}",
            rank + 1,
            evaluation.exposure,
            evaluation.maximum_receptor,
            evaluation.fan_power,
            evaluation.final_mass_fraction,
            evaluation.scalar_objective(),
            population[index]
        );
    }
    let design = Design::decode(&population[selected]).ok_or("MODE returned a malformed design")?;
    let evaluation = evaluations[selected].clone();
    print_evaluation("MO_SELECTED", design, &evaluation);
    let pareto = front
        .into_iter()
        .map(|index| MultiPoint {
            design: Design::decode(&population[index])
                .expect("MODE population candidates have the expected dimension"),
            evaluation: evaluations[index].clone(),
        })
        .collect();
    Ok(MultiOutcome {
        selected: MultiPoint { design, evaluation },
        pareto,
        evaluations: generations * args.popsize,
        generations,
        elapsed,
        convergence,
    })
}

#[derive(Clone, Debug)]
struct QdPoint {
    niche_id: usize,
    grid_x: usize,
    grid_y: usize,
    design: Design,
    quality: f64,
    descriptors: [f64; 2],
    visits: u64,
    evaluation: Evaluation,
}

#[derive(Clone, Copy, Debug)]
struct QdProgress {
    evaluations: usize,
    elapsed_seconds: f64,
    coverage: f64,
    qd_score: f64,
    best_quality: f64,
}

#[derive(Clone, Debug)]
struct QdOutcome {
    selected: QdPoint,
    elites: Vec<QdPoint>,
    evaluations: usize,
    occupied: usize,
    capacity: usize,
    qd_score: f64,
    elapsed: Duration,
    convergence: Vec<QdProgress>,
}

fn run_qd(problem: &RoomProblem, args: &Args) -> Result<QdOutcome, Box<dyn Error>> {
    const DESCRIPTOR_LOWER: [f64; 2] = [0.09, 0.0];
    const DESCRIPTOR_UPPER: [f64; 2] = [2.025, 1.0];

    let generations = args.qd_evaluations.div_ceil(args.qd_chunk_size);
    let evaluations = generations * args.qd_chunk_size;
    let side = (args.qd_capacity as f64).sqrt() as usize;
    let mut rng = Rng::new(args.seed ^ 0x8EBC_6AF0_9C88_C6E3);
    let mut archive = Archive::try_new(
        DIMENSION,
        &DESCRIPTOR_LOWER,
        &DESCRIPTOR_UPPER,
        args.qd_capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER_BOUNDS, &UPPER_BOUNDS, &mut rng);
    let mut batch = |xs: &[Vec<f64>]| {
        parallel_batch(xs, args.workers as i32, |x| {
            let evaluation = problem.evaluate(x);
            let descriptors = vec![evaluation.flow_rate_m2_s, evaluation.low_velocity_fraction];
            if evaluation.feasible() {
                (evaluation.scalar_objective(), descriptors)
            } else {
                (f64::INFINITY, descriptors)
            }
        })
    };
    let parameters = MapElitesParams {
        generations,
        chunk_size: args.qd_chunk_size,
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
        &mut |generation, committed| {
            convergence.push(QdProgress {
                evaluations: generation * args.qd_chunk_size,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: committed.occupied() as f64 / committed.capacity() as f64,
                qd_score: committed.qd_score(),
                best_quality: committed.best_y(),
            });
        },
    )?;
    let elapsed = started.elapsed();
    let occupied_indices: Vec<usize> = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .collect();
    let elite_evaluations = parallel_batch(
        &occupied_indices
            .iter()
            .map(|&index| archive.xs()[index].clone())
            .collect::<Vec<_>>(),
        args.workers as i32,
        |x| problem.evaluate(x),
    );
    let mut elites = Vec::with_capacity(occupied_indices.len());
    for (&niche_id, evaluation) in occupied_indices.iter().zip(elite_evaluations) {
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            design: Design::decode(&archive.xs()[niche_id])
                .expect("MAP-Elites candidates have the expected dimension"),
            quality: archive.ys()[niche_id],
            descriptors: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            visits: archive.counts()[niche_id],
            evaluation,
        });
    }
    elites.sort_by(|left, right| left.quality.total_cmp(&right.quality));
    let selected = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites did not find a feasible room design")?;
    println!(
        "QD_RUN occupied={} capacity={} coverage={:.6} qd_score={:.9} evaluations={} generations={} seconds={:.6} evaluations_per_second={:.2}",
        archive.occupied(),
        archive.capacity(),
        archive.occupied() as f64 / archive.capacity() as f64,
        archive.qd_score(),
        evaluations,
        generations,
        elapsed.as_secs_f64(),
        evaluations as f64 / elapsed.as_secs_f64().max(1.0e-9)
    );
    for point in elites.iter().take(12) {
        println!(
            "QD_ELITE niche={} quality={:.9} flow_rate={:.6} low_velocity_fraction={:.6} design={:?}",
            point.niche_id,
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            point.design.as_array()
        );
    }
    print_evaluation("QD_SELECTED", selected.design, &selected.evaluation);
    Ok(QdOutcome {
        selected,
        elites,
        evaluations,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        elapsed,
        convergence,
    })
}

fn append_design_columns(output: &mut String) -> Result<(), std::fmt::Error> {
    for name in [
        "inlet_y",
        "inlet_width",
        "outlet_y",
        "outlet_width",
        "inlet_velocity",
        "baffle_x",
        "baffle_y",
        "baffle_length",
        "baffle_angle",
    ] {
        write!(output, ",decision_{name}")?;
    }
    Ok(())
}

fn append_design(output: &mut String, design: Design) -> Result<(), std::fmt::Error> {
    for value in design.as_array() {
        write!(output, ",{value}")?;
    }
    Ok(())
}

fn write_multi_results(directory: &PathBuf, outcome: &MultiOutcome) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut convergence =
        String::from("evaluations,elapsed_seconds,best_quality,feasible_fraction\n");
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.best_quality,
            sample.feasible_fraction
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let mut pareto = String::from(
        "point_id,selected,quality,exposure,maximum_receptor,fan_power,final_mass_fraction,clearance_time,flow_rate_m2_s,low_velocity_fraction",
    );
    append_design_columns(&mut pareto)?;
    pareto.push('\n');
    for (index, point) in outcome.pareto.iter().enumerate() {
        write!(
            pareto,
            "{index},{},{},{},{},{},{},{},{},{}",
            u8::from(index == 0),
            point.evaluation.scalar_objective(),
            point.evaluation.exposure,
            point.evaluation.maximum_receptor,
            point.evaluation.fan_power,
            point.evaluation.final_mass_fraction,
            point.evaluation.clearance_time,
            point.evaluation.flow_rate_m2_s,
            point.evaluation.low_velocity_fraction
        )?;
        append_design(&mut pareto, point.design)?;
        pareto.push('\n');
    }
    fs::write(directory.join("pareto.csv"), pareto)?;
    println!(
        "MO_OUTPUT directory={} files=pareto.csv,convergence.csv evaluations={} generations={} seconds={:.6}",
        directory.display(),
        outcome.evaluations,
        outcome.generations,
        outcome.elapsed.as_secs_f64()
    );
    Ok(())
}

fn write_qd_results(directory: &PathBuf, outcome: &QdOutcome) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut convergence =
        String::from("evaluations,elapsed_seconds,coverage,qd_score,best_quality\n");
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,selected,quality,descriptor_flow_rate_m2_s,descriptor_low_velocity_fraction,visits,exposure,maximum_receptor,fan_power,final_mass_fraction,clearance_time",
    );
    append_design_columns(&mut archive)?;
    archive.push('\n');
    for point in &outcome.elites {
        write!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            u8::from(point.niche_id == outcome.selected.niche_id),
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            point.visits,
            point.evaluation.exposure,
            point.evaluation.maximum_receptor,
            point.evaluation.fan_power,
            point.evaluation.final_mass_fraction,
            point.evaluation.clearance_time
        )?;
        append_design(&mut archive, point.design)?;
        archive.push('\n');
    }
    fs::write(directory.join("archive.csv"), archive)?;
    println!(
        "QD_OUTPUT directory={} files=archive.csv,convergence.csv evaluations={} occupied={}/{} qd_score={:.9} seconds={:.6}",
        directory.display(),
        outcome.evaluations,
        outcome.occupied,
        outcome.capacity,
        outcome.qd_score,
        outcome.elapsed.as_secs_f64()
    );
    Ok(())
}

fn write_validation_result(
    directory: &PathBuf,
    design: Design,
    evaluation: &Evaluation,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut output = String::from(
        "quality,exposure,maximum_receptor,fan_power,final_mass_fraction,clearance_time,flow_rate_m2_s,low_velocity_fraction,source_count,worst_source_x,worst_source_y",
    );
    append_design_columns(&mut output)?;
    output.push('\n');
    write!(
        output,
        "{},{},{},{},{},{},{},{},{},{},{}",
        evaluation.scalar_objective(),
        evaluation.exposure,
        evaluation.maximum_receptor,
        evaluation.fan_power,
        evaluation.final_mass_fraction,
        evaluation.clearance_time,
        evaluation.flow_rate_m2_s,
        evaluation.low_velocity_fraction,
        evaluation.source_count,
        evaluation.worst_exposure_source[0],
        evaluation.worst_exposure_source[1]
    )?;
    append_design(&mut output, design)?;
    output.push('\n');
    fs::write(directory.join("validation.csv"), output)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let config = RoomConfig {
        nx: args.nx,
        ny: args.ny,
        flow_steps: args.flow_steps,
        scalar_steps: args.scalar_steps,
        ..Default::default()
    };
    let problem = RoomProblem::new(config)?;
    let validation_problem = problem.validation_problem()?;
    println!(
        "CONFIG mode={} dimension={} grid={}x{} flow_steps={} scalar_steps={} workers={} retries={} evaluations_per_retry={} mo_evaluations={} popsize={} qd_evaluations={} qd_capacity={} qd_chunk_size={} seed={}",
        args.mode.name(),
        DIMENSION,
        args.nx,
        args.ny,
        args.flow_steps,
        args.scalar_steps,
        args.workers,
        args.retries,
        args.evaluations,
        args.mo_evaluations,
        args.popsize,
        args.qd_evaluations,
        args.qd_capacity,
        args.qd_chunk_size,
        args.seed
    );
    let baseline = Design::default();
    let baseline_evaluation = problem.evaluate_design(baseline);
    print_evaluation("BASELINE", baseline, &baseline_evaluation);
    print_evaluation(
        "BASELINE_VALIDATION",
        baseline,
        &validation_problem.evaluate_design(baseline),
    );

    let mut selected = (baseline, baseline_evaluation);
    if args.mode.includes_single() {
        selected = run_single(&problem, &args)?;
        print_evaluation(
            "SO_VALIDATION",
            selected.0,
            &validation_problem.evaluate_design(selected.0),
        );
    }
    if args.mode.includes_multi() {
        let multi = run_multi(&problem, &args)?;
        if let Some(directory) = &args.output {
            let directory = if args.mode.includes_qd() {
                directory.join("mode")
            } else {
                directory.clone()
            };
            write_multi_results(&directory, &multi)?;
            write_validation_result(
                &directory,
                multi.selected.design,
                &validation_problem.evaluate_design(multi.selected.design),
            )?;
        }
        print_evaluation(
            "MO_VALIDATION",
            multi.selected.design,
            &validation_problem.evaluate_design(multi.selected.design),
        );
        if !args.mode.includes_single()
            || multi.selected.evaluation.scalar_objective() < selected.1.scalar_objective()
        {
            selected = (multi.selected.design, multi.selected.evaluation);
        }
    }
    if args.mode.includes_qd() {
        let qd = run_qd(&problem, &args)?;
        if let Some(directory) = &args.output {
            let directory = if args.mode.includes_multi() {
                directory.join("qd")
            } else {
                directory.clone()
            };
            write_qd_results(&directory, &qd)?;
            write_validation_result(
                &directory,
                qd.selected.design,
                &validation_problem.evaluate_design(qd.selected.design),
            )?;
        }
        print_evaluation(
            "QD_VALIDATION",
            qd.selected.design,
            &validation_problem.evaluate_design(qd.selected.design),
        );
        if (!args.mode.includes_single() && !args.mode.includes_multi())
            || qd.selected.evaluation.scalar_objective() < selected.1.scalar_objective()
        {
            selected = (qd.selected.design, qd.selected.evaluation);
        }
    }
    if let Some(path) = &args.csv {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let detailed = problem
            .evaluate_detailed(selected.0)
            .ok_or("failed to reproduce selected design for CSV output")?;
        detailed.field.write_csv(path, problem.config())?;
        println!("CSV path={} rows={}", path.display(), args.nx * args.ny);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> std::vec::IntoIter<String> {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn defaults_are_a_small_research_run() {
        let args = Args::default();
        assert_eq!(args.mode, RunMode::Both);
        assert_eq!((args.workers, args.retries, args.evaluations), (4, 4, 200));
        assert_eq!((args.nx, args.ny), (40, 24));
        assert_eq!(
            (args.qd_evaluations, args.qd_capacity, args.qd_chunk_size),
            (512, 100, 64)
        );
    }

    #[test]
    fn parses_solver_optimizer_and_output_controls() {
        let args = Args::from_args(arguments(&[
            "--mode",
            "multi",
            "--workers",
            "8",
            "--retries",
            "2",
            "--evaluations",
            "50",
            "--mo-evaluations",
            "128",
            "--popsize",
            "16",
            "--qd-evaluations",
            "256",
            "--qd-capacity",
            "64",
            "--qd-chunk-size",
            "32",
            "--nx",
            "24",
            "--ny",
            "16",
            "--flow-steps",
            "100",
            "--scalar-steps",
            "80",
            "--seed",
            "7",
            "--csv",
            "result.csv",
            "--output",
            "results/test",
        ]))
        .unwrap();
        assert_eq!(args.mode, RunMode::Multi);
        assert_eq!((args.workers, args.popsize), (8, 16));
        assert_eq!((args.nx, args.ny), (24, 16));
        assert_eq!(args.csv, Some("result.csv".into()));
        assert_eq!(args.output, Some("results/test".into()));
    }

    #[test]
    fn rejects_invalid_counts_and_grids() {
        assert!(Args::from_args(arguments(&["--retries", "0"])).is_err());
        assert!(Args::from_args(arguments(&["--popsize", "3"])).is_err());
        assert!(Args::from_args(arguments(&["--nx", "4"])).is_err());
        assert!(Args::from_args(arguments(&["--mode", "unknown"])).is_err());
        assert!(Args::from_args(arguments(&["--qd-capacity", "15"])).is_err());
        assert!(Args::from_args(arguments(&["--qd-chunk-size", "3"])).is_err());
    }

    #[test]
    fn tiny_map_elites_run_fills_a_feasible_archive() {
        let problem = RoomProblem::new(RoomConfig {
            nx: 20,
            ny: 12,
            flow_steps: 120,
            scalar_steps: 80,
            flow_tolerance: 5.0e-3,
            minimum_fresh_air_m2_s: 0.0,
            maximum_mass_imbalance: 1.0,
            ..Default::default()
        })
        .unwrap();
        let outcome = run_qd(
            &problem,
            &Args {
                workers: 2,
                qd_evaluations: 16,
                qd_capacity: 16,
                qd_chunk_size: 4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(outcome.evaluations, 16);
        assert!(!outcome.elites.is_empty());
        assert!(outcome.selected.evaluation.feasible());
    }
}
