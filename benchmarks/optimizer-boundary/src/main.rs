mod nelder_mead;

use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use egobox_ego::{EgorBuilder, InfillStrategy};
use fcmaes_core::{De, DeParams, Fitness, Objective, Rng, parallel_batch};
use ndarray::{Array2, ArrayView2};

use nelder_mead::optimize as optimize_nm;

const POPULATION: usize = 16;
const BO_EVALUATIONS: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProblemKind {
    OpticalLens,
    CfdVentilation,
    RebopCrn,
    RebopResampled,
}

impl ProblemKind {
    fn name(self) -> &'static str {
        match self {
            Self::OpticalLens => "optical-lens",
            Self::CfdVentilation => "cfd-ventilation",
            Self::RebopCrn => "rebop-crn",
            Self::RebopResampled => "rebop-resampled",
        }
    }

    fn lower(self) -> Vec<f64> {
        match self {
            Self::OpticalLens => optical_lens_design::LOWER_BOUNDS.to_vec(),
            Self::CfdVentilation => cfd_room_ventilation::LOWER_BOUNDS.to_vec(),
            Self::RebopCrn | Self::RebopResampled => rebop_oscillator::lower_bounds().to_vec(),
        }
    }

    fn upper(self) -> Vec<f64> {
        match self {
            Self::OpticalLens => optical_lens_design::UPPER_BOUNDS.to_vec(),
            Self::CfdVentilation => cfd_room_ventilation::UPPER_BOUNDS.to_vec(),
            Self::RebopCrn | Self::RebopResampled => rebop_oscillator::upper_bounds().to_vec(),
        }
    }

    fn evaluate(self, x: &[f64], root_seed: u64, evaluation: u64) -> f64 {
        match self {
            Self::OpticalLens => {
                optical_lens_design::evaluate(x, optical_lens_design::PUBLICATION_GRID_RADIUS)
                    .map_or(1.0e99, |value| value.scalar_score())
            }
            Self::CfdVentilation => cfd_room_ventilation::RoomProblem::default()
                .evaluate(x)
                .scalar_objective(),
            Self::RebopCrn => rebop_oscillator::scalar_objective(
                x,
                &rebop_oscillator::EvaluationConfig::default(),
            ),
            Self::RebopResampled => {
                let Ok(rates) = rebop_oscillator::LogRates::from_slice(x) else {
                    return 1.0e99;
                };
                let mut state = splitmix64(root_seed ^ evaluation);
                let mut seeds = [0_u64; 4];
                for seed in &mut seeds {
                    state = splitmix64(state);
                    *seed = state;
                }
                rebop_oscillator::evaluate_with_seeds(&rates, 20.0, &seeds, false).scalar_score
            }
        }
    }

    fn validation(self, x: &[f64]) -> f64 {
        match self {
            Self::RebopCrn | Self::RebopResampled => rebop_oscillator::evaluate_validation(
                x,
                &rebop_oscillator::EvaluationConfig {
                    target_period: 20.0,
                    replications: 8,
                },
                false,
            )
            .map_or(1.0e99, |value| value.scalar_score),
            _ => self.evaluate(x, 0, 0),
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct MeasuredObjective {
    problem: ProblemKind,
    root_seed: u64,
    calls: AtomicU64,
    simulator_nanos: AtomicU64,
}

impl MeasuredObjective {
    fn new(problem: ProblemKind, root_seed: u64) -> Self {
        Self {
            problem,
            root_seed,
            calls: AtomicU64::new(0),
            simulator_nanos: AtomicU64::new(0),
        }
    }

    fn evaluate_indexed(&self, x: &[f64], evaluation: u64) -> f64 {
        let started = Instant::now();
        let value = self.problem.evaluate(x, self.root_seed, evaluation);
        self.simulator_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        value
    }

    fn evaluate_batch(&self, xs: &[Vec<f64>], workers: usize) -> Vec<f64> {
        let base = self.calls.fetch_add(xs.len() as u64, Ordering::Relaxed);
        if workers == 1 {
            return xs
                .iter()
                .enumerate()
                .map(|(offset, x)| self.evaluate_indexed(x, base + offset as u64))
                .collect();
        }
        parallel_batch(xs, workers as i32, |x| {
            let offset = xs
                .iter()
                .position(|candidate| candidate.as_slice() == x)
                .unwrap_or(0);
            self.evaluate_indexed(x, base + offset as u64)
        })
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn simulator_seconds(&self) -> f64 {
        self.simulator_nanos.load(Ordering::Relaxed) as f64 / 1.0e9
    }
}

impl Objective for MeasuredObjective {
    fn nobj(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self.eval_scalar(x)]
    }

    fn eval_scalar(&self, x: &[f64]) -> f64 {
        let evaluation = self.calls.fetch_add(1, Ordering::Relaxed);
        self.evaluate_indexed(x, evaluation)
    }
}

fn as_obj_fn<F>(function: F) -> F
where
    F: for<'a, 'b> Fn(&'a ArrayView2<'b, f64>) -> Array2<f64> + Clone,
{
    function
}

fn start_points(problem: ProblemKind, seed: u64, count: usize) -> Vec<Vec<f64>> {
    let lower = problem.lower();
    let upper = problem.upper();
    let mut rng = Rng::new(seed ^ 0x5eed_1234_a55a_19c3);
    (0..count)
        .map(|_| {
            lower
                .iter()
                .zip(&upper)
                .map(|(low, high)| low + rng.uniform01() * (high - low))
                .collect()
        })
        .collect()
}

fn steps(problem: ProblemKind, scale: f64) -> Vec<f64> {
    problem
        .lower()
        .iter()
        .zip(problem.upper())
        .map(|(low, high)| scale * (high - low))
        .collect()
}

fn make_de(problem: ProblemKind, seed: u64, guess: &[f64]) -> De {
    let lower = problem.lower();
    let upper = problem.upper();
    let fitness = Fitness::bounded(guess.len(), 1, &lower, &upper);
    De::new(
        fitness,
        guess,
        &steps(problem, 0.3),
        None,
        &DeParams {
            popsize: POPULATION as i32,
            max_evaluations: u64::MAX,
            stop_fitness: f64::NEG_INFINITY,
            seed: seed ^ 0xd3_d3_d3_d3,
            ..Default::default()
        },
    )
}

fn run_de_batches(
    problem: ProblemKind,
    seed: u64,
    batches: usize,
    workers: usize,
    objective: &MeasuredObjective,
) -> fcmaes_core::DeResult {
    let guess = start_points(problem, seed, 1).remove(0);
    let mut de = make_de(problem, seed, &guess);
    for _ in 0..batches {
        let xs = de.ask();
        let ys = objective.evaluate_batch(&xs, workers);
        de.tell(&ys);
    }
    de.result()
}

#[derive(Clone, Copy)]
struct RefinerProtocol {
    name: &'static str,
    workers: usize,
    rounds: usize,
}

struct RefinerRecord {
    protocol: &'static str,
    problem: &'static str,
    seed: u64,
    arm: &'static str,
    workers: usize,
    rounds: usize,
    calls: u64,
    training_score: f64,
    validation_score: f64,
    simulator_seconds: f64,
    wall_seconds: f64,
}

struct CandidateOutcome<'a> {
    objective: &'a MeasuredObjective,
    x: &'a [f64],
    training_score: f64,
    wall_seconds: f64,
}

fn record(
    protocol: RefinerProtocol,
    problem: ProblemKind,
    seed: u64,
    arm: &'static str,
    outcome: CandidateOutcome<'_>,
) -> RefinerRecord {
    RefinerRecord {
        protocol: protocol.name,
        problem: problem.name(),
        seed,
        arm,
        workers: protocol.workers,
        rounds: protocol.rounds,
        calls: outcome.objective.calls(),
        training_score: outcome.training_score,
        validation_score: problem.validation(outcome.x),
        simulator_seconds: outcome.objective.simulator_seconds(),
        wall_seconds: outcome.wall_seconds,
    }
}

fn run_refiner_case(
    problem: ProblemKind,
    seed: u64,
    protocol: RefinerProtocol,
) -> Vec<RefinerRecord> {
    let batch_rounds = POPULATION.div_ceil(protocol.workers);
    assert_eq!(protocol.rounds % batch_rounds, 0);
    let full_batches = protocol.rounds / batch_rounds;
    let dimension = problem.lower().len();
    let minimum_tail = 2 * (dimension + 1);
    let requested_tail = (protocol.rounds as f64 * 0.2).round() as usize;
    let mut tail_rounds = requested_tail.max(minimum_tail);
    let mut head_rounds = protocol.rounds.saturating_sub(tail_rounds);
    head_rounds -= head_rounds % batch_rounds;
    tail_rounds = protocol.rounds - head_rounds;
    let head_batches = head_rounds / batch_rounds;
    let starts = start_points(problem, seed, protocol.workers.max(1));
    let lower = problem.lower();
    let upper = problem.upper();
    let nm_step = steps(problem, 0.02);
    let mut records = Vec::new();

    let objective = MeasuredObjective::new(problem, seed ^ 0x1111);
    let started = Instant::now();
    let full = run_de_batches(problem, seed, full_batches, protocol.workers, &objective);
    records.push(record(
        protocol,
        problem,
        seed,
        "de",
        CandidateOutcome {
            objective: &objective,
            x: &full.x,
            training_score: full.y,
            wall_seconds: started.elapsed().as_secs_f64(),
        },
    ));

    let objective = MeasuredObjective::new(problem, seed ^ 0x1111);
    let started = Instant::now();
    let head = run_de_batches(problem, seed, head_batches, protocol.workers, &objective);
    let head_wall = started.elapsed().as_secs_f64();
    records.push(record(
        protocol,
        problem,
        seed,
        "de-head",
        CandidateOutcome {
            objective: &objective,
            x: &head.x,
            training_score: head.y,
            wall_seconds: head_wall,
        },
    ));
    let tail_started = Instant::now();
    let tail = optimize_nm(
        &objective,
        &head.x,
        &nm_step,
        &lower,
        &upper,
        tail_rounds as u64,
    );
    records.push(record(
        protocol,
        problem,
        seed,
        "de+nm",
        CandidateOutcome {
            objective: &objective,
            x: &tail.x,
            training_score: tail.y,
            wall_seconds: head_wall + tail_started.elapsed().as_secs_f64(),
        },
    ));

    let objective = MeasuredObjective::new(problem, seed ^ 0x2222);
    let started = Instant::now();
    let serial = optimize_nm(
        &objective,
        &starts[0],
        &steps(problem, 0.15),
        &lower,
        &upper,
        protocol.rounds as u64,
    );
    records.push(record(
        protocol,
        problem,
        seed,
        "nm-serial",
        CandidateOutcome {
            objective: &objective,
            x: &serial.x,
            training_score: serial.y,
            wall_seconds: started.elapsed().as_secs_f64(),
        },
    ));

    if protocol.workers > 1 {
        let started = Instant::now();
        let results = parallel_batch(&starts, protocol.workers as i32, |start| {
            let index = starts
                .iter()
                .position(|candidate| candidate.as_slice() == start)
                .unwrap_or(0);
            let objective = MeasuredObjective::new(
                problem,
                seed ^ 0x3333 ^ (index as u64).wrapping_mul(0x9e37_79b9),
            );
            let result = optimize_nm(
                &objective,
                start,
                &steps(problem, 0.15),
                &lower,
                &upper,
                protocol.rounds as u64,
            );
            (result, objective.simulator_seconds())
        });
        let (best, _) = results
            .iter()
            .min_by(|left, right| left.0.y.total_cmp(&right.0.y))
            .expect("at least one multistart");
        records.push(RefinerRecord {
            protocol: protocol.name,
            problem: problem.name(),
            seed,
            arm: "nm-multistart",
            workers: protocol.workers,
            rounds: protocol.rounds,
            calls: results.iter().map(|value| value.0.evaluations).sum(),
            training_score: best.y,
            validation_score: problem.validation(&best.x),
            simulator_seconds: results.iter().map(|value| value.1).sum(),
            wall_seconds: started.elapsed().as_secs_f64(),
        });
    }
    records
}

fn write_refiner(path: &Path, seeds: u64) {
    let protocols = [
        RefinerProtocol {
            name: "serial-r160-w1",
            workers: 1,
            rounds: 160,
        },
        RefinerProtocol {
            name: "parallel-r60-w16",
            workers: 16,
            rounds: 60,
        },
    ];
    let problems = [
        ProblemKind::OpticalLens,
        ProblemKind::CfdVentilation,
        ProblemKind::RebopCrn,
        ProblemKind::RebopResampled,
    ];
    let file = File::create(path.join("refiner-raw.tsv")).expect("create refiner results");
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "protocol\tproblem\tseed\tarm\tworkers\trounds\tcalls\ttraining_score\tvalidation_score\tsimulator_seconds\twall_seconds"
    )
    .unwrap();
    for protocol in protocols {
        for problem in problems {
            let mut blocks: Vec<(u64, Vec<RefinerRecord>)> = if protocol.workers == 1 {
                let next = AtomicU64::new(0);
                let collected = Mutex::new(Vec::new());
                let jobs = usize::try_from(seeds.min(16)).unwrap_or(1).max(1);
                std::thread::scope(|scope| {
                    for _ in 0..jobs {
                        scope.spawn(|| {
                            loop {
                                let seed = next.fetch_add(1, Ordering::Relaxed);
                                if seed >= seeds {
                                    break;
                                }
                                eprintln!(
                                    "refiner {} {} seed {}/{}",
                                    protocol.name,
                                    problem.name(),
                                    seed + 1,
                                    seeds
                                );
                                collected
                                    .lock()
                                    .expect("result mutex")
                                    .push((seed, run_refiner_case(problem, seed, protocol)));
                            }
                        });
                    }
                });
                collected.into_inner().expect("result mutex")
            } else {
                (0..seeds)
                    .map(|seed| {
                        eprintln!(
                            "refiner {} {} seed {}/{}",
                            protocol.name,
                            problem.name(),
                            seed + 1,
                            seeds
                        );
                        (seed, run_refiner_case(problem, seed, protocol))
                    })
                    .collect()
            };
            blocks.sort_by_key(|(seed, _)| *seed);
            for (_, block) in blocks {
                for value in block {
                    writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12e}\t{:.12e}\t{:.6}\t{:.6}",
                        value.protocol,
                        value.problem,
                        value.seed,
                        value.arm,
                        value.workers,
                        value.rounds,
                        value.calls,
                        value.training_score,
                        value.validation_score,
                        value.simulator_seconds,
                        value.wall_seconds
                    )
                    .unwrap();
                }
                output.flush().expect("flush refiner result block");
            }
        }
    }
}

#[derive(Clone)]
struct TracePoint {
    call: usize,
    best: f64,
    overhead_seconds: f64,
}

struct TraceObjective {
    problem: ProblemKind,
    root_seed: u64,
    started: Instant,
    calls: AtomicU64,
    simulator_nanos: AtomicU64,
    best: Mutex<f64>,
    trace: Mutex<Vec<TracePoint>>,
}

impl TraceObjective {
    fn new(problem: ProblemKind, root_seed: u64) -> Self {
        Self {
            problem,
            root_seed,
            started: Instant::now(),
            calls: AtomicU64::new(0),
            simulator_nanos: AtomicU64::new(0),
            best: Mutex::new(f64::INFINITY),
            trace: Mutex::new(Vec::new()),
        }
    }

    fn trace(&self) -> Vec<TracePoint> {
        self.trace.lock().expect("trace mutex").clone()
    }
}

impl Objective for TraceObjective {
    fn nobj(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self.eval_scalar(x)]
    }

    fn eval_scalar(&self, x: &[f64]) -> f64 {
        let evaluation = self.calls.fetch_add(1, Ordering::Relaxed);
        let simulator_started = Instant::now();
        let value = self.problem.evaluate(x, self.root_seed, evaluation);
        self.simulator_nanos.fetch_add(
            simulator_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        let mut best = self.best.lock().expect("best mutex");
        *best = best.min(value);
        let simulator = self.simulator_nanos.load(Ordering::Relaxed) as f64 / 1.0e9;
        let overhead = (self.started.elapsed().as_secs_f64() - simulator).max(0.0);
        self.trace.lock().expect("trace mutex").push(TracePoint {
            call: evaluation as usize + 1,
            best: *best,
            overhead_seconds: overhead,
        });
        value
    }
}

fn run_bo_trace(problem: ProblemKind, seed: u64) -> Vec<TracePoint> {
    let objective = TraceObjective::new(problem, seed ^ 0x8080);
    let lower = problem.lower();
    let upper = problem.upper();
    let limits = Array2::from_shape_vec(
        (lower.len(), 2),
        lower
            .iter()
            .zip(&upper)
            .flat_map(|(low, high)| [*low, *high])
            .collect(),
    )
    .expect("limits shape");
    let evaluate = |x: &ArrayView2<f64>| {
        let mut values = Array2::zeros((x.nrows(), 1));
        for row in 0..x.nrows() {
            values[[row, 0]] = objective.eval_scalar(&x.row(row).to_vec());
        }
        values
    };
    EgorBuilder::optimize(as_obj_fn(evaluate))
        .configure(|config| {
            config
                .infill_strategy(InfillStrategy::EI)
                .n_doe(16)
                .max_iters(BO_EVALUATIONS - 16)
                .seed(seed ^ 0xb0)
        })
        .min_within(&limits)
        .and_then(|optimizer| optimizer.run())
        .expect("egobox run");
    objective.trace()
}

fn run_de_trace(problem: ProblemKind, seed: u64) -> Vec<TracePoint> {
    let objective = TraceObjective::new(problem, seed ^ 0x8080);
    let guess = start_points(problem, seed, 1).remove(0);
    let mut de = make_de(problem, seed, &guess);
    for _ in 0..BO_EVALUATIONS / POPULATION {
        let xs = de.ask();
        let ys: Vec<f64> = xs.iter().map(|x| objective.eval_scalar(x)).collect();
        de.tell(&ys);
    }
    objective.trace()
}

fn write_bo(path: &Path, seeds: u64) {
    let file = File::create(path.join("bo-trace.tsv")).expect("create BO trace");
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "problem\tseed\tarm\tcall\tbest_score\toverhead_seconds"
    )
    .unwrap();
    for problem in [ProblemKind::OpticalLens, ProblemKind::CfdVentilation] {
        for seed in 0..seeds {
            for arm in ["de", "bo"] {
                eprintln!("bo {} {} seed {}/{}", problem.name(), arm, seed + 1, seeds);
                let trace = if arm == "de" {
                    run_de_trace(problem, seed)
                } else {
                    run_bo_trace(problem, seed)
                };
                for point in trace {
                    writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{:.12e}\t{:.9}",
                        problem.name(),
                        seed,
                        arm,
                        point.call,
                        point.best,
                        point.overhead_seconds
                    )
                    .unwrap();
                }
                output.flush().expect("flush BO trace block");
            }
        }
    }
}

fn main() {
    let mut mode = String::from("all");
    let mut seeds = 20_u64;
    let mut output = PathBuf::from("results/decision-v2");
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut next = || arguments.next().expect("missing option value");
        match argument.as_str() {
            "--mode" => mode = next(),
            "--seeds" => seeds = next().parse().expect("integer seeds"),
            "--output" => output = PathBuf::from(next()),
            other => panic!("unknown option {other}"),
        }
    }
    fs::create_dir_all(&output).expect("create output directory");
    if mode == "all" || mode == "refiner" {
        write_refiner(&output, seeds);
    }
    if mode == "all" || mode == "bo" {
        write_bo(&output, seeds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_protocols_are_exact() {
        for protocol in [
            RefinerProtocol {
                name: "serial",
                workers: 1,
                rounds: 160,
            },
            RefinerProtocol {
                name: "parallel",
                workers: 16,
                rounds: 60,
            },
        ] {
            let batch_rounds = POPULATION.div_ceil(protocol.workers);
            assert_eq!(protocol.rounds % batch_rounds, 0);
            let full_calls = protocol.rounds / batch_rounds * POPULATION;
            assert_eq!(full_calls.div_ceil(protocol.workers), protocol.rounds);
        }
    }

    #[test]
    fn fixed_and_resampled_rebop_have_expected_repeatability() {
        let x = rebop_oscillator::base_log_rates();
        let fixed_a = ProblemKind::RebopCrn.evaluate(&x, 7, 0);
        let fixed_b = ProblemKind::RebopCrn.evaluate(&x, 7, 99);
        assert_eq!(fixed_a, fixed_b);
        let noisy_a = ProblemKind::RebopResampled.evaluate(&x, 7, 0);
        let noisy_b = ProblemKind::RebopResampled.evaluate(&x, 7, 1);
        assert_ne!(noisy_a, noisy_b);
    }
}
