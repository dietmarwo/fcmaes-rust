//! Shared native optimizer sequence used by both example binaries.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use fcmaes_core::{
    AdvancedRetryConfig, BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryConfig,
    RetryContext, RetryResult, RetryRunResult, Rng, advanced_retry, optimize_bite, retry,
};

use crate::problems::Problem;

#[derive(Clone, Debug)]
pub struct Cli {
    pub retries: usize,
    pub evaluations: u64,
    pub workers: usize,
    pub seed: u64,
    pub max_eval_fac: f64,
    pub check_interval: usize,
    pub value_limit: f64,
    pub stop_fitness: f64,
    pub problem: Option<String>,
    pub progress_interval: f64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            retries: 32,
            evaluations: 50_000,
            workers: 0,
            seed: 0,
            max_eval_fac: 50.0,
            check_interval: 100,
            value_limit: f64::INFINITY,
            stop_fitness: f64::NEG_INFINITY,
            problem: None,
            progress_interval: 0.0,
        }
    }
}

impl Cli {
    pub const USAGE: &str = "options:\n\
        \x20 --problem NAME          run one problem (for example messenger-full)\n\
        \x20 --retries N             optimization retries per problem\n\
        \x20 --evaluations N         initial evaluations per retry\n\
        \x20 --workers N             retry workers; 0 selects available parallelism\n\
        \x20 --seed N                random seed\n\
        \x20 --value-limit N         only retain results below this value\n\
        \x20 --stop-fitness N        stop retry after reaching this objective value\n\
        \x20 --progress-interval N   print live progress every N seconds; 0 disables it\n\
        \x20 --max-eval-fac N        advanced retry maximum evaluation factor\n\
        \x20 --check-interval N      advanced retry checkpoint interval\n\
        \x20 --help                  show this help";

    pub fn from_env() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!("{}", Self::USAGE);
            std::process::exit(0);
        }
        Self::from_args(args).unwrap_or_else(|message| {
            eprintln!("{message}\n\n{}", Self::USAGE);
            std::process::exit(2);
        })
    }

    pub fn from_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cli = Self::default();
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            let argument = argument.as_ref();
            let value = args
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?;
            let value = value.as_ref();
            match argument {
                "--problem" => cli.problem = Some(value.to_owned()),
                "--retries" => {
                    cli.retries = value
                        .parse()
                        .map_err(|_| "--retries requires an integer".to_owned())?
                }
                "--evaluations" => {
                    cli.evaluations = value
                        .parse()
                        .map_err(|_| "--evaluations requires an integer".to_owned())?
                }
                "--workers" => {
                    cli.workers = value
                        .parse()
                        .map_err(|_| "--workers requires an integer".to_owned())?
                }
                "--seed" => {
                    cli.seed = value
                        .parse()
                        .map_err(|_| "--seed requires an integer".to_owned())?
                }
                "--value-limit" => {
                    cli.value_limit = value
                        .parse()
                        .map_err(|_| "--value-limit requires a number".to_owned())?;
                    if cli.value_limit.is_nan() {
                        return Err("--value-limit must not be NaN".to_owned());
                    }
                }
                "--stop-fitness" => {
                    cli.stop_fitness = value
                        .parse()
                        .map_err(|_| "--stop-fitness requires a number".to_owned())?;
                    if cli.stop_fitness.is_nan() {
                        return Err("--stop-fitness must not be NaN".to_owned());
                    }
                }
                "--progress-interval" => {
                    cli.progress_interval = value
                        .parse()
                        .map_err(|_| "--progress-interval requires a number".to_owned())?;
                    if !cli.progress_interval.is_finite() || cli.progress_interval < 0.0 {
                        return Err(
                            "--progress-interval must be finite and non-negative".to_owned()
                        );
                    }
                }
                "--max-eval-fac" => {
                    cli.max_eval_fac = value
                        .parse()
                        .map_err(|_| "--max-eval-fac requires a number".to_owned())?;
                    if !cli.max_eval_fac.is_finite() || cli.max_eval_fac < 1.0 {
                        return Err("--max-eval-fac must be finite and at least 1".to_owned());
                    }
                }
                "--check-interval" => {
                    cli.check_interval = value
                        .parse()
                        .map_err(|_| "--check-interval requires an integer".to_owned())?
                }
                _ => return Err(format!("unknown option '{argument}'")),
            }
        }
        Ok(cli)
    }
}

#[derive(Debug)]
struct ProgressState {
    problem: &'static str,
    total_retries: usize,
    started: Instant,
    evaluations: AtomicU64,
    completed_retries: AtomicUsize,
    best_bits: AtomicU64,
}

impl ProgressState {
    fn new(problem: &'static str, total_retries: usize) -> Self {
        Self {
            problem,
            total_retries,
            started: Instant::now(),
            evaluations: AtomicU64::new(0),
            completed_retries: AtomicUsize::new(0),
            best_bits: AtomicU64::new(f64::INFINITY.to_bits()),
        }
    }

    #[inline]
    fn record_evaluation(&self, value: f64) {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        if !value.is_finite() {
            return;
        }
        let mut current = self.best_bits.load(Ordering::Relaxed);
        loop {
            if value >= f64::from_bits(current) {
                return;
            }
            match self.best_bits.compare_exchange_weak(
                current,
                value.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn retry_completed(&self) {
        self.completed_retries.fetch_add(1, Ordering::Relaxed);
    }

    fn print(&self, final_update: bool) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let evaluations = self.evaluations.load(Ordering::Relaxed);
        let completed = self.completed_retries.load(Ordering::Relaxed);
        let best = f64::from_bits(self.best_bits.load(Ordering::Relaxed));
        let rate = evaluations as f64 / elapsed.max(1.0e-9);
        eprintln!(
            "progress problem=\"{}\" final={} elapsed={:.1}s evaluations={} evals_per_second={:.0} retries={}/{} best={:.12}",
            self.problem,
            final_update,
            elapsed,
            evaluations,
            rate,
            completed,
            self.total_retries,
            best
        );
    }
}

struct ProgressReporter {
    state: Arc<ProgressState>,
    stop: mpsc::Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl ProgressReporter {
    fn start(problem: &'static str, total_retries: usize, interval_seconds: f64) -> Self {
        let state = Arc::new(ProgressState::new(problem, total_retries));
        let thread_state = Arc::clone(&state);
        let interval = Duration::from_secs_f64(interval_seconds);
        let (stop, receiver) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("fcmaes-progress".to_owned())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = receiver.recv_timeout(interval) {
                    thread_state.print(false);
                }
            })
            .expect("failed to start progress reporter");
        Self {
            state,
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) {
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.state.print(true);
    }
}

fn de_cma_run<O>(objective: &O, context: &RetryContext) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let dim = context.bounds.dim();
    let de_budget = (context.max_evaluations * 2 / 5).max(31);
    let cma_budget = context.max_evaluations.saturating_sub(de_budget).max(31);
    let de_fit = Fitness::bounded(dim, 1, context.bounds.lower(), context.bounds.upper());
    let de_sigma: Vec<f64> = context
        .sdev
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&sigma, (&lo, &hi))| sigma * (hi - lo))
        .collect();
    let guess = context.guess.as_deref().unwrap_or(&[]);
    let mut de = De::new(
        de_fit,
        guess,
        if guess.is_empty() { &[] } else { &de_sigma },
        None,
        &DeParams {
            max_evaluations: de_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: context.run_id as i64,
            ..Default::default()
        },
    );
    let de_result = de.optimize(objective);

    let mut cma_fit = Fitness::bounded(dim, 1, context.bounds.lower(), context.bounds.upper());
    cma_fit.set_normalize(true);
    let mut cma = Cmaes::new(
        cma_fit,
        &de_result.x,
        &context.sdev,
        &CmaesParams {
            max_evaluations: cma_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed ^ 0xA076_1D64_78BD_642F,
            runid: context.run_id as i64,
            ..Default::default()
        },
    );
    let cma_result = cma.optimize(objective, 1);
    let (x, y) = if cma_result.y < de_result.y {
        (cma_result.x, cma_result.y)
    } else {
        (de_result.x, de_result.y)
    };
    RetryRunResult {
        x,
        y,
        evaluations: de_result.evaluations + cma_result.evaluations,
    }
}

fn bite_run<O>(objective: &O, context: &RetryContext, stop_fitness: f64) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    // The retry coordinator has already consumed the worker stream's sdev
    // draw and provided this independent child seed. Use it for the uniform
    // initial guess and then BiteOpt's own seed.
    let mut rng = Rng::new(context.seed);
    let sampled_guess: Vec<f64> = context
        .bounds
        .lower()
        .iter()
        .zip(context.bounds.upper())
        .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
        .collect();
    let guess = context.guess.as_deref().unwrap_or(&sampled_guess);
    let bite_seed = (rng.uniform01() * u32::MAX as f64) as u64;
    let result = optimize_bite(
        objective,
        context.bounds.lower(),
        context.bounds.upper(),
        Some(guess),
        &BiteParams {
            max_evaluations: context.max_evaluations,
            stop_fitness,
            seed: bite_seed,
            runid: context.run_id as i64,
            ..Default::default()
        },
        1,
    );
    RetryRunResult {
        x: result.x,
        y: result.y,
        evaluations: result.evaluations,
    }
}

pub fn run_basic(problem: &Problem, cli: &Cli) -> RetryResult {
    let config = RetryConfig {
        num_retries: cli.retries,
        workers: cli.workers,
        max_evaluations: cli.evaluations,
        seed: cli.seed,
        value_limit: cli.value_limit,
        stop_fitness: cli.stop_fitness,
        statistic_num: 0,
        ..Default::default()
    };
    if cli.progress_interval <= 0.0 {
        return retry(&problem.objective, &problem.bounds, &config, de_cma_run);
    }

    let reporter = ProgressReporter::start(problem.name, cli.retries, cli.progress_interval);
    let evaluation_state = Arc::clone(&reporter.state);
    let completion_state = Arc::clone(&reporter.state);
    let objective = move |x: &[f64]| {
        let value = (problem.objective)(x);
        evaluation_state.record_evaluation(value);
        value
    };
    let result = retry(
        &objective,
        &problem.bounds,
        &config,
        move |objective, context| {
            let result = de_cma_run(objective, context);
            completion_state.retry_completed();
            result
        },
    );
    reporter.finish();
    result
}

/// Run independent BiteOpt restarts through the basic retry coordinator.
pub fn run_bite(problem: &Problem, cli: &Cli) -> RetryResult {
    let config = RetryConfig {
        num_retries: cli.retries,
        workers: cli.workers,
        max_evaluations: cli.evaluations,
        seed: cli.seed,
        value_limit: cli.value_limit,
        stop_fitness: cli.stop_fitness,
        statistic_num: 0,
        ..Default::default()
    };
    if cli.progress_interval <= 0.0 {
        return retry(
            &problem.objective,
            &problem.bounds,
            &config,
            |objective, context| bite_run(objective, context, cli.stop_fitness),
        );
    }

    let reporter = ProgressReporter::start(problem.name, cli.retries, cli.progress_interval);
    let evaluation_state = Arc::clone(&reporter.state);
    let completion_state = Arc::clone(&reporter.state);
    let objective = move |x: &[f64]| {
        let value = (problem.objective)(x);
        evaluation_state.record_evaluation(value);
        value
    };
    let result = retry(
        &objective,
        &problem.bounds,
        &config,
        move |objective, context| {
            let result = bite_run(objective, context, cli.stop_fitness);
            completion_state.retry_completed();
            result
        },
    );
    reporter.finish();
    result
}

pub fn run_advanced(problem: &Problem, cli: &Cli) -> RetryResult {
    let config = AdvancedRetryConfig {
        retry: RetryConfig {
            num_retries: cli.retries,
            workers: cli.workers,
            max_evaluations: cli.evaluations,
            seed: cli.seed,
            value_limit: cli.value_limit,
            stop_fitness: cli.stop_fitness,
            statistic_num: 0,
            ..Default::default()
        },
        check_interval: cli.check_interval,
        max_eval_fac: cli.max_eval_fac,
        ..Default::default()
    };
    if cli.progress_interval <= 0.0 {
        return advanced_retry(&problem.objective, &problem.bounds, &config, de_cma_run);
    }

    let reporter = ProgressReporter::start(problem.name, cli.retries, cli.progress_interval);
    let evaluation_state = Arc::clone(&reporter.state);
    let completion_state = Arc::clone(&reporter.state);
    let objective = move |x: &[f64]| {
        let value = (problem.objective)(x);
        evaluation_state.record_evaluation(value);
        value
    };
    let result = advanced_retry(
        &objective,
        &problem.bounds,
        &config,
        move |objective, context| {
            let result = de_cma_run(objective, context);
            completion_state.retry_completed();
            result
        },
    );
    reporter.finish();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|value| value * value).sum()
    }

    #[test]
    fn native_sequence_converges() {
        let problem = Problem {
            name: "sphere",
            objective: sphere,
            bounds: fcmaes_core::RetryBounds::new(vec![-5.0; 3], vec![5.0; 3]).unwrap(),
        };
        let result = run_basic(
            &problem,
            &Cli {
                retries: 2,
                evaluations: 2_000,
                workers: 1,
                seed: 7,
                max_eval_fac: 50.0,
                check_interval: 100,
                ..Default::default()
            },
        );
        assert!(result.y < 1.0e-6, "sphere not solved: {}", result.y);
    }

    #[test]
    fn native_advanced_sequence_converges() {
        let problem = Problem {
            name: "sphere",
            objective: sphere,
            bounds: fcmaes_core::RetryBounds::new(vec![-5.0; 2], vec![5.0; 2]).unwrap(),
        };
        let result = run_advanced(
            &problem,
            &Cli {
                retries: 2,
                evaluations: 1_000,
                workers: 1,
                seed: 11,
                max_eval_fac: 50.0,
                check_interval: 100,
                ..Default::default()
            },
        );
        assert!(result.success);
        assert!(result.y < 1.0e-6, "sphere not solved: {}", result.y);
    }

    #[test]
    fn native_bite_retry_converges_and_honors_the_budget() {
        let problem = Problem {
            name: "sphere",
            objective: sphere,
            bounds: fcmaes_core::RetryBounds::new(vec![-5.0; 3], vec![5.0; 3]).unwrap(),
        };
        let result = run_bite(
            &problem,
            &Cli {
                retries: 2,
                evaluations: 3_000,
                workers: 1,
                seed: 17,
                ..Default::default()
            },
        );
        assert!(result.y < 1.0e-6, "sphere not solved: {}", result.y);
        assert_eq!(result.runs, 2);
        assert_eq!(result.evaluations, 6_000);
    }

    #[test]
    fn cli_parses_problem_and_retry_controls() {
        let cli = Cli::from_args([
            "--problem",
            "messenger-full",
            "--retries",
            "50000",
            "--evaluations",
            "1500",
            "--workers",
            "16",
            "--value-limit",
            "12",
            "--stop-fitness",
            "1.96769",
            "--progress-interval",
            "10",
            "--max-eval-fac",
            "50",
            "--check-interval",
            "100",
            "--seed",
            "7",
        ])
        .unwrap();
        assert_eq!(cli.problem.as_deref(), Some("messenger-full"));
        assert_eq!(cli.retries, 50_000);
        assert_eq!(cli.evaluations, 1_500);
        assert_eq!(cli.workers, 16);
        assert_eq!(cli.value_limit, 12.0);
        assert_eq!(cli.stop_fitness, 1.96769);
        assert_eq!(cli.progress_interval, 10.0);
        assert_eq!(cli.max_eval_fac, 50.0);
        assert_eq!(cli.check_interval, 100);
        assert_eq!(cli.seed, 7);
    }

    #[test]
    fn cli_rejects_missing_invalid_and_unknown_values() {
        assert!(Cli::from_args(["--workers"]).is_err());
        assert!(Cli::from_args(["--workers", "many"]).is_err());
        assert!(Cli::from_args(["--max-eval-fac", "0"]).is_err());
        assert!(Cli::from_args(["--progress-interval", "-1"]).is_err());
        assert!(Cli::from_args(["--stop-fitness", "NaN"]).is_err());
        assert!(Cli::from_args(["--unknown", "1"]).is_err());
    }

    #[test]
    fn progress_state_counts_evaluations_retries_and_atomic_minimum() {
        let state = ProgressState::new("test", 2);
        for value in [5.0, f64::NAN, 3.0, 4.0, -1.0] {
            state.record_evaluation(value);
        }
        state.retry_completed();
        assert_eq!(state.evaluations.load(Ordering::Relaxed), 5);
        assert_eq!(state.completed_retries.load(Ordering::Relaxed), 1);
        assert_eq!(
            f64::from_bits(state.best_bits.load(Ordering::Relaxed)),
            -1.0
        );
    }
}
