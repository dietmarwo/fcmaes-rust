//! Buckingham–Pi enumeration and data-driven continuous exponent search.

use std::env;
use std::error::Error;
use std::time::Instant;

use fcmaes_core::{
    BiteParams, Fitness, Mode, ModeParams, RetryBounds, RetryConfig, RetryRunResult, Rng,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use fcmaes_examples::buckingham::{
    BuckinghamModel, CandidateMetrics, PreparedProblem, catalog, format_pi_group, prepare_problem,
    problem_by_slug,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Enumerate,
    Rank,
    Optimize,
    Multi,
    All,
}

impl RunMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "enumerate" => Ok(Self::Enumerate),
            "rank" => Ok(Self::Rank),
            "optimize" | "single" => Ok(Self::Optimize),
            "multi" | "mode" => Ok(Self::Multi),
            "all" => Ok(Self::All),
            _ => Err("--mode must be enumerate, rank, optimize, multi, or all".to_owned()),
        }
    }

    fn includes_enumeration(self) -> bool {
        matches!(self, Self::Enumerate | Self::All)
    }

    fn includes_ranking(self) -> bool {
        matches!(self, Self::Rank | Self::All)
    }

    fn includes_scalar(self) -> bool {
        matches!(self, Self::Optimize | Self::All)
    }

    fn includes_multi(self) -> bool {
        matches!(self, Self::Multi | Self::All)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Enumerate => "enumerate",
            Self::Rank => "rank",
            Self::Optimize => "optimize",
            Self::Multi => "multi",
            Self::All => "all",
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    problem: String,
    mode: RunMode,
    groups: usize,
    samples: usize,
    workers: usize,
    retries: usize,
    evaluations: u64,
    mo_evaluations: usize,
    popsize: usize,
    rank_limit: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            problem: "cylinder".to_owned(),
            mode: RunMode::All,
            groups: 0,
            samples: 300,
            workers: 16,
            retries: 32,
            evaluations: 2_000,
            mo_evaluations: 20_000,
            popsize: 128,
            rank_limit: 10,
            seed: 42,
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
                "--problem" => parsed.problem = next_value(&mut arguments, "--problem")?,
                "--mode" => parsed.mode = RunMode::parse(&next_value(&mut arguments, "--mode")?)?,
                "--groups" => parsed.groups = parse_value(&mut arguments, "--groups")?,
                "--samples" => parsed.samples = parse_value(&mut arguments, "--samples")?,
                "--workers" => parsed.workers = parse_value(&mut arguments, "--workers")?,
                "--retries" => parsed.retries = parse_value(&mut arguments, "--retries")?,
                "--evaluations" => {
                    parsed.evaluations = parse_value(&mut arguments, "--evaluations")?
                }
                "--mo-evaluations" => {
                    parsed.mo_evaluations = parse_value(&mut arguments, "--mo-evaluations")?
                }
                "--popsize" => parsed.popsize = parse_value(&mut arguments, "--popsize")?,
                "--rank-limit" => parsed.rank_limit = parse_value(&mut arguments, "--rank-limit")?,
                "--seed" => parsed.seed = parse_value(&mut arguments, "--seed")?,
                "--list-problems" => {
                    print_problem_list();
                    std::process::exit(0);
                }
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
        if self.samples < 8 {
            return Err("--samples must be at least eight".to_owned());
        }
        if self.retries == 0 || self.evaluations == 0 || self.mo_evaluations == 0 {
            return Err("retry and evaluation counts must be positive".to_owned());
        }
        if self.popsize < 4 {
            return Err("--popsize must be at least four".to_owned());
        }
        if self.rank_limit == 0 {
            return Err("--rank-limit must be positive".to_owned());
        }
        if problem_by_slug(&self.problem).is_none() {
            return Err(format!(
                "unknown problem '{}'; use --list-problems",
                self.problem
            ));
        }
        Ok(())
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

fn print_problem_list() {
    for problem in catalog() {
        println!("{}\t{}", problem.slug, problem.name);
    }
}

fn print_help() {
    println!(
        "Native Buckingham-Pi analysis and optimization\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin buckingham-pi -- [OPTIONS]\n\
         \n  --problem NAME          Catalog slug (cylinder)\n\
         \n  --list-problems         Print available problem slugs\n\
         \n  --mode NAME             enumerate, rank, optimize, multi, or all (all)\n\
         \n  --groups N              Continuous pi groups; 0 uses full nullity (0)\n\
         \n  --samples N             Samples in each train/holdout split (300)\n\
         \n  --workers N             Retry/batch workers; 0 uses available CPUs (16)\n\
         \n  --retries N             Independent BiteOpt retries (32)\n\
         \n  --evaluations N         Evaluations per BiteOpt retry (2000)\n\
         \n  --mo-evaluations N      Requested MODE evaluation budget (20000)\n\
         \n  --popsize N             MODE population size (128)\n\
         \n  --rank-limit N          Repeating sets printed by rank mode (10)\n\
         \n  --seed N                Data and optimizer root seed (42)"
    );
}

fn print_problem(problem: &PreparedProblem) {
    println!(
        "PROBLEM slug={} name={:?} dependent={} variables={} dimensions={} rank={} nullity={} removed_dimensionless={:?}",
        problem.slug,
        problem.name,
        problem.dependent,
        problem.variables.len(),
        problem.dimensions.len(),
        problem.rank(),
        problem.nullity(),
        problem.removed_dimensionless()
    );
    println!("VARIABLES {:?}", problem.variables);
}

fn run_enumeration(problem: &PreparedProblem) -> Result<(), Box<dyn Error>> {
    let sets = problem.repeating_sets();
    let unique = problem.unique_pi_groups()?;
    println!(
        "ENUMERATION repeating_sets={} unique_groups={}",
        sets.len(),
        unique.len()
    );
    for (set_index, repeating) in sets.iter().enumerate() {
        let names: Vec<&str> = repeating
            .iter()
            .map(|&index| problem.variables[index])
            .collect();
        let groups = problem.pi_groups(repeating)?;
        println!(
            "REPEATING rank={} variables={:?} groups={}",
            set_index + 1,
            names,
            groups.len()
        );
        for (group_index, group) in groups.iter().enumerate() {
            println!(
                "PI set={} group={} non_repeating={} expression={}",
                set_index + 1,
                group_index + 1,
                problem.variables[group.non_repeating],
                format_pi_group(&problem.variables, &group.exponents)
            );
        }
    }
    for (index, group) in unique.iter().enumerate() {
        println!(
            "UNIQUE_PI index={} expression={}",
            index + 1,
            format_pi_group(&problem.variables, &group.exponents)
        );
    }
    Ok(())
}

fn run_ranking(model: &BuckinghamModel, limit: usize) -> Result<(), Box<dyn Error>> {
    let ranked = model.rank_repeating_sets()?;
    println!(
        "RANKING repeating_sets={} shown={} criterion=holdout_r2",
        ranked.len(),
        ranked.len().min(limit)
    );
    for (rank, entry) in ranked.iter().take(limit).enumerate() {
        let names: Vec<&str> = entry
            .repeating
            .iter()
            .map(|&index| model.problem().variables[index])
            .collect();
        println!(
            "RANK rank={} repeating={:?} validation_r2={:.9} train_r2={:.9} mean_cv={:.6} complexity={:.6} condition_ratio={:.6}",
            rank + 1,
            names,
            entry.metrics.validation_r2,
            entry.metrics.train_r2,
            entry.metrics.mean_coefficient_of_variation,
            entry.metrics.complexity,
            entry.metrics.condition_ratio
        );
        print_groups(model.problem(), &entry.metrics);
    }
    Ok(())
}

fn run_scalar(model: &BuckinghamModel, args: &Args) -> Result<(), Box<dyn Error>> {
    let dimension = model.decision_dimension();
    let bounds = RetryBounds::new(vec![-9.0; dimension], vec![9.0; dimension])?;
    let config = RetryConfig {
        num_retries: args.retries,
        workers: args.workers,
        capacity: args.retries.min(500),
        max_evaluations: args.evaluations,
        seed: args.seed ^ 0x243F_6A88_85A3_08D3,
        ..Default::default()
    };
    let objective = |x: &[f64]| model.evaluate(x).scalar_objective;
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
    let metrics = model.evaluate(&result.x);
    if !metrics.valid {
        return Err("BiteOpt retry returned an invalid candidate".into());
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "SO objective={:.9} validation_r2={:.9} train_r2={:.9} mean_cv={:.6} complexity={:.6} dependence={:.6} condition_ratio={:.6} dimensional_residual={:.3e} evaluations={} completed_retries={} seconds={:.6} evaluations_per_second={:.0}",
        metrics.scalar_objective,
        metrics.validation_r2,
        metrics.train_r2,
        metrics.mean_coefficient_of_variation,
        metrics.complexity,
        metrics.dependence,
        metrics.condition_ratio,
        metrics.dimensional_residual,
        result.evaluations,
        result.runs,
        seconds,
        result.evaluations as f64 / seconds.max(1.0e-9)
    );
    print_groups(model.problem(), &metrics);
    Ok(())
}

fn run_multi(model: &BuckinghamModel, args: &Args) -> Result<(), Box<dyn Error>> {
    let dimension = model.decision_dimension();
    let lower = vec![-9.0; dimension];
    let upper = vec![9.0; dimension];
    let fitness = Fitness::bounded(dimension, 4, &lower, &upper);
    let params = ModeParams {
        popsize: args.popsize as i32,
        nsga_update: true,
        seed: args.seed ^ 0x1319_8A2E_0370_7344,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, 3, 1, None, &params)?;
    let generations = args.mo_evaluations.div_ceil(args.popsize);
    let started = Instant::now();
    for _ in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, args.workers as i32, |x| {
            model.evaluate(x).mode_values()
        });
        mode.tell(&ys);
    }

    let population = mode.population();
    let metrics: Vec<CandidateMetrics> = population
        .iter()
        .map(|candidate| model.evaluate(candidate))
        .collect();
    let feasible_indices: Vec<usize> = metrics
        .iter()
        .enumerate()
        .filter(|(_, item)| item.valid && item.cv_violation <= 1.0e-12)
        .map(|(index, _)| index)
        .collect();
    let values: Vec<Vec<f64>> = feasible_indices
        .iter()
        .map(|&index| metrics[index].mode_values()[..3].to_vec())
        .collect();
    let front_local = pareto_indices(&values, 3)?;
    let front: Vec<usize> = front_local
        .iter()
        .map(|&index| feasible_indices[index])
        .collect();
    let best_validation_r2 = front
        .iter()
        .map(|&index| metrics[index].validation_r2)
        .fold(f64::NEG_INFINITY, f64::max);
    let actual_evaluations = generations * args.popsize;
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "MO pareto={} feasible={} best_validation_r2={:.9} evaluations={} generations={} seconds={:.6} evaluations_per_second={:.0}",
        front.len(),
        feasible_indices.len(),
        best_validation_r2,
        actual_evaluations,
        generations,
        seconds,
        actual_evaluations as f64 / seconds.max(1.0e-9)
    );
    for (rank, &index) in front.iter().take(12).enumerate() {
        let item = &metrics[index];
        println!(
            "MO_POINT rank={} validation_r2={:.9} train_r2={:.9} complexity={:.6} dependence={:.6} mean_cv={:.6} dimensional_residual={:.3e}",
            rank + 1,
            item.validation_r2,
            item.train_r2,
            item.complexity,
            item.dependence,
            item.mean_coefficient_of_variation,
            item.dimensional_residual
        );
        print_groups(model.problem(), item);
    }
    Ok(())
}

fn print_groups(problem: &PreparedProblem, metrics: &CandidateMetrics) {
    for column in 0..metrics.exponents.ncols() {
        let exponents: Vec<f64> = metrics.exponents.column(column).iter().copied().collect();
        println!(
            "GROUP index={} expression={}",
            column + 1,
            format_pi_group(&problem.variables, &exponents)
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let raw_problem = problem_by_slug(&args.problem)
        .ok_or_else(|| format!("unknown problem {}", args.problem))?;
    let problem = prepare_problem(raw_problem)?;
    print_problem(&problem);
    if problem.nullity() == 0 {
        return Err("problem has no dimensionless group after preprocessing".into());
    }
    let groups = if args.groups == 0 {
        problem.nullity()
    } else {
        args.groups
    };
    if groups > problem.nullity() {
        return Err(format!(
            "--groups must not exceed the problem nullity ({})",
            problem.nullity()
        )
        .into());
    }
    println!(
        "CONFIG language=rust mode={} groups={} samples_per_split={} workers={} retries={} evaluations_per_retry={} mo_evaluations={} popsize={} seed={}",
        args.mode.name(),
        groups,
        args.samples,
        args.workers,
        args.retries,
        args.evaluations,
        args.mo_evaluations,
        args.popsize,
        args.seed
    );

    if args.mode.includes_enumeration() {
        run_enumeration(&problem)?;
    }
    if args.mode.includes_ranking() || args.mode.includes_scalar() || args.mode.includes_multi() {
        let model = BuckinghamModel::new(problem, groups, args.samples, args.seed)?;
        println!(
            "DATA synthetic=true train={} holdout={} log10_range=[-3,3] relative_noise=0.02 decision_dimension={}",
            model.samples_per_split(),
            model.samples_per_split(),
            model.decision_dimension()
        );
        if args.mode.includes_ranking() {
            run_ranking(&model, args.rank_limit)?;
        }
        if args.mode.includes_scalar() {
            run_scalar(&model, &args)?;
        }
        if args.mode.includes_multi() {
            run_multi(&model, &args)?;
        }
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
    fn defaults_match_the_documented_comparison() {
        let args = Args::default();
        assert_eq!(args.problem, "cylinder");
        assert_eq!(args.mode, RunMode::All);
        assert_eq!(
            (args.workers, args.retries, args.evaluations),
            (16, 32, 2_000)
        );
    }

    #[test]
    fn parses_all_controls() {
        let args = Args::from_args(arguments(&[
            "--problem",
            "pipe",
            "--mode",
            "mode",
            "--groups",
            "1",
            "--samples",
            "64",
            "--workers",
            "4",
            "--retries",
            "8",
            "--evaluations",
            "100",
            "--mo-evaluations",
            "256",
            "--popsize",
            "32",
            "--rank-limit",
            "3",
            "--seed",
            "7",
        ]))
        .unwrap();
        assert_eq!(args.problem, "pipe");
        assert_eq!(args.mode, RunMode::Multi);
        assert_eq!(args.groups, 1);
        assert_eq!((args.workers, args.retries), (4, 8));
        assert_eq!((args.mo_evaluations, args.popsize), (256, 32));
    }

    #[test]
    fn rejects_unknown_problem_and_invalid_counts() {
        assert!(Args::from_args(arguments(&["--problem", "missing"])).is_err());
        assert!(Args::from_args(arguments(&["--samples", "4"])).is_err());
        assert!(Args::from_args(arguments(&["--retries", "0"])).is_err());
        assert!(Args::from_args(arguments(&["--popsize", "3"])).is_err());
    }

    #[test]
    fn smoke_runs_enumeration_ranking_and_optimizers() {
        let problem = prepare_problem(problem_by_slug("pipe").unwrap()).unwrap();
        run_enumeration(&problem).unwrap();
        let model = BuckinghamModel::new(problem, 1, 32, 3).unwrap();
        run_ranking(&model, 2).unwrap();
        let args = Args {
            problem: "pipe".to_owned(),
            mode: RunMode::All,
            groups: 1,
            samples: 32,
            workers: 1,
            retries: 1,
            evaluations: 32,
            mo_evaluations: 16,
            popsize: 8,
            rank_limit: 2,
            seed: 3,
        };
        run_scalar(&model, &args).unwrap();
        run_multi(&model, &args).unwrap();
    }
}
