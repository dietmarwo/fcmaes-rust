//! Command-line entry point: `list`, `measure` and `run`.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use candidate_harness::campaign::{
    BudgetRow, CampaignConfig, Protocol, budget_for, measure_problem, read_budgets, run_campaign,
    write_budgets,
};
use candidate_harness::{problems, solvers};

const USAGE: &str = "usage:
  candidate-harness list
  candidate-harness measure --out results/budgets.tsv
  candidate-harness run --budgets results/budgets.tsv --out results/raw.tsv
      [--problems a,b] [--solvers x,y] [--protocols S,R] [--reps 20]
      [--threads 2] [--budget-scale 1.0]";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("measure") => measure(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_owned()),
    };
    if let Err(message) = result {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn options(args: &[String], allowed: &[&str]) -> Result<Vec<(String, String)>, String> {
    let mut parsed = Vec::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        if !allowed.contains(&flag.as_str()) {
            return Err(format!("unknown option '{flag}'\n{USAGE}"));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        parsed.push((flag.clone(), value.clone()));
    }
    Ok(parsed)
}

fn option<'a>(parsed: &'a [(String, String)], flag: &str) -> Option<&'a str> {
    parsed
        .iter()
        .rev()
        .find(|(name, _)| name == flag)
        .map(|(_, value)| value.as_str())
}

fn list() -> Result<(), String> {
    println!("problems:");
    for problem in problems::all() {
        println!(
            "  {:<15} {:<10} D={}",
            problem.name,
            problem.domain,
            problem.dim()
        );
    }
    println!("solvers:");
    for solver in solvers::registry() {
        let role = if solvers::INCUMBENTS.contains(&solver.name()) {
            "incumbent"
        } else {
            "candidate"
        };
        println!("  {:<15} {role}", solver.name());
    }
    Ok(())
}

fn measure(args: &[String]) -> Result<(), String> {
    let parsed = options(args, &["--out"])?;
    let out = PathBuf::from(option(&parsed, "--out").ok_or("measure needs --out")?);
    // Timings decide the budget classes; other load on the host inflates them.
    if let Ok(load) = std::fs::read_to_string("/proc/loadavg") {
        println!("load average before measuring: {}", load.trim());
    }
    let mut rows = Vec::new();
    for problem in problems::all() {
        let mean = measure_problem(&problem);
        let budget = budget_for(mean);
        println!(
            "{:<15} D={:<4} mean {:>10.3} us  B={budget}",
            problem.name,
            problem.dim(),
            mean
        );
        rows.push(BudgetRow {
            problem: problem.name.to_owned(),
            domain: problem.domain.to_owned(),
            dim: problem.dim(),
            mean_eval_us: mean,
            budget,
        });
    }
    write_budgets(&out, &rows)?;
    println!("wrote {}", out.display());
    Ok(())
}

fn split_list(value: Option<&str>) -> Option<Vec<String>> {
    value.map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    })
}

fn run(args: &[String]) -> Result<(), String> {
    let parsed = options(
        args,
        &[
            "--budgets",
            "--out",
            "--problems",
            "--solvers",
            "--protocols",
            "--reps",
            "--threads",
            "--budget-scale",
        ],
    )?;
    let budgets_path = PathBuf::from(option(&parsed, "--budgets").ok_or("run needs --budgets")?);
    let out = PathBuf::from(option(&parsed, "--out").ok_or("run needs --out")?);
    let repetitions: u32 = option(&parsed, "--reps")
        .unwrap_or("20")
        .parse()
        .map_err(|_| "--reps needs a positive integer")?;
    let threads: usize = option(&parsed, "--threads")
        .unwrap_or("2")
        .parse()
        .map_err(|_| "--threads needs a positive integer")?;
    let budget_scale: f64 = option(&parsed, "--budget-scale")
        .unwrap_or("1.0")
        .parse()
        .map_err(|_| "--budget-scale needs a number")?;
    if repetitions == 0 || threads == 0 {
        return Err("--reps and --threads must be positive".into());
    }
    if !(budget_scale > 0.0 && budget_scale <= 1.0) {
        return Err("--budget-scale must lie in (0, 1]".into());
    }

    let budgets = read_budgets(&budgets_path)?;
    let all_problems = problems::all();
    let problem_names = split_list(option(&parsed, "--problems"))
        .unwrap_or_else(|| all_problems.iter().map(|p| p.name.to_owned()).collect());
    let mut selected_problems = Vec::new();
    for name in &problem_names {
        let problem = all_problems
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| format!("unknown problem '{name}'"))?;
        let row = budgets
            .iter()
            .find(|row| row.problem == *name)
            .ok_or_else(|| format!("{}: no budget for '{name}'", budgets_path.display()))?;
        if row.dim != problem.dim() {
            return Err(format!("{name}: budgets file has D={}", row.dim));
        }
        selected_problems.push((problem, row.budget));
    }

    let registry = solvers::registry();
    let solver_names = split_list(option(&parsed, "--solvers"))
        .unwrap_or_else(|| registry.iter().map(|s| s.name().to_owned()).collect());
    let mut selected_solvers = Vec::new();
    for name in &solver_names {
        let solver = registry
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| format!("unknown solver '{name}' (see `list`)"))?;
        selected_solvers.push(solver.as_ref());
    }

    let protocols = split_list(option(&parsed, "--protocols"))
        .unwrap_or_else(|| vec!["S".into(), "R".into()])
        .iter()
        .map(|p| Protocol::parse(p))
        .collect::<Result<Vec<_>, _>>()?;

    let config = CampaignConfig {
        problems: selected_problems,
        solvers: selected_solvers,
        protocols,
        repetitions,
        threads,
        budget_scale,
    };
    let written = run_campaign(&config, &out)?;
    println!("wrote {written} rows to {}", out.display());
    Ok(())
}
