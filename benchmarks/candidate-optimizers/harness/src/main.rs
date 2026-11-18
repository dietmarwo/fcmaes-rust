//! Command-line entry point: `list`, `measure` and `run`.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use candidate_harness::campaign::{
    BudgetRow, CampaignConfig, Protocol, T_MIN_RUN, T_RUN_CAP, T_TOTAL, TSettings, budget_for,
    is_deterministic, measure_problem, read_budgets, run_campaign, run_campaign_t, write_budgets,
};
use candidate_harness::problems::Problem;
use candidate_harness::seeds::ROOT_SEED;
use candidate_harness::{problems, solvers};

const USAGE: &str = "usage:
  candidate-harness list
  candidate-harness measure --out results/budgets.tsv [--problems a,b|round2]
  candidate-harness run --budgets results/budgets.tsv --out results/raw.tsv
      [--problems a,b] [--solvers x,y] [--protocols S,R] [--reps 20]
      [--threads 2] [--budget-scale 1.0]
  candidate-harness run --protocols T --out results/round2/raw.tsv
      [--problems round2|a,b] [--solvers round2|x,y] [--reps 20] [--threads 2]
      [--root-seed 1] [--total 400000] [--run-cap 100000] [--min-run 1000]

  --problems round2 and --solvers round2 select the round-2 sets
  (ROUND2.md); they are the defaults for protocol T. Round 2 uses
  --root-seed 2.";

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
    for problem in problems::catalog() {
        println!(
            "  {:<15} {:<10} D={}",
            problem.name,
            problem.domain,
            problem.dim()
        );
    }
    println!("round-2 problems: {}", problems::round2_names().join(","));
    println!("solvers:");
    for solver in solvers::registry() {
        let role = if solvers::INCUMBENTS.contains(&solver.name()) {
            "incumbent"
        } else {
            "candidate"
        };
        println!("  {:<15} {role}", solver.name());
    }
    println!(
        "round-2 solvers: {}",
        solvers::round2_solver_names().join(",")
    );
    Ok(())
}

/// Problems named by `--problems` (`round2` expands to the round-2 set;
/// default: the thirteen round-1 problems).
fn select_problems(value: Option<&str>) -> Result<Vec<Problem>, String> {
    let Some(names) = split_list(value) else {
        return Ok(problems::all());
    };
    let mut expanded = Vec::new();
    for name in names {
        if name == "round2" {
            expanded.extend(problems::round2_names().into_iter().map(str::to_owned));
        } else {
            expanded.push(name);
        }
    }
    expanded
        .iter()
        .map(|name| problems::by_name(name).ok_or_else(|| format!("unknown problem '{name}'")))
        .collect()
}

fn measure(args: &[String]) -> Result<(), String> {
    let parsed = options(args, &["--out", "--problems"])?;
    let out = PathBuf::from(option(&parsed, "--out").ok_or("measure needs --out")?);
    let selected = select_problems(option(&parsed, "--problems"))?;
    // Timings decide the budget classes; other load on the host inflates them.
    if let Ok(load) = std::fs::read_to_string("/proc/loadavg") {
        println!("load average before measuring: {}", load.trim());
    }
    let mut rows = Vec::new();
    for problem in selected {
        let mean = measure_problem(&problem);
        let budget = budget_for(mean);
        let deterministic = is_deterministic(&problem);
        println!(
            "{:<15} D={:<4} mean {:>10.3} us  B={budget}  deterministic={deterministic}",
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
            "--root-seed",
            "--total",
            "--run-cap",
            "--min-run",
        ],
    )?;
    let protocols = split_list(option(&parsed, "--protocols"))
        .unwrap_or_else(|| vec!["S".into(), "R".into()])
        .iter()
        .map(|p| Protocol::parse(p))
        .collect::<Result<Vec<_>, _>>()?;
    if protocols.contains(&Protocol::T) {
        return run_t(&parsed, protocols);
    }
    for flag in ["--root-seed", "--total", "--run-cap", "--min-run"] {
        if option(&parsed, flag).is_some() {
            return Err(format!("{flag} applies to protocol T only"));
        }
    }
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
    if problem_names.iter().any(|name| name == "round2") {
        return Err("--problems round2 belongs to protocol T".into());
    }
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

fn parse_u64(parsed: &[(String, String)], flag: &str, default: u64) -> Result<u64, String> {
    option(parsed, flag).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("{flag} needs a non-negative integer"))
    })
}

/// `run --protocols T`: the round-2 restart protocol.
fn run_t(parsed: &[(String, String)], protocols: Vec<Protocol>) -> Result<(), String> {
    if protocols != [Protocol::T] {
        return Err("protocol T cannot be combined with S or R (separate file format)".into());
    }
    for flag in ["--budgets", "--budget-scale"] {
        if option(parsed, flag).is_some() {
            return Err(format!("{flag} does not apply to protocol T"));
        }
    }
    let out = PathBuf::from(option(parsed, "--out").ok_or("run needs --out")?);
    let repetitions: u32 = option(parsed, "--reps")
        .unwrap_or("20")
        .parse()
        .map_err(|_| "--reps needs a positive integer")?;
    let threads: usize = option(parsed, "--threads")
        .unwrap_or("2")
        .parse()
        .map_err(|_| "--threads needs a positive integer")?;
    if repetitions == 0 || threads == 0 {
        return Err("--reps and --threads must be positive".into());
    }
    let settings = TSettings {
        root_seed: parse_u64(parsed, "--root-seed", ROOT_SEED)?,
        total: parse_u64(parsed, "--total", T_TOTAL)?,
        run_cap: parse_u64(parsed, "--run-cap", T_RUN_CAP)?,
        min_run: parse_u64(parsed, "--min-run", T_MIN_RUN)?,
    };
    if settings.total == 0 || settings.run_cap == 0 || settings.min_run == 0 {
        return Err("--total, --run-cap and --min-run must be positive".into());
    }
    let selected_problems =
        select_problems(Some(option(parsed, "--problems").unwrap_or("round2")))?;

    let registry = solvers::registry();
    let solver_names: Vec<String> = match split_list(option(parsed, "--solvers")) {
        None => vec!["round2".to_owned()],
        Some(names) => names,
    }
    .into_iter()
    .flat_map(|name| {
        if name == "round2" {
            solvers::round2_solver_names()
                .into_iter()
                .map(str::to_owned)
                .collect()
        } else {
            vec![name]
        }
    })
    .collect();
    let mut selected_solvers = Vec::new();
    for name in &solver_names {
        let solver = registry
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| format!("unknown solver '{name}' (see `list`)"))?;
        selected_solvers.push(solver.as_ref());
    }

    println!(
        "protocol T: root seed {}, total {}, run cap {}, min run {}{}",
        settings.root_seed,
        settings.total,
        settings.run_cap,
        settings.min_run,
        if settings.is_declared() {
            ""
        } else {
            " (not the declared round-2 settings)"
        }
    );
    let config = CampaignConfig {
        problems: selected_problems
            .iter()
            .map(|problem| (problem, settings.total))
            .collect(),
        solvers: selected_solvers,
        protocols,
        repetitions,
        threads,
        budget_scale: 1.0,
    };
    let written = run_campaign_t(&config, &settings, &out)?;
    println!("wrote {written} rows to {}", out.display());
    Ok(())
}
