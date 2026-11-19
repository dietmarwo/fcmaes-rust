//! Command-line entry point: `list`, `measure`, `run`, `survey`,
//! `survey3b`, `serve` and `seeds`.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use candidate_harness::campaign::{
    BudgetRow, CampaignConfig, Protocol, T_MIN_RUN, T_RUN_CAP, T_TOTAL, TSettings, budget_for,
    is_deterministic, measure_problem, read_budgets, run_campaign, run_campaign_t,
    run_campaign_t_totals, t_run_seed, write_budgets,
};
use candidate_harness::problems::Problem;
use candidate_harness::seeds::ROOT_SEED;
use candidate_harness::{problems, round3, round3b, serve, solvers};

const USAGE: &str = "usage:
  candidate-harness list
  candidate-harness measure --out results/budgets.tsv [--problems a,b|round2]
  candidate-harness run --budgets results/budgets.tsv --out results/raw.tsv
      [--problems a,b] [--solvers x,y] [--protocols S,R] [--reps 20]
      [--threads 2] [--budget-scale 1.0]
  candidate-harness survey --out results/round3 [--problems a,b]
  candidate-harness survey3b --out results/round3b
  candidate-harness run --protocols T --out results/round2/raw.tsv
      [--problems round2|round3|round3b|a,b] [--solvers round2|round3|x,y] [--reps 20]
      [--threads 2] [--root-seed 1] [--total 400000] [--run-cap 100000]
      [--min-run 1000] [--totals 2000,20000]
  candidate-harness serve
  candidate-harness seeds [--root-seed 3] [--problems round3|round3b|a,b]
      [--totals 2000,20000] [--reps 3] [--restarts 4]

  --problems round2 and --solvers round2 select the round-2 sets
  (ROUND2.md); they are the defaults for protocol T. Round 2 uses
  --root-seed 2.
  --problems round3 and --solvers round3 select the round-3 sets
  (ROUND3.md, DEVIATIONS.md round 3). --totals runs one set of cells per
  total, each with run cap equal to the total, into one file (round 3:
  --totals 2000,20000 --root-seed 3 --min-run 100); it excludes --total and
  --run-cap.
  --problems round3b selects the second round-3 batch (ROUND3B.md); it
  runs with --solvers round3 and the round-3 settings.
  survey examines the round-3 problems (ROUND3.md sections 3 and 4) and
  writes problems.tsv and types/<problem>.json; survey3b does the same for
  the round-3b batch (ROUND3B.md section 2).
  serve answers objective requests on stdin (protocol in src/serve.rs) for
  run_catcmawm.py.
  seeds prints the protocol-T run seeds of a grid (problem, total,
  repetition, restart), tab-separated.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("measure") => measure(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("survey") => survey(&args[1..]),
        Some("survey3b") => survey3b(&args[1..]),
        Some("serve") => {
            if args.len() > 1 {
                Err("serve takes no options".to_owned())
            } else {
                let stdin = std::io::stdin();
                serve::serve(stdin.lock(), std::io::stdout().lock())
            }
        }
        Some("seeds") => seeds(&args[1..]),
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
    println!("round-3 problems: {}", round3::PROBLEMS.join(","));
    println!(
        "round-3 solvers: {} (candidate catcmawm runs through run_catcmawm.py)",
        round3::SOLVERS.join(",")
    );
    println!("round-3b problems: {}", round3b::PROBLEMS.join(","));
    Ok(())
}

/// Problems named by `--problems` (`round2`, `round3` and `round3b` expand
/// to the sets of those rounds; default: the thirteen round-1 problems).
fn select_problems(value: Option<&str>) -> Result<Vec<Problem>, String> {
    let Some(names) = split_list(value) else {
        return Ok(problems::all());
    };
    let mut expanded = Vec::new();
    for name in names {
        if name == "round2" {
            expanded.extend(problems::round2_names().into_iter().map(str::to_owned));
        } else if name == "round3" {
            expanded.extend(round3::PROBLEMS.iter().map(|s| (*s).to_owned()));
        } else if name == "round3b" {
            expanded.extend(round3b::PROBLEMS.iter().map(|s| (*s).to_owned()));
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
            "--totals",
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
    for flag in [
        "--root-seed",
        "--total",
        "--run-cap",
        "--min-run",
        "--totals",
    ] {
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
    if problem_names
        .iter()
        .any(|name| name == "round2" || name == "round3" || name == "round3b")
    {
        return Err("--problems round2, round3 and round3b belong to protocol T".into());
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
    let root_seed = parse_u64(parsed, "--root-seed", ROOT_SEED)?;
    let min_run = parse_u64(parsed, "--min-run", T_MIN_RUN)?;
    let totals = parse_totals(option(parsed, "--totals"))?;
    if totals.is_some() {
        for flag in ["--total", "--run-cap"] {
            if option(parsed, flag).is_some() {
                return Err(format!(
                    "{flag} does not combine with --totals (run cap = total)"
                ));
            }
        }
    }
    let all_settings: Vec<TSettings> = match &totals {
        Some(totals) => totals
            .iter()
            .map(|&total| round3::settings_for(root_seed, total, min_run))
            .collect(),
        None => vec![TSettings {
            root_seed,
            total: parse_u64(parsed, "--total", T_TOTAL)?,
            run_cap: parse_u64(parsed, "--run-cap", T_RUN_CAP)?,
            min_run,
        }],
    };
    for settings in &all_settings {
        if settings.total == 0 || settings.run_cap == 0 || settings.min_run == 0 {
            return Err("--total(s), --run-cap and --min-run must be positive".into());
        }
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
        } else if name == "round3" {
            round3::SOLVERS.iter().map(|s| (*s).to_owned()).collect()
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
    // ROUND3.md section 2: `de` and `de_cma` would need the problem's own DE
    // integer mask. No round-3 problem passes one (round3::MASK_SOURCES); the
    // runner has no mask wiring and refuses a problem that has one.
    for problem in &selected_problems {
        if round3::de_mask(problem.name).is_some()
            && solver_names.iter().any(|s| s == "de" || s == "de_cma")
        {
            return Err(format!(
                "{}: its own DE call passes an integer mask, which `de` and `de_cma` \
                 would have to use; the runner does not wire masks (no round-3 problem \
                 has one)",
                problem.name
            ));
        }
    }

    let config = CampaignConfig {
        problems: selected_problems
            .iter()
            .map(|problem| (problem, all_settings[0].total))
            .collect(),
        solvers: selected_solvers,
        protocols,
        repetitions,
        threads,
        budget_scale: 1.0,
    };
    let written = if totals.is_some() {
        let declared = all_settings == round3::declared_settings();
        println!(
            "protocol T, totals {}: root seed {root_seed}, run cap = total, min run {min_run}{}",
            all_settings
                .iter()
                .map(|s| s.total.to_string())
                .collect::<Vec<_>>()
                .join(","),
            if declared {
                " (the declared round-3 settings)"
            } else {
                " (not the declared round-3 settings)"
            }
        );
        run_campaign_t_totals(&config, &all_settings, &out)?
    } else {
        let settings = all_settings[0];
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
        run_campaign_t(&config, &settings, &out)?
    };
    println!("wrote {written} rows to {}", out.display());
    Ok(())
}

/// `--totals a,b`: distinct positive totals.
fn parse_totals(value: Option<&str>) -> Result<Option<Vec<u64>>, String> {
    let Some(list) = split_list(value) else {
        return Ok(None);
    };
    let totals = list
        .iter()
        .map(|t| {
            t.parse::<u64>()
                .ok()
                .filter(|t| *t > 0)
                .ok_or_else(|| format!("--totals: bad total '{t}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if totals.is_empty() {
        return Err("--totals needs at least one total".into());
    }
    let mut distinct = totals.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != totals.len() {
        return Err("--totals must be distinct".into());
    }
    Ok(Some(totals))
}

/// `seeds`: protocol-T run seeds of a grid, as the runner derives them.
fn seeds(args: &[String]) -> Result<(), String> {
    let parsed = options(
        args,
        &[
            "--root-seed",
            "--problems",
            "--totals",
            "--reps",
            "--restarts",
        ],
    )?;
    let root_seed = parse_u64(&parsed, "--root-seed", round3::ROOT_SEED)?;
    let names: Vec<String> = split_list(Some(option(&parsed, "--problems").unwrap_or("round3")))
        .unwrap_or_default()
        .into_iter()
        .flat_map(|name| {
            if name == "round3" {
                round3::PROBLEMS.iter().map(|s| (*s).to_owned()).collect()
            } else if name == "round3b" {
                round3b::PROBLEMS.iter().map(|s| (*s).to_owned()).collect()
            } else {
                vec![name]
            }
        })
        .collect();
    let totals = parse_totals(option(&parsed, "--totals"))?.unwrap_or(round3::TOTALS.to_vec());
    let reps = parse_u64(&parsed, "--reps", 3)?;
    let restarts = parse_u64(&parsed, "--restarts", 4)?;
    let reps = u32::try_from(reps).map_err(|_| "--reps too large")?;
    let restarts = u32::try_from(restarts).map_err(|_| "--restarts too large")?;
    println!("problem\ttotal\trep\trestart\tseed");
    for name in &names {
        for &total in &totals {
            let settings = round3::settings_for(root_seed, total, round3::MIN_RUN);
            for rep in 0..reps {
                for restart in 0..restarts {
                    println!(
                        "{name}\t{total}\t{rep}\t{restart}\t{}",
                        t_run_seed(&settings, name, rep, restart)
                    );
                }
            }
        }
    }
    Ok(())
}

/// `survey`: the round-3 problem survey (ROUND3.md sections 3 and 4).
fn survey(args: &[String]) -> Result<(), String> {
    use candidate_harness::round3::{
        self, ENTRY_LIMIT_US, EXAMINED, Kind, NOT_BUILT, deterministic_on, measure_capped,
        type_table,
    };
    let parsed = options(args, &["--out", "--problems"])?;
    let out = PathBuf::from(option(&parsed, "--out").ok_or("survey needs --out")?);
    let names: Vec<String> = split_list(option(&parsed, "--problems"))
        .unwrap_or_else(|| EXAMINED.iter().map(|s| (*s).to_owned()).collect());
    let types_dir = out.join("types");
    std::fs::create_dir_all(&types_dir).map_err(|e| e.to_string())?;
    if let Ok(load) = std::fs::read_to_string("/proc/loadavg") {
        println!("load average before measuring: {}", load.trim());
    }
    let mut tsv = String::from(
        "problem\tdomain\tdim\tcontinuous\tinteger\tcategorical\tmean_eval_us\ttimed_points\t\
         deterministic\tde_mask\tentered\treason\n",
    );
    let mut entered = Vec::new();
    for name in &names {
        let problem = problems::by_name(name).ok_or_else(|| format!("unknown problem '{name}'"))?;
        let table = type_table(name).ok_or_else(|| format!("no type table for '{name}'"))?;
        if table.dim() != problem.dim() {
            return Err(format!("{name}: type table has D={}", table.dim()));
        }
        let timing = measure_capped(&problem, ENTRY_LIMIT_US);
        // 200 points twice (ROUND3 survey); 10 for problems already out by timing.
        let det_points = if timing.complete { 200 } else { 10 };
        let deterministic = deterministic_on(&problem, det_points);
        let mask = round3::de_mask(name).is_some();
        let discrete = table.count(Kind::Integer) + table.count(Kind::Categorical);
        let fast = timing.complete && timing.mean_us <= ENTRY_LIMIT_US;
        let ok = deterministic && discrete > 0 && fast;
        let mut reasons = Vec::new();
        if !deterministic {
            reasons.push("not deterministic".to_owned());
        }
        if discrete == 0 {
            reasons.push("no discrete coordinate in its single-objective cost".to_owned());
        }
        if !fast {
            reasons.push(if timing.complete {
                format!(
                    "mean evaluation time {:.1} us above 2,000 us",
                    timing.mean_us
                )
            } else {
                format!(
                    "stopped after {} of 2,000 points: their time alone exceeds 2,000 x 2 ms",
                    timing.points
                )
            });
        }
        let reason = if ok {
            "meets the entry rule".to_owned()
        } else {
            reasons.join("; ")
        };
        println!(
            "{name:<22} D={:<5} cont={:<4} int={:<4} cat={:<3} mean {:>12.3} us ({} pts) det={deterministic} ({det_points}) mask={mask} entered={}",
            problem.dim(),
            table.count(Kind::Continuous),
            table.count(Kind::Integer),
            table.count(Kind::Categorical),
            timing.mean_us,
            timing.points,
            if ok { "yes" } else { "no" }
        );
        tsv.push_str(&format!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{reason}\n",
            problem.domain,
            problem.dim(),
            table.count(Kind::Continuous),
            table.count(Kind::Integer),
            table.count(Kind::Categorical),
            timing.mean_us,
            timing.points,
            if deterministic { "yes" } else { "no" },
            if mask { "yes" } else { "no" },
            if ok { "yes" } else { "no" }
        ));
        std::fs::write(types_dir.join(format!("{name}.json")), table.to_json())
            .map_err(|e| e.to_string())?;
        if ok {
            entered.push((name.clone(), table.count(Kind::Categorical) > 0));
        }
    }
    if option(&parsed, "--problems").is_none() {
        for (name, domain, reason) in NOT_BUILT {
            tsv.push_str(&format!(
                "{name}\t{domain}\t-\t-\t-\t-\t-\t-\t-\t-\tno\t{reason}\n"
            ));
        }
    }
    std::fs::write(out.join("problems.tsv"), tsv).map_err(|e| e.to_string())?;
    let with_categorical = entered.iter().filter(|(_, c)| *c).count();
    println!(
        "entered: {} ({} with a categorical coordinate); ROUND3.md needs at least 5 and 3: {}",
        entered.len(),
        with_categorical,
        if entered.len() >= 5 && with_categorical >= 3 {
            "met"
        } else {
            "NOT met, the round stops"
        }
    );
    Ok(())
}

/// `survey3b`: the round-3b batch (ROUND3B.md section 2): timing over the
/// 2,000-point measurement sample, determinism over 200 points evaluated
/// twice, type tables, `problems.tsv`.
fn survey3b(args: &[String]) -> Result<(), String> {
    use candidate_harness::campaign::{DETERMINISM_POINTS, MEASURE_POINTS};
    use candidate_harness::round3::{Kind, deterministic_on};
    let parsed = options(args, &["--out"])?;
    let out = PathBuf::from(option(&parsed, "--out").ok_or("survey3b needs --out")?);
    let types_dir = out.join("types");
    std::fs::create_dir_all(&types_dir).map_err(|e| e.to_string())?;
    if let Ok(load) = std::fs::read_to_string("/proc/loadavg") {
        println!("load average before measuring: {}", load.trim());
    }
    let mut tsv = String::from(
        "problem\tdomain\tdim\tcontinuous\tinteger\tcategorical\tmean_eval_us\ttimed_points\t\
         deterministic\tde_mask\tentered\treason\n",
    );
    let mut entered = 0;
    for name in round3b::PROBLEMS {
        let problem = problems::by_name(name).ok_or_else(|| format!("unknown problem '{name}'"))?;
        let table =
            round3b::type_table(name).ok_or_else(|| format!("no type table for '{name}'"))?;
        if table.dim() != problem.dim() {
            return Err(format!("{name}: type table has D={}", table.dim()));
        }
        let mean = measure_problem(&problem);
        let deterministic = deterministic_on(&problem, DETERMINISM_POINTS);
        let categorical = table.count(Kind::Categorical);
        let ok = deterministic && categorical > 0;
        let reason = if ok {
            "deterministic, categorical coordinate; definition checked by tests/round3b.rs"
                .to_owned()
        } else {
            let mut reasons = Vec::new();
            if !deterministic {
                reasons.push("not deterministic");
            }
            if categorical == 0 {
                reasons.push("no categorical coordinate");
            }
            reasons.join("; ")
        };
        println!(
            "{name:<18} D={:<3} cont={:<2} int={:<2} cat={:<2} mean {mean:>10.3} us ({MEASURE_POINTS} pts) det={deterministic} ({DETERMINISM_POINTS}) entered={}",
            problem.dim(),
            table.count(Kind::Continuous),
            table.count(Kind::Integer),
            categorical,
            if ok { "yes" } else { "no" }
        );
        tsv.push_str(&format!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{mean:.3}\t{MEASURE_POINTS}\t{}\t{}\t{}\t{reason}\n",
            problem.domain,
            problem.dim(),
            table.count(Kind::Continuous),
            table.count(Kind::Integer),
            categorical,
            if deterministic { "yes" } else { "no" },
            if round3b::de_mask(name).is_some() {
                "yes"
            } else {
                "no"
            },
            if ok { "yes" } else { "no" }
        ));
        std::fs::write(types_dir.join(format!("{name}.json")), table.to_json())
            .map_err(|e| e.to_string())?;
        if ok {
            entered += 1;
        }
    }
    for (name, reason) in round3b::DROPPED {
        tsv.push_str(&format!(
            "{name}\t-\t-\t-\t-\t-\t-\t-\t-\t-\tno\t{reason}\n"
        ));
    }
    std::fs::write(out.join("problems.tsv"), tsv).map_err(|e| e.to_string())?;
    println!(
        "entered: {entered} of {}; ROUND3B.md needs at least 2: {}",
        round3b::PROBLEMS.len() + round3b::DROPPED.len(),
        if entered >= 2 {
            "met"
        } else {
            "NOT met, the batch stops"
        }
    );
    Ok(())
}
