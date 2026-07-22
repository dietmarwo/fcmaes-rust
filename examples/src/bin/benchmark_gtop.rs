use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use fcmaes_examples::benchmark_gtop::{
    RunRecord, render_adoc, run_seed, selected_cases, summarize,
};
use fcmaes_examples::runner::{Cli, run_advanced};

#[derive(Debug)]
struct BenchmarkCli {
    runs: usize,
    workers: usize,
    evaluations: u64,
    seed: u64,
    max_eval_fac: f64,
    check_interval: usize,
    progress_interval: f64,
    problem: Option<String>,
    include_slow: bool,
    raw_output: Option<PathBuf>,
    table_output: Option<PathBuf>,
}

impl Default for BenchmarkCli {
    fn default() -> Self {
        Self {
            runs: 100,
            workers: 0,
            evaluations: 1_500,
            seed: 1,
            max_eval_fac: 50.0,
            check_interval: 100,
            progress_interval: 0.0,
            problem: None,
            include_slow: false,
            raw_output: None,
            table_output: None,
        }
    }
}

impl BenchmarkCli {
    const USAGE: &str = "options:\n\
      --runs N                 independent experiments per problem (default 100)\n\
      --workers N              retry workers; 0 uses available parallelism\n\
      --evaluations N          initial evaluations per retry (default 1500)\n\
      --seed N                 deterministic root seed\n\
      --max-eval-fac N         advanced final evaluation factor\n\
      --check-interval N       advanced diversity checkpoint interval\n\
      --progress-interval N    live interval inside each experiment\n\
      --problem NAME           run one benchmark problem\n\
      --include-slow           include Tandem and Messenger Full\n\
      --raw-output PATH        write one TSV row after every experiment\n\
      --table-output PATH      update the AsciiDoc table after every problem\n\
      --help                   show this help";

    fn from_env() -> Result<Option<Self>, String> {
        Self::from_args(std::env::args().skip(1))
    }

    fn from_args<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cli = Self::default();
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            let argument = argument.as_ref();
            match argument {
                "--help" | "-h" => return Ok(None),
                "--include-slow" => cli.include_slow = true,
                _ => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{argument} requires a value"))?;
                    let value = value.as_ref();
                    match argument {
                        "--runs" => {
                            cli.runs = value
                                .parse()
                                .map_err(|_| "--runs requires an integer".to_owned())?
                        }
                        "--workers" => {
                            cli.workers = value
                                .parse()
                                .map_err(|_| "--workers requires an integer".to_owned())?
                        }
                        "--evaluations" => {
                            cli.evaluations = value
                                .parse()
                                .map_err(|_| "--evaluations requires an integer".to_owned())?
                        }
                        "--seed" => {
                            cli.seed = value
                                .parse()
                                .map_err(|_| "--seed requires an integer".to_owned())?
                        }
                        "--max-eval-fac" => {
                            cli.max_eval_fac = value
                                .parse()
                                .map_err(|_| "--max-eval-fac requires a number".to_owned())?
                        }
                        "--check-interval" => {
                            cli.check_interval = value
                                .parse()
                                .map_err(|_| "--check-interval requires an integer".to_owned())?
                        }
                        "--progress-interval" => {
                            cli.progress_interval = value
                                .parse()
                                .map_err(|_| "--progress-interval requires a number".to_owned())?
                        }
                        "--problem" => cli.problem = Some(value.to_owned()),
                        "--raw-output" => cli.raw_output = Some(value.into()),
                        "--table-output" => cli.table_output = Some(value.into()),
                        _ => return Err(format!("unknown option '{argument}'")),
                    }
                }
            }
        }
        if cli.runs == 0 || cli.evaluations == 0 {
            return Err("--runs and --evaluations must be positive".to_owned());
        }
        if !cli.max_eval_fac.is_finite() || cli.max_eval_fac < 1.0 {
            return Err("--max-eval-fac must be finite and at least 1".to_owned());
        }
        if !cli.progress_interval.is_finite() || cli.progress_interval < 0.0 {
            return Err("--progress-interval must be finite and non-negative".to_owned());
        }
        Ok(Some(cli))
    }
}

fn main() {
    let cli = match BenchmarkCli::from_env() {
        Ok(Some(cli)) => cli,
        Ok(None) => {
            println!("{}", BenchmarkCli::USAGE);
            return;
        }
        Err(message) => {
            eprintln!("{message}\n\n{}", BenchmarkCli::USAGE);
            std::process::exit(2);
        }
    };
    let cases =
        selected_cases(cli.problem.as_deref(), cli.include_slow).unwrap_or_else(|message| {
            eprintln!("{message}");
            std::process::exit(2);
        });
    let mut raw_output = cli.raw_output.as_ref().map(|path| {
        let file = File::create(path)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display()));
        let mut writer = BufWriter::new(file);
        writeln!(
            writer,
            "problem\trun\tseed\tsuccess\tvalue\tevaluations\tretries\twall_seconds"
        )
        .expect("failed to write raw header");
        writer
    });

    println!(
        "benchmark runs={} workers={} evaluations={} max_eval_fac={} check_interval={} slow={}",
        cli.runs,
        cli.workers,
        cli.evaluations,
        cli.max_eval_fac,
        cli.check_interval,
        cli.include_slow
    );
    let mut summaries = Vec::with_capacity(cases.len());
    for (case_index, case) in cases.iter().enumerate() {
        let mut records = Vec::with_capacity(cli.runs);
        for run_index in 0..cli.runs {
            let seed = run_seed(cli.seed, case_index, run_index);
            let run_cli = Cli {
                retries: case.max_retries,
                evaluations: cli.evaluations,
                workers: cli.workers,
                seed,
                max_eval_fac: cli.max_eval_fac,
                check_interval: cli.check_interval,
                value_limit: case.value_limit,
                stop_fitness: case.stop_value,
                problem: None,
                progress_interval: cli.progress_interval,
            };
            let started = Instant::now();
            let result = run_advanced(&case.problem, &run_cli);
            let wall_seconds = started.elapsed().as_secs_f64();
            let success = result.success && result.y <= case.stop_value;
            let record = RunRecord {
                run: run_index + 1,
                seed,
                success,
                value: result.y,
                evaluations: result.evaluations,
                retries: result.runs,
                wall_seconds,
            };
            println!(
                "problem=\"{}\" run={}/{} success={} value={:.12} evaluations={} retries={} wall={:.3}s",
                case.problem.name,
                record.run,
                cli.runs,
                record.success,
                record.value,
                record.evaluations,
                record.retries,
                record.wall_seconds
            );
            std::io::stdout().flush().expect("failed to flush stdout");
            if let Some(writer) = raw_output.as_mut() {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{:.17}\t{}\t{}\t{:.9}",
                    case.key,
                    record.run,
                    record.seed,
                    record.success,
                    record.value,
                    record.evaluations,
                    record.retries,
                    record.wall_seconds
                )
                .expect("failed to write raw result");
                writer.flush().expect("failed to flush raw result");
            }
            records.push(record);
        }
        summaries.push(summarize(case, &records));
        if let Some(path) = cli.table_output.as_ref() {
            std::fs::write(path, render_adoc(&summaries))
                .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        }
    }
    let table = render_adoc(&summaries);
    println!("\n{table}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete_configuration() {
        let cli = BenchmarkCli::from_args([
            "--runs",
            "7",
            "--workers",
            "3",
            "--evaluations",
            "2000",
            "--seed",
            "9",
            "--max-eval-fac",
            "4",
            "--check-interval",
            "20",
            "--progress-interval",
            "0.5",
            "--problem",
            "sagas",
            "--include-slow",
            "--raw-output",
            "raw.tsv",
            "--table-output",
            "table.adoc",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(cli.runs, 7);
        assert_eq!(cli.workers, 3);
        assert_eq!(cli.evaluations, 2_000);
        assert_eq!(cli.seed, 9);
        assert_eq!(cli.max_eval_fac, 4.0);
        assert_eq!(cli.check_interval, 20);
        assert_eq!(cli.progress_interval, 0.5);
        assert_eq!(cli.problem.as_deref(), Some("sagas"));
        assert!(cli.include_slow);
        assert_eq!(cli.raw_output, Some(PathBuf::from("raw.tsv")));
        assert_eq!(cli.table_output, Some(PathBuf::from("table.adoc")));
    }

    #[test]
    fn handles_help_and_rejects_invalid_options() {
        assert!(BenchmarkCli::from_args(["--help"]).unwrap().is_none());
        assert!(BenchmarkCli::from_args(["--runs", "0"]).is_err());
        assert!(BenchmarkCli::from_args(["--evaluations", "0"]).is_err());
        assert!(BenchmarkCli::from_args(["--max-eval-fac", "0.9"]).is_err());
        assert!(BenchmarkCli::from_args(["--progress-interval", "nan"]).is_err());
        assert!(BenchmarkCli::from_args(["--unknown", "1"]).is_err());
        assert!(BenchmarkCli::from_args(["--runs"]).is_err());
    }
}
