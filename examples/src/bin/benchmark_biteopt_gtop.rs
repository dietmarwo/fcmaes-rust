use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use fcmaes_examples::benchmark_gtop::{
    BenchmarkCase, ProblemSummary, RunRecord, mean_sdev, run_seed, selected_bite_cases, summarize,
};
use fcmaes_examples::runner::{Cli, run_basic, run_bite};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Algorithm {
    Biteopt,
    DeCma,
}

impl Algorithm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "biteopt" => Ok(Self::Biteopt),
            "de_cma" => Ok(Self::DeCma),
            _ => Err("--algo must be 'biteopt' or 'de_cma'".to_owned()),
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Biteopt => "biteopt",
            Self::DeCma => "de_cma",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Biteopt => "BiteOpt",
            Self::DeCma => "DE-CMA",
        }
    }
}

#[derive(Clone, Debug)]
struct BenchmarkSummary {
    base: ProblemSummary,
    mean_optimum: f64,
    sdev_optimum: f64,
}

fn summarize_results(case: &BenchmarkCase, records: &[RunRecord]) -> BenchmarkSummary {
    let values: Vec<f64> = records.iter().map(|record| record.value).collect();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{} has a non-finite final optimum",
        case.display_name
    );
    let (mean_optimum, sdev_optimum) = mean_sdev(&values);
    BenchmarkSummary {
        base: summarize(case, records),
        mean_optimum,
        sdev_optimum,
    }
}

fn render_markdown(algorithm: Algorithm, summaries: &[BenchmarkSummary]) -> String {
    let table_title = format!(
        "GTOP {} retry results for stopVal = 1.005*absolute_best (Rust)",
        algorithm.label()
    );
    let mut output = format!(
        "## {table_title}\n\n\
         | Problem | Runs | Absolute best | Stop value | Success rate | Mean optimum | Sdev optimum | Mean time | Sdev time |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    );
    for summary in summaries {
        let base = &summary.base;
        let success_rate = if base.runs == 0 {
            0.0
        } else {
            100.0 * base.successes as f64 / base.runs as f64
        };
        writeln!(
            output,
            "| {} | {} | {} | {} | {:.0}% | {:.6} | {:.6} | {:.2}s | {:.2}s |",
            base.problem,
            base.runs,
            base.absolute_best_label,
            base.stop_value_label,
            success_rate,
            summary.mean_optimum,
            summary.sdev_optimum,
            base.mean_seconds,
            base.sdev_seconds
        )
        .expect("writing to a String cannot fail");
    }
    output
}

#[derive(Debug)]
struct BenchmarkCli {
    algorithm: Algorithm,
    runs: usize,
    workers: usize,
    retries: usize,
    evaluations: u64,
    seed: u64,
    problem: Option<String>,
    progress_interval: f64,
    raw_output: Option<PathBuf>,
    table_output: Option<PathBuf>,
}

impl Default for BenchmarkCli {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Biteopt,
            runs: 100,
            workers: 24,
            retries: 24,
            evaluations: 10_000,
            seed: 1,
            problem: None,
            progress_interval: 0.0,
            raw_output: None,
            table_output: None,
        }
    }
}

impl BenchmarkCli {
    const USAGE: &str = "options:\n\
      --algo NAME              biteopt or de_cma (default biteopt)\n\
      --runs N                 independent experiments per problem (default 100)\n\
      --workers N              basic-retry worker threads (default 24)\n\
      --retries N              optimizer retries per experiment (default 24)\n\
      --evaluations N          maximum evaluations per retry (default 10000)\n\
      --seed N                 deterministic root seed (default 1)\n\
      --problem NAME           run one included benchmark problem\n\
      --progress-interval N    live interval inside each experiment\n\
      --raw-output PATH        write one TSV row after every experiment\n\
      --table-output PATH      update the Markdown table after every problem\n\
      --help                   show this help\n\
\n\
Messenger Full is always excluded; Tandem is included by default.";

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
            if argument == "--help" || argument == "-h" {
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?;
            let value = value.as_ref();
            match argument {
                "--algo" => cli.algorithm = Algorithm::parse(value)?,
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
                "--seed" => {
                    cli.seed = value
                        .parse()
                        .map_err(|_| "--seed requires an integer".to_owned())?
                }
                "--problem" => cli.problem = Some(value.to_owned()),
                "--progress-interval" => {
                    cli.progress_interval = value
                        .parse()
                        .map_err(|_| "--progress-interval requires a number".to_owned())?
                }
                "--raw-output" => cli.raw_output = Some(value.into()),
                "--table-output" => cli.table_output = Some(value.into()),
                _ => return Err(format!("unknown option '{argument}'")),
            }
        }
        if cli.runs == 0 || cli.workers == 0 || cli.retries == 0 || cli.evaluations == 0 {
            return Err(
                "--runs, --workers, --retries and --evaluations must be positive".to_owned(),
            );
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
    let cases = selected_bite_cases(cli.problem.as_deref()).unwrap_or_else(|message| {
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
        "benchmark optimizer={} retry=basic runs={} workers={} retries={} evaluations={}",
        cli.algorithm.key(),
        cli.runs,
        cli.workers,
        cli.retries,
        cli.evaluations
    );
    let mut summaries = Vec::with_capacity(cases.len());
    for (case_index, case) in cases.iter().enumerate() {
        let mut records = Vec::with_capacity(cli.runs);
        for run_index in 0..cli.runs {
            let seed = run_seed(cli.seed, case_index, run_index);
            let run_cli = Cli {
                retries: cli.retries,
                evaluations: cli.evaluations,
                workers: cli.workers,
                seed,
                // Keep every finite final result so unsuccessful experiments
                // contribute to the reported optimum statistics.
                value_limit: f64::INFINITY,
                stop_fitness: case.stop_value,
                progress_interval: cli.progress_interval,
                ..Default::default()
            };
            let started = Instant::now();
            let result = match cli.algorithm {
                Algorithm::Biteopt => run_bite(&case.problem, &run_cli),
                Algorithm::DeCma => run_basic(&case.problem, &run_cli),
            };
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
        summaries.push(summarize_results(case, &records));
        if let Some(path) = cli.table_output.as_ref() {
            std::fs::write(path, render_markdown(cli.algorithm, &summaries))
                .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        }
    }
    println!("\n{}", render_markdown(cli.algorithm, &summaries));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_requested_benchmark() {
        let cli = BenchmarkCli::default();
        assert_eq!(cli.algorithm, Algorithm::Biteopt);
        assert_eq!(cli.runs, 100);
        assert_eq!(cli.workers, 24);
        assert_eq!(cli.retries, 24);
        assert_eq!(cli.evaluations, 10_000);
    }

    #[test]
    fn parses_configuration_and_rejects_invalid_values() {
        let cli = BenchmarkCli::from_args([
            "--algo",
            "de_cma",
            "--runs",
            "2",
            "--workers",
            "3",
            "--retries",
            "4",
            "--evaluations",
            "500",
            "--seed",
            "7",
            "--problem",
            "tandem",
            "--progress-interval",
            "0.5",
            "--raw-output",
            "raw.tsv",
            "--table-output",
            "table.md",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(cli.algorithm, Algorithm::DeCma);
        assert_eq!(cli.runs, 2);
        assert_eq!(cli.workers, 3);
        assert_eq!(cli.retries, 4);
        assert_eq!(cli.evaluations, 500);
        assert_eq!(cli.seed, 7);
        assert_eq!(cli.problem.as_deref(), Some("tandem"));
        assert_eq!(cli.progress_interval, 0.5);
        assert_eq!(cli.raw_output, Some(PathBuf::from("raw.tsv")));
        assert_eq!(cli.table_output, Some(PathBuf::from("table.md")));

        assert!(BenchmarkCli::from_args(["--help"]).unwrap().is_none());
        assert!(BenchmarkCli::from_args(["--workers", "0"]).is_err());
        assert!(BenchmarkCli::from_args(["--runs"]).is_err());
        assert!(BenchmarkCli::from_args(["--unknown", "1"]).is_err());
        assert!(BenchmarkCli::from_args(["--algo", "cma"]).is_err());
    }

    #[test]
    fn reports_population_statistics_for_final_optima() {
        let case = selected_bite_cases(Some("cassini1")).unwrap().remove(0);
        let records = [
            RunRecord {
                run: 1,
                seed: 1,
                success: true,
                value: 4.95,
                evaluations: 10,
                retries: 1,
                wall_seconds: 1.0,
            },
            RunRecord {
                run: 2,
                seed: 2,
                success: false,
                value: 5.0,
                evaluations: 10,
                retries: 1,
                wall_seconds: 3.0,
            },
        ];
        let summary = summarize_results(&case, &records);
        assert!((summary.mean_optimum - 4.975).abs() < 1.0e-12);
        assert!((summary.sdev_optimum - 0.025).abs() < 1.0e-12);
        assert!(
            render_markdown(Algorithm::Biteopt, std::slice::from_ref(&summary)).contains(
                "| Cassini1 | 2 | 4.9307 | 4.95535 | 50% | 4.975000 | 0.025000 | 2.00s | 1.00s |"
            )
        );
        assert!(
            render_markdown(Algorithm::DeCma, &[summary]).contains("GTOP DE-CMA retry results")
        );
    }
}
