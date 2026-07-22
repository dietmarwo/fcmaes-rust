//! Configuration and result formatting for the native GTOP benchmark.

use std::fmt::Write;

use crate::problems::{self, Problem};

/// One GTOP benchmark definition with its relaxed success threshold.
pub struct BenchmarkCase {
    pub key: &'static str,
    pub display_name: &'static str,
    pub problem: Problem,
    pub max_retries: usize,
    pub value_limit: f64,
    pub absolute_best: f64,
    pub absolute_best_label: &'static str,
    pub stop_value: f64,
    pub stop_value_label: &'static str,
    pub slow: bool,
}

/// Result of one independent coordinated-retry experiment.
#[derive(Clone, Debug)]
pub struct RunRecord {
    pub run: usize,
    pub seed: u64,
    pub success: bool,
    pub value: f64,
    pub evaluations: u64,
    pub retries: usize,
    pub wall_seconds: f64,
}

/// Aggregate row rendered into the tutorial-style table.
#[derive(Clone, Debug)]
pub struct ProblemSummary {
    pub problem: &'static str,
    pub runs: usize,
    pub absolute_best_label: &'static str,
    pub stop_value_label: &'static str,
    pub successes: usize,
    pub mean_seconds: f64,
    pub sdev_seconds: f64,
}

/// Full benchmark catalog. Tandem and Messenger Full are marked as slow and
/// excluded by the default binary invocation.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            key: "cassini1",
            display_name: "Cassini1",
            problem: problems::cassini1(),
            max_retries: 4_000,
            value_limit: 20.0,
            absolute_best: 4.9307,
            absolute_best_label: "4.9307",
            stop_value: 4.95535,
            stop_value_label: "4.95535",
            slow: false,
        },
        BenchmarkCase {
            key: "cassini2",
            display_name: "Cassini2",
            problem: problems::cassini2(),
            max_retries: 6_000,
            value_limit: 20.0,
            absolute_best: 8.383,
            absolute_best_label: "8.383",
            stop_value: 8.42491,
            stop_value_label: "8.42491",
            slow: false,
        },
        BenchmarkCase {
            key: "gtoc1",
            display_name: "Gtoc1",
            problem: problems::gtoc1(),
            max_retries: 10_000,
            value_limit: -300_000.0,
            absolute_best: -1_581_950.0,
            absolute_best_label: "-1581950",
            stop_value: -1_574_080.0,
            stop_value_label: "-1574080",
            slow: false,
        },
        BenchmarkCase {
            key: "messenger",
            display_name: "Messenger",
            problem: problems::messenger(),
            max_retries: 8_000,
            value_limit: 20.0,
            absolute_best: 8.6299,
            absolute_best_label: "8.6299",
            stop_value: 8.673,
            stop_value_label: "8.673",
            slow: false,
        },
        BenchmarkCase {
            key: "rosetta",
            display_name: "Rosetta",
            problem: problems::rosetta(),
            max_retries: 4_000,
            value_limit: 20.0,
            absolute_best: 1.3433,
            absolute_best_label: "1.3433",
            stop_value: 1.35,
            stop_value_label: "1.35",
            slow: false,
        },
        BenchmarkCase {
            key: "tandem",
            display_name: "Tandem",
            problem: problems::tandem_5(),
            max_retries: 20_000,
            value_limit: -300.0,
            absolute_best: -1_500.46,
            absolute_best_label: "-1500.46",
            stop_value: -1_493.0,
            stop_value_label: "-1493",
            slow: true,
        },
        BenchmarkCase {
            key: "sagas",
            display_name: "Sagas",
            problem: problems::sagas(),
            max_retries: 4_000,
            value_limit: 100.0,
            absolute_best: 18.188,
            absolute_best_label: "18.188",
            stop_value: 18.279,
            stop_value_label: "18.279",
            slow: false,
        },
        BenchmarkCase {
            key: "messenger-full",
            display_name: "Messenger Full",
            problem: problems::messenger_full(),
            max_retries: 50_000,
            value_limit: 12.0,
            absolute_best: 1.9579,
            absolute_best_label: "1.9579",
            stop_value: 1.96769,
            stop_value_label: "1.96769",
            slow: true,
        },
    ]
}

/// Select the default quick catalog or one explicitly requested problem.
pub fn selected_cases(
    problem: Option<&str>,
    include_slow: bool,
) -> Result<Vec<BenchmarkCase>, String> {
    let mut all = cases();
    if let Some(requested) = problem {
        let selected = problems::by_name(requested)
            .ok_or_else(|| format!("unknown GTOP problem '{requested}'"))?;
        let display_name = selected.name;
        return all
            .drain(..)
            .find(|case| case.problem.name == display_name)
            .map(|case| vec![case])
            .ok_or_else(|| format!("problem '{requested}' is not in benchmark_gtop.py"));
    }
    if !include_slow {
        all.retain(|case| !case.slow);
    }
    Ok(all)
}

/// Select the matched basic-retry benchmark catalog: every original GTOP
/// benchmark except Messenger Full. Tandem is deliberately included.
pub fn selected_bite_cases(problem: Option<&str>) -> Result<Vec<BenchmarkCase>, String> {
    let mut selected = cases();
    selected.retain(|case| case.key != "messenger-full");
    if let Some(requested) = problem {
        let requested = problems::by_name(requested)
            .ok_or_else(|| format!("unknown GTOP problem '{requested}'"))?;
        return selected
            .drain(..)
            .find(|case| case.problem.name == requested.name)
            .map(|case| vec![case])
            .ok_or_else(|| "Messenger Full is excluded from this benchmark".to_owned());
    }
    Ok(selected)
}

/// Stable seed assignment independent of scheduling order.
pub fn run_seed(base_seed: u64, case_index: usize, run_index: usize) -> u64 {
    base_seed
        .wrapping_add((case_index as u64).wrapping_mul(1_000_003))
        .wrapping_add(run_index as u64)
}

/// Population mean and standard deviation, matching NumPy's default `std`.
pub fn mean_sdev(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

pub fn summarize(case: &BenchmarkCase, records: &[RunRecord]) -> ProblemSummary {
    let times: Vec<f64> = records.iter().map(|record| record.wall_seconds).collect();
    let (mean_seconds, sdev_seconds) = mean_sdev(&times);
    ProblemSummary {
        problem: case.display_name,
        runs: records.len(),
        absolute_best_label: case.absolute_best_label,
        stop_value_label: case.stop_value_label,
        successes: records.iter().filter(|record| record.success).count(),
        mean_seconds,
        sdev_seconds,
    }
}

/// Render a compact AsciiDoc result table.
pub fn render_adoc(summaries: &[ProblemSummary]) -> String {
    render_adoc_with_title(
        "GTOP coordinated retry results for stopVal = 1.005*absolute_best (Rust)",
        summaries,
    )
}

/// Render a tutorial-style result table with a caller-supplied title.
pub fn render_adoc_with_title(title: &str, summaries: &[ProblemSummary]) -> String {
    let mut output = format!(
        ".{title}\n\
         [width=\"80%\",cols=\"3,^2,^2,^2,^2,^2,^2\",options=\"header\"]\n\
         |=========================================================\n\
         |problem |runs |absolute best |stopVal |success rate |mean time |sdev time\n"
    );
    for summary in summaries {
        let success_rate = if summary.runs == 0 {
            0.0
        } else {
            100.0 * summary.successes as f64 / summary.runs as f64
        };
        writeln!(
            output,
            "|{} |{} |{} |{} |{:.0}% |{:.2}s |{:.2}s",
            summary.problem,
            summary.runs,
            summary.absolute_best_label,
            summary.stop_value_label,
            success_rate,
            summary.mean_seconds,
            summary.sdev_seconds
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("|=========================================================\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_selection_excludes_only_the_two_slow_cases() {
        let selected = selected_cases(None, false).unwrap();
        assert_eq!(selected.len(), 6);
        assert!(selected.iter().all(|case| !case.slow));
        assert!(!selected.iter().any(|case| case.key == "tandem"));
        assert!(!selected.iter().any(|case| case.key == "messenger-full"));
    }

    #[test]
    fn explicit_problem_allows_slow_aliases() {
        let selected = selected_cases(Some("messenger_full"), false).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].key, "messenger-full");
        assert!(selected_cases(Some("cassini1-minlp"), true).is_err());
    }

    #[test]
    fn bite_selection_includes_tandem_and_excludes_only_messenger_full() {
        let selected = selected_bite_cases(None).unwrap();
        assert_eq!(selected.len(), 7);
        assert!(selected.iter().any(|case| case.key == "tandem"));
        assert!(!selected.iter().any(|case| case.key == "messenger-full"));
        assert_eq!(
            selected_bite_cases(Some("tandem")).unwrap()[0].key,
            "tandem"
        );
        assert!(selected_bite_cases(Some("messenger-full")).is_err());
    }

    #[test]
    fn statistics_and_table_match_the_tutorial_shape() {
        let (mean, sdev) = mean_sdev(&[1.0, 2.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert!((sdev - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
        let summary = ProblemSummary {
            problem: "Cassini1",
            runs: 100,
            absolute_best_label: "4.9307",
            stop_value_label: "4.95535",
            successes: 99,
            mean_seconds: 0.125,
            sdev_seconds: 0.025,
        };
        let table = render_adoc(&[summary]);
        assert!(table.contains("GTOP coordinated retry results"));
        assert!(table.contains("|Cassini1 |100 |4.9307 |4.95535 |99% |0.12s |0.03s"));
    }

    #[test]
    fn run_seeds_are_stable_and_distinct() {
        assert_eq!(run_seed(7, 0, 0), 7);
        assert_ne!(run_seed(7, 0, 1), run_seed(7, 1, 0));
        assert_eq!(run_seed(7, 2, 3), run_seed(7, 2, 3));
    }
}
