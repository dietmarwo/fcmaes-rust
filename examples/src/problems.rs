//! Named GTOP problems and their search boxes.

use std::f64::consts::PI;

use fcmaes_core::RetryBounds;

use crate::gtop;

pub type ProblemFn = fn(&[f64]) -> f64;

#[derive(Clone)]
pub struct Problem {
    pub name: &'static str,
    pub objective: ProblemFn,
    pub bounds: RetryBounds,
}

fn problem(name: &'static str, objective: ProblemFn, lower: &[f64], upper: &[f64]) -> Problem {
    Problem {
        name,
        objective,
        bounds: RetryBounds::new(lower.to_vec(), upper.to_vec()).expect("static bounds are valid"),
    }
}

pub fn cassini1() -> Problem {
    problem(
        "Cassini1",
        gtop::cassini1,
        &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
        &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
    )
}

pub fn cassini2() -> Problem {
    problem(
        "Cassini2",
        gtop::cassini2,
        &[
            -1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.05, 1.05, 1.15, 1.7, -PI, -PI, -PI, -PI,
        ],
        &[
            0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0,
            6.0, 6.5, 291.0, PI, PI, PI, PI,
        ],
    )
}

pub fn messenger() -> Problem {
    problem(
        "Messenger reduced",
        gtop::messenger,
        &[
            1000.0, 1.0, 0.0, 0.0, 200.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1,
            -PI, -PI, -PI,
        ],
        &[
            4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0,
            6.0, PI, PI, PI,
        ],
    )
}

pub fn messenger_full() -> Problem {
    problem(
        "Messenger full",
        gtop::messenger_full,
        &[
            1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI, -PI,
        ],
        &[
            2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0, 0.99, 0.99, 0.99,
            0.99, 0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0, PI, PI, PI, PI, PI,
        ],
    )
}

pub fn rosetta() -> Problem {
    problem(
        "Rosetta",
        gtop::rosetta,
        &[
            1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.05, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI,
        ],
        &[
            1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9,
            9.0, 9.0, 9.0, 9.0, PI, PI, PI, PI,
        ],
    )
}

pub fn sagas() -> Problem {
    problem(
        "Sagas",
        gtop::sagas,
        &[
            7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0, -PI, -PI,
        ],
        &[
            9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0, PI, PI,
        ],
    )
}

pub fn gtoc1() -> Problem {
    fn shifted(x: &[f64]) -> f64 {
        gtop::gtoc1(x) - 2_000_000.0
    }
    problem(
        "GTOC1",
        shifted,
        &[3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0],
        &[
            10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0,
        ],
    )
}

pub fn cassini1_minlp() -> Problem {
    fn fixed_sequence(x: &[f64]) -> f64 {
        let mut values = Vec::with_capacity(10);
        values.extend_from_slice(x);
        values.extend([2.0, 2.0, 3.0, 5.0]);
        gtop::cassini1_minlp(&values).0
    }
    problem(
        "Cassini1 MINLP",
        fixed_sequence,
        &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
        &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
    )
}

pub fn tandem_5() -> Problem {
    fn objective(x: &[f64]) -> f64 {
        gtop::tandem(x, &[3, 2, 3, 3, 6])
    }
    problem(
        "Tandem 6",
        objective,
        &[
            5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05,
            1.05, -PI, -PI, -PI,
        ],
        &[
            9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99, 0.99, 0.99, 0.99, 10.0,
            10.0, 10.0, PI, PI, PI,
        ],
    )
}

pub fn all() -> Vec<Problem> {
    vec![
        cassini1(),
        cassini2(),
        rosetta(),
        tandem_5(),
        messenger(),
        gtoc1(),
        messenger_full(),
        sagas(),
        cassini1_minlp(),
    ]
}

fn normalized_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Resolve a CLI-friendly problem name. Separators and case are ignored.
pub fn by_name(name: &str) -> Option<Problem> {
    match normalized_name(name).as_str() {
        "cassini1" => Some(cassini1()),
        "cassini2" => Some(cassini2()),
        "rosetta" => Some(rosetta()),
        "tandem" | "tandem5" | "tandem6" => Some(tandem_5()),
        "messenger" | "messengerreduced" => Some(messenger()),
        "gtoc1" => Some(gtoc1()),
        "messfull" | "messengerfull" => Some(messenger_full()),
        "sagas" => Some(sagas()),
        "cassini1minlp" => Some(cassini1_minlp()),
        _ => None,
    }
}

/// Return all problems or the single problem requested by the CLI.
pub fn selected(name: Option<&str>) -> Result<Vec<Problem>, String> {
    match name {
        None => Ok(all()),
        Some(name) if normalized_name(name) == "all" => Ok(all()),
        Some(name) => by_name(name).map(|problem| vec![problem]).ok_or_else(|| {
            format!(
                "unknown problem '{name}'; expected one of: cassini1, cassini2, rosetta, \
                 tandem, messenger, gtoc1, messenger-full, sagas, cassini1-minlp"
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_catalog_has_valid_bounds_and_finite_midpoints() {
        let problems = all();
        assert_eq!(problems.len(), 9);
        for problem in problems {
            assert_eq!(problem.bounds.lower().len(), problem.bounds.upper().len());
            let midpoint: Vec<f64> = problem
                .bounds
                .lower()
                .iter()
                .zip(problem.bounds.upper())
                .map(|(&lower, &upper)| 0.5 * (lower + upper))
                .collect();
            assert!(
                (problem.objective)(&midpoint).is_finite(),
                "{}",
                problem.name
            );
        }
    }

    #[test]
    fn cli_problem_names_accept_aliases_and_reject_unknown_names() {
        for alias in [
            "messenger-full",
            "Messenger Full",
            "messenger_full",
            "messfull",
        ] {
            assert_eq!(by_name(alias).unwrap().name, "Messenger full");
        }
        assert_eq!(by_name("messenger").unwrap().name, "Messenger reduced");
        assert_eq!(selected(Some("all")).unwrap().len(), 9);
        assert!(selected(Some("not-a-problem")).is_err());
    }
}
