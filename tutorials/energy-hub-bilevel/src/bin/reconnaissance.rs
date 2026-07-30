//! M1 solver-API and horizon reconnaissance.

use std::error::Error;
use std::time::Instant;

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolveOutcome, Variable};

use energy_hub_bilevel::config::{CURTAILMENT_COST, ELECTRICITY_VOLL};

#[derive(Clone, Copy)]
struct HourVars {
    import: Variable,
    export: Variable,
    charge: Variable,
    discharge: Variable,
    soc: Variable,
    curtail: Variable,
    unserved: Variable,
}

fn synthetic_hour(hour: usize) -> (f64, f64, f64) {
    let day_fraction = (hour % 24) as f64 / 24.0;
    let year_fraction = hour as f64 / 8_760.0;
    let solar = (std::f64::consts::TAU * (day_fraction - 0.25))
        .sin()
        .max(0.0)
        * (0.72 + 0.28 * (std::f64::consts::TAU * (year_fraction - 0.25)).sin());
    let wind = (0.38
        + 0.12 * (std::f64::consts::TAU * year_fraction * 7.0).sin()
        + 0.08 * (std::f64::consts::TAU * day_fraction * 3.0).cos())
    .clamp(0.05, 0.8);
    let load = 1_100.0
        + 170.0 * (std::f64::consts::TAU * (day_fraction - 0.2)).sin()
        + 130.0 * (std::f64::consts::TAU * (year_fraction + 0.15)).cos();
    (solar, wind, load)
}

fn build_dispatch(hours: usize) -> (Problem, Vec<HourVars>) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let mut variables = Vec::with_capacity(hours);
    for hour in 0..hours {
        let (solar, wind, _) = synthetic_hour(hour);
        let generation = 1_500.0 * solar + 800.0 * wind;
        let import_price = if matches!(hour % 24, 7..=10 | 17..=21) {
            0.24
        } else {
            0.11
        };
        variables.push(HourVars {
            import: problem.add_var(import_price, (0.0, 1_500.0)),
            export: problem.add_var(-0.04, (0.0, 1_500.0)),
            charge: problem.add_var(1.0e-7, (0.0, 700.0)),
            discharge: problem.add_var(1.0e-7, (0.0, 700.0)),
            soc: problem.add_var(0.0, (0.0, 3_000.0)),
            curtail: problem.add_var(CURTAILMENT_COST, (0.0, generation)),
            unserved: problem.add_var(ELECTRICITY_VOLL, (0.0, f64::INFINITY)),
        });
    }
    for hour in 0..hours {
        let (solar, wind, load) = synthetic_hour(hour);
        let generation = 1_500.0 * solar + 800.0 * wind;
        let current = variables[hour];
        let previous = variables[(hour + hours - 1) % hours];
        problem.add_constraint(
            [
                (current.import, 1.0),
                (current.export, -1.0),
                (current.charge, -1.0),
                (current.discharge, 1.0),
                (current.curtail, -1.0),
                (current.unserved, 1.0),
            ],
            ComparisonOp::Eq,
            load - generation,
        );
        problem.add_constraint(
            [
                (current.soc, 1.0),
                (previous.soc, -1.0),
                (current.charge, -0.95),
                (current.discharge, 1.0 / 0.95),
            ],
            ComparisonOp::Eq,
            0.0,
        );
    }
    (problem, variables)
}

fn main() -> Result<(), Box<dyn Error>> {
    fn assert_send<T: Send>() {}
    assert_send::<Problem>();
    assert_send::<SolveOutcome>();

    println!("hours,variables,constraints_proxy,wall_seconds,lp_iterations,objective");
    for hours in [24, 288, 2_016, 8_760] {
        let (problem, variables) = build_dispatch(hours);
        let started = Instant::now();
        let outcome = problem.solve()?;
        let elapsed = started.elapsed().as_secs_f64();
        let solution = outcome.into_solution().map_err(|interrupted| {
            format!(
                "unexpected interrupted pure LP at {hours} h: {:?}",
                interrupted.termination_reason()
            )
        })?;
        println!(
            "{hours},{},{},{elapsed:.9},{},{}",
            variables.len() * 7,
            2 * hours,
            solution.stats().lp_iterations,
            solution.objective()
        );
    }
    Ok(())
}
