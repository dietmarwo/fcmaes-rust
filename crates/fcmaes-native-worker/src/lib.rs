//! Native objective registry and solver dispatch, shared by
//! `fcmaes-native-worker` (`fcmaes.job/0.1`) and `fcmaes-job-worker`
//! (`fcmaes.retry/1`), so both job profiles evaluate the same objectives with
//! the same BiteOpt and DE settings and retry seeds.
#![forbid(unsafe_code)]
use fcmaes_core::{BiteOpt, BiteParams, De, DeParams, Fitness, RetryBounds, RetryRunResult};
use optimization_job_api::legacy::JobSpec;

/// Runs one retry of the job's optimizer from `seed`.
pub fn solve(
    spec: &JobSpec,
    bounds: &RetryBounds,
    seed: u64,
    obj: &(impl Fn(&[f64]) -> f64 + Sync),
) -> RetryRunResult {
    if spec.optimizer.kind == "biteopt" {
        let mut solver = BiteOpt::new(
            bounds.lower(),
            bounds.upper(),
            None,
            &BiteParams {
                seed,
                runid: 0,
                popsize: spec.optimizer.parameters.population.unwrap() as i32,
                max_evaluations: spec.budget.evaluations_per_retry,
                ..Default::default()
            },
        );
        let r = solver.optimize(obj);
        RetryRunResult {
            x: r.x,
            y: r.y,
            evaluations: r.evaluations,
        }
    } else {
        let fit = Fitness::new(
            bounds.dim(),
            1,
            bounds.lower().to_vec(),
            bounds.upper().to_vec(),
        );
        let mut solver = De::new(
            fit,
            &[],
            &[],
            None,
            &DeParams {
                seed,
                runid: 0,
                popsize: spec.optimizer.parameters.population.unwrap() as i32,
                max_evaluations: spec.budget.evaluations_per_retry,
                ..Default::default()
            },
        );
        let r = solver.optimize(obj);
        RetryRunResult {
            x: r.x,
            y: r.y,
            evaluations: r.evaluations,
        }
    }
}
/// Sum of squares; minimum 0 at the origin.
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
/// Rosenbrock's valley; minimum 0 at (1, ..., 1).
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}
/// Resolve the objective once, so the evaluation loop does not dispatch on
/// the problem name for every candidate.
pub fn objective_for(spec: &JobSpec) -> fn(&[f64]) -> f64 {
    objective_by_id(&spec.problem.id)
}
/// The registered scalar objective with this identifier.
///
/// # Panics
///
/// Panics for an identifier that `Problem::resolve` would reject.
pub fn objective_by_id(id: &str) -> fn(&[f64]) -> f64 {
    match id {
        "benchmark.sphere" => sphere,
        "benchmark.rosenbrock" => rosenbrock,
        "gtop.cassini1" => fcmaes_gtop::cassini1,
        _ => unreachable!("problem id is validated by Problem::resolve"),
    }
}
