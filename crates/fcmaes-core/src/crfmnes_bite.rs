use std::sync::{Mutex, PoisonError};

use crate::{
    BiteParams, Crfmnes, CrfmnesParams, DeepBiteOpt, Fitness, NAN_REPLACEMENT, Objective,
    RetryBounds, RetryContext, RetryRunResult, Rng, validate_bite_inputs,
};

/// Settings for the opt-in [`crfmnes_bite`] retry sequence.
#[derive(Clone, Copy, Debug)]
pub struct CrfmnesBiteParams {
    /// Fraction of each retry's budget available to CR-FM-NES, in `0..=1`.
    /// Whole populations fit within this share. BiteOpt receives the balance.
    pub crfmnes_share: f64,
    /// Population size for both stages, at least four. CR-FM-NES rounds odd
    /// sizes up to preserve mirrored sampling.
    pub popsize: i32,
    /// Initial CR-FM-NES step size as a fraction of each variable's range.
    /// CR-FM-NES samples in normalized coordinates, so `0.1` is a step of a
    /// tenth of every interval whatever its width. On a unit box such as
    /// random keys in `[0, 1]` this equals a step in decision units. `None`
    /// draws a value in `[0.05, 0.10)` from the retry's deterministic seed.
    pub sigma: Option<f64>,
    /// Number of populations in DeepBiteOpt, from 1 to 36.
    pub bite_depth: i32,
    /// Stop after a finite evaluated objective reaches this value or lower.
    pub stop_fitness: f64,
}

impl Default for CrfmnesBiteParams {
    fn default() -> Self {
        Self {
            crfmnes_share: 0.30,
            popsize: 500,
            sigma: None,
            bite_depth: 4,
            stop_fitness: f64::NEG_INFINITY,
        }
    }
}

impl CrfmnesBiteParams {
    /// Check settings, bounds, and an optional initial point before starting a campaign.
    ///
    /// # Errors
    ///
    /// Returns an error for a share outside `[0, 1]`, an invalid step size,
    /// population, depth or target, unrepresentable box widths, or a guess
    /// with the wrong dimension or non-finite coordinates.
    pub fn validate(&self, bounds: &RetryBounds, guess: Option<&[f64]>) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.crfmnes_share) {
            return Err("crfmnes_share must be in [0, 1]".into());
        }
        if self
            .sigma
            .is_some_and(|sigma| !(2.0 * sigma).is_finite() || sigma <= 0.0)
        {
            return Err("sigma must be finite and positive".into());
        }
        if self.popsize < 4 || self.popsize == i32::MAX {
            return Err("popsize must be at least 4 and allow an even CR-FM-NES population".into());
        }
        if !(1..=36).contains(&self.bite_depth) {
            return Err("bite_depth must be from 1 to 36".into());
        }
        validate_bite_inputs(
            bounds.lower(),
            bounds.upper(),
            guess,
            &BiteParams {
                popsize: self.popsize,
                stop_fitness: self.stop_fitness,
                ..Default::default()
            },
            self.bite_depth,
        )
    }
}

/// Run one CR-FM-NES → BiteOpt sequence within a retry's hard evaluation budget.
///
/// Pass this to a [`crate::retry()`] closure to parallelize independent runs.
/// CR-FM-NES samples complete, even-sized populations. DeepBiteOpt starts at
/// the best point actually evaluated by the first stage and receives the
/// remaining calls, including any unused first-stage allowance. The returned
/// point and value describe the best finite observation across both stages.
/// With zero calls or no finite observation, the result contains the initial
/// point and positive infinity. A target can stop the run after a population
/// of CR-FM-NES or after an individual BiteOpt evaluation.
///
/// The CR-FM-NES step size is a fraction of each variable's range, so one
/// setting suits boxes of any width.
///
/// # Examples
///
/// ```
/// use fcmaes_core::{CrfmnesBiteParams, RetryBounds, RetryConfig, crfmnes_bite, retry};
///
/// let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
/// let bounds = RetryBounds::new(vec![-1.0; 4], vec![1.0; 4]).unwrap();
/// let settings = CrfmnesBiteParams { popsize: 16, ..Default::default() };
/// settings.validate(&bounds, None).unwrap();
/// let config = RetryConfig {
///     num_retries: 4,
///     workers: 2,
///     max_evaluations: 2_000,
///     seed: 42,
///     ..Default::default()
/// };
/// let result = retry(&sphere, &bounds, &config, |objective, context| {
///     crfmnes_bite(objective, context, &settings).unwrap()
/// });
/// assert_eq!(result.evaluations, 8_000);
/// assert!(result.y < 1e-6);
/// ```
///
/// # Errors
///
/// Returns an error if the objective is nonscalar or if the settings, bounds,
/// or initial guess fail [`CrfmnesBiteParams::validate`].
pub fn crfmnes_bite(
    objective: &impl Objective,
    context: &RetryContext,
    params: &CrfmnesBiteParams,
) -> Result<RetryRunResult, String> {
    crfmnes_bite_until(
        objective,
        &context.bounds,
        context.guess.as_deref(),
        context.run_seed,
        context.max_evaluations,
        params,
        &|| false,
    )
}

/// [`crfmnes_bite`] on explicit inputs that stops once `stop` returns true.
///
/// `seed` plays the role of [`RetryContext::run_seed`]: it fixes the initial
/// point without a `guess`, the drawn step size and both stages' streams.
/// `stop` is checked before every evaluation. Once it fires, the run makes
/// no further evaluation, skips BiteOpt if CR-FM-NES was running, and returns
/// the best finite observation so far. Use it for cancellation and deadlines.
///
/// # Errors
///
/// As [`crfmnes_bite`].
pub fn crfmnes_bite_until(
    objective: &impl Objective,
    bounds: &RetryBounds,
    guess: Option<&[f64]>,
    seed: u64,
    max_evaluations: u64,
    params: &CrfmnesBiteParams,
    stop: &dyn Fn() -> bool,
) -> Result<RetryRunResult, String> {
    if objective.nobj() != 1 {
        return Err("CR-FM-NES → BiteOpt requires a scalar objective".into());
    }
    params.validate(bounds, guess)?;
    let mut fit = Fitness::bounded(bounds.dim(), 1, bounds.lower(), bounds.upper());
    let mut rng = Rng::new(seed);
    let initial = match guess {
        Some(point) => fit.closest_feasible(point),
        None => fit.sample(&mut rng),
    };
    let sigma = params
        .sigma
        .unwrap_or_else(|| 0.05 + 0.05 * rng.uniform01());
    let nes_seed = rng.next_u64();
    let bite_seed = rng.next_u64();
    // Normalized coordinates map every interval to [-1, 1], a width of two.
    fit.set_normalize(true);
    let normalized_sigma = 2.0 * sigma;

    // CR-FM-NES ranks points with a bounds penalty, so record the decoded
    // points and their observed objective values before either solver updates.
    let best = Mutex::new(RetryRunResult {
        x: initial.clone(),
        y: f64::INFINITY,
        evaluations: 0,
    });
    let lock_best = || best.lock().unwrap_or_else(PoisonError::into_inner);
    let observe = |x: &[f64]| {
        let y = objective.eval_scalar(x);
        let mut best = lock_best();
        best.evaluations += 1;
        if y.is_finite() && y < best.y {
            best.x.copy_from_slice(x);
            best.y = y;
        }
        y
    };
    let reached_target = || {
        let y = lock_best().y;
        y.is_finite() && y <= params.stop_fitness
    };

    let population = params.popsize + params.popsize % 2;
    let first_cap = ((max_evaluations as f64 * params.crfmnes_share) as u64).min(max_evaluations);
    let generations = first_cap / population as u64;
    let mut stopped = false;
    if generations > 0 {
        let mut nes = Crfmnes::new(
            fit,
            &initial,
            normalized_sigma,
            &CrfmnesParams {
                popsize: population,
                max_evaluations: generations * population as u64,
                seed: nes_seed,
                ..Default::default()
            },
        );
        for _ in 0..generations {
            let points = nes.ask_pop();
            if points
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                break;
            }
            let mut values = Vec::with_capacity(points.len());
            for point in &points {
                if stop() {
                    stopped = true;
                    break;
                }
                let y = observe(point);
                values.push(if y.is_finite() { y } else { NAN_REPLACEMENT });
            }
            // A stopped population is not told: no stage runs after it.
            if stopped || reached_target() || nes.tell_pop(&values) != 0 {
                break;
            }
        }
    }

    let (remaining, start) = {
        let best = lock_best();
        (max_evaluations - best.evaluations, best.x.clone())
    };
    if !stopped && remaining > 0 && !reached_target() {
        let mut bite = DeepBiteOpt::new(
            bounds.lower(),
            bounds.upper(),
            Some(&start),
            &BiteParams {
                popsize: params.popsize,
                max_evaluations: remaining,
                seed: bite_seed,
                ..Default::default()
            },
            params.bite_depth,
        );
        bite.optimize_until(&observe, &|| reached_target() || stop());
    }
    Ok(best.into_inner().unwrap_or_else(PoisonError::into_inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RetryConfig, retry};

    fn evaluate_trace(
        budget: u64,
        params: &CrfmnesBiteParams,
        value: impl Fn(usize, &[f64]) -> f64 + Sync,
    ) -> (RetryRunResult, Vec<Vec<f64>>) {
        let bounds = RetryBounds::new(vec![0.0; 2], vec![1.0; 2]).unwrap();
        let trace = Mutex::new(Vec::new());
        let result = Mutex::new(None);
        retry(
            &|x: &[f64]| {
                assert!(x.iter().all(|x| (0.0..=1.0).contains(x)));
                let mut trace = trace.lock().unwrap();
                let y = value(trace.len(), x);
                trace.push(x.to_vec());
                y
            },
            &bounds,
            &RetryConfig {
                num_retries: 1,
                workers: 1,
                max_evaluations: budget,
                seed: 45,
                ..Default::default()
            },
            |objective, context| {
                let mut context = context.clone();
                context.guess = Some(vec![0.0; 2]);
                let run = crfmnes_bite(objective, &context, params).unwrap();
                *result.lock().unwrap() = Some(run.clone());
                run
            },
        );
        (
            result.into_inner().unwrap().unwrap(),
            trace.into_inner().unwrap(),
        )
    }

    #[test]
    fn evaluated_best_is_handed_off_and_unused_population_budget_is_spent() {
        let params = CrfmnesBiteParams {
            crfmnes_share: 0.5,
            popsize: 5,
            sigma: Some(3.0),
            ..Default::default()
        };
        // Eleven evaluations are allocated to CR-FM-NES. Its mirrored
        // population is six. Its raw minimum is in that first population.
        let (run, trace) =
            evaluate_trace(23, &params, |index, _| if index == 2 { 0.0 } else { 1.0 });
        assert_eq!(trace.len(), 23);
        assert_eq!(run.evaluations, 23);
        assert_eq!(run.y, 0.0);
        assert_eq!(run.x, trace[2]);
        assert!(
            trace[6]
                .iter()
                .zip(&run.x)
                .all(|(a, b)| (a - b).abs() <= 1e-12)
        );
    }

    #[test]
    fn small_and_empty_budgets_and_exact_targets() {
        let params = CrfmnesBiteParams {
            popsize: 8,
            stop_fitness: 1.0,
            ..Default::default()
        };
        let (run, trace) = evaluate_trace(3, &params, |_, _| 1.0);
        assert_eq!(trace.len(), 1);
        assert_eq!(run.evaluations, 1);
        let (run, trace) = evaluate_trace(30, &params, |_, _| 1.0);
        assert_eq!(trace.len(), 8);
        assert_eq!(run.evaluations, 8);
        let (run, trace) = evaluate_trace(0, &params, |_, _| panic!("unexpected evaluation"));
        assert!(trace.is_empty());
        assert_eq!(run.evaluations, 0);
        assert_eq!(run.y, f64::INFINITY);
    }

    #[test]
    fn nonfinite_values_do_not_become_the_incumbent() {
        let params = CrfmnesBiteParams {
            popsize: 4,
            ..Default::default()
        };
        let (run, trace) = evaluate_trace(20, &params, |_, _| f64::NAN);
        assert_eq!(trace.len(), 20);
        assert_eq!(run.evaluations, 20);
        assert_eq!(run.y, f64::INFINITY);
    }

    #[test]
    fn reproducible_for_each_retry_across_worker_counts() {
        let bounds = RetryBounds::new(vec![-2.0; 3], vec![2.0; 3]).unwrap();
        let params = CrfmnesBiteParams {
            popsize: 8,
            ..Default::default()
        };
        let run = |workers| {
            let records = Mutex::new(Vec::new());
            retry(
                &|x: &[f64]| x.iter().map(|x| x * x).sum(),
                &bounds,
                &RetryConfig {
                    num_retries: 4,
                    workers,
                    max_evaluations: 100,
                    seed: 93,
                    ..Default::default()
                },
                |objective, context| {
                    let run = crfmnes_bite(objective, context, &params).unwrap();
                    records.lock().unwrap().push((context.run_id, run.clone()));
                    run
                },
            );
            let mut records = records.into_inner().unwrap();
            records.sort_by_key(|entry| entry.0);
            records
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn a_stop_ends_the_run_before_the_next_evaluation() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let bounds = RetryBounds::new(vec![0.0; 2], vec![1.0; 2]).unwrap();
        let params = CrfmnesBiteParams {
            popsize: 8,
            ..Default::default()
        };
        // 100 calls: three CR-FM-NES populations (24 calls), then BiteOpt.
        for limit in [0, 5, 20, 24, 30, 60, 100] {
            let calls = AtomicU64::new(0);
            let run = crfmnes_bite_until(
                &|x: &[f64]| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    x.iter().sum()
                },
                &bounds,
                None,
                3,
                100,
                &params,
                &|| calls.load(Ordering::Relaxed) >= limit,
            )
            .unwrap();
            assert_eq!(run.evaluations, limit, "stop after {limit}");
            assert_eq!(calls.load(Ordering::Relaxed), limit);
            assert_eq!(run.y.is_finite(), limit > 0);
            assert!(run.x.iter().all(|x| (0.0..=1.0).contains(x)));
        }
    }

    /// One retry starting at the center of the box.
    fn single_context(bounds: RetryBounds, budget: u64) -> RetryContext {
        RetryContext {
            run_id: 0,
            run_seed: 7,
            seed: 7,
            guess: Some(
                bounds
                    .lower()
                    .iter()
                    .zip(bounds.upper())
                    .map(|(l, u)| 0.5 * (l + u))
                    .collect(),
            ),
            bounds,
            sdev: Vec::new(),
            max_evaluations: budget,
            value_limit: f64::INFINITY,
            crossover: false,
        }
    }

    #[test]
    fn crfmnes_steps_are_relative_to_each_range() {
        // The same retry on the unit box and on a box with very different
        // widths and offsets samples the same CR-FM-NES populations after
        // mapping back. The interior start and small step keep the points
        // inside the box, where CR-FM-NES's raw bound penalty is zero.
        let lower = [-5.0, 2.0, 100.0];
        let width = [10.0, 0.25, 1000.0];
        let unit = RetryBounds::new(vec![0.0; 3], vec![1.0; 3]).unwrap();
        let wide = RetryBounds::new(
            lower.to_vec(),
            lower.iter().zip(&width).map(|(l, w)| l + w).collect(),
        )
        .unwrap();
        let to_unit = |x: &[f64]| -> Vec<f64> {
            x.iter()
                .zip(&lower)
                .zip(&width)
                .map(|((x, l), w)| (x - l) / w)
                .collect()
        };
        let value = |u: &[f64]| -> f64 {
            u.iter()
                .enumerate()
                .map(|(i, v)| (i as f64 + 1.0) * (v - 0.3).powi(2))
                .sum()
        };
        let params = CrfmnesBiteParams {
            crfmnes_share: 1.0,
            popsize: 8,
            sigma: Some(0.02),
            ..Default::default()
        };
        let unit_trace = Mutex::new(Vec::new());
        let wide_trace = Mutex::new(Vec::new());
        crfmnes_bite(
            &|x: &[f64]| {
                unit_trace.lock().unwrap().push(x.to_vec());
                value(x)
            },
            &single_context(unit, 32),
            &params,
        )
        .unwrap();
        crfmnes_bite(
            &|x: &[f64]| {
                let u = to_unit(x);
                wide_trace.lock().unwrap().push(u.clone());
                value(&u)
            },
            &single_context(wide, 32),
            &params,
        )
        .unwrap();
        let unit_trace = unit_trace.into_inner().unwrap();
        let wide_trace = wide_trace.into_inner().unwrap();
        assert_eq!(unit_trace.len(), 32);
        assert_eq!(wide_trace.len(), 32);
        for (a, b) in unit_trace.iter().zip(&wide_trace) {
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-9));
        }
    }

    #[test]
    fn rejects_invalid_settings_and_vector_objectives() {
        let bounds = RetryBounds::new(vec![0.0; 2], vec![1.0; 2]).unwrap();
        let valid = CrfmnesBiteParams::default();
        assert!(valid.validate(&bounds, None).is_ok());
        assert!(valid.validate(&bounds, Some(&[0.5, 2.0])).is_ok());
        for invalid in [
            CrfmnesBiteParams {
                crfmnes_share: -0.1,
                ..valid
            },
            CrfmnesBiteParams {
                crfmnes_share: f64::NAN,
                ..valid
            },
            CrfmnesBiteParams {
                sigma: Some(0.0),
                ..valid
            },
            CrfmnesBiteParams {
                sigma: Some(f64::INFINITY),
                ..valid
            },
            CrfmnesBiteParams {
                sigma: Some(f64::MAX),
                ..valid
            },
            CrfmnesBiteParams {
                popsize: 3,
                ..valid
            },
            CrfmnesBiteParams {
                popsize: i32::MAX,
                ..valid
            },
            CrfmnesBiteParams {
                bite_depth: 0,
                ..valid
            },
            CrfmnesBiteParams {
                bite_depth: 37,
                ..valid
            },
            CrfmnesBiteParams {
                stop_fitness: f64::NAN,
                ..valid
            },
        ] {
            assert!(invalid.validate(&bounds, None).is_err(), "{invalid:?}");
        }
        assert!(valid.validate(&bounds, Some(&[0.5])).is_err());
        assert!(valid.validate(&bounds, Some(&[0.5, f64::NAN])).is_err());

        struct Pair;
        impl Objective for Pair {
            fn nobj(&self) -> usize {
                2
            }
            fn eval(&self, _: &[f64]) -> Vec<f64> {
                vec![0.0, 0.0]
            }
        }
        assert!(crfmnes_bite(&Pair, &single_context(bounds, 10), &valid).is_err());
    }
}
