use crate::{
    Error, MAX_BATCH, MAX_OBSERVATIONS, Observation, Result, Space, Value, design,
    gp::{self, ExactGp, Hyper, Kernel},
    special, stream_seed,
    transform::{OutputTransform, Transform},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acquisition {
    #[default]
    LogEi,
    Lcb,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub kernel: Kernel,
    pub acquisition: Acquisition,
    pub output_transform: OutputTransform,
    pub initial_points: usize,
    pub candidates: usize,
    pub fit_restarts: usize,
    /// L-BFGS iterations per MAP restart. Each accepted iteration costs one
    /// factorization plus one analytic gradient; rejected line-search trials
    /// cost a factorization only.
    pub fit_iterations: usize,
    pub refit_interval: usize,
    pub lcb_beta: f64,
    pub minimum_distance: f64,
    pub failure_threshold: f64,
    /// Diagonal nugget (variance, standardized outputs) added to the kernel
    /// matrix. Raise it for rough or numerically noisy simulations.
    pub nugget: f64,
    /// Search a TuRBO-style trust region around the incumbent (shaped by the
    /// lengthscales, resized by the success and failure counts of previous
    /// batches) for two of every three proposals and the whole box for the
    /// third. `false` searches the whole box for every proposal.
    pub trust_region: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            kernel: Kernel::Matern52,
            acquisition: Acquisition::LogEi,
            output_transform: OutputTransform::Standardize,
            initial_points: 8,
            candidates: 256,
            fit_restarts: 2,
            fit_iterations: 50,
            refit_interval: 3,
            lcb_beta: 2.0,
            minimum_distance: 1e-6,
            failure_threshold: 1e99,
            nugget: gp::DEFAULT_NUGGET,
            trust_region: true,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if !(1..=128).contains(&self.initial_points)
            || !(32..=4096).contains(&self.candidates)
            || !(1..=4).contains(&self.fit_restarts)
            || !(1..=gp::MAX_FIT_ITERATIONS).contains(&self.fit_iterations)
            || !(1..=64).contains(&self.refit_interval)
            || !self.lcb_beta.is_finite()
            || !(0.0..=20.0).contains(&self.lcb_beta)
            || !self.minimum_distance.is_finite()
            || !(1e-12..=0.01).contains(&self.minimum_distance)
            || !self.failure_threshold.is_finite()
            || self.failure_threshold <= 0.0
            || !gp::NUGGET_RANGE.contains(&self.nugget)
        {
            return Err(Error::Invalid(
                "invalid surrogate configuration or resource cap".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyState {
    pub round: u64,
    pub hyper: Option<Hyper>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust: Option<TrustRegion>,
}

/// TuRBO-1 trust-region state (Eriksson et al., 2019): side length in unit
/// coordinates before lengthscale shaping, consecutive successes/failures and
/// the incumbent value that defines the next success.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustRegion {
    pub length: f64,
    pub successes: u32,
    pub failures: u32,
    pub best: f64,
}

const TR_INITIAL: f64 = 0.8;
const TR_MIN: f64 = 0.0078125;
const TR_MAX: f64 = 1.6;
const TR_SUCCESSES: u32 = 3;
/// With a trust region, every third proposal maximizes the acquisition over
/// the whole box instead.
const GLOBAL_EVERY: u64 = 3;

impl TrustRegion {
    /// Updates the region with the incumbent after the previous batch.
    fn update(&mut self, best: f64, batch: usize, d: usize) {
        if best < self.best - 1e-3 * self.best.abs() {
            self.successes += 1;
            self.failures = 0;
        } else {
            self.successes = 0;
            self.failures += 1;
        }
        let failures = (4.0 / batch as f64).max(d as f64 / batch as f64).ceil() as u32;
        if self.successes >= TR_SUCCESSES {
            self.length = (2.0 * self.length).min(TR_MAX);
            self.successes = 0;
        } else if self.failures >= failures {
            self.length *= 0.5;
            self.failures = 0;
        }
        if self.length < TR_MIN {
            self.length = TR_INITIAL;
        }
        self.best = self.best.min(best);
    }

    /// Box around `centre`, side `length * w_i` with `w = l / geomean(l)`.
    fn bounds(&self, centre: &[f64], hyper: &Hyper) -> (Vec<f64>, Vec<f64>) {
        let d = centre.len() as f64;
        let mean_log = hyper
            .lengthscales
            .iter()
            .map(|l| libm::log(*l))
            .sum::<f64>()
            / d;
        let mut lower = Vec::with_capacity(centre.len());
        let mut upper = Vec::with_capacity(centre.len());
        for (c, l) in centre.iter().zip(&hyper.lengthscales) {
            let half = 0.5 * self.length * libm::exp(libm::log(*l) - mean_log);
            lower.push((c - half).max(0.0));
            upper.push((c + half).min(1.0));
        }
        (lower, upper)
    }
}

pub struct Round<'a> {
    pub space: &'a Space,
    pub history: &'a [Observation],
    pub pending: &'a [Vec<f64>],
    pub state: &'a StrategyState,
    pub config: &'a Config,
    pub seed: u64,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub points: Vec<Vec<f64>>,
    pub state: StrategyState,
    pub model_fits: usize,
    pub fit_evaluations: usize,
    pub fallback: Option<String>,
}

fn distinct(x: &[f64], occupied: &[Vec<f64>], distance: f64) -> bool {
    occupied
        .iter()
        .all(|p| design::squared_distance(x, p) >= distance * distance)
}
fn canonical(space: &Space, x: &[f64]) -> Result<Vec<f64>> {
    space.to_unit(&space.from_unit(x)?)
}

fn score(gp: &ExactGp, x: &[f64], best: f64, config: &Config) -> Result<f64> {
    let (mean, variance) = gp.posterior(x)?;
    Ok(match config.acquisition {
        Acquisition::LogEi => special::log_ei(best, mean, variance),
        Acquisition::Lcb => -mean + config.lcb_beta * libm::sqrt(variance),
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize(
    gp: &ExactGp,
    candidates: &[Vec<f64>],
    occupied: &[Vec<f64>],
    best: f64,
    config: &Config,
    lower: &[f64],
    upper: &[f64],
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Vec<f64>> {
    let scores: Vec<Result<f64>> = candidates
        .par_iter()
        .map(|x| {
            if stop() {
                return Err(Error::Stopped);
            }
            if !distinct(x, occupied, config.minimum_distance) {
                Ok(f64::NEG_INFINITY)
            } else {
                score(gp, x, best, config)
            }
        })
        .collect();
    let mut ranked = Vec::new();
    for (index, s) in scores.into_iter().enumerate() {
        ranked.push((index, s?));
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let starts: Vec<usize> = ranked
        .into_iter()
        .filter(|(_, s)| s.is_finite())
        .take(4)
        .map(|(i, _)| i)
        .collect();
    if starts.is_empty() {
        return Err(Error::Numerical(
            "no distinct finite acquisition candidates".into(),
        ));
    }
    let refined: Vec<Result<(Vec<f64>, f64)>> = starts
        .par_iter()
        .map(|&i| {
            let mut x = candidates[i].clone();
            let mut value = score(gp, &x, best, config)?;
            let mut step = 0.1;
            let mut trial = x.clone();
            for _ in 0..12 {
                let mut improved = false;
                for j in 0..x.len() {
                    let base = x[j];
                    for direction in [-1.0, 1.0] {
                        if stop() {
                            return Err(Error::Stopped);
                        }
                        trial[j] = (base + direction * step).clamp(lower[j], upper[j]);
                        if !distinct(&trial, occupied, config.minimum_distance) {
                            trial[j] = x[j];
                            continue;
                        }
                        let s = score(gp, &trial, best, config)?;
                        if s > value {
                            x[j] = trial[j];
                            value = s;
                            improved = true;
                        } else {
                            trial[j] = x[j];
                        }
                    }
                }
                if !improved {
                    step *= 0.5;
                }
                if step < 1e-4 {
                    break;
                }
            }
            Ok((x, value))
        })
        .collect();
    let mut answer = None;
    for result in refined {
        let (x, value) = result?;
        if answer.as_ref().is_none_or(|(_, s)| value > *s) {
            answer = Some((x, value));
        }
    }
    Ok(answer.expect("nonempty starts").0)
}

/// Pure proposal from explicit history, pending points, state and seed.
/// Collection and tie-breaking are independent of the installed pool size.
pub fn propose(round: &Round<'_>, stop: &(dyn Fn() -> bool + Sync)) -> Result<Proposal> {
    round.space.validate()?;
    round.config.validate()?;
    if round.count == 0
        || round.count > MAX_BATCH
        || round.history.len() + round.pending.len() + round.count > MAX_OBSERVATIONS
    {
        return Err(Error::Invalid(
            "proposal count or 256-observation cap".into(),
        ));
    }
    if stop() {
        return Err(Error::Stopped);
    }
    let mut ids = BTreeSet::new();
    let mut occupied = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for observation in round.history {
        if !ids.insert(observation.id) {
            return Err(Error::Invalid("duplicate observation id".into()));
        }
        let x = round.space.to_unit(&observation.x)?;
        occupied.push(x.clone());
        if let Value::Finite(y) = observation.value {
            if !y.is_finite() || y >= round.config.failure_threshold {
                return Err(Error::Invalid("unclassified observation value".into()));
            }
            xs.push(x);
            ys.push(y);
        }
    }
    for x in round.pending {
        occupied.push(round.space.to_unit(x)?);
    }
    let d = round.space.dimension();
    let seed = stream_seed(round.seed, 41, round.state.round);
    let mut state = round.state.clone();
    state.round = state
        .round
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("round overflow".into()))?;
    let mut model = None;
    let mut model_fits = 0;
    let mut fit_evaluations = 0;
    let mut fallback = None;
    let mut best = 0.0;
    if ys.len() >= 2 && round.history.len() >= round.config.initial_points {
        let transform = Transform::fit(&ys, round.config.output_transform)?;
        let transformed: Vec<f64> = ys.iter().map(|&y| transform.apply(y)).collect();
        best = transformed.iter().copied().fold(f64::INFINITY, f64::min);
        let fitted = if round.state.hyper.is_none()
            || round
                .state
                .round
                .is_multiple_of(round.config.refit_interval as u64)
        {
            model_fits = 1;
            gp::fit(
                gp::Fit {
                    xs: &xs,
                    ys: &transformed,
                    kernel: round.config.kernel,
                    warm: round.state.hyper.as_ref(),
                    seed,
                    restarts: round.config.fit_restarts,
                    iterations: round.config.fit_iterations,
                    nugget: round.config.nugget,
                },
                stop,
            )
            .map(|(gp, count)| {
                fit_evaluations = count;
                gp
            })
        } else {
            ExactGp::with_nugget(
                &xs,
                &transformed,
                round.config.kernel,
                round.state.hyper.clone().expect("checked hyper"),
                round.config.nugget,
            )
        };
        match fitted {
            Ok(mut gp) => {
                state.hyper = Some(gp.hyper.clone());
                for x in round.pending {
                    let x = round.space.to_unit(x)?;
                    let mean = gp.posterior(&x)?.0;
                    gp = gp.fantasize(x, mean)?;
                }
                model = Some(gp);
            }
            Err(Error::Numerical(s)) => {
                fallback = Some(s);
                state.hyper = None;
            }
            Err(e) => return Err(e),
        }
    }
    let mut ordered: Vec<usize> = (0..ys.len()).collect();
    ordered.sort_by(|&a, &b| ys[a].total_cmp(&ys[b]).then(a.cmp(&b)));
    let centres: Vec<Vec<f64>> = ordered.iter().take(3).map(|&i| xs[i].clone()).collect();
    let mut lower = vec![0.0; d];
    let mut upper = vec![1.0; d];
    if round.config.trust_region
        && let Some(gp) = &model
    {
        let incumbent = ys[ordered[0]];
        let region = match &round.state.trust {
            Some(previous) => {
                let mut region = previous.clone();
                region.update(incumbent, round.count, d);
                region
            }
            None => TrustRegion {
                length: TR_INITIAL,
                successes: 0,
                failures: 0,
                best: incumbent,
            },
        };
        (lower, upper) = region.bounds(&xs[ordered[0]], &gp.hyper);
        state.trust = Some(region);
    }
    // Candidates are generated in the unit cube of the search box.
    let boxed = |lower: &[f64], upper: &[f64]| -> Result<Vec<Vec<f64>>> {
        let inside: Vec<Vec<f64>> = centres
            .iter()
            .filter(|c| {
                c.iter()
                    .zip(lower)
                    .zip(upper)
                    .all(|((c, l), u)| (l..=u).contains(&c))
            })
            .map(|c| {
                c.iter()
                    .zip(lower)
                    .zip(upper)
                    .map(|((c, l), u)| ((c - l) / (u - l)).clamp(0.0, 1.0))
                    .collect()
            })
            .collect();
        let mut candidates = design::candidates(round.config.candidates, d, seed, &inside);
        for x in &mut candidates {
            for ((v, l), u) in x.iter_mut().zip(lower).zip(upper) {
                *v = (l + *v * (u - l)).clamp(*l, *u);
            }
            *x = canonical(round.space, x)?;
        }
        Ok(candidates)
    };
    let candidates = boxed(&lower, &upper)?;
    let global_lower = vec![0.0; d];
    let global_upper = vec![1.0; d];
    let global = if state.trust.is_some() && round.config.trust_region {
        boxed(&global_lower, &global_upper)?
    } else {
        Vec::new()
    };

    let initial = design::initial_design(round.config.initial_points, d, round.seed);
    let mut points = Vec::new();
    for i in 0..round.count {
        if stop() {
            return Err(Error::Stopped);
        }
        let index = round.history.len() + round.pending.len() + i;
        let mut unit = None;
        if index < initial.len() {
            let candidate = canonical(round.space, &initial[index])?;
            if distinct(&candidate, &occupied, round.config.minimum_distance) {
                unit = Some(candidate);
            }
        }
        if unit.is_none()
            && let Some(gp) = &model
        {
            // Every GLOBAL_EVERY-th proposal searches the whole box, so a
            // region around a local minimum cannot hold the run forever.
            let slot = round.state.round * round.count as u64 + i as u64;
            let use_global = !global.is_empty() && slot % GLOBAL_EVERY == GLOBAL_EVERY - 1;
            let (cands, lo, hi) = if use_global {
                (&global, &global_lower, &global_upper)
            } else {
                (&candidates, &lower, &upper)
            };
            let candidate = canonical(
                round.space,
                &optimize(gp, cands, &occupied, best, round.config, lo, hi, stop)?,
            )?;
            if distinct(&candidate, &occupied, round.config.minimum_distance) {
                unit = Some(candidate);
            }
        }
        if unit.is_none() {
            // Deterministic space-filling fallback, also used before a GP exists.
            let mut distance = -1.0;
            for candidate in &candidates {
                let nearest = occupied
                    .iter()
                    .map(|x| design::squared_distance(candidate, x))
                    .fold(f64::INFINITY, f64::min);
                if nearest >= round.config.minimum_distance.powi(2) && nearest > distance {
                    distance = nearest;
                    unit = Some(candidate.clone());
                }
            }
        }
        let unit = unit.ok_or_else(|| {
            Error::Numerical("candidate space exhausted at the configured minimum distance".into())
        })?;
        points.push(round.space.from_unit(&unit)?);
        occupied.push(unit.clone());
        if i + 1 < round.count
            && let Some(gp) = &model
        {
            let mean = gp.posterior(&unit)?.0;
            model = Some(gp.fantasize(unit, mean)?);
        }
    }
    Ok(Proposal {
        points,
        state,
        model_fits,
        fit_evaluations,
        fallback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_region_expands_shrinks_and_restarts() {
        let mut region = TrustRegion {
            length: TR_INITIAL,
            successes: 0,
            failures: 0,
            best: 10.0,
        };
        for k in 1..=3 {
            region.update(10.0 - k as f64, 1, 2);
        }
        assert_eq!(
            (region.length, region.successes, region.best),
            (1.6, 0, 7.0)
        );
        region.update(6.0, 1, 2);
        region.update(5.0, 1, 2);
        region.update(4.0, 1, 2);
        assert_eq!(region.length, TR_MAX, "expansion is capped");
        // Sequential, d=2: four failures halve the region.
        for _ in 0..3 {
            region.update(4.0, 1, 2);
        }
        assert_eq!((region.length, region.failures), (TR_MAX, 3));
        region.update(4.0 - 1e-6, 1, 2);
        assert_eq!((region.length, region.failures), (0.8, 0));
        // Batches of four in d=10 halve after ceil(10/4) = 3 failures.
        for _ in 0..3 {
            region.update(4.0, 4, 10);
        }
        assert_eq!(region.length, 0.4);
        region.length = TR_MIN;
        for _ in 0..3 {
            region.update(4.0, 4, 10);
        }
        assert_eq!(region.length, TR_INITIAL, "a collapsed region restarts");
    }

    #[test]
    fn trust_region_box_follows_lengthscales_and_stays_in_the_cube() {
        let region = TrustRegion {
            length: 0.8,
            successes: 0,
            failures: 0,
            best: 0.0,
        };
        let (lower, upper) = region.bounds(
            &[0.5, 0.05],
            &Hyper {
                lengthscales: vec![0.1, 0.4],
            },
        );
        for (a, b) in lower
            .iter()
            .zip([0.3, 0.0])
            .chain(upper.iter().zip([0.7, 0.85]))
        {
            assert!((a - b).abs() < 1e-12, "{lower:?} {upper:?}");
        }
    }

    #[test]
    fn two_of_three_proposals_lie_in_the_trust_region() {
        let space = Space::new(vec![-1.0; 3], vec![2.0; 3], vec![false; 3]).unwrap();
        let history: Vec<Observation> = design::latin_hypercube(12, 3, 9)
            .into_iter()
            .enumerate()
            .map(|(i, u)| {
                let x = space.from_unit(&u).unwrap();
                let y = x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum::<f64>();
                Observation {
                    id: i as u64,
                    x,
                    value: Value::Finite(y),
                }
            })
            .collect();
        let config = Config {
            candidates: 64,
            ..Config::default()
        };
        let state = StrategyState::default();
        let proposal = propose(
            &Round {
                space: &space,
                history: &history,
                pending: &[],
                state: &state,
                config: &config,
                seed: 3,
                count: 3,
            },
            &|| false,
        )
        .unwrap();
        let region = proposal.state.trust.clone().expect("region");
        let best = history
            .iter()
            .min_by(|a, b| match (&a.value, &b.value) {
                (Value::Finite(a), Value::Finite(b)) => a.total_cmp(b),
                _ => std::cmp::Ordering::Equal,
            })
            .unwrap();
        let (lower, upper) = region.bounds(
            &space.to_unit(&best.x).unwrap(),
            proposal.state.hyper.as_ref().unwrap(),
        );
        for point in &proposal.points[..2] {
            let u = space.to_unit(point).unwrap();
            for ((v, l), h) in u.iter().zip(&lower).zip(&upper) {
                assert!(
                    *v >= l - 1e-12 && *v <= h + 1e-12,
                    "{u:?} {lower:?} {upper:?}"
                );
            }
        }
        let off = Config {
            trust_region: false,
            ..config
        };
        let global = propose(
            &Round {
                space: &space,
                history: &history,
                pending: &[],
                state: &state,
                config: &off,
                seed: 3,
                count: 3,
            },
            &|| false,
        )
        .unwrap();
        assert!(global.state.trust.is_none());
    }
}
