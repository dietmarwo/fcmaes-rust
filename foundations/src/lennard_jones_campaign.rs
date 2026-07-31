//! Reproducible Lennard-Jones scaling campaign.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fcmaes_core::{
    BiteOpt, BiteParams, Cmaes, CmaesParams, Crfmnes, CrfmnesParams, De, DeParams, Fitness,
    RetryBounds, RetryConfig, RetryRunResult, parallel_batch, retry, retry_run_seed,
};
use serde::Serialize;
use serde_json::json;

use crate::artifacts::{write_json, write_text};
use crate::suites::Suite;
use crate::suites::lennard_jones::{
    LennardJones, Parameterization, REFERENCE_SOURCE, SUCCESS_TOLERANCE,
};

const DERIVATIVE_FREE_RETRIES: usize = 4;
const CRFMNES_SIGMA: f64 = 0.05;
const CRFMNES_POPULATION: i32 = 16;
const CRFMNES_CHECK_SIGMAS: [f64; 3] = [0.05, 0.15, 0.50];
const CRFMNES_CHECK_POPULATIONS: [i32; 3] = [16, 32, 64];

/// Lennard-Jones campaign size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LjPreset {
    /// Fast model/adapter conformance run; not ranking evidence.
    Smoke,
    /// Ten-seed scaling protocol.
    Publication,
}

impl LjPreset {
    /// Parse `smoke` or `publication`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "publication" => Some(Self::Publication),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Publication => "publication",
        }
    }

    fn atoms(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[13, 38],
            Self::Publication => &[13, 38, 55, 75, 98],
        }
    }

    fn seeds(self) -> usize {
        match self {
            Self::Smoke => 2,
            Self::Publication => 10,
        }
    }

    fn pair_budget(self) -> u64 {
        match self {
            Self::Smoke => 600,
            Self::Publication => 20_000,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ScalingRow {
    n_atoms: usize,
    dimension: usize,
    parameterization: String,
    optimizer: String,
    seed: u64,
    best_energy: f64,
    target_energy: f64,
    gap: f64,
    target_relative_gap: f64,
    success: bool,
    within_1_percent: bool,
    within_5_percent: bool,
    within_10_percent: bool,
    population_size: Option<usize>,
    nominal_full_generations_per_restart: Option<u64>,
    restarts: Option<usize>,
    pair_traversals: u64,
    pair_terms_evaluated: u64,
    objective_calls: u64,
    gradient_calls: u64,
    wall_seconds: f64,
    measured_pair_seconds: f64,
    estimated_optimizer_overhead_seconds: f64,
    overlap_pairs: u64,
    projected_candidates: u64,
    reference_structure_audited: bool,
    best_decision: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct ReferenceAuditRow {
    n_atoms: usize,
    coordinate_source: String,
    local_file_name: String,
    coordinate_sha256: String,
    measured_energy: f64,
    target_energy: f64,
    absolute_error: f64,
    tolerance: f64,
    matches: bool,
}

#[derive(Debug)]
struct Best {
    energy: f64,
    decision: Vec<f64>,
}

struct Meter<'a> {
    problem: &'a LennardJones,
    budget: u64,
    pair_traversals: AtomicU64,
    objective_calls: AtomicU64,
    pair_nanoseconds: AtomicU64,
    overlap_pairs: AtomicU64,
    projected_candidates: AtomicU64,
    lower: Vec<f64>,
    upper: Vec<f64>,
    best: Mutex<Best>,
}

impl<'a> Meter<'a> {
    fn new(problem: &'a LennardJones, budget: u64) -> Self {
        let (lower, upper) = problem.bounds();
        Self {
            problem,
            budget,
            pair_traversals: AtomicU64::new(0),
            objective_calls: AtomicU64::new(0),
            pair_nanoseconds: AtomicU64::new(0),
            overlap_pairs: AtomicU64::new(0),
            projected_candidates: AtomicU64::new(0),
            lower,
            upper,
            best: Mutex::new(Best {
                energy: f64::INFINITY,
                decision: Vec::new(),
            }),
        }
    }

    fn evaluate(&self, decision: &[f64]) -> f64 {
        self.objective_calls.fetch_add(1, Ordering::Relaxed);
        let next = self.pair_traversals.fetch_add(1, Ordering::Relaxed);
        if next >= self.budget {
            self.pair_traversals.fetch_sub(1, Ordering::Relaxed);
            return 1.0e12;
        }
        let mut projected = decision.to_vec();
        let mut changed = false;
        for ((value, &lower), &upper) in projected.iter_mut().zip(&self.lower).zip(&self.upper) {
            let bounded = value.clamp(lower, upper);
            changed |= bounded != *value;
            *value = bounded;
        }
        if changed {
            self.projected_candidates.fetch_add(1, Ordering::Relaxed);
        }
        let started = Instant::now();
        let evaluation = self
            .problem
            .energy(&projected)
            .expect("explicit projection must satisfy Lennard-Jones bounds");
        self.pair_nanoseconds.fetch_add(
            started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        self.overlap_pairs
            .fetch_add(evaluation.overlap_pairs as u64, Ordering::Relaxed);
        let mut best = self.best.lock().unwrap_or_else(|error| error.into_inner());
        if evaluation.energy < best.energy {
            best.energy = evaluation.energy;
            best.decision = projected;
        }
        evaluation.energy
    }

    fn metrics(&self) -> (f64, Vec<f64>, u64, u64, f64, u64, u64) {
        let best = self.best.lock().unwrap_or_else(|error| error.into_inner());
        (
            best.energy,
            best.decision.clone(),
            self.pair_traversals.load(Ordering::Relaxed),
            self.objective_calls.load(Ordering::Relaxed),
            self.pair_nanoseconds.load(Ordering::Relaxed) as f64 * 1.0e-9,
            self.overlap_pairs.load(Ordering::Relaxed),
            self.projected_candidates.load(Ordering::Relaxed),
        )
    }
}

struct ArmResult {
    best_energy: f64,
    best_decision: Vec<f64>,
    pair_traversals: u64,
    objective_calls: u64,
    gradient_calls: u64,
    wall_seconds: f64,
    measured_pair_seconds: f64,
    overlap_pairs: u64,
    projected_candidates: u64,
}

fn arm_result(meter: &Meter<'_>, started: Instant) -> ArmResult {
    let (
        best_energy,
        best_decision,
        pair_traversals,
        objective_calls,
        pair_seconds,
        overlaps,
        projected_candidates,
    ) = meter.metrics();
    ArmResult {
        best_energy,
        best_decision,
        pair_traversals,
        objective_calls,
        gradient_calls: 0,
        wall_seconds: started.elapsed().as_secs_f64(),
        measured_pair_seconds: pair_seconds,
        overlap_pairs: overlaps,
        projected_candidates,
    }
}

fn random_arm(problem: &LennardJones, budget: u64, seed: u64) -> Result<ArmResult, String> {
    let meter = Meter::new(problem, budget);
    let started = Instant::now();
    for sample in 0..budget {
        let sample_seed = retry_run_seed(seed, sample as usize);
        let decision = problem
            .initial_decision(sample_seed)
            .map_err(|error| error.to_string())?;
        meter.evaluate(&decision);
    }
    Ok(arm_result(&meter, started))
}

#[derive(Clone, Copy)]
enum DerivativeFreeArm {
    De,
    Cma,
    Crfmnes { sigma: f64, population: i32 },
    Bite,
}

fn retry_arm(
    problem: &LennardJones,
    budget: u64,
    seed: u64,
    arm: DerivativeFreeArm,
) -> Result<ArmResult, String> {
    let meter = Meter::new(problem, budget);
    let started = Instant::now();
    let (lower, upper) = problem.bounds();
    let retry_bounds = RetryBounds::new(lower, upper).map_err(str::to_owned)?;
    let retries = DERIVATIVE_FREE_RETRIES;
    let config = RetryConfig {
        num_retries: retries,
        workers: 1,
        capacity: retries,
        max_evaluations: budget / retries as u64,
        seed,
        ..Default::default()
    };
    let objective = |decision: &[f64]| meter.evaluate(decision);
    let result = retry(&objective, &retry_bounds, &config, |objective, context| {
        let guess = problem
            .initial_decision(context.run_seed)
            .expect("tested compact initializer must produce a bounded point");
        match arm {
            DerivativeFreeArm::De => {
                // DE consumes the supplied normal-sampling mean directly in
                // its working coordinates. Keep it in real coordinates so
                // the compact real-space start is not decoded a second time.
                let fitness = Fitness::bounded(
                    problem.dimension(),
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                let result = De::new(
                    fitness,
                    &guess,
                    &vec![0.15; problem.dimension()],
                    None,
                    &DeParams {
                        max_evaluations: context.max_evaluations,
                        seed: context.run_seed,
                        ..Default::default()
                    },
                )
                .optimize(objective);
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
            DerivativeFreeArm::Cma => {
                let mut fitness = Fitness::bounded(
                    problem.dimension(),
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                fitness.set_normalize(true);
                let result = Cmaes::new(
                    fitness,
                    &guess,
                    &[0.15],
                    &CmaesParams {
                        max_evaluations: context.max_evaluations,
                        seed: context.run_seed,
                        ..Default::default()
                    },
                )
                .optimize(objective, 1);
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
            DerivativeFreeArm::Crfmnes { sigma, population } => {
                let mut fitness = Fitness::bounded(
                    problem.dimension(),
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                fitness.set_normalize(true);
                let mut optimizer = Crfmnes::new(
                    fitness,
                    &guess,
                    sigma,
                    &CrfmnesParams {
                        popsize: population,
                        max_evaluations: context.max_evaluations,
                        seed: context.run_seed,
                        ..Default::default()
                    },
                );
                let result = optimizer.optimize_batch(|population| {
                    population
                        .iter()
                        .map(|decision| objective(decision))
                        .collect()
                });
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
            DerivativeFreeArm::Bite => {
                let result = BiteOpt::new(
                    context.bounds.lower(),
                    context.bounds.upper(),
                    Some(&guess),
                    &BiteParams {
                        max_evaluations: context.max_evaluations,
                        seed: context.run_seed,
                        ..Default::default()
                    },
                )
                .optimize(objective);
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
        }
    });
    if !result.success {
        return Err("derivative-free retry returned no finite candidate".to_owned());
    }
    Ok(arm_result(&meter, started))
}

fn de_arm(problem: &LennardJones, budget: u64, seed: u64) -> Result<ArmResult, String> {
    retry_arm(problem, budget, seed, DerivativeFreeArm::De)
}

fn cma_arm(problem: &LennardJones, budget: u64, seed: u64) -> Result<ArmResult, String> {
    retry_arm(problem, budget, seed, DerivativeFreeArm::Cma)
}

fn crfmnes_arm(problem: &LennardJones, budget: u64, seed: u64) -> Result<ArmResult, String> {
    retry_arm(
        problem,
        budget,
        seed,
        DerivativeFreeArm::Crfmnes {
            sigma: CRFMNES_SIGMA,
            population: CRFMNES_POPULATION,
        },
    )
}

fn bite_arm(problem: &LennardJones, budget: u64, seed: u64) -> Result<ArmResult, String> {
    retry_arm(problem, budget, seed, DerivativeFreeArm::Bite)
}

#[cfg(feature = "gradient-reference")]
mod gradient {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use argmin::core::{CostFunction, Error, Executor, Gradient};
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;
    use fcmaes_core::{Rng, retry_run_seed};

    use super::{ArmResult, LennardJones, Suite};

    struct StateData {
        pair_traversals: u64,
        objective_calls: u64,
        gradient_calls: u64,
        pair_seconds: f64,
        overlap_pairs: u64,
        best_energy: f64,
        best_decision: Vec<f64>,
        cache_z: Vec<f64>,
        cache_energy: f64,
        cache_gradient: Vec<f64>,
    }

    struct Shared<'a> {
        problem: &'a LennardJones,
        lower: Vec<f64>,
        upper: Vec<f64>,
        budget: u64,
        state: Mutex<StateData>,
    }

    #[derive(Clone)]
    struct ArgminProblem<'a> {
        shared: Arc<Shared<'a>>,
    }

    impl ArgminProblem<'_> {
        fn decode(&self, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let mut decision = Vec::with_capacity(z.len());
            let mut derivative = Vec::with_capacity(z.len());
            for ((&value, &lower), &upper) in
                z.iter().zip(&self.shared.lower).zip(&self.shared.upper)
            {
                let scaled = value.tanh();
                let half = 0.5 * (upper - lower);
                decision.push(0.5 * (upper + lower) + half * scaled);
                derivative.push(half * (1.0 - scaled * scaled));
            }
            (decision, derivative)
        }

        fn evaluate(&self, z: &[f64]) -> (f64, Vec<f64>) {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.cache_z == z {
                return (state.cache_energy, state.cache_gradient.clone());
            }
            if state.pair_traversals >= self.shared.budget {
                return (1.0e12, vec![0.0; z.len()]);
            }
            let (decision, chain) = self.decode(z);
            let started = Instant::now();
            let (evaluation, mut gradient) = self
                .shared
                .problem
                .value_gradient(&decision)
                .expect("tanh transform must stay inside bounds");
            state.pair_seconds += started.elapsed().as_secs_f64();
            state.pair_traversals += 1;
            state.overlap_pairs += evaluation.overlap_pairs as u64;
            for (component, factor) in gradient.iter_mut().zip(chain) {
                *component *= factor;
            }
            if evaluation.energy < state.best_energy {
                state.best_energy = evaluation.energy;
                state.best_decision = decision;
            }
            state.cache_z = z.to_vec();
            state.cache_energy = evaluation.energy;
            state.cache_gradient.clone_from(&gradient);
            (evaluation.energy, gradient)
        }
    }

    impl CostFunction for ArgminProblem<'_> {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, parameter: &Self::Param) -> Result<Self::Output, Error> {
            self.shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .objective_calls += 1;
            Ok(self.evaluate(parameter).0)
        }
    }

    impl Gradient for ArgminProblem<'_> {
        type Param = Vec<f64>;
        type Gradient = Vec<f64>;

        fn gradient(&self, parameter: &Self::Param) -> Result<Self::Gradient, Error> {
            self.shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .gradient_calls += 1;
            Ok(self.evaluate(parameter).1)
        }
    }

    fn unconstrained(decision: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        decision
            .iter()
            .zip(lower)
            .zip(upper)
            .map(|((&value, &low), &high)| {
                let scaled =
                    (2.0 * (value - low) / (high - low) - 1.0).clamp(-0.999_999, 0.999_999);
                scaled.atanh()
            })
            .collect()
    }

    fn local_minimize(problem: ArgminProblem<'_>, start: &[f64], iterations: u64) {
        let initial = unconstrained(start, &problem.shared.lower, &problem.shared.upper);
        let linesearch = MoreThuenteLineSearch::new();
        let solver: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 10);
        let _ = Executor::new(problem, solver)
            .configure(|state| state.param(initial).max_iters(iterations))
            .ctrlc(false)
            .run();
    }

    fn shared(problem: &LennardJones, budget: u64) -> Arc<Shared<'_>> {
        let (lower, upper) = problem.bounds();
        Arc::new(Shared {
            problem,
            lower,
            upper,
            budget,
            state: Mutex::new(StateData {
                pair_traversals: 0,
                objective_calls: 0,
                gradient_calls: 0,
                pair_seconds: 0.0,
                overlap_pairs: 0,
                best_energy: f64::INFINITY,
                best_decision: Vec::new(),
                cache_z: Vec::new(),
                cache_energy: f64::INFINITY,
                cache_gradient: Vec::new(),
            }),
        })
    }

    fn finish(shared: Arc<Shared<'_>>, started: Instant) -> ArmResult {
        let state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ArmResult {
            best_energy: state.best_energy,
            best_decision: state.best_decision.clone(),
            pair_traversals: state.pair_traversals,
            objective_calls: state.objective_calls,
            gradient_calls: state.gradient_calls,
            wall_seconds: started.elapsed().as_secs_f64(),
            measured_pair_seconds: state.pair_seconds,
            overlap_pairs: state.overlap_pairs,
            projected_candidates: 0,
        }
    }

    pub(super) fn multistart(
        problem: &LennardJones,
        budget: u64,
        seed: u64,
    ) -> Result<ArmResult, String> {
        let shared = shared(problem, budget);
        let started = Instant::now();
        let starts = 4_u64;
        for run in 0..starts {
            if shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pair_traversals
                >= budget
            {
                break;
            }
            let decision = problem
                .initial_decision(retry_run_seed(seed, run as usize))
                .map_err(|error| error.to_string())?;
            local_minimize(
                ArgminProblem {
                    shared: Arc::clone(&shared),
                },
                &decision,
                budget / starts,
            );
        }
        Ok(finish(shared, started))
    }

    pub(super) fn basin_hopping(
        problem: &LennardJones,
        budget: u64,
        seed: u64,
    ) -> Result<ArmResult, String> {
        let shared = shared(problem, budget);
        let started = Instant::now();
        let hops = 5_u64;
        let mut current = problem
            .initial_decision(seed)
            .map_err(|error| error.to_string())?;
        let mut rng = Rng::new(seed ^ 0x4241_5349_4e48_4f50);
        for _ in 0..hops {
            local_minimize(
                ArgminProblem {
                    shared: Arc::clone(&shared),
                },
                &current,
                budget / hops,
            );
            let state = shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !state.best_decision.is_empty() {
                current.clone_from(&state.best_decision);
            }
            drop(state);
            for ((value, &lower), &upper) in
                current.iter_mut().zip(&shared.lower).zip(&shared.upper)
            {
                *value = (*value + 0.08 * (upper - lower) * rng.gaussian()).clamp(lower, upper);
            }
        }
        Ok(finish(shared, started))
    }
}

#[cfg(feature = "gradient-reference")]
fn gradient_arms(
    problem: &LennardJones,
    budget: u64,
    seed: u64,
) -> Result<Vec<(&'static str, ArmResult)>, String> {
    Ok(vec![
        (
            "lbfgs-multistart",
            gradient::multistart(problem, budget, seed)?,
        ),
        (
            "basin-hopping",
            gradient::basin_hopping(problem, budget, seed)?,
        ),
    ])
}

#[cfg(not(feature = "gradient-reference"))]
fn gradient_arms(
    _problem: &LennardJones,
    _budget: u64,
    _seed: u64,
) -> Result<Vec<(&'static str, ArmResult)>, String> {
    Err("Lennard-Jones campaigns require --features gradient-reference so the mandatory argmin L-BFGS controls are present".to_owned())
}

fn arm_protocol(
    optimizer: &str,
    dimension: usize,
    budget: u64,
) -> (Option<usize>, Option<u64>, Option<usize>) {
    let population = match optimizer {
        "de-retry" => Some(15 * dimension),
        "cma-retry" => Some((4.0 + 3.0 * (dimension as f64).ln()).floor() as usize),
        "crfmnes-retry" => Some(CRFMNES_POPULATION as usize),
        _ => None,
    };
    let restarts = match optimizer {
        "de-retry" | "cma-retry" | "crfmnes-retry" | "bite-retry" => Some(DERIVATIVE_FREE_RETRIES),
        "lbfgs-multistart" => Some(4),
        "basin-hopping" => Some(5),
        _ => None,
    };
    let generations = population
        .zip(restarts)
        .map(|(population, restarts)| (budget / restarts as u64) / population as u64);
    (population, generations, restarts)
}

fn audit_references(preset: LjPreset, directory: &Path) -> Result<Vec<ReferenceAuditRow>, String> {
    let mut rows = Vec::with_capacity(preset.atoms().len());
    for &atoms in preset.atoms() {
        let file_name = atoms.to_string();
        let path = directory.join(&file_name);
        let problem =
            LennardJones::new(atoms, Parameterization::Free).map_err(|error| error.to_string())?;
        let audit = problem
            .audit_reference(&path)
            .map_err(|error| format!("LJ{atoms} reference audit failed: {error}"))?;
        if !audit.matches {
            return Err(format!(
                "LJ{atoms} reference energy differs from its source-cited target by {:.3e}",
                audit.absolute_error
            ));
        }
        rows.push(ReferenceAuditRow {
            n_atoms: atoms,
            coordinate_source: format!(
                "{}points/{atoms}",
                REFERENCE_SOURCE.trim_end_matches("tables.150.html")
            ),
            local_file_name: file_name,
            coordinate_sha256: audit.coordinate_sha256,
            measured_energy: audit.measured_energy,
            target_energy: audit.target_energy,
            absolute_error: audit.absolute_error,
            tolerance: 1.0e-6,
            matches: audit.matches,
        });
    }
    Ok(rows)
}

#[derive(Clone, Debug, Serialize)]
struct CrfmnesConfigurationRow {
    n_atoms: usize,
    dimension: usize,
    seed: u64,
    sigma_normalized: f64,
    population_size: i32,
    primary_configuration: bool,
    best_energy: f64,
    target_energy: f64,
    gap: f64,
    target_relative_gap: f64,
    pair_traversals: u64,
    wall_seconds: f64,
}

fn crfmnes_configuration_check(
    root_seed: u64,
    workers: i32,
    budget: u64,
) -> Result<Vec<CrfmnesConfigurationRow>, String> {
    let mut cases = Vec::new();
    for (size_index, atoms) in [13, 38, 98].into_iter().enumerate() {
        for (sigma_index, sigma) in CRFMNES_CHECK_SIGMAS.into_iter().enumerate() {
            for (population_index, population) in CRFMNES_CHECK_POPULATIONS.into_iter().enumerate()
            {
                for seed_index in 0..3 {
                    cases.push((
                        size_index,
                        sigma_index,
                        population_index,
                        atoms,
                        sigma,
                        population,
                        seed_index,
                    ));
                }
            }
        }
    }
    let results = parallel_batch(&cases, workers, |case| {
        let (size_index, sigma_index, population_index, atoms, sigma, population, seed_index) =
            *case;
        let problem = LennardJones::new(atoms, Parameterization::FixedFrame)
            .map_err(|error| error.to_string())?;
        let target = problem
            .target()
            .ok_or_else(|| format!("missing target for LJ{atoms}"))?;
        let case_id = (((size_index * 3 + sigma_index) * 3 + population_index) * 3) + seed_index;
        let seed = retry_run_seed(root_seed ^ 0x4352_464d_4348_454b, case_id);
        let result = retry_arm(
            &problem,
            budget,
            seed,
            DerivativeFreeArm::Crfmnes { sigma, population },
        )?;
        let gap = result.best_energy - target.energy;
        Ok::<CrfmnesConfigurationRow, String>(CrfmnesConfigurationRow {
            n_atoms: atoms,
            dimension: problem.dimension(),
            seed,
            sigma_normalized: sigma,
            population_size: population,
            primary_configuration: sigma == CRFMNES_SIGMA && population == CRFMNES_POPULATION,
            best_energy: result.best_energy,
            target_energy: target.energy,
            gap,
            target_relative_gap: gap.max(0.0) / target.energy.abs(),
            pair_traversals: result.pair_traversals,
            wall_seconds: result.wall_seconds,
        })
    });
    let mut rows = Vec::with_capacity(cases.len());
    for result in results {
        rows.push(result?);
    }
    rows.sort_by(|left, right| {
        left.n_atoms
            .cmp(&right.n_atoms)
            .then(left.sigma_normalized.total_cmp(&right.sigma_normalized))
            .then(left.population_size.cmp(&right.population_size))
            .then(left.seed.cmp(&right.seed))
    });
    Ok(rows)
}

fn write_scaling_csv(path: &Path, rows: &[ScalingRow]) -> Result<(), String> {
    let mut output = String::from(
        "n_atoms,dimension,parameterization,optimizer,seed,best_energy,target_energy,gap,target_relative_gap,success,within_1_percent,within_5_percent,within_10_percent,population_size,nominal_full_generations_per_restart,restarts,pair_traversals,pair_terms_evaluated,objective_calls,gradient_calls,wall_seconds,measured_pair_seconds,estimated_optimizer_overhead_seconds,overlap_pairs,projected_candidates,reference_structure_audited\n",
    );
    for row in rows {
        let optional = |value: Option<usize>| value.map_or(String::new(), |item| item.to_string());
        let optional_u64 =
            |value: Option<u64>| value.map_or(String::new(), |item| item.to_string());
        output.push_str(&format!(
            "{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{},{},{}\n",
            row.n_atoms,
            row.dimension,
            row.parameterization,
            row.optimizer,
            row.seed,
            row.best_energy,
            row.target_energy,
            row.gap,
            row.target_relative_gap,
            row.success,
            row.within_1_percent,
            row.within_5_percent,
            row.within_10_percent,
            optional(row.population_size),
            optional_u64(row.nominal_full_generations_per_restart),
            optional(row.restarts),
            row.pair_traversals,
            row.pair_terms_evaluated,
            row.objective_calls,
            row.gradient_calls,
            row.wall_seconds,
            row.measured_pair_seconds,
            row.estimated_optimizer_overhead_seconds,
            row.overlap_pairs,
            row.projected_candidates,
            row.reference_structure_audited,
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

fn write_crfmnes_configuration_csv(
    path: &Path,
    rows: &[CrfmnesConfigurationRow],
) -> Result<(), String> {
    let mut output = String::from(
        "n_atoms,dimension,seed,sigma_normalized,population_size,primary_configuration,best_energy,target_energy,gap,target_relative_gap,pair_traversals,wall_seconds\n",
    );
    for row in rows {
        output.push_str(&format!(
            "{},{},{},{:.6},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.9}\n",
            row.n_atoms,
            row.dimension,
            row.seed,
            row.sigma_normalized,
            row.population_size,
            row.primary_configuration,
            row.best_energy,
            row.target_energy,
            row.gap,
            row.target_relative_gap,
            row.pair_traversals,
            row.wall_seconds,
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

/// Run the frozen scaling protocol and write CSV/JSON evidence.
pub fn run(
    preset: LjPreset,
    root_seed: u64,
    workers: i32,
    root: &Path,
    reference_directory: Option<&Path>,
) -> Result<(), String> {
    let resolved_workers = if workers <= 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        workers as usize
    };
    let reference_audits = reference_directory
        .map(|directory| audit_references(preset, directory))
        .transpose()?
        .unwrap_or_default();
    let reference_structure_audited = !reference_audits.is_empty();
    let mut cases = Vec::new();
    for (size_index, &atoms) in preset.atoms().iter().enumerate() {
        for (parameter_index, parameterization) in
            [Parameterization::Free, Parameterization::FixedFrame]
                .into_iter()
                .enumerate()
        {
            for seed_index in 0..preset.seeds() {
                cases.push((
                    size_index,
                    parameter_index,
                    atoms,
                    parameterization,
                    seed_index,
                ));
            }
        }
    }
    let nested_workers = 1;
    let case_rows = parallel_batch(&cases, workers, |case| {
        let (size_index, parameter_index, atoms, parameterization, seed_index) = *case;
        let problem =
            LennardJones::new(atoms, parameterization).map_err(|error| error.to_string())?;
        let target = problem
            .target()
            .ok_or_else(|| format!("missing target for LJ{atoms}"))?;
        let case_id = (size_index * 2 + parameter_index) * preset.seeds() + seed_index;
        let seed = retry_run_seed(root_seed, case_id);
        let budget = preset.pair_budget();
        let mut arms = vec![
            ("random", random_arm(&problem, budget, seed)?),
            ("de-retry", de_arm(&problem, budget, seed ^ 0x4445)?),
            ("cma-retry", cma_arm(&problem, budget, seed ^ 0x0043_4d41)?),
            (
                "crfmnes-retry",
                crfmnes_arm(&problem, budget, seed ^ 0x4352_464d)?,
            ),
            (
                "bite-retry",
                bite_arm(&problem, budget, seed ^ 0x4249_5445)?,
            ),
        ];
        arms.extend(gradient_arms(&problem, budget, seed)?);
        Ok::<Vec<ScalingRow>, String>(
            arms.into_iter()
                .map(|(name, result)| {
                    let gap = result.best_energy - target.energy;
                    let target_relative_gap = gap.max(0.0) / target.energy.abs();
                    let (population_size, nominal_generations, restarts) =
                        arm_protocol(name, problem.dimension(), budget);
                    ScalingRow {
                        n_atoms: atoms,
                        dimension: problem.dimension(),
                        parameterization: parameterization.label().to_owned(),
                        optimizer: name.to_owned(),
                        seed,
                        best_energy: result.best_energy,
                        target_energy: target.energy,
                        gap,
                        target_relative_gap,
                        success: gap <= SUCCESS_TOLERANCE,
                        within_1_percent: target_relative_gap <= 0.01,
                        within_5_percent: target_relative_gap <= 0.05,
                        within_10_percent: target_relative_gap <= 0.10,
                        population_size,
                        nominal_full_generations_per_restart: nominal_generations,
                        restarts,
                        pair_traversals: result.pair_traversals,
                        pair_terms_evaluated: result.pair_traversals
                            * ((atoms * (atoms - 1) / 2) as u64),
                        objective_calls: result.objective_calls,
                        gradient_calls: result.gradient_calls,
                        wall_seconds: result.wall_seconds,
                        measured_pair_seconds: result.measured_pair_seconds,
                        estimated_optimizer_overhead_seconds: (result.wall_seconds
                            - result.measured_pair_seconds)
                            .max(0.0),
                        overlap_pairs: result.overlap_pairs,
                        projected_candidates: result.projected_candidates,
                        reference_structure_audited,
                        best_decision: result.best_decision,
                    }
                })
                .collect(),
        )
    });
    let mut rows = Vec::new();
    for result in case_rows {
        rows.extend(result?);
    }
    rows.sort_by(|left, right| {
        left.n_atoms
            .cmp(&right.n_atoms)
            .then(left.parameterization.cmp(&right.parameterization))
            .then(left.optimizer.cmp(&right.optimizer))
            .then(left.seed.cmp(&right.seed))
    });
    let output = root.join("lennard-jones");
    write_scaling_csv(&output.join("scaling.csv"), &rows)?;
    let crfmnes_configuration = if preset == LjPreset::Publication {
        crfmnes_configuration_check(root_seed, workers, preset.pair_budget())?
    } else {
        Vec::new()
    };
    if !crfmnes_configuration.is_empty() {
        write_crfmnes_configuration_csv(
            &output.join("crfmnes-configuration.csv"),
            &crfmnes_configuration,
        )?;
    }
    write_json(
        &output.join("reference-audit.json"),
        &if reference_structure_audited {
            json!({
                "schema_version": 1,
                "status": "completed",
                "source": REFERENCE_SOURCE,
                "coordinate_files_redistributed": false,
                "actual_evaluations": reference_audits.len(),
                "tolerance": 1.0e-6,
                "all_match": reference_audits.iter().all(|audit| audit.matches),
                "audits": reference_audits,
                "artifacts": {}
            })
        } else {
            json!({
                "schema_version": 1,
                "status": "not-run",
                "reason": "no-reference-directory-supplied",
                "coordinate_files_redistributed": false,
                "actual_evaluations": null,
                "artifacts": {}
            })
        },
    )
    .map_err(|error| error.to_string())?;
    let claim_scope = if preset == LjPreset::Publication {
        "ten-seed scaling comparison against source-cited putative targets"
    } else {
        "deterministic conformance pilot; not an optimizer ranking"
    };
    write_json(
        &output.join("run.json"),
        &json!({
            "schema_version": 2,
            "status": "completed",
            "study": "lennard-jones-scaling",
            "preset": preset.label(),
            "claim_scope": claim_scope,
            "command": format!(
                "cargo run --release --locked --features gradient-reference -- --lj-campaign --preset {} --workers {} --seed {} --output {}{}",
                preset.label(), workers, root_seed, root.display(),
                if reference_directory.is_some() { " --reference-directory <directory>" } else { "" }
            ),
            "root_seed": root_seed,
            "requested_workers": workers,
            "resolved_workers": resolved_workers,
            "nested_optimizer_workers": nested_workers,
            "retries_per_derivative_free_arm": DERIVATIVE_FREE_RETRIES,
            "sizes": preset.atoms(),
            "seeds_per_case": preset.seeds(),
            "pair_traversal_budget_per_arm": preset.pair_budget(),
            "success_tolerance": SUCCESS_TOLERANCE,
            "reference_source": REFERENCE_SOURCE,
            "reference_structure_audited": reference_structure_audited,
            "reference_audit_artifact": "reference-audit.json",
            "gradient_reference": "argmin-0.11-lbfgs",
            "population_policy": {
                "de-retry": "fcmaes-core default 15*dimension",
                "cma-retry": "fcmaes-core default floor(4+3*ln(dimension))",
                "crfmnes-retry": CRFMNES_POPULATION,
                "bite-retry": null
            },
            "crfmnes_primary_sigma_normalized": CRFMNES_SIGMA,
            "crfmnes_configuration_check": if crfmnes_configuration.is_empty() {
                json!({"status": "not-run", "reason": "smoke-preset"})
            } else {
                json!({
                    "status": "completed",
                    "claim_scope": "post-hoc configuration sensitivity; not optimizer ranking evidence",
                    "sizes": [13, 38, 98],
                    "seeds_per_configuration": 3,
                    "sigmas_normalized": CRFMNES_CHECK_SIGMAS,
                    "population_sizes": CRFMNES_CHECK_POPULATIONS,
                    "artifact": "crfmnes-configuration.csv"
                })
            },
            "timing_claim_scope": "diagnostic elapsed time; host exclusivity is not enforced",
            "rows": rows,
        }),
    )
    .map_err(|error| error.to_string())?;
    super::lennard_jones_pilot::run(root_seed, &output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_protocol_distinguishes_smoke_from_publication() {
        assert_eq!(LjPreset::Smoke.seeds(), 2);
        assert_eq!(LjPreset::Publication.seeds(), 10);
        assert_eq!(LjPreset::Publication.atoms(), &[13, 38, 55, 75, 98]);
        assert!(LjPreset::Publication.pair_budget() > LjPreset::Smoke.pair_budget());
    }

    #[test]
    fn compact_random_control_consumes_exact_budget() {
        let problem = LennardJones::new(13, Parameterization::FixedFrame).unwrap();
        let result = random_arm(&problem, 7, 42).unwrap();
        assert_eq!(result.pair_traversals, 7);
        assert_eq!(result.objective_calls, 7);
        assert!(result.best_energy.is_finite());
    }

    #[cfg(feature = "gradient-reference")]
    #[test]
    fn lbfgs_reference_stays_inside_pair_budget_and_improves() {
        let problem = LennardJones::new(13, Parameterization::FixedFrame).unwrap();
        let start = problem.initial_decision(42).unwrap();
        let initial = problem.energy(&start).unwrap().energy;
        let result = gradient::multistart(&problem, 100, 42).unwrap();
        assert!(result.pair_traversals <= 100);
        assert!(result.gradient_calls > 0);
        assert!(result.best_energy < initial);
    }

    #[cfg(feature = "gradient-reference")]
    #[test]
    fn campaign_results_do_not_depend_on_worker_count() {
        let root = std::env::temp_dir().join(format!("fcmaes-lj-workers-{}", std::process::id()));
        let serial = root.join("serial");
        let parallel = root.join("parallel");
        run(LjPreset::Smoke, 123, 1, &serial, None).unwrap();
        run(LjPreset::Smoke, 123, 2, &parallel, None).unwrap();
        let read_rows = |path: &Path| {
            let document: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path.join("lennard-jones/run.json")).unwrap(),
            )
            .unwrap();
            let mut rows = document["rows"].as_array().unwrap().clone();
            for row in &mut rows {
                let object = row.as_object_mut().unwrap();
                object.remove("wall_seconds");
                object.remove("measured_pair_seconds");
                object.remove("estimated_optimizer_overhead_seconds");
            }
            rows
        };
        assert_eq!(read_rows(&serial), read_rows(&parallel));
        std::fs::remove_dir_all(root).unwrap();
    }
}
