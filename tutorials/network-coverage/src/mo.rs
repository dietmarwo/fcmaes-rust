//! Integer-aware cost-versus-coverage MODE campaign.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, Rng, parallel_batch, pareto_indices};

use crate::coverage::{
    CoverageMetrics, GROUP_WEIGHT_EXPONENT, evaluate, marginal_greedy, nondominated,
};
use crate::decode::{UPPER, controls, decode};
use crate::instance::Instance;
use crate::oracle::{matching_cover, weighted_primal_dual};

/// Number of minimized objectives.
pub const OBJECTIVES: usize = 2;

/// MODE budget.
#[derive(Clone, Debug)]
pub struct MoConfig {
    /// Candidate-call budget.
    pub evaluations: usize,
    /// Even population size.
    pub population: usize,
    /// Candidate worker threads.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// Convergence observation.
#[derive(Clone, Debug)]
pub struct MoProgress {
    /// Candidate calls.
    pub evaluations: usize,
    /// Elapsed seconds.
    pub elapsed_seconds: f64,
    /// Nondominated population members.
    pub pareto_population: usize,
}

/// Replayed nondominated point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Whether this mask was already present in the deterministic seed set.
    pub origin: &'static str,
    /// Physical metrics.
    pub metrics: CoverageMetrics,
    /// Minimized normalized cost and uncovered ROI.
    pub objectives: [f64; OBJECTIVES],
    /// Documentation representative.
    pub selected: bool,
}

/// Completed MODE campaign.
#[derive(Clone, Debug)]
pub struct MoResult {
    /// Requested calls.
    pub requested_evaluations: usize,
    /// Actual calls.
    pub actual_evaluations: usize,
    /// Wall time.
    pub elapsed: Duration,
    /// Replayed nondominated front.
    pub pareto: Vec<ParetoPoint>,
    /// Full deterministic greedy frontier.
    pub greedy: Vec<CoverageMetrics>,
    /// Every deterministic greedy prefix, before dominance filtering.
    pub greedy_prefixes: Vec<CoverageMetrics>,
    /// Progress trace.
    pub progress: Vec<MoProgress>,
}

#[derive(Clone, Debug)]
struct SeedCandidate {
    controls: Vec<f64>,
    selected: Vec<bool>,
    origin: &'static str,
}

fn objective_values(instance: &Instance, candidate: &[f64]) -> Vec<f64> {
    let selected = match decode(candidate) {
        Some(selected) => selected,
        None => return vec![f64::INFINITY; OBJECTIVES],
    };
    let metrics = evaluate(instance, &selected, GROUP_WEIGHT_EXPONENT)
        .expect("decoded decision has the fixed dimension");
    vec![
        metrics.cost / instance.costs.iter().sum::<f64>(),
        1.0 - metrics.roi,
    ]
}

fn seed_candidate(selected: Vec<bool>, origin: &'static str) -> SeedCandidate {
    SeedCandidate {
        controls: controls(&selected),
        selected,
        origin,
    }
}

fn seeded_population(instance: &Instance, population: usize, seed: u64) -> Vec<SeedCandidate> {
    let greedy = marginal_greedy(instance, GROUP_WEIGHT_EXPONENT);
    let matching = matching_cover(instance).selected;
    let weighted = weighted_primal_dual(instance).selected;
    let mut result = Vec::with_capacity(population);
    result.push(seed_candidate(
        vec![false; instance.nodes()],
        "mode-retained-empty-seed",
    ));
    result.push(seed_candidate(
        vec![true; instance.nodes()],
        "mode-retained-full-seed",
    ));
    result.push(seed_candidate(matching, "mode-retained-matching-seed"));
    result.push(seed_candidate(weighted, "mode-retained-primal-dual-seed"));
    while result.len() < population {
        let index = result.len();
        if index < population.saturating_sub(2) {
            let greedy_index =
                index.saturating_sub(3) * (greedy.len() - 1) / population.saturating_sub(4).max(1);
            result.push(seed_candidate(
                greedy[greedy_index].metrics.selected.clone(),
                "mode-retained-greedy-seed",
            ));
        } else {
            let mut rng = Rng::new(seed.wrapping_add(index as u64));
            let candidate = (0..instance.nodes())
                .map(|_| if rng.uniform01() < 0.5 { 0.5 } else { 1.5 })
                .collect::<Vec<_>>();
            result.push(SeedCandidate {
                selected: decode(&candidate).expect("binary initial candidate must decode"),
                controls: candidate,
                origin: "mode-retained-random-initial",
            });
        }
    }
    result
}

fn progress(values: &[Vec<f64>], evaluations: usize, started: Instant) -> MoProgress {
    MoProgress {
        evaluations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        pareto_population: pareto_indices(values, OBJECTIVES).map_or(0, |front| front.len()),
    }
}

/// Run the integer-aware bi-objective campaign.
pub fn optimize(instance: &Instance, config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.population < 4
        || !config.population.is_multiple_of(2)
        || config.evaluations < config.population
    {
        return Err("MODE requires an even population and at least one population budget".into());
    }
    let dimension = instance.nodes();
    let fitness = Fitness::bounded(
        dimension,
        OBJECTIVES,
        &vec![0.0; dimension],
        &vec![UPPER; dimension],
    );
    let mut mode = Mode::try_new(
        fitness,
        OBJECTIVES,
        0,
        Some(vec![true; dimension]),
        &ModeParams {
            popsize: config.population as i32,
            seed: config.seed,
            nsga_update: true,
            ..Default::default()
        },
    )?;
    let started = Instant::now();
    let initial = seeded_population(instance, config.population, config.seed);
    let initial_x = initial
        .iter()
        .map(|candidate| candidate.controls.clone())
        .collect::<Vec<_>>();
    let initial_y = parallel_batch(&initial_x, config.workers, |candidate| {
        objective_values(instance, candidate)
    });
    mode.set_population(&initial_x, &initial_y);
    let mut actual = initial_y.len();
    let mut trace = vec![progress(&initial_y, actual, started)];
    let generations = config
        .evaluations
        .saturating_sub(config.population)
        .div_ceil(config.population);
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, config.workers, |candidate| {
            objective_values(instance, candidate)
        });
        actual += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1).is_multiple_of(4) || generation + 1 == generations {
            trace.push(progress(&mode.result().y, actual, started));
        }
    }
    let result = mode.result();
    let front = pareto_indices(&result.y, OBJECTIVES)?;
    let mut unique = Vec::<CoverageMetrics>::new();
    for index in front {
        let selected = decode(&result.x[index]).ok_or("MODE replay decode failed")?;
        let metrics = evaluate(instance, &selected, GROUP_WEIGHT_EXPONENT)
            .ok_or("MODE replay dimension failed")?;
        if !unique
            .iter()
            .any(|existing| existing.selected == metrics.selected)
        {
            unique.push(metrics);
        }
    }
    let kept = nondominated(&unique);
    let total_cost = instance.costs.iter().sum::<f64>();
    let mut pareto = kept
        .into_iter()
        .map(|index| {
            let metrics = unique[index].clone();
            ParetoPoint {
                origin: "mode-generated",
                objectives: [metrics.cost / total_cost, 1.0 - metrics.roi],
                metrics,
                selected: false,
            }
        })
        .collect::<Vec<_>>();
    pareto.sort_by(|left, right| left.metrics.cost.total_cmp(&right.metrics.cost));
    for index in [0, pareto.len().saturating_sub(1)] {
        if let Some(point) = pareto.get_mut(index) {
            point.selected = true;
        }
    }
    if pareto.len() > 2 {
        let knee = (1..pareto.len() - 1)
            .max_by(|left, right| {
                let score = |index: usize| {
                    let point = &pareto[index].metrics;
                    point.roi - point.cost / total_cost
                };
                score(*left).total_cmp(&score(*right))
            })
            .unwrap();
        pareto[knee].selected = true;
    }
    for point in &mut pareto {
        if let Some(seed) = initial
            .iter()
            .find(|seed| seed.selected == point.metrics.selected)
        {
            point.origin = seed.origin;
        }
    }
    let greedy_prefixes = marginal_greedy(instance, GROUP_WEIGHT_EXPONENT)
        .into_iter()
        .map(|point| point.metrics)
        .collect::<Vec<_>>();
    let greedy_indices = nondominated(&greedy_prefixes);
    let greedy = greedy_indices
        .into_iter()
        .map(|index| greedy_prefixes[index].clone())
        .collect();
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: actual,
        elapsed: started.elapsed(),
        pareto,
        greedy,
        greedy_prefixes,
        progress: trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{FIXTURES, generate};

    #[test]
    fn tiny_mode_keeps_replayable_endpoints() {
        let instance = generate(&FIXTURES[0]).unwrap();
        let result = optimize(
            &instance,
            &MoConfig {
                evaluations: 32,
                population: 8,
                workers: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(result.pareto.iter().all(|point| point.metrics.roi >= 0.0));
        assert_eq!(result.greedy_prefixes.len(), instance.nodes() + 1);
        assert_eq!(nondominated(&result.greedy).len(), result.greedy.len());
    }

    #[test]
    fn origin_labels_cover_the_actual_seed_population_only() {
        let instance = generate(&FIXTURES[0]).unwrap();
        let population = seeded_population(&instance, 8, 42);
        let matching = matching_cover(&instance).selected;
        let weighted = weighted_primal_dual(&instance).selected;
        assert_eq!(population[2].selected, matching);
        assert_eq!(population[2].origin, "mode-retained-matching-seed");
        assert_eq!(population[3].selected, weighted);
        assert_eq!(population[3].origin, "mode-retained-primal-dual-seed");
    }
}
