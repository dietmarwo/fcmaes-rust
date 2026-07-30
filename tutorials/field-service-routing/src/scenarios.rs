//! Named training disruptions and structurally different holdouts.

use crate::decode::{Decoded, decode_active};
use crate::evaluate::{DistanceMode, EvalConfig, SolutionMetrics, constraints, evaluate};
use crate::instance::{BASE_TASKS, Instance, TASKS};

/// One deterministic evaluation case.
#[derive(Clone, Debug)]
pub struct Scenario {
    /// Stable artifact name.
    pub name: &'static str,
    /// Scenario-specific physical data.
    pub instance: Instance,
    /// Active task mask over the fixed superset.
    pub active: Vec<bool>,
    /// Available vehicle mask.
    pub available: Vec<bool>,
    /// Travel convention.
    pub config: EvalConfig,
}

fn base_mask() -> Vec<bool> {
    (0..TASKS).map(|task| task < BASE_TASKS).collect()
}

fn nominal(instance: &Instance, name: &'static str) -> Scenario {
    Scenario {
        name,
        instance: instance.clone(),
        active: base_mask(),
        available: vec![true; instance.vehicles.len()],
        config: EvalConfig::default(),
    }
}

/// Five named scenarios optimized jointly by SO and QD.
#[must_use]
pub fn training(instance: &Instance) -> Vec<Scenario> {
    let mut traffic = nominal(instance, "traffic_x1_3");
    traffic.config.traffic_factor = 1.3;
    let mut cancelled = nominal(instance, "cancel_3_tasks");
    for task in [3, 17, 41] {
        cancelled.active[task] = false;
    }
    let mut urgent = nominal(instance, "insert_2_urgent");
    urgent.active[BASE_TASKS..].fill(true);
    let mut unavailable = nominal(instance, "vehicle_7_unavailable");
    unavailable.available[7] = false;
    vec![
        nominal(instance, "nominal"),
        traffic,
        cancelled,
        urgent,
        unavailable,
    ]
}

/// Four holdouts that change a different modelling assumption.
#[must_use]
pub fn holdout(instance: &Instance) -> Vec<Scenario> {
    let mut geography = nominal(instance, "geography_uniform");
    for task in &mut geography.instance.tasks {
        let angle = (task.id as f64 * 2.399_963_229_728_653).rem_euclid(std::f64::consts::TAU);
        let radius = 5.0 + 30.0 * ((task.id * 37 % 101) as f64 / 100.0).sqrt();
        task.x_km = radius * angle.cos();
        task.y_km = radius * angle.sin();
    }
    let mut tightened = nominal(instance, "windows_tightened_50pct");
    for task in &mut tightened.instance.tasks {
        let middle = 0.5 * (task.earliest_s + task.latest_s);
        let half_width = 0.25 * (task.latest_s - task.earliest_s);
        task.earliest_s = middle - half_width;
        task.latest_s = middle + half_width;
    }
    let mut fleet = nominal(instance, "fleet_mix_changed");
    for vehicle in &mut fleet.instance.vehicles {
        vehicle.capacity_kg *= if vehicle.id.is_multiple_of(2) {
            0.86
        } else {
            1.08
        };
        vehicle.fixed_cost *= if vehicle.id < 4 { 1.15 } else { 0.9 };
    }
    let mut rounded = nominal(instance, "distance_rounding_integer");
    rounded.config.distance_mode = DistanceMode::RoundedKm;
    vec![geography, tightened, fleet, rounded]
}

/// Structured non-anticipative seed that avoids vehicle 7 in every scenario.
///
/// Assignment keys are chosen from the intersection of the nominal and
/// vehicle-unavailable bins, so the same target vehicle is decoded under both
/// compatible-vehicle lists.
#[must_use]
pub fn robust_seed_controls(instance: &Instance) -> Vec<f64> {
    let mut routes = instance.witness_routes.clone();
    let displaced = std::mem::take(&mut routes[7]);
    let mut loads = routes
        .iter()
        .map(|route| {
            route
                .iter()
                .map(|task| instance.tasks[*task].demand_kg)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    for task in displaced {
        let target = instance
            .vehicles
            .iter()
            .take(7)
            .filter(|vehicle| {
                vehicle.skills & instance.tasks[task].skill != 0
                    && loads[vehicle.id] + instance.tasks[task].demand_kg <= vehicle.capacity_kg
            })
            .max_by(|left, right| {
                (left.capacity_kg - loads[left.id])
                    .total_cmp(&(right.capacity_kg - loads[right.id]))
            })
            .map(|vehicle| vehicle.id)
            .expect("generated fleet has reserve capacity");
        routes[target].push(task);
        loads[target] += instance.tasks[task].demand_kg;
    }
    let mut controls = vec![0.5; 2 * instance.tasks.len()];
    for (vehicle, route) in routes.iter().enumerate().take(7) {
        for (order, task) in route.iter().copied().enumerate() {
            let nominal = instance
                .vehicles
                .iter()
                .filter(|candidate| candidate.skills & instance.tasks[task].skill != 0)
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            let unavailable = nominal
                .iter()
                .copied()
                .filter(|candidate| *candidate != 7)
                .collect::<Vec<_>>();
            let full_position = nominal
                .iter()
                .position(|candidate| *candidate == vehicle)
                .expect("target compatible nominally");
            let reduced_position = unavailable
                .iter()
                .position(|candidate| *candidate == vehicle)
                .expect("target compatible without vehicle 7");
            let lower = (full_position as f64 / nominal.len() as f64)
                .max(reduced_position as f64 / unavailable.len() as f64);
            let upper = ((full_position + 1) as f64 / nominal.len() as f64)
                .min((reduced_position + 1) as f64 / unavailable.len() as f64);
            assert!(lower < upper, "assignment bins must overlap");
            controls[task] = 0.5 * (lower + upper);
            controls[instance.tasks.len() + task] =
                (order as f64 + 0.5) / route.len().max(1) as f64;
        }
    }
    // Urgent reserve visits remain near their anchor routes.
    let witness = crate::decode::witness_controls(instance);
    for task in BASE_TASKS..TASKS {
        controls[task] = witness[task];
        controls[instance.tasks.len() + task] = witness[instance.tasks.len() + task];
    }
    controls
}

/// One scenario replay.
#[derive(Clone, Debug)]
pub struct ScenarioEvaluation {
    /// Scenario name.
    pub name: &'static str,
    /// Scenario-specific decoded plan.
    pub decoded: Decoded,
    /// Forward-pass metrics.
    pub metrics: SolutionMetrics,
}

/// Robust hard-window evaluation.
#[derive(Clone, Debug)]
pub struct RobustEvaluation {
    /// Original normalized decision.
    pub controls: Vec<f64>,
    /// Individual scenario results.
    pub scenarios: Vec<ScenarioEvaluation>,
    /// Worst monetary cost.
    pub worst_cost: f64,
    /// Worst normalized capacity, lateness and shift violation.
    pub constraints: [f64; 3],
    /// Penalized scalar objective.
    pub objective: f64,
}

impl RobustEvaluation {
    /// Nominal result.
    #[must_use]
    pub fn nominal(&self) -> &ScenarioEvaluation {
        &self.scenarios[0]
    }

    /// Strict hard-window feasibility.
    #[must_use]
    pub fn feasible(&self) -> bool {
        self.constraints.iter().all(|value| *value <= 1.0e-9)
    }
}

/// Replay controls on an explicit scenario list.
#[must_use]
pub fn evaluate_cases(
    controls: &[f64],
    cases: &[Scenario],
    hard_windows: bool,
) -> Option<RobustEvaluation> {
    let mut evaluations = Vec::with_capacity(cases.len());
    let mut worst_cost: f64 = 0.0;
    let mut worst = [0.0_f64; 3];
    for case in cases {
        let decoded =
            decode_active(controls, &case.instance, &case.active, &case.available).ok()?;
        let metrics = evaluate(&decoded, &case.instance, case.config);
        let current = constraints(&metrics);
        for index in 0..3 {
            if !hard_windows && index == 1 {
                continue;
            }
            worst[index] = worst[index].max(current[index]);
        }
        worst_cost = worst_cost.max(metrics.cost);
        evaluations.push(ScenarioEvaluation {
            name: case.name,
            decoded,
            metrics,
        });
    }
    let objective = worst_cost + 10_000.0 * worst.iter().sum::<f64>();
    objective.is_finite().then_some(RobustEvaluation {
        controls: controls.to_vec(),
        scenarios: evaluations,
        worst_cost,
        constraints: worst,
        objective,
    })
}

/// Robust training evaluation.
#[must_use]
pub fn evaluate_training(controls: &[f64], instance: &Instance) -> Option<RobustEvaluation> {
    evaluate_cases(controls, &training(instance), true)
}

/// Holdout replay, not used for optimizer selection.
#[must_use]
pub fn evaluate_holdout(controls: &[f64], instance: &Instance) -> Option<RobustEvaluation> {
    evaluate_cases(controls, &holdout(instance), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::witness_controls;
    use crate::instance::{SEEDS, generate};

    #[test]
    fn scenario_names_and_dimension_are_stable() {
        let instance = generate(SEEDS[0], 0);
        let training = training(&instance);
        assert_eq!(
            training.iter().map(|case| case.name).collect::<Vec<_>>(),
            [
                "nominal",
                "traffic_x1_3",
                "cancel_3_tasks",
                "insert_2_urgent",
                "vehicle_7_unavailable"
            ]
        );
        assert!(training.iter().all(|case| case.active.len() == TASKS));
        let evaluated = evaluate_cases(&witness_controls(&instance), &training, true).unwrap();
        assert_eq!(evaluated.scenarios.len(), 5);
        assert!(evaluated.scenarios.iter().any(|row| {
            (row.metrics.cost - evaluated.scenarios[0].metrics.cost).abs() > 1.0e-9
                || (row.metrics.total_lateness_s - evaluated.scenarios[0].metrics.total_lateness_s)
                    .abs()
                    > 1.0e-9
        }));
    }
}
