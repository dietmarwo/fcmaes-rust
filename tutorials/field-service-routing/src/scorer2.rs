//! Independent supplied-route scorer written directly from `COST_SPEC.md`.
//!
//! Unlike `evaluate`, this module does not use a shared distance helper or
//! build route metrics. It is still an in-repository cross-check, not external
//! validation.

use crate::decode::Decoded;
use crate::evaluate::{DistanceMode, EvalConfig, SolutionMetrics};
use crate::instance::Instance;

/// Compact independent score.
#[derive(Clone, Copy, Debug, Default)]
pub struct IndependentScore {
    /// Monetary cost.
    pub cost: f64,
    /// Travel distance.
    pub distance_km: f64,
    /// Used routes.
    pub used_vehicles: usize,
    /// Maximum duration.
    pub makespan_s: f64,
    /// Total late service starts.
    pub lateness_s: f64,
    /// Capacity excess.
    pub capacity_excess_kg: f64,
    /// Shift excess.
    pub shift_excess_s: f64,
}

fn coordinates(instance: &Instance, task: Option<usize>) -> (f64, f64) {
    match task {
        Some(task) => (instance.tasks[task].x_km, instance.tasks[task].y_km),
        None => (instance.depot_x_km, instance.depot_y_km),
    }
}

/// Independently score explicit route ordering.
#[must_use]
pub fn score(decoded: &Decoded, instance: &Instance, config: EvalConfig) -> IndependentScore {
    let mut total = IndependentScore::default();
    for route in decoded
        .routes
        .iter()
        .filter(|route| !route.tasks.is_empty())
    {
        let vehicle = &instance.vehicles[route.vehicle];
        let mut time = vehicle.shift_start_s;
        let mut previous = None;
        let mut route_distance = 0.0;
        let mut load = 0.0;
        let mut late = 0.0;
        for current in route.tasks.iter().copied().map(Some).chain([None]) {
            let a = coordinates(instance, previous);
            let b = coordinates(instance, current);
            let raw = (a.0 - b.0).hypot(a.1 - b.1);
            let distance = match config.distance_mode {
                DistanceMode::Euclidean => raw,
                DistanceMode::RoundedKm => raw.round(),
            };
            route_distance += distance;
            time += 3600.0 * distance * config.traffic_factor / instance.speed_km_h;
            if let Some(task_index) = current {
                let task = &instance.tasks[task_index];
                time = time.max(task.earliest_s);
                late += (time - task.latest_s).max(0.0);
                time += task.service_s;
                load += task.demand_kg;
            }
            previous = current;
        }
        let duration = time - vehicle.shift_start_s;
        total.cost += vehicle.fixed_cost + route_distance * vehicle.cost_per_km;
        total.distance_km += route_distance;
        total.used_vehicles += 1;
        total.makespan_s = total.makespan_s.max(duration);
        total.lateness_s += late;
        total.capacity_excess_kg += (load - vehicle.capacity_kg).max(0.0);
        total.shift_excess_s += (time - vehicle.shift_end_s).max(0.0);
    }
    total
}

/// Maximum absolute discrepancy across shared metrics.
#[must_use]
pub fn max_discrepancy(primary: &SolutionMetrics, independent: IndependentScore) -> f64 {
    [
        (primary.cost - independent.cost).abs(),
        (primary.distance_km - independent.distance_km).abs(),
        (primary.used_vehicles as f64 - independent.used_vehicles as f64).abs(),
        (primary.makespan_s - independent.makespan_s).abs(),
        (primary.total_lateness_s - independent.lateness_s).abs(),
        (primary.capacity_excess_kg - independent.capacity_excess_kg).abs(),
        (primary.shift_excess_s - independent.shift_excess_s).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use fcmaes_core::Rng;

    use super::*;
    use crate::decode::decode;
    use crate::evaluate::evaluate;
    use crate::instance::{DIMENSION, SEEDS, generate};

    #[test]
    fn independent_scorer_matches_random_supplied_routes() {
        let mut rng = Rng::new(42);
        for (index, seed) in SEEDS.iter().copied().enumerate() {
            let instance = generate(seed, index);
            for _ in 0..100 {
                let controls = (0..DIMENSION).map(|_| rng.uniform01()).collect::<Vec<_>>();
                let decoded = decode(&controls, &instance).unwrap();
                let primary = evaluate(&decoded, &instance, EvalConfig::default());
                let second = score(&decoded, &instance, EvalConfig::default());
                assert!(max_discrepancy(&primary, second) <= 1.0e-9);
            }
        }
    }
}
