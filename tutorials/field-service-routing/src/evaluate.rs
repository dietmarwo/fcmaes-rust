//! Native distance-matrix route forward pass.

use crate::decode::Decoded;
use crate::instance::Instance;

/// Distance convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMode {
    /// Exact Euclidean distance.
    Euclidean,
    /// Each leg rounded to the nearest kilometre.
    RoundedKm,
}

/// Evaluation settings not encoded by a route.
#[derive(Clone, Copy, Debug)]
pub struct EvalConfig {
    /// Travel-time multiplier.
    pub traffic_factor: f64,
    /// Distance convention.
    pub distance_mode: DistanceMode,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            traffic_factor: 1.0,
            distance_mode: DistanceMode::Euclidean,
        }
    }
}

/// Metrics for one route.
#[derive(Clone, Debug, Default)]
pub struct RouteMetrics {
    /// Vehicle index.
    pub vehicle: usize,
    /// Travel distance.
    pub distance_km: f64,
    /// Return minus shift start.
    pub duration_s: f64,
    /// Capacity used.
    pub load_kg: f64,
    /// Sum of late service starts.
    pub lateness_s: f64,
    /// Free waiting time.
    pub waiting_s: f64,
    /// Return time.
    pub return_s: f64,
}

/// Whole-plan metrics.
#[derive(Clone, Debug, Default)]
pub struct SolutionMetrics {
    /// Fixed plus distance monetary cost.
    pub cost: f64,
    /// Total distance.
    pub distance_km: f64,
    /// Emergent non-empty route count.
    pub used_vehicles: usize,
    /// Longest route duration.
    pub makespan_s: f64,
    /// Aggregate lateness.
    pub total_lateness_s: f64,
    /// Sum of positive route overload.
    pub capacity_excess_kg: f64,
    /// Sum of positive shift excess.
    pub shift_excess_s: f64,
    /// Coefficient of variation of non-empty route distance.
    pub imbalance_cv: f64,
    /// Mean waiting over used routes.
    pub mean_waiting_s: f64,
    /// Route details.
    pub per_route: Vec<RouteMetrics>,
}

fn leg_distance(
    a: Option<usize>,
    b: Option<usize>,
    instance: &Instance,
    mode: DistanceMode,
) -> f64 {
    let (ax, ay) = a.map_or((instance.depot_x_km, instance.depot_y_km), |task| {
        (instance.tasks[task].x_km, instance.tasks[task].y_km)
    });
    let (bx, by) = b.map_or((instance.depot_x_km, instance.depot_y_km), |task| {
        (instance.tasks[task].x_km, instance.tasks[task].y_km)
    });
    let distance = (ax - bx).hypot(ay - by);
    match mode {
        DistanceMode::Euclidean => distance,
        DistanceMode::RoundedKm => distance.round(),
    }
}

/// Evaluate an explicit decoded solution.
#[must_use]
pub fn evaluate(decoded: &Decoded, instance: &Instance, config: EvalConfig) -> SolutionMetrics {
    let mut metrics = SolutionMetrics::default();
    let mut non_empty_distances = Vec::new();
    for route in &decoded.routes {
        if route.tasks.is_empty() {
            continue;
        }
        let vehicle = &instance.vehicles[route.vehicle];
        let mut route_metrics = RouteMetrics {
            vehicle: route.vehicle,
            ..Default::default()
        };
        let mut clock = vehicle.shift_start_s;
        let mut previous = None;
        for task_index in &route.tasks {
            let task = &instance.tasks[*task_index];
            let distance =
                leg_distance(previous, Some(*task_index), instance, config.distance_mode);
            route_metrics.distance_km += distance;
            clock += 3600.0 * distance / instance.speed_km_h * config.traffic_factor;
            if clock < task.earliest_s {
                route_metrics.waiting_s += task.earliest_s - clock;
                clock = task.earliest_s;
            }
            route_metrics.lateness_s += (clock - task.latest_s).max(0.0);
            clock += task.service_s;
            route_metrics.load_kg += task.demand_kg;
            previous = Some(*task_index);
        }
        let return_distance = leg_distance(previous, None, instance, config.distance_mode);
        route_metrics.distance_km += return_distance;
        clock += 3600.0 * return_distance / instance.speed_km_h * config.traffic_factor;
        route_metrics.return_s = clock;
        route_metrics.duration_s = clock - vehicle.shift_start_s;
        metrics.cost += vehicle.fixed_cost + vehicle.cost_per_km * route_metrics.distance_km;
        metrics.distance_km += route_metrics.distance_km;
        metrics.total_lateness_s += route_metrics.lateness_s;
        metrics.capacity_excess_kg += (route_metrics.load_kg - vehicle.capacity_kg).max(0.0);
        metrics.shift_excess_s += (clock - vehicle.shift_end_s).max(0.0);
        metrics.makespan_s = metrics.makespan_s.max(route_metrics.duration_s);
        non_empty_distances.push(route_metrics.distance_km);
        metrics.per_route.push(route_metrics);
    }
    metrics.used_vehicles = non_empty_distances.len();
    if !non_empty_distances.is_empty() {
        let mean = non_empty_distances.iter().sum::<f64>() / non_empty_distances.len() as f64;
        let variance = non_empty_distances
            .iter()
            .map(|distance| (distance - mean).powi(2))
            .sum::<f64>()
            / non_empty_distances.len() as f64;
        metrics.imbalance_cv = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };
        metrics.mean_waiting_s = metrics
            .per_route
            .iter()
            .map(|route| route.waiting_s)
            .sum::<f64>()
            / metrics.used_vehicles as f64;
    }
    metrics
}

/// Normalized hard constraints, feasible at zero.
#[must_use]
pub fn constraints(metrics: &SolutionMetrics) -> [f64; 3] {
    [
        metrics.capacity_excess_kg / 100.0,
        metrics.total_lateness_s / 3600.0,
        metrics.shift_excess_s / 3600.0,
    ]
}

/// Whether all hard requirements hold within arithmetic tolerance.
#[must_use]
pub fn feasible(metrics: &SolutionMetrics) -> bool {
    constraints(metrics).iter().all(|value| *value <= 1.0e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Decoded, Route, decode, witness_controls};
    use crate::instance::{SEEDS, generate};

    #[test]
    fn witness_proves_generator_feasibility() {
        for (index, seed) in SEEDS.iter().copied().enumerate() {
            let instance = generate(seed, index);
            let decoded = decode(&witness_controls(&instance), &instance).unwrap();
            assert!(feasible(&evaluate(
                &decoded,
                &instance,
                EvalConfig::default()
            )));
        }
    }

    #[test]
    fn hand_computed_wait_late_and_empty_route() {
        let mut instance = generate(SEEDS[0], 0);
        instance.tasks[0].x_km = 4.0;
        instance.tasks[0].y_km = 3.0;
        instance.tasks[0].earliest_s = 9.0 * 3600.0;
        instance.tasks[0].latest_s = 9.0 * 3600.0 - 1.0;
        instance.tasks[0].service_s = 600.0;
        instance.speed_km_h = 60.0;
        let decoded = Decoded {
            routes: (0..instance.vehicles.len())
                .map(|vehicle| Route {
                    vehicle,
                    tasks: if vehicle == 0 { vec![0] } else { vec![] },
                })
                .collect(),
            used_vehicles: 1,
        };
        let result = evaluate(&decoded, &instance, EvalConfig::default());
        assert_eq!(result.used_vehicles, 1);
        assert!((result.distance_km - 10.0).abs() < 1.0e-12);
        assert!((result.per_route[0].waiting_s - 3_300.0).abs() < 1.0e-9);
        assert!((result.per_route[0].lateness_s - 1.0).abs() < 1.0e-9);
        assert!((result.per_route[0].duration_s - 4_500.0).abs() < 1.0e-9);
    }
}
