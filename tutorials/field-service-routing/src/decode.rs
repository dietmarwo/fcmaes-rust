//! Deterministic assignment-plus-priority random-key decoding.

use std::fmt;

use crate::instance::Instance;

/// One decoded vehicle route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Route {
    /// Vehicle index.
    pub vehicle: usize,
    /// Ordered task indices.
    pub tasks: Vec<usize>,
}

/// Complete decoded plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Decoded {
    /// One route per vehicle, including empty routes.
    pub routes: Vec<Route>,
    /// Emergent non-empty route count.
    pub used_vehicles: usize,
}

/// Invalid random-key vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Wrong number of coordinates.
    Dimension { expected: usize, actual: usize },
    /// Non-finite coordinate.
    NonFinite(usize),
    /// Active-mask length mismatch.
    ActiveMask,
    /// Availability-mask length mismatch.
    AvailabilityMask,
    /// No available compatible vehicle.
    NoCompatibleVehicle(usize),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DecodeError {}

fn bin(key: f64, count: usize) -> usize {
    ((key.clamp(0.0, 1.0) * count as f64).floor() as usize).min(count - 1)
}

/// Decode a vector for the nominal base task set.
pub fn decode(x: &[f64], instance: &Instance) -> Result<Decoded, DecodeError> {
    let active = instance
        .tasks
        .iter()
        .map(|task| task.base)
        .collect::<Vec<_>>();
    decode_active(x, instance, &active, &vec![true; instance.vehicles.len()])
}

/// Decode the fixed task superset under scenario masks.
pub fn decode_active(
    x: &[f64],
    instance: &Instance,
    active: &[bool],
    available: &[bool],
) -> Result<Decoded, DecodeError> {
    if x.len() != 2 * instance.tasks.len() {
        return Err(DecodeError::Dimension {
            expected: 2 * instance.tasks.len(),
            actual: x.len(),
        });
    }
    if active.len() != instance.tasks.len() {
        return Err(DecodeError::ActiveMask);
    }
    if available.len() != instance.vehicles.len() {
        return Err(DecodeError::AvailabilityMask);
    }
    if let Some(index) = x.iter().position(|value| !value.is_finite()) {
        return Err(DecodeError::NonFinite(index));
    }
    let mut routed = vec![Vec::<(usize, f64)>::new(); instance.vehicles.len()];
    for task in 0..instance.tasks.len() {
        if !active[task] {
            continue;
        }
        let compatible = instance
            .vehicles
            .iter()
            .filter(|vehicle| {
                available[vehicle.id] && vehicle.skills & instance.tasks[task].skill != 0
            })
            .map(|vehicle| vehicle.id)
            .collect::<Vec<_>>();
        if compatible.is_empty() {
            return Err(DecodeError::NoCompatibleVehicle(task));
        }
        let vehicle = compatible[bin(x[task], compatible.len())];
        routed[vehicle].push((task, x[instance.tasks.len() + task]));
    }
    let routes = routed
        .into_iter()
        .enumerate()
        .map(|(vehicle, mut tasks)| {
            tasks.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            Route {
                vehicle,
                tasks: tasks.into_iter().map(|(task, _)| task).collect(),
            }
        })
        .collect::<Vec<_>>();
    let used_vehicles = routes
        .iter()
        .filter(|route| !route.tasks.is_empty())
        .count();
    Ok(Decoded {
        routes,
        used_vehicles,
    })
}

/// Encode the checked-in witness as interior-bin keys.
#[must_use]
pub fn witness_controls(instance: &Instance) -> Vec<f64> {
    let mut controls = vec![0.5; 2 * instance.tasks.len()];
    for (vehicle, route) in instance.witness_routes.iter().enumerate() {
        for (order, task) in route.iter().copied().enumerate() {
            let compatible = instance
                .vehicles
                .iter()
                .filter(|candidate| candidate.skills & instance.tasks[task].skill != 0)
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            let selected = compatible
                .iter()
                .position(|candidate| *candidate == vehicle)
                .expect("witness skill compatibility");
            controls[task] = (selected as f64 + 0.5) / compatible.len() as f64;
            controls[instance.tasks.len() + task] =
                (order as f64 + 0.5) / route.len().max(1) as f64;
        }
    }
    // Reserve urgent tasks are inactive nominally, but receive meaningful keys
    // beside their generator anchor so the insertion scenario has a feasible
    // structured seed without changing the decision dimension.
    for reserve in 0..crate::instance::RESERVE_TASKS {
        let task = crate::instance::BASE_TASKS + reserve;
        let vehicle = reserve;
        let compatible = instance
            .vehicles
            .iter()
            .filter(|candidate| candidate.skills & instance.tasks[task].skill != 0)
            .map(|candidate| candidate.id)
            .collect::<Vec<_>>();
        let selected = compatible
            .iter()
            .position(|candidate| *candidate == vehicle)
            .expect("reserve anchor compatibility");
        controls[task] = (selected as f64 + 0.5) / compatible.len() as f64;
        controls[instance.tasks.len() + task] =
            2.75 / instance.witness_routes[vehicle].len() as f64;
    }
    controls
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use fcmaes_core::{Rng, parallel_batch};

    use super::*;
    use crate::instance::{BASE_TASKS, DIMENSION, SEEDS, TASKS, generate};

    fn completeness(decoded: &Decoded, expected: usize) {
        let tasks = decoded
            .routes
            .iter()
            .flat_map(|route| route.tasks.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(tasks.len(), expected);
        assert_eq!(
            tasks.iter().copied().collect::<HashSet<_>>().len(),
            expected
        );
    }

    #[test]
    fn endpoints_reach_first_and_last_compatible_vehicle() {
        let instance = generate(SEEDS[0], 0);
        let task = 0;
        let compatible = instance
            .vehicles
            .iter()
            .filter(|vehicle| vehicle.skills & instance.tasks[task].skill != 0)
            .map(|vehicle| vehicle.id)
            .collect::<Vec<_>>();
        let mut x = vec![0.5; DIMENSION];
        x[task] = 0.0;
        let first = decode(&x, &instance).unwrap();
        x[task] = 1.0;
        let last = decode(&x, &instance).unwrap();
        assert!(first.routes[compatible[0]].tasks.contains(&task));
        assert!(
            last.routes[*compatible.last().unwrap()]
                .tasks
                .contains(&task)
        );
    }

    #[test]
    fn completeness_skills_ties_and_errors() {
        let instance = generate(SEEDS[0], 0);
        let x = vec![0.5; DIMENSION];
        let decoded = decode(&x, &instance).unwrap();
        completeness(&decoded, BASE_TASKS);
        for route in &decoded.routes {
            for task in &route.tasks {
                assert_ne!(
                    instance.vehicles[route.vehicle].skills & instance.tasks[*task].skill,
                    0
                );
            }
            assert!(route.tasks.windows(2).all(|pair| pair[0] < pair[1]));
        }
        let mut invalid = x.clone();
        invalid[7] = f64::NAN;
        assert_eq!(decode(&invalid, &instance), Err(DecodeError::NonFinite(7)));
    }

    #[test]
    fn fixed_superset_masks_preserve_active_completeness() {
        let instance = generate(SEEDS[0], 0);
        let mut active = vec![true; TASKS];
        active[3] = false;
        active[17] = false;
        let decoded = decode_active(&vec![0.5; DIMENSION], &instance, &active, &[true; 8]).unwrap();
        completeness(&decoded, TASKS - 2);
        assert!(!decoded.routes.iter().any(|route| route.tasks.contains(&3)));
    }

    #[test]
    fn exact_single_coordinate_plateau_bounds_hold() {
        let instance = generate(SEEDS[0], 0);
        let base = witness_controls(&instance);
        let task = instance.witness_routes[0][2];
        let route_size = instance.witness_routes[0].len();
        let compatible = instance
            .vehicles
            .iter()
            .filter(|vehicle| vehicle.skills & instance.tasks[task].skill != 0)
            .count();
        let mut priority_states = HashSet::new();
        let mut assignment_states = HashSet::new();
        for step in 0..=1_000 {
            let value = step as f64 / 1_000.0;
            let mut x = base.clone();
            x[TASKS + task] = value;
            priority_states.insert(format!("{:?}", decode(&x, &instance).unwrap()));
            let mut x = base.clone();
            x[task] = value;
            assignment_states.insert(format!("{:?}", decode(&x, &instance).unwrap()));
        }
        assert!(priority_states.len() <= route_size);
        assert!(assignment_states.len() <= compatible);
    }

    #[test]
    fn assignment_bins_are_flat_and_all_reachable() {
        for count in 1..=8 {
            let mut histogram = vec![0_usize; count];
            let samples = 200_000;
            for index in 0..samples {
                let key = (index as f64 + 0.5) / samples as f64;
                histogram[bin(key, count)] += 1;
            }
            let expected = samples as f64 / count as f64;
            assert!(
                histogram
                    .iter()
                    .all(|observed| { (*observed as f64 - expected).abs() / expected < 0.01 })
            );
        }
    }

    #[test]
    fn serial_and_parallel_batch_decoding_are_identical() {
        let instance = generate(SEEDS[0], 0);
        let mut rng = Rng::new(123);
        let candidates = (0..10_000)
            .map(|_| (0..DIMENSION).map(|_| rng.uniform01()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let serial = candidates
            .iter()
            .map(|candidate| decode(candidate, &instance).unwrap())
            .collect::<Vec<_>>();
        let parallel = parallel_batch(&candidates, 4, |candidate| {
            decode(candidate, &instance).unwrap()
        });
        assert_eq!(serial, parallel);
    }
}
