//! Native Multi-UAV Task Assignment benchmark objective.
//!
//! This is an independent Rust implementation of the continuous random-key
//! formulation in the enhanced Multi-UAV-Task-Assignment-Benchmark. The first
//! `vehicles - 1` coordinates place route separators and the remaining
//! coordinates sort the targets into one unique visit sequence.
//!
//! The evaluator intentionally preserves the benchmark's timing semantics:
//! service time at the current target is charged before travel to the next
//! target, and a next target earns its reward when arrival is strictly before
//! the horizon. The energy proxy is `sum(speed^2 * travel_time)` for rewarded
//! visits.

use fcmaes_core::Rng;

pub const SPEED_CHOICES: [f64; 3] = [10.0, 15.0, 30.0];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub x: f64,
    pub y: f64,
    pub reward: f64,
    pub service_time: f64,
}

impl Target {
    pub const fn new(x: f64, y: f64, reward: f64, service_time: f64) -> Self {
        Self {
            x,
            y,
            reward,
            service_time,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UavMetrics {
    pub reward: f64,
    pub max_time: f64,
    pub energy: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UavSolution {
    pub metrics: UavMetrics,
    /// Target indices are one-based, matching the Python benchmark.
    pub assignments: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct UavProblem {
    speeds: Vec<f64>,
    /// Includes the depot at index zero.
    targets: Vec<Target>,
    distances: Vec<f64>,
    time_limit: f64,
    total_reward: f64,
}

impl UavProblem {
    pub fn try_new(
        speeds: Vec<f64>,
        targets: Vec<Target>,
        time_limit: f64,
    ) -> Result<Self, &'static str> {
        if speeds.is_empty()
            || speeds
                .iter()
                .any(|speed| !speed.is_finite() || *speed <= 0.0)
        {
            return Err("at least one finite positive UAV speed is required");
        }
        if targets.is_empty() {
            return Err("at least one target is required");
        }
        if !time_limit.is_finite() || time_limit <= 0.0 {
            return Err("time limit must be finite and positive");
        }
        if targets.iter().any(|target| {
            !target.x.is_finite()
                || !target.y.is_finite()
                || !target.reward.is_finite()
                || target.reward < 0.0
                || !target.service_time.is_finite()
                || target.service_time < 0.0
        }) {
            return Err("targets must have finite coordinates and non-negative rewards and times");
        }

        let mut with_depot = Vec::with_capacity(targets.len() + 1);
        with_depot.push(Target::new(0.0, 0.0, 0.0, 0.0));
        with_depot.extend(targets);
        let count = with_depot.len();
        let mut distances = vec![0.0; count * count];
        for left in 0..count {
            for right in 0..left {
                let dx = with_depot[left].x - with_depot[right].x;
                let dy = with_depot[left].y - with_depot[right].y;
                let distance = dx.hypot(dy);
                distances[left * count + right] = distance;
                distances[right * count + left] = distance;
            }
        }
        let total_reward = with_depot.iter().map(|target| target.reward).sum();
        Ok(Self {
            speeds,
            targets: with_depot,
            distances,
            time_limit,
            total_reward,
        })
    }

    /// Generate the same benchmark distribution with a deterministic PCG
    /// stream. Seeded instances are statistically equivalent to, but not
    /// bit-identical with, Python's Mersenne-Twister instances.
    pub fn generate(
        vehicle_count: usize,
        target_count: usize,
        map_size: usize,
        seed: u64,
    ) -> Result<Self, &'static str> {
        if vehicle_count == 0 || target_count == 0 || map_size == 0 {
            return Err("vehicle count, target count, and map size must be positive");
        }
        let mut rng = Rng::new(seed);
        let speeds = (0..vehicle_count)
            .map(|_| SPEED_CHOICES[rng.int_below(SPEED_CHOICES.len() as i64) as usize])
            .collect();
        let offset = map_size as f64 * 0.5;
        let targets = (0..target_count)
            .map(|_| {
                Target::new(
                    rng.int_below(map_size as i64) as f64 + 1.0 - offset,
                    rng.int_below(map_size as i64) as f64 + 1.0 - offset,
                    rng.int_below(10) as f64 + 1.0,
                    rng.int_below(26) as f64 + 5.0,
                )
            })
            .collect();
        Self::try_new(speeds, targets, map_size as f64 / SPEED_CHOICES[1])
    }

    #[inline]
    pub fn vehicle_count(&self) -> usize {
        self.speeds.len()
    }

    #[inline]
    pub fn target_count(&self) -> usize {
        self.targets.len() - 1
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.vehicle_count() - 1 + self.target_count()
    }

    #[inline]
    pub fn speeds(&self) -> &[f64] {
        &self.speeds
    }

    #[inline]
    pub fn targets(&self) -> &[Target] {
        &self.targets[1..]
    }

    #[inline]
    pub fn time_limit(&self) -> f64 {
        self.time_limit
    }

    #[inline]
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    #[inline]
    fn distance(&self, left: usize, right: usize) -> f64 {
        self.distances[left * self.targets.len() + right]
    }

    fn separators_and_sequence(&self, x: &[f64]) -> (Vec<usize>, Vec<usize>) {
        assert_eq!(x.len(), self.dimension());
        let target_count = self.target_count();
        let separator_count = self.vehicle_count() - 1;
        let mut separators = vec![0usize; target_count + 1];
        // The final sentinel closes the final vehicle route.
        separators[target_count] = 1;
        for value in &x[..separator_count] {
            let position = (value.clamp(0.0, 1.0) * target_count as f64) as usize;
            separators[position.min(target_count - 1)] += 1;
        }
        let keys = &x[separator_count..];
        let mut sequence: Vec<usize> = (1..=target_count).collect();
        sequence.sort_by(|&left, &right| keys[left - 1].total_cmp(&keys[right - 1]));
        (separators, sequence)
    }

    fn evaluate_internal(
        &self,
        x: &[f64],
        mut assignments: Option<&mut [Vec<usize>]>,
    ) -> UavMetrics {
        let (mut separators, sequence) = self.separators_and_sequence(x);
        let mut metrics = UavMetrics::default();
        let mut post = 0usize;

        for (vehicle, speed) in self.speeds.iter().copied().enumerate() {
            let mut previous = 0usize;
            let mut time = 0.0;
            while separators[post] == 0 {
                let target_index = sequence[post];
                time += self.targets[previous].service_time;
                let travel_time = self.distance(previous, target_index) / speed;
                time += travel_time;
                if time >= self.time_limit {
                    // Time only increases, so the remaining targets before the
                    // next separator cannot earn reward. Skip their distance
                    // calculations while preserving the encoded route split.
                    post += 1;
                    while separators[post] == 0 {
                        post += 1;
                    }
                    break;
                }
                metrics.reward += self.targets[target_index].reward;
                metrics.max_time = metrics.max_time.max(time);
                metrics.energy += speed * speed * travel_time;
                if let Some(routes) = assignments.as_deref_mut() {
                    routes[vehicle].push(target_index);
                }
                previous = target_index;
                post += 1;
            }
            separators[post] -= 1;
        }
        metrics
    }

    pub fn evaluate(&self, x: &[f64]) -> UavMetrics {
        self.evaluate_internal(x, None)
    }

    /// Scalar minimization objective corresponding to reward maximization.
    pub fn fitness(&self, x: &[f64]) -> f64 {
        -self.evaluate(x).reward
    }

    /// MODE row: negative reward, makespan, and energy, all minimized.
    pub fn multi_objective(&self, x: &[f64]) -> Vec<f64> {
        let metrics = self.evaluate(x);
        vec![-metrics.reward, metrics.max_time, metrics.energy]
    }

    pub fn solution(&self, x: &[f64]) -> UavSolution {
        let mut assignments = vec![Vec::new(); self.vehicle_count()];
        let metrics = self.evaluate_internal(x, Some(&mut assignments));
        UavSolution {
            metrics,
            assignments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(time_limit: f64) -> UavProblem {
        UavProblem::try_new(
            vec![1.0, 1.0],
            vec![
                Target::new(1.0, 0.0, 10.0, 2.0),
                Target::new(3.0, 0.0, 20.0, 4.0),
            ],
            time_limit,
        )
        .unwrap()
    }

    #[test]
    fn split_routes_match_the_benchmark_encoding() {
        let problem = fixture(10.0);
        let solution = problem.solution(&[0.5, 0.1, 0.2]);
        assert_eq!(solution.assignments, [vec![1], vec![2]]);
        assert_eq!(
            solution.metrics,
            UavMetrics {
                reward: 30.0,
                max_time: 3.0,
                energy: 4.0,
            }
        );
        assert_eq!(problem.fitness(&[0.5, 0.1, 0.2]), -30.0);
        assert_eq!(problem.multi_objective(&[0.5, 0.1, 0.2]), [-30.0, 3.0, 4.0]);
    }

    #[test]
    fn repeated_separator_and_horizon_filter_routes() {
        let problem = fixture(4.0);
        let solution = problem.solution(&[0.0, 0.1, 0.2]);
        assert_eq!(solution.assignments, [Vec::<usize>::new(), vec![1]]);
        assert_eq!(solution.metrics.reward, 10.0);
        assert_eq!(solution.metrics.max_time, 1.0);
        assert_eq!(solution.metrics.energy, 1.0);
    }

    #[test]
    fn permutation_keys_are_unique_and_deterministic() {
        let problem = fixture(10.0);
        let ascending = problem.solution(&[0.5, 0.1, 0.2]);
        let descending = problem.solution(&[0.5, 0.2, 0.1]);
        assert_eq!(ascending.assignments, [vec![1], vec![2]]);
        assert_eq!(descending.assignments, [vec![2], vec![1]]);
        assert_eq!(ascending.metrics.reward, descending.metrics.reward);
    }

    #[test]
    fn generated_instances_are_valid_and_repeatable() {
        let first = UavProblem::generate(5, 30, 5_000, 65).unwrap();
        let second = UavProblem::generate(5, 30, 5_000, 65).unwrap();
        let different = UavProblem::generate(5, 30, 5_000, 66).unwrap();
        assert_eq!(first.speeds(), second.speeds());
        assert_eq!(first.targets(), second.targets());
        assert_ne!(first.targets(), different.targets());
        assert_eq!(first.dimension(), 34);
        assert_eq!(first.time_limit(), 5_000.0 / 15.0);
        assert!(first.total_reward() > 0.0);
        assert!(
            first
                .speeds()
                .iter()
                .all(|speed| SPEED_CHOICES.contains(speed))
        );
    }

    #[test]
    fn validates_problem_definition() {
        let target = Target::new(0.0, 0.0, 1.0, 1.0);
        assert!(UavProblem::try_new(vec![], vec![target], 1.0).is_err());
        assert!(UavProblem::try_new(vec![1.0], vec![], 1.0).is_err());
        assert!(UavProblem::try_new(vec![0.0], vec![target], 1.0).is_err());
        assert!(UavProblem::try_new(vec![1.0], vec![target], 0.0).is_err());
        assert!(
            UavProblem::try_new(vec![1.0], vec![Target::new(0.0, 0.0, -1.0, 1.0)], 1.0).is_err()
        );
        assert!(UavProblem::generate(0, 1, 1, 1).is_err());
    }

    #[test]
    fn one_vehicle_needs_no_separator_key() {
        let problem =
            UavProblem::try_new(vec![2.0], vec![Target::new(2.0, 0.0, 7.0, 1.0)], 2.0).unwrap();
        assert_eq!(problem.dimension(), 1);
        let solution = problem.solution(&[0.25]);
        assert_eq!(solution.assignments, [vec![1]]);
        assert_eq!(solution.metrics.reward, 7.0);
    }
}
