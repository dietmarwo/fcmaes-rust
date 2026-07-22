//! Lotka-Volterra fox-control example from `examples/lotka.py`.

use crate::integration::dopri5;

pub const DIMENSION: usize = 20;
pub const INITIAL_POPULATION: [f64; 2] = [10.0, 5.0];

pub const REFERENCE_SOLUTION: [f64; DIMENSION] = [
    0.7764942271302568,
    9.831131324541304e-13,
    -0.4392523575954558,
    0.9999999991093724,
    0.9999999993419174,
    0.877806604524956,
    -0.21969547982373291,
    0.9877830923045987,
    0.21691094924304902,
    -0.016089523522436144,
    1.0,
    0.7622848572479829,
    -0.0004231871176822595,
    -0.015617623735551967,
    -0.9227281069513724,
    0.8517521143397784,
    8.397851857275901e-19,
    1.0,
    1.0,
    0.1509108812092751,
];

pub fn derivative([rabbits, foxes]: [f64; 2]) -> [f64; 2] {
    [
        rabbits - 0.1 * rabbits * foxes,
        -1.5 * foxes + 0.075 * rabbits * foxes,
    ]
}

pub fn kill_times(x: &[f64]) -> Vec<f64> {
    x.iter()
        .enumerate()
        .filter(|(_, value)| **value > 0.0)
        .map(|(year, value)| year as f64 + value)
        .collect()
}

pub fn fitness(x: &[f64]) -> f64 {
    assert_eq!(x.len(), DIMENSION);
    let mut state = INITIAL_POPULATION;
    let mut time = 0.0;
    for target in kill_times(x) {
        let Ok(next) = integrate(state, time, target) else {
            return 1e10;
        };
        state = [next[0], (next[1] - 1.0).max(1.0)];
        time = target;
    }
    let mut maximum_rabbits = f64::NEG_INFINITY;
    for sample in 0..50 {
        let target = DIMENSION as f64 + 5.0 * sample as f64 / 49.0;
        let Ok(next) = integrate(state, time, target) else {
            return 1e10;
        };
        state = next;
        time = target;
        maximum_rabbits = maximum_rabbits.max(state[0]);
    }
    -maximum_rabbits
}

fn integrate(state: [f64; 2], start: f64, end: f64) -> Result<[f64; 2], &'static str> {
    dopri5(state, start, end, 0.05, 1e-8, 1e-8, |_, state| {
        derivative(*state)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_time_encoding() {
        assert_eq!(kill_times(&[-0.5, 0.25, 0.0, 0.75]), [1.25, 3.75]);
    }

    #[test]
    fn objectives_match_scipy_references() {
        let never = fitness(&[-0.5; DIMENSION]);
        let yearly = fitness(&[0.5; DIMENSION]);
        let best = fitness(&REFERENCE_SOLUTION);
        assert!((never + 40.588063291621026).abs() < 2e-5, "{never}");
        assert!((yearly + 62.19130800844176).abs() < 2e-3, "{yearly}");
        assert!((best + 132.26163547824325).abs() < 5e-3, "{best}");
    }
}
