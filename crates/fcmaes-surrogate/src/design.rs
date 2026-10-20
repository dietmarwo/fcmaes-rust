//! Seeded Latin-hypercube and local candidates; no Sobol claim is made.
use crate::stream_seed;
use fcmaes_core::Rng;

pub fn latin_hypercube(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    if n == 0 {
        return Vec::new();
    }
    let mut rng = Rng::new(seed);
    let mut xs = vec![vec![0.0; d]; n];
    for j in 0..d {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = rng.int_below((i + 1) as i64) as usize;
            order.swap(i, k);
        }
        for (i, x) in xs.iter_mut().enumerate() {
            x[j] = (order[i] as f64 + rng.uniform01()) / n as f64;
        }
    }
    xs
}
pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Centre plus the best of four randomized LHS designs (maximin distance).
pub fn initial_design(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    if n == 0 {
        return Vec::new();
    }
    let mut best = Vec::new();
    let mut best_distance = -1.0_f64;
    for k in 0..4 {
        let mut xs = vec![vec![0.5; d]];
        xs.extend(latin_hypercube(n - 1, d, stream_seed(seed, 11, k)));
        let mut distance = f64::INFINITY;
        for i in 0..xs.len() {
            for j in 0..i {
                distance = distance.min(squared_distance(&xs[i], &xs[j]));
            }
        }
        if distance > best_distance {
            best_distance = distance;
            best = xs;
        }
    }
    best
}

pub fn candidates(count: usize, d: usize, seed: u64, centres: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut xs = latin_hypercube(count, d, stream_seed(seed, 21, 0));
    let mut rng = Rng::new(stream_seed(seed, 22, 0));
    let probability = (20.0 / d as f64).min(1.0);
    for i in 0..count / 2 {
        if centres.is_empty() {
            break;
        }
        let mut x = centres[i % centres.len()].clone();
        let mut changed = false;
        let sigma = [0.01, 0.05, 0.2][i % 3];
        for v in &mut x {
            if rng.uniform01() < probability {
                *v = (*v + sigma * rng.gaussian()).clamp(0.0, 1.0);
                changed = true;
            }
        }
        if !changed {
            let j = rng.int_below(d as i64) as usize;
            x[j] = rng.uniform01();
        }
        xs.push(x);
    }
    xs
}
