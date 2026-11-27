//! Evaluations and seconds active CMA-ES needs to reach 1e-8, per seed.
//!
//! `cma_target PROBLEM DIM LAST_SEED BUDGET [POPSIZE [FIRST_SEED]]` with
//! PROBLEM one of sphere, rosen, ellipsoid (rotated, condition 1e6) and cigar
//! (rotated). Each line is `problem,dim,popsize,seed,evaluations,best,stop,
//! seconds`. Run it in two checkouts to compare releases.
use fcmaes_core::{Cmaes, CmaesParams, Fitness};
fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| (v - 0.7) * (v - 0.7)).sum()
}
fn ellipsoid(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + i as f64))
        .sum();
    let norm: f64 = (1..=x.len()).map(|i| (i * i) as f64).sum();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let r = v - 2.0 * s * (1.0 + i as f64) / norm;
            10f64.powf(6.0 * i as f64 / (n - 1.0)) * r * r
        })
        .sum()
}
fn cigar(x: &[f64]) -> f64 {
    let s: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + i as f64))
        .sum();
    let norm: f64 = (1..=x.len()).map(|i| (i * i) as f64).sum();
    let r: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v - 2.0 * s * (1.0 + i as f64) / norm)
        .collect();
    r[0] * r[0] + 1e6 * r[1..].iter().map(|v| v * v).sum::<f64>()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (name, d, seeds, budget): (&str, usize, u64, u64) = (
        &a[1],
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    let pop: i32 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0);
    let f: fn(&[f64]) -> f64 = match name {
        "rosen" => rosen,
        "sphere" => sphere,
        "cigar" => cigar,
        _ => ellipsoid,
    };
    let first: u64 = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(1);
    for seed in first..=seeds {
        let p = CmaesParams {
            popsize: pop,
            seed,
            max_evaluations: budget,
            stop_fitness: 1e-8,
            ..Default::default()
        };
        let mut es = Cmaes::new(
            Fitness::new(d, 1, vec![], vec![]),
            &vec![0.0; d],
            &[0.5],
            &p,
        );
        let t = std::time::Instant::now();
        let r = es.optimize(&f, 1);
        println!(
            "{name},{d},{pop},{seed},{},{:e},{},{:.3}",
            r.evaluations,
            r.y,
            r.stop,
            t.elapsed().as_secs_f64()
        );
    }
}
