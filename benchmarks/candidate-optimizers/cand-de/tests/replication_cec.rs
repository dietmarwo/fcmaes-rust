//! Informational replication of the L-SRTDE paper's CEC 2024 results
//! (Stanovov and Semenkin, CEC 2024, Table X: 30-variable CEC 2017
//! functions, 300,000 evaluations, `[-100, 100]^30`).
//!
//! Ignored by default; run with
//! `cargo test -p cand-de --test replication_cec -- --ignored --nocapture`.
//!
//! The official CEC rotation matrices and shift vectors are not used. Each
//! function gets a seeded random orthogonal rotation `M` and a seeded uniform
//! random shift `o` in `[-80, 80]^30` as a documented substitute, so values
//! are comparable with the paper in level, not digit for digit. Values below
//! 1e-8 are reported as 0, as in the paper.
//!
//! Both L-SRTDE forms are run (the printed Eq. 7 default and
//! `pbest_difference_from_base = true`), with L-SHADE and jSO as references.

use cand_de::{Jso, LShade, LSrtde};
use fcmaes_core::Rng;
use solver_api::{Guard, Solver};

const DIM: usize = 30;
const BUDGET: u64 = 300_000;
const SEEDS: u64 = 10;

/// Seeded random orthogonal matrix (Gram–Schmidt on Gaussian rows).
fn rotation(seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(DIM);
    while rows.len() < DIM {
        let mut v = rng.normal_vec(DIM);
        for r in &rows {
            let dot: f64 = v.iter().zip(r).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(r).for_each(|(a, b)| *a -= dot * b);
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 1e-8 {
            rows.push(v.into_iter().map(|a| a / norm).collect());
        }
    }
    rows
}

/// Seeded uniform shift in `[-80, 80]^30`.
fn shift(seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..DIM).map(|_| -80.0 + 160.0 * rng.uniform01()).collect()
}

/// Shifted, scaled, rotated argument `z = M (scale · (x - o))`.
struct Transform {
    m: Vec<Vec<f64>>,
    o: Vec<f64>,
    scale: f64,
}

impl Transform {
    fn new(seed: u64, scale: f64) -> Self {
        Self {
            m: rotation(seed),
            o: shift(seed + 1000),
            scale,
        }
    }

    fn z(&self, x: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = x
            .iter()
            .zip(&self.o)
            .map(|(v, o)| self.scale * (v - o))
            .collect();
        self.m
            .iter()
            .map(|row| row.iter().zip(&d).map(|(a, b)| a * b).sum())
            .collect()
    }
}

fn bent_cigar(t: &Transform, x: &[f64]) -> f64 {
    let z = t.z(x);
    z[0] * z[0] + 1e6 * z[1..].iter().map(|v| v * v).sum::<f64>()
}

fn rastrigin(t: &Transform, x: &[f64]) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    t.z(x)
        .iter()
        .map(|v| v * v - 10.0 * (two_pi * v).cos() + 10.0)
        .sum()
}

fn rosenbrock(t: &Transform, x: &[f64]) -> f64 {
    let z: Vec<f64> = t.z(x).iter().map(|v| v + 1.0).collect();
    z.windows(2)
        .map(|w| 100.0 * (w[0] * w[0] - w[1]).powi(2) + (w[0] - 1.0).powi(2))
        .sum()
}

/// Mean, median, min and max of the errors (below 1e-8 counted as 0).
fn stats(values: &mut [f64]) -> (f64, f64, f64, f64) {
    for v in values.iter_mut() {
        if *v < 1e-8 {
            *v = 0.0;
        }
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let median = if n.is_multiple_of(2) {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    } else {
        values[n / 2]
    };
    (mean, median, values[0], values[n - 1])
}

#[test]
#[ignore = "informational replication; several minutes of CPU"]
fn replicate_table_x() {
    let cigar = Transform::new(11, 1.0);
    let rast = Transform::new(12, 5.12 / 100.0);
    let rosen = Transform::new(13, 2.048 / 100.0);
    let f_cigar = |x: &[f64]| bent_cigar(&cigar, x);
    let f_rast = |x: &[f64]| rastrigin(&rast, x);
    let f_rosen = |x: &[f64]| rosenbrock(&rosen, x);
    type Problem<'a> = (&'a str, &'a (dyn Fn(&[f64]) -> f64 + Sync), &'a str);
    let problems: [Problem; 3] = [
        (
            "bent-cigar",
            &f_cigar,
            "paper (F1): mean 0, median 0, min 0, max 0",
        ),
        (
            "rastrigin",
            &f_rast,
            "paper (row F4): mean 8.3607, median 7.9597, min 3.9804, max 18.904",
        ),
        (
            "rosenbrock",
            &f_rosen,
            "paper (row F3): mean 58.562, median 58.562, min 58.562, max 58.562",
        ),
    ];
    let solvers: Vec<(&str, Box<dyn Solver>)> = vec![
        ("lsrtde (Eq. 7 default)", Box::new(LSrtde::default())),
        (
            "lsrtde (base difference)",
            Box::new(LSrtde {
                pbest_difference_from_base: true,
                ..LSrtde::default()
            }),
        ),
        ("lshade", Box::new(LShade::default())),
        ("jso", Box::new(Jso::default())),
    ];
    let (lower, upper) = (vec![-100.0; DIM], vec![100.0; DIM]);
    for (fname, f, paper) in problems {
        println!("{fname}: {paper}");
        for (sname, solver) in &solvers {
            let mut values: Vec<f64> = (1..=SEEDS)
                .map(|seed| {
                    let guard = Guard::new(f, &lower, &upper, BUDGET);
                    solver.minimize(&|x| guard.eval(x), &lower, &upper, BUDGET, seed);
                    assert_eq!(guard.over_budget() + guard.out_of_bounds(), 0);
                    guard.best_y()
                })
                .collect();
            let (mean, median, min, max) = stats(&mut values);
            println!(
                "  {sname:<25} mean {mean:.4e}  median {median:.4e}  min {min:.4e}  max {max:.4e}"
            );
        }
    }
}
