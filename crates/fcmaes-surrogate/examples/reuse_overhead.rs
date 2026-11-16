//! Fixed-hyperparameter covariance append versus reconstruction; not full BO overhead.
use fcmaes_surrogate::{
    design,
    gp::{ExactGp, Hyper, Kernel},
};
use std::{hint::black_box, time::Instant};
fn main() {
    let xs = design::latin_hypercube(225, 5, 12);
    let ys: Vec<_> = xs
        .iter()
        .map(|x| x.iter().map(|v| (v * 3.0).sin()).sum::<f64>())
        .collect();
    let hyper = Hyper {
        lengthscales: vec![0.3; 5],
    };
    let gp = ExactGp::with_hyper(&xs[..224], &ys[..224], Kernel::Matern52, hyper.clone()).unwrap();
    let start = Instant::now();
    for _ in 0..50 {
        black_box(gp.fantasize(xs[224].clone(), ys[224]).unwrap());
    }
    let append = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..50 {
        black_box(ExactGp::with_hyper(&xs, &ys, Kernel::Matern52, hyper.clone()).unwrap());
    }
    println!(
        "{}",
        serde_json::json!({"append_seconds":append,"rebuild_seconds":start.elapsed().as_secs_f64(),"iterations":50,"training":224,"dimension":5,"scope":"fixed-transform fixed-hyperparameter microbenchmark"})
    );
}
