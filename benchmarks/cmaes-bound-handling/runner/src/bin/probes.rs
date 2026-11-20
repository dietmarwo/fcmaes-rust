//! The two boundary-start probes of `PROTOCOL.md` section 3: plain CMA-ES on
//! one variable on `[0, 1]`, started at 0 with sigma 0.15 and population 31,
//! seeds 1 to 5, budget 3,100 evaluations (that of the REPAIR-Z unit test of
//! the first probe), in the modes clip, repair and transform.
//!
//! - `quadratic`: `(x - 0.3)^2`.
//! - `damp`: the amplitude of the damped spring of `examples/src/damp.rs` as
//!   a function of the first duration, the other coordinates fixed at that
//!   file's test point `[0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9]`
//!   (`DEVIATIONS.md`, section 3).
//!
//! ```text
//! probes [--out results/probes.tsv]
//! ```
//!
//! One TSV row per run: probe, mode, seed, evaluations, value, x, stop. The
//! smallest value of each probe on a grid of 10^6 + 1 points is printed to
//! stderr for reference.
#![forbid(unsafe_code)]

use bound_handling::{Mode, Table, option};
use fcmaes_core::{Cmaes, CmaesParams, Fitness};
use std::path::Path;

const MAX_ALPHA: f64 = 0.1;
const MAX_TIME: f64 = 40.0;
const REFERENCE: [f64; 8] = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9];

/// `damp::fitness` of `examples/src/damp.rs` (closed-form segments), copied
/// to keep this crate free of the examples' dependencies.
fn damp_amplitude(x: &[f64]) -> f64 {
    let n = x.len() / 2;
    let duration_scale = 2.0 * MAX_TIME / n as f64;
    let mut state = [1.0_f64, 0.0_f64];
    for i in 0..n {
        let duration = x[i] * duration_scale;
        let alpha = x[n + i] * 2.0 * MAX_ALPHA - MAX_ALPHA;
        let (sin, cos) = duration.sin_cos();
        let displaced = state[0] - alpha;
        state = [
            alpha + displaced * cos + state[1] * sin,
            -displaced * sin + state[1] * cos,
        ];
    }
    state[0].abs() + state[1].abs()
}

fn damp_probe(t: &[f64]) -> f64 {
    let mut x = REFERENCE;
    x[0] = t[0];
    damp_amplitude(&x)
}

fn quadratic(t: &[f64]) -> f64 {
    (t[0] - 0.3).powi(2)
}

/// A probe's name and objective.
type Probe = (&'static str, fn(&[f64]) -> f64);

fn main() {
    // The value of the test in damp.rs.
    assert!((damp_amplitude(&REFERENCE) - 1.4984988964913732).abs() < 1e-12);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut table = Table::open(
        option(&args, "--out").map(Path::new),
        "probe\tmode\tseed\tevaluations\tvalue\tx\tstop",
        3,
    );
    let probes: [Probe; 2] = [("quadratic", quadratic), ("damp", damp_probe)];
    for (name, f) in probes {
        let n = 1_000_000;
        let (best_t, best) = (0..=n)
            .map(|k| k as f64 / n as f64)
            .map(|t| (t, f(&[t])))
            .fold((0.0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
        eprintln!("{name}: grid minimum {best:.12e} at x = {best_t}");
        for mode in Mode::ALL {
            for seed in 1..=5u64 {
                let key = [name.to_owned(), mode.name().to_owned(), seed.to_string()];
                if table.is_done(&key) {
                    continue;
                }
                let mut fitness = Fitness::bounded(1, 1, &[0.0], &[1.0]);
                fitness.set_normalize(true);
                let params = mode.apply(CmaesParams {
                    popsize: 31,
                    max_evaluations: 3100,
                    seed,
                    ..Default::default()
                });
                let mut cma = Cmaes::new(fitness, &[0.0], &[0.15], &params);
                let r = cma.optimize(&f, 1);
                let mut row = key.to_vec();
                row.extend([
                    r.evaluations.to_string(),
                    format!("{:.17e}", r.y),
                    format!("{:.17e}", r.x[0]),
                    r.stop.to_string(),
                ]);
                table.write(&row);
            }
        }
    }
}
