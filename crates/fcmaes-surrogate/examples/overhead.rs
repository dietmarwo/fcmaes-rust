//! Proposal overhead against history size, dimension and batch size.
//!
//! `cargo run --release -p fcmaes-surrogate --example overhead -- [threads]`
//! prints one JSON line per configuration: the median wall time of a full
//! proposal (fit plus acquisition) and of a proposal with carried
//! hyperparameters, and the fitted negative log posterior.
use fcmaes_surrogate::{
    Config, Observation, Round, Space, StrategyState, Value, design, propose, stream_seed,
};
use std::time::Instant;

fn objective(x: &[f64]) -> f64 {
    // A smooth multimodal test function on the unit cube (Levy-like).
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 1.0 + (4.0 * v - 2.0 + 0.1 * i as f64) / 4.0;
            (w - 1.0).powi(2) * (1.0 + 10.0 * (std::f64::consts::PI * w + 1.0).sin().powi(2))
        })
        .sum()
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("threads"))
        .unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        for &d in &[2usize, 5, 10, 20, 32] {
            for &n in &[16usize, 32, 64, 128, 248] {
                let space = Space::new(vec![0.0; d], vec![1.0; d], vec![false; d]).unwrap();
                let xs = design::latin_hypercube(n, d, stream_seed(7, d as u64, n as u64));
                let history: Vec<Observation> = xs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| Observation {
                        id: i as u64,
                        x: x.clone(),
                        value: Value::Finite(objective(x)),
                    })
                    .collect();
                let config = Config::default();
                for &count in &[1usize, 8] {
                    let mut fit = Vec::new();
                    let mut carried = Vec::new();
                    let mut nlp = 0.0;
                    let mut fit_evaluations = 0;
                    for repeat in 0..3 {
                        let state = StrategyState::default();
                        let round = Round {
                            space: &space,
                            history: &history,
                            pending: &[],
                            state: &state,
                            config: &config,
                            seed: 11 + repeat,
                            count,
                        };
                        let t = Instant::now();
                        let p = propose(&round, &|| false).unwrap();
                        fit.push(t.elapsed().as_secs_f64());
                        fit_evaluations = p.fit_evaluations;
                        // Carried hyperparameters: a round that does not refit.
                        let mut state = p.state.clone();
                        state.round = 1;
                        let round = Round {
                            state: &state,
                            ..round
                        };
                        let t = Instant::now();
                        let q = propose(&round, &|| false).unwrap();
                        carried.push(t.elapsed().as_secs_f64());
                        assert_eq!(q.model_fits, 0);
                        let gp = fcmaes_surrogate::gp::ExactGp::with_hyper(
                            &history.iter().map(|o| space.to_unit(&o.x).unwrap()).collect::<Vec<_>>(),
                            &{
                                let ys: Vec<f64> = history
                                    .iter()
                                    .map(|o| match o.value {
                                        Value::Finite(y) => y,
                                        _ => unreachable!(),
                                    })
                                    .collect();
                                let t = fcmaes_surrogate::transform::Transform::fit(
                                    &ys,
                                    config.output_transform,
                                )
                                .unwrap();
                                ys.iter().map(|&y| t.apply(y)).collect::<Vec<_>>()
                            },
                            config.kernel,
                            p.state.hyper.clone().unwrap(),
                        )
                        .unwrap();
                        nlp = gp.negative_log_posterior;
                    }
                    fit.sort_by(f64::total_cmp);
                    carried.sort_by(f64::total_cmp);
                    println!(
                        "{{\"threads\":{threads},\"d\":{d},\"n\":{n},\"count\":{count},\"fit_ms\":{:.3},\"carried_ms\":{:.3},\"fit_evaluations\":{fit_evaluations},\"nlp\":{nlp:.6}}}",
                        fit[1] * 1e3,
                        carried[1] * 1e3
                    );
                }
            }
        }
    });
}
