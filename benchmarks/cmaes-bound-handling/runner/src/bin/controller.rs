//! Plain CMA-ES on the neural-controller objective of
//! `tutorials/neural-controller-policy-search`, run as the job worker runs it
//! (`crates/fcmaes-job-worker/src/retry.rs`, `es_retry` without restarts:
//! population 31, sigma 0.3 in normalized coordinates, a uniform random
//! start), in the modes clip, repair and transform.
//!
//! ```text
//! controller --modes clip,repair,transform --seeds 1-10 --budget 110000 \
//!     [--out results/controller.tsv] [--verify]
//! ```
//!
//! One TSV row per run: mode, seed, budget, evaluations used, final loss and
//! stop code. With `--out` finished rows are skipped, so an interrupted
//! campaign resumes. `--verify` checks that clip and repair runs equal the
//! job worker's own `es_retry`.
#![forbid(unsafe_code)]

use bound_handling::{Mode, Table, option, parse_seeds};
use fcmaes_core::{Cmaes, CmaesParams, Fitness, RetryBounds, Rng, retry_run_seed};
use fcmaes_job_worker::retry::{CMAES_POPULATION, CMAES_SIGMA, EsSettings, es_retry};
use fcmaes_job_worker::threads::Threads;
use neural_controller_policy_search::{LOWER_WEIGHT, PARAMS, UPPER_WEIGHT, evaluate_policy};
use std::path::Path;

const ROOT: u64 = 42;
const SCENARIOS: u64 = 4;
const HORIZON: usize = 300;

/// The tutorial's `mix_seed` (private there; `FastRng::new` followed by one
/// `next_u64`): the fixed training panel is `mix_seed(42, 0, i)`, `i < 4`.
fn mix_seed(root: u64, stream: u64, index: u64) -> u64 {
    let mut state = (root
        ^ stream.wrapping_mul(0xd2b7_4407_b1ce_6e93)
        ^ index.wrapping_mul(0xca5a_8263_9512_1157))
        ^ 0x9e37_79b9_7f4a_7c15;
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One run of `es_run` for the CMA-ES kind with the start of `es_retry`.
fn run(
    mode: Mode,
    retry_seed: u64,
    budget: u64,
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
) -> (u64, f64, i32) {
    let mut start = Rng::new(retry_seed ^ 0x9E37_79B9_7F4A_7C15);
    let guess: Vec<f64> = (0..PARAMS)
        .map(|_| LOWER_WEIGHT + (UPPER_WEIGHT - LOWER_WEIGHT) * start.uniform01())
        .collect();
    let mut fitness = Fitness::bounded(PARAMS, 1, &[LOWER_WEIGHT; PARAMS], &[UPPER_WEIGHT; PARAMS]);
    fitness.set_normalize(true);
    let params = mode.apply(CmaesParams {
        popsize: CMAES_POPULATION as i32,
        max_evaluations: budget,
        seed: retry_seed,
        ..Default::default()
    });
    let mut cma = Cmaes::new(fitness, &guess, &[CMAES_SIGMA], &params);
    let r = cma.optimize_until(&|x: &[f64]| f(x), 1, &|| false);
    (r.evaluations, r.y, r.stop)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let modes: Vec<Mode> = option(&args, "--modes")
        .unwrap_or("clip,repair,transform")
        .split(',')
        .map(|m| Mode::parse(m).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let seeds = parse_seeds(option(&args, "--seeds").unwrap_or("1-10")).unwrap();
    let budget: u64 = option(&args, "--budget")
        .unwrap_or("110000")
        .parse()
        .expect("--budget needs a number");
    let verify = args.iter().any(|a| a == "--verify");
    let mut table = Table::open(
        option(&args, "--out").map(Path::new),
        "mode\tseed\tbudget\tevaluations\tloss\tstop",
        3,
    );
    let panel: Vec<u64> = (0..SCENARIOS).map(|i| mix_seed(ROOT, 0, i)).collect();
    let f = |x: &[f64]| evaluate_policy(x, &panel, HORIZON).score;
    for &seed in &seeds {
        for &mode in &modes {
            let key = [mode.name().to_owned(), seed.to_string(), budget.to_string()];
            if table.is_done(&key) {
                continue;
            }
            let retry_seed = retry_run_seed(seed, 0);
            let (evaluations, loss, stop) = run(mode, retry_seed, budget, &f);
            if verify && mode != Mode::Transform {
                let settings = EsSettings {
                    repair_z: mode == Mode::Repair,
                    ..EsSettings::cmaes(CMAES_POPULATION, CMAES_SIGMA)
                };
                let bounds =
                    RetryBounds::new(vec![LOWER_WEIGHT; PARAMS], vec![UPPER_WEIGHT; PARAMS])
                        .unwrap();
                let w = es_retry(
                    &settings,
                    &bounds,
                    retry_seed,
                    budget,
                    &f,
                    &Threads::new(1).unwrap(),
                );
                assert_eq!(
                    w.result.y.to_bits(),
                    loss.to_bits(),
                    "{mode:?} seed {seed}: differs from es_retry"
                );
                assert_eq!(w.result.evaluations, evaluations);
                assert_eq!(w.stop, stop);
                eprintln!("verified: {} seed {seed} equals es_retry", mode.name());
            }
            let mut row: Vec<String> = key.to_vec();
            row.extend([
                evaluations.to_string(),
                format!("{loss:.17e}"),
                stop.to_string(),
            ]);
            table.write(&row);
        }
    }
}
