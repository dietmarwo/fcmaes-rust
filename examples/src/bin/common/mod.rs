use std::env;

use fcmaes_core::{BiteParams, DeepBiteOpt};

#[derive(Clone, Debug)]
pub struct Options {
    pub evaluations: u64,
    pub batch: usize,
    pub seed: u64,
    pub data: Option<String>,
    pub positional: Vec<String>,
}

impl Options {
    pub fn parse() -> Result<Self, String> {
        let mut result = Self {
            evaluations: 20_000,
            batch: 16,
            seed: 42,
            data: None,
            positional: Vec::new(),
        };
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--evals" => result.evaluations = value(&mut args, "--evals")?,
                "--batch" => result.batch = value(&mut args, "--batch")?,
                "--seed" => result.seed = value(&mut args, "--seed")?,
                "--data" => result.data = Some(value(&mut args, "--data")?),
                "-h" | "--help" => {
                    println!(
                        "Options:\n  --evals N  objective evaluations (20000)\n  --batch N  ask/tell batch (16)\n  --seed N   random seed (42)\n  --data P   optional input dataset"
                    );
                    std::process::exit(0);
                }
                _ if argument.starts_with('-') => {
                    return Err(format!("unknown option: {argument}"));
                }
                _ => result.positional.push(argument),
            }
        }
        if result.evaluations == 0 || result.batch == 0 {
            return Err("--evals and --batch must be positive".into());
        }
        Ok(result)
    }
}

fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

pub fn optimize(
    lower: &[f64],
    upper: &[f64],
    options: &Options,
    objective: impl Fn(&[f64]) -> f64,
) -> (Vec<f64>, f64, u64) {
    let params = BiteParams {
        max_evaluations: options.evaluations,
        seed: options.seed,
        ..Default::default()
    };
    let mut optimizer = DeepBiteOpt::new(lower, upper, None, &params, 1);
    loop {
        let candidates = optimizer.ask(options.batch);
        if candidates.is_empty() {
            break;
        }
        let values: Vec<f64> = candidates.iter().map(|x| objective(x)).collect();
        optimizer.tell(&values);
    }
    let result = optimizer.result_public();
    (result.x, result.y, result.evaluations)
}
