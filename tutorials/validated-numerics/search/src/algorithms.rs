//! Bounded comparisons. Every method sees the nominal point first.
use crate::bridge::{Failure, Result};
use fcmaes_core::{BiteOpt, BiteParams, Cmaes, CmaesParams, Fitness, Rng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Reference,
    Grid,
    Random,
    Local,
    Cmaes,
    Biteopt,
}

#[derive(Debug, Serialize)]
pub struct Best {
    pub parameters: Vec<f64>,
    pub score: f64,
    pub evaluations: usize,
}

pub fn run(
    method: Method,
    dim: usize,
    budget: usize,
    seed: u64,
    mut evaluate: impl FnMut(&[f64]) -> Result<f64>,
) -> Result<Best> {
    if !(1..=2).contains(&dim) || !(9..=256).contains(&budget) {
        return Err(Failure::new(
            "invalid_protocol",
            "dimension 1..2 and budget 9..256 required",
        ));
    }
    let mut best = Best {
        parameters: vec![1.0; dim],
        score: f64::INFINITY,
        evaluations: 0,
    };
    let mut call = |point: &[f64]| -> Result<f64> {
        if point.len() != dim
            || point
                .iter()
                .any(|x| !x.is_finite() || !(0.5..=1.5).contains(x))
        {
            return Err(Failure::new(
                "optimizer_error",
                "optimizer proposed an invalid parameter",
            ));
        }
        let value = evaluate(point)?;
        if !value.is_finite() {
            return Err(Failure::new("protocol_error", "nonfinite estimated score"));
        }
        best.evaluations += 1;
        if value < best.score {
            best.score = value;
            best.parameters = point.to_vec();
        }
        Ok(value)
    };
    let nominal = vec![1.0; dim];
    let initial_score = call(&nominal)?;
    match method {
        Method::Reference => {}
        Method::Grid => {
            let levels = if dim == 1 {
                budget
            } else {
                (budget as f64).sqrt() as usize
            };
            if levels.pow(dim as u32) != budget || levels % 2 == 0 {
                return Err(Failure::new(
                    "invalid_protocol",
                    "grid budget must be an odd perfect dimension-power",
                ));
            }
            for index in 0..budget {
                let point = (0..dim)
                    .map(|axis| {
                        0.5 + ((index / levels.pow(axis as u32)) % levels) as f64
                            / (levels - 1) as f64
                    })
                    .collect::<Vec<_>>();
                if point != nominal {
                    call(&point)?;
                }
            }
        }
        Method::Random => {
            let mut rng = Rng::new(seed);
            for _ in 1..budget {
                call(&(0..dim).map(|_| 0.5 + rng.uniform01()).collect::<Vec<_>>())?;
            }
        }
        Method::Local => {
            let mut centre = nominal;
            let mut centre_score = initial_score;
            let mut step = 0.25;
            let mut improved = false;
            for index in 1..budget {
                let axis = ((index - 1) / 2) % dim;
                let sign = if (index - 1) % 2 == 0 { -1.0 } else { 1.0 };
                let mut point = centre.clone();
                point[axis] = (point[axis] + sign * step).clamp(0.5, 1.5);
                let value = call(&point)?;
                if value < centre_score {
                    centre = point;
                    centre_score = value;
                    improved = true;
                }
                if index % (2 * dim) == 0 {
                    if !improved {
                        step *= 0.5;
                    }
                    improved = false;
                }
            }
        }
        Method::Cmaes => {
            let fitness = Fitness::new(dim, 1, vec![0.5; dim], vec![1.5; dim]);
            let params = CmaesParams {
                popsize: 4,
                max_evaluations: (budget - 1) as u64,
                seed,
                ..Default::default()
            };
            let mut opt = Cmaes::new(fitness, &nominal, &vec![0.2; dim], &params);
            let mut remaining = budget - 1;
            while remaining > 0 {
                let population = opt.ask();
                let count = remaining.min(population.len());
                let mut values = Vec::new();
                for point in population.iter().take(count) {
                    values.push(call(point)?);
                }
                remaining -= count;
                // A final partial population is observed but never padded with invented evaluations.
                if count == population.len() {
                    opt.tell(&values);
                }
            }
        }
        Method::Biteopt => {
            let params = BiteParams {
                max_evaluations: (budget - 1) as u64,
                popsize: 8,
                seed,
                ..Default::default()
            };
            let mut opt = BiteOpt::new(&vec![0.5; dim], &vec![1.5; dim], Some(&nominal), &params);
            for _ in 1..budget {
                let points = opt.ask(1);
                let point = points.first().ok_or_else(|| {
                    Failure::new("optimizer_error", "BiteOpt stopped before budget")
                })?;
                let value = call(point)?;
                opt.tell(&[value]);
            }
        }
    }
    Ok(best)
}
