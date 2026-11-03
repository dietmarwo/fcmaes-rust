//! Independent six-channel reference for CRN's additive-Hill tutorial.
//! This uses this tutorial's documented local ReBop compatibility backend.
#![forbid(unsafe_code)]
use rebop::expr::Expr;
use rebop::gillespie::{Gillespie, Rate};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    basal: f64,
    vmax: f64,
    k: f64,
    n: f64,
    decay: f64,
    edges: Vec<(usize, usize, i32)>,
    times: Vec<f64>,
    paths: usize,
    seed: u64,
    /// Initial counts of X0, X1, X2, taken from the CRN model.
    initial: [isize; 3],
    states: Vec<[isize; 3]>,
}

fn production(r: &Request, target: usize) -> Expr {
    let mut sum = Expr::Constant(r.basal);
    for &(source, destination, sign) in &r.edges {
        if destination != target {
            continue;
        }
        let power = Expr::Pow(
            Box::new(Expr::Concentration(source)),
            Box::new(Expr::Constant(r.n)),
        );
        let kp = r.k.powf(r.n);
        let numerator = if sign == -1 {
            Expr::Constant(r.vmax * kp)
        } else {
            Expr::Mul(Box::new(Expr::Constant(r.vmax)), Box::new(power.clone()))
        };
        let term = Expr::Div(
            Box::new(numerator),
            Box::new(Expr::Add(Box::new(Expr::Constant(kp)), Box::new(power))),
        );
        sum = Expr::Add(Box::new(sum), Box::new(term));
    }
    sum
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: crn_reference REQUEST.json")?;
    let r: Request = serde_json::from_slice(&std::fs::read(path)?)?;
    if !(1..=4096).contains(&r.paths)
        || r.times.len() < 2
        || r.times.len() > 1024
        || r.times[0] != 0.0
        || r.times.last().is_none_or(|t| !t.is_finite() || *t > 1000.0)
        || r.times
            .windows(2)
            .any(|w| !w[0].is_finite() || w[0] >= w[1])
        || [r.basal, r.vmax, r.k, r.n, r.decay]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0 || *x > 32.0)
        || r.edges.len() > 9
        || r.edges
            .iter()
            .any(|&(s, t, k)| s > 2 || t > 2 || ![-1, 1].contains(&k))
        || r.states.len() > 1024
        || r.states.iter().flatten().any(|x| *x < 0 || *x > 100000)
        || r.initial.iter().any(|x| *x < 0 || *x > 100000)
    {
        return Err("request exceeds the bounded reference domain".into());
    }
    let expressions: Vec<_> = (0..3).map(|i| production(&r, i)).collect();
    let births: Vec<Vec<_>> = r
        .states
        .iter()
        .map(|s| expressions.iter().map(|e| e.eval(s)).collect())
        .collect();
    let mut paths = Vec::new();
    for i in 0..r.paths {
        let mut model = Gillespie::new_with_seed(r.initial, true, r.seed.wrapping_add(i as u64));
        for (j, expression) in expressions.iter().enumerate() {
            let mut jump = [0; 3];
            jump[j] = 1;
            model.add_reaction(Rate::expr(expression.clone()), jump);
            jump[j] = -1;
            model.add_reaction(Rate::lma_sparse(r.decay, [(j as u32, 1)]), jump);
        }
        let mut values = Vec::new();
        for &t in &r.times {
            model.advance_until(t);
            values.push([
                model.get_species(0),
                model.get_species(1),
                model.get_species(2),
            ]);
        }
        paths.push(values);
    }
    println!(
        "{}",
        json!({"backend":"local-rebop-compatibility-0.9.7", "births":births, "paths":paths})
    );
    Ok(())
}
