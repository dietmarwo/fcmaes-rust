//! Runtime ReBop assembly for a proposed topology.

use rebop::expr::Expr;
use rebop::gillespie::{Gillespie, Rate};
use serde::{Deserialize, Serialize};

use crate::config::{
    BURN_IN, HILL_K, INITIAL_COPIES, MAX_MOLECULES, SAMPLE_COUNT, SAMPLE_INTERVAL,
};
use crate::grammar::{EDGE_INDEX, Edge, GENES, Topology};

/// Decoded edge kinetics.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EdgeKinetics {
    pub strength: f64,
    pub hill: f64,
}

/// Positive physical parameters decoded from the mixed log/linear vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kinetics {
    pub basal: [f64; GENES],
    pub degradation: [f64; GENES],
    pub edges: Vec<(usize, EdgeKinetics)>,
}

impl Kinetics {
    /// Decode `[log10 basal_i, log10 degradation_i, ..., log10 alpha_e, n_e]`.
    pub fn decode(topology: &Topology, decision: &[f64]) -> Result<Self, &'static str> {
        if decision.len() != topology.parameter_dimension() {
            return Err("parameter dimension does not match topology");
        }
        let mut basal = [0.0; GENES];
        let mut degradation = [0.0; GENES];
        let mut pointer = 0;
        for gene in 0..GENES {
            basal[gene] = 10.0_f64.powf(decision[pointer]);
            degradation[gene] = 10.0_f64.powf(decision[pointer + 1]);
            pointer += 2;
        }
        let mut edges = Vec::with_capacity(topology.active_edges());
        for (index, &kind) in topology.edges.iter().enumerate() {
            if kind == Edge::Absent as u8 {
                continue;
            }
            edges.push((
                index,
                EdgeKinetics {
                    strength: 10.0_f64.powf(decision[pointer]),
                    hill: decision[pointer + 1],
                },
            ));
            pointer += 2;
        }
        let decoded = Self {
            basal,
            degradation,
            edges,
        };
        if decoded
            .basal
            .iter()
            .chain(&decoded.degradation)
            .any(|value| !value.is_finite() || *value <= 0.0)
            || decoded.edges.iter().any(|(_, edge)| {
                !edge.strength.is_finite()
                    || edge.strength <= 0.0
                    || !edge.hill.is_finite()
                    || edge.hill <= 0.0
            })
        {
            return Err("decoded kinetics are not finite and positive");
        }
        Ok(decoded)
    }
}

/// Optimizer bounds for one topology.
pub fn bounds(topology: &Topology) -> (Vec<f64>, Vec<f64>) {
    let mut lower = Vec::with_capacity(topology.parameter_dimension());
    let mut upper = Vec::with_capacity(topology.parameter_dimension());
    for _ in 0..GENES {
        lower.extend([-1.0, 0.005_f64.log10()]);
        upper.extend([50.0_f64.log10(), 0.0]);
    }
    for _ in 0..topology.active_edges() {
        lower.extend([-1.0, 1.0]);
        upper.extend([2.0, 5.0]);
    }
    (lower, upper)
}

fn hill_expression(source: usize, kind: Edge, parameters: EdgeKinetics) -> Expr {
    let count = Expr::Concentration(source);
    let exponent = Expr::Constant(parameters.hill);
    let count_power = Expr::Pow(Box::new(count), Box::new(exponent.clone()));
    let k_power = HILL_K.powf(parameters.hill);
    let numerator = match kind {
        Edge::Activation => Expr::Mul(
            Box::new(Expr::Constant(parameters.strength)),
            Box::new(count_power.clone()),
        ),
        Edge::Inhibition => Expr::Constant(parameters.strength * k_power),
        Edge::Absent => unreachable!("absent edges have no parameters"),
    };
    Expr::Div(
        Box::new(numerator),
        Box::new(Expr::Add(
            Box::new(Expr::Constant(k_power)),
            Box::new(count_power),
        )),
    )
}

fn production_expression(topology: &Topology, kinetics: &Kinetics, target: usize) -> Expr {
    let mut expression = Expr::Constant(kinetics.basal[target]);
    for &(index, parameters) in &kinetics.edges {
        let (source, edge_target) = EDGE_INDEX[index];
        if edge_target != target {
            continue;
        }
        let kind = match topology.edges[index] {
            1 => Edge::Activation,
            2 => Edge::Inhibition,
            _ => unreachable!("decoded edge must be active"),
        };
        expression = Expr::Add(
            Box::new(expression),
            Box::new(hill_expression(source, kind, parameters)),
        );
    }
    expression
}

/// Evaluate one production propensity analytically; used by correctness tests
/// and model documentation.
pub fn production_rate(
    topology: &Topology,
    kinetics: &Kinetics,
    target: usize,
    state: &[isize; GENES],
) -> f64 {
    production_expression(topology, kinetics, target).eval(state)
}

/// Assemble the six-reaction runtime network.
pub fn build_model(topology: &Topology, kinetics: &Kinetics, seed: u64) -> Gillespie {
    let mut model = Gillespie::new_with_seed([INITIAL_COPIES; GENES], true, seed);
    for gene in 0..GENES {
        let mut production_jump = [0; GENES];
        production_jump[gene] = 1;
        model.add_reaction(
            Rate::expr(production_expression(topology, kinetics, gene)),
            production_jump,
        );

        let mut degradation_jump = [0; GENES];
        degradation_jump[gene] = -1;
        model.add_reaction(
            Rate::lma_sparse(kinetics.degradation[gene], [(gene as u32, 1)]),
            degradation_jump,
        );
    }
    model
}

/// One regularly sampled stochastic path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trace {
    pub seed: u64,
    pub time: Vec<f64>,
    pub values: Vec<[f64; GENES]>,
    pub runaway: bool,
}

/// Simulate burn-in and the fixed scoring window.
pub fn simulate(topology: &Topology, decision: &[f64], seed: u64) -> Result<Trace, &'static str> {
    let kinetics = Kinetics::decode(topology, decision)?;
    let mut model = build_model(topology, &kinetics, seed);
    model.advance_until(BURN_IN);
    let mut time = Vec::with_capacity(SAMPLE_COUNT);
    let mut values = Vec::with_capacity(SAMPLE_COUNT);
    let mut runaway = false;
    for sample in 1..=SAMPLE_COUNT {
        let current = BURN_IN + sample as f64 * SAMPLE_INTERVAL;
        model.advance_until(current);
        let state = [
            model.get_species(0) as f64,
            model.get_species(1) as f64,
            model.get_species(2) as f64,
        ];
        runaway |= state.iter().sum::<f64>() > MAX_MOLECULES
            || state.iter().any(|value| !value.is_finite() || *value < 0.0);
        time.push(current);
        values.push(state);
        if runaway {
            break;
        }
    }
    Ok(Trace {
        seed,
        time,
        values,
        runaway,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::REFERENCES;

    fn midpoint(topology: &Topology) -> Vec<f64> {
        let (lower, upper) = bounds(topology);
        lower
            .into_iter()
            .zip(upper)
            .map(|(left, right)| 0.5 * (left + right))
            .collect()
    }

    #[test]
    fn runtime_network_has_three_species_and_six_reactions() {
        let topology = REFERENCES[0].1;
        let kinetics = Kinetics::decode(&topology, &midpoint(&topology)).unwrap();
        let model = build_model(&topology, &kinetics, 42);
        assert_eq!(model.nb_species(), 3);
        assert_eq!(model.nb_reactions(), 6);
    }

    #[test]
    fn activation_rises_and_inhibition_falls() {
        let topology = REFERENCES[0].1;
        let kinetics = Kinetics::decode(&topology, &midpoint(&topology)).unwrap();
        // C inhibits A in the canonical repressilator.
        let low = production_rate(&topology, &kinetics, 0, &[10, 10, 0]);
        let high = production_rate(&topology, &kinetics, 0, &[10, 10, 100]);
        assert!(low > high);

        let activation = REFERENCES[2].1;
        let kinetics = Kinetics::decode(&activation, &midpoint(&activation)).unwrap();
        let low = production_rate(&activation, &kinetics, 1, &[0, 10, 10]);
        let high = production_rate(&activation, &kinetics, 1, &[100, 10, 10]);
        assert!(high > low);
    }

    #[test]
    fn absent_source_does_not_change_a_propensity() {
        let topology = REFERENCES[0].1;
        let kinetics = Kinetics::decode(&topology, &midpoint(&topology)).unwrap();
        let first = production_rate(&topology, &kinetics, 0, &[0, 10, 20]);
        let second = production_rate(&topology, &kinetics, 0, &[1000, 999, 20]);
        assert_eq!(first, second);
    }

    #[test]
    fn seeded_runtime_replays_bit_for_bit() {
        let topology = REFERENCES[0].1;
        let decision = midpoint(&topology);
        let first = simulate(&topology, &decision, 123).unwrap();
        let second = simulate(&topology, &decision, 123).unwrap();
        assert_eq!(first.values, second.values);
    }
}
