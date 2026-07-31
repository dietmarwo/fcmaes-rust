//! Function-based Gillespie direct-method API derived from ReBop 0.9.7.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::Exp1;

pub use crate::expr::Expr;

/// A mass-action or runtime expression propensity.
#[derive(Clone, Debug, PartialEq)]
pub enum Rate {
    LmaSparse(f64, Vec<(u32, u32)>),
    Expr(Expr),
}

impl Rate {
    /// Construct a sparse law-of-mass-action propensity.
    pub fn lma_sparse<V: AsRef<[(u32, u32)]>>(rate: f64, stoichiometry: V) -> Self {
        Self::LmaSparse(rate, stoichiometry.as_ref().to_vec())
    }

    /// Construct a runtime expression propensity.
    pub fn expr(expr: Expr) -> Self {
        Self::Expr(expr)
    }

    fn value(&self, species: &[isize]) -> f64 {
        match self {
            Self::LmaSparse(base, terms) => {
                let mut value = *base;
                for &(index, exponent) in terms {
                    let count = species[index as usize];
                    for factor in count + 1 - exponent as isize..=count {
                        value *= factor as f64;
                    }
                }
                value
            }
            Self::Expr(expr) => expr.eval(species),
        }
    }
}

/// A runtime well-mixed stochastic reaction network.
#[derive(Clone, Debug)]
pub struct Gillespie {
    species: Vec<isize>,
    time: f64,
    reactions: Vec<(Rate, Vec<(usize, isize)>)>,
    rng: SmallRng,
}

impl Gillespie {
    /// Construct a seeded runtime model.
    pub fn new_with_seed<V: AsRef<[isize]>>(species: V, _sparse: bool, seed: u64) -> Self {
        Self {
            species: species.as_ref().to_vec(),
            time: 0.0,
            reactions: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Add one reaction and its dense state change.
    pub fn add_reaction<V: AsRef<[isize]>>(&mut self, rate: Rate, differences: V) {
        assert_eq!(differences.as_ref().len(), self.species.len());
        let sparse = differences
            .as_ref()
            .iter()
            .enumerate()
            .filter_map(|(index, &change)| (change != 0).then_some((index, change)))
            .collect();
        self.reactions.push((rate, sparse));
    }

    /// Return the number of species.
    pub fn nb_species(&self) -> usize {
        self.species.len()
    }

    /// Return the number of reactions.
    pub fn nb_reactions(&self) -> usize {
        self.reactions.len()
    }

    /// Return one species count.
    pub fn get_species(&self, index: usize) -> isize {
        self.species[index]
    }

    /// Return the current model time.
    pub fn get_time(&self) -> f64 {
        self.time
    }

    /// Advance with Gillespie's direct method until `maximum`.
    pub fn advance_until(&mut self, maximum: f64) {
        let mut cumulative = vec![0.0; self.reactions.len()];
        loop {
            let mut total = 0.0;
            for ((rate, _), slot) in self.reactions.iter().zip(&mut cumulative) {
                let value = rate.value(&self.species);
                if !value.is_finite() || value < 0.0 {
                    self.time = maximum;
                    return;
                }
                total += value;
                *slot = total;
            }
            if total <= 0.0 {
                self.time = maximum;
                return;
            }
            let event_time = self.time + self.rng.sample::<f64, _>(Exp1) / total;
            if event_time > maximum {
                self.time = maximum;
                return;
            }
            self.time = event_time;
            let selected = total * self.rng.random::<f64>();
            let reaction = cumulative.partition_point(|&value| value < selected);
            for &(index, change) in &self.reactions[reaction].1 {
                self.species[index] += change;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_models_replay() {
        let mut first = Gillespie::new_with_seed([10], true, 42);
        first.add_reaction(Rate::expr(Expr::Constant(2.0)), [1]);
        first.add_reaction(Rate::lma_sparse(0.1, [(0, 1)]), [-1]);
        let mut second = first.clone();
        first.advance_until(20.0);
        second.advance_until(20.0);
        assert_eq!(first.get_species(0), second.get_species(0));
    }
}
