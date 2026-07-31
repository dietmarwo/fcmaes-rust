//! Runtime propensity expression from ReBop 0.9.7.

/// An arithmetic expression evaluated against integer species counts.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Constant(f64),
    Concentration(usize),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Exp(Box<Expr>),
}

impl Expr {
    /// Evaluate the expression for one state.
    pub fn eval(&self, species: &[isize]) -> f64 {
        match self {
            Self::Constant(value) => *value,
            Self::Concentration(index) => species[*index] as f64,
            Self::Neg(value) => -value.eval(species),
            Self::Add(left, right) => left.eval(species) + right.eval(species),
            Self::Sub(left, right) => left.eval(species) - right.eval(species),
            Self::Mul(left, right) => left.eval(species) * right.eval(species),
            Self::Div(left, right) => left.eval(species) / right.eval(species),
            Self::Pow(left, right) => left.eval(species).powf(right.eval(species)),
            Self::Max(left, right) => left.eval(species).max(right.eval(species)),
            Self::Min(left, right) => left.eval(species).min(right.eval(species)),
            Self::Exp(value) => value.eval(species).exp(),
        }
    }
}
