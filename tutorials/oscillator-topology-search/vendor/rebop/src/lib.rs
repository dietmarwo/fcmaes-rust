//! Narrow compatibility copy of the ReBop 0.9.7 runtime API.
//!
//! The upstream crate keeps [`Expr`] private even though
//! `gillespie::Rate::expr` accepts it. This copy exports that existing type so
//! dependent Rust code can construct runtime Hill propensities.

pub mod expr;
pub mod gillespie;

pub use expr::Expr;
