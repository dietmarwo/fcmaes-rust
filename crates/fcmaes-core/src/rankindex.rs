//! Immutable rank-space indexes for exact closed, axis-aligned box queries.
//! Build once over a finite row-major matrix, then query from any thread.
//! Borda returns shortest-axis rank order. Grid returns cell-key then row-ID order.
//! Both retain every matching row when coordinates tie.

mod borda;
mod grid;
mod rank;

pub use borda::{BordaIndex, BordaResult};
pub use grid::{GridIndex, GridResult};
