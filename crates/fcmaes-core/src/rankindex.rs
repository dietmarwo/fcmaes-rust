// Derived from two C++ programs: grid_point_index.cpp (sparse uniform grid in
// rank space, Copyright (c) 2026 Aleksey Vaneev) and borda_box_query.cpp
// (Borda rank-band selector, no author notice). Neither is part of the
// published BiteOpt repository.
// Rust port: Copyright (c) 2026 Dietmar Wolz.
// MIT license; keep NOTICE.fronts with copies of this file.
//! Immutable rank-space indexes for exact closed, axis-aligned box queries.
//! Build once over a finite row-major matrix, then query from any thread.
//!
//! Ranks sort each axis by coordinate; equal coordinates (including `-0.0`
//! and `0.0`) keep ascending row order. Borda returns hits in that order on
//! the axis with the shortest rank interval. Grid returns cell-key, then
//! row-ID order. Both report every matching row when coordinates tie, and
//! both give the same IDs, in the same order, as the C++ indexes of
//! fast-cma-es (`fcmaes.rankindex`).

mod borda;
mod grid;
mod rank;

pub use borda::{BordaIndex, BordaResult};
pub use grid::{GridIndex, GridResult};
