//! Self-contained problem modules of the examples crate, compiled from their
//! source files. The examples crate itself is not a dependency (it pulls in
//! tokio and a finance client); these modules use only `std` and
//! `fcmaes_core`, so the objectives are the examples' own code, unchanged.

#[path = "../../../../examples/src/jobshop.rs"]
pub mod jobshop;

#[path = "../../../../examples/src/material_flow_planning.rs"]
pub mod material_flow_planning;

#[path = "../../../../examples/src/uav.rs"]
pub mod uav;
