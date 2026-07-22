//! PyO3 bindings for fcmaes.
//!
//! The extension exposes optimizer, retry, and GTOP bindings through the
//! existing `fcmaes._fcmaes_ext` import path.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod acma;
mod biteopt;
mod common;
mod crfmnes;
mod da;
mod de;
mod gtop;
mod mapelites;
mod mode;
mod moretry;
mod pgpe;
mod retry;

/// Small dict proving the Rust extension module was built and imported.
#[pyfunction]
fn phase1_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = PyDict::new(py);
    info.set_item("module", "_fcmaes_ext")?;
    info.set_item("phase", 0)?;
    info.set_item("backend", "rust")?;
    info.set_item("nanobind", false)?;
    info.set_item("core_version", fcmaes_core::CORE_VERSION)?;
    info.set_item("binding_version", env!("CARGO_PKG_VERSION"))?;
    Ok(info.into())
}

/// Internal smoke test: sum a 1-D float64 array through the Rust core.
#[pyfunction]
fn _phase1_probe_sum(values: PyReadonlyArray1<'_, f64>) -> f64 {
    let slice = values.as_slice().unwrap_or(&[]);
    fcmaes_core::probe_sum(slice)
}

#[pymodule]
fn _fcmaes_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phase1_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(_phase1_probe_sum, m)?)?;
    acma::register(m)?;
    biteopt::register(m)?;
    crfmnes::register(m)?;
    da::register(m)?;
    de::register(m)?;
    gtop::register(m)?;
    mapelites::register(m)?;
    mode::register(m)?;
    moretry::register(m)?;
    pgpe::register(m)?;
    retry::register(m)?;
    Ok(())
}
