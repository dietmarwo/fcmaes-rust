//! PyO3 bindings for fcmaes.
//!
//! The extension exposes optimizer, retry, and GTOP bindings through the
//! private `fcmaes_rust._fcmaes_ext` module. The public Python facade lives
//! in `python/fcmaes_rust/__init__.py`.

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

/// Return native-extension build information.
///
/// The dictionary contains the extension module name, implementation backend,
/// core and binding versions, and compatibility flags. This is useful in bug
/// reports and installation smoke tests.
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

/// Sum a contiguous one-dimensional ``float64`` array in the Rust core.
///
/// This internal function is retained as an installation probe. Applications
/// should not use it as a numerical reduction API.
#[pyfunction]
fn _phase1_probe_sum(values: PyReadonlyArray1<'_, f64>) -> f64 {
    let slice = values.as_slice().unwrap_or(&[]);
    fcmaes_core::probe_sum(slice)
}

/// Native implementation backing the public :mod:`fcmaes_rust` facade.
///
/// Import :mod:`fcmaes_rust` in application code. The extension module remains
/// available as ``fcmaes_rust.native`` for signature inspection and advanced
/// use.
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
