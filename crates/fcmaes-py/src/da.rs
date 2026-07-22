//! PyO3 binding for Dual Annealing: the `optimize_da` free function, backed by
//! `fcmaes_core::optimize_da`. DA has no ask/tell interface.

use fcmaes_core::{DaParams, optimize_da};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{PyObjective, result_tuple, slice_or_vec};

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "optimize_da")]
#[pyo3(signature = (fun, guess, lower, upper, *, seed, runid=0,
    max_evaluations=100000, use_local_search=true))]
pub fn optimize_da_py<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    guess: PyReadonlyArray1<f64>,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    use_local_search: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let guess = slice_or_vec(&guess);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let params = DaParams {
        max_evaluations,
        use_local_search,
        seed,
        runid,
    };
    let obj = PyObjective::new(fun);
    let result = py.allow_threads(move || optimize_da(&obj, &guess, lower, upper, &params));
    result_tuple(
        py,
        &result.x,
        result.y,
        result.evaluations,
        result.iterations,
        result.stop,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_da_py, m)?)?;
    Ok(())
}
