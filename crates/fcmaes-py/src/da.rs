//! PyO3 binding for Dual Annealing: the `optimize_da` free function, backed by
//! `fcmaes_core::optimize_da`. DA has no ask/tell interface.

use fcmaes_core::{DaParams, optimize_da};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{F64Vec, PyObjective, check_box, check_vector, result_tuple, slice_or_vec};

#[allow(clippy::too_many_arguments)]
/// Minimize a bounded scalar objective with Dual Annealing.
///
/// ``fun(x)`` receives a one-dimensional decoded ``float64`` array. ``lower``
/// and ``upper`` must match ``guess`` in length, or both be empty for an
/// unbounded search. Set ``use_local_search`` to enable the bounded
/// finite-difference quasi-Newton phase after annealing steps.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``.
///
/// ``evaluations`` counts the objective calls actually made and never exceeds
/// ``max_evaluations``, which includes the attempts to find a start point.
/// ``stop == -1`` means that no start point with an objective value below
/// ``1e16`` was found within 1,000 attempts or the budget; ``x`` and ``fun``
/// then describe the best point evaluated (``fun`` is the largest float if
/// every value was NaN).
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have one entry per
/// element of ``guess`` (both may be empty for an unbounded search) or do not
/// satisfy finite ``lower < upper``, or if ``guess`` is empty or not finite.
/// Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. As in fcmaes, an exception raised by the
/// objective callback is not re-raised: that call, like a non-finite result,
/// counts as a failed evaluation and ranks worst. The run's first exception or
/// unusable return value is logged once as a warning on the ``fcmaes_rust``
/// logger; non-finite values are not logged.
#[pyfunction]
#[pyo3(name = "optimize_da")]
#[pyo3(signature = (fun, guess, lower, upper, *, seed, runid=0,
    max_evaluations=100000, use_local_search=true))]
pub fn optimize_da_py<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    guess: F64Vec<'_>,
    lower: F64Vec<'_>,
    upper: F64Vec<'_>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    use_local_search: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let guess = slice_or_vec(&guess);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    check_box("Dual Annealing", guess.len(), &lower, &upper, true)?;
    check_vector("Dual Annealing", "guess", &guess, &[guess.len()])?;
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
