//! Shared PyO3 helpers used by the optimizer bindings.

use fcmaes_core::Objective;
use numpy::ndarray::Array2;
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::atomic::{AtomicBool, Ordering};

/// One-dimensional float argument: a ``float64`` NumPy array is borrowed
/// without a copy; lists, tuples and arrays of other numeric dtypes are
/// converted by NumPy, as the type stub (``Sequence[float] | NDArray``)
/// promises.
pub type F64Vec<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;

/// One-dimensional Boolean argument (the ``ints`` masks): a ``bool`` NumPy
/// array, or a sequence or array NumPy converts to ``bool``.
pub type BoolVec<'py> = PyArrayLike1<'py, bool, AllowTypeChange>;

/// Two-dimensional float argument: a NumPy matrix or a nested sequence of rows.
pub type F64Mat<'py> = PyArrayLike2<'py, f64, AllowTypeChange>;

/// Check a search box for `dim` decision variables: `lower` and `upper` must
/// have `dim` entries each with finite `lower < upper` (and a finite width).
/// With `allow_unbounded`, both may instead be empty.
pub fn check_box(
    optimizer: &str,
    dim: usize,
    lower: &[f64],
    upper: &[f64],
    allow_unbounded: bool,
) -> PyResult<()> {
    if dim == 0 {
        return Err(PyValueError::new_err(format!(
            "{optimizer} needs at least one decision variable"
        )));
    }
    if allow_unbounded && lower.is_empty() && upper.is_empty() {
        return Ok(());
    }
    if lower.len() != dim || upper.len() != dim {
        let or_empty = if allow_unbounded {
            " (or both be empty)"
        } else {
            ""
        };
        return Err(PyValueError::new_err(format!(
            "{optimizer} lower and upper must have {dim} entries each{or_empty}, got {} and {}",
            lower.len(),
            upper.len()
        )));
    }
    let invalid = |(&lo, &hi): (&f64, &f64)| {
        !lo.is_finite() || !hi.is_finite() || lo >= hi || !(hi - lo).is_finite()
    };
    if lower.iter().zip(upper).any(invalid) {
        return Err(PyValueError::new_err(format!(
            "{optimizer} bounds must be finite with lower < upper"
        )));
    }
    Ok(())
}

/// Check that a per-variable argument has one of the `allowed` lengths and
/// only finite values.
pub fn check_vector(
    optimizer: &str,
    name: &str,
    values: &[f64],
    allowed: &[usize],
) -> PyResult<()> {
    check_length(optimizer, name, values.len(), allowed)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{optimizer} {name} must contain only finite values"
        )));
    }
    Ok(())
}

/// Check that a per-variable argument has one of the `allowed` lengths.
pub fn check_length(optimizer: &str, name: &str, len: usize, allowed: &[usize]) -> PyResult<()> {
    if allowed.contains(&len) {
        return Ok(());
    }
    let mut lengths = allowed.to_vec();
    lengths.sort_unstable();
    lengths.dedup();
    let expected = lengths
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" or ");
    Err(PyValueError::new_err(format!(
        "{optimizer} {name} must have {expected} entries, got {len}"
    )))
}

/// Copy a read-only 2-D float array into row vectors.
pub fn matrix_rows(a: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    a.as_array().outer_iter().map(|r| r.to_vec()).collect()
}

/// Name of the Python logger that reports failed objective callbacks.
pub const LOGGER: &str = "fcmaes_rust";

/// Reports failed Python callbacks once per optimization run.
///
/// As in fcmaes, a callback that raises, or returns something that is not a
/// value of the expected shape, does not abort the run: the call counts as a
/// failed evaluation and ranks worst. The first failure of a run is logged as
/// a warning on the `fcmaes_rust` logger, with the traceback of a raised
/// exception; later failures in the same run are not logged, so an objective
/// that fails millions of times produces one message.
#[derive(Default)]
pub struct FailureLog {
    logged: AtomicBool,
}

impl FailureLog {
    /// Record that `callback` raised `error`; logs only the run's first failure.
    pub fn raised(&self, py: Python<'_>, callback: &str, error: &PyErr) {
        if !self.logged.swap(true, Ordering::Relaxed) {
            let detail = format!("raised {}", describe(py, error));
            let _ = log_failure(py, callback, &detail, Some(error));
        }
    }

    /// Record that `callback` returned a value that could not be used.
    pub fn bad_value(
        &self,
        py: Python<'_>,
        callback: &str,
        value: &Bound<'_, PyAny>,
        expected: &str,
    ) {
        if !self.logged.swap(true, Ordering::Relaxed) {
            let shown = value
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_else(|_| "an unprintable value".into());
            let shown: String = if shown.chars().count() > 200 {
                shown.chars().take(200).chain("...".chars()).collect()
            } else {
                shown
            };
            let detail = format!("returned {shown}, expected {expected}");
            let _ = log_failure(py, callback, &detail, None);
        }
    }
}

fn describe(py: Python<'_>, error: &PyErr) -> String {
    let name = error
        .get_type(py)
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "an exception".into());
    match error.value(py).str() {
        Ok(message) if !message.to_string().is_empty() => format!("{name}: {message}"),
        _ => name,
    }
}

/// Log one failure; errors inside logging itself are ignored by the caller.
fn log_failure(
    py: Python<'_>,
    callback: &str,
    detail: &str,
    error: Option<&PyErr>,
) -> PyResult<()> {
    let logger = py.import("logging")?.call_method1("getLogger", (LOGGER,))?;
    let kwargs = PyDict::new(py);
    if let Some(error) = error {
        kwargs.set_item("exc_info", error.value(py))?;
    }
    logger.call_method(
        "warning",
        (
            "%s %s; the call counts as a failed evaluation (worst value). Further failed calls in this optimization run are not logged.",
            callback,
            detail,
        ),
        Some(&kwargs),
    )?;
    Ok(())
}

/// Objective backed by a Python callable `fun(x: np.ndarray) -> float`.
///
/// `eval` re-acquires the GIL per call (optimizer loops run under
/// `py.allow_threads`), mirroring the C++ `without_gil` + `gil_scoped_acquire`
/// trampoline and avoiding a rayon/GIL deadlock. Non-finite / failed calls
/// return `NaN`, which the core sanitizes to its large-value sentinel; the
/// first failed call of a run is logged ([`FailureLog`]).
pub struct PyObjective {
    fun: Py<PyAny>,
    failures: FailureLog,
}

impl PyObjective {
    pub fn new(fun: Py<PyAny>) -> Self {
        Self {
            fun,
            failures: FailureLog::default(),
        }
    }
}

impl Objective for PyObjective {
    fn nobj(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self.eval_scalar(x)]
    }

    fn eval_scalar(&self, x: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let arr = PyArray1::from_slice(py, x);
            match self.fun.call1(py, (arr,)) {
                Ok(v) => match v.extract::<f64>(py) {
                    Ok(y) => y,
                    Err(_) => {
                        self.failures
                            .bad_value(py, "objective fun", v.bind(py), "a float");
                        f64::NAN
                    }
                },
                Err(error) => {
                    self.failures.raised(py, "objective fun", &error);
                    f64::NAN
                }
            }
        })
    }
}

/// Copy a read-only 1-D float array into a `Vec`, tolerating non-contiguity.
pub fn slice_or_vec(a: &PyReadonlyArray1<f64>) -> Vec<f64> {
    match a.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => a.as_array().iter().copied().collect(),
    }
}

/// Build a `(popsize, dim)` numpy array from row vectors.
pub fn rows_to_pyarray<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let popsize = rows.len();
    let dim = if popsize == 0 { 0 } else { rows[0].len() };
    let mut flat = Vec::with_capacity(popsize * dim);
    for r in rows {
        flat.extend_from_slice(r);
    }
    Array2::from_shape_vec((popsize, dim), flat)
        .expect("consistent row lengths")
        .into_pyarray(py)
        .to_owned()
}

/// Standard optimizer result tuple `(x, y, evaluations, iterations, stop)`.
pub fn result_tuple<'py>(
    py: Python<'py>,
    x: &[f64],
    y: f64,
    evaluations: u64,
    iterations: i32,
    stop: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let x = PyArray1::from_slice(py, x);
    PyTuple::new(
        py,
        [
            x.into_any(),
            y.into_pyobject(py)?.into_any(),
            (evaluations as i64).into_pyobject(py)?.into_any(),
            iterations.into_pyobject(py)?.into_any(),
            stop.into_pyobject(py)?.into_any(),
        ],
    )
}
