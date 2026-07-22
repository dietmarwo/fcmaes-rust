//! Shared PyO3 helpers used by the optimizer bindings.

use fcmaes_core::Objective;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Copy a read-only 2-D float array into row vectors.
pub fn matrix_rows(a: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    a.as_array().outer_iter().map(|r| r.to_vec()).collect()
}

/// Objective backed by a Python callable `fun(x: np.ndarray) -> float`.
///
/// `eval` re-acquires the GIL per call (optimizer loops run under
/// `py.allow_threads`), mirroring the C++ `without_gil` + `gil_scoped_acquire`
/// trampoline and avoiding a rayon/GIL deadlock. Non-finite / failed calls
/// return `NaN`, which the core sanitizes to its large-value sentinel.
pub struct PyObjective {
    fun: Py<PyAny>,
}

impl PyObjective {
    pub fn new(fun: Py<PyAny>) -> Self {
        Self { fun }
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
                Ok(v) => v.extract::<f64>(py).unwrap_or(f64::NAN),
                Err(_) => f64::NAN,
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
