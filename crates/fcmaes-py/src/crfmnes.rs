//! PyO3 bindings for CR-FM-NES: the `optimize_crfmnes` free function (batch
//! objective) and the `CRFMNES` ask/tell class, backed by `fcmaes_core::Crfmnes`.

use fcmaes_core::{Crfmnes, CrfmnesParams, Fitness};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{result_tuple, rows_to_pyarray, slice_or_vec};

fn build_fitness(dim: usize, lower: &[f64], upper: &[f64], normalize: bool) -> Fitness {
    if lower.is_empty() {
        Fitness::new(dim, 1, vec![], vec![])
    } else {
        let mut f = Fitness::bounded(dim, 1, lower, upper);
        f.set_normalize(normalize);
        f
    }
}

fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    penalty_coef: f64,
    use_constraint_violation: bool,
) -> CrfmnesParams {
    CrfmnesParams {
        popsize,
        max_evaluations,
        stop_fitness,
        penalty_coef,
        use_constraint_violation,
        seed,
        runid,
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (batch_fun, guess, lower, upper, sigma=0.3, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, popsize=32,
    penalty_coef=1e5, use_constraint_violation=true, normalize=false))]
pub fn optimize_crfmnes<'py>(
    py: Python<'py>,
    batch_fun: Py<PyAny>,
    guess: PyReadonlyArray1<f64>,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    sigma: f64,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    penalty_coef: f64,
    use_constraint_violation: bool,
    normalize: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let guess = slice_or_vec(&guess);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let dim = guess.len();

    let fitness = build_fitness(dim, &lower, &upper, normalize);
    let params = make_params(
        seed,
        runid,
        max_evaluations,
        stop_fitness,
        popsize,
        penalty_coef,
        use_constraint_violation,
    );

    let result = py.allow_threads(move || {
        let mut opt = Crfmnes::new(fitness, &guess, sigma, &params);
        opt.optimize_batch(|rows| eval_batch(&batch_fun, rows))
    });
    result_tuple(
        py,
        &result.x,
        result.y,
        result.evaluations,
        result.iterations,
        result.stop,
    )
}

/// Evaluate a decoded population through the Python batch callable, re-acquiring
/// the GIL for the call (the loop runs under `py.allow_threads`).
fn eval_batch(batch_fun: &Py<PyAny>, rows: &[Vec<f64>]) -> Vec<f64> {
    Python::with_gil(|py| {
        let arr = rows_to_pyarray(py, rows);
        match batch_fun.call1(py, (arr,)) {
            Ok(v) => v
                .extract::<Vec<f64>>(py)
                .unwrap_or_else(|_| vec![f64::NAN; rows.len()]),
            Err(_) => vec![f64::NAN; rows.len()],
        }
    })
}

/// Stateful CR-FM-NES with an ask/tell interface.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct CRFMNES {
    inner: Crfmnes,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl CRFMNES {
    #[new]
    #[pyo3(signature = (guess, lower, upper, sigma=0.3, popsize=32, *, seed,
        runid=0, penalty_coef=1e5, use_constraint_violation=true, normalize=false))]
    fn new(
        guess: PyReadonlyArray1<f64>,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        sigma: f64,
        popsize: i32,
        seed: u64,
        runid: i64,
        penalty_coef: f64,
        use_constraint_violation: bool,
        normalize: bool,
    ) -> Self {
        let guess = slice_or_vec(&guess);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let dim = guess.len();
        let fitness = build_fitness(dim, &lower, &upper, normalize);
        let params = make_params(
            seed,
            runid,
            0,
            f64::NEG_INFINITY,
            popsize,
            penalty_coef,
            use_constraint_violation,
        );
        CRFMNES {
            inner: Crfmnes::new(fitness, &guess, sigma, &params),
        }
    }

    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask_pop())
    }

    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> i32 {
        self.inner.tell_pop(&slice_or_vec(&ys))
    }

    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.population())
    }

    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let r = self.inner.result();
        result_tuple(py, &r.x, r.y, r.evaluations, r.iterations, r.stop)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn popsize(&self) -> usize {
        self.inner.popsize()
    }
    #[getter]
    fn stop(&self) -> i32 {
        self.inner.stop()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_crfmnes, m)?)?;
    m.add_class::<CRFMNES>()?;
    Ok(())
}
