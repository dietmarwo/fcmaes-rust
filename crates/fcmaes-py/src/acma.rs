//! PyO3 bindings for active CMA-ES: the `optimize_acma` free function and the
//! `ACMA` ask/tell class, backed by `fcmaes_core::Cmaes`.

use fcmaes_core::{Cmaes, CmaesParams, Fitness};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{PyObjective, result_tuple, rows_to_pyarray, slice_or_vec};

fn build_fitness(dim: usize, lower: &[f64], upper: &[f64], normalize: bool) -> (Fitness, bool) {
    if lower.is_empty() {
        // Unbounded: normalization is disabled (matches the C++ path).
        (Fitness::new(dim, 1, vec![], vec![]), false)
    } else {
        let mut f = Fitness::bounded(dim, 1, lower, upper);
        f.set_normalize(normalize);
        (f, normalize)
    }
}

fn acma_result_tuple<'py>(
    py: Python<'py>,
    r: &fcmaes_core::AcmaResult,
) -> PyResult<Bound<'py, PyTuple>> {
    result_tuple(py, &r.x, r.y, r.evaluations, r.iterations, r.stop)
}

#[allow(clippy::too_many_arguments)]
fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    stop_hist: f64,
    mu: i32,
    popsize: i32,
    accuracy: f64,
    update_gap: i32,
) -> CmaesParams {
    CmaesParams {
        popsize,
        mu,
        max_evaluations,
        accuracy,
        stop_fitness,
        stop_tol_hist_fun: stop_hist,
        update_gap,
        seed,
        runid,
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (fun, batch_fun, guess, lower, upper, sigma, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, stop_hist=-1.0, mu=0,
    popsize=31, accuracy=1.0, normalize=true, delayed_update=true, update_gap=-1,
    workers=1))]
pub fn optimize_acma<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    batch_fun: Py<PyAny>,
    guess: PyReadonlyArray1<f64>,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    stop_hist: f64,
    mu: i32,
    popsize: i32,
    accuracy: f64,
    normalize: bool,
    delayed_update: bool,
    update_gap: i32,
    workers: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = (batch_fun, delayed_update); // accepted for API compatibility
    let guess = slice_or_vec(&guess);
    let sigma = slice_or_vec(&sigma);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let dim = guess.len();

    let (fitness, _) = build_fitness(dim, &lower, &upper, normalize);
    let params = make_params(
        seed,
        runid,
        max_evaluations,
        stop_fitness,
        stop_hist,
        mu,
        popsize,
        accuracy,
        update_gap,
    );
    let obj = PyObjective::new(fun);

    let result = py.allow_threads(move || {
        let mut opt = Cmaes::new(fitness, &guess, &sigma, &params);
        opt.optimize(&obj, workers)
    });
    acma_result_tuple(py, &result)
}

/// Stateful active CMA-ES with an ask/tell interface.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct ACMA {
    inner: Cmaes,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl ACMA {
    #[new]
    #[pyo3(signature = (guess, lower, upper, sigma, *, max_evaluations=100000,
        stop_fitness=f64::NEG_INFINITY, stop_hist=-1.0, mu=0, popsize=31,
        accuracy=1.0, seed, runid=0, normalize=true, delayed_update=true,
        update_gap=-1))]
    fn new(
        guess: PyReadonlyArray1<f64>,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        sigma: PyReadonlyArray1<f64>,
        max_evaluations: u64,
        stop_fitness: f64,
        stop_hist: f64,
        mu: i32,
        popsize: i32,
        accuracy: f64,
        seed: u64,
        runid: i64,
        normalize: bool,
        delayed_update: bool,
        update_gap: i32,
    ) -> Self {
        let _ = delayed_update;
        let guess = slice_or_vec(&guess);
        let sigma = slice_or_vec(&sigma);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let dim = guess.len();
        let (fitness, _) = build_fitness(dim, &lower, &upper, normalize);
        let params = make_params(
            seed,
            runid,
            max_evaluations,
            stop_fitness,
            stop_hist,
            mu,
            popsize,
            accuracy,
            update_gap,
        );
        ACMA {
            inner: Cmaes::new(fitness, &guess, &sigma, &params),
        }
    }

    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let pop = self.inner.ask();
        rows_to_pyarray(py, &pop)
    }

    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> i32 {
        self.inner.tell(&slice_or_vec(&ys))
    }

    fn tell_x(&mut self, ys: PyReadonlyArray1<f64>, xs: PyReadonlyArray2<f64>) -> i32 {
        let ys = slice_or_vec(&ys);
        let arr = xs.as_array();
        let rows: Vec<Vec<f64>> = arr.outer_iter().map(|r| r.to_vec()).collect();
        self.inner.tell_x(&ys, &rows)
    }

    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.population())
    }

    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        acma_result_tuple(py, &self.inner.result())
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
    m.add_function(wrap_pyfunction!(optimize_acma, m)?)?;
    m.add_class::<ACMA>()?;
    Ok(())
}
