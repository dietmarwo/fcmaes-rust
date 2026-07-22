//! PyO3 bindings for Differential Evolution: the `optimize_de` free function
//! and the `DE` ask/tell class, backed by `fcmaes_core::De`.

use fcmaes_core::{De, DeParams, Fitness};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{PyObjective, result_tuple, rows_to_pyarray, slice_or_vec};

fn build_fitness(dim: usize, lower: &[f64], upper: &[f64]) -> Fitness {
    if lower.is_empty() {
        Fitness::new(dim, 1, vec![], vec![])
    } else {
        Fitness::bounded(dim, 1, lower, upper)
    }
}

fn opt_ints(ints: &PyReadonlyArray1<bool>) -> Option<Vec<bool>> {
    if ints.len().unwrap_or(0) == 0 {
        None
    } else {
        Some(ints.as_array().iter().copied().collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    keep: f64,
    stop_fitness: f64,
    popsize: i32,
    f: f64,
    cr: f64,
    min_sigma: f64,
    min_mutate: f64,
    max_mutate: f64,
) -> DeParams {
    DeParams {
        popsize,
        max_evaluations,
        keep,
        stop_fitness,
        f,
        cr,
        min_mutate,
        max_mutate,
        min_sigma,
        seed,
        runid,
    }
}

#[allow(clippy::too_many_arguments, non_snake_case)]
#[pyfunction]
#[pyo3(signature = (fun, dim, lower, upper, guess, sigma, ints, *, seed, runid=0,
    max_evaluations=100000, keep=200.0, stop_fitness=f64::NEG_INFINITY, popsize=31,
    F=0.5, CR=0.9, min_sigma=0.0, min_mutate=0.1, max_mutate=0.5, workers=1,
    terminate=None))]
pub fn optimize_de<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    dim: usize,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    guess: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    ints: PyReadonlyArray1<bool>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    keep: f64,
    stop_fitness: f64,
    popsize: i32,
    F: f64,
    CR: f64,
    min_sigma: f64,
    min_mutate: f64,
    max_mutate: f64,
    workers: i32,
    terminate: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = (workers, terminate); // accepted for API compatibility
    let (f, cr) = (F, CR);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let guess = slice_or_vec(&guess);
    let sigma = slice_or_vec(&sigma);
    let ints = opt_ints(&ints);

    let fitness = build_fitness(dim, &lower, &upper);
    let params = make_params(
        seed,
        runid,
        max_evaluations,
        keep,
        stop_fitness,
        popsize,
        f,
        cr,
        min_sigma,
        min_mutate,
        max_mutate,
    );
    let obj = PyObjective::new(fun);

    let result = py.allow_threads(move || {
        let mut opt = De::new(fitness, &guess, &sigma, ints, &params);
        opt.optimize(&obj)
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

/// Stateful DE with an ask/tell interface.
#[pyclass]
pub struct DE {
    inner: De,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl DE {
    #[new]
    #[allow(non_snake_case)]
    #[pyo3(signature = (dim, lower, upper, guess, sigma, ints, popsize=31,
        keep=200.0, F=0.5, CR=0.9, min_sigma=0.0, min_mutate=0.1, max_mutate=0.5,
        *, seed, runid=0))]
    fn new(
        dim: usize,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        guess: PyReadonlyArray1<f64>,
        sigma: PyReadonlyArray1<f64>,
        ints: PyReadonlyArray1<bool>,
        popsize: i32,
        keep: f64,
        F: f64,
        CR: f64,
        min_sigma: f64,
        min_mutate: f64,
        max_mutate: f64,
        seed: u64,
        runid: i64,
    ) -> Self {
        let (f, cr) = (F, CR);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let guess = slice_or_vec(&guess);
        let sigma = slice_or_vec(&sigma);
        let ints = opt_ints(&ints);
        let fitness = build_fitness(dim, &lower, &upper);
        let params = make_params(
            seed,
            runid,
            0,
            keep,
            f64::NEG_INFINITY,
            popsize,
            f,
            cr,
            min_sigma,
            min_mutate,
            max_mutate,
        );
        DE {
            inner: De::new(fitness, &guess, &sigma, ints, &params),
        }
    }

    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask())
    }

    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> i32 {
        self.inner.tell(&slice_or_vec(&ys))
    }

    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
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
    m.add_function(wrap_pyfunction!(optimize_de, m)?)?;
    m.add_class::<DE>()?;
    Ok(())
}
