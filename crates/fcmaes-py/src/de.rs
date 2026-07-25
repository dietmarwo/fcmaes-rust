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
/// Minimize a scalar objective with Differential Evolution.
///
/// ``fun(x)`` receives a one-dimensional ``float64`` NumPy array and must
/// return a scalar to minimize. ``lower`` and ``upper`` define the finite
/// search box; ``guess`` and ``sigma`` may be empty. Set entries of ``ints`` to
/// true for integer coordinates. ``seed`` and ``runid`` make the run
/// reproducible.
///
/// ``workers`` and ``terminate`` are accepted for compatibility; this
/// one-shot Python-callback path currently evaluates serially because each
/// callback reacquires the GIL.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``. ``stop == 1`` means
/// ``stop_fitness`` was reached.
///
/// Raises ``ValueError`` if the bound arrays are empty, of unequal length, or
/// do not satisfy finite ``lower < upper``, or if ``ints`` does not have one
/// entry per decision variable, and ``TypeError`` if an array argument cannot
/// be converted to contiguous ``float64``. Exceptions raised inside the
/// objective callback propagate to the caller.
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

/// Stateful Differential Evolution with an external ask/tell evaluator.
///
/// The constructor arguments match :func:`optimize_de` except that evaluation
/// budgets and objective callbacks belong to the caller. Call :meth:`ask`,
/// evaluate every returned row in the same order, and pass the scalar values
/// to :meth:`tell`.
///
/// Raises ``ValueError`` if the bound arrays do not match the decision
/// dimension or do not satisfy finite ``lower < upper``, and ``TypeError`` if
/// an array argument cannot be converted to contiguous ``float64``.
/// :meth:`tell` raises ``ValueError`` if it does not receive exactly one value
/// per row returned by the most recent :meth:`ask`.
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

    /// Return the next decoded candidate population, shape ``(popsize, dim)``.
    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask())
    }

    /// Submit one minimized objective value per row returned by :meth:`ask`.
    ///
    /// Returns the optimizer stop code.
    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> i32 {
        self.inner.tell(&slice_or_vec(&ys))
    }

    /// Return the current decoded population, shape ``(popsize, dim)``.
    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.population())
    }

    /// Return ``(x, fun, evaluations, iterations, stop)`` for the best point.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let r = self.inner.result();
        result_tuple(py, &r.x, r.y, r.evaluations, r.iterations, r.stop)
    }

    /// Number of decision variables.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    /// Number of candidates in each ask/tell population.
    #[getter]
    fn popsize(&self) -> usize {
        self.inner.popsize()
    }
    /// Current termination code; zero means no stop criterion has fired.
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
