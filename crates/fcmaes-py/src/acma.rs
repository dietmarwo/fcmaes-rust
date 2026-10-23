//! PyO3 bindings for active CMA-ES: the `optimize_acma` free function and the
//! `ACMA` ask/tell class, backed by `fcmaes_core::Cmaes`.

use fcmaes_core::{Cmaes, CmaesParams, Fitness};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{
    F64Mat, F64Vec, PyObjective, check_box, check_vector, result_tuple, rows_to_pyarray,
    slice_or_vec,
};

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

/// Validate the inputs `Cmaes::new` would otherwise panic on.
fn check_acma_inputs(
    dim: usize,
    guess: &[f64],
    lower: &[f64],
    upper: &[f64],
    sigma: &[f64],
) -> PyResult<()> {
    check_box("ACMA", dim, lower, upper, true)?;
    check_vector("ACMA", "guess", guess, &[dim])?;
    check_vector("ACMA", "sigma", sigma, &[1, dim])
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
/// Minimize a scalar objective with active CMA-ES.
///
/// ``fun(x)`` receives one decoded candidate and returns a scalar to minimize.
/// ``guess`` defines the initial center and ``sigma`` is either one shared
/// standard deviation or one value per coordinate. Empty bounds select an
/// unbounded search; otherwise ``lower`` and ``upper`` must match ``guess``.
///
/// Candidate evaluation can use ``workers`` native threads, but Python
/// callbacks still reacquire the GIL. ``batch_fun`` and ``delayed_update`` are
/// currently accepted for API compatibility and do not change evaluation.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``.
///
/// ``evaluations`` counts the objective calls actually made.
/// ``max_evaluations`` is checked before each generation and whole generations
/// are evaluated, so a run can make up to ``popsize - 1`` calls more.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have one entry per
/// element of ``guess`` (both may be empty for an unbounded search) or do not
/// satisfy finite ``lower < upper``, or if ``guess`` or ``sigma`` holds
/// non-finite values or ``sigma`` has neither one nor ``len(guess)`` entries.
/// Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. As in fcmaes, an exception raised by the
/// objective callback is not re-raised: that call, like a non-finite result,
/// counts as a failed evaluation and ranks worst.
#[pyfunction]
#[pyo3(signature = (fun, batch_fun, guess, lower, upper, sigma, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, stop_hist=-1.0, mu=0,
    popsize=31, accuracy=1.0, normalize=true, delayed_update=true, update_gap=-1,
    workers=1))]
pub fn optimize_acma<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    batch_fun: Py<PyAny>,
    guess: F64Vec<'_>,
    lower: F64Vec<'_>,
    upper: F64Vec<'_>,
    sigma: F64Vec<'_>,
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
    check_acma_inputs(dim, &guess, &lower, &upper, &sigma)?;

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

/// Stateful active CMA-ES with an external ask/tell evaluator.
///
/// Construct the search distribution from ``guess`` and ``sigma``, then
/// alternate :meth:`ask` with :meth:`tell` or :meth:`tell_x`. Bounds may be
/// empty for an unbounded search.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have one entry per
/// element of ``guess`` (both may be empty for an unbounded search) or do not
/// satisfy finite ``lower < upper``, or if ``guess`` or ``sigma`` holds
/// non-finite values or ``sigma`` has neither one nor ``len(guess)`` entries.
/// Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. :meth:`tell` raises ``ValueError`` if it
/// does not receive exactly one value per row returned by the most recent
/// :meth:`ask`.
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
        guess: F64Vec<'_>,
        lower: F64Vec<'_>,
        upper: F64Vec<'_>,
        sigma: F64Vec<'_>,
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
    ) -> PyResult<Self> {
        let _ = delayed_update;
        let guess = slice_or_vec(&guess);
        let sigma = slice_or_vec(&sigma);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let dim = guess.len();
        check_acma_inputs(dim, &guess, &lower, &upper, &sigma)?;
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
        Ok(ACMA {
            inner: Cmaes::new(fitness, &guess, &sigma, &params),
        })
    }

    /// Return the next decoded offspring population, shape ``(popsize, dim)``.
    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let pop = self.inner.ask();
        rows_to_pyarray(py, &pop)
    }

    /// Update the distribution with values for the most recent :meth:`ask`.
    ///
    /// ``ys`` must contain one minimized scalar value per offspring. Returns
    /// the current stop code.
    fn tell(&mut self, ys: F64Vec<'_>) -> i32 {
        self.inner.tell(&slice_or_vec(&ys))
    }

    /// Update with externally supplied candidate rows and their objective values.
    ///
    /// ``xs`` has shape ``(popsize, dim)`` and ``ys`` has length ``popsize``.
    /// This supports evaluators that repair or otherwise modify candidates.
    fn tell_x(&mut self, ys: F64Vec<'_>, xs: F64Mat<'_>) -> i32 {
        let ys = slice_or_vec(&ys);
        let arr = xs.as_array();
        let rows: Vec<Vec<f64>> = arr.outer_iter().map(|r| r.to_vec()).collect();
        self.inner.tell_x(&ys, &rows)
    }

    /// Return the current decoded offspring population.
    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.population())
    }

    /// Return ``(x, fun, evaluations, iterations, stop)`` for the best point.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        acma_result_tuple(py, &self.inner.result())
    }

    /// Number of decision variables.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    /// Number of offspring in each generation.
    #[getter]
    fn popsize(&self) -> usize {
        self.inner.popsize()
    }
    /// Current termination code; zero means optimization is still active.
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
