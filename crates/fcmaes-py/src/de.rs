//! PyO3 bindings for Differential Evolution: the `optimize_de` free function
//! and the `DE` ask/tell class, backed by `fcmaes_core::De`.

use fcmaes_core::{De, DeParams, Fitness};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{
    BoolVec, F64Vec, PyObjective, check_box, check_length, check_vector, result_tuple,
    rows_to_pyarray, slice_or_vec,
};

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

/// Validate the inputs `De::new` would otherwise panic on. Without bounds,
/// the start distribution comes from `guess` and `sigma`, so both are needed.
fn check_de_inputs(
    dim: usize,
    lower: &[f64],
    upper: &[f64],
    guess: &[f64],
    sigma: &[f64],
    ints: Option<&[bool]>,
) -> PyResult<()> {
    check_box("DE", dim, lower, upper, !guess.is_empty())?;
    check_vector("DE", "guess", guess, &[0, dim])?;
    let sigma_lengths: &[usize] = if guess.is_empty() { &[0, dim] } else { &[dim] };
    check_vector("DE", "sigma", sigma, sigma_lengths)?;
    check_length("DE", "ints", ints.map_or(0, <[bool]>::len), &[0, dim])
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
/// ``evaluations`` counts the objective calls actually made.
/// ``max_evaluations`` is checked before each generation, and a generation
/// costs one or two calls per population member, so a run can make up to
/// ``2 * popsize - 1`` calls more than ``max_evaluations``.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have ``dim`` entries
/// each with finite ``lower < upper`` (both may be empty only with a
/// ``guess``), if a non-empty ``guess`` does not have ``dim`` finite entries,
/// if ``sigma`` does not have ``dim`` finite entries when a ``guess`` is given
/// (or is neither empty nor of length ``dim`` without one), or if a non-empty
/// ``ints`` does not have one entry per decision variable. Array arguments
/// accept NumPy arrays and sequences: one that is not one-dimensional raises
/// ``TypeError``, values NumPy cannot convert to ``float64`` (``bool`` for
/// ``ints``) raise ``ValueError``. As in fcmaes, an exception raised by the
/// objective callback is not re-raised: that call, like a non-finite result,
/// counts as a failed evaluation and ranks worst.
#[pyfunction]
#[pyo3(signature = (fun, dim, lower, upper, guess, sigma, ints, *, seed, runid=0,
    max_evaluations=100000, keep=200.0, stop_fitness=f64::NEG_INFINITY, popsize=31,
    F=0.5, CR=0.9, min_sigma=0.0, min_mutate=0.1, max_mutate=0.5, workers=1,
    terminate=None))]
pub fn optimize_de<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    dim: usize,
    lower: F64Vec<'_>,
    upper: F64Vec<'_>,
    guess: F64Vec<'_>,
    sigma: F64Vec<'_>,
    ints: BoolVec<'_>,
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
    check_de_inputs(dim, &lower, &upper, &guess, &sigma, ints.as_deref())?;

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
/// Raises ``ValueError`` on the same ``lower``, ``upper``, ``guess``, ``sigma``
/// and ``ints`` violations as :func:`optimize_de`. Array arguments accept NumPy
/// arrays and sequences: one that is not one-dimensional raises ``TypeError``,
/// values NumPy cannot convert to ``float64`` (``bool`` for ``ints``) raise
/// ``ValueError``. :meth:`tell` raises ``ValueError`` if it does not receive
/// exactly one value per row returned by the most recent :meth:`ask`.
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
        lower: F64Vec<'_>,
        upper: F64Vec<'_>,
        guess: F64Vec<'_>,
        sigma: F64Vec<'_>,
        ints: BoolVec<'_>,
        popsize: i32,
        keep: f64,
        F: f64,
        CR: f64,
        min_sigma: f64,
        min_mutate: f64,
        max_mutate: f64,
        seed: u64,
        runid: i64,
    ) -> PyResult<Self> {
        let (f, cr) = (F, CR);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let guess = slice_or_vec(&guess);
        let sigma = slice_or_vec(&sigma);
        let ints = opt_ints(&ints);
        check_de_inputs(dim, &lower, &upper, &guess, &sigma, ints.as_deref())?;
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
        Ok(DE {
            inner: De::new(fitness, &guess, &sigma, ints, &params),
        })
    }

    /// Return the next decoded candidate population, shape ``(popsize, dim)``.
    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask())
    }

    /// Submit one minimized objective value per row returned by :meth:`ask`.
    ///
    /// Returns the optimizer stop code.
    fn tell(&mut self, ys: F64Vec<'_>) -> i32 {
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
