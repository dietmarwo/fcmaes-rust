//! PyO3 bindings for CR-FM-NES: the `optimize_crfmnes` free function (batch
//! objective) and the `CRFMNES` ask/tell class, backed by `fcmaes_core::Crfmnes`.

use fcmaes_core::{Crfmnes, CrfmnesParams, Fitness};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{F64Vec, check_box, check_vector, result_tuple, rows_to_pyarray, slice_or_vec};

fn build_fitness(dim: usize, lower: &[f64], upper: &[f64], normalize: bool) -> Fitness {
    if lower.is_empty() {
        Fitness::new(dim, 1, vec![], vec![])
    } else {
        let mut f = Fitness::bounded(dim, 1, lower, upper);
        f.set_normalize(normalize);
        f
    }
}

/// Validate the inputs `Crfmnes::new` would otherwise panic on.
fn check_crfmnes_inputs(
    dim: usize,
    guess: &[f64],
    lower: &[f64],
    upper: &[f64],
    sigma: f64,
) -> PyResult<()> {
    check_box("CR-FM-NES", dim, lower, upper, true)?;
    check_vector("CR-FM-NES", "guess", guess, &[dim])?;
    check_vector("CR-FM-NES", "sigma", &[sigma], &[1])
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
/// Minimize a scalar objective with CR-FM-NES.
///
/// ``batch_fun(xs)`` receives a ``(popsize, dim)`` ``float64`` array and must
/// return one minimized value per row. ``guess`` is the distribution center,
/// ``sigma`` its initial scalar scale, and optional bounds constrain decoded
/// candidates. The batch callback is invoked with the GIL held.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``.
///
/// ``evaluations`` counts the rows actually passed to ``batch_fun``.
/// ``max_evaluations`` is checked before each generation and whole generations
/// are evaluated, so a run can evaluate up to ``popsize - 1`` rows more.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have one entry per
/// element of ``guess`` (both may be empty for an unbounded search) or do not
/// satisfy finite ``lower < upper``, or if ``guess`` or ``sigma`` is not
/// finite. Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. As in fcmaes, an exception raised by the
/// objective callback is not re-raised: that call, like a non-finite result,
/// counts as a failed evaluation and ranks worst.
#[pyfunction]
#[pyo3(signature = (batch_fun, guess, lower, upper, sigma=0.3, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, popsize=32,
    penalty_coef=1e5, use_constraint_violation=true, normalize=false))]
pub fn optimize_crfmnes<'py>(
    py: Python<'py>,
    batch_fun: Py<PyAny>,
    guess: F64Vec<'_>,
    lower: F64Vec<'_>,
    upper: F64Vec<'_>,
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
    check_crfmnes_inputs(dim, &guess, &lower, &upper, sigma)?;

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

/// Stateful CR-FM-NES with an external ask/tell evaluator.
///
/// CR-FM-NES uses a restricted covariance representation for
/// high-dimensional search. Alternate :meth:`ask` and :meth:`tell`; evaluate
/// every returned row in order.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have one entry per
/// element of ``guess`` (both may be empty for an unbounded search) or do not
/// satisfy finite ``lower < upper``, or if ``guess`` or ``sigma`` is not
/// finite. Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. :meth:`tell` raises ``ValueError`` if it
/// does not receive exactly one value per row returned by the most recent
/// :meth:`ask`.
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
        guess: F64Vec<'_>,
        lower: F64Vec<'_>,
        upper: F64Vec<'_>,
        sigma: f64,
        popsize: i32,
        seed: u64,
        runid: i64,
        penalty_coef: f64,
        use_constraint_violation: bool,
        normalize: bool,
    ) -> PyResult<Self> {
        let guess = slice_or_vec(&guess);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let dim = guess.len();
        check_crfmnes_inputs(dim, &guess, &lower, &upper, sigma)?;
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
        Ok(CRFMNES {
            inner: Crfmnes::new(fitness, &guess, sigma, &params),
        })
    }

    /// Return the next decoded candidate population, shape ``(popsize, dim)``.
    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask_pop())
    }

    /// Submit one minimized value per row from the most recent :meth:`ask`.
    ///
    /// Returns the current stop code.
    fn tell(&mut self, ys: F64Vec<'_>) -> i32 {
        self.inner.tell_pop(&slice_or_vec(&ys))
    }

    /// Return the current decoded population.
    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
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
    /// Number of candidates in one mirrored population.
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
    m.add_function(wrap_pyfunction!(optimize_crfmnes, m)?)?;
    m.add_class::<CRFMNES>()?;
    Ok(())
}
