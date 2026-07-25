//! PyO3 bindings for PGPE: the `optimize_pgpe` free function (batch objective)
//! and the `PGPE` ask/tell class, backed by `fcmaes_core::Pgpe`.

use fcmaes_core::{Fitness, Pgpe, PgpeParams};
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

#[allow(clippy::too_many_arguments)]
fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    lr_decay_steps: i32,
    use_ranking: bool,
    center_learning_rate: f64,
    stdev_learning_rate: f64,
    stdev_max_change: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    decay_coef: f64,
) -> PgpeParams {
    PgpeParams {
        popsize,
        max_evaluations,
        stop_fitness,
        lr_decay_steps,
        use_ranking,
        center_learning_rate,
        stdev_learning_rate,
        stdev_max_change,
        b1,
        b2,
        eps,
        decay_coef,
        seed,
        runid,
    }
}

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

#[allow(clippy::too_many_arguments)]
/// Minimize a scalar objective with Parameter-Exploring Policy Gradients.
///
/// ``batch_fun(xs)`` receives a ``(popsize, dim)`` ``float64`` array and must
/// return one minimized value per row. ``guess`` and ``sigma`` initialize the
/// center and per-coordinate spread of the sampling distribution. Bounds may
/// be empty for unbounded search. The Python batch callback runs with the GIL.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``.
///
/// Raises ``ValueError`` if the bound arrays are empty, of unequal length, or
/// do not satisfy finite ``lower < upper``, and ``TypeError`` if an array
/// argument cannot be converted to contiguous ``float64``. Exceptions raised
/// inside the objective callback propagate to the caller.
#[pyfunction]
#[pyo3(signature = (batch_fun, guess, lower, upper, sigma, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, popsize=32,
    lr_decay_steps=1000, use_ranking=true, center_learning_rate=0.15,
    stdev_learning_rate=0.1, stdev_max_change=0.2, b1=0.9, b2=0.999, eps=1e-8,
    decay_coef=1.0, normalize=true))]
pub fn optimize_pgpe<'py>(
    py: Python<'py>,
    batch_fun: Py<PyAny>,
    guess: PyReadonlyArray1<f64>,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    lr_decay_steps: i32,
    use_ranking: bool,
    center_learning_rate: f64,
    stdev_learning_rate: f64,
    stdev_max_change: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    decay_coef: f64,
    normalize: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let guess = slice_or_vec(&guess);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let sigma = slice_or_vec(&sigma);
    let dim = guess.len();

    let fitness = build_fitness(dim, &lower, &upper, normalize);
    let params = make_params(
        seed,
        runid,
        max_evaluations,
        stop_fitness,
        popsize,
        lr_decay_steps,
        use_ranking,
        center_learning_rate,
        stdev_learning_rate,
        stdev_max_change,
        b1,
        b2,
        eps,
        decay_coef,
    );

    let result = py.allow_threads(move || {
        let mut opt = Pgpe::new(fitness, &guess, &sigma, &params);
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

/// Stateful PGPE with an external ask/tell evaluator.
///
/// The optimizer uses mirrored parameter samples and an Adam center update.
/// Alternate :meth:`ask` and :meth:`tell`, preserving row order.
///
/// Raises ``ValueError`` if the bound arrays do not match the decision
/// dimension or do not satisfy finite ``lower < upper``, and ``TypeError`` if
/// an array argument cannot be converted to contiguous ``float64``.
/// :meth:`tell` raises ``ValueError`` if it does not receive exactly one value
/// per row returned by the most recent :meth:`ask`.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct PGPE {
    inner: Pgpe,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PGPE {
    #[new]
    #[pyo3(signature = (guess, lower, upper, sigma, popsize=32, *, seed, runid=0,
        lr_decay_steps=1000, use_ranking=true, center_learning_rate=0.15,
        stdev_learning_rate=0.1, stdev_max_change=0.2, b1=0.9, b2=0.999,
        eps=1e-8, decay_coef=1.0, normalize=true))]
    fn new(
        guess: PyReadonlyArray1<f64>,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        sigma: PyReadonlyArray1<f64>,
        popsize: i32,
        seed: u64,
        runid: i64,
        lr_decay_steps: i32,
        use_ranking: bool,
        center_learning_rate: f64,
        stdev_learning_rate: f64,
        stdev_max_change: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        decay_coef: f64,
        normalize: bool,
    ) -> Self {
        let guess = slice_or_vec(&guess);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let sigma = slice_or_vec(&sigma);
        let dim = guess.len();
        let fitness = build_fitness(dim, &lower, &upper, normalize);
        let params = make_params(
            seed,
            runid,
            0,
            f64::NEG_INFINITY,
            popsize,
            lr_decay_steps,
            use_ranking,
            center_learning_rate,
            stdev_learning_rate,
            stdev_max_change,
            b1,
            b2,
            eps,
            decay_coef,
        );
        PGPE {
            inner: Pgpe::new(fitness, &guess, &sigma, &params),
        }
    }

    /// Return the next mirrored candidate population, shape ``(popsize, dim)``.
    fn ask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.ask_pop())
    }

    /// Submit one minimized value per row from the most recent :meth:`ask`.
    ///
    /// Returns the current stop code.
    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> i32 {
        self.inner.tell_pop(&slice_or_vec(&ys))
    }

    /// Return the current decoded candidate population.
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
    /// Number of mirrored candidates per update.
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
    m.add_function(wrap_pyfunction!(optimize_pgpe, m)?)?;
    m.add_class::<PGPE>()?;
    Ok(())
}
