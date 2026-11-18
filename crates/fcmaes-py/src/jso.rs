//! PyO3 bindings for jSO: the `optimize_jso` free function and the `JSO`
//! ask/tell class, backed by `fcmaes_core::Jso`.

use fcmaes_core::{Fitness, Jso, JsoParams};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{F64Vec, PyObjective, check_box, result_tuple, rows_to_pyarray, slice_or_vec};

/// Validate the inputs `Jso::new` would otherwise panic on and build the
/// optimizer. jSO samples its initial population uniformly in the box, so
/// finite bounds are required, and its population schedule needs a positive
/// evaluation budget.
fn build(dim: usize, lower: &[f64], upper: &[f64], params: &JsoParams) -> PyResult<Jso> {
    check_box("jSO", dim, lower, upper, false)?;
    if params.max_evaluations == 0 {
        return Err(PyValueError::new_err(
            "jSO max_evaluations must be positive",
        ));
    }
    Jso::try_new(Fitness::bounded(dim, 1, lower, upper), params).map_err(PyValueError::new_err)
}

fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
) -> JsoParams {
    JsoParams {
        popsize,
        max_evaluations,
        stop_fitness,
        seed,
        runid,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
/// Minimize a scalar objective with jSO, an adaptive Differential Evolution
/// with linear population size reduction (Brest, Maučec and Bošković, CEC
/// 2017).
///
/// ``fun(x)`` receives a one-dimensional ``float64`` NumPy array and must
/// return a scalar to minimize. ``lower`` and ``upper`` define the finite
/// search box; the initial population is uniform in it. ``popsize`` is the
/// initial population size; ``0`` (the default) selects the published
/// ``round(25 * ln(dim) * sqrt(dim))``. The population shrinks linearly to four
/// members over ``max_evaluations``, so the budget shapes the whole run.
/// ``seed`` and ``runid`` make the run reproducible.
///
/// Each generation is evaluated as one batch: ``workers=1`` evaluates
/// serially, larger values use that many native threads (``0`` or less: all
/// cores), but Python callbacks still reacquire the GIL. The result does not
/// depend on ``workers``, except that a run reaching ``stop_fitness``
/// finishes its current generation when ``workers`` is not 1.
///
/// Returns ``(x, fun, evaluations, iterations, stop)``. ``stop == 1`` means
/// ``stop_fitness`` was reached; ``iterations`` counts generations after the
/// initial population.
///
/// ``evaluations`` counts the objective calls actually made and never exceeds
/// ``max_evaluations``.
///
/// Raises ``ValueError`` if ``lower`` and ``upper`` do not have ``dim`` entries
/// each with finite ``lower < upper``, or if ``max_evaluations`` is zero.
/// Array arguments accept NumPy arrays and sequences: one that is not
/// one-dimensional raises ``TypeError``, values NumPy cannot convert to
/// ``float64`` raise ``ValueError``. As in fcmaes, an exception raised by the
/// objective callback is not re-raised: that call, like a non-finite result,
/// counts as a failed evaluation and ranks worst. The run's first exception or
/// unusable return value is logged once as a warning on the ``fcmaes_rust``
/// logger; non-finite values are not logged.
#[pyfunction]
#[pyo3(signature = (fun, dim, lower, upper, *, seed, runid=0, max_evaluations=100000,
    stop_fitness=f64::NEG_INFINITY, popsize=0, workers=1))]
pub fn optimize_jso<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    dim: usize,
    lower: F64Vec<'_>,
    upper: F64Vec<'_>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    workers: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let params = make_params(seed, runid, max_evaluations, stop_fitness, popsize);
    let mut opt = build(dim, &lower, &upper, &params)?;
    let obj = PyObjective::new(fun);

    let result = py.allow_threads(move || {
        if workers == 1 {
            opt.optimize(&obj)
        } else {
            opt.optimize_parallel(&obj, workers)
        }
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

/// Stateful jSO with an external ask/tell evaluator.
///
/// The constructor arguments match :func:`optimize_jso` except that the
/// objective callback belongs to the caller. Unlike :class:`DE`, the
/// evaluation budget ``max_evaluations`` is a constructor argument: jSO's
/// population size and parameter schedule follow the fraction of it already
/// spent. Call :meth:`ask`, evaluate every returned row in the same order,
/// and pass the scalar values to :meth:`tell`.
///
/// One ask/tell pair is one generation. The first :meth:`ask` returns the
/// initial population, later ones one trial per current member; the
/// population shrinks over the run, so the number of rows varies. The last
/// batch is cut so that the rows asked never exceed ``max_evaluations`` in
/// total. Once the budget is spent (or ``stop_fitness`` is reached) :meth:`ask`
/// returns an empty ``(0, dim)`` array and no :meth:`tell` is due. With serial
/// evaluation the run equals :func:`optimize_jso` with ``workers=1``.
///
/// Raises ``ValueError`` on the same ``lower``, ``upper`` and
/// ``max_evaluations`` violations as :func:`optimize_jso`. Array arguments
/// accept NumPy arrays and sequences: one that is not one-dimensional raises
/// ``TypeError``, values NumPy cannot convert to ``float64`` raise
/// ``ValueError``. :meth:`ask` raises ``ValueError`` while a batch is pending,
/// and :meth:`tell` raises ``ValueError`` if no batch is pending or if it does
/// not receive exactly one value per row returned by the most recent
/// :meth:`ask`.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct JSO {
    inner: Jso,
}

#[pymethods]
impl JSO {
    #[new]
    #[pyo3(signature = (dim, lower, upper, popsize=0, *, seed, runid=0,
        max_evaluations=100000, stop_fitness=f64::NEG_INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        lower: F64Vec<'_>,
        upper: F64Vec<'_>,
        popsize: i32,
        seed: u64,
        runid: i64,
        max_evaluations: u64,
        stop_fitness: f64,
    ) -> PyResult<Self> {
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let params = make_params(seed, runid, max_evaluations, stop_fitness, popsize);
        Ok(JSO {
            inner: build(dim, &lower, &upper, &params)?,
        })
    }

    /// Return the next candidate batch, shape ``(rows, dim)``; empty once the
    /// budget is spent or ``stop_fitness`` is reached.
    fn ask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.inner.try_ask().map_err(PyValueError::new_err)?;
        if rows.is_empty() {
            return Ok(Array2::<f64>::zeros((0, self.inner.dim())).into_pyarray(py));
        }
        Ok(rows_to_pyarray(py, &rows))
    }

    /// Submit one minimized objective value per row returned by :meth:`ask`.
    ///
    /// Returns the optimizer stop code.
    fn tell(&mut self, ys: F64Vec<'_>) -> PyResult<i32> {
        self.inner
            .try_tell(&slice_or_vec(&ys))
            .map_err(PyValueError::new_err)
    }

    /// Return the current population, shape ``(popsize, dim)``.
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
    /// Current population size; it shrinks over the run.
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
    m.add_function(wrap_pyfunction!(optimize_jso, m)?)?;
    m.add_class::<JSO>()?;
    Ok(())
}
