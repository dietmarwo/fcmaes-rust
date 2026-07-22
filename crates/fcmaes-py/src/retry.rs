//! Python callback bridge for the native retry coordinators.

use std::sync::Mutex;

use fcmaes_core::{
    AdvancedRetryConfig, RetryBounds, RetryConfig, RetryContext, RetryResult, RetryRunResult,
    advanced_retry, retry,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common::{rows_to_pyarray, slice_or_vec};

/// Minimal store surface consumed by `fcmaes.optimizer.Optimizer`.
#[pyclass]
struct RetryStoreView {
    eval_factor: f64,
    run_id: usize,
}

#[pymethods]
impl RetryStoreView {
    fn eval_num(&self, max_evaluations: u64) -> u64 {
        ((max_evaluations as f64) * self.eval_factor)
            .round()
            .clamp(1.0, u64::MAX as f64) as u64
    }

    fn get_count_runs(&self) -> usize {
        self.run_id
    }
}

pub(crate) fn call_optimizer(
    fun: &Py<PyAny>,
    optimize: &Py<PyAny>,
    context: &RetryContext,
    base_evaluations: u64,
) -> PyResult<RetryRunResult> {
    Python::with_gil(|py| {
        let scipy = py.import("scipy.optimize")?;
        let lower = PyArray1::from_slice(py, context.bounds.lower());
        let upper = PyArray1::from_slice(py, context.bounds.upper());
        let bounds = scipy.getattr("Bounds")?.call1((lower, upper))?;

        let numpy_random = py.import("numpy.random")?;
        let bit_generator = numpy_random.getattr("PCG64DXSM")?.call1((context.seed,))?;
        let rng = numpy_random.getattr("Generator")?.call1((bit_generator,))?;
        let guess = context.guess.as_ref().map_or_else(
            || py.None(),
            |values| PyArray1::from_slice(py, values).into_any().unbind(),
        );
        let sdev = PyArray1::from_slice(py, &context.sdev);
        let eval_factor = context.max_evaluations as f64 / base_evaluations.max(1) as f64;
        let store = Py::new(
            py,
            RetryStoreView {
                eval_factor,
                run_id: context.run_id,
            },
        )?;
        let result = optimize.call1(py, (fun.clone_ref(py), bounds, guess, sdev, rng, store))?;
        let tuple = result.bind(py).downcast::<pyo3::types::PyTuple>()?;
        if tuple.len() < 3 {
            return Err(PyValueError::new_err(
                "optimizer must return (x, value, evaluations)",
            ));
        }
        let x_obj = tuple.get_item(0)?;
        let x = if let Ok(array) = x_obj.extract::<PyReadonlyArray1<'_, f64>>() {
            slice_or_vec(&array)
        } else {
            x_obj.extract::<Vec<f64>>()?
        };
        Ok(RetryRunResult {
            x,
            y: tuple.get_item(1)?.extract()?,
            evaluations: tuple.get_item(2)?.extract()?,
        })
    })
}

fn result_dict(py: Python<'_>, result: RetryResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", PyArray1::from_slice(py, &result.x))?;
    dict.set_item("fun", result.y)?;
    dict.set_item("nfev", result.evaluations)?;
    dict.set_item("runs", result.runs)?;
    dict.set_item("success", result.success)?;
    let rows: Vec<Vec<f64>> = result.entries.iter().map(|entry| entry.x.clone()).collect();
    dict.set_item("xs", rows_to_pyarray(py, &rows))?;
    dict.set_item(
        "ys",
        PyArray1::from_vec(py, result.entries.iter().map(|entry| entry.y).collect()),
    )?;
    let improvements: Vec<Vec<f64>> = result
        .improvements
        .iter()
        .map(|sample| {
            vec![
                sample.elapsed_seconds,
                sample.evaluations as f64,
                sample.value,
            ]
        })
        .collect();
    dict.set_item("improvements", rows_to_pyarray(py, &improvements))?;
    Ok(dict.unbind())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (fun, optimize, lower, upper, num_retries=1024, workers=0, capacity=500, value_limit=f64::INFINITY, stop_fitness=f64::NEG_INFINITY, max_evaluations=50_000, statistic_num=0, seed=0))]
fn minimize_retry(
    py: Python<'_>,
    fun: Py<PyAny>,
    optimize: Py<PyAny>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    num_retries: usize,
    workers: usize,
    capacity: usize,
    value_limit: f64,
    stop_fitness: f64,
    max_evaluations: u64,
    statistic_num: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let bounds = RetryBounds::new(slice_or_vec(&lower), slice_or_vec(&upper))
        .map_err(PyValueError::new_err)?;
    let config = RetryConfig {
        num_retries,
        workers,
        capacity,
        value_limit,
        stop_fitness,
        max_evaluations,
        statistic_num,
        seed,
    };
    let first_error = Mutex::new(None);
    let result = py.allow_threads(|| {
        retry(&|_: &[f64]| 0.0, &bounds, &config, |_, context| {
            call_optimizer(&fun, &optimize, context, max_evaluations).unwrap_or_else(|error| {
                *first_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
                RetryRunResult {
                    x: Vec::new(),
                    y: f64::NAN,
                    evaluations: 0,
                }
            })
        })
    });
    if !result.success
        && let Some(message) = first_error
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(PyRuntimeError::new_err(message));
    }
    result_dict(py, result)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (fun, optimize, lower, upper, num_retries=5000, workers=0, capacity=500, value_limit=f64::INFINITY, stop_fitness=f64::NEG_INFINITY, min_evaluations=1500, max_eval_fac=50.0, check_interval=100, statistic_num=0, seed=0))]
fn minimize_advanced_retry(
    py: Python<'_>,
    fun: Py<PyAny>,
    optimize: Py<PyAny>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    num_retries: usize,
    workers: usize,
    capacity: usize,
    value_limit: f64,
    stop_fitness: f64,
    min_evaluations: u64,
    max_eval_fac: f64,
    check_interval: usize,
    statistic_num: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let bounds = RetryBounds::new(slice_or_vec(&lower), slice_or_vec(&upper))
        .map_err(PyValueError::new_err)?;
    let config = AdvancedRetryConfig {
        retry: RetryConfig {
            num_retries,
            workers,
            capacity,
            value_limit,
            stop_fitness,
            max_evaluations: min_evaluations,
            statistic_num,
            seed,
        },
        check_interval,
        max_eval_fac,
        ..Default::default()
    };
    let first_error = Mutex::new(None);
    let result = py.allow_threads(|| {
        advanced_retry(&|_: &[f64]| 0.0, &bounds, &config, |_, context| {
            call_optimizer(&fun, &optimize, context, min_evaluations).unwrap_or_else(|error| {
                *first_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
                RetryRunResult {
                    x: Vec::new(),
                    y: f64::NAN,
                    evaluations: 0,
                }
            })
        })
    });
    if !result.success
        && let Some(message) = first_error
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(PyRuntimeError::new_err(message));
    }
    result_dict(py, result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(minimize_retry, module)?)?;
    module.add_function(wrap_pyfunction!(minimize_advanced_retry, module)?)?;
    Ok(())
}
