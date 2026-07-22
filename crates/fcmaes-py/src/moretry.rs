//! Python callback bridge for native multi-objective weighted retry.

use std::sync::Mutex;

use fcmaes_core::{
    MoRetryConfig, MoRetryResult, MultiObjective, NAN_REPLACEMENT, RetryBounds, RetryConfig,
    RetryRunResult, WeightedObjective, moretry, scalarize,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common::{rows_to_pyarray, slice_or_vec};
use crate::retry::call_optimizer;

struct PyMultiObjective {
    fun: Py<PyAny>,
    width: usize,
}

impl MultiObjective for PyMultiObjective {
    fn eval(&self, x: &[f64]) -> Vec<f64> {
        Python::with_gil(|py| call_vector(py, &self.fun, x, self.width))
    }
}

#[pyclass]
struct PyWeightedObjective {
    fun: Py<PyAny>,
    weights: Vec<f64>,
    ncon: usize,
    value_exp: f64,
}

#[pymethods]
impl PyWeightedObjective {
    fn __call__(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> f64 {
        let x = slice_or_vec(&x);
        let values = call_vector(py, &self.fun, &x, self.weights.len());
        scalarize(&values, &self.weights, self.ncon, self.value_exp)
    }
}

fn call_vector(py: Python<'_>, fun: &Py<PyAny>, x: &[f64], width: usize) -> Vec<f64> {
    let array = PyArray1::from_slice(py, x);
    let Ok(result) = fun.call1(py, (array,)) else {
        return vec![NAN_REPLACEMENT; width];
    };
    if let Ok(array) = result.extract::<PyReadonlyArray1<'_, f64>>(py) {
        slice_or_vec(&array)
    } else {
        result
            .extract::<Vec<f64>>(py)
            .unwrap_or_else(|_| vec![NAN_REPLACEMENT; width])
    }
}

fn result_dict(py: Python<'_>, result: MoRetryResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", PyArray1::from_slice(py, &result.x))?;
    dict.set_item("fun", PyArray1::from_slice(py, &result.y))?;
    dict.set_item("scalar_fun", result.scalar_value)?;
    dict.set_item("nfev", result.evaluations)?;
    dict.set_item("runs", result.runs)?;
    dict.set_item("success", result.success)?;
    let xs: Vec<Vec<f64>> = result.entries.iter().map(|entry| entry.x.clone()).collect();
    let ys: Vec<Vec<f64>> = result.entries.iter().map(|entry| entry.y.clone()).collect();
    let weights: Vec<Vec<f64>> = result
        .entries
        .iter()
        .map(|entry| entry.weights.clone())
        .collect();
    dict.set_item("xs", rows_to_pyarray(py, &xs))?;
    dict.set_item("ys", rows_to_pyarray(py, &ys))?;
    dict.set_item("weights", rows_to_pyarray(py, &weights))?;
    dict.set_item(
        "scalar_ys",
        PyArray1::from_vec(
            py,
            result
                .entries
                .iter()
                .map(|entry| entry.scalar_value)
                .collect(),
        ),
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
#[pyo3(signature = (fun, optimize, lower, upper, weight_lower, weight_upper,
    ncon=0, value_exp=2.0, value_limits=None, num_retries=1024, workers=0,
    capacity=1024, value_limit=f64::INFINITY,
    stop_fitness=f64::NEG_INFINITY, max_evaluations=50_000,
    statistic_num=0, seed=0))]
fn minimize_moretry(
    py: Python<'_>,
    fun: Py<PyAny>,
    optimize: Py<PyAny>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    weight_lower: PyReadonlyArray1<'_, f64>,
    weight_upper: PyReadonlyArray1<'_, f64>,
    ncon: usize,
    value_exp: f64,
    value_limits: Option<PyReadonlyArray1<'_, f64>>,
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
    let weight_lower = slice_or_vec(&weight_lower);
    let width = weight_lower.len();
    let config = MoRetryConfig {
        retry: RetryConfig {
            num_retries,
            workers,
            capacity,
            value_limit,
            stop_fitness,
            max_evaluations,
            statistic_num,
            seed,
        },
        weight_lower,
        weight_upper: slice_or_vec(&weight_upper),
        ncon,
        value_exp,
        value_limits: value_limits.as_ref().map(slice_or_vec),
    };
    config.validate().map_err(PyValueError::new_err)?;
    let objective = PyMultiObjective {
        fun: fun.clone_ref(py),
        width,
    };
    let first_error = Mutex::new(None);
    let result = py
        .allow_threads(|| {
            moretry(
                &objective,
                &bounds,
                &config,
                |weighted: &WeightedObjective<'_, PyMultiObjective>, context| {
                    let callback = Python::with_gil(|py| {
                        Py::new(
                            py,
                            PyWeightedObjective {
                                fun: fun.clone_ref(py),
                                weights: weighted.weights().to_vec(),
                                ncon: weighted.ncon(),
                                value_exp: weighted.value_exp(),
                            },
                        )
                        .map(Py::into_any)
                    });
                    match callback.and_then(|callback| {
                        call_optimizer(&callback, &optimize, context, max_evaluations)
                    }) {
                        Ok(result) => result,
                        Err(error) => {
                            *first_error
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(error.to_string());
                            RetryRunResult {
                                x: Vec::new(),
                                y: f64::NAN,
                                evaluations: 0,
                            }
                        }
                    }
                },
            )
        })
        .map_err(PyValueError::new_err)?;
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
    module.add_function(wrap_pyfunction!(minimize_moretry, module)?)?;
    Ok(())
}
