//! PyO3 bindings for BiteOpt: the `optimize_bite` free function and the `Bite`
//! ask/tell class, backed by `fcmaes_core::BiteOpt`.

use fcmaes_core::{BiteParams, DeepBiteOpt, optimize_bite, validate_bite_inputs};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{PyObjective, result_tuple, rows_to_pyarray, slice_or_vec};

fn make_params(
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    popsize: i32,
    stall_criterion: i32,
) -> BiteParams {
    BiteParams {
        popsize,
        max_evaluations,
        stop_fitness,
        stall_criterion,
        seed,
        runid,
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "optimize_bite")]
#[pyo3(signature = (fun, guess, lower, upper, *, seed, runid=0,
    max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, M=1, popsize=0,
    stall_criterion=0))]
pub fn optimize_bite_py<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    guess: PyReadonlyArray1<f64>,
    lower: PyReadonlyArray1<f64>,
    upper: PyReadonlyArray1<f64>,
    seed: u64,
    runid: i64,
    max_evaluations: u64,
    stop_fitness: f64,
    #[allow(non_snake_case)] M: i32,
    popsize: i32,
    stall_criterion: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let guess = slice_or_vec(&guess);
    let lower = slice_or_vec(&lower);
    let upper = slice_or_vec(&upper);
    let params = make_params(
        seed,
        runid,
        max_evaluations,
        stop_fitness,
        popsize,
        stall_criterion,
    );
    let init = if guess.is_empty() {
        None
    } else {
        Some(guess.as_slice())
    };
    validate_bite_inputs(&lower, &upper, init, &params, M).map_err(PyValueError::new_err)?;
    let obj = PyObjective::new(fun);
    let result = py.allow_threads(move || optimize_bite(&obj, &lower, &upper, init, &params, M));
    result_tuple(
        py,
        &result.x,
        result.y,
        result.evaluations,
        result.iterations,
        result.stop,
    )
}

/// Stateful BiteOpt with a batched ask/tell interface.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct Bite {
    inner: DeepBiteOpt,
    batch_size: usize,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Bite {
    #[new]
    #[pyo3(signature = (guess, lower, upper, M=1, popsize=0, batch_size=8, *,
        max_evaluations=100000, stop_fitness=f64::NEG_INFINITY, stall_criterion=0,
        seed, runid=0))]
    fn new(
        guess: PyReadonlyArray1<f64>,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        #[allow(non_snake_case)] M: i32,
        popsize: i32,
        batch_size: i32,
        max_evaluations: u64,
        stop_fitness: f64,
        stall_criterion: i32,
        seed: u64,
        runid: i64,
    ) -> PyResult<Self> {
        let guess = slice_or_vec(&guess);
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        let params = make_params(
            seed,
            runid,
            max_evaluations,
            stop_fitness,
            popsize,
            stall_criterion,
        );
        let init = if guess.is_empty() {
            None
        } else {
            Some(guess.as_slice())
        };
        validate_bite_inputs(&lower, &upper, init, &params, M).map_err(PyValueError::new_err)?;
        Ok(Bite {
            inner: DeepBiteOpt::new(&lower, &upper, init, &params, M),
            batch_size: batch_size.max(1) as usize,
        })
    }

    fn ask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.inner.current_batch_size() != 0 {
            return Err(PyRuntimeError::new_err(
                "Bite ask() called twice without tell()",
            ));
        }
        let rows = self.inner.ask(self.batch_size);
        if rows.is_empty() {
            Ok(Array2::zeros((0, self.inner.dim()))
                .into_pyarray(py)
                .to_owned())
        } else {
            Ok(rows_to_pyarray(py, &rows))
        }
    }

    fn tell(&mut self, ys: PyReadonlyArray1<f64>) -> PyResult<i32> {
        let expected = self.inner.current_batch_size();
        if expected == 0 {
            return Err(PyRuntimeError::new_err("Bite tell() called before ask()"));
        }
        let values = slice_or_vec(&ys);
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "ys must match the current batch size ({expected})"
            )));
        }
        Ok(self.inner.tell(&values))
    }

    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let r = self.inner.result_public();
        result_tuple(py, &r.x, r.y, r.evaluations, r.iterations, r.stop)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn popsize(&self) -> usize {
        self.batch_size
    }
    #[getter]
    fn population_size(&self) -> usize {
        self.inner.population_size()
    }
    #[getter]
    fn current_batch_size(&self) -> usize {
        self.inner.current_batch_size()
    }
    #[getter]
    fn stop(&self) -> i32 {
        self.inner.stop_code()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_bite_py, m)?)?;
    m.add_class::<Bite>()?;
    Ok(())
}
