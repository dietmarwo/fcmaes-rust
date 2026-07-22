//! PyO3 binding for MODE: the `MODE` multi-objective ask/tell class, backed by
//! `fcmaes_core::Mode`. Ask/tell only (the caller evaluates objectives +
//! constraints). Arrays are row-per-individual: `ask()` returns
//! `(popsize, dim)`, `tell(ys)` takes `(popsize, nobj + ncon)`.

use fcmaes_core::{Fitness, Mode, ModeParams};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::{matrix_rows, rows_to_pyarray, slice_or_vec};

fn opt_ints(ints: &PyReadonlyArray1<bool>) -> Option<Vec<bool>> {
    if ints.len().unwrap_or(0) == 0 {
        None
    } else {
        Some(ints.as_array().iter().copied().collect())
    }
}

/// Stateful multi-objective / constrained MODE with an ask/tell interface.
#[allow(clippy::upper_case_acronyms)]
#[pyclass]
pub struct MODE {
    inner: Mode,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl MODE {
    #[new]
    #[pyo3(signature = (dim, nobj, ncon, lower, upper, ints, popsize=64, F=0.5,
        CR=0.9, pro_c=0.5, dis_c=15.0, pro_m=0.9, dis_m=20.0, nsga_update=true,
        pareto_update=0.0, min_mutate=0.1, max_mutate=0.5, *, seed, runid=0))]
    fn new(
        dim: usize,
        nobj: usize,
        ncon: usize,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        ints: PyReadonlyArray1<bool>,
        popsize: i32,
        #[allow(non_snake_case)] F: f64,
        #[allow(non_snake_case)] CR: f64,
        pro_c: f64,
        dis_c: f64,
        pro_m: f64,
        dis_m: f64,
        nsga_update: bool,
        pareto_update: f64,
        min_mutate: f64,
        max_mutate: f64,
        seed: u64,
        runid: i64,
    ) -> PyResult<Self> {
        let lower = slice_or_vec(&lower);
        let upper = slice_or_vec(&upper);
        if lower.len() != dim
            || upper.len() != dim
            || lower
                .iter()
                .zip(&upper)
                .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        {
            return Err(PyValueError::new_err(
                "MODE bounds must match dim and satisfy finite lower < upper",
            ));
        }
        let fitness = Fitness::bounded(dim, nobj + ncon, &lower, &upper);
        let params = ModeParams {
            popsize,
            f: F,
            cr: CR,
            pro_c,
            dis_c,
            pro_m,
            dis_m,
            nsga_update,
            pareto_update,
            min_mutate,
            max_mutate,
            seed,
            runid,
        };
        Ok(MODE {
            inner: Mode::try_new(fitness, nobj, ncon, opt_ints(&ints), &params)
                .map_err(PyValueError::new_err)?,
        })
    }

    fn ask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(rows_to_pyarray(
            py,
            &self.inner.try_ask().map_err(PyValueError::new_err)?,
        ))
    }

    fn tell(&mut self, ys: PyReadonlyArray2<f64>) -> PyResult<i32> {
        self.inner
            .try_tell(&matrix_rows(&ys))
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (ys, nsga_update=true, pareto_update=0.0))]
    fn tell_switch(
        &mut self,
        ys: PyReadonlyArray2<f64>,
        nsga_update: bool,
        pareto_update: f64,
    ) -> PyResult<i32> {
        self.inner
            .try_tell_switch(&matrix_rows(&ys), nsga_update, pareto_update)
            .map_err(PyValueError::new_err)
    }

    fn set_population(
        &mut self,
        xs: PyReadonlyArray2<f64>,
        ys: PyReadonlyArray2<f64>,
    ) -> PyResult<i32> {
        self.inner
            .try_set_population(&matrix_rows(&xs), &matrix_rows(&ys))
            .map_err(PyValueError::new_err)
    }

    fn population<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, &self.inner.population())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn nobj(&self) -> usize {
        self.inner.nobj()
    }
    #[getter]
    fn ncon(&self) -> usize {
        self.inner.ncon()
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
    m.add_class::<MODE>()?;
    Ok(())
}
