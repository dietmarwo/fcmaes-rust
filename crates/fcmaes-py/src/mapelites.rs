//! PyO3 bindings for Quality-Diversity: the CVT-MAP-Elites `Archive` class with
//! `optimize_map_elites` and `diversify`, backed by `fcmaes_core::mapelites`.
//!
//! The `qd_fitness` callback maps `x -> (fitness, behavior_descriptor)`. The
//! optimization loop runs under `py.allow_threads`; the callback re-acquires the
//! GIL per call.

use fcmaes_core::mapelites::{
    DiversifierParams, MapElitesParams, QdFitness, diversify, map_elites,
};
use fcmaes_core::{Archive as CoreArchive, Rng};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::common::{rows_to_pyarray, slice_or_vec};
use numpy::PyReadonlyArray1;

/// Quality-diversity fitness backed by a Python callable
/// `fun(x) -> (float, np.ndarray)`. Re-acquires the GIL per call.
struct PyQdFitness {
    fun: Py<PyAny>,
    qd_dim: usize,
}

impl QdFitness for PyQdFitness {
    fn eval(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        Python::with_gil(|py| {
            let arr = PyArray1::from_slice(py, x);
            match self.fun.call1(py, (arr,)) {
                Ok(v) => match v.extract::<(f64, Vec<f64>)>(py) {
                    Ok((y, d)) if y.is_finite() && d.iter().all(|v| v.is_finite()) => (y, d),
                    _ => (f64::INFINITY, vec![0.0; self.qd_dim]),
                },
                Err(_) => (f64::INFINITY, vec![0.0; self.qd_dim]),
            }
        })
    }
}

/// CVT-MAP-Elites archive with the MAP-Elites and Diversifier drivers.
#[pyclass]
pub struct Archive {
    inner: CoreArchive,
    rng: Rng,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Archive {
    #[new]
    #[pyo3(signature = (dim, lower, upper, qd_lower, qd_upper, capacity=4000,
        samples_per_niche=20, *, seed, seed_parents=true))]
    fn new(
        dim: usize,
        lower: PyReadonlyArray1<f64>,
        upper: PyReadonlyArray1<f64>,
        qd_lower: PyReadonlyArray1<f64>,
        qd_upper: PyReadonlyArray1<f64>,
        capacity: usize,
        samples_per_niche: usize,
        seed: u64,
        seed_parents: bool,
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
                "decision bounds must match dim and satisfy finite lower < upper",
            ));
        }
        let qd_lo = slice_or_vec(&qd_lower);
        let qd_up = slice_or_vec(&qd_upper);
        let mut rng = Rng::new(seed);
        let mut inner =
            CoreArchive::try_new(dim, &qd_lo, &qd_up, capacity, samples_per_niche, &mut rng)
                .map_err(PyValueError::new_err)?;
        if seed_parents {
            inner.seed_uniform(&lower, &upper, &mut rng);
        }
        Ok(Archive {
            inner,
            rng,
            lower,
            upper,
        })
    }

    /// Run CVT-MAP-Elites with the SBX / Iso+LineDD (+ optional CMA-ES) emitter.
    #[pyo3(signature = (qd_fitness, generations=100, chunk_size=20, use_sbx=true,
        dis_c=20.0, dis_m=20.0, iso_sigma=0.02, line_sigma=0.2, cma_generations=0))]
    fn optimize_map_elites(
        &mut self,
        py: Python<'_>,
        qd_fitness: Py<PyAny>,
        generations: usize,
        chunk_size: usize,
        use_sbx: bool,
        dis_c: f64,
        dis_m: f64,
        iso_sigma: f64,
        line_sigma: f64,
        cma_generations: usize,
    ) {
        let qd_dim = self.inner.qd_dim();
        let mut fit = PyQdFitness {
            fun: qd_fitness,
            qd_dim,
        };
        let params = MapElitesParams {
            generations,
            chunk_size,
            use_sbx,
            dis_c,
            dis_m,
            iso_sigma,
            line_sigma,
            cma_generations,
        };
        let inner = &mut self.inner;
        let rng = &mut self.rng;
        let lower = &self.lower;
        let upper = &self.upper;
        py.allow_threads(|| {
            map_elites(inner, &mut fit, lower, upper, &params, rng);
        });
    }

    /// Run the Diversifier (CMA-ME-style); returns `(best_x, best_y)`.
    #[pyo3(signature = (qd_fitness, max_evaluations=100000, popsize=31, stall_criterion=20))]
    fn diversify<'py>(
        &mut self,
        py: Python<'py>,
        qd_fitness: Py<PyAny>,
        max_evaluations: u64,
        popsize: i32,
        stall_criterion: i32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let qd_dim = self.inner.qd_dim();
        let mut fit = PyQdFitness {
            fun: qd_fitness,
            qd_dim,
        };
        let params = DiversifierParams {
            max_evaluations,
            popsize,
            stall_criterion,
        };
        let inner = &mut self.inner;
        let rng = &mut self.rng;
        let lower = &self.lower;
        let upper = &self.upper;
        let (bx, by) = py.allow_threads(|| diversify(inner, &mut fit, lower, upper, &params, rng));
        PyTuple::new(
            py,
            [
                PyArray1::from_slice(py, &bx).into_any(),
                by.into_pyobject(py)?.into_any(),
            ],
        )
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn qd_dim(&self) -> usize {
        self.inner.qd_dim()
    }
    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    #[getter]
    fn occupied(&self) -> usize {
        self.inner.occupied()
    }
    #[getter]
    fn best_y(&self) -> f64 {
        self.inner.best_y()
    }
    #[getter]
    fn qd_score(&self) -> f64 {
        self.inner.qd_score()
    }

    /// Niche fitness values (`inf` for empty niches).
    fn ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.ys())
    }

    /// Niche solutions, shape `(capacity, dim)`.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, self.inner.xs())
    }

    /// Niche behavior descriptors, shape `(capacity, qd_dim)`.
    fn descriptors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_pyarray(py, self.inner.descriptors())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Archive>()?;
    Ok(())
}
