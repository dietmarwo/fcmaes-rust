//! NumPy interfaces to the Rust rank indexes and multi-objective front tools.
use crate::common::{F64Mat, F64Vec, matrix_rows, slice_or_vec};
use fcmaes_core::archivede::{ArchiveDe, ArchiveDeParams, ParetoArchiveND};
use fcmaes_core::rankindex::{BordaIndex, GridIndex};
use fcmaes_core::{ParetoArchive2D, ParetoError};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

type Matrix<'py> = Bound<'py, PyArray2<f64>>;
type Front<'py> = (Matrix<'py>, Matrix<'py>);
type BoxBatch<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Vec<Vec<u128>>,
);
fn invalid(msg: &'static str) -> PyErr {
    PyValueError::new_err(msg)
}
fn matrix<'py>(py: Python<'py>, n: usize, d: usize, values: Vec<f64>) -> Matrix<'py> {
    Array2::from_shape_vec((n, d), values)
        .expect("rows have validated widths")
        .into_pyarray(py)
}
fn snapshot<'py, 'a>(
    py: Python<'py>,
    dx: usize,
    dy: usize,
    rows: impl Iterator<Item = (&'a [f64], &'a [f64])>,
) -> Front<'py> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut n = 0;
    for (x, y) in rows {
        xs.extend_from_slice(x);
        ys.extend_from_slice(y);
        n += 1
    }
    (matrix(py, n, dx, xs), matrix(py, n, dy, ys))
}
trait BoxLookup: Sync {
    fn width(&self) -> usize;
    fn hits(&self, lo: &[f64], hi: &[f64]) -> Vec<u32>;
    fn details(&self, lo: &[f64], hi: &[f64]) -> (Vec<u32>, Vec<u128>);
}
impl BoxLookup for BordaIndex {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn hits(&self, lo: &[f64], hi: &[f64]) -> Vec<u32> {
        self.query(lo, hi).ids
    }
    fn details(&self, lo: &[f64], hi: &[f64]) -> (Vec<u32>, Vec<u128>) {
        let r = self.query(lo, hi);
        (
            r.ids,
            vec![r.axis_candidates.into(), r.score_candidates.into()],
        )
    }
}
impl BoxLookup for GridIndex {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn hits(&self, lo: &[f64], hi: &[f64]) -> Vec<u32> {
        self.query(lo, hi).ids
    }
    fn details(&self, lo: &[f64], hi: &[f64]) -> (Vec<u32>, Vec<u128>) {
        let r = self.query(lo, hi);
        (
            r.ids,
            vec![
                r.touched_cells,
                r.nonempty_cells.into(),
                r.full_cells.into(),
                r.point_tests.into(),
            ],
        )
    }
}
fn query<'py>(
    py: Python<'py>,
    index: &impl BoxLookup,
    low: F64Vec<'py>,
    high: F64Vec<'py>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let (lo, hi) = (slice_or_vec(&low), slice_or_vec(&high));
    if lo.len() != index.width() || hi.len() != index.width() {
        return Err(invalid("bounds must match the index dimension"));
    }
    Ok(py.allow_threads(|| index.hits(&lo, &hi)).into_pyarray(py))
}
fn batch<'py>(
    py: Python<'py>,
    index: &impl BoxLookup,
    low: F64Mat<'py>,
    high: F64Mat<'py>,
) -> PyResult<BoxBatch<'py>> {
    let d = index.width();
    if low.as_array().dim() != high.as_array().dim() || low.as_array().ncols() != d {
        return Err(invalid("bounds must have shape (queries, index dimension)"));
    }
    let lo: Vec<_> = low.as_array().iter().copied().collect();
    let hi: Vec<_> = high.as_array().iter().copied().collect();
    let (offsets, ids, counts) = py.allow_threads(|| {
        let mut offsets = vec![0u64];
        let mut ids = Vec::new();
        let mut counts = Vec::new();
        for (a, b) in lo.chunks_exact(d).zip(hi.chunks_exact(d)) {
            let (mut hits, c) = index.details(a, b);
            ids.append(&mut hits);
            offsets.push(ids.len() as u64);
            counts.push(c);
        }
        (offsets, ids, counts)
    });
    Ok((offsets.into_pyarray(py), ids.into_pyarray(py), counts))
}

macro_rules! box_class {
    ($class:ident, $name:literal, $core:ty, $doc:literal $(, $arg:ident: $ty:ty)?) => {
        #[doc = $doc]
        #[pyclass(name = $name, module = "fcmaes_rust._fcmaes_ext")]
        struct $class {
            index: $core,
        }

        #[pymethods]
        impl $class {
            #[new]
            fn new(py: Python<'_>, points: F64Mat<'_> $(, $arg: $ty)?) -> PyResult<Self> {
                let matrix = points.as_array();
                let dimension = matrix.ncols();
                let data: Vec<_> = matrix.iter().copied().collect();
                let index = py
                    .allow_threads(|| <$core>::try_new(&data, dimension $(, $arg)?))
                    .map_err(invalid)?;
                Ok(Self { index })
            }

            /// Input row IDs inside the closed box, as a uint32 array.
            fn query<'py>(
                &self, py: Python<'py>, low: F64Vec<'py>, high: F64Vec<'py>,
            ) -> PyResult<Bound<'py, PyArray1<u32>>> {
                query(py, &self.index, low, high)
            }

            /// Return uint64 offsets, flat uint32 IDs, and one row of integer counters per box.
            fn query_batch<'py>(
                &self, py: Python<'py>, low: F64Mat<'py>, high: F64Mat<'py>,
            ) -> PyResult<BoxBatch<'py>> {
                batch(py, &self.index, low, high)
            }
        }
    }
}
box_class!(
    PyBorda,
    "BordaBoxIndex",
    BordaIndex,
    "Immutable Borda rank index over rows. Counters are shortest-axis and score-band candidates."
);
box_class!(
    PyGrid, "GridBoxIndex", GridIndex,
    "Immutable sparse rank grid. Counters are touched, occupied and full cells, and point tests.",
    query_volume: f64
);

enum Archive {
    Hv(ParetoArchive2D<Vec<f64>>),
    Nd(ParetoArchiveND<Vec<f64>, Vec<f64>>),
}
struct Store {
    archive: Archive,
    dx: usize,
    dy: usize,
}
fn hv_error(e: ParetoError) -> PyErr {
    match e {
        ParetoError::NonFiniteArea => PyOverflowError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}
impl Store {
    fn add(&mut self, y: &[f64], x: &[f64]) -> PyResult<bool> {
        if y.len() != self.dy || x.len() != self.dx {
            return Err(invalid("objective or payload width mismatch"));
        }
        match &mut self.archive {
            Archive::Hv(a) => a
                .insert_with([y[0], y[1]], || x.to_vec())
                .map(|v| v.accepted())
                .map_err(hv_error),
            Archive::Nd(a) => {
                if y.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("objectives must be finite"));
                }
                Ok(a.insert_with(y.to_vec(), || x.to_vec()))
            }
        }
    }
    fn add_batch<'py>(
        &mut self,
        py: Python<'py>,
        ys: F64Mat<'py>,
        xs: Option<F64Mat<'py>>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let y = ys.as_array();
        let n = y.nrows();
        if y.ncols() != self.dy
            || xs
                .as_ref()
                .is_some_and(|a| a.as_array().dim() != (n, self.dx))
            || (xs.is_none() && self.dx > 0 && n > 0)
        {
            return Err(invalid("objective or payload batch shape mismatch"));
        }
        let x = xs.as_ref().map(|m| m.as_array());
        let mut result = Vec::with_capacity(n);
        let mut ybuf = Vec::new();
        let mut xbuf = Vec::new();
        for (i, row) in y.outer_iter().enumerate() {
            let xr = x.as_ref().map(|m| m.row(i));
            let data = match xr.as_ref() {
                Some(r) => r.as_slice().unwrap_or_else(|| {
                    xbuf.clear();
                    xbuf.extend(r.iter().copied());
                    &xbuf
                }),
                None => &[],
            };
            let objective = row.as_slice().unwrap_or_else(|| {
                ybuf.clear();
                ybuf.extend(row.iter().copied());
                &ybuf
            });
            result.push(self.add(objective, data).inspect_err(|e| {
                let _ = e
                    .value(py)
                    .call_method1("add_note", (format!("at batch row {i}"),));
            })?);
        }
        Ok(result.into_pyarray(py))
    }
    fn front<'py>(&self, py: Python<'py>) -> Front<'py> {
        match &self.archive {
            Archive::Hv(a) => {
                let mut x = vec![];
                let mut y = vec![];
                for e in a.entries() {
                    x.extend_from_slice(e.payload());
                    y.extend(e.point());
                }
                (matrix(py, a.len(), self.dx, x), matrix(py, a.len(), 2, y))
            }
            Archive::Nd(a) => snapshot(
                py,
                self.dx,
                self.dy,
                a.ordered_entries()
                    .into_iter()
                    .map(|(y, x)| (x.as_slice(), y.as_slice())),
            ),
        }
    }
}

macro_rules! archive_class {
    ($class:ident, $name:literal, $doc:literal, $dim:ident: $arg:ty, $create:ident $(, $extra:item)*) => {
        #[doc = $doc]
        #[pyclass(name = $name, module = "fcmaes_rust._fcmaes_ext")]
        struct $class {
            inner: Store,
        }

        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature = ($dim, payload_dim=0))]
            fn new($dim: $arg, payload_dim: usize) -> PyResult<Self> {
                Ok(Self { inner: $create($dim, payload_dim)? })
            }

            /// Insert one objective row and optional decision payload.
            #[pyo3(signature = (y, x=None))]
            fn add(&mut self, y: F64Vec<'_>, x: Option<F64Vec<'_>>) -> PyResult<bool> {
                let values = slice_or_vec(&y);
                let payload = x.as_ref().map(|row| slice_or_vec(row)).unwrap_or_default();
                self.inner.add(&values, &payload)
            }

            /// Insert observations in arrival order and return their Boolean admission mask.
            #[pyo3(signature = (ys, xs=None))]
            fn add_batch<'py>(
                &mut self, py: Python<'py>, ys: F64Mat<'py>, xs: Option<F64Mat<'py>>,
            ) -> PyResult<Bound<'py, PyArray1<bool>>> {
                self.inner.add_batch(py, ys, xs)
            }

            /// Return independent payload and objective matrices.
            fn front<'py>(&self, py: Python<'py>) -> Front<'py> {
                self.inner.front(py)
            }

            /// Length of the payload stored with each row.
            #[getter]
            fn payload_dim(&self) -> usize {
                self.inner.dx
            }

            fn __len__(&self) -> usize {
                match &self.inner.archive {
                    Archive::Hv(a) => a.len(),
                    Archive::Nd(a) => a.len(),
                }
            }

            /// Discard all entries and retain the configured dimensions and reference.
            fn clear(&mut self) {
                match &mut self.inner.archive {
                    Archive::Hv(a) => a.clear(),
                    Archive::Nd(a) => a.clear(),
                }
            }

            $($extra)*
        }
    }
}
fn make_hv(reference: F64Vec<'_>, dx: usize) -> PyResult<Store> {
    let r = slice_or_vec(&reference);
    let pair = r
        .try_into()
        .map_err(|_| invalid("reference must have two coordinates"))?;
    Ok(Store {
        archive: Archive::Hv(ParetoArchive2D::new(pair).map_err(hv_error)?),
        dx,
        dy: 2,
    })
}
fn make_nd(nobj: usize, dx: usize) -> PyResult<Store> {
    if nobj == 0 {
        return Err(invalid("objective count must be positive"));
    }
    Ok(Store {
        archive: Archive::Nd(ParetoArchiveND::new(nobj)),
        dx,
        dy: nobj,
    })
}
archive_class!(
    PyNd, "ParetoArchiveND", "Strict Pareto archive with duplicate observations and optional payloads.",
    nobj: usize, make_nd,
    /// Number of objectives.
    #[getter]
    fn nobj(&self) -> usize {
        self.inner.dy
    }
);
archive_class!(
    PyHv, "ParetoArchive2D", "Exact reference-bounded 2D archive with incremental hypervolume.",
    reference: F64Vec<'_>, make_hv,
    /// Hypervolume reference point.
    #[getter]
    fn reference(&self) -> (f64, f64) {
        match &self.inner.archive {
            Archive::Hv(a) => a.reference().into(),
            Archive::Nd(_) => unreachable!("ParetoArchive2D always stores the 2D archive"),
        }
    },
    /// Dominated area up to the configured reference point.
    fn hypervolume(&self) -> f64 {
        match &self.inner.archive {
            Archive::Hv(a) => a.hypervolume(),
            Archive::Nd(_) => unreachable!("ParetoArchive2D always stores the 2D archive"),
        }
    }
);

/// Synchronous archive-gated DE/rand/1/bin (F = 0.5, CR = 0.3) with a hard evaluation cap.
#[pyclass(name = "ArchiveDE", module = "fcmaes_rust._fcmaes_ext")]
struct PyDe {
    inner: ArchiveDe<Vec<f64>, Vec<f64>>,
}
#[pymethods]
impl PyDe {
    #[new]
    #[pyo3(signature=(nobj,lower,upper,*,popsize=100,max_evaluations=100_000,seed=0,gate=true,tiered=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nobj: usize,
        lower: F64Vec<'_>,
        upper: F64Vec<'_>,
        popsize: usize,
        max_evaluations: u64,
        seed: u32,
        gate: bool,
        tiered: bool,
    ) -> PyResult<Self> {
        let params = ArchiveDeParams {
            population: popsize,
            max_evaluations,
            seed,
            gate,
            tiered,
        };
        Ok(Self {
            inner: ArchiveDe::new(slice_or_vec(&lower), slice_or_vec(&upper), nobj, params)
                .map_err(invalid)?,
        })
    }
    /// An independent decision matrix, empty once the evaluation budget is exhausted.
    fn ask<'py>(&mut self, py: Python<'py>) -> Matrix<'py> {
        let d = self.inner.dimension();
        let x = self.inner.ask();
        matrix(py, x.len(), d, x.iter().flatten().copied().collect())
    }
    /// Submit the finite objective matrix for the entire pending batch in order.
    fn tell(&mut self, py: Python<'_>, ys: F64Mat<'_>) -> PyResult<()> {
        let rows = matrix_rows(&ys);
        py.allow_threads(|| self.inner.tell(&rows)).map_err(invalid)
    }
    /// Return the current breeding decisions and objectives.
    fn population<'py>(&self, py: Python<'py>) -> Front<'py> {
        let rows = self.inner.population();
        snapshot(
            py,
            self.inner.dimension(),
            self.inner.objectives(),
            rows.iter().map(|(x, y)| (x.as_slice(), y.as_slice())),
        )
    }
    /// Return all nondominated decisions and objectives.
    fn front<'py>(&self, py: Python<'py>) -> Front<'py> {
        snapshot(
            py,
            self.inner.dimension(),
            self.inner.objectives(),
            self.inner
                .archive()
                .ordered_entries()
                .into_iter()
                .map(|(y, x)| (x.as_slice(), y.as_slice())),
        )
    }
    /// Number of submitted objective rows.
    #[getter]
    fn nfev(&self) -> u64 {
        self.inner.evaluations()
    }
}

/// Inverted generational distance: the mean Euclidean distance from each
/// reference row to its nearest front row. The approximation front comes first.
#[pyfunction]
fn igd(py: Python<'_>, front: F64Mat<'_>, reference: F64Mat<'_>) -> PyResult<f64> {
    let (a, b) = (matrix_rows(&front), matrix_rows(&reference));
    py.allow_threads(|| fcmaes_core::igd(&a, &b))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBorda>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyNd>()?;
    m.add_class::<PyHv>()?;
    m.add_class::<PyDe>()?;
    m.add_function(wrap_pyfunction!(igd, m)?)?;
    Ok(())
}
