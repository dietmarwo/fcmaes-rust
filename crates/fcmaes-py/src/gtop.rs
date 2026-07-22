//! PyO3 bindings for the native GTOP benchmark crate.

use fcmaes_examples::gtop;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::slice_or_vec;

macro_rules! scalar_benchmark {
    ($rust_name:ident, $python_name:literal, $function:path) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(x: PyReadonlyArray1<'_, f64>) -> f64 {
            $function(&slice_or_vec(&x))
        }
    };
}

scalar_benchmark!(gtoc1, "gtop_gtoc1", gtop::gtoc1);
scalar_benchmark!(cassini1, "gtop_cassini1", gtop::cassini1);
scalar_benchmark!(messenger, "gtop_messenger", gtop::messenger);
scalar_benchmark!(messenger_full, "gtop_messengerfull", gtop::messenger_full);
scalar_benchmark!(cassini2, "gtop_cassini2", gtop::cassini2);
scalar_benchmark!(rosetta, "gtop_rosetta", gtop::rosetta);
scalar_benchmark!(sagas, "gtop_sagas", gtop::sagas);
scalar_benchmark!(cassini2_minlp, "gtop_cassini2_minlp", gtop::cassini2_minlp);

fn extract_sequence(sequence: Vec<i64>) -> PyResult<Vec<usize>> {
    if sequence.len() != 5 || sequence.iter().any(|&body| !(1..=6).contains(&body)) {
        return Err(PyValueError::new_err(
            "sequence must contain exactly five planet ids in 1..=6",
        ));
    }
    Ok(sequence.into_iter().map(|body| body as usize).collect())
}

#[pyfunction(name = "gtop_tandem")]
fn tandem(x: PyReadonlyArray1<'_, f64>, sequence: Vec<i64>) -> PyResult<f64> {
    Ok(gtop::tandem(
        &slice_or_vec(&x),
        &extract_sequence(sequence)?,
    ))
}

#[pyfunction(name = "gtop_tandem_unconstrained")]
fn tandem_unconstrained(x: PyReadonlyArray1<'_, f64>, sequence: Vec<i64>) -> PyResult<f64> {
    Ok(gtop::tandem_unconstrained(
        &slice_or_vec(&x),
        &extract_sequence(sequence)?,
    ))
}

#[pyfunction(name = "gtop_cassini1_minlp")]
fn cassini1_minlp(x: PyReadonlyArray1<'_, f64>) -> (f64, f64) {
    gtop::cassini1_minlp(&slice_or_vec(&x))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gtoc1, module)?)?;
    module.add_function(wrap_pyfunction!(cassini1, module)?)?;
    module.add_function(wrap_pyfunction!(messenger, module)?)?;
    module.add_function(wrap_pyfunction!(messenger_full, module)?)?;
    module.add_function(wrap_pyfunction!(cassini2, module)?)?;
    module.add_function(wrap_pyfunction!(rosetta, module)?)?;
    module.add_function(wrap_pyfunction!(sagas, module)?)?;
    module.add_function(wrap_pyfunction!(tandem, module)?)?;
    module.add_function(wrap_pyfunction!(tandem_unconstrained, module)?)?;
    module.add_function(wrap_pyfunction!(cassini1_minlp, module)?)?;
    module.add_function(wrap_pyfunction!(cassini2_minlp, module)?)?;
    Ok(())
}
