//! PyO3 bindings for the native GTOP benchmark crate.

use fcmaes_gtop as gtop;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::slice_or_vec;

macro_rules! scalar_benchmark {
    ($rust_name:ident, $python_name:literal, $function:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction(name = $python_name)]
        fn $rust_name(x: PyReadonlyArray1<'_, f64>) -> f64 {
            $function(&slice_or_vec(&x))
        }
    };
}

scalar_benchmark!(
    gtoc1,
    "gtop_gtoc1",
    gtop::gtoc1,
    "Evaluate the eight-variable GTOC1 asteroid-impact objective.\n\n\
     ``x`` is the native GTOP decision vector. The minimized scalar objective \
     is returned; invalid or undersized vectors receive a finite penalty."
);
scalar_benchmark!(
    cassini1,
    "gtop_cassini1",
    gtop::cassini1,
    "Evaluate the six-variable continuous Cassini 1 objective.\n\n\
     Returns the minimized mission delta-v objective. Invalid or undersized \
     vectors receive a finite penalty."
);
scalar_benchmark!(
    messenger,
    "gtop_messenger",
    gtop::messenger,
    "Evaluate the 18-variable reduced Messenger trajectory objective.\n\n\
     Returns the minimized rendezvous delta-v objective. Invalid or undersized \
     vectors receive a finite penalty."
);
scalar_benchmark!(
    messenger_full,
    "gtop_messengerfull",
    gtop::messenger_full,
    "Evaluate the 26-variable full Messenger trajectory objective.\n\n\
     Returns the minimized orbit-insertion objective. Invalid or undersized \
     vectors receive a finite penalty."
);
scalar_benchmark!(
    cassini2,
    "gtop_cassini2",
    gtop::cassini2,
    "Evaluate the 22-variable continuous Cassini 2 objective.\n\n\
     Returns the minimized rendezvous delta-v objective. Invalid or undersized \
     vectors receive a finite penalty."
);
scalar_benchmark!(
    rosetta,
    "gtop_rosetta",
    gtop::rosetta,
    "Evaluate the 22-variable Rosetta trajectory objective.\n\n\
     Returns the minimized rendezvous objective. Invalid or undersized vectors \
     receive a finite penalty."
);
scalar_benchmark!(
    sagas,
    "gtop_sagas",
    gtop::sagas,
    "Evaluate the 12-variable SAGAS trajectory objective.\n\n\
     Returns the minimized time-to-50-AU objective with delta-v penalties. \
     Invalid or undersized vectors receive a finite penalty."
);
scalar_benchmark!(
    cassini2_minlp,
    "gtop_cassini2_minlp",
    gtop::cassini2_minlp,
    "Evaluate the 26-variable mixed-integer Cassini 2 objective.\n\n\
     The final four coordinates encode intermediate planet identifiers. \
     Invalid or undersized vectors receive a finite penalty."
);

fn extract_sequence(sequence: Vec<i64>) -> PyResult<Vec<usize>> {
    if sequence.len() != 5 || sequence.iter().any(|&body| !(1..=6).contains(&body)) {
        return Err(PyValueError::new_err(
            "sequence must contain exactly five planet ids in 1..=6",
        ));
    }
    Ok(sequence.into_iter().map(|body| body as usize).collect())
}

/// Evaluate the constrained 18-variable TandEM objective for a planet sequence.
///
/// ``sequence`` must contain exactly five integer planet identifiers in
/// ``1..=6``. The objective includes the flight-time constraint penalty.
///
/// Raises ``ValueError`` if ``sequence`` is not five identifiers in ``1..=6``.
/// Invalid or undersized ``x`` receives a finite penalty rather than raising.
#[pyfunction(name = "gtop_tandem")]
fn tandem(x: PyReadonlyArray1<'_, f64>, sequence: Vec<i64>) -> PyResult<f64> {
    Ok(gtop::tandem(
        &slice_or_vec(&x),
        &extract_sequence(sequence)?,
    ))
}

/// Evaluate the unconstrained 18-variable TandEM objective.
///
/// ``sequence`` must contain exactly five integer planet identifiers in
/// ``1..=6``. Unlike :func:`gtop_tandem`, this value omits the flight-time
/// penalty.
///
/// Raises ``ValueError`` if ``sequence`` is not five identifiers in ``1..=6``.
/// Invalid or undersized ``x`` receives a finite penalty rather than raising.
#[pyfunction(name = "gtop_tandem_unconstrained")]
fn tandem_unconstrained(x: PyReadonlyArray1<'_, f64>, sequence: Vec<i64>) -> PyResult<f64> {
    Ok(gtop::tandem_unconstrained(
        &slice_or_vec(&x),
        &extract_sequence(sequence)?,
    ))
}

/// Evaluate mixed-integer Cassini 1 and return objective and launch delta-v.
///
/// ``x`` contains six continuous trajectory coordinates followed by four
/// integer-valued planet identifiers. Invalid or undersized vectors return
/// finite penalty values.
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
