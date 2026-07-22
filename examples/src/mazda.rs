//! Rust-side adapter for the Mazda factory-design benchmark.
//!
//! The supplied Mazda model contains roughly 65,000 lines of generated C++
//! response-surface equations. This module loads its stable C entry point and
//! keeps decision decoding, constraint convention, MO/QD scalarization, and
//! optimization orchestration in Rust.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::path::Path;

pub const MAZDA_DIM: usize = 222;
pub const MAZDA_OBJECTIVES: usize = 2;
pub const MAZDA_CONSTRAINTS: usize = 54;
pub const MAZDA_VALUE_WIDTH: usize = MAZDA_OBJECTIVES + MAZDA_CONSTRAINTS;
pub const MAZDA_QD_LOWER: [f64; 2] = [2.0, -74.0];
pub const MAZDA_QD_UPPER: [f64; 2] = [3.5, 0.0];

type MazdaFitnessFn = unsafe extern "C" fn(*mut f64, c_int, *mut f64, *mut f64);

#[cfg(unix)]
mod loader {
    use super::*;

    const RTLD_NOW: c_int = 2;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    pub(super) struct DynamicLibrary {
        handle: *mut c_void,
    }

    impl DynamicLibrary {
        pub(super) fn open(path: &Path) -> Result<Self, String> {
            let path = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| "Mazda library path contains a NUL byte".to_string())?;
            // SAFETY: `path` is a live NUL-terminated string and RTLD_NOW is a
            // valid POSIX loader flag. A null handle is converted to an error.
            let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return Err(last_error("unable to load Mazda library"));
            }
            Ok(Self { handle })
        }

        pub(super) fn symbol(&self, name: &str) -> Result<MazdaFitnessFn, String> {
            let name = CString::new(name).map_err(|_| "symbol contains NUL".to_string())?;
            // SAFETY: `self.handle` remains open for this object's lifetime and
            // `name` is NUL terminated. The benchmark publishes this exact C ABI.
            let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
            if symbol.is_null() {
                return Err(last_error("unable to resolve Mazda fitness symbol"));
            }
            // SAFETY: the symbol is `fitness_MazdaMop_C`, whose declaration in
            // mazda_mop.cpp exactly matches `MazdaFitnessFn`.
            Ok(unsafe { std::mem::transmute::<*mut c_void, MazdaFitnessFn>(symbol) })
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            // SAFETY: this handle was returned by `dlopen` and is closed once.
            unsafe {
                dlclose(self.handle);
            }
        }
    }

    fn last_error(prefix: &str) -> String {
        // SAFETY: POSIX `dlerror` returns either null or a NUL-terminated string
        // owned by the loader; it is copied before another loader call.
        let detail = unsafe {
            let error = dlerror();
            if error.is_null() {
                "unknown dynamic-loader error".to_string()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        };
        format!("{prefix}: {detail}")
    }
}

#[cfg(not(unix))]
mod loader {
    use super::*;

    pub(super) struct DynamicLibrary;

    impl DynamicLibrary {
        pub(super) fn open(_path: &Path) -> Result<Self, String> {
            Err("the Mazda dynamic-library adapter currently supports Unix targets".to_string())
        }

        pub(super) fn symbol(&self, _name: &str) -> Result<MazdaFitnessFn, String> {
            Err("the Mazda dynamic-library adapter currently supports Unix targets".to_string())
        }
    }
}

/// Discrete Mazda thickness choices parsed from the supplied Python sample.
#[derive(Clone, Debug)]
pub struct MazdaDecisionSpace {
    choices: Vec<Vec<f64>>,
}

impl MazdaDecisionSpace {
    /// Load the three-car `decision_x` table from `mazda.py`.
    pub fn from_python_sample(path: impl AsRef<Path>) -> Result<Self, String> {
        let source = fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "unable to read Mazda decision table {}: {error}",
                path.as_ref().display()
            )
        })?;
        let marker = "decision_x = [";
        let start = source
            .match_indices(marker)
            .nth(1)
            .map(|(index, _)| index + marker.len() - 1)
            .ok_or_else(|| "three-car decision_x table not found in Mazda sample".to_string())?;
        let choices = parse_nested_float_lists(&source[start..])?;
        if choices.len() != MAZDA_DIM || choices.iter().any(Vec::is_empty) {
            return Err(format!(
                "expected {MAZDA_DIM} non-empty Mazda choice lists, found {}",
                choices.len()
            ));
        }
        Ok(Self { choices })
    }

    pub fn dim(&self) -> usize {
        self.choices.len()
    }

    pub fn lower(&self) -> Vec<f64> {
        vec![0.0; self.dim()]
    }

    pub fn upper(&self) -> Vec<f64> {
        self.choices
            .iter()
            .map(|values| values.len() as f64 - 1e-9)
            .collect()
    }

    /// Convert MODE/MAP-Elites index coordinates into physical thicknesses.
    /// Coordinates follow Python's `int(xi)` convention.
    pub fn decode(&self, indices: &[f64]) -> Result<Vec<f64>, String> {
        if indices.len() != self.dim() {
            return Err(format!(
                "Mazda decision vector has length {}, expected {}",
                indices.len(),
                self.dim()
            ));
        }
        indices
            .iter()
            .zip(&self.choices)
            .map(|(&index, values)| {
                if !index.is_finite() {
                    return Err("Mazda decision index is not finite".to_string());
                }
                let selected = index.trunc().clamp(0.0, (values.len() - 1) as f64) as usize;
                Ok(values[selected])
            })
            .collect()
    }

    pub fn choices(&self) -> &[Vec<f64>] {
        &self.choices
    }
}

/// Loaded Mazda three-car response-surface evaluator.
pub struct MazdaEvaluator {
    _library: loader::DynamicLibrary,
    fitness: MazdaFitnessFn,
}

impl MazdaEvaluator {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let library = loader::DynamicLibrary::open(path.as_ref())?;
        let fitness = library.symbol("fitness_MazdaMop_C")?;
        Ok(Self {
            _library: library,
            fitness,
        })
    }

    /// Return `[mass, -common_parts, constraints...]`, converting the Mazda C++
    /// convention (feasible `con >= 0`) to MODE's feasible `constraint <= 0`.
    pub fn evaluate_physical(&self, physical: &[f64]) -> Result<Vec<f64>, String> {
        if physical.len() != MAZDA_DIM {
            return Err(format!(
                "Mazda physical vector has length {}, expected {MAZDA_DIM}",
                physical.len()
            ));
        }
        let mut input = physical.to_vec();
        let mut objectives = [0.0; 5];
        let mut constraints = [0.0; MAZDA_CONSTRAINTS];
        // SAFETY: all buffers are writable and sized according to the published
        // three-car C interface; the dynamic library stays loaded in `self`.
        unsafe {
            (self.fitness)(
                input.as_mut_ptr(),
                MAZDA_DIM as c_int,
                objectives.as_mut_ptr(),
                constraints.as_mut_ptr(),
            );
        }
        if objectives.iter().any(|value| !value.is_finite())
            || constraints.iter().any(|value| !value.is_finite())
        {
            return Err("Mazda evaluator returned a non-finite value".to_string());
        }
        let mut values = Vec::with_capacity(MAZDA_VALUE_WIDTH);
        values.extend_from_slice(&objectives[..MAZDA_OBJECTIVES]);
        values.extend(constraints.iter().map(|&constraint| -constraint));
        Ok(values)
    }

    pub fn evaluate_indices(
        &self,
        space: &MazdaDecisionSpace,
        indices: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.evaluate_physical(&space.decode(indices)?)
    }
}

/// Mazda QD fitness and behavior descriptor used by the Python sample.
pub fn qd_value(values: &[f64]) -> Result<(f64, Vec<f64>), String> {
    if values.len() != MAZDA_VALUE_WIDTH || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "Mazda QD input must contain {MAZDA_VALUE_WIDTH} finite values"
        ));
    }
    let constraint_penalty: f64 = values[MAZDA_OBJECTIVES..]
        .iter()
        .filter(|&&constraint| constraint > 0.0)
        .map(|&constraint| 1_000.0 * (constraint + 1.0))
        .sum();
    let mass =
        ((values[0] - MAZDA_QD_LOWER[0]) / (MAZDA_QD_UPPER[0] - MAZDA_QD_LOWER[0])).min(100_000.0);
    let common_parts =
        ((values[1] - MAZDA_QD_LOWER[1]) / (MAZDA_QD_UPPER[1] - MAZDA_QD_LOWER[1])).min(100_000.0);
    Ok((
        mass + common_parts + constraint_penalty,
        values[..MAZDA_OBJECTIVES].to_vec(),
    ))
}

pub fn is_feasible(values: &[f64]) -> bool {
    values.len() == MAZDA_VALUE_WIDTH
        && values[MAZDA_OBJECTIVES..]
            .iter()
            .all(|&constraint| constraint <= 0.0)
}

fn parse_nested_float_lists(source: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut depth = 0usize;
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut token = String::new();
    let mut started = false;

    let flush_number = |token: &mut String, row: &mut Vec<f64>| -> Result<(), String> {
        if token.is_empty() {
            return Ok(());
        }
        let value = token
            .parse::<f64>()
            .map_err(|_| format!("invalid number in Mazda decision table: {token}"))?;
        row.push(value);
        token.clear();
        Ok(())
    };

    for character in source.chars() {
        match character {
            '[' => {
                depth += 1;
                started = true;
                if depth == 2 {
                    row.clear();
                }
            }
            ']' if started => {
                if depth == 2 {
                    flush_number(&mut token, &mut row)?;
                    rows.push(std::mem::take(&mut row));
                }
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Ok(rows);
                }
            }
            ',' if depth == 2 => flush_number(&mut token, &mut row)?,
            '0'..='9' | '.' | '-' | '+' | 'e' | 'E' if depth == 2 => token.push(character),
            _ => {}
        }
    }
    Err("unterminated Mazda decision_x table".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_lists() {
        let rows = parse_nested_float_lists("[[0.3, 1.0], [-2, 4e-1]] trailing").unwrap();
        assert_eq!(rows, vec![vec![0.3, 1.0], vec![-2.0, 0.4]]);
    }

    #[test]
    fn qd_conversion_penalizes_violations() {
        let mut values = vec![0.0; MAZDA_VALUE_WIDTH];
        values[0] = 2.75;
        values[1] = -37.0;
        let (valid, descriptor) = qd_value(&values).unwrap();
        assert_eq!(valid, 1.0);
        assert_eq!(descriptor, vec![2.75, -37.0]);
        values[2] = 0.25;
        assert_eq!(qd_value(&values).unwrap().0, 1_251.0);
        assert!(!is_feasible(&values));
    }

    #[test]
    fn rejects_bad_qd_width() {
        assert!(qd_value(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn supplied_decision_table_has_expected_shape() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mazda/mazda_cpp");
        let source = root.join("Mazda_CdMOBP/src/mazda.py");
        if !source.exists() {
            return;
        }
        let space = MazdaDecisionSpace::from_python_sample(source).unwrap();
        assert_eq!(space.dim(), MAZDA_DIM);
        assert!(space.choices().iter().all(|choices| choices.len() >= 5));
        assert_eq!(space.upper().len(), MAZDA_DIM);
    }

    #[test]
    fn dynamic_adapter_matches_supplied_reference_row() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mazda/mazda_cpp");
        let library = root.join("Mazda_CdMOBP/src/libmazda.so");
        let variables = root.join("Mazda_CdMOBP/sample/pop_vars_eval.txt");
        let objectives = root.join("Mazda_CdMOBP/sample/ref_pop_objs_eval.txt");
        let constraints = root.join("Mazda_CdMOBP/sample/ref_pop_cons_eval.txt");
        if !library.exists() || !variables.exists() || !objectives.exists() || !constraints.exists()
        {
            return;
        }
        let first_row = |path: &Path| -> Vec<f64> {
            fs::read_to_string(path)
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .map(|token| token.parse().unwrap())
                .collect()
        };
        let x = first_row(&variables);
        let expected_objectives = first_row(&objectives);
        let expected_constraints = first_row(&constraints);
        assert_eq!(x.len(), MAZDA_DIM);
        assert_eq!(expected_constraints.len(), MAZDA_CONSTRAINTS);
        let actual = MazdaEvaluator::load(library)
            .unwrap()
            .evaluate_physical(&x)
            .unwrap();
        assert!((actual[0] - expected_objectives[0]).abs() < 1e-7);
        assert!((actual[1] - expected_objectives[1]).abs() < 1e-7);
        for (&value, &expected) in actual[2..].iter().zip(&expected_constraints) {
            assert!((value + expected).abs() < 1e-6);
        }
    }
}
