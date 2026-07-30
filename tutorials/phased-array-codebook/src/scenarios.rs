//! Named deterministic training and holdout perturbations.

use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

use num_complex::Complex64;

use crate::array::{AngleGrid, Array, ArrayLayout};
use crate::kernel::SteeringMatrix;
use crate::metrics::{PatternMetrics, analyse_linear, degenerate_metrics};

const ELEMENTS: usize = 16;
const TRAINING_PHASE: &str = include_str!("../scenarios/phase-training.csv");
const TRAINING_AMPLITUDE: &str = include_str!("../scenarios/amplitude-training.csv");
const HOLDOUT_PHASE: &str = include_str!("../scenarios/phase-holdout.csv");
const HOLDOUT_SPACING: &str = include_str!("../scenarios/spacing-holdout.csv");
const DUAL_FAILURE_PAIRS: [(usize, usize); 24] = [
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (10, 13),
    (11, 14),
    (12, 15),
    (13, 0),
    (14, 1),
    (15, 2),
    (0, 15),
    (1, 7),
    (2, 13),
    (3, 12),
    (4, 11),
    (5, 10),
    (6, 12),
    (7, 8),
];

/// Scenario family, used to assert train/holdout disjointness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioKind {
    /// No perturbation.
    Nominal,
    /// Fixed phase-error table.
    PhaseError5Deg,
    /// Fixed amplitude-error table.
    AmplitudeError0p5Db,
    /// Exhaustive single-element failures.
    SingleFailure,
    /// Fixed non-adjacent dual failures.
    DualFailure,
    /// Stronger phase errors used only for holdout.
    PhaseError10Deg,
    /// Adjacent dual failures used only for holdout.
    AdjacentFailure,
    /// Element-position errors used only for holdout.
    SpacingError,
}

/// Aggregated robust cut metrics.
#[derive(Clone, Debug)]
pub struct RobustMetrics {
    /// Unperturbed pattern.
    pub nominal: PatternMetrics,
    /// Largest (worst) PSLL in dB.
    pub worst_psll_db: f64,
    /// Empirical 90th percentile PSLL in dB.
    pub quantile_psll_db: f64,
    /// Number of physical scenario evaluations.
    pub scenario_count: usize,
    /// Physically degenerate patterns, penalized through worst-case PSLL.
    pub degenerate_scenarios: usize,
    /// Genuine field-kernel evaluation failures.
    pub kernel_failures: usize,
}

/// A frozen 16-element scenario table was applied to another array size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScenarioDimensionError {
    /// Scenario-table element count.
    pub expected: usize,
    /// Excitation/geometry element count.
    pub actual: usize,
}

impl Display for ScenarioDimensionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "scenario table has {} elements, evaluation has {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for ScenarioDimensionError {}

/// Precomputed holdout geometry and steering matrices.
#[derive(Clone, Debug)]
pub struct HoldoutContext {
    array: Array,
    grid: AngleGrid,
    nominal: SteeringMatrix,
    spacing: Vec<SteeringMatrix>,
}

impl HoldoutContext {
    /// Build the nominal and fixed spacing-error matrices once.
    pub fn new(array: Array, grid: AngleGrid) -> Result<Self, ScenarioDimensionError> {
        ensure_elements(array.positions.len())?;
        let nominal = SteeringMatrix::build(&array, &grid);
        let spacing = spacing_holdout()
            .iter()
            .map(|draw| {
                let mut perturbed = array.clone();
                for (position, offset) in perturbed.positions.iter_mut().zip(draw) {
                    position[0] += offset * perturbed.wavelength;
                }
                perturbed.layout = ArrayLayout::General;
                SteeringMatrix::build(&perturbed, &grid)
            })
            .collect();
        Ok(Self {
            array,
            grid,
            nominal,
            spacing,
        })
    }

    /// Frozen holdout array.
    #[must_use]
    pub const fn array(&self) -> &Array {
        &self.array
    }

    /// Frozen holdout cut.
    #[must_use]
    pub const fn grid(&self) -> &AngleGrid {
        &self.grid
    }
}

fn parse_table(contents: &str) -> Vec<[f64; ELEMENTS]> {
    contents
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let values = line
                .split(',')
                .skip(1)
                .map(|value| {
                    value
                        .parse::<f64>()
                        .expect("checked-in scenario value is numeric")
                })
                .collect::<Vec<_>>();
            values
                .try_into()
                .expect("checked-in scenario row has exactly 16 elements")
        })
        .collect()
}

fn phase_training() -> &'static [[f64; ELEMENTS]] {
    static TABLE: OnceLock<Vec<[f64; ELEMENTS]>> = OnceLock::new();
    TABLE.get_or_init(|| parse_table(TRAINING_PHASE))
}

fn amplitude_training() -> &'static [[f64; ELEMENTS]] {
    static TABLE: OnceLock<Vec<[f64; ELEMENTS]>> = OnceLock::new();
    TABLE.get_or_init(|| parse_table(TRAINING_AMPLITUDE))
}

fn phase_holdout() -> &'static [[f64; ELEMENTS]] {
    static TABLE: OnceLock<Vec<[f64; ELEMENTS]>> = OnceLock::new();
    TABLE.get_or_init(|| parse_table(HOLDOUT_PHASE))
}

fn spacing_holdout() -> &'static [[f64; ELEMENTS]] {
    static TABLE: OnceLock<Vec<[f64; ELEMENTS]>> = OnceLock::new();
    TABLE.get_or_init(|| parse_table(HOLDOUT_SPACING))
}

#[derive(Clone, Debug)]
struct ScenarioMetrics {
    metrics: PatternMetrics,
    kernel_failed: bool,
}

fn metrics_with_field(
    matrix: &SteeringMatrix,
    grid: &AngleGrid,
    excitation: &[Complex64],
) -> (ScenarioMetrics, Vec<Complex64>) {
    let mut field = vec![Complex64::new(0.0, 0.0); grid.len()];
    if matrix.field_direct(excitation, &mut field).is_err() {
        return (
            ScenarioMetrics {
                metrics: degenerate_metrics(),
                kernel_failed: true,
            },
            field,
        );
    }
    (
        ScenarioMetrics {
            metrics: analyse_linear(&field, grid, excitation, None),
            kernel_failed: false,
        },
        field,
    )
}

fn metrics(matrix: &SteeringMatrix, grid: &AngleGrid, excitation: &[Complex64]) -> ScenarioMetrics {
    metrics_with_field(matrix, grid, excitation).0
}

fn ensure_elements(actual: usize) -> Result<(), ScenarioDimensionError> {
    (actual == ELEMENTS)
        .then_some(())
        .ok_or(ScenarioDimensionError {
            expected: ELEMENTS,
            actual,
        })
}

fn phase_perturbed(
    excitation: &[Complex64],
    errors_deg: &[f64],
) -> Result<Vec<Complex64>, ScenarioDimensionError> {
    ensure_elements(excitation.len())?;
    ensure_elements(errors_deg.len())?;
    Ok(excitation
        .iter()
        .zip(errors_deg)
        .map(|(weight, error)| weight * Complex64::from_polar(1.0, error * PI / 180.0))
        .collect())
}

fn amplitude_perturbed(
    excitation: &[Complex64],
    errors_db: &[f64],
) -> Result<Vec<Complex64>, ScenarioDimensionError> {
    ensure_elements(excitation.len())?;
    ensure_elements(errors_db.len())?;
    Ok(excitation
        .iter()
        .zip(errors_db)
        .map(|(weight, error)| weight * 10_f64.powf(error / 20.0))
        .collect())
}

fn failures(
    excitation: &[Complex64],
    indices: &[usize],
) -> Result<Vec<Complex64>, ScenarioDimensionError> {
    ensure_elements(excitation.len())?;
    if indices.iter().any(|index| *index >= excitation.len()) {
        return Err(ScenarioDimensionError {
            expected: excitation.len(),
            actual: indices.iter().copied().max().unwrap_or(0) + 1,
        });
    }
    let mut perturbed = excitation.to_vec();
    for index in indices {
        perturbed[*index] = Complex64::new(0.0, 0.0);
    }
    Ok(perturbed)
}

fn aggregate(nominal: PatternMetrics, patterns: Vec<ScenarioMetrics>) -> RobustMetrics {
    let mut psll = patterns
        .iter()
        .map(|scenario| {
            if scenario.metrics.degenerate || !scenario.metrics.psll_db.is_finite() {
                0.0
            } else {
                scenario.metrics.psll_db
            }
        })
        .collect::<Vec<_>>();
    psll.sort_by(f64::total_cmp);
    let degenerate_scenarios = patterns
        .iter()
        .filter(|scenario| scenario.metrics.degenerate)
        .count();
    let kernel_failures = patterns
        .iter()
        .filter(|scenario| scenario.kernel_failed)
        .count();
    let worst_psll_db = psll.last().copied().unwrap_or(0.0);
    let quantile_index = ((psll.len() as f64 * 0.9).ceil() as usize)
        .saturating_sub(1)
        .min(psll.len().saturating_sub(1));
    let quantile_psll_db = psll.get(quantile_index).copied().unwrap_or(0.0);
    RobustMetrics {
        nominal,
        worst_psll_db,
        quantile_psll_db,
        scenario_count: patterns.len(),
        degenerate_scenarios,
        kernel_failures,
    }
}

fn evaluate_training_impl(
    excitation: &[Complex64],
    matrix: &SteeringMatrix,
    grid: &AngleGrid,
) -> Result<(RobustMetrics, Vec<Complex64>), ScenarioDimensionError> {
    ensure_elements(excitation.len())?;
    let (nominal_scenario, nominal_field) = metrics_with_field(matrix, grid, excitation);
    let nominal = nominal_scenario.metrics.clone();
    let mut patterns = vec![nominal_scenario];
    for draw in phase_training() {
        patterns.push(metrics(matrix, grid, &phase_perturbed(excitation, draw)?));
    }
    for draw in amplitude_training() {
        patterns.push(metrics(
            matrix,
            grid,
            &amplitude_perturbed(excitation, draw)?,
        ));
    }
    for failed in 0..excitation.len() {
        patterns.push(metrics(matrix, grid, &failures(excitation, &[failed])?));
    }
    for &(left, right) in &DUAL_FAILURE_PAIRS {
        patterns.push(metrics(
            matrix,
            grid,
            &failures(excitation, &[left, right])?,
        ));
    }
    Ok((aggregate(nominal, patterns), nominal_field))
}

/// Evaluate the frozen training set.
pub fn evaluate_training(
    excitation: &[Complex64],
    matrix: &SteeringMatrix,
    grid: &AngleGrid,
) -> Result<RobustMetrics, ScenarioDimensionError> {
    evaluate_training_impl(excitation, matrix, grid).map(|(robust, _)| robust)
}

/// Evaluate training scenarios and retain the already-computed nominal field.
pub fn evaluate_training_with_field(
    excitation: &[Complex64],
    matrix: &SteeringMatrix,
    grid: &AngleGrid,
) -> Result<(RobustMetrics, Vec<Complex64>), ScenarioDimensionError> {
    evaluate_training_impl(excitation, matrix, grid)
}

/// Evaluate disjoint holdout perturbation kinds.
pub fn evaluate_holdout(
    excitation: &[Complex64],
    context: &HoldoutContext,
) -> Result<RobustMetrics, ScenarioDimensionError> {
    ensure_elements(excitation.len())?;
    let nominal_scenario = metrics(&context.nominal, &context.grid, excitation);
    let nominal = nominal_scenario.metrics.clone();
    let mut patterns = vec![nominal_scenario];
    for draw in phase_holdout() {
        patterns.push(metrics(
            &context.nominal,
            &context.grid,
            &phase_perturbed(excitation, draw)?,
        ));
    }
    for failed in 0..excitation.len().saturating_sub(1) {
        patterns.push(metrics(
            &context.nominal,
            &context.grid,
            &failures(excitation, &[failed, failed + 1])?,
        ));
    }
    for matrix in &context.spacing {
        patterns.push(metrics(matrix, &context.grid, excitation));
    }
    Ok(aggregate(nominal, patterns))
}

/// Representative holdout pattern used only for descriptor migration.
///
/// Quality still uses the complete worst-case holdout aggregation. Keeping the
/// descriptor scenario fixed prevents selecting a different perturbation for
/// every candidate.
pub fn representative_holdout_metrics(
    excitation: &[Complex64],
    context: &HoldoutContext,
) -> Result<PatternMetrics, ScenarioDimensionError> {
    Ok(metrics(
        &context.nominal,
        &context.grid,
        &phase_perturbed(excitation, &phase_holdout()[0])?,
    )
    .metrics)
}

/// Scenario families disclosed in the optimization set.
pub const TRAINING_KINDS: [ScenarioKind; 5] = [
    ScenarioKind::Nominal,
    ScenarioKind::PhaseError5Deg,
    ScenarioKind::AmplitudeError0p5Db,
    ScenarioKind::SingleFailure,
    ScenarioKind::DualFailure,
];

/// Scenario families reserved for validation.
pub const HOLDOUT_KINDS: [ScenarioKind; 3] = [
    ScenarioKind::PhaseError10Deg,
    ScenarioKind::AdjacentFailure,
    ScenarioKind::SpacingError,
];

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;

    fn fixture() -> (Array, AngleGrid, Vec<Complex64>) {
        (
            Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0),
            AngleGrid::linear_cut(2_001),
            vec![Complex64::new(1.0, 0.0); 16],
        )
    }

    #[test]
    fn nominal_reproduces_direct_analysis_and_single_failure_is_exhaustive() {
        let (array, grid, excitation) = fixture();
        let matrix = SteeringMatrix::build(&array, &grid);
        let robust = evaluate_training(&excitation, &matrix, &grid).unwrap();
        let direct = metrics(&matrix, &grid, &excitation).metrics;
        assert_eq!(robust.nominal.psll_db.to_bits(), direct.psll_db.to_bits());
        let explicit_worst = (0..16)
            .map(|failed| {
                metrics(&matrix, &grid, &failures(&excitation, &[failed]).unwrap())
                    .metrics
                    .psll_db
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(robust.worst_psll_db >= explicit_worst);
        assert_eq!(robust.scenario_count, 49);
        assert_eq!(robust.kernel_failures, 0);
    }

    #[test]
    fn training_and_holdout_differ_by_kind_and_change_the_pattern() {
        assert!(
            TRAINING_KINDS
                .iter()
                .all(|kind| !HOLDOUT_KINDS.contains(kind))
        );
        let (array, grid, excitation) = fixture();
        let matrix = SteeringMatrix::build(&array, &grid);
        let training = evaluate_training(&excitation, &matrix, &grid).unwrap();
        let holdout_context = HoldoutContext::new(array, grid).unwrap();
        let holdout = evaluate_holdout(&excitation, &holdout_context).unwrap();
        assert_ne!(
            training.worst_psll_db.to_bits(),
            training.nominal.psll_db.to_bits()
        );
        assert_ne!(
            holdout.worst_psll_db.to_bits(),
            holdout.nominal.psll_db.to_bits()
        );
        assert_eq!(holdout.scenario_count, 22);
    }

    #[test]
    fn fixed_scenarios_reject_other_array_sizes_instead_of_truncating() {
        let array = Array::uniform_linear(17, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(361);
        let matrix = SteeringMatrix::build(&array, &grid);
        let error =
            evaluate_training(&vec![Complex64::new(1.0, 0.0); 17], &matrix, &grid).unwrap_err();
        assert_eq!(
            error,
            ScenarioDimensionError {
                expected: 16,
                actual: 17
            }
        );
        assert!(HoldoutContext::new(array, grid).is_err());
    }

    #[test]
    fn degeneracy_and_kernel_failures_are_distinct_channels() {
        let (array, grid, mut excitation) = fixture();
        excitation.fill(Complex64::new(0.0, 0.0));
        let matrix = SteeringMatrix::build(&array, &grid);
        let robust = evaluate_training(&excitation, &matrix, &grid).unwrap();
        assert!(robust.degenerate_scenarios > 0);
        assert_eq!(robust.kernel_failures, 0);
        assert_eq!(robust.worst_psll_db, 0.0);
    }

    #[test]
    fn all_fixed_dual_failure_pairs_are_distinct() {
        let normalized = DUAL_FAILURE_PAIRS
            .iter()
            .map(|&(left, right)| (left.min(right), left.max(right)))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(normalized.len(), DUAL_FAILURE_PAIRS.len());
    }
}
