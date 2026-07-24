//! Numerical Buckingham–Pi analysis for the native optimization example.
//!
//! This module deliberately implements the small numerical core needed by the
//! example instead of porting BuckinghamPy's symbolic parser. A problem is
//! supplied as variable names plus an integer dimension matrix `A`. Valid
//! exponent vectors satisfy `A e = 0`.

use std::collections::HashSet;

use fcmaes_core::Rng;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

const LOG10: f64 = std::f64::consts::LN_10;
const MAX_LOG_PI: f64 = 80.0;
const INVALID_SCORE: f64 = 1.0e12;
const MATRIX_TOLERANCE: f64 = 1.0e-10;

/// One dimensional-analysis problem before preprocessing.
#[derive(Clone, Debug)]
pub struct BuckinghamProblem {
    pub slug: &'static str,
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub dimensions: &'static [&'static str],
    /// Row-major dimension matrix.
    pub matrix: &'static [&'static [i32]],
    pub dependent: &'static str,
}

/// A problem with the dependent variable and dimensionless input columns
/// removed, matching the continuous-exponent analysis in BuckinghamExamples.
#[derive(Clone, Debug)]
pub struct PreparedProblem {
    pub slug: &'static str,
    pub name: &'static str,
    pub variables: Vec<&'static str>,
    pub dimensions: &'static [&'static str],
    pub dependent: &'static str,
    matrix: DMatrix<f64>,
    removed_dimensionless: Vec<&'static str>,
}

impl PreparedProblem {
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    pub fn rank(&self) -> usize {
        matrix_rank(&self.matrix)
    }

    pub fn nullity(&self) -> usize {
        self.variables.len().saturating_sub(self.rank())
    }

    pub fn removed_dimensionless(&self) -> &[&'static str] {
        &self.removed_dimensionless
    }

    /// Orthonormal numerical basis of `ker(A)`, stored as basis vectors in
    /// columns.
    pub fn nullspace(&self) -> Result<DMatrix<f64>, String> {
        nullspace_basis(&self.matrix)
    }

    /// Enumerate every full-rank repeating-variable set.
    pub fn repeating_sets(&self) -> Vec<Vec<usize>> {
        find_repeating_sets(&self.matrix)
    }

    /// Construct the conventional π groups belonging to a repeating set.
    pub fn pi_groups(&self, repeating: &[usize]) -> Result<Vec<PiGroup>, String> {
        compute_pi_groups(&self.matrix, &self.variables, repeating)
    }

    /// All conventional groups from all repeating sets, with scalar multiples
    /// and reciprocals represented once.
    pub fn unique_pi_groups(&self) -> Result<Vec<PiGroup>, String> {
        let mut keys = HashSet::new();
        let mut unique = Vec::new();
        for repeating in self.repeating_sets() {
            for group in self.pi_groups(&repeating)? {
                let key = canonical_exponent_key(&group.exponents);
                if keys.insert(key) {
                    unique.push(group);
                }
            }
        }
        Ok(unique)
    }
}

/// One conventional π group. The exponent vector follows the prepared
/// problem's variable order and is normalized to exponent one on its
/// non-repeating variable.
#[derive(Clone, Debug)]
pub struct PiGroup {
    pub non_repeating: usize,
    pub exponents: Vec<f64>,
}

/// Deterministic synthetic train/holdout data for continuous π-group search.
#[derive(Clone, Debug)]
pub struct BuckinghamModel {
    problem: PreparedProblem,
    nullspace: DMatrix<f64>,
    groups: usize,
    train_log_x: DMatrix<f64>,
    train_y: DVector<f64>,
    validation_log_x: DMatrix<f64>,
    validation_y: DVector<f64>,
}

/// Diagnostics for a candidate set of continuous π groups.
#[derive(Clone, Debug)]
pub struct CandidateMetrics {
    pub scalar_objective: f64,
    pub train_r2: f64,
    pub validation_r2: f64,
    pub mean_coefficient_of_variation: f64,
    pub cv_violation: f64,
    pub complexity: f64,
    pub dependence: f64,
    pub condition_ratio: f64,
    pub dimensional_residual: f64,
    pub exponents: DMatrix<f64>,
    pub valid: bool,
}

impl CandidateMetrics {
    fn invalid(variable_count: usize, groups: usize) -> Self {
        Self {
            scalar_objective: INVALID_SCORE,
            train_r2: f64::NEG_INFINITY,
            validation_r2: f64::NEG_INFINITY,
            mean_coefficient_of_variation: 0.0,
            cv_violation: 1.0e6,
            complexity: 1.0e6,
            dependence: 1.0,
            condition_ratio: 0.0,
            dimensional_residual: 1.0e6,
            exponents: DMatrix::zeros(variable_count, groups),
            valid: false,
        }
    }

    /// MODE row: validation error, exponent complexity, feature dependence,
    /// followed by the coefficient-of-variation constraint (`<= 0`).
    pub fn mode_values(&self) -> Vec<f64> {
        if !self.valid {
            return vec![1.0e6, 1.0e6, 1.0, 1.0e6];
        }
        vec![
            1.0 - self.validation_r2,
            self.complexity,
            self.dependence,
            self.cv_violation,
        ]
    }
}

/// One enumerated repeating set ranked against the same holdout data used by
/// continuous optimization.
#[derive(Clone, Debug)]
pub struct RankedRepeatingSet {
    pub repeating: Vec<usize>,
    pub groups: Vec<PiGroup>,
    pub metrics: CandidateMetrics,
}

impl BuckinghamModel {
    pub fn new(
        problem: PreparedProblem,
        groups: usize,
        samples_per_split: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if samples_per_split < 8 {
            return Err("at least eight samples per split are required".to_owned());
        }
        let nullspace = problem.nullspace()?;
        let nullity = nullspace.ncols();
        if nullity == 0 {
            return Err("the preprocessed problem has no dimensionless groups".to_owned());
        }
        if groups == 0 || groups > nullity {
            return Err(format!(
                "group count must be between one and the nullity ({nullity})"
            ));
        }

        let mut weight_rng = Rng::new(seed ^ 0xD1B5_4A32_D192_ED03);
        let weights = DVector::from_iterator(nullity, (0..nullity).map(|_| weight_rng.gaussian()));
        let mut train_rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut validation_rng = Rng::new(seed ^ 0xA076_1D64_78BD_642F);
        let train_log_x =
            sample_log_inputs(samples_per_split, problem.variables.len(), &mut train_rng);
        let validation_log_x = sample_log_inputs(
            samples_per_split,
            problem.variables.len(),
            &mut validation_rng,
        );

        let train_signal = response(&train_log_x, &nullspace, &weights)?;
        let validation_signal = response(&validation_log_x, &nullspace, &weights)?;
        let signal_scale = population_stddev(train_signal.as_slice()).max(1.0e-12);
        let noise_scale = 0.02 * signal_scale;
        let train_y = add_noise(train_signal, noise_scale, &mut train_rng);
        let validation_y = add_noise(validation_signal, noise_scale, &mut validation_rng);

        Ok(Self {
            problem,
            nullspace,
            groups,
            train_log_x,
            train_y,
            validation_log_x,
            validation_y,
        })
    }

    pub fn problem(&self) -> &PreparedProblem {
        &self.problem
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn decision_dimension(&self) -> usize {
        self.nullspace.ncols() * self.groups
    }

    pub fn samples_per_split(&self) -> usize {
        self.train_log_x.nrows()
    }

    pub fn evaluate(&self, coefficients: &[f64]) -> CandidateMetrics {
        if coefficients.len() != self.decision_dimension()
            || coefficients.iter().any(|value| !value.is_finite())
        {
            return CandidateMetrics::invalid(self.problem.variables.len(), self.groups);
        }
        let coefficients =
            DMatrix::from_row_slice(self.nullspace.ncols(), self.groups, coefficients);
        let exponents = &self.nullspace * coefficients;
        self.evaluate_exponents(&exponents)
    }

    pub fn evaluate_exponents(&self, exponents: &DMatrix<f64>) -> CandidateMetrics {
        if exponents.nrows() != self.problem.variables.len()
            || exponents.ncols() == 0
            || exponents.iter().any(|value| !value.is_finite())
        {
            return CandidateMetrics::invalid(self.problem.variables.len(), self.groups);
        }

        let Some((train_pi, cvs)) = pi_features(&self.train_log_x, exponents) else {
            return CandidateMetrics::invalid(self.problem.variables.len(), exponents.ncols());
        };
        let Some((validation_pi, _)) = pi_features(&self.validation_log_x, exponents) else {
            return CandidateMetrics::invalid(self.problem.variables.len(), exponents.ncols());
        };
        let Some(regression) =
            fit_and_score(&train_pi, &self.train_y, &validation_pi, &self.validation_y)
        else {
            return CandidateMetrics::invalid(self.problem.variables.len(), exponents.ncols());
        };

        let cv_floor = 0.1;
        let cv_cap = 10.0;
        let cv_violation = cvs
            .iter()
            .map(|&cv| (cv_floor - cv).max(0.0) + (cv - cv_cap).max(0.0))
            .sum::<f64>();
        let conditioning_penalty = if regression.condition_ratio < 0.05 {
            10.0 * (0.05 - regression.condition_ratio) / 0.05
        } else {
            0.0
        };
        let scalar_objective =
            ((1.0 - regression.validation_r2) + 100.0 * cv_violation + conditioning_penalty)
                .clamp(-1.0e6, INVALID_SCORE);
        let residual = (&self.problem.matrix * exponents)
            .iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        let metrics = CandidateMetrics {
            scalar_objective,
            train_r2: regression.train_r2,
            validation_r2: regression.validation_r2,
            mean_coefficient_of_variation: cvs.iter().sum::<f64>() / cvs.len() as f64,
            cv_violation,
            complexity: exponents.iter().map(|value| value.abs()).sum(),
            dependence: 1.0 - regression.condition_ratio,
            condition_ratio: regression.condition_ratio,
            dimensional_residual: residual,
            exponents: exponents.clone(),
            valid: scalar_objective.is_finite()
                && regression.train_r2.is_finite()
                && regression.validation_r2.is_finite()
                && residual <= 1.0e-7,
        };
        if metrics.valid {
            metrics
        } else {
            CandidateMetrics::invalid(self.problem.variables.len(), exponents.ncols())
        }
    }

    /// Rank conventional repeating-variable bases using the deterministic
    /// train/holdout response. Complete bases have `nullity` groups.
    pub fn rank_repeating_sets(&self) -> Result<Vec<RankedRepeatingSet>, String> {
        let mut ranked = Vec::new();
        for repeating in self.problem.repeating_sets() {
            let groups = self.problem.pi_groups(&repeating)?;
            if groups.len() != self.problem.nullity() {
                continue;
            }
            let mut exponents = DMatrix::zeros(self.problem.variables.len(), groups.len());
            for (column, group) in groups.iter().enumerate() {
                for (row, &value) in group.exponents.iter().enumerate() {
                    exponents[(row, column)] = value;
                }
            }
            ranked.push(RankedRepeatingSet {
                repeating,
                groups,
                metrics: self.evaluate_exponents(&exponents),
            });
        }
        ranked.sort_by(|left, right| {
            right
                .metrics
                .validation_r2
                .total_cmp(&left.metrics.validation_r2)
        });
        Ok(ranked)
    }
}

#[derive(Clone, Copy, Debug)]
struct RegressionMetrics {
    train_r2: f64,
    validation_r2: f64,
    condition_ratio: f64,
}

fn sample_log_inputs(rows: usize, columns: usize, rng: &mut Rng) -> DMatrix<f64> {
    DMatrix::from_fn(rows, columns, |_, _| (-3.0 + 6.0 * rng.uniform01()) * LOG10)
}

fn response(
    log_x: &DMatrix<f64>,
    nullspace: &DMatrix<f64>,
    weights: &DVector<f64>,
) -> Result<DVector<f64>, String> {
    let logs = log_x * nullspace;
    let mut pi = DMatrix::zeros(logs.nrows(), logs.ncols());
    for row in 0..logs.nrows() {
        for column in 0..logs.ncols() {
            let value = logs[(row, column)];
            if value.abs() > MAX_LOG_PI {
                return Err("synthetic π feature exceeded the safe log range".to_owned());
            }
            pi[(row, column)] = value.exp();
        }
    }
    Ok(pi * weights)
}

fn add_noise(mut signal: DVector<f64>, scale: f64, rng: &mut Rng) -> DVector<f64> {
    for value in signal.iter_mut() {
        *value += scale * rng.gaussian();
    }
    signal
}

fn population_stddev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn pi_features(log_x: &DMatrix<f64>, exponents: &DMatrix<f64>) -> Option<(DMatrix<f64>, Vec<f64>)> {
    let logs = log_x * exponents;
    if logs
        .iter()
        .any(|value| !value.is_finite() || value.abs() > MAX_LOG_PI)
    {
        return None;
    }
    let pi = logs.map(f64::exp);
    let mut cvs = Vec::with_capacity(pi.ncols());
    for column in 0..pi.ncols() {
        let mean = (0..pi.nrows()).map(|row| pi[(row, column)]).sum::<f64>() / pi.nrows() as f64;
        let variance = (0..pi.nrows())
            .map(|row| {
                let delta = pi[(row, column)] - mean;
                delta * delta
            })
            .sum::<f64>()
            / pi.nrows() as f64;
        let cv = variance.sqrt() / mean.abs().max(1.0e-12);
        if !cv.is_finite() {
            return None;
        }
        cvs.push(cv);
    }
    Some((pi, cvs))
}

fn fit_and_score(
    train_x: &DMatrix<f64>,
    train_y: &DVector<f64>,
    validation_x: &DMatrix<f64>,
    validation_y: &DVector<f64>,
) -> Option<RegressionMetrics> {
    if train_x.nrows() != train_y.len()
        || validation_x.nrows() != validation_y.len()
        || train_x.ncols() == 0
        || train_x.ncols() != validation_x.ncols()
    {
        return None;
    }

    let rows = train_x.nrows();
    let columns = train_x.ncols();
    let mut means = vec![0.0; columns];
    let mut scales = vec![0.0; columns];
    let mut standardized = DMatrix::zeros(rows, columns);
    for column in 0..columns {
        means[column] = (0..rows).map(|row| train_x[(row, column)]).sum::<f64>() / rows as f64;
        scales[column] = ((0..rows)
            .map(|row| {
                let delta = train_x[(row, column)] - means[column];
                delta * delta
            })
            .sum::<f64>()
            / rows as f64)
            .sqrt();
        if !scales[column].is_finite() || scales[column] <= 1.0e-12 {
            return None;
        }
        for row in 0..rows {
            standardized[(row, column)] = (train_x[(row, column)] - means[column]) / scales[column];
        }
    }

    let y_mean = train_y.iter().sum::<f64>() / train_y.len() as f64;
    let centered_y = train_y.map(|value| value - y_mean);
    let singular_values = standardized.clone().svd(false, false).singular_values;
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let smallest = singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if largest <= 0.0 || !smallest.is_finite() {
        return None;
    }
    let condition_ratio = if columns == 1 {
        1.0
    } else {
        (smallest / largest).clamp(0.0, 1.0)
    };
    if condition_ratio <= 1.0e-10 {
        return None;
    }

    let coefficients = standardized
        .clone()
        .svd(true, true)
        .solve(&centered_y, 1.0e-10)
        .ok()?;
    let train_prediction = DVector::from_iterator(
        rows,
        (0..rows).map(|row| {
            y_mean
                + (0..columns)
                    .map(|column| standardized[(row, column)] * coefficients[column])
                    .sum::<f64>()
        }),
    );
    let validation_prediction = DVector::from_iterator(
        validation_x.nrows(),
        (0..validation_x.nrows()).map(|row| {
            y_mean
                + (0..columns)
                    .map(|column| {
                        ((validation_x[(row, column)] - means[column]) / scales[column])
                            * coefficients[column]
                    })
                    .sum::<f64>()
        }),
    );
    let train_r2 = r_squared(train_y, &train_prediction)?;
    let validation_r2 = r_squared(validation_y, &validation_prediction)?;
    Some(RegressionMetrics {
        train_r2,
        validation_r2,
        condition_ratio,
    })
}

fn r_squared(actual: &DVector<f64>, predicted: &DVector<f64>) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let total = actual
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    if !total.is_finite() || total <= 1.0e-30 {
        return None;
    }
    let residual = actual
        .iter()
        .zip(predicted)
        .map(|(&value, &estimate)| {
            let delta = value - estimate;
            delta * delta
        })
        .sum::<f64>();
    let r2 = 1.0 - residual / total;
    r2.is_finite().then_some(r2)
}

fn matrix_rank(matrix: &DMatrix<f64>) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    let singular_values = matrix.clone().svd(false, false).singular_values;
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = MATRIX_TOLERANCE * matrix.nrows().max(matrix.ncols()) as f64 * largest.max(1.0);
    singular_values
        .iter()
        .filter(|&&value| value > tolerance)
        .count()
}

fn nullspace_basis(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let rank = matrix_rank(matrix);
    let nullity = matrix.ncols().saturating_sub(rank);
    if nullity == 0 {
        return Ok(DMatrix::zeros(matrix.ncols(), 0));
    }
    let gram = matrix.transpose() * matrix;
    let eigen = SymmetricEigen::new(gram);
    let mut indices: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
    indices.sort_by(|&left, &right| {
        eigen.eigenvalues[left]
            .abs()
            .total_cmp(&eigen.eigenvalues[right].abs())
    });
    let mut basis = DMatrix::zeros(matrix.ncols(), nullity);
    for (basis_column, &eigen_column) in indices.iter().take(nullity).enumerate() {
        for row in 0..matrix.ncols() {
            basis[(row, basis_column)] = eigen.eigenvectors[(row, eigen_column)];
        }
    }
    let residual = (matrix * &basis)
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if residual > 1.0e-7 {
        return Err(format!(
            "failed to compute a stable nullspace (residual {residual:.3e})"
        ));
    }
    Ok(basis)
}

fn find_repeating_sets(matrix: &DMatrix<f64>) -> Vec<Vec<usize>> {
    let rank = matrix_rank(matrix);
    if rank == 0 {
        return vec![Vec::new()];
    }
    let mut combinations = Vec::new();
    combinations_of(matrix.ncols(), rank, 0, &mut Vec::new(), &mut combinations);
    combinations
        .into_iter()
        .filter(|indices| {
            let selected = DMatrix::from_fn(matrix.nrows(), indices.len(), |row, column| {
                matrix[(row, indices[column])]
            });
            matrix_rank(&selected) == rank
        })
        .collect()
}

fn combinations_of(
    count: usize,
    choose: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == choose {
        output.push(current.clone());
        return;
    }
    let remaining = choose - current.len();
    for index in start..=count - remaining {
        current.push(index);
        combinations_of(count, choose, index + 1, current, output);
        current.pop();
    }
}

fn compute_pi_groups(
    matrix: &DMatrix<f64>,
    variables: &[&'static str],
    repeating: &[usize],
) -> Result<Vec<PiGroup>, String> {
    let rank = matrix_rank(matrix);
    if repeating.len() != rank {
        return Err(format!("a repeating set must contain {rank} variables"));
    }
    if repeating.iter().any(|&index| index >= matrix.ncols()) {
        return Err("repeating-variable index is out of range".to_owned());
    }
    let unique: HashSet<usize> = repeating.iter().copied().collect();
    if unique.len() != repeating.len() {
        return Err("repeating-variable indices must be unique".to_owned());
    }
    let repeating_matrix = DMatrix::from_fn(matrix.nrows(), rank, |row, column| {
        matrix[(row, repeating[column])]
    });
    if matrix_rank(&repeating_matrix) != rank {
        return Err("repeating-variable columns are not full rank".to_owned());
    }

    let decomposition = repeating_matrix.svd(true, true);
    let mut groups = Vec::new();
    for non_repeating in 0..variables.len() {
        if unique.contains(&non_repeating) {
            continue;
        }
        let target = -matrix.column(non_repeating).into_owned();
        let solution = decomposition
            .solve(&target, MATRIX_TOLERANCE)
            .map_err(|_| "failed to solve a repeating-variable system")?;
        let mut exponents = vec![0.0; variables.len()];
        exponents[non_repeating] = 1.0;
        for (position, &variable) in repeating.iter().enumerate() {
            exponents[variable] = clean_exponent(solution[position]);
        }
        groups.push(PiGroup {
            non_repeating,
            exponents,
        });
    }
    Ok(groups)
}

fn clean_exponent(value: f64) -> f64 {
    let (numerator, denominator) = rational_approximation(value, 32);
    let rational = numerator as f64 / denominator as f64;
    if (rational - value).abs() <= 1.0e-9 {
        rational
    } else if value.abs() <= 1.0e-12 {
        0.0
    } else {
        value
    }
}

fn canonical_exponent_key(exponents: &[f64]) -> Vec<i64> {
    let Some(first) = exponents
        .iter()
        .copied()
        .find(|value| value.abs() > 1.0e-10)
    else {
        return vec![0; exponents.len()];
    };
    exponents
        .iter()
        .map(|value| (value / first * 1.0e8).round() as i64)
        .collect()
}

/// Compact rational form used only for stable human-readable output.
pub fn format_exponent(value: f64) -> String {
    let (numerator, denominator) = rational_approximation(value, 32);
    let rational = numerator as f64 / denominator as f64;
    if (rational - value).abs() <= 1.0e-7 {
        if denominator == 1 {
            numerator.to_string()
        } else {
            format!("{numerator}/{denominator}")
        }
    } else {
        format!("{value:.5}")
    }
}

fn rational_approximation(value: f64, max_denominator: i64) -> (i64, i64) {
    let mut best = (value.round() as i64, 1_i64);
    let mut best_error = (best.0 as f64 - value).abs();
    for denominator in 1..=max_denominator {
        let numerator = (value * denominator as f64).round() as i64;
        let error = (numerator as f64 / denominator as f64 - value).abs();
        if error < best_error {
            best = (numerator, denominator);
            best_error = error;
        }
    }
    let divisor = gcd(best.0.unsigned_abs(), best.1 as u64).max(1) as i64;
    (best.0 / divisor, best.1 / divisor)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Prepare a catalog problem by removing the dependent response and
/// dimensionless input columns. The removed zero columns are reported rather
/// than silently disappearing.
pub fn prepare_problem(problem: BuckinghamProblem) -> Result<PreparedProblem, String> {
    if problem.matrix.len() != problem.dimensions.len()
        || problem
            .matrix
            .iter()
            .any(|row| row.len() != problem.variables.len())
    {
        return Err("dimension-matrix shape does not match variables/dimensions".to_owned());
    }
    let dependent_index = problem
        .variables
        .iter()
        .position(|&variable| variable == problem.dependent)
        .ok_or_else(|| "dependent variable is missing from the problem".to_owned())?;

    let mut keep = Vec::new();
    let mut removed_dimensionless = Vec::new();
    for column in 0..problem.variables.len() {
        if column == dependent_index {
            continue;
        }
        if problem.matrix.iter().all(|row| row[column] == 0) {
            removed_dimensionless.push(problem.variables[column]);
        } else {
            keep.push(column);
        }
    }
    if keep.is_empty() {
        return Err("preprocessing removed every independent variable".to_owned());
    }
    let matrix = DMatrix::from_fn(problem.matrix.len(), keep.len(), |row, column| {
        problem.matrix[row][keep[column]] as f64
    });
    Ok(PreparedProblem {
        slug: problem.slug,
        name: problem.name,
        variables: keep
            .iter()
            .map(|&column| problem.variables[column])
            .collect(),
        dimensions: problem.dimensions,
        dependent: problem.dependent,
        matrix,
        removed_dimensionless,
    })
}

const MLT: &[&str] = &["M", "L", "T"];
const MLT_THETA: &[&str] = &["M", "L", "T", "Theta"];

const PIPE_VARIABLES: &[&str] = &["DeltaP", "R", "d", "mu", "Q"];
const PIPE_MATRIX: &[&[i32]] = &[&[1, 0, 0, 1, 0], &[-1, 1, 1, -1, 3], &[-2, 0, 0, -1, -1]];

const PUMP_VARIABLES: &[&str] = &["DeltaP", "R", "V", "Q", "E", "G"];
const PUMP_MATRIX: &[&[i32]] = &[
    &[1, 1, 1, 0, 0, 0],
    &[-3, 2, -1, 3, 1, 0],
    &[0, -3, -1, -1, 0, -1],
];

const FLOW_VARIABLES: &[&str] = &["DeltaF", "rho", "v", "D", "eta"];
const FLOW_MATRIX: &[&[i32]] = &[&[1, 1, 0, 0, 1], &[1, -3, 1, 1, -1], &[-2, 0, -1, 0, -1]];

const PACKED_BED_VARIABLES: &[&str] = &["DeltaP", "rho", "mu", "U", "D_p", "epsilon", "L"];
const PACKED_BED_MATRIX: &[&[i32]] = &[
    &[1, 1, 1, 0, 0, 0, 0],
    &[-1, -3, -1, 1, 1, 0, 1],
    &[-2, 0, -1, -1, 0, 0, 0],
];

const CYLINDER_VARIABLES: &[&str] = &["h", "D", "k", "U", "mu", "rho", "c_p"];
const CYLINDER_MATRIX: &[&[i32]] = &[
    &[1, 0, 1, 0, 1, 1, 0],
    &[-2, 1, 1, 1, -1, -3, 2],
    &[-3, 0, -3, -1, -1, 0, -2],
    &[-1, 0, -1, 0, 0, 0, -1],
];

const NATURAL_CONVECTION_VARIABLES: &[&str] = &[
    "Nu", "Gr", "Pr", "L", "beta", "DeltaT", "rho", "mu", "k", "g",
];
const NATURAL_CONVECTION_MATRIX: &[&[i32]] = &[
    &[0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    &[0, 0, 0, 1, 0, 0, -3, -1, 1, 1],
    &[0, 0, 0, 0, 0, 0, 0, -1, -3, -2],
    &[0, 0, 0, 0, -1, 1, 0, 0, -1, 0],
];

const RAYLEIGH_VARIABLES: &[&str] = &[
    "Nu", "Ra", "Pr", "H", "k", "rho", "c_p", "mu", "g", "beta", "DeltaT",
];
const RAYLEIGH_MATRIX: &[&[i32]] = &[
    &[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    &[0, 0, 0, 1, 1, -3, 2, -1, 1, 0, 0],
    &[0, 0, 0, 0, -3, 0, -2, -1, -2, 0, 0],
    &[0, 0, 0, 0, -1, 0, -1, 0, 0, -1, 1],
];

/// Built-in problems adapted from the MIT-licensed BuckinghamExamples
/// catalog. Add a user problem by constructing [`BuckinghamProblem`] directly.
pub fn catalog() -> Vec<BuckinghamProblem> {
    vec![
        BuckinghamProblem {
            slug: "pipe",
            name: "Pressure Drop in Pipe",
            variables: PIPE_VARIABLES,
            dimensions: MLT,
            matrix: PIPE_MATRIX,
            dependent: "DeltaP",
        },
        BuckinghamProblem {
            slug: "pump",
            name: "Centrifugal Pump",
            variables: PUMP_VARIABLES,
            dimensions: MLT,
            matrix: PUMP_MATRIX,
            dependent: "DeltaP",
        },
        BuckinghamProblem {
            slug: "flow",
            name: "Flow Around a Body",
            variables: FLOW_VARIABLES,
            dimensions: MLT,
            matrix: FLOW_MATRIX,
            dependent: "DeltaF",
        },
        BuckinghamProblem {
            slug: "packed-bed",
            name: "Packed-Bed Pressure Drop",
            variables: PACKED_BED_VARIABLES,
            dimensions: MLT,
            matrix: PACKED_BED_MATRIX,
            dependent: "DeltaP",
        },
        BuckinghamProblem {
            slug: "cylinder",
            name: "Laminar Forced Convection over a Cylinder",
            variables: CYLINDER_VARIABLES,
            dimensions: MLT_THETA,
            matrix: CYLINDER_MATRIX,
            dependent: "h",
        },
        BuckinghamProblem {
            slug: "natural-convection",
            name: "Natural Convection from a Horizontal Plate",
            variables: NATURAL_CONVECTION_VARIABLES,
            dimensions: MLT_THETA,
            matrix: NATURAL_CONVECTION_MATRIX,
            dependent: "Nu",
        },
        BuckinghamProblem {
            slug: "rayleigh-benard",
            name: "Rayleigh-Benard Convection",
            variables: RAYLEIGH_VARIABLES,
            dimensions: MLT_THETA,
            matrix: RAYLEIGH_MATRIX,
            dependent: "Nu",
        },
    ]
}

pub fn problem_by_slug(slug: &str) -> Option<BuckinghamProblem> {
    catalog().into_iter().find(|problem| problem.slug == slug)
}

/// Format one exponent vector as a product for CLI output.
pub fn format_pi_group(variables: &[&str], exponents: &[f64]) -> String {
    variables
        .iter()
        .zip(exponents)
        .filter(|(_, exponent)| exponent.abs() > 1.0e-9)
        .map(|(variable, &exponent)| {
            if (exponent - 1.0).abs() <= 1.0e-9 {
                (*variable).to_owned()
            } else {
                format!("{variable}^{}", format_exponent(exponent))
            }
        })
        .collect::<Vec<_>>()
        .join(" * ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_nullspace_is_dimensionless() {
        let prepared = prepare_problem(problem_by_slug("pipe").unwrap()).unwrap();
        assert_eq!(prepared.rank(), 3);
        assert_eq!(prepared.nullity(), 1);
        let basis = prepared.nullspace().unwrap();
        let residual = prepared.matrix() * basis;
        assert!(residual.iter().all(|value| value.abs() < 1.0e-8));
    }

    #[test]
    fn conventional_pipe_groups_are_valid_and_canonical() {
        let prepared = prepare_problem(problem_by_slug("pipe").unwrap()).unwrap();
        let sets = prepared.repeating_sets();
        assert_eq!(sets.len(), 2);
        for repeating in sets {
            let groups = prepared.pi_groups(&repeating).unwrap();
            assert_eq!(groups.len(), prepared.nullity());
            for group in groups {
                let exponent = DVector::from_vec(group.exponents);
                assert!(
                    (prepared.matrix() * exponent)
                        .iter()
                        .all(|value| value.abs() < 1.0e-8)
                );
            }
        }
        let unique = prepared.unique_pi_groups().unwrap();
        assert_eq!(unique.len(), 1);
        assert_eq!(
            format_pi_group(&prepared.variables, &unique[0].exponents),
            "R^-1 * d"
        );
    }

    #[test]
    fn preprocessing_reports_removed_dimensionless_inputs() {
        let prepared = prepare_problem(problem_by_slug("natural-convection").unwrap()).unwrap();
        assert_eq!(prepared.removed_dimensionless(), &["Gr", "Pr"]);
        assert!(!prepared.variables.contains(&"Nu"));
    }

    #[test]
    fn complete_catalog_has_valid_dimensionless_bases() {
        for raw in catalog() {
            let prepared = prepare_problem(raw).unwrap();
            assert!(prepared.rank() > 0, "{}", prepared.slug);
            assert!(prepared.nullity() > 0, "{}", prepared.slug);
            assert!(!prepared.repeating_sets().is_empty(), "{}", prepared.slug);
            let unique = prepared.unique_pi_groups().unwrap();
            assert!(!unique.is_empty(), "{}", prepared.slug);
            for group in unique {
                let exponent = DVector::from_vec(group.exponents);
                assert!(
                    (prepared.matrix() * exponent)
                        .iter()
                        .all(|value| value.abs() < 1.0e-8),
                    "{}",
                    prepared.slug
                );
            }
        }
    }

    #[test]
    fn deterministic_data_and_candidate_scores_reproduce() {
        let problem = prepare_problem(problem_by_slug("cylinder").unwrap()).unwrap();
        let first = BuckinghamModel::new(problem.clone(), 2, 64, 42).unwrap();
        let second = BuckinghamModel::new(problem, 2, 64, 42).unwrap();
        let coefficients = vec![1.0, 0.0, 0.0, 1.0];
        let a = first.evaluate(&coefficients);
        let b = second.evaluate(&coefficients);
        assert!(a.valid);
        assert_eq!(a.scalar_objective, b.scalar_objective);
        assert_eq!(a.validation_r2, b.validation_r2);
        assert!(a.dimensional_residual < 1.0e-8);
    }

    #[test]
    fn malformed_and_extreme_candidates_are_rejected() {
        let problem = prepare_problem(problem_by_slug("cylinder").unwrap()).unwrap();
        let model = BuckinghamModel::new(problem, 2, 32, 7).unwrap();
        assert!(!model.evaluate(&[0.0]).valid);
        assert!(!model.evaluate(&vec![9.0; model.decision_dimension()]).valid);
        let mut non_finite = vec![0.0; model.decision_dimension()];
        non_finite[0] = f64::NAN;
        assert!(!model.evaluate(&non_finite).valid);
    }

    #[test]
    fn rank_uses_holdout_metrics_for_complete_bases() {
        let problem = prepare_problem(problem_by_slug("packed-bed").unwrap()).unwrap();
        let nullity = problem.nullity();
        let model = BuckinghamModel::new(problem, nullity, 64, 11).unwrap();
        let ranking = model.rank_repeating_sets().unwrap();
        assert!(!ranking.is_empty());
        assert!(
            ranking
                .iter()
                .all(|entry| entry.groups.len() == nullity && entry.metrics.valid)
        );
        assert!(
            ranking
                .windows(2)
                .all(|pair| { pair[0].metrics.validation_r2 >= pair[1].metrics.validation_r2 })
        );
    }

    #[test]
    fn formatter_prefers_small_rational_exponents() {
        assert_eq!(format_exponent(-0.5), "-1/2");
        assert_eq!(
            format_pi_group(&["a", "b", "c"], &[1.0, -0.5, 0.0]),
            "a * b^-1/2"
        );
    }

    #[test]
    fn bad_problem_shapes_and_repeating_sets_are_rejected() {
        let malformed = BuckinghamProblem {
            slug: "bad",
            name: "bad",
            variables: &["x", "y"],
            dimensions: &["L"],
            matrix: &[&[1]],
            dependent: "x",
        };
        assert!(prepare_problem(malformed).is_err());
        let prepared = prepare_problem(problem_by_slug("pipe").unwrap()).unwrap();
        assert!(prepared.pi_groups(&[]).is_err());
        assert!(prepared.pi_groups(&[0, 0, 1]).is_err());
    }
}
