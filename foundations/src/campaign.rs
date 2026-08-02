//! Reproducible classic/ZDT/DTLZ comparison campaign.

use std::path::Path;
use std::time::Instant;

use fcmaes_core::{
    De, DeParams, Fitness, HypervolumeEstimate, Mode, ModeParams, ReferencePoint, Rng,
    additive_epsilon, gd, gd_plus, hypervolume, igd, igd_plus, pareto_indices, spacing, spread,
};
use serde::Serialize;
use serde_json::json;

use crate::artifacts::{write_json, write_text};
use crate::lessons;
use crate::suites::classic::Classic;
use crate::suites::dtlz::Dtlz;
use crate::suites::zdt::Zdt;
use crate::suites::{Suite, bbob, wfg};

/// Campaign-size preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    /// CI-sized conformance run.
    Smoke,
    /// Checked-in publication evidence.
    Publication,
}

impl Preset {
    /// Parse `smoke` or `publication`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "publication" => Some(Self::Publication),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Publication => "publication",
        }
    }

    fn so_budget(self) -> u64 {
        match self {
            Self::Smoke => 400,
            Self::Publication => 4_000,
        }
    }

    fn mo_budget(self) -> usize {
        match self {
            Self::Smoke => 512,
            Self::Publication => 4_096,
        }
    }

    fn reference_points(self) -> usize {
        match self {
            Self::Smoke => 257,
            Self::Publication => 2_001,
        }
    }
}

#[derive(Serialize)]
struct SoRow {
    problem: String,
    arm: String,
    dimension: usize,
    requested_evaluations: u64,
    actual_evaluations: u64,
    best: f64,
    wall_seconds: f64,
}

#[derive(Clone, Serialize)]
struct MoRow {
    problem: String,
    arm: String,
    dimension: usize,
    objectives: usize,
    requested_evaluations: usize,
    actual_evaluations: usize,
    front_size: usize,
    deterministic_recheck_points: usize,
    deterministic_recheck_max_abs_error: f64,
    hypervolume: f64,
    hypervolume_kind: String,
    fixed_hypervolume: Option<f64>,
    fixed_hypervolume_kind: String,
    igd: f64,
    igd_plus: f64,
    gd: f64,
    gd_plus: f64,
    additive_epsilon: f64,
    spacing: f64,
    spread: f64,
    duplicates_collapsed: usize,
    dominated_removed: usize,
    fixed_outside_reference: usize,
    normalization_ideal: Vec<f64>,
    normalization_nadir: Vec<f64>,
    reference_point: Vec<f64>,
    fixed_reference_point: Vec<f64>,
    wall_seconds: f64,
}

struct FrontRow {
    problem: String,
    arm: String,
    point_id: usize,
    decision: Vec<f64>,
    objectives: Vec<f64>,
}

struct ReplayedFront {
    decisions: Vec<Vec<f64>>,
    objectives: Vec<Vec<f64>>,
}

struct ConvergenceRow {
    problem: String,
    evaluations: usize,
    hypervolume: f64,
    fixed_hypervolume: Option<f64>,
    igd_plus: f64,
    fixed_outside_reference: usize,
}

struct MetricSnapshot {
    arm: &'static str,
    requested_evaluations: usize,
    actual_evaluations: usize,
    decisions: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    wall_seconds: f64,
}

type MultiResults = (Vec<MoRow>, Vec<FrontRow>, Vec<ConvergenceRow>);

fn csv_value(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn write_so_csv(path: &Path, rows: &[SoRow]) -> Result<(), String> {
    let mut output = String::from(
        "problem,arm,dimension,requested_evaluations,actual_evaluations,best,wall_seconds\n",
    );
    for row in rows {
        output.push_str(&format!(
            "{},{},{},{},{},{:.17e},{:.9}\n",
            row.problem,
            row.arm,
            row.dimension,
            row.requested_evaluations,
            row.actual_evaluations,
            row.best,
            row.wall_seconds
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

fn write_mo_csv(path: &Path, rows: &[MoRow]) -> Result<(), String> {
    let mut output = String::from(
        "problem,arm,dimension,objectives,requested_evaluations,actual_evaluations,front_size,deterministic_recheck_points,deterministic_recheck_max_abs_error,hypervolume,hypervolume_kind,fixed_hypervolume,fixed_hypervolume_kind,igd,igd_plus,gd,gd_plus,additive_epsilon,spacing,spread,duplicates_collapsed,dominated_removed,fixed_outside_reference,normalization_ideal,normalization_nadir,reference_point,fixed_reference_point,wall_seconds\n",
    );
    for row in rows {
        let reference = serde_json::to_string(&row.reference_point).map_err(|e| e.to_string())?;
        let fixed_reference =
            serde_json::to_string(&row.fixed_reference_point).map_err(|e| e.to_string())?;
        let ideal = serde_json::to_string(&row.normalization_ideal).map_err(|e| e.to_string())?;
        let nadir = serde_json::to_string(&row.normalization_nadir).map_err(|e| e.to_string())?;
        let fields = [
            row.problem.clone(),
            row.arm.clone(),
            row.dimension.to_string(),
            row.objectives.to_string(),
            row.requested_evaluations.to_string(),
            row.actual_evaluations.to_string(),
            row.front_size.to_string(),
            row.deterministic_recheck_points.to_string(),
            format!("{:.17e}", row.deterministic_recheck_max_abs_error),
            format!("{:.17e}", row.hypervolume),
            row.hypervolume_kind.clone(),
            row.fixed_hypervolume
                .map(|value| format!("{value:.17e}"))
                .unwrap_or_default(),
            row.fixed_hypervolume_kind.clone(),
            format!("{:.17e}", row.igd),
            format!("{:.17e}", row.igd_plus),
            format!("{:.17e}", row.gd),
            format!("{:.17e}", row.gd_plus),
            format!("{:.17e}", row.additive_epsilon),
            format!("{:.17e}", row.spacing),
            format!("{:.17e}", row.spread),
            row.duplicates_collapsed.to_string(),
            row.dominated_removed.to_string(),
            row.fixed_outside_reference.to_string(),
            csv_value(&ideal),
            csv_value(&nadir),
            csv_value(&reference),
            csv_value(&fixed_reference),
            format!("{:.9}", row.wall_seconds),
        ];
        output.push_str(&fields.join(","));
        output.push('\n');
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

fn write_front_csv(path: &Path, rows: &[FrontRow]) -> Result<(), String> {
    let mut output = String::from("problem,arm,point_id,decision,normalized_objectives\n");
    for row in rows {
        let decision = serde_json::to_string(&row.decision).map_err(|e| e.to_string())?;
        let objectives = serde_json::to_string(&row.objectives).map_err(|e| e.to_string())?;
        output.push_str(&format!(
            "{},{},{},{},{}\n",
            row.problem,
            row.arm,
            row.point_id,
            csv_value(&decision),
            csv_value(&objectives)
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

fn write_convergence_csv(path: &Path, rows: &[ConvergenceRow]) -> Result<(), String> {
    let mut output = String::from(
        "problem,evaluations,hypervolume,fixed_hypervolume,igd_plus,fixed_outside_reference\n",
    );
    for row in rows {
        let fixed = row
            .fixed_hypervolume
            .map(|value| format!("{value:.17e}"))
            .unwrap_or_default();
        output.push_str(&format!(
            "{},{},{:.17e},{},{:.17e},{}\n",
            row.problem,
            row.evaluations,
            row.hypervolume,
            fixed,
            row.igd_plus,
            row.fixed_outside_reference
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

fn convergence_row(row: &MoRow) -> ConvergenceRow {
    ConvergenceRow {
        problem: row.problem.clone(),
        evaluations: row.actual_evaluations,
        hypervolume: row.hypervolume,
        fixed_hypervolume: row.fixed_hypervolume,
        igd_plus: row.igd_plus,
        fixed_outside_reference: row.fixed_outside_reference,
    }
}

fn problem_seed(root: u64, name: &str, arm: u64) -> u64 {
    name.bytes().fold(root ^ arm, |state, byte| {
        state.rotate_left(7) ^ u64::from(byte).wrapping_mul(0x9e37_79b9)
    })
}

fn random_decision(rng: &mut Rng, lower: &[f64], upper: &[f64]) -> Vec<f64> {
    lower
        .iter()
        .zip(upper)
        .map(|(&lo, &hi)| lo + (hi - lo) * rng.uniform01())
        .collect()
}

fn run_single_objective(preset: Preset, seed: u64) -> Result<Vec<SoRow>, String> {
    let mut rows = Vec::new();
    let budget = preset.so_budget();
    for problem in Classic::all(10) {
        let (lower, upper) = problem.bounds();
        let initial_population = 31;
        let mut initial_rng = Rng::new(problem_seed(seed, problem.name(), 0x5241_4e44));
        let initial_best = (0..initial_population)
            .map(|_| {
                let decision = random_decision(&mut initial_rng, &lower, &upper);
                problem.evaluate(&decision).map(|value| value[0])
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        rows.push(SoRow {
            problem: problem.name().to_owned(),
            arm: "initial-population".to_owned(),
            dimension: problem.dimension(),
            requested_evaluations: initial_population,
            actual_evaluations: initial_population,
            best: initial_best,
            wall_seconds: 0.0,
        });

        let started = Instant::now();
        let mut rng = Rng::new(problem_seed(seed, problem.name(), 0x5241_4e44));
        let mut random_best = f64::INFINITY;
        for _ in 0..budget {
            let decision = random_decision(&mut rng, &lower, &upper);
            random_best =
                random_best.min(problem.evaluate(&decision).map_err(|e| e.to_string())?[0]);
        }
        rows.push(SoRow {
            problem: problem.name().to_owned(),
            arm: "random".to_owned(),
            dimension: problem.dimension(),
            requested_evaluations: budget,
            actual_evaluations: budget,
            best: random_best,
            wall_seconds: started.elapsed().as_secs_f64(),
        });

        let started = Instant::now();
        let fitness = Fitness::bounded(problem.dimension(), 1, &lower, &upper);
        let parameters = DeParams {
            max_evaluations: budget,
            seed: problem_seed(seed, problem.name(), 0x4445),
            ..Default::default()
        };
        let objective = |decision: &[f64]| problem.evaluate(decision).map_or(1.0e99, |v| v[0]);
        let result = De::new(fitness, &[], &[], None, &parameters).optimize(&objective);
        rows.push(SoRow {
            problem: problem.name().to_owned(),
            arm: "de".to_owned(),
            dimension: problem.dimension(),
            requested_evaluations: budget,
            actual_evaluations: result.evaluations,
            best: result.y,
            wall_seconds: started.elapsed().as_secs_f64(),
        });
    }
    Ok(rows)
}

fn normalization(reference: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let dimension = reference[0].len();
    let ideal = (0..dimension)
        .map(|axis| {
            reference
                .iter()
                .map(|point| point[axis])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let nadir = (0..dimension)
        .map(|axis| {
            reference
                .iter()
                .map(|point| point[axis])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect::<Vec<_>>();
    (ideal, nadir)
}

fn normalize(points: &[Vec<f64>], ideal: &[f64], nadir: &[f64]) -> Vec<Vec<f64>> {
    points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(ideal)
                .zip(nadir)
                .map(|((&value, &lo), &hi)| (value - lo) / (hi - lo).max(1.0e-15))
                .collect()
        })
        .collect()
}

fn estimate_kind(estimate: &HypervolumeEstimate) -> &'static str {
    match estimate {
        HypervolumeEstimate::Exact(_) => "exact",
        HypervolumeEstimate::MonteCarlo { .. } => "monte-carlo",
    }
}

fn campaign_reference(
    problem: &dyn Suite,
    reference_native: &[Vec<f64>],
    snapshots: &[&MetricSnapshot],
) -> Result<ReferencePoint, String> {
    let (ideal, nadir) = normalization(reference_native);
    let mut union = Vec::new();
    for snapshot in snapshots {
        let indices =
            pareto_indices(&snapshot.values, problem.objectives()).map_err(str::to_owned)?;
        let raw: Vec<Vec<f64>> = indices
            .into_iter()
            .map(|index| snapshot.values[index].clone())
            .collect();
        union.extend(normalize(&raw, &ideal, &nadir));
    }
    if union.is_empty() {
        return Err(format!("{} campaign has no front points", problem.name()));
    }
    let mut coordinates = Vec::with_capacity(problem.objectives());
    for axis in 0..problem.objectives() {
        let minimum = union
            .iter()
            .map(|point| point[axis])
            .fold(f64::INFINITY, f64::min);
        let maximum = union
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        coordinates.push(maximum + 0.1 * (maximum - minimum).max(1.0));
    }
    ReferencePoint::new(coordinates).map_err(|error| error.to_string())
}

fn metric_row(
    problem: &dyn Suite,
    snapshot: &MetricSnapshot,
    reference_native: &[Vec<f64>],
    reference_point: &ReferencePoint,
) -> Result<(MoRow, ReplayedFront), String> {
    if snapshot.decisions.len() != snapshot.values.len() {
        return Err(format!(
            "{} {} has {} decisions but {} objective vectors",
            problem.name(),
            snapshot.arm,
            snapshot.decisions.len(),
            snapshot.values.len()
        ));
    }
    let retained = pareto_indices(&snapshot.values, problem.objectives()).map_err(str::to_owned)?;
    let mut deterministic_recheck_max_abs_error = 0.0_f64;
    for &index in &retained {
        let replayed = problem
            .evaluate(&snapshot.decisions[index])
            .map_err(|error| error.to_string())?;
        if replayed.len() != snapshot.values[index].len() {
            return Err(format!(
                "{} deterministic recheck changed objective count",
                problem.name()
            ));
        }
        deterministic_recheck_max_abs_error = deterministic_recheck_max_abs_error.max(
            replayed
                .iter()
                .zip(&snapshot.values[index])
                .map(|(&left, &right)| (left - right).abs())
                .fold(0.0, f64::max),
        );
    }
    if deterministic_recheck_max_abs_error != 0.0 {
        return Err(format!(
            "{} {} deterministic recheck changed a retained objective by {deterministic_recheck_max_abs_error:e}",
            problem.name(),
            snapshot.arm
        ));
    }
    let raw_front: Vec<Vec<f64>> = retained
        .iter()
        .map(|&index| snapshot.values[index].clone())
        .collect();
    let retained_decisions: Vec<Vec<f64>> = retained
        .iter()
        .map(|&index| snapshot.decisions[index].clone())
        .collect();
    let (ideal, nadir) = normalization(reference_native);
    let front = normalize(&raw_front, &ideal, &nadir);
    let reference_set = normalize(reference_native, &ideal, &nadir);
    let report = hypervolume(&front, reference_point).map_err(|error| error.to_string())?;
    let hypervolume_kind = estimate_kind(&report.estimate).to_owned();
    let hypervolume_value = report.estimate.value();

    let fixed_reference_point =
        ReferencePoint::new(vec![1.1; problem.objectives()]).map_err(|error| error.to_string())?;
    let fixed_outside_reference = front
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(fixed_reference_point.as_slice())
                .any(|(value, limit)| value > limit)
        })
        .count();
    let (fixed_hypervolume, fixed_hypervolume_kind) = if fixed_outside_reference == 0 {
        let fixed_report =
            hypervolume(&front, &fixed_reference_point).map_err(|error| error.to_string())?;
        (
            Some(fixed_report.estimate.value()),
            estimate_kind(&fixed_report.estimate).to_owned(),
        )
    } else {
        (None, "not-applicable-outside-reference".to_owned())
    };
    let extremes: Vec<Vec<f64>> = (0..problem.objectives())
        .map(|axis| {
            reference_set
                .iter()
                .min_by(|left, right| left[axis].total_cmp(&right[axis]))
                .expect("reference set is non-empty")
                .clone()
        })
        .collect();
    Ok((
        MoRow {
            problem: problem.name().to_owned(),
            arm: snapshot.arm.to_owned(),
            dimension: problem.dimension(),
            objectives: problem.objectives(),
            requested_evaluations: snapshot.requested_evaluations,
            actual_evaluations: snapshot.actual_evaluations,
            front_size: front.len(),
            deterministic_recheck_points: retained.len(),
            deterministic_recheck_max_abs_error,
            hypervolume: hypervolume_value,
            hypervolume_kind,
            fixed_hypervolume,
            fixed_hypervolume_kind,
            igd: igd(&front, &reference_set).map_err(|e| e.to_string())?,
            igd_plus: igd_plus(&front, &reference_set).map_err(|e| e.to_string())?,
            gd: gd(&front, &reference_set).map_err(|e| e.to_string())?,
            gd_plus: gd_plus(&front, &reference_set).map_err(|e| e.to_string())?,
            additive_epsilon: additive_epsilon(&front, &reference_set)
                .map_err(|e| e.to_string())?,
            spacing: spacing(&front).map_err(|e| e.to_string())?,
            spread: if front.len() > 1 {
                spread(&front, &extremes).map_err(|e| e.to_string())?
            } else {
                0.0
            },
            duplicates_collapsed: report.duplicates_collapsed,
            dominated_removed: report.dominated_removed,
            fixed_outside_reference,
            normalization_ideal: ideal,
            normalization_nadir: nadir,
            reference_point: reference_point.as_slice().to_vec(),
            fixed_reference_point: fixed_reference_point.as_slice().to_vec(),
            wall_seconds: snapshot.wall_seconds,
        },
        ReplayedFront {
            decisions: retained_decisions,
            objectives: front,
        },
    ))
}

fn front_rows(problem: &dyn Suite, arm: &str, front: ReplayedFront) -> Vec<FrontRow> {
    front
        .decisions
        .into_iter()
        .zip(front.objectives)
        .enumerate()
        .map(|(point_id, (decision, objectives))| FrontRow {
            problem: problem.name().to_owned(),
            arm: arm.to_owned(),
            point_id,
            decision,
            objectives,
        })
        .collect()
}

fn run_multi_problem(
    problem: &dyn Suite,
    preset: Preset,
    seed: u64,
) -> Result<MultiResults, String> {
    let budget = preset.mo_budget();
    let reference = problem
        .reference_front(preset.reference_points())
        .ok_or_else(|| format!("{} has no reference front", problem.name()))?;
    let (lower, upper) = problem.bounds();
    let population = 64;
    let fitness = Fitness::bounded(problem.dimension(), problem.objectives(), &lower, &upper);
    let mut mode = Mode::try_new(
        fitness,
        problem.objectives(),
        0,
        None,
        &ModeParams {
            popsize: population as i32,
            seed: problem_seed(seed, problem.name(), 0x4d4f_4445),
            ..Default::default()
        },
    )
    .map_err(str::to_owned)?;
    let started = Instant::now();
    let first = mode.ask();
    let first_values = first
        .iter()
        .map(|decision| problem.evaluate(decision).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    mode.try_tell(&first_values).map_err(str::to_owned)?;
    let initial_snapshot = MetricSnapshot {
        arm: "initial",
        requested_evaluations: population,
        actual_evaluations: population,
        decisions: first,
        values: first_values,
        wall_seconds: 0.0,
    };
    let mut evaluations = population;
    let checkpoints = [budget / 4, budget / 2, budget];
    let mut checkpoint = 0;
    let mut checkpoint_snapshots = Vec::new();
    while evaluations < budget {
        let decisions = mode.ask();
        let values = decisions
            .iter()
            .map(|decision| problem.evaluate(decision).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        evaluations += values.len();
        mode.try_tell(&values).map_err(str::to_owned)?;
        while checkpoint < checkpoints.len() && evaluations >= checkpoints[checkpoint] {
            let checkpoint_decisions = mode.population();
            let checkpoint_values = checkpoint_decisions
                .iter()
                .map(|decision| problem.evaluate(decision).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            checkpoint_snapshots.push(MetricSnapshot {
                arm: "mode",
                requested_evaluations: checkpoints[checkpoint],
                actual_evaluations: evaluations,
                decisions: checkpoint_decisions,
                values: checkpoint_values,
                wall_seconds: started.elapsed().as_secs_f64(),
            });
            checkpoint += 1;
        }
    }
    let mode_decisions = mode.population();
    let mode_values = mode_decisions
        .iter()
        .map(|decision| problem.evaluate(decision).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let mode_snapshot = MetricSnapshot {
        arm: "mode",
        requested_evaluations: budget,
        actual_evaluations: evaluations,
        decisions: mode_decisions,
        values: mode_values,
        wall_seconds: started.elapsed().as_secs_f64(),
    };

    let started = Instant::now();
    let mut rng = Rng::new(problem_seed(seed, problem.name(), 0x5241_4e44));
    let random_decisions: Vec<Vec<f64>> = (0..budget)
        .map(|_| random_decision(&mut rng, &lower, &upper))
        .collect();
    let random_values = random_decisions
        .iter()
        .map(|decision| problem.evaluate(decision).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let random_snapshot = MetricSnapshot {
        arm: "random",
        requested_evaluations: budget,
        actual_evaluations: budget,
        decisions: random_decisions,
        values: random_values,
        wall_seconds: started.elapsed().as_secs_f64(),
    };

    let reference_snapshots: Vec<&MetricSnapshot> = std::iter::once(&initial_snapshot)
        .chain(checkpoint_snapshots.iter())
        .chain([&mode_snapshot, &random_snapshot])
        .collect();
    let shared_reference = campaign_reference(problem, &reference, &reference_snapshots)?;

    let (initial_row, initial_front) =
        metric_row(problem, &initial_snapshot, &reference, &shared_reference)?;
    let mut convergence = vec![convergence_row(&initial_row)];
    for snapshot in &checkpoint_snapshots {
        let (row, _) = metric_row(problem, snapshot, &reference, &shared_reference)?;
        convergence.push(convergence_row(&row));
    }
    let (mode_row, mode_front) =
        metric_row(problem, &mode_snapshot, &reference, &shared_reference)?;
    let (random_row, random_front) =
        metric_row(problem, &random_snapshot, &reference, &shared_reference)?;
    let rows = vec![initial_row, mode_row, random_row];
    let mut fronts = front_rows(problem, "initial", initial_front);
    fronts.extend(front_rows(problem, "mode", mode_front));
    fronts.extend(front_rows(problem, "random", random_front));
    Ok((rows, fronts, convergence))
}

fn run_multi_objective(preset: Preset, seed: u64) -> Result<MultiResults, String> {
    let mut problems: Vec<Box<dyn Suite>> = Zdt::publication()
        .into_iter()
        .map(|problem| Box::new(problem) as Box<dyn Suite>)
        .collect();
    problems.extend(
        Dtlz::all(3)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|problem| Box::new(problem) as Box<dyn Suite>),
    );
    let mut rows = Vec::new();
    let mut fronts = Vec::new();
    let mut convergence = Vec::new();
    for problem in &problems {
        let (problem_rows, problem_fronts, problem_convergence) =
            run_multi_problem(problem.as_ref(), preset, seed)?;
        rows.extend(problem_rows);
        fronts.extend(problem_fronts);
        convergence.extend(problem_convergence);
    }
    Ok((rows, fronts, convergence))
}

/// Execute all ungated publication stages and write schema-v2 conformance evidence.
///
/// # Errors
///
/// Returns a message when evaluation, indicators, or artifact I/O fails.
pub fn run(preset: Preset, seed: u64, workers: i32, output: &Path) -> Result<(), String> {
    let started = Instant::now();
    let so = run_single_objective(preset, seed)?;
    write_so_csv(&output.join("so/arms.csv"), &so)?;
    write_json(
        &output.join("so/run.json"),
        &json!({
            "schema_version": 2,
            "tutorial": "foundations",
            "formulation": "classic-single-objective",
            "status": "completed",
            "preset": preset.label(),
            "seed": seed,
            "seed_count": 1,
            "workers": workers,
            "claim_scope": "deterministic-conformance-demonstration",
            "problems": 8,
            "dimensions": [10],
            "arms": ["initial-population", "random", "de"],
            "baseline_relation": "initial-population is the first 31 points of the random arm stream",
            "requested_evaluations_per_optimized_arm": preset.so_budget(),
            "artifacts": {"arms": "arms.csv"}
        }),
    )
    .map_err(|error| error.to_string())?;

    let (mo, fronts, convergence) = run_multi_objective(preset, seed)?;
    write_mo_csv(&output.join("mo/indicators.csv"), &mo)?;
    write_front_csv(&output.join("mo/fronts.csv"), &fronts)?;
    write_convergence_csv(&output.join("mo/convergence.csv"), &convergence)?;
    write_json(
        &output.join("mo/run.json"),
        &json!({
            "schema_version": 2,
            "tutorial": "foundations",
            "formulation": "analytic-front-multi-objective",
            "status": "completed",
            "preset": preset.label(),
            "seed": seed,
            "seed_count": 1,
            "workers": workers,
            "claim_scope": "deterministic-conformance-demonstration",
            "problems": 12,
            "arms": ["initial", "random", "mode"],
            "mode": {
                "population": 64,
                "evaluation_workers": 1,
                "population_update": "nsga-ii-style",
                "nsga_update": true,
                "de_update_evaluated": false
            },
            "requested_evaluations_per_control_or_optimizer": preset.mo_budget(),
            "reference_points": preset.reference_points(),
            "normalization": "analytic-reference-set ideal/nadir",
            "hypervolume_reference_point": "shared per-problem union-front nadir plus 10% of max(observed range, 1) after analytic-front normalization",
            "fixed_hypervolume_reference_point": "[1.1; objectives] after analytic-front normalization",
            "fixed_hypervolume_policy": "null when any arm-front point lies outside; points are never filtered",
            "deterministic_recheck": "same-evaluator repeatability/bookkeeping check, not independent model validation",
            "artifacts": {
                "indicators": "indicators.csv",
                "fronts": "fronts.csv",
                "convergence": "convergence.csv"
            }
        }),
    )
    .map_err(|error| error.to_string())?;

    let ladder = lessons::run("all", workers)?;
    write_text(&output.join("ladder/output.txt"), &ladder).map_err(|error| error.to_string())?;
    write_json(
        &output.join("ladder/run.json"),
        &json!({
            "schema_version": 2,
            "tutorial": "foundations",
            "formulation": "seven-lesson-ladder",
            "status": "completed",
            "preset": preset.label(),
            "seed": seed,
            "workers": workers,
            "lessons": 7,
            "artifacts": {"output": "output.txt"}
        }),
    )
    .map_err(|error| error.to_string())?;

    let command = format!(
        "cargo run --release --locked -- --campaign --preset {} --workers {workers} --seed {seed} --output <DIR>",
        preset.label()
    );
    for (suite, reason) in [("wfg", wfg::SKIP_REASON), ("bbob", bbob::SKIP_REASON)] {
        write_json(
            &output.join(suite).join("run.json"),
            &json!({
                "schema_version": 2,
                "tutorial": "foundations",
                "formulation": suite,
                "status": "skipped",
                "preset": preset.label(),
                "seed": seed,
                "workers": workers,
                "command": command.clone(),
                "reason": reason,
                "reason_detail": "implementation was not attempted because independently sourced fixed-point fixtures are a prerequisite",
                "actual_evaluations": null,
                "artifacts": {}
            }),
        )
        .map_err(|error| error.to_string())?;
    }
    write_json(
        &output.join("run.json"),
        &json!({
            "schema_version": 2,
            "tutorial": "foundations",
            "formulation": "foundations-conformance-campaign",
            "status": "completed",
            "preset": preset.label(),
            "seed": seed,
            "workers": workers,
            "claim_scope": "deterministic-conformance-demonstration; not a statistical optimizer benchmark",
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "completed_stages": ["classic", "zdt", "dtlz", "ladder"],
            "available_components": ["cec-loader"],
            "skipped_gates": ["wfg", "bbob"],
            "artifacts": {
                "single_objective": "so/run.json",
                "multi_objective": "mo/run.json",
                "ladder": "ladder/run.json",
                "wfg_gate": "wfg/run.json",
                "bbob_gate": "bbob/run.json"
            }
        }),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(problem: &dyn Suite, decision: Vec<f64>) -> MetricSnapshot {
        let value = problem.evaluate(&decision).unwrap();
        MetricSnapshot {
            arm: "test",
            requested_evaluations: 1,
            actual_evaluations: 1,
            decisions: vec![decision],
            values: vec![value],
            wall_seconds: 0.0,
        }
    }

    #[test]
    fn shared_reference_covers_complete_front_and_fixed_box_never_filters() {
        let problem = Zdt::Zdt1(30);
        let analytic = problem.reference_front(101).unwrap();
        let outside = snapshot(&problem, vec![0.2; 30]);
        let shared = campaign_reference(&problem, &analytic, &[&outside]).unwrap();
        let (row, front) = metric_row(&problem, &outside, &analytic, &shared).unwrap();
        assert_eq!(front.objectives.len(), 1);
        assert!(row.hypervolume > 0.0);
        assert!(row.fixed_hypervolume.is_none());
        assert_eq!(
            row.fixed_hypervolume_kind,
            "not-applicable-outside-reference"
        );
        assert_eq!(row.fixed_outside_reference, 1);
    }

    #[test]
    fn fixed_box_is_reported_only_for_a_complete_eligible_front() {
        let problem = Zdt::Zdt1(30);
        let analytic = problem.reference_front(101).unwrap();
        let mut decision = vec![0.0; 30];
        decision[0] = 0.2;
        let eligible = snapshot(&problem, decision);
        let shared = campaign_reference(&problem, &analytic, &[&eligible]).unwrap();
        let (row, _) = metric_row(&problem, &eligible, &analytic, &shared).unwrap();
        assert!(row.fixed_hypervolume.is_some_and(|value| value > 0.0));
        assert_eq!(row.fixed_hypervolume_kind, "exact");
        assert_eq!(row.fixed_outside_reference, 0);
    }
}
