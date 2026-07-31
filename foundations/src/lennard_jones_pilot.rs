//! Pre-registered descriptor pilot for the optional Lennard-Jones QD arm.

use std::collections::BTreeSet;
use std::path::Path;

use fcmaes_core::{Rng, retry_run_seed};
use serde::Serialize;
use serde_json::json;

use crate::artifacts::{write_json, write_text};
use crate::suites::Suite;
use crate::suites::lennard_jones::{COORDINATION_CUTOFF, LennardJones, Parameterization};

const ATOMS: usize = 38;
const ARMS: usize = 3;
const CANDIDATES_PER_ARM: usize = 64;
const FINE: usize = 12;
const COARSE: usize = 6;
const RG_BOUNDS: [f64; 2] = [0.25, 0.75];
const COORD_BOUNDS: [f64; 2] = [0.0, 12.0];

#[derive(Serialize)]
struct PilotRow {
    seed_arm: usize,
    candidate: usize,
    radius_gyration_normalized: f64,
    mean_coordination: f64,
    holdout_radius_gyration_normalized: f64,
    holdout_mean_coordination: f64,
    fine_niche: Option<usize>,
    holdout_fine_niche: Option<usize>,
    coarse_niche: Option<usize>,
    holdout_coarse_niche: Option<usize>,
    cutoff_low_niche: Option<usize>,
    cutoff_high_niche: Option<usize>,
}

fn bin(value: f64, bounds: [f64; 2], resolution: usize) -> Option<usize> {
    if value < bounds[0] || value > bounds[1] {
        return None;
    }
    let fraction = ((value - bounds[0]) / (bounds[1] - bounds[0])).clamp(0.0, 1.0);
    Some(((fraction * resolution as f64) as usize).min(resolution - 1))
}

fn niche(descriptor: [f64; 2], resolution: usize) -> Option<usize> {
    let x = bin(descriptor[0], RG_BOUNDS, resolution)?;
    let y = bin(descriptor[1], COORD_BOUNDS, resolution)?;
    Some(y * resolution + x)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut result = vec![0.0; values.len()];
    let mut begin = 0;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let rank = 0.5 * (begin + end - 1) as f64;
        for &index in &order[begin..end] {
            result[index] = rank;
        }
        begin = end;
    }
    result
}

fn correlation(left: &[f64], right: &[f64]) -> f64 {
    let left = ranks(left);
    let right = ranks(right);
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(&right)
        .map(|(&x, &y)| (x - left_mean) * (y - right_mean))
        .sum::<f64>();
    let left_scale = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_scale = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    if left_scale == 0.0 || right_scale == 0.0 {
        0.0
    } else {
        covariance / (left_scale * right_scale)
    }
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

fn write_csv(path: &Path, rows: &[PilotRow]) -> Result<(), String> {
    let mut output = String::from(
        "seed_arm,candidate,radius_gyration_normalized,mean_coordination,holdout_radius_gyration_normalized,holdout_mean_coordination,fine_niche,holdout_fine_niche,coarse_niche,holdout_coarse_niche,cutoff_low_niche,cutoff_high_niche\n",
    );
    for row in rows {
        let optional = |value: Option<usize>| value.map_or(String::new(), |item| item.to_string());
        output.push_str(&format!(
            "{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{},{},{},{}\n",
            row.seed_arm,
            row.candidate,
            row.radius_gyration_normalized,
            row.mean_coordination,
            row.holdout_radius_gyration_normalized,
            row.holdout_mean_coordination,
            optional(row.fine_niche),
            optional(row.holdout_fine_niche),
            optional(row.coarse_niche),
            optional(row.holdout_coarse_niche),
            optional(row.cutoff_low_niche),
            optional(row.cutoff_high_niche),
        ));
    }
    write_text(path, &output).map_err(|error| error.to_string())
}

/// Run the frozen descriptor gate and write an explicit QD decision.
pub fn run(root_seed: u64, output: &Path) -> Result<(), String> {
    let problem = LennardJones::new(ATOMS, Parameterization::FixedFrame)
        .map_err(|error| error.to_string())?;
    let (lower, upper) = problem.bounds();
    let mut rows = Vec::with_capacity(ARMS * CANDIDATES_PER_ARM);
    let mut coverage = Vec::new();
    for arm in 0..ARMS {
        let arm_seed = retry_run_seed(root_seed ^ 0x0051_4450_494c_4f54, arm);
        let mut occupied = BTreeSet::new();
        for candidate in 0..CANDIDATES_PER_ARM {
            let seed = retry_run_seed(arm_seed, candidate);
            let decision = problem
                .initial_decision(seed)
                .map_err(|error| error.to_string())?;
            let train = problem
                .descriptors(&decision)
                .map_err(|error| error.to_string())?;
            let mut rng = Rng::new(seed ^ 0x0048_4f4c_444f_5554);
            let holdout_decision: Vec<f64> = decision
                .iter()
                .zip(&lower)
                .zip(&upper)
                .map(|((&value, &low), &high)| (value + 0.01 * rng.gaussian()).clamp(low, high))
                .collect();
            let holdout = problem
                .descriptors(&holdout_decision)
                .map_err(|error| error.to_string())?;
            let low = problem
                .descriptors_at_cutoff(&decision, COORDINATION_CUTOFF - 0.01)
                .map_err(|error| error.to_string())?;
            let high = problem
                .descriptors_at_cutoff(&decision, COORDINATION_CUTOFF + 0.01)
                .map_err(|error| error.to_string())?;
            let fine_niche = niche(train, FINE);
            if let Some(cell) = fine_niche {
                occupied.insert(cell);
            }
            rows.push(PilotRow {
                seed_arm: arm,
                candidate,
                radius_gyration_normalized: train[0],
                mean_coordination: train[1],
                holdout_radius_gyration_normalized: holdout[0],
                holdout_mean_coordination: holdout[1],
                fine_niche,
                holdout_fine_niche: niche(holdout, FINE),
                coarse_niche: niche(train, COARSE),
                holdout_coarse_niche: niche(holdout, COARSE),
                cutoff_low_niche: niche(low, FINE),
                cutoff_high_niche: niche(high, FINE),
            });
        }
        coverage.push(fraction(occupied.len(), FINE * FINE));
    }
    let radius: Vec<f64> = rows
        .iter()
        .map(|row| row.radius_gyration_normalized)
        .collect();
    let coordination: Vec<f64> = rows.iter().map(|row| row.mean_coordination).collect();
    let clipped = rows.iter().filter(|row| row.fine_niche.is_none()).count();
    let fine_comparable = rows
        .iter()
        .filter(|row| row.fine_niche.is_some() && row.holdout_fine_niche.is_some())
        .count();
    let fine_retained = rows
        .iter()
        .filter(|row| row.fine_niche.is_some() && row.fine_niche == row.holdout_fine_niche)
        .count();
    let coarse_comparable = rows
        .iter()
        .filter(|row| row.coarse_niche.is_some() && row.holdout_coarse_niche.is_some())
        .count();
    let coarse_retained = rows
        .iter()
        .filter(|row| row.coarse_niche.is_some() && row.coarse_niche == row.holdout_coarse_niche)
        .count();
    let sensitivity_retained = rows
        .iter()
        .filter(|row| {
            row.fine_niche.is_some()
                && row.fine_niche == row.cutoff_low_niche
                && row.fine_niche == row.cutoff_high_niche
        })
        .count();
    let clipping_fraction = fraction(clipped, rows.len());
    let rank_correlation = correlation(&radius, &coordination);
    let mean_coverage = coverage.iter().sum::<f64>() / coverage.len() as f64;
    let holdout_retention = fraction(fine_retained, fine_comparable.max(1));
    let coarse_retention = fraction(coarse_retained, coarse_comparable.max(1));
    let cutoff_sensitivity_retention = fraction(sensitivity_retained, rows.len());
    let accepted = clipping_fraction <= 0.05
        && rank_correlation.abs() <= 0.90
        && mean_coverage >= 0.08
        && holdout_retention >= 0.60
        && coarse_retention >= 0.75
        && cutoff_sensitivity_retention >= 0.60;
    let verdict = if accepted { "accepted" } else { "rejected" };
    let pilot = output.join("pilot");
    write_csv(&pilot.join("pilot.csv"), &rows)?;
    let radius_min = radius.iter().copied().fold(f64::INFINITY, f64::min);
    let radius_max = radius.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let coordination_min = coordination.iter().copied().fold(f64::INFINITY, f64::min);
    let coordination_max = coordination
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    write_json(
        &pilot.join("run.json"),
        &json!({
            "schema_version": 1,
            "status": "completed",
            "verdict": verdict,
            "descriptor_pair": ["radius_gyration_normalized", "mean_coordination"],
            "fallback_pair": ["radius_gyration_normalized", "overlap_pairs"],
            "atoms": ATOMS,
            "seed_arms": ARMS,
            "candidates_per_arm": CANDIDATES_PER_ARM,
            "generator": "compact-separated-initializer",
            "holdout": "independent N(0, 0.01) coordinate perturbation",
            "bounds": [RG_BOUNDS, COORD_BOUNDS],
            "fine_grid": [FINE, FINE],
            "coarse_grid": [COARSE, COARSE],
            "reachable": [[radius_min, radius_max], [coordination_min, coordination_max]],
            "rank_correlation": rank_correlation,
            "lower_upper_clipping_fraction": clipping_fraction,
            "coverage_by_seed_arm": coverage,
            "mean_coverage": mean_coverage,
            "holdout_niche_retention": holdout_retention,
            "coarse_holdout_niche_retention": coarse_retention,
            "cutoff_sensitivity_retention": cutoff_sensitivity_retention,
            "gates": {
                "clipping_fraction_max": 0.05,
                "absolute_rank_correlation_max": 0.90,
                "mean_coverage_min": 0.08,
                "holdout_niche_retention_min": 0.60,
                "coarse_holdout_niche_retention_min": 0.75,
                "cutoff_sensitivity_retention_min": 0.60
            }
        }),
    )
    .map_err(|error| error.to_string())?;
    let markdown = format!(
        "# Lennard-Jones descriptor pilot\n\n\
         Verdict: **{verdict}**. The pre-registered pair is normalized radius of gyration × mean coordination at cutoff `{COORDINATION_CUTOFF}`.\n\n\
         | Measure | Result | Gate |\n|---|---:|---:|\n\
         | Reachable normalized radius | {radius_min:.4} … {radius_max:.4} | diagnostic |\n\
         | Reachable mean coordination | {coordination_min:.4} … {coordination_max:.4} | diagnostic |\n\
         | Absolute rank correlation | {correlation:.4} | ≤ 0.90 |\n\
         | Bound clipping | {clipping:.4} | ≤ 0.05 |\n\
         | Mean 12×12 coverage | {coverage:.4} | ≥ 0.08 |\n\
         | Perturbation same-niche retention | {holdout:.4} | ≥ 0.60 |\n\
         | Coarse 6×6 retention | {coarse:.4} | ≥ 0.75 |\n\
         | Cutoff ±0.01 retention | {sensitivity:.4} | ≥ 0.60 |\n\n\
         The candidate generator and thresholds were frozen before this archive was inspected. A rejected gate is evidence about descriptor reachability or stability; it is not repaired by changing bins after seeing the result.\n",
        correlation = rank_correlation.abs(),
        clipping = clipping_fraction,
        coverage = mean_coverage,
        holdout = holdout_retention,
        coarse = coarse_retention,
        sensitivity = cutoff_sensitivity_retention,
    );
    write_text(&pilot.join("pilot.md"), &markdown).map_err(|error| error.to_string())?;
    write_json(
        &output.join("qd/run.json"),
        &if accepted {
            json!({
                "schema_version": 1,
                "status": "not-run",
                "reason": "descriptor-pilot-accepted-but-publication-archive-not-requested",
                "claim_ceiling": "pilot-only",
                "actual_evaluations": null,
                "artifacts": {}
            })
        } else {
            json!({
                "schema_version": 1,
                "status": "skipped",
                "reason": "descriptor-pilot-rejected",
                "claim_ceiling": "no-quality-diversity-claim",
                "actual_evaluations": null,
                "artifacts": {}
            })
        },
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_includes_upper_bound_and_rejects_clipping() {
        assert_eq!(bin(0.25, RG_BOUNDS, 12), Some(0));
        assert_eq!(bin(0.75, RG_BOUNDS, 12), Some(11));
        assert_eq!(bin(0.24, RG_BOUNDS, 12), None);
    }

    #[test]
    fn tied_rank_correlation_is_well_defined() {
        assert!((correlation(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 1.0).abs() < 1.0e-12);
        assert_eq!(correlation(&[1.0, 1.0], &[2.0, 3.0]), 0.0);
    }
}
