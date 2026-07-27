//! Reproducibility studies: design grid, timestep convergence, and scaling.

use std::error::Error;
use std::time::Instant;

use fcmaes_core::{Rng, parallel_batch};

use crate::{DEFAULT_STOP_S, GateDesign, GateMetrics, measure_waveform, simulate_thevenin};

#[derive(Clone, Debug)]
/// One `thevenin` result on the cross-simulator validation grid.
pub struct ValidationRow {
    /// Stable row identifier shared with the ngspice output.
    pub point_id: usize,
    /// Normalized driver and snubber coordinates.
    pub controls: [f64; 2],
    /// Decoded physical resistances.
    pub design: GateDesign,
    /// Metrics measured from the transient.
    pub metrics: GateMetrics,
    /// Number of transient samples returned by the solver.
    pub timepoints: usize,
}

#[derive(Clone, Debug)]
/// One design/timestep result in the discretization-convergence study.
pub struct TimestepRow {
    /// Stable identifier of the representative design.
    pub design_id: usize,
    /// Normalized driver and snubber coordinates.
    pub controls: [f64; 2],
    /// Maximum solver timestep, in seconds.
    pub step_s: f64,
    /// Metrics measured from the transient.
    pub metrics: GateMetrics,
    /// Number of transient samples returned by the solver.
    pub timepoints: usize,
}

#[derive(Clone, Debug)]
/// Throughput observation for one worker count and repetition.
pub struct ScalingRow {
    /// Number of parallel simulation workers.
    pub workers: i32,
    /// Zero-based repetition index.
    pub repeat: usize,
    /// Number of identical, pre-generated candidates evaluated.
    pub candidates: usize,
    /// Batch wall time, in seconds.
    pub elapsed_seconds: f64,
    /// Candidate evaluations per wall-clock second.
    pub evaluations_per_second: f64,
    /// Number of candidates for which simulation or measurement failed.
    pub failures: usize,
}

fn simulate_row(point_id: usize, controls: [f64; 2], step_s: f64) -> Option<ValidationRow> {
    let design = GateDesign::decode(&controls)?;
    let waveform = simulate_thevenin(design, step_s, DEFAULT_STOP_S).ok()?;
    let metrics = measure_waveform(&waveform, design.resistance_ohm)?;
    Some(ValidationRow {
        point_id,
        controls,
        design,
        metrics,
        timepoints: waveform.time_s.len(),
    })
}

/// Inclusive Cartesian grid used for the independent ngspice comparison.
pub fn validation_grid(side: usize, step_s: f64) -> Result<Vec<ValidationRow>, Box<dyn Error>> {
    if side < 2 {
        return Err("validation side must be at least two".into());
    }
    let mut rows = Vec::with_capacity(side * side);
    for resistance in 0..side {
        for snubber in 0..side {
            let controls = [
                resistance as f64 / (side - 1) as f64,
                snubber as f64 / (side - 1) as f64,
            ];
            let point_id = rows.len();
            rows.push(simulate_row(point_id, controls, step_s).ok_or_else(|| {
                format!("thevenin failed on validation point {point_id} at controls {controls:?}")
            })?);
        }
    }
    Ok(rows)
}

/// Three representative designs at progressively refined maximum timesteps.
pub fn timestep_study() -> Result<Vec<TimestepRow>, Box<dyn Error>> {
    const CONTROLS: [[f64; 2]; 3] = [[0.2, 0.0], [0.5, 0.5], [0.8, 1.0]];
    const STEPS: [f64; 3] = [100.0e-12, 50.0e-12, 25.0e-12];
    let mut rows = Vec::with_capacity(CONTROLS.len() * STEPS.len());
    for (design_id, controls) in CONTROLS.iter().copied().enumerate() {
        for step_s in STEPS {
            let row = simulate_row(design_id, controls, step_s)
                .ok_or("thevenin failed during the timestep study")?;
            rows.push(TimestepRow {
                design_id,
                controls,
                step_s,
                metrics: row.metrics,
                timepoints: row.timepoints,
            });
        }
    }
    Ok(rows)
}

/// Repeat the identical candidate batch at each requested worker count.
pub fn scaling_study(
    candidates: usize,
    repeats: usize,
    worker_counts: &[i32],
    seed: u64,
) -> Result<Vec<ScalingRow>, Box<dyn Error>> {
    if candidates == 0 || repeats == 0 {
        return Err("scaling candidates and repeats must be positive".into());
    }
    if worker_counts.is_empty() || worker_counts.iter().any(|workers| *workers < 1) {
        return Err("scaling worker counts must be positive".into());
    }
    let mut rng = Rng::new(seed);
    let controls = (0..candidates)
        .map(|_| [rng.uniform01(), rng.uniform01()])
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for &workers in worker_counts {
        for repeat in 0..repeats {
            let started = Instant::now();
            let successful = parallel_batch(&controls, workers, |u| crate::evaluate(u).is_some());
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let failures = successful.iter().filter(|success| !**success).count();
            rows.push(ScalingRow {
                workers,
                repeat,
                candidates,
                elapsed_seconds,
                evaluations_per_second: candidates as f64 / elapsed_seconds.max(1.0e-12),
                failures,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_step_is_converged_against_refinement() {
        let rows = timestep_study().unwrap();
        for design_id in 0..3 {
            let publication = rows
                .iter()
                .find(|row| design_id == row.design_id && row.step_s == 50.0e-12)
                .unwrap();
            let refined = rows
                .iter()
                .find(|row| design_id == row.design_id && row.step_s == 25.0e-12)
                .unwrap();
            assert!((publication.metrics.rise_time_ns - refined.metrics.rise_time_ns).abs() < 0.01);
            assert!(
                (publication.metrics.overshoot_percent - refined.metrics.overshoot_percent).abs()
                    < 0.01
            );
            assert!(
                (publication.metrics.peak_driver_current_a - refined.metrics.peak_driver_current_a)
                    .abs()
                    < 0.01
            );
            assert!(
                (publication.metrics.settling_time_ns - refined.metrics.settling_time_ns).abs()
                    < 0.1
            );
            assert!(
                (publication.metrics.final_gate_voltage_v - refined.metrics.final_gate_voltage_v)
                    .abs()
                    < 0.01
            );
        }
    }

    #[test]
    fn validation_grid_is_boundary_inclusive() {
        assert!(validation_grid(1, crate::DEFAULT_STEP_S).is_err());
        let rows = validation_grid(2, crate::DEFAULT_STEP_S).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.first().unwrap().controls, [0.0, 0.0]);
        assert_eq!(rows.last().unwrap().controls, [1.0, 1.0]);
    }
}
