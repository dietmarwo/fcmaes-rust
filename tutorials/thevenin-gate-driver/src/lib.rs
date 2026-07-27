//! Transient optimization of a lumped MOSFET gate-input network.
//!
//! The hot path is pure Rust: each candidate is rendered as a short SPICE
//! netlist and simulated by `thevenin`. The model intentionally stops at the
//! gate input. It captures the driver resistance, package/trace inductance,
//! effective input capacitance, and a series-RC gate snubber; it does not claim
//! drain switching, Miller charge, or device heating.

#![warn(missing_docs)]

use std::error::Error;

use thevenin_cirq::simulate_spice_tran;
use thevenin_types::{SimPlot, VectorData};

pub mod artifacts;
pub mod mode;
pub mod studies;

/// Number of normalized design variables.
pub const DIMENSION: usize = 2;
/// Number of minimized objectives.
pub const OBJECTIVES: usize = 2;
/// Number of inequality constraints.
pub const CONSTRAINTS: usize = 2;
/// Width of the objective-plus-constraint vector passed to MODE.
pub const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;

/// High-state voltage of the ideal pulse driver, in volts.
pub const DRIVE_VOLTAGE_V: f64 = 10.0;
/// Start of the rising pulse edge, in seconds.
pub const EDGE_START_S: f64 = 5.0e-9;
/// Default transient stop time, in seconds.
pub const DEFAULT_STOP_S: f64 = 120.0e-9;
/// Default maximum transient timestep, in seconds.
pub const DEFAULT_STEP_S: f64 = 50.0e-12;
/// Lumped package and PCB-trace inductance, in henries.
pub const TRACE_INDUCTANCE_H: f64 = 8.0e-9;
/// Linearized MOSFET gate-input capacitance, in farads.
pub const BASE_GATE_CAPACITANCE_F: f64 = 4.0e-9;

/// Lower bound for driver resistance, in ohms.
pub const RESISTANCE_LOWER_OHM: f64 = 0.2;
/// Upper bound for driver resistance, in ohms.
pub const RESISTANCE_UPPER_OHM: f64 = 6.0;
/// Lower bound for series snubber resistance, in ohms.
pub const SNUBBER_RESISTANCE_LOWER_OHM: f64 = 0.2;
/// Upper bound for series snubber resistance, in ohms.
pub const SNUBBER_RESISTANCE_UPPER_OHM: f64 = 30.0;
/// Fixed series-snubber capacitance, in farads.
pub const SNUBBER_CAPACITANCE_F: f64 = 2.0e-9;

/// Maximum permitted absolute driver current, in amperes.
pub const PEAK_CURRENT_LIMIT_A: f64 = 5.0;
/// Maximum permitted 2% settling time, in nanoseconds.
pub const SETTLING_LIMIT_NS: f64 = 75.0;

/// Physical controls decoded from the normalized optimizer coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateDesign {
    /// Driver output resistance.
    pub resistance_ohm: f64,
    /// Resistance of the series-RC gate snubber.
    pub snubber_resistance_ohm: f64,
}

impl GateDesign {
    /// Decode `u ∈ [0,1]²`. Both positive resistances use logarithmic maps.
    pub fn decode(u: &[f64]) -> Option<Self> {
        if u.len() != DIMENSION || u.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let resistance_coordinate = u[0].clamp(0.0, 1.0);
        let snubber_coordinate = u[1].clamp(0.0, 1.0);
        Some(Self {
            resistance_ohm: RESISTANCE_LOWER_OHM
                * (RESISTANCE_UPPER_OHM / RESISTANCE_LOWER_OHM).powf(resistance_coordinate),
            snubber_resistance_ohm: SNUBBER_RESISTANCE_LOWER_OHM
                * (SNUBBER_RESISTANCE_UPPER_OHM / SNUBBER_RESISTANCE_LOWER_OHM)
                    .powf(snubber_coordinate),
        })
    }
}

/// Direct time-domain measurements used by the optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateMetrics {
    /// Interpolated 10–90% gate-voltage rise time.
    pub rise_time_ns: f64,
    /// Peak gate voltage above the 10 V target, in percent.
    pub overshoot_percent: f64,
    /// Peak absolute current delivered through the driver resistance.
    pub peak_driver_current_a: f64,
    /// Time from the edge start to the final exit from the 2% band.
    pub settling_time_ns: f64,
    /// Mean gate voltage over the final 16 transient samples.
    pub final_gate_voltage_v: f64,
}

/// Complete replayable evaluation of one design.
#[derive(Clone, Debug)]
pub struct GateEvaluation {
    /// Clamped normalized optimizer coordinates.
    pub controls: [f64; DIMENSION],
    /// Physical design decoded from `controls`.
    pub design: GateDesign,
    /// Direct measurements extracted from the transient waveform.
    pub metrics: GateMetrics,
    /// Minimized rise-time and overshoot objectives.
    pub objectives: [f64; OBJECTIVES],
    /// Peak-current and settling-time residuals; nonpositive is feasible.
    pub constraints: [f64; CONSTRAINTS],
}

impl GateEvaluation {
    /// Concatenate objectives and constraints in the order expected by MODE.
    pub fn values(&self) -> Vec<f64> {
        self.objectives
            .iter()
            .chain(self.constraints.iter())
            .copied()
            .collect()
    }

    /// Return whether every inequality-constraint residual is nonpositive.
    pub fn is_feasible(&self) -> bool {
        self.constraints.iter().all(|value| *value <= 0.0)
    }
}

/// Waveform retained for plots and cross-simulator validation.
#[derive(Clone, Debug)]
pub struct GateWaveform {
    /// Transient sample times, in seconds.
    pub time_s: Vec<f64>,
    /// Ideal driver-node voltage, in volts.
    pub drive_v: Vec<f64>,
    /// Voltage after the driver resistance and before the trace inductance.
    pub trace_v: Vec<f64>,
    /// Gate-node voltage, in volts.
    pub gate_v: Vec<f64>,
}

/// Render the exact circuit sent to both `thevenin` and the ngspice reference
/// harness.
pub fn gate_netlist(design: GateDesign, step_s: f64, stop_s: f64) -> String {
    let values = [
        ("DRIVE_VOLTAGE_V", DRIVE_VOLTAGE_V),
        ("EDGE_START_S", EDGE_START_S),
        ("RESISTANCE_OHM", design.resistance_ohm),
        ("TRACE_INDUCTANCE_H", TRACE_INDUCTANCE_H),
        ("BASE_GATE_CAPACITANCE_F", BASE_GATE_CAPACITANCE_F),
        ("SNUBBER_RESISTANCE_OHM", design.snubber_resistance_ohm),
        ("SNUBBER_CAPACITANCE_F", SNUBBER_CAPACITANCE_F),
        ("STEP_S", step_s),
        ("STOP_S", stop_s),
    ];
    let mut rendered = include_str!("../netlists/gate-driver.cir").to_owned();
    for (name, value) in values {
        rendered = rendered.replace(&format!("{{{{{name}}}}}"), &format!("{value:.17e}"));
    }
    debug_assert!(!rendered.contains("{{"));
    rendered
}

fn real_vector<'a>(plot: &'a SimPlot, name: &str) -> Option<&'a [f64]> {
    plot.vecs
        .iter()
        .find(|vector| vector.name.eq_ignore_ascii_case(name))
        .and_then(|vector| match &vector.data {
            VectorData::Real(values) => Some(values.as_slice()),
            VectorData::Complex(_) => None,
        })
}

/// Simulate one design with `thevenin`.
pub fn simulate_thevenin(
    design: GateDesign,
    step_s: f64,
    stop_s: f64,
) -> Result<GateWaveform, Box<dyn Error>> {
    if !(step_s.is_finite() && step_s > 0.0 && stop_s.is_finite() && stop_s > EDGE_START_S) {
        return Err("transient step and stop time must be finite and positive".into());
    }
    let result = simulate_spice_tran(&gate_netlist(design, step_s, stop_s))?;
    let plot = result
        .plots
        .iter()
        .find(|plot| plot.name.starts_with("tran"))
        .ok_or("thevenin returned no transient plot")?;
    let time_s = real_vector(plot, "time").ok_or("missing time vector")?;
    let drive_v = real_vector(plot, "v(drive)").ok_or("missing v(drive) vector")?;
    let trace_v = real_vector(plot, "v(trace)").ok_or("missing v(trace) vector")?;
    let gate_v = real_vector(plot, "v(gate)").ok_or("missing v(gate) vector")?;
    let length = time_s.len();
    if length < 3 || drive_v.len() != length || trace_v.len() != length || gate_v.len() != length {
        return Err("transient vectors have inconsistent lengths".into());
    }
    Ok(GateWaveform {
        time_s: time_s.to_vec(),
        drive_v: drive_v.to_vec(),
        trace_v: trace_v.to_vec(),
        gate_v: gate_v.to_vec(),
    })
}

fn rising_crossing(time: &[f64], values: &[f64], threshold: f64, start_s: f64) -> Option<f64> {
    time.windows(2)
        .zip(values.windows(2))
        .find_map(|(times, samples)| {
            let t0 = times[0];
            let t1 = times[1];
            let y0 = samples[0];
            let y1 = samples[1];
            if t1 < start_s || !(y0 <= threshold && y1 >= threshold) || y1 == y0 {
                return None;
            }
            let fraction = ((threshold - y0) / (y1 - y0)).clamp(0.0, 1.0);
            Some(t0 + fraction * (t1 - t0))
        })
}

/// Measure rise time, overshoot, peak driver current, and 2% settling time.
///
/// Threshold crossings are linearly interpolated between transient samples.
/// Settling time is the final exit from the ±2% band during the recorded
/// high-state window.
pub fn measure_waveform(waveform: &GateWaveform, resistance_ohm: f64) -> Option<GateMetrics> {
    let time = &waveform.time_s;
    let gate = &waveform.gate_v;
    if resistance_ohm <= 0.0
        || time.len() != gate.len()
        || time.len() != waveform.drive_v.len()
        || time.len() != waveform.trace_v.len()
    {
        return None;
    }
    let t10 = rising_crossing(time, gate, 0.1 * DRIVE_VOLTAGE_V, EDGE_START_S)?;
    let t90 = rising_crossing(time, gate, 0.9 * DRIVE_VOLTAGE_V, t10)?;
    let rise_time_ns = (t90 - t10) * 1.0e9;
    let high_samples = time
        .iter()
        .zip(gate)
        .filter(|(sample_time, _)| **sample_time >= t10);
    let maximum = high_samples
        .map(|(_, voltage)| *voltage)
        .fold(f64::NEG_INFINITY, f64::max);
    let overshoot_percent = (100.0 * (maximum - DRIVE_VOLTAGE_V) / DRIVE_VOLTAGE_V).max(0.0);
    let peak_driver_current_a = waveform
        .drive_v
        .iter()
        .zip(&waveform.trace_v)
        .map(|(drive, trace)| ((drive - trace) / resistance_ohm).abs())
        .fold(0.0, f64::max);
    let band = 0.02 * DRIVE_VOLTAGE_V;
    let last_outside = time
        .iter()
        .zip(gate)
        .filter(|(sample_time, voltage)| {
            **sample_time >= EDGE_START_S && (*voltage - DRIVE_VOLTAGE_V).abs() > band
        })
        .map(|(sample_time, _)| *sample_time)
        .next_back()
        .unwrap_or(EDGE_START_S);
    let settling_time_ns = (last_outside - EDGE_START_S).max(0.0) * 1.0e9;
    let final_count = gate.len().min(16);
    let final_gate_voltage_v =
        gate[gate.len() - final_count..].iter().sum::<f64>() / final_count as f64;
    [
        rise_time_ns,
        overshoot_percent,
        peak_driver_current_a,
        settling_time_ns,
        final_gate_voltage_v,
    ]
    .iter()
    .all(|value| value.is_finite())
    .then_some(GateMetrics {
        rise_time_ns,
        overshoot_percent,
        peak_driver_current_a,
        settling_time_ns,
        final_gate_voltage_v,
    })
}

/// Evaluate one normalized candidate with the default publication transient.
pub fn evaluate(u: &[f64]) -> Option<GateEvaluation> {
    evaluate_with_step(u, DEFAULT_STEP_S)
}

/// Evaluate one normalized candidate at a selected maximum timestep.
pub fn evaluate_with_step(u: &[f64], step_s: f64) -> Option<GateEvaluation> {
    let design = GateDesign::decode(u)?;
    let waveform = simulate_thevenin(design, step_s, DEFAULT_STOP_S).ok()?;
    let metrics = measure_waveform(&waveform, design.resistance_ohm)?;
    let controls = [u[0].clamp(0.0, 1.0), u[1].clamp(0.0, 1.0)];
    let objectives = [metrics.rise_time_ns, metrics.overshoot_percent];
    let constraints = [
        metrics.peak_driver_current_a - PEAK_CURRENT_LIMIT_A,
        metrics.settling_time_ns - SETTLING_LIMIT_NS,
    ];
    objectives
        .iter()
        .chain(constraints.iter())
        .all(|value| value.is_finite())
        .then_some(GateEvaluation {
            controls,
            design,
            metrics,
            objectives,
            constraints,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_preserves_endpoints() {
        let low = GateDesign::decode(&[0.0, 0.0]).unwrap();
        let high = GateDesign::decode(&[1.0, 1.0]).unwrap();
        assert_eq!(low.resistance_ohm, RESISTANCE_LOWER_OHM);
        assert_eq!(low.snubber_resistance_ohm, SNUBBER_RESISTANCE_LOWER_OHM);
        assert!((high.resistance_ohm - RESISTANCE_UPPER_OHM).abs() < 1.0e-12);
        assert!((high.snubber_resistance_ohm - SNUBBER_RESISTANCE_UPPER_OHM).abs() < 1.0e-12);
    }

    #[test]
    fn decode_rejects_malformed_coordinates() {
        assert!(GateDesign::decode(&[0.5]).is_none());
        assert!(GateDesign::decode(&[0.5, f64::NAN]).is_none());
        assert!(GateDesign::decode(&[0.5, 0.5, 0.5]).is_none());
    }

    #[test]
    fn rendered_netlist_has_no_template_placeholders() {
        let design = GateDesign::decode(&[0.5, 0.5]).unwrap();
        let netlist = gate_netlist(design, DEFAULT_STEP_S, DEFAULT_STOP_S);
        assert!(!netlist.contains("{{"));
        assert!(netlist.contains("VDRIVE drive 0 PULSE"));
        assert!(netlist.contains("RDRIVE drive trace"));
        assert!(netlist.contains("CGATE gate 0"));
        assert!(netlist.contains(".tran"));
    }

    #[test]
    fn crossing_is_interpolated() {
        let time = [0.0, 1.0, 2.0];
        let values = [0.0, 0.5, 1.5];
        assert_eq!(rising_crossing(&time, &values, 1.0, 0.0), Some(1.5));
    }

    #[test]
    fn centre_design_simulates() {
        let evaluation = evaluate(&[0.5, 0.5]).unwrap();
        assert!(evaluation.metrics.rise_time_ns > 0.0);
        assert!(evaluation.metrics.final_gate_voltage_v > 9.0);
    }
}
