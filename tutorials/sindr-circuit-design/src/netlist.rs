//! Small-signal filter netlists rebuilt for every objective evaluation.

use sindr::{Circuit, CircuitElement};

fn voltage_source() -> CircuitElement {
    CircuitElement::VoltageSource {
        id: "V1".into(),
        nodes: ["vin".into(), "0".into()],
        voltage: 0.0,
        waveform: None,
    }
}

fn resistor(id: &str, from: &str, to: &str, resistance: f64) -> CircuitElement {
    CircuitElement::Resistor {
        id: id.into(),
        nodes: [from.into(), to.into()],
        resistance,
    }
}

fn capacitor(id: &str, from: &str, to: &str, capacitance: f64) -> CircuitElement {
    CircuitElement::Capacitor {
        id: id.into(),
        nodes: [from.into(), to.into()],
        capacitance,
    }
}

fn op_amp(id: &str, non_inverting: &str, inverting: &str, output: &str) -> CircuitElement {
    CircuitElement::OpAmp {
        id: id.into(),
        nodes: [non_inverting.into(), inverting.into(), output.into()],
        v_pos: 15.0,
        v_neg: -15.0,
    }
}

/// Multiple-feedback band-pass filter with `[R1, R2, R3, C1, C2]`.
pub fn mfb_bandpass(components: &[f64; 5]) -> Circuit {
    let [r1, r2, r3, c1, c2] = *components;
    Circuit {
        ground_node: "0".into(),
        components: vec![
            voltage_source(),
            resistor("R1", "vin", "n1", r1),
            resistor("R2", "n1", "0", r2),
            capacitor("C1", "n1", "inv", c1),
            capacitor("C2", "n1", "out", c2),
            resistor("R3", "inv", "out", r3),
            op_amp("U1", "0", "inv", "out"),
        ],
    }
}

/// Fourth-order low-pass formed from two passive RC sections buffered by ideal VCVS followers.
///
/// The eight values are `[R1, R2, R3, R4, C1, C2, C3, C4]`.
pub fn sallen_key_lowpass4(values: &[f64; 8]) -> Circuit {
    let mut components = vec![voltage_source()];
    // A high-gain follower cannot use the same node for both input and output;
    // each section therefore has a distinct sense and driven-output node.
    components.push(resistor("R1", "vin", "s1a", values[0]));
    components.push(resistor("R2", "s1a", "s1sense", values[1]));
    components.push(capacitor("C1", "s1a", "mid", values[4]));
    components.push(capacitor("C2", "s1sense", "0", values[5]));
    components.push(op_amp("U1", "s1sense", "mid", "mid"));

    components.push(resistor("R3", "mid", "s2a", values[2]));
    components.push(resistor("R4", "s2a", "s2sense", values[3]));
    components.push(capacitor("C3", "s2a", "out", values[6]));
    components.push(capacitor("C4", "s2sense", "0", values[7]));
    components.push(op_amp("U2", "s2sense", "out", "out"));
    Circuit {
        ground_node: "0".into(),
        components,
    }
}

/// Analytic one-pole RC low-pass used to verify feature extraction.
pub fn rc_lowpass(resistance: f64, capacitance: f64) -> Circuit {
    Circuit {
        ground_node: "0".into(),
        components: vec![
            voltage_source(),
            resistor("R1", "vin", "out", resistance),
            capacitor("C1", "out", "0", capacitance),
        ],
    }
}
