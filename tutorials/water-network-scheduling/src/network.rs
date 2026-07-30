//! Deterministic tutorial-owned network loading and serialization.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use epanet_rs::model::network::Network;
use epanet_rs::simulation::Simulation;

/// Checked-in deterministic EPANET input.
pub const INP_TEXT: &str = include_str!("../network/synthetic-zone.inp");

/// Path of the checked-in network.
#[must_use]
pub fn checked_in_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("network")
        .join("synthetic-zone.inp")
}

/// Load the checked-in synthetic network.
pub fn load() -> Result<Network, Box<dyn Error>> {
    Ok(Network::from_file(
        checked_in_path()
            .to_str()
            .ok_or("network path is not valid UTF-8")?,
    )?)
}

/// Write the deterministic network input verbatim.
pub fn write(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, INP_TEXT)?;
    Ok(())
}

/// Compile-time thread-safety contract for independent candidates.
pub fn assert_thread_safety() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Network>();
    assert_send_sync::<Simulation>();
}

/// Relative error for a laminar one-pipe Darcy-Weisbach/Hagen-Poiseuille case.
pub fn analytic_pipe_relative_error() -> Result<f64, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("network")
        .join("analytic-pipe.inp");
    let network = Network::from_file(path.to_str().ok_or("invalid analytic path")?)?;
    let junction = network.node_map["J1"];
    let mut simulation = Simulation::new(network);
    let result = simulation.solve_hydraulics(false)?;
    let observed_headloss_m = 50.0 - result.heads[0][junction];
    // epanet-rs 0.2.3 uses EPANET's fixed internal viscosity of
    // 1.1e-5 ft²/s in its Darcy-Weisbach coefficients. The parsed
    // `[OPTIONS] VISCOSITY` value is not consulted by that code path.
    let nu_m2_s = 1.1e-5 * 0.3048_f64.powi(2);
    let length_m = 100.0;
    let flow_m3_s = 0.05 / 1_000.0;
    let diameter_m: f64 = 0.1;
    let backend_gravity_m_s2 = 32.2 * 0.3048;
    let expected_headloss_m = 128.0 * nu_m2_s * length_m * flow_m3_s
        / (backend_gravity_m_s2 * std::f64::consts::PI * diameter_m.powi(4));
    Ok((observed_headloss_m - expected_headloss_m).abs() / expected_headloss_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epanet_rs::model::node::NodeType;
    use std::collections::VecDeque;

    #[test]
    fn checked_in_network_parses_and_has_expected_shape() {
        let network = load().unwrap();
        assert_eq!(network.nodes.len(), 25); // 20 demands + header + 2 valve nodes + source + tank
        assert_eq!(network.links.len(), 39); // 36 pipes + 2 pumps + PRV
        assert_eq!(network.controls.len(), 0);
        assert!(network.has_tanks());
        assert!(network.contains_pressure_control_valve);
    }

    #[test]
    fn network_and_simulation_are_send_sync() {
        assert_thread_safety();
    }

    #[test]
    fn laminar_pipe_matches_closed_form() {
        let error = analytic_pipe_relative_error().unwrap();
        assert!(error < 1.0e-5, "{error}");
    }

    #[test]
    fn every_node_is_connected_to_the_reservoir() {
        let network = load().unwrap();
        let source = network.node_map["R1"];
        let mut seen = vec![false; network.nodes.len()];
        let mut queue = VecDeque::from([source]);
        seen[source] = true;
        while let Some(node) = queue.pop_front() {
            for link in &network.links {
                let other = if link.start_node == node {
                    Some(link.end_node)
                } else if link.end_node == node {
                    Some(link.start_node)
                } else {
                    None
                };
                if let Some(other) = other
                    && !seen[other]
                {
                    seen[other] = true;
                    queue.push_back(other);
                }
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn tank_geometry_matches_decoder_contract() {
        let network = load().unwrap();
        let tank = &network.nodes[network.node_map["T1"]];
        let NodeType::Tank(tank) = &tank.node_type else {
            panic!("T1 is not a tank");
        };
        assert!(tank.min_level * 0.3048 < 1.2);
        assert!(tank.max_level * 0.3048 > 9.8);
        assert!(tank.initial_level * 0.3048 > 5.5);
    }

    #[test]
    fn embedded_input_matches_checked_in_file() {
        assert_eq!(fs::read_to_string(checked_in_path()).unwrap(), INP_TEXT);
    }
}
