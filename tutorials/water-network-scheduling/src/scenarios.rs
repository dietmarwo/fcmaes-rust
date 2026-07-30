//! Named training and holdout scenarios with an explicit DDA/PDA split.

use epanet_rs::model::link::LinkStatus;
use epanet_rs::model::network::{LinkUpdate, Network, PipeUpdate, ReservoirUpdate};
use epanet_rs::model::options::DemandModel;

/// Hydraulic analysis type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalysisType {
    Dda,
    Pda,
}

impl AnalysisType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dda => "DDA",
            Self::Pda => "PDA",
        }
    }
}

/// Deterministic physical or economic perturbation.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub analysis: AnalysisType,
    pub demand_multiplier: f64,
    pub reservoir_delta_m: f64,
    pub roughness_factor: f64,
    pub pump_outage: Option<usize>,
    pub pipe_outage: Option<&'static str>,
    pub tariff_shift_h: i32,
    pub profile_phase_h: usize,
}

impl Scenario {
    /// Apply all topology-independent changes before initializing a simulation.
    pub fn configure(&self, network: &mut Network, timestep_s: usize) -> Result<(), String> {
        network.options.time_options.duration = 24 * 3_600;
        network.options.time_options.hydraulic_timestep = timestep_s;
        network.options.time_options.report_timestep = timestep_s;
        network.options.time_options.pattern_timestep = 3_600;
        network.options.time_options.pattern_start = self.profile_phase_h * 3_600;
        network.options.demand_multiplier = self.demand_multiplier;
        network.options.demand_model = match self.analysis {
            AnalysisType::Dda => DemandModel::DDA,
            AnalysisType::Pda => DemandModel::PDA,
        };
        // SimulationOptions are stored in internal feet.
        network.options.minimum_pressure = 0.0;
        network.options.required_pressure = 20.0 / 0.3048;
        network.options.pressure_exponent = 0.5;
        network
            .update_reservoir(
                "R1",
                &ReservoirUpdate {
                    elevation: Some(43.0 + self.reservoir_delta_m),
                    ..Default::default()
                },
            )
            .map_err(|error| error.to_string())?;
        if (self.roughness_factor - 1.0).abs() > f64::EPSILON {
            for index in 1..=36 {
                let id = format!("P{index:02}");
                network
                    .update_pipe(
                        &id,
                        &PipeUpdate {
                            roughness: Some(0.20 * self.roughness_factor),
                            ..Default::default()
                        },
                    )
                    .map_err(|error| error.to_string())?;
            }
        }
        if let Some(pipe) = self.pipe_outage {
            network
                .update_link(
                    pipe,
                    &LinkUpdate {
                        initial_status: Some(LinkStatus::Closed),
                        ..Default::default()
                    },
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

/// Six DDA training scenarios.
#[must_use]
pub fn training() -> Vec<Scenario> {
    vec![
        scenario("normal_weekday"),
        Scenario {
            name: "high_peak_day",
            demand_multiplier: 1.15,
            ..scenario("x")
        },
        Scenario {
            name: "low_night_demand",
            demand_multiplier: 0.84,
            ..scenario("x")
        },
        Scenario {
            name: "pattern_perturbed",
            profile_phase_h: 1,
            ..scenario("x")
        },
        Scenario {
            name: "reservoir_head_reduced_2m",
            reservoir_delta_m: -2.0,
            ..scenario("x")
        },
        Scenario {
            name: "roughness_degraded_10pct",
            roughness_factor: 1.10,
            ..scenario("x")
        },
    ]
}

/// Five structurally different holdout scenarios.
#[must_use]
pub fn holdout() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "unseen_demand_profile",
            demand_multiplier: 1.06,
            profile_phase_h: 3,
            ..scenario("x")
        },
        Scenario {
            name: "pump_1_outage",
            analysis: AnalysisType::Pda,
            pump_outage: Some(0),
            ..scenario("x")
        },
        Scenario {
            name: "pipe_outage_trunk",
            analysis: AnalysisType::Pda,
            pipe_outage: Some("P01"),
            ..scenario("x")
        },
        Scenario {
            name: "tariff_peak_shifted_3h",
            tariff_shift_h: 3,
            ..scenario("x")
        },
        Scenario {
            name: "hydraulic_timestep_halved",
            ..scenario("x")
        },
    ]
}

const fn scenario(name: &'static str) -> Scenario {
    Scenario {
        name,
        analysis: AnalysisType::Dda,
        demand_multiplier: 1.0,
        reservoir_delta_m: 0.0,
        roughness_factor: 1.0,
        pump_outage: None,
        pipe_outage: None,
        tariff_shift_h: 0,
        profile_phase_h: 0,
    }
}

/// Tariff in currency/kWh at a clock hour.
#[must_use]
pub fn tariff(hour: usize, shift_h: i32) -> f64 {
    let shifted = (hour as i32 - shift_h).rem_euclid(24) as usize;
    match shifted {
        0..=5 => 0.09,
        16..=20 => 0.31,
        _ => 0.16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_and_holdout_analysis_contract_is_explicit() {
        assert!(
            training()
                .iter()
                .all(|scenario| scenario.analysis == AnalysisType::Dda)
        );
        assert!(
            holdout()
                .iter()
                .any(|scenario| scenario.analysis == AnalysisType::Pda)
        );
    }

    #[test]
    fn tariff_shift_changes_peak_window() {
        assert_ne!(tariff(18, 0), tariff(18, 3));
    }
}
