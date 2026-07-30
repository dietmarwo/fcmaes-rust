//! Named deterministic training and holdout scenarios.

use std::sync::OnceLock;

use crate::config::Preset;
use crate::dispatch::HubCapacities;
use crate::profiles::{Profile, ProfileModifiers, quarter_hour_day, representative_days};

const TABLE: &str = include_str!("../scenarios/scenario-modifiers.csv");

/// Scenario set membership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioSet {
    /// Used inside the optimizer objective.
    Training,
    /// Used only after selection.
    Holdout,
}

/// Scenario perturbation kind.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScenarioKind {
    /// Unmodified synthetic year.
    Base,
    /// Reduced PV availability.
    LowSolar,
    /// Winter-specific load growth.
    HighWinterLoad,
    /// Shifted time-of-use tariff.
    TariffShift,
    /// No export compensation.
    ZeroExportPrice,
    /// Extended wind outage proxy.
    WindOutage,
    /// Reduced usable battery power and energy.
    BatteryDerate,
    /// Uniform electrical-load growth.
    LoadGrowth,
    /// Finer timestep validation.
    QuarterHour,
}

/// Parsed checked-in scenario definition.
#[derive(Clone, Debug)]
pub struct Scenario {
    /// Stable name.
    pub name: &'static str,
    /// Training or holdout.
    pub set: ScenarioSet,
    /// Structural kind.
    pub kind: ScenarioKind,
    /// Synthetic-profile changes.
    pub modifiers: ProfileModifiers,
    /// Multiplicative usable battery capacity.
    pub battery_factor: f64,
    /// Whether this is the quarter-hour replay.
    pub quarter_hour: bool,
}

impl Scenario {
    /// Build the deterministic dispatch profile for this scenario.
    #[must_use]
    pub fn profile(&self, preset: Preset) -> Profile {
        if self.quarter_hour {
            quarter_hour_day(self.modifiers)
        } else {
            representative_days(preset.protocol().representative_days, self.modifiers)
        }
    }

    /// Apply capacity-side derating.
    #[must_use]
    pub fn capacities(&self, mut capacities: HubCapacities) -> HubCapacities {
        capacities.battery_kwh *= self.battery_factor;
        capacities.battery_kw *= self.battery_factor;
        capacities
    }
}

fn kind(name: &str) -> ScenarioKind {
    match name {
        "base_year" => ScenarioKind::Base,
        "low_solar_year" => ScenarioKind::LowSolar,
        "high_load_winter" => ScenarioKind::HighWinterLoad,
        "tariff_peak_shifted" => ScenarioKind::TariffShift,
        "export_price_zero" => ScenarioKind::ZeroExportPrice,
        "wind_outage" => ScenarioKind::WindOutage,
        "battery_derated_80pct" => ScenarioKind::BatteryDerate,
        "load_growth_15pct" => ScenarioKind::LoadGrowth,
        "quarter_hour_replay" => ScenarioKind::QuarterHour,
        _ => panic!("unknown checked-in scenario {name}"),
    }
}

fn stable_name(name: &str) -> &'static str {
    match name {
        "base_year" => "base_year",
        "low_solar_year" => "low_solar_year",
        "high_load_winter" => "high_load_winter",
        "tariff_peak_shifted" => "tariff_peak_shifted",
        "export_price_zero" => "export_price_zero",
        "wind_outage" => "wind_outage",
        "battery_derated_80pct" => "battery_derated_80pct",
        "load_growth_15pct" => "load_growth_15pct",
        "quarter_hour_replay" => "quarter_hour_replay",
        _ => panic!("unknown checked-in scenario {name}"),
    }
}

fn parse() -> Vec<Scenario> {
    TABLE
        .lines()
        .skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let columns = line.split(',').collect::<Vec<_>>();
            assert_eq!(columns.len(), 10);
            let name = stable_name(columns[0]);
            Scenario {
                name,
                set: match columns[1] {
                    "training" => ScenarioSet::Training,
                    "holdout" => ScenarioSet::Holdout,
                    value => panic!("unknown scenario set {value}"),
                },
                kind: kind(name),
                modifiers: ProfileModifiers {
                    solar_factor: columns[2].parse().expect("numeric solar factor"),
                    wind_factor: columns[3].parse().expect("numeric wind factor"),
                    load_factor: columns[4].parse().expect("numeric load factor"),
                    winter_load_boost: columns[5].parse().expect("numeric winter boost"),
                    tariff_shift_hours: columns[6].parse().expect("integer tariff shift"),
                    export_price_factor: columns[7].parse().expect("numeric export factor"),
                    wind_outage: columns[0] == "wind_outage",
                },
                battery_factor: columns[8].parse().expect("numeric battery factor"),
                quarter_hour: columns[9] == "quarter-hour",
            }
        })
        .collect()
}

/// All checked-in scenarios.
#[must_use]
pub fn all() -> &'static [Scenario] {
    static SCENARIOS: OnceLock<Vec<Scenario>> = OnceLock::new();
    SCENARIOS.get_or_init(parse)
}

/// Training scenarios in stable order.
#[must_use]
pub fn training() -> Vec<&'static Scenario> {
    all()
        .iter()
        .filter(|scenario| scenario.set == ScenarioSet::Training)
        .collect()
}

/// Holdout scenarios in stable order.
#[must_use]
pub fn holdout() -> Vec<&'static Scenario> {
    all()
        .iter()
        .filter(|scenario| scenario.set == ScenarioSet::Holdout)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn checked_in_sets_are_complete_and_disjoint_by_kind() {
        assert_eq!(training().len(), 5);
        assert_eq!(holdout().len(), 4);
        let training_kinds = training()
            .iter()
            .map(|scenario| scenario.kind)
            .collect::<HashSet<_>>();
        assert!(
            holdout()
                .iter()
                .all(|scenario| !training_kinds.contains(&scenario.kind))
        );
        assert_eq!(
            holdout()
                .iter()
                .find(|scenario| scenario.quarter_hour)
                .unwrap()
                .profile(Preset::Publication)
                .dt_hours,
            0.25
        );
    }
}
