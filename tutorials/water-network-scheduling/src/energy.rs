//! Tutorial-owned pump energy accounting.

/// Water density in kg/m³.
pub const RHO: f64 = 998.2;
/// Standard gravity in m/s².
pub const GRAVITY: f64 = 9.806_65;

const ORACLE_CSV: &str = include_str!("../scenarios/energy-power-oracle.csv");

/// One frozen power calculation made outside the Rust implementation.
#[derive(Clone, Copy, Debug)]
pub struct EnergyOraclePoint {
    /// Zero-based pump index.
    pub pump: usize,
    /// Hydraulic flow.
    pub flow_m3_s: f64,
    /// Pump head gain.
    pub head_gain_m: f64,
    /// Independently tabulated efficiency.
    pub expected_efficiency: f64,
    /// Independently tabulated electrical power.
    pub expected_power_kw: f64,
}

/// Comparison between the implementation and one frozen oracle point.
#[derive(Clone, Copy, Debug)]
pub struct EnergyOracleCheck {
    /// Source oracle point.
    pub point: EnergyOraclePoint,
    /// Efficiency returned by the implementation.
    pub observed_efficiency: f64,
    /// Power returned by the implementation.
    pub observed_power_kw: f64,
    /// Relative power error.
    pub relative_error: f64,
}

/// Smooth synthetic efficiency curve, bounded away from zero.
#[must_use]
pub fn efficiency(pump: usize, flow_m3_s: f64) -> f64 {
    let (peak, design, curvature) = if pump == 0 {
        (0.86, 0.028, 145.0)
    } else {
        (0.78, 0.022, 190.0)
    };
    (peak - curvature * (flow_m3_s - design).powi(2)).clamp(0.55, peak)
}

/// Electrical power from hydraulic flow and head.
#[must_use]
pub fn pump_power_kw(pump: usize, flow_m3_s: f64, head_gain_m: f64) -> f64 {
    if flow_m3_s <= 0.0 || head_gain_m <= 0.0 {
        return 0.0;
    }
    RHO * GRAVITY * flow_m3_s * head_gain_m / efficiency(pump, flow_m3_s) / 1_000.0
}

/// Integrate a left-continuous power sample over one interval.
#[must_use]
pub fn interval_energy_kwh(power_kw: f64, seconds: usize) -> f64 {
    power_kw.max(0.0) * seconds as f64 / 3_600.0
}

/// Compare the implementation against the checked-in offline oracle table.
pub fn validate_energy_oracle() -> Result<Vec<EnergyOracleCheck>, String> {
    ORACLE_CSV
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 5 {
                return Err(format!("invalid energy oracle row: {line}"));
            }
            let parse = |index: usize| {
                fields[index]
                    .parse::<f64>()
                    .map_err(|error| format!("invalid energy oracle value: {error}"))
            };
            let pump = fields[0]
                .parse::<usize>()
                .map_err(|error| format!("invalid energy oracle pump: {error}"))?
                .checked_sub(1)
                .ok_or("energy oracle pump ids are one-based")?;
            let point = EnergyOraclePoint {
                pump,
                flow_m3_s: parse(1)?,
                head_gain_m: parse(2)?,
                expected_efficiency: parse(3)?,
                expected_power_kw: parse(4)?,
            };
            let observed_efficiency = efficiency(point.pump, point.flow_m3_s);
            let observed_power_kw = pump_power_kw(point.pump, point.flow_m3_s, point.head_gain_m);
            Ok(EnergyOracleCheck {
                point,
                observed_efficiency,
                observed_power_kw,
                relative_error: (observed_power_kw - point.expected_power_kw).abs()
                    / point.expected_power_kw,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_oracle_covers_both_pumps_and_is_not_identity() {
        let checks = validate_energy_oracle().unwrap();
        assert_eq!(checks.len(), 4);
        assert!(checks.iter().any(|check| check.point.pump == 0));
        assert!(checks.iter().any(|check| check.point.pump == 1));
        let maximum = checks
            .iter()
            .map(|check| check.relative_error)
            .fold(0.0_f64, f64::max);
        assert!(maximum > 0.0);
        assert!(maximum < 1.0e-6, "{maximum}");
        assert!(checks.iter().all(|check| {
            (check.observed_efficiency - check.point.expected_efficiency).abs() < 1.0e-12
        }));
    }

    #[test]
    fn off_pump_has_no_energy() {
        assert_eq!(pump_power_kw(0, 0.0, 40.0), 0.0);
        assert_eq!(interval_energy_kwh(-1.0, 3600), 0.0);
    }
}
