//! Annualized capital and fixed operating costs.

use crate::decode::OuterDesign;

/// Grid-tier one-off capital costs.
pub const GRID_TIER_CAPEX: [f64; 6] = [
    55_000.0, 82_000.0, 120_000.0, 178_000.0, 265_000.0, 390_000.0,
];

/// Capital-cost breakdown.
#[derive(Clone, Copy, Debug, Default)]
pub struct CapexBreakdown {
    /// Total one-off investment.
    pub investment: f64,
    /// Annualized investment plus fixed O&M.
    pub annualized: f64,
}

/// Capital-recovery factor.
#[must_use]
pub fn capital_recovery_factor(rate: f64, years: u32) -> f64 {
    let growth = (1.0 + rate).powi(years as i32);
    rate * growth / (growth - 1.0)
}

fn annual_cost(capex: f64, lifetime: u32, fixed_om_fraction: f64) -> f64 {
    capex * (capital_recovery_factor(0.06, lifetime) + fixed_om_fraction)
}

/// Tiered annualized cost of one decoded design.
#[must_use]
pub fn annualized_capex(design: &OuterDesign) -> CapexBreakdown {
    let capacities = design.capacities;
    let pv = 700.0 * capacities.pv_kwp;
    let wind = if design.include_wind {
        80_000.0 + 1_180.0 * capacities.wind_kw
    } else {
        0.0
    };
    let battery = if design.include_battery {
        40_000.0 + 285.0 * capacities.battery_kwh + 145.0 * capacities.battery_kw
    } else {
        0.0
    };
    let hydrogen = if design.include_hydrogen {
        60_000.0 + 610.0 * capacities.electrolyser_kw + 18.0 * capacities.hydrogen_kwh
    } else {
        0.0
    };
    let grid = GRID_TIER_CAPEX[design.grid_tier];
    CapexBreakdown {
        investment: pv + wind + battery + hydrogen + grid,
        annualized: annual_cost(pv, 25, 0.015)
            + annual_cost(wind, 25, 0.025)
            + annual_cost(battery, 15, 0.02)
            + annual_cost(hydrogen, 20, 0.025)
            + annual_cost(grid, 30, 0.01),
    }
}

/// Smooth linear grid-connection cost used only by the convex baseline.
#[must_use]
pub fn annualized_capex_linear_grid(design: &OuterDesign) -> CapexBreakdown {
    let tiered = annualized_capex(design);
    let tier_cost = GRID_TIER_CAPEX[design.grid_tier];
    let linear_cost = 96.0 * design.capacities.grid_kw;
    CapexBreakdown {
        investment: tiered.investment - tier_cost + linear_cost,
        annualized: tiered.annualized - annual_cost(tier_cost, 30, 0.01)
            + annual_cost(linear_cost, 30, 0.01),
    }
}

/// Levelized cost for a positive served-energy denominator.
#[must_use]
pub fn lcoe(annual_cost: f64, served_energy_kwh: f64) -> f64 {
    annual_cost / served_energy_kwh.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{DIMENSION, decode_outer};

    #[test]
    fn excluded_assets_have_no_hidden_fixed_cost() {
        let excluded = decode_outer(&[0.49; DIMENSION], false).unwrap();
        let included = decode_outer(&[0.51; DIMENSION], true).unwrap();
        assert!(!excluded.include_wind);
        assert!(included.include_wind);
        assert!(annualized_capex(&included).investment > annualized_capex(&excluded).investment);
    }

    #[test]
    fn lcoe_known_answer_and_recovery_factor() {
        assert!((lcoe(100_000.0, 2_000_000.0) - 0.05).abs() < 1.0e-12);
        assert!((capital_recovery_factor(0.06, 20) - 0.087_184_556).abs() < 1.0e-8);
    }
}
