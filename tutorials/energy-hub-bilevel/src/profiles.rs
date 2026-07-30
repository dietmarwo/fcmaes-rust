//! Deterministic synthetic electricity, renewable, tariff, and hydrogen profiles.

use std::f64::consts::TAU;

/// One dispatch horizon.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Timestep duration in hours.
    pub dt_hours: f64,
    /// Number of steps in each independently cyclic block.
    pub cyclic_block_steps: usize,
    /// Normalized PV production per installed kWp.
    pub solar_cf: Vec<f64>,
    /// Normalized wind production per installed kW.
    pub wind_cf: Vec<f64>,
    /// Electrical demand in kW.
    pub load_kw: Vec<f64>,
    /// Grid import price in currency units per kWh.
    pub import_price: Vec<f64>,
    /// Grid export price in currency units per kWh.
    pub export_price: Vec<f64>,
    /// Industrial hydrogen demand in kW-H₂.
    pub hydrogen_demand_kw: Vec<f64>,
    /// Purchased hydrogen price per kWh-H₂.
    pub hydrogen_price: Vec<f64>,
}

impl Profile {
    /// Number of dispatch timesteps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.load_kw.len()
    }

    /// Whether the profile contains no timesteps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.load_kw.is_empty()
    }

    /// Total electrical demand.
    #[must_use]
    pub fn load_energy_kwh(&self) -> f64 {
        self.dt_hours * self.load_kw.iter().sum::<f64>()
    }

    /// Total hydrogen demand.
    #[must_use]
    pub fn hydrogen_energy_kwh(&self) -> f64 {
        self.dt_hours * self.hydrogen_demand_kw.iter().sum::<f64>()
    }
}

/// Deterministic changes defining one named scenario.
#[derive(Clone, Copy, Debug)]
pub struct ProfileModifiers {
    /// Multiplicative solar availability.
    pub solar_factor: f64,
    /// Multiplicative wind availability.
    pub wind_factor: f64,
    /// Multiplicative electrical load.
    pub load_factor: f64,
    /// Additional winter-only load fraction.
    pub winter_load_boost: f64,
    /// Circular shift of tariff bands in hours.
    pub tariff_shift_hours: i32,
    /// Multiplicative export price.
    pub export_price_factor: f64,
    /// Whether wind is unavailable during a fixed interval.
    pub wind_outage: bool,
}

impl Default for ProfileModifiers {
    fn default() -> Self {
        Self {
            solar_factor: 1.0,
            wind_factor: 1.0,
            load_factor: 1.0,
            winter_load_boost: 0.0,
            tariff_shift_hours: 0,
            export_price_factor: 1.0,
            wind_outage: false,
        }
    }
}

fn base_hour(hour_of_year: f64, modifiers: ProfileModifiers) -> [f64; 7] {
    let year_fraction = hour_of_year / 8_760.0;
    let local_hour = hour_of_year.rem_euclid(24.0);
    let day_fraction = local_hour / 24.0;
    let season = (TAU * (year_fraction - 0.25)).sin();
    let daylight = (TAU * (day_fraction - 0.25)).sin().max(0.0);
    let solar = (daylight * (0.70 + 0.28 * season)).max(0.0) * modifiers.solar_factor;
    let mut wind = (0.38
        + 0.13 * (TAU * year_fraction * 7.0 + 0.4).sin()
        + 0.08 * (TAU * day_fraction * 3.0).cos())
    .clamp(0.04, 0.82)
        * modifiers.wind_factor;
    let outage_start = 24.0 * 20.0;
    if modifiers.wind_outage && (outage_start..outage_start + 24.0 * 14.0).contains(&hour_of_year) {
        wind = 0.0;
    }
    let winter_weight = ((-season + 1.0) * 0.5).clamp(0.0, 1.0);
    let load = (1_050.0
        + 155.0 * (TAU * (day_fraction - 0.18)).sin()
        + 105.0 * (TAU * year_fraction + 0.7).cos())
        * modifiers.load_factor
        * (1.0 + modifiers.winter_load_boost * winter_weight);
    let shifted_hour = (local_hour.round() as i32 - modifiers.tariff_shift_hours).rem_euclid(24);
    let import_price = if matches!(shifted_hour, 7..=10 | 17..=21) {
        0.245
    } else if matches!(shifted_hour, 0..=5) {
        0.095
    } else {
        0.145
    };
    let export_price = 0.045 * modifiers.export_price_factor;
    let hydrogen_demand = 145.0
        * (0.92 + 0.08 * (TAU * (day_fraction - 0.1)).sin())
        * (0.96 + 0.04 * (TAU * year_fraction).cos());
    let hydrogen_price = 0.19 + 0.015 * winter_weight;
    [
        solar,
        wind,
        load,
        import_price,
        export_price,
        hydrogen_demand,
        hydrogen_price,
    ]
}

fn push_sample(profile: &mut Profile, hour: f64, modifiers: ProfileModifiers) {
    let sample = base_hour(hour, modifiers);
    profile.solar_cf.push(sample[0]);
    profile.wind_cf.push(sample[1]);
    profile.load_kw.push(sample[2]);
    profile.import_price.push(sample[3]);
    profile.export_price.push(sample[4]);
    profile.hydrogen_demand_kw.push(sample[5]);
    profile.hydrogen_price.push(sample[6]);
}

fn empty(dt_hours: f64, cyclic_block_steps: usize, capacity: usize) -> Profile {
    Profile {
        dt_hours,
        cyclic_block_steps,
        solar_cf: Vec::with_capacity(capacity),
        wind_cf: Vec::with_capacity(capacity),
        load_kw: Vec::with_capacity(capacity),
        import_price: Vec::with_capacity(capacity),
        export_price: Vec::with_capacity(capacity),
        hydrogen_demand_kw: Vec::with_capacity(capacity),
        hydrogen_price: Vec::with_capacity(capacity),
    }
}

/// Representative, independently cyclic days distributed across the year.
#[must_use]
pub fn representative_days(days: usize, modifiers: ProfileModifiers) -> Profile {
    assert!(days > 0);
    let mut profile = empty(1.0, 24, 24 * days);
    for day in 0..days {
        let day_of_year = ((day as f64 + 0.5) * 365.0 / days as f64).floor();
        for hour in 0..24 {
            push_sample(
                &mut profile,
                24.0 * day_of_year + f64::from(hour),
                modifiers,
            );
        }
    }
    profile
}

fn validation_day(dt_hours: f64, modifiers: ProfileModifiers) -> Profile {
    let steps = (24.0 / dt_hours).round() as usize;
    let mut profile = empty(dt_hours, steps, steps);
    let middle_spring_day = 105.0;
    for step in 0..steps {
        push_sample(
            &mut profile,
            24.0 * middle_spring_day + step as f64 * dt_hours,
            modifiers,
        );
    }
    profile
}

/// One fixed validation day at quarter-hour resolution.
#[must_use]
pub fn quarter_hour_day(modifiers: ProfileModifiers) -> Profile {
    validation_day(0.25, modifiers)
}

/// The same fixed validation day at hourly resolution.
#[must_use]
pub fn hourly_validation_day(modifiers: ProfileModifiers) -> Profile {
    validation_day(1.0, modifiers)
}

/// Chronological annual profile sampled at one or six-hour resolution.
#[must_use]
pub fn chronological_year(dt_hours: usize, modifiers: ProfileModifiers) -> Profile {
    assert!(matches!(dt_hours, 1 | 6));
    let steps = 8_760 / dt_hours;
    let mut profile = empty(dt_hours as f64, steps, steps);
    for step in 0..steps {
        let start = step * dt_hours;
        let mut average = [0.0; 7];
        for offset in 0..dt_hours {
            let sample = base_hour((start + offset) as f64, modifiers);
            for (target, value) in average.iter_mut().zip(sample) {
                *target += value / dt_hours as f64;
            }
        }
        profile.solar_cf.push(average[0]);
        profile.wind_cf.push(average[1]);
        profile.load_kw.push(average[2]);
        profile.import_price.push(average[3]);
        profile.export_price.push(average[4]);
        profile.hydrogen_demand_kw.push(average[5]);
        profile.hydrogen_price.push(average[6]);
    }
    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_horizons_and_energy_are_consistent() {
        let smoke = representative_days(4, ProfileModifiers::default());
        let publication = representative_days(12, ProfileModifiers::default());
        let quarter = quarter_hour_day(ProfileModifiers::default());
        let hourly_day = hourly_validation_day(ProfileModifiers::default());
        let coarse_year = chronological_year(6, ProfileModifiers::default());
        let fine_year = chronological_year(1, ProfileModifiers::default());
        assert_eq!(smoke.len(), 96);
        assert_eq!(publication.len(), 288);
        assert_eq!(quarter.len(), 96);
        assert_eq!(hourly_day.len(), 24);
        assert_eq!(coarse_year.len(), 1_460);
        assert_eq!(fine_year.len(), 8_760);
        let relative = (coarse_year.load_energy_kwh() - fine_year.load_energy_kwh()).abs()
            / fine_year.load_energy_kwh();
        assert!(relative < 1.0e-12);
    }
}
