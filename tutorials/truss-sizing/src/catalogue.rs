//! Circular-hollow-section catalogue and material data.

use std::f64::consts::PI;

/// Young's modulus used by the educational steel model.
pub const STEEL_E_PA: f64 = 210.0e9;
/// Material density.
pub const STEEL_DENSITY_KG_M3: f64 = 7_850.0;
/// Educational allowable tensile/compressive yield stress.
pub const ALLOWABLE_STRESS_PA: f64 = 215.0e6;
/// Indicative cradle-to-gate factor, not a procurement declaration.
pub const CARBON_KG_CO2E_PER_KG: f64 = 1.70;

/// One circular hollow section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    /// Stable designation used in artifacts.
    pub name: &'static str,
    /// Nominal outer diameter in metres.
    pub outer_diameter_m: f64,
    /// Nominal wall thickness in metres.
    pub wall_m: f64,
    /// Cross-sectional area in square metres.
    pub area_m2: f64,
    /// Second moment of area in metres to the fourth power.
    pub inertia_m4: f64,
    /// Radius of gyration in metres.
    pub radius_gyration_m: f64,
    /// Linear mass in kilograms per metre.
    pub mass_kg_m: f64,
    /// Indicative embodied-carbon factor.
    pub carbon_kg_co2e_per_kg: f64,
}

impl Section {
    /// Build a section from nominal circular geometry.
    #[must_use]
    pub fn circular(name: &'static str, diameter_mm: f64, wall_mm: f64) -> Self {
        let outer = diameter_mm * 1.0e-3;
        let wall = wall_mm * 1.0e-3;
        let inner = outer - 2.0 * wall;
        let area = PI * (outer.powi(2) - inner.powi(2)) / 4.0;
        let inertia = PI * (outer.powi(4) - inner.powi(4)) / 64.0;
        Self {
            name,
            outer_diameter_m: outer,
            wall_m: wall,
            area_m2: area,
            inertia_m4: inertia,
            radius_gyration_m: (inertia / area).sqrt(),
            mass_kg_m: area * STEEL_DENSITY_KG_M3,
            carbon_kg_co2e_per_kg: CARBON_KG_CO2E_PER_KG,
        }
    }
}

/// The twelve nominal sections used by the tutorial.
#[must_use]
pub fn sections() -> [Section; 12] {
    [
        Section::circular("CHS33.7x2.6", 33.7, 2.6),
        Section::circular("CHS42.4x2.6", 42.4, 2.6),
        Section::circular("CHS48.3x3.2", 48.3, 3.2),
        Section::circular("CHS60.3x3.2", 60.3, 3.2),
        Section::circular("CHS76.1x3.6", 76.1, 3.6),
        Section::circular("CHS88.9x4.0", 88.9, 4.0),
        Section::circular("CHS101.6x4.0", 101.6, 4.0),
        Section::circular("CHS114.3x4.5", 114.3, 4.5),
        Section::circular("CHS139.7x5.0", 139.7, 5.0),
        Section::circular("CHS168.3x6.3", 168.3, 6.3),
        Section::circular("CHS193.7x6.3", 193.7, 6.3),
        Section::circular("CHS219.1x8.0", 219.1, 8.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_geometry_is_self_consistent() {
        for section in sections() {
            let reconstructed = (section.inertia_m4 / section.area_m2).sqrt();
            assert!((reconstructed - section.radius_gyration_m).abs() < 1.0e-15);
            assert!((section.mass_kg_m - section.area_m2 * STEEL_DENSITY_KG_M3).abs() < 1.0e-12);
            assert!(section.wall_m * 2.0 < section.outer_diameter_m);
        }
    }

    #[test]
    fn catalogue_is_strictly_increasing_by_area() {
        let catalogue = sections();
        assert!(
            catalogue
                .windows(2)
                .all(|pair| pair[0].area_m2 < pair[1].area_m2)
        );
    }
}
