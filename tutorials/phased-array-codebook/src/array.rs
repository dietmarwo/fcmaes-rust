//! Array geometry and angular grids.

use std::f64::consts::{FRAC_PI_2, PI};

/// Geometry family retained for kernel capability checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrayLayout {
    /// Arbitrary element positions.
    General,
    /// Uniform line along x.
    Linear {
        /// Element count.
        count: usize,
        /// Spacing measured in wavelengths.
        spacing_lambda: f64,
    },
    /// Uniform rectangular array, x varying fastest.
    Rectangular {
        /// X element count.
        nx: usize,
        /// Y element count.
        ny: usize,
        /// X spacing measured in wavelengths.
        dx_lambda: f64,
        /// Y spacing measured in wavelengths.
        dy_lambda: f64,
    },
}

/// Planar array geometry.
#[derive(Clone, Debug)]
pub struct Array {
    /// Element positions `[x,y]` in metres.
    pub positions: Vec<[f64; 2]>,
    /// Wavelength in metres.
    pub wavelength: f64,
    /// Exponent in the idealized `cos(theta)^q` element factor.
    pub element_exponent: f64,
    /// Geometry family.
    pub layout: ArrayLayout,
}

impl Array {
    /// Construct a centered uniform linear array.
    #[must_use]
    pub fn uniform_linear(count: usize, spacing_lambda: f64, wavelength: f64) -> Self {
        assert!(count > 0, "an array needs at least one element");
        assert!(
            spacing_lambda.is_finite() && spacing_lambda > 0.0,
            "spacing must be positive"
        );
        assert!(
            wavelength.is_finite() && wavelength > 0.0,
            "wavelength must be positive"
        );
        let center = (count - 1) as f64 / 2.0;
        let spacing = spacing_lambda * wavelength;
        let positions = (0..count)
            .map(|index| [(index as f64 - center) * spacing, 0.0])
            .collect();
        Self {
            positions,
            wavelength,
            element_exponent: 1.0,
            layout: ArrayLayout::Linear {
                count,
                spacing_lambda,
            },
        }
    }

    /// Construct a centered uniform rectangular array.
    #[must_use]
    pub fn uniform_rectangular(
        nx: usize,
        ny: usize,
        dx_lambda: f64,
        dy_lambda: f64,
        wavelength: f64,
    ) -> Self {
        assert!(nx > 0 && ny > 0, "an array needs at least one element");
        assert!(
            dx_lambda.is_finite() && dy_lambda.is_finite() && dx_lambda > 0.0 && dy_lambda > 0.0,
            "spacing must be positive"
        );
        assert!(
            wavelength.is_finite() && wavelength > 0.0,
            "wavelength must be positive"
        );
        let x_center = (nx - 1) as f64 / 2.0;
        let y_center = (ny - 1) as f64 / 2.0;
        let mut positions = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                positions.push([
                    (ix as f64 - x_center) * dx_lambda * wavelength,
                    (iy as f64 - y_center) * dy_lambda * wavelength,
                ]);
            }
        }
        Self {
            positions,
            wavelength,
            element_exponent: 1.0,
            layout: ArrayLayout::Rectangular {
                nx,
                ny,
                dx_lambda,
                dy_lambda,
            },
        }
    }

    /// Return a copy with a different idealized element exponent.
    #[must_use]
    pub fn with_element_exponent(mut self, exponent: f64) -> Self {
        assert!(
            exponent.is_finite() && exponent >= 0.0,
            "element exponent must be finite and non-negative"
        );
        self.element_exponent = exponent;
        self
    }

    /// Smallest pairwise element spacing in wavelengths.
    #[must_use]
    pub fn minimum_spacing_lambda(&self) -> f64 {
        if self.positions.len() < 2 {
            return f64::INFINITY;
        }
        let mut minimum = f64::INFINITY;
        for left in 0..self.positions.len() {
            for right in left + 1..self.positions.len() {
                let dx = self.positions[left][0] - self.positions[right][0];
                let dy = self.positions[left][1] - self.positions[right][1];
                minimum = minimum.min(dx.hypot(dy) / self.wavelength);
            }
        }
        minimum
    }
}

/// One sampled propagation direction.
#[derive(Clone, Copy, Debug)]
pub struct Direction {
    /// Direction cosine along x.
    pub u: f64,
    /// Direction cosine along y.
    pub v: f64,
    /// Polar angle from +z in radians.
    pub theta_rad: f64,
    /// Azimuth from +x in radians.
    pub phi_rad: f64,
    /// Solid-angle quadrature weight; zero for a 1-D cut.
    pub solid_angle_weight: f64,
}

/// Angular grid kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridKind {
    /// Signed x-z cut uniformly sampled in `u`.
    LinearCut,
    /// Midpoint upper-hemisphere polar quadrature.
    UpperHemisphere {
        /// Polar-angle bins.
        theta_points: usize,
        /// Azimuth bins.
        phi_points: usize,
    },
}

/// Directions used by an array-factor kernel.
#[derive(Clone, Debug)]
pub struct AngleGrid {
    /// Grid family.
    pub kind: GridKind,
    /// Directions in deterministic row-major order.
    pub directions: Vec<Direction>,
}

impl AngleGrid {
    /// Uniform signed x-z cut over `u ∈ [-1,1]`.
    #[must_use]
    pub fn linear_cut(points: usize) -> Self {
        assert!(points >= 3, "a linear cut needs at least three points");
        let denominator = (points - 1) as f64;
        let directions = (0..points)
            .map(|index| {
                let u = -1.0 + 2.0 * index as f64 / denominator;
                let signed_theta = u.clamp(-1.0, 1.0).asin();
                Direction {
                    u,
                    v: 0.0,
                    theta_rad: signed_theta.abs(),
                    phi_rad: if signed_theta < 0.0 { PI } else { 0.0 },
                    solid_angle_weight: 0.0,
                }
            })
            .collect();
        Self {
            kind: GridKind::LinearCut,
            directions,
        }
    }

    /// Midpoint quadrature over the upper hemisphere.
    #[must_use]
    pub fn upper_hemisphere(theta_points: usize, phi_points: usize) -> Self {
        assert!(
            theta_points >= 2 && phi_points >= 4,
            "hemisphere grid is too small"
        );
        let d_theta = FRAC_PI_2 / theta_points as f64;
        let d_phi = 2.0 * PI / phi_points as f64;
        let mut directions = Vec::with_capacity(theta_points * phi_points);
        for theta_index in 0..theta_points {
            let theta = (theta_index as f64 + 0.5) * d_theta;
            let sin_theta = theta.sin();
            for phi_index in 0..phi_points {
                let phi = (phi_index as f64 + 0.5) * d_phi;
                directions.push(Direction {
                    u: sin_theta * phi.cos(),
                    v: sin_theta * phi.sin(),
                    theta_rad: theta,
                    phi_rad: phi,
                    solid_angle_weight: sin_theta * d_theta * d_phi,
                });
            }
        }
        Self {
            kind: GridKind::UpperHemisphere {
                theta_points,
                phi_points,
            },
            directions,
        }
    }

    /// Number of sampled directions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Whether the grid contains no directions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// Signed `u` values for a linear cut.
    #[must_use]
    pub fn linear_u(&self) -> Option<Vec<f64>> {
        (self.kind == GridKind::LinearCut).then(|| {
            self.directions
                .iter()
                .map(|direction| direction.u)
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_geometries_are_centered_and_have_requested_spacing() {
        let ula = Array::uniform_linear(16, 0.5, 0.1);
        assert!(
            ula.positions
                .iter()
                .map(|position| position[0])
                .sum::<f64>()
                .abs()
                < 1.0e-15
        );
        assert!((ula.minimum_spacing_lambda() - 0.5).abs() < 1.0e-14);

        let ura = Array::uniform_rectangular(8, 8, 0.5, 0.5, 0.1);
        assert_eq!(ura.positions.len(), 64);
        assert!((ura.minimum_spacing_lambda() - 0.5).abs() < 1.0e-14);
    }

    #[test]
    fn hemisphere_weights_converge_to_two_pi() {
        let grid = AngleGrid::upper_hemisphere(180, 360);
        let solid_angle = grid
            .directions
            .iter()
            .map(|direction| direction.solid_angle_weight)
            .sum::<f64>();
        assert!((solid_angle - 2.0 * PI).abs() < 1.0e-4);
    }
}
