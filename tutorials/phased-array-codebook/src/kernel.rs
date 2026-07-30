//! Direct-sum and optional FFT array-factor kernels.

use std::f64::consts::PI;
use std::fmt::{Display, Formatter};

use num_complex::Complex64;

use crate::array::{AngleGrid, Array};
#[cfg(feature = "fft")]
use crate::array::{ArrayLayout, Direction, GridKind};

/// Kernel setup/evaluation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    /// Excitation and array dimensions differ.
    Dimension,
    /// FFT path requested for a geometry it cannot represent exactly.
    NonUniformSpacing,
    /// Output and angular-grid dimensions differ.
    OutputDimension,
}

impl Display for KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dimension => formatter.write_str("excitation dimension does not match array"),
            Self::NonUniformSpacing => {
                formatter.write_str("FFT kernel requires a uniform half-wave array")
            }
            Self::OutputDimension => formatter.write_str("output dimension does not match grid"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Precomputed `EF(theta) exp(j psi_n)` matrix.
#[derive(Clone, Debug)]
pub struct SteeringMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl SteeringMatrix {
    /// Build a direction-by-element steering matrix.
    #[must_use]
    pub fn build(array: &Array, grid: &AngleGrid) -> Self {
        let wave_number = 2.0 * PI / array.wavelength;
        let mut data = Vec::with_capacity(grid.len() * array.positions.len());
        for direction in &grid.directions {
            let element_factor = direction
                .theta_rad
                .cos()
                .max(0.0)
                .powf(array.element_exponent);
            for position in &array.positions {
                let phase = wave_number * (position[0] * direction.u + position[1] * direction.v);
                data.push(Complex64::from_polar(element_factor, phase));
            }
        }
        Self {
            rows: grid.len(),
            cols: array.positions.len(),
            data,
        }
    }

    /// Matrix allocation in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<Complex64>()
    }

    /// Evaluate all directions without allocation.
    pub fn field_direct(
        &self,
        excitation: &[Complex64],
        out: &mut [Complex64],
    ) -> Result<(), KernelError> {
        if excitation.len() != self.cols {
            return Err(KernelError::Dimension);
        }
        if out.len() != self.rows {
            return Err(KernelError::OutputDimension);
        }
        for (row, output) in out.iter_mut().enumerate() {
            *output = self.data[row * self.cols..(row + 1) * self.cols]
                .iter()
                .zip(excitation)
                .map(|(steering, weight)| steering * weight)
                .sum();
        }
        Ok(())
    }
}

/// Evaluate one field without precomputation, used by the geometry arm.
pub fn field_direct(
    array: &Array,
    grid: &AngleGrid,
    excitation: &[Complex64],
) -> Result<Vec<Complex64>, KernelError> {
    let matrix = SteeringMatrix::build(array, grid);
    let mut field = vec![Complex64::new(0.0, 0.0); grid.len()];
    matrix.field_direct(excitation, &mut field)?;
    Ok(field)
}

#[cfg(feature = "fft")]
enum FftKind {
    Linear {
        count: usize,
        length: usize,
    },
    Planar {
        nx: usize,
        ny: usize,
        fft_nx: usize,
        fft_ny: usize,
    },
}

/// Exact DFT-node kernel for uniform half-wave arrays.
///
/// It intentionally does not interpolate onto arbitrary angular grids.
#[cfg(feature = "fft")]
pub struct FftKernel {
    kind: FftKind,
    x_fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
    y_fft: Option<std::sync::Arc<dyn rustfft::Fft<f64>>>,
}

#[cfg(feature = "fft")]
impl FftKernel {
    /// Construct a 1-D inverse-DFT kernel.
    pub fn linear(array: &Array, length: usize) -> Result<Self, KernelError> {
        let ArrayLayout::Linear {
            count,
            spacing_lambda,
        } = array.layout
        else {
            return Err(KernelError::NonUniformSpacing);
        };
        if (spacing_lambda - 0.5).abs() > 1.0e-14 || array.element_exponent != 0.0 || length < count
        {
            return Err(KernelError::NonUniformSpacing);
        }
        let mut planner = rustfft::FftPlanner::new();
        Ok(Self {
            kind: FftKind::Linear { count, length },
            x_fft: planner.plan_fft_inverse(length),
            y_fft: None,
        })
    }

    /// Construct a 2-D inverse-DFT kernel.
    pub fn planar(array: &Array, fft_nx: usize, fft_ny: usize) -> Result<Self, KernelError> {
        let ArrayLayout::Rectangular {
            nx,
            ny,
            dx_lambda,
            dy_lambda,
        } = array.layout
        else {
            return Err(KernelError::NonUniformSpacing);
        };
        if (dx_lambda - 0.5).abs() > 1.0e-14
            || (dy_lambda - 0.5).abs() > 1.0e-14
            || array.element_exponent != 0.0
            || fft_nx < nx
            || fft_ny < ny
        {
            return Err(KernelError::NonUniformSpacing);
        }
        let mut planner = rustfft::FftPlanner::new();
        Ok(Self {
            kind: FftKind::Planar {
                nx,
                ny,
                fft_nx,
                fft_ny,
            },
            x_fft: planner.plan_fft_inverse(fft_nx),
            y_fft: Some(planner.plan_fft_inverse(fft_ny)),
        })
    }

    /// Native FFT-node directions in output order.
    #[must_use]
    pub fn native_grid(&self) -> AngleGrid {
        match self.kind {
            FftKind::Linear { length, .. } => AngleGrid {
                kind: GridKind::LinearCut,
                directions: (0..length)
                    .map(|index| {
                        let u = -1.0 + 2.0 * index as f64 / length as f64;
                        let signed_theta = u.asin();
                        Direction {
                            u,
                            v: 0.0,
                            theta_rad: signed_theta.abs(),
                            phi_rad: if signed_theta < 0.0 { PI } else { 0.0 },
                            solid_angle_weight: 0.0,
                        }
                    })
                    .collect(),
            },
            FftKind::Planar { fft_nx, fft_ny, .. } => {
                let mut directions = Vec::with_capacity(fft_nx * fft_ny);
                for iy in 0..fft_ny {
                    let v = -1.0 + 2.0 * iy as f64 / fft_ny as f64;
                    for ix in 0..fft_nx {
                        let u = -1.0 + 2.0 * ix as f64 / fft_nx as f64;
                        let radius = u.hypot(v);
                        directions.push(Direction {
                            u,
                            v,
                            theta_rad: radius.min(1.0).asin(),
                            phi_rad: v.atan2(u),
                            solid_angle_weight: 0.0,
                        });
                    }
                }
                AngleGrid {
                    kind: GridKind::UpperHemisphere {
                        theta_points: fft_ny,
                        phi_points: fft_nx,
                    },
                    directions,
                }
            }
        }
    }

    /// Evaluate at native FFT nodes.
    pub fn field(&self, excitation: &[Complex64]) -> Result<Vec<Complex64>, KernelError> {
        match self.kind {
            FftKind::Linear { count, length } => {
                if excitation.len() != count {
                    return Err(KernelError::Dimension);
                }
                let mut buffer = vec![Complex64::new(0.0, 0.0); length];
                for (index, value) in excitation.iter().enumerate() {
                    buffer[index] = if index.is_multiple_of(2) {
                        *value
                    } else {
                        -*value
                    };
                }
                self.x_fft.process(&mut buffer);
                for (index, value) in buffer.iter_mut().enumerate() {
                    let u = -1.0 + 2.0 * index as f64 / length as f64;
                    *value *= Complex64::from_polar(1.0, -PI * (count - 1) as f64 * u / 2.0);
                }
                Ok(buffer)
            }
            FftKind::Planar {
                nx,
                ny,
                fft_nx,
                fft_ny,
            } => {
                if excitation.len() != nx * ny {
                    return Err(KernelError::Dimension);
                }
                let mut buffer = vec![Complex64::new(0.0, 0.0); fft_nx * fft_ny];
                for iy in 0..ny {
                    for ix in 0..nx {
                        let value = excitation[iy * nx + ix];
                        buffer[iy * fft_nx + ix] = if (ix + iy).is_multiple_of(2) {
                            value
                        } else {
                            -value
                        };
                    }
                }
                for row in buffer.chunks_exact_mut(fft_nx) {
                    self.x_fft.process(row);
                }
                let y_fft = self.y_fft.as_ref().expect("planar kernel owns a y FFT");
                let mut column = vec![Complex64::new(0.0, 0.0); fft_ny];
                for ix in 0..fft_nx {
                    for iy in 0..fft_ny {
                        column[iy] = buffer[iy * fft_nx + ix];
                    }
                    y_fft.process(&mut column);
                    for iy in 0..fft_ny {
                        buffer[iy * fft_nx + ix] = column[iy];
                    }
                }
                for iy in 0..fft_ny {
                    let v = -1.0 + 2.0 * iy as f64 / fft_ny as f64;
                    for ix in 0..fft_nx {
                        let u = -1.0 + 2.0 * ix as f64 / fft_nx as f64;
                        buffer[iy * fft_nx + ix] *= Complex64::from_polar(
                            1.0,
                            -PI * ((nx - 1) as f64 * u + (ny - 1) as f64 * v) / 2.0,
                        );
                    }
                }
                Ok(buffer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(count: usize) -> Vec<Complex64> {
        vec![Complex64::new(1.0, 0.0); count]
    }

    #[test]
    fn broadside_and_dirichlet_closed_form_match() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(4_001);
        let field = field_direct(&array, &grid, &unit(16)).unwrap();
        let broadside = grid
            .directions
            .iter()
            .position(|direction| direction.u.abs() < 1.0e-15)
            .unwrap();
        assert!((field[broadside].norm() - 16.0).abs() < 1.0e-12);
        for (direction, actual) in grid.directions.iter().zip(field) {
            let psi = PI * direction.u;
            let expected = if psi.abs() < 1.0e-14 {
                16.0
            } else {
                (8.0 * psi).sin().abs() / (0.5 * psi).sin().abs()
            };
            assert!(
                (actual.norm() - expected).abs() <= 2.0e-11 * expected.max(1.0),
                "u={} actual={} expected={expected}",
                direction.u,
                actual.norm()
            );
        }
    }

    #[test]
    fn half_wave_visible_region_has_exact_energy_identity() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(4_001);
        let tapers = [
            vec![1.0; 16],
            (0..16)
                .map(|index| 0.4 - 0.6 * (2.0 * PI * index as f64 / 15.0).cos())
                .collect(),
            (0..16).map(|index| 0.25 + index as f64 / 20.0).collect(),
        ];
        let du = 2.0 / (grid.len() - 1) as f64;
        for taper in tapers {
            let excitation = taper
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            let field = field_direct(&array, &grid, &excitation).unwrap();
            let integral = field
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let endpoint = usize::from(index == 0 || index + 1 == field.len());
                    let weight = if endpoint == 1 { 0.5 } else { 1.0 };
                    weight * value.norm_sqr()
                })
                .sum::<f64>()
                * du
                / 2.0;
            let expected = taper.iter().map(|value| value * value).sum::<f64>();
            assert!((integral - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn symmetric_taper_is_symmetric_and_progressive_phase_steers() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(4_001);
        let taper = [
            0.2, 0.3, 0.45, 0.6, 0.75, 0.86, 0.94, 1.0, 1.0, 0.94, 0.86, 0.75, 0.6, 0.45, 0.3, 0.2,
        ];
        let symmetric = field_direct(
            &array,
            &grid,
            &taper
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for index in 0..symmetric.len() {
            assert!(
                (symmetric[index].norm() - symmetric[symmetric.len() - 1 - index].norm()).abs()
                    < 1.0e-13
            );
        }
        for target in [0.0, 0.25, 0.5, 0.7] {
            let excitation = (0..16)
                .map(|index| Complex64::from_polar(1.0, -PI * index as f64 * target))
                .collect::<Vec<_>>();
            let field = field_direct(&array, &grid, &excitation).unwrap();
            let peak = field
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.norm_sqr().total_cmp(&right.1.norm_sqr()))
                .unwrap()
                .0;
            assert!((grid.directions[peak].u - target).abs() <= 2.0 / 4_000.0);
        }
    }

    #[test]
    fn first_uniform_null_is_at_two_over_n() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(4_001);
        let field = field_direct(&array, &grid, &unit(16)).unwrap();
        let target = 0.125;
        let nearest = grid
            .directions
            .iter()
            .enumerate()
            .min_by(|left, right| {
                (left.1.u - target)
                    .abs()
                    .total_cmp(&(right.1.u - target).abs())
            })
            .unwrap()
            .0;
        assert!(field[nearest].norm() < 1.0e-12);
        assert!((grid.directions[nearest].theta_rad.to_degrees() - 7.180_755_78).abs() < 0.04);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn fft_and_direct_sum_match_at_native_linear_nodes() {
        use fcmaes_core::Rng;

        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let fft = FftKernel::linear(&array, 256).unwrap();
        let grid = fft.native_grid();
        let mut rng = Rng::new(42);
        for _ in 0..200 {
            let excitation = (0..16)
                .map(|_| Complex64::from_polar(rng.uniform01(), 2.0 * PI * rng.uniform01()))
                .collect::<Vec<_>>();
            let direct = field_direct(&array, &grid, &excitation).unwrap();
            let transformed = fft.field(&excitation).unwrap();
            for (left, right) in direct.iter().zip(transformed) {
                assert!((*left - right).norm() <= 5.0e-12 * left.norm().max(1.0));
            }
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    fn fft_and_direct_sum_match_at_native_planar_nodes() {
        use fcmaes_core::Rng;

        let array = Array::uniform_rectangular(8, 8, 0.5, 0.5, 1.0).with_element_exponent(0.0);
        let fft = FftKernel::planar(&array, 32, 32).unwrap();
        let grid = fft.native_grid();
        let mut rng = Rng::new(7);
        for _ in 0..20 {
            let excitation = (0..64)
                .map(|_| Complex64::from_polar(rng.uniform01(), 2.0 * PI * rng.uniform01()))
                .collect::<Vec<_>>();
            let direct = field_direct(&array, &grid, &excitation).unwrap();
            let transformed = fft.field(&excitation).unwrap();
            for (left, right) in direct.iter().zip(transformed) {
                assert!((*left - right).norm() <= 1.0e-11 * left.norm().max(1.0));
            }
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    fn fft_rejects_nonuniform_geometry() {
        let mut array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        array.positions[3][0] += 0.01;
        array.layout = crate::array::ArrayLayout::General;
        assert!(matches!(
            FftKernel::linear(&array, 256),
            Err(KernelError::NonUniformSpacing)
        ));
    }
}
