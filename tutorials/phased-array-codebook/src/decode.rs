//! Authoritative quantized decision decoding.

use std::f64::consts::TAU;
use std::fmt::{Display, Formatter};

use num_complex::Complex64;

use crate::config::Quantization;

/// Invalid quantized control vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Vector length differs from the documented layout.
    Dimension {
        /// Required coordinate/register count.
        expected: usize,
        /// Supplied coordinate/register count.
        actual: usize,
    },
    /// A coordinate or quantization setting is non-finite/invalid.
    NonFinite,
    /// Requested activation cardinality exceeds the array size.
    Cardinality,
}

impl Display for DecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dimension { expected, actual } => {
                write!(
                    formatter,
                    "decision dimension is {actual}, expected {expected}"
                )
            }
            Self::NonFinite => formatter.write_str("quantized controls must be finite"),
            Self::Cardinality => formatter.write_str("activation cardinality exceeds array size"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decoded hardware state.
#[derive(Clone, Debug, PartialEq)]
pub struct Excitation {
    /// Complex element weights, zero for inactive elements.
    pub weights: Vec<Complex64>,
    /// Element activation mask.
    pub active: Vec<bool>,
    /// Phase-shifter register codes.
    pub phase_codes: Vec<u32>,
    /// Attenuator register codes.
    pub attenuator_codes: Vec<u32>,
}

fn levels(bits: u32) -> Result<u32, DecodeError> {
    (bits > 0 && bits < 31)
        .then(|| 1_u32 << bits)
        .ok_or(DecodeError::NonFinite)
}

/// Decode an equal-width normalized coordinate into `0..levels`.
pub fn equal_width_code(value: f64, bits: u32) -> Result<u32, DecodeError> {
    if !value.is_finite() {
        return Err(DecodeError::NonFinite);
    }
    let count = levels(bits)?;
    let unit = value.clamp(0.0, 1.0);
    Ok(((unit * f64::from(count)).floor() as u32).min(count - 1))
}

fn phase_from_code(code: u32, bits: u32) -> Result<f64, DecodeError> {
    let count = levels(bits)?;
    (code < count)
        .then(|| TAU * f64::from(code) / f64::from(count))
        .ok_or(DecodeError::NonFinite)
}

fn amplitude_from_code(code: u32, bits: u32, step_db: f64) -> Result<f64, DecodeError> {
    let count = levels(bits)?;
    if code >= count || !step_db.is_finite() || step_db <= 0.0 {
        return Err(DecodeError::NonFinite);
    }
    Ok(10_f64.powf(-f64::from(code) * step_db / 20.0))
}

/// Decode a phase-shifter coordinate to radians.
pub fn decode_phase(value: f64, bits: u32) -> Result<f64, DecodeError> {
    phase_from_code(equal_width_code(value, bits)?, bits)
}

/// Decode an attenuator coordinate to a linear voltage amplitude.
pub fn decode_amplitude(value: f64, bits: u32, step_db: f64) -> Result<f64, DecodeError> {
    amplitude_from_code(equal_width_code(value, bits)?, bits, step_db)
}

fn validate_quantization(quantization: &Quantization) -> Result<(), DecodeError> {
    let _ = levels(quantization.phase_bits)?;
    let _ = levels(quantization.attenuator_bits)?;
    if !quantization.attenuator_step_db.is_finite() || quantization.attenuator_step_db <= 0.0 {
        return Err(DecodeError::NonFinite);
    }
    Ok(())
}

/// Decode `[phase × N, attenuation × N]`.
pub fn decode_beam(
    x: &[f64],
    array_len: usize,
    quantization: &Quantization,
) -> Result<Excitation, DecodeError> {
    validate_quantization(quantization)?;
    let expected = 2 * array_len;
    if x.len() != expected {
        return Err(DecodeError::Dimension {
            expected,
            actual: x.len(),
        });
    }
    decode_parts(
        &x[..array_len],
        &x[array_len..],
        vec![true; array_len],
        quantization,
    )
}

/// Decode `[phase × N, attenuation × N, activation keys × N]` with exact `k`.
pub fn decode_with_activation(
    x: &[f64],
    array_len: usize,
    active_count: usize,
    quantization: &Quantization,
) -> Result<Excitation, DecodeError> {
    validate_quantization(quantization)?;
    if active_count > array_len {
        return Err(DecodeError::Cardinality);
    }
    let expected = 3 * array_len;
    if x.len() != expected {
        return Err(DecodeError::Dimension {
            expected,
            actual: x.len(),
        });
    }
    let keys = &x[2 * array_len..];
    if keys.iter().any(|value| !value.is_finite()) {
        return Err(DecodeError::NonFinite);
    }
    let mut order = (0..array_len).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        keys[*left]
            .total_cmp(&keys[*right])
            .then_with(|| left.cmp(right))
    });
    let mut active = vec![false; array_len];
    for index in order.into_iter().take(active_count) {
        active[index] = true;
    }
    decode_parts(
        &x[..array_len],
        &x[array_len..2 * array_len],
        active,
        quantization,
    )
}

fn decode_parts(
    phase: &[f64],
    attenuation: &[f64],
    active: Vec<bool>,
    quantization: &Quantization,
) -> Result<Excitation, DecodeError> {
    let mut weights = Vec::with_capacity(phase.len());
    let mut phase_codes = Vec::with_capacity(phase.len());
    let mut attenuator_codes = Vec::with_capacity(phase.len());
    for index in 0..phase.len() {
        let phase_code = equal_width_code(phase[index], quantization.phase_bits)?;
        let attenuator_code = equal_width_code(attenuation[index], quantization.attenuator_bits)?;
        let phase_rad = phase_from_code(phase_code, quantization.phase_bits)?;
        let amplitude = amplitude_from_code(
            attenuator_code,
            quantization.attenuator_bits,
            quantization.attenuator_step_db,
        )?;
        weights.push(if active[index] {
            Complex64::from_polar(amplitude, phase_rad)
        } else {
            Complex64::new(0.0, 0.0)
        });
        phase_codes.push(phase_code);
        attenuator_codes.push(attenuator_code);
    }
    Ok(Excitation {
        weights,
        active,
        phase_codes,
        attenuator_codes,
    })
}

/// Recreate excitation weights from machine register codes.
pub fn from_codes(
    phase_codes: &[u32],
    attenuator_codes: &[u32],
    active: &[bool],
    quantization: &Quantization,
) -> Result<Excitation, DecodeError> {
    validate_quantization(quantization)?;
    if phase_codes.len() != attenuator_codes.len() || phase_codes.len() != active.len() {
        return Err(DecodeError::Dimension {
            expected: phase_codes.len(),
            actual: attenuator_codes.len().min(active.len()),
        });
    }
    let phase_levels = levels(quantization.phase_bits)?;
    let attenuator_levels = levels(quantization.attenuator_bits)?;
    if phase_codes.iter().any(|code| *code >= phase_levels)
        || attenuator_codes
            .iter()
            .any(|code| *code >= attenuator_levels)
    {
        return Err(DecodeError::NonFinite);
    }
    let weights = phase_codes
        .iter()
        .zip(attenuator_codes)
        .zip(active)
        .map(
            |((&phase, &attenuation), &enabled)| -> Result<Complex64, DecodeError> {
                if !enabled {
                    return Ok(Complex64::new(0.0, 0.0));
                }
                Ok(Complex64::from_polar(
                    amplitude_from_code(
                        attenuation,
                        quantization.attenuator_bits,
                        quantization.attenuator_step_db,
                    )?,
                    phase_from_code(phase, quantization.phase_bits)?,
                ))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Excitation {
        weights,
        active: active.to_vec(),
        phase_codes: phase_codes.to_vec(),
        attenuator_codes: attenuator_codes.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use fcmaes_core::{Rng, parallel_batch};

    use super::*;

    #[test]
    fn equal_width_codes_cover_endpoints_and_all_bins_uniformly() {
        assert_eq!(equal_width_code(0.0, 6).unwrap(), 0);
        assert_eq!(equal_width_code(1.0, 6).unwrap(), 63);
        let samples = 1_000_000;
        let mut phase_histogram = [0usize; 64];
        let mut attenuation_histogram = [0usize; 32];
        for index in 0..samples {
            let value = (index as f64 + 0.5) / samples as f64;
            phase_histogram[equal_width_code(value, 6).unwrap() as usize] += 1;
            attenuation_histogram[equal_width_code(value, 5).unwrap() as usize] += 1;
        }
        let expected_phase = samples as f64 / 64.0;
        let expected_attenuation = samples as f64 / 32.0;
        assert!(
            phase_histogram
                .iter()
                .all(|count| { (*count as f64 - expected_phase).abs() / expected_phase < 0.01 })
        );
        assert!(attenuation_histogram.iter().all(|count| {
            (*count as f64 - expected_attenuation).abs() / expected_attenuation < 0.01
        }));
    }

    #[test]
    fn bad_coordinates_fail_and_activation_is_exact_and_sorted_by_key() {
        let quantization = Quantization::default();
        assert_eq!(
            decode_beam(&[f64::NAN; 32], 16, &quantization),
            Err(DecodeError::NonFinite)
        );
        let mut rng = Rng::new(42);
        for active_count in 0..=16 {
            let x = (0..48).map(|_| rng.uniform01()).collect::<Vec<_>>();
            let decoded = decode_with_activation(&x, 16, active_count, &quantization).unwrap();
            assert_eq!(
                decoded.active.iter().filter(|active| **active).count(),
                active_count
            );
        }
    }

    #[test]
    fn serial_and_parallel_decoding_are_identical() {
        let quantization = Quantization::default();
        let mut rng = Rng::new(7);
        let xs = (0..10_000)
            .map(|_| (0..32).map(|_| rng.uniform01()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let serial = xs
            .iter()
            .map(|x| decode_beam(x, 16, &quantization).unwrap())
            .collect::<Vec<_>>();
        let parallel = parallel_batch(&xs, 4, |x| decode_beam(x, 16, &quantization).unwrap());
        assert_eq!(serial, parallel);
    }

    #[test]
    fn a_one_coordinate_sweep_changes_only_at_known_bin_boundaries() {
        let quantization = Quantization::default();
        let mut x = vec![0.0; 32];
        let mut previous = None;
        let mut transitions = Vec::new();
        for sample in 0..=1_280 {
            x[0] = sample as f64 / 1_280.0;
            let code = decode_beam(&x, 16, &quantization).unwrap().phase_codes[0];
            if previous.is_some_and(|value| value != code) {
                transitions.push(x[0]);
            }
            previous = Some(code);
        }
        assert_eq!(transitions.len(), 63);
        for (index, transition) in transitions.iter().enumerate() {
            let expected = (index + 1) as f64 / 64.0;
            assert!((transition - expected).abs() <= 1.0 / 1_280.0);
        }
    }
}
