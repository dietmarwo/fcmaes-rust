//! Compact native evaluator for the Mazda three-car response-surface model.

use std::sync::OnceLock;

use crate::mazda::{MAZDA_CONSTRAINTS, MAZDA_DIM};

const MODEL_BYTES: &[u8] = include_bytes!("../data/mazda_model.bin");
const MAGIC: &[u8; 8] = b"FCMAZ01\0";
const CAR_DIM: usize = 74;
const CARS: usize = 3;
const RAW_OBJECTIVES: usize = 5;
const MAX_RBF_SAMPLES: usize = 1_271;

static MODEL: OnceLock<Result<Model, String>> = OnceLock::new();

pub(crate) struct RawEvaluation {
    pub(crate) objectives: [f64; RAW_OBJECTIVES],
    pub(crate) constraints: [f64; MAZDA_CONSTRAINTS],
}

struct Model {
    decisions: Vec<Vec<f64>>,
    masses: Vec<MassModel>,
    groups: Vec<ResponseGroup>,
    #[cfg(test)]
    references: Vec<Reference>,
}

struct MassModel {
    ranges: Vec<(f64, f64)>,
    y_range: (f64, f64),
    coefficients: Vec<f64>,
}

struct ResponseGroup {
    car: usize,
    indices: Vec<usize>,
    ranges: Vec<(f64, f64)>,
    samples: usize,
    points: Vec<f64>,
    responses: Vec<Response>,
}

struct Response {
    output: usize,
    y_range: (f64, f64),
    delta: f64,
    intercept: f64,
    coefficients: Vec<f64>,
}

#[cfg(test)]
struct Reference {
    x: Vec<f64>,
    objectives: [f64; RAW_OBJECTIVES],
    constraints: [f64; MAZDA_CONSTRAINTS],
}

fn model() -> Result<&'static Model, String> {
    match MODEL.get_or_init(|| Model::parse(MODEL_BYTES)) {
        Ok(model) => Ok(model),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn decision_choices() -> Result<Vec<Vec<f64>>, String> {
    Ok(model()?.decisions.clone())
}

pub(crate) fn validate() -> Result<(), String> {
    model().map(|_| ())
}

pub(crate) fn evaluate(physical: &[f64]) -> Result<RawEvaluation, String> {
    if physical.len() != MAZDA_DIM {
        return Err(format!(
            "Mazda physical vector has length {}, expected {MAZDA_DIM}",
            physical.len()
        ));
    }
    if physical.iter().any(|value| !value.is_finite()) {
        return Err("Mazda physical vector contains a non-finite value".to_string());
    }
    model()?.evaluate(physical)
}

impl Model {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err("invalid embedded Mazda model signature".to_string());
        }

        let decision_count = reader.usize()?;
        if decision_count != MAZDA_DIM {
            return Err(format!(
                "embedded Mazda model has {decision_count} decisions, expected {MAZDA_DIM}"
            ));
        }
        let mut decisions = Vec::with_capacity(decision_count);
        for _ in 0..decision_count {
            let count = reader.usize()?;
            if count == 0 {
                return Err("embedded Mazda model contains an empty decision list".to_string());
            }
            decisions.push(reader.f64s(count)?);
        }

        let mass_count = reader.usize()?;
        if mass_count != CARS {
            return Err(format!(
                "embedded Mazda model has {mass_count} mass models, expected {CARS}"
            ));
        }
        let mut masses = Vec::with_capacity(mass_count);
        for _ in 0..mass_count {
            let ranges = reader.ranges(CAR_DIM)?;
            let y_range = reader.range()?;
            let coefficients = reader.f64s(CAR_DIM + 1)?;
            masses.push(MassModel {
                ranges,
                y_range,
                coefficients,
            });
        }

        let group_count = reader.usize()?;
        if group_count == 0 {
            return Err("embedded Mazda model contains no response groups".to_string());
        }
        let mut groups = Vec::with_capacity(group_count);
        let mut generated_outputs = [false; MAZDA_CONSTRAINTS];
        for _ in 0..group_count {
            let car = reader.usize()?;
            let dim = reader.usize()?;
            let samples = reader.usize()?;
            let response_count = reader.usize()?;
            if car >= CARS
                || dim == 0
                || dim > CAR_DIM
                || samples == 0
                || samples > MAX_RBF_SAMPLES
                || response_count == 0
            {
                return Err("invalid embedded Mazda response-group dimensions".to_string());
            }
            let indices = reader
                .take(dim)?
                .iter()
                .map(|&index| index as usize)
                .collect::<Vec<_>>();
            if indices.iter().any(|&index| index >= CAR_DIM) {
                return Err("invalid embedded Mazda response input index".to_string());
            }
            let ranges = reader.ranges(dim)?;
            let points = reader.f64s(
                samples
                    .checked_mul(dim)
                    .ok_or_else(|| "embedded Mazda response matrix size overflow".to_string())?,
            )?;
            let mut responses = Vec::with_capacity(response_count);
            for _ in 0..response_count {
                let output = reader.usize()?;
                if output >= MAZDA_CONSTRAINTS || generated_outputs[output] {
                    return Err("duplicate or invalid embedded Mazda constraint output".to_string());
                }
                generated_outputs[output] = true;
                let y_range = reader.range()?;
                let delta = reader.f64()?;
                let intercept = reader.f64()?;
                let coefficients = reader.f64s(samples)?;
                if !(delta.is_finite() && delta > 0.0) {
                    return Err("invalid embedded Mazda radial scale".to_string());
                }
                responses.push(Response {
                    output,
                    y_range,
                    delta,
                    intercept,
                    coefficients,
                });
            }
            groups.push(ResponseGroup {
                car,
                indices,
                ranges,
                samples,
                points,
                responses,
            });
        }

        // Four simple difference constraints per car are evaluated directly.
        for output in [14, 15, 16, 17, 32, 33, 34, 35, 50, 51, 52, 53] {
            generated_outputs[output] = true;
        }
        if generated_outputs.iter().any(|&present| !present) {
            return Err("embedded Mazda model does not cover all constraints".to_string());
        }

        let reference_count = reader.usize()?;
        #[cfg(test)]
        let mut references = Vec::with_capacity(reference_count);
        for _ in 0..reference_count {
            let x = reader.f64s(MAZDA_DIM)?;
            let objectives: [f64; RAW_OBJECTIVES] = reader
                .f64s(RAW_OBJECTIVES)?
                .try_into()
                .map_err(|_| "invalid Mazda objective reference width".to_string())?;
            let constraints: [f64; MAZDA_CONSTRAINTS] = reader
                .f64s(MAZDA_CONSTRAINTS)?
                .try_into()
                .map_err(|_| "invalid Mazda constraint reference width".to_string())?;
            #[cfg(test)]
            references.push(Reference {
                x,
                objectives,
                constraints,
            });
            #[cfg(not(test))]
            let _ = (x, objectives, constraints);
        }
        if !reader.is_empty() {
            return Err("trailing bytes in embedded Mazda model".to_string());
        }

        Ok(Self {
            decisions,
            masses,
            groups,
            #[cfg(test)]
            references,
        })
    }

    fn evaluate(&self, physical: &[f64]) -> Result<RawEvaluation, String> {
        let cars = [
            &physical[..CAR_DIM],
            &physical[CAR_DIM..2 * CAR_DIM],
            &physical[2 * CAR_DIM..],
        ];
        let mut mass = [0.0; CARS];
        for (output, (model, input)) in mass
            .iter_mut()
            .zip(self.masses.iter().zip(cars.iter().copied()))
        {
            *output = model.evaluate(input);
        }

        let common_parts = (0..CAR_DIM)
            .filter(|&index| {
                let upper = cars[0][index].max(cars[1][index].max(cars[2][index]));
                let lower = cars[0][index].min(cars[1][index].min(cars[2][index]));
                upper - lower < 0.05
            })
            .count() as f64;
        let objectives = [
            mass[0] + mass[1] + mass[2],
            -common_parts,
            mass[0],
            mass[1],
            mass[2],
        ];

        let mut constraints = [0.0; MAZDA_CONSTRAINTS];
        let mut radii = [0.0; MAX_RBF_SAMPLES];
        for group in &self.groups {
            group.evaluate(cars[group.car], &mut radii, &mut constraints);
        }
        for (car, input) in cars.into_iter().enumerate() {
            let output = car * 18 + 14;
            constraints[output] = input[13] - input[12];
            constraints[output + 1] = input[15] - input[14];
            constraints[output + 2] = input[12] - input[63];
            constraints[output + 3] = input[14] - input[63];
        }

        if objectives.iter().any(|value| !value.is_finite())
            || constraints.iter().any(|value| !value.is_finite())
        {
            return Err("native Mazda evaluator produced a non-finite value".to_string());
        }
        Ok(RawEvaluation {
            objectives,
            constraints,
        })
    }
}

impl MassModel {
    fn evaluate(&self, input: &[f64]) -> f64 {
        let mut normalized = [0.0; CAR_DIM];
        for (index, value) in normalized.iter_mut().enumerate() {
            let (lower, upper) = self.ranges[index];
            *value = (input[index] - lower) / (upper - lower);
        }
        let mut scaled = self.coefficients[0];
        for (coefficient, value) in self.coefficients[1..].iter().zip(normalized) {
            scaled += coefficient * value;
        }
        self.y_range.0 + (self.y_range.1 - self.y_range.0) * scaled
    }
}

impl ResponseGroup {
    fn evaluate(
        &self,
        input: &[f64],
        radii: &mut [f64],
        constraints: &mut [f64; MAZDA_CONSTRAINTS],
    ) {
        let mut normalized = [0.0; CAR_DIM];
        for (position, (&index, &(lower, upper))) in
            self.indices.iter().zip(&self.ranges).enumerate()
        {
            normalized[position] = (input[index] - lower) / (upper - lower);
        }

        for (radius, point) in radii[..self.samples]
            .iter_mut()
            .zip(self.points.chunks_exact(self.indices.len()))
        {
            let mut squared = 0.0;
            for (&value, &reference) in normalized[..self.indices.len()].iter().zip(point) {
                let difference = value - reference;
                squared += difference * difference;
            }
            *radius = squared.sqrt();
        }

        for response in &self.responses {
            let mut scaled = 0.0;
            for (&coefficient, &radius) in response.coefficients.iter().zip(&radii[..self.samples])
            {
                let relative = radius / response.delta;
                scaled += coefficient * (1.0 + relative * relative).sqrt();
            }
            scaled += response.intercept;
            constraints[response.output] =
                response.y_range.0 + (response.y_range.1 - response.y_range.0) * scaled;
        }
    }
}

struct Reader<'a> {
    remaining: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        if self.remaining.len() < count {
            return Err("truncated embedded Mazda model".to_string());
        }
        let (value, remaining) = self.remaining.split_at(count);
        self.remaining = remaining;
        Ok(value)
    }

    fn usize(&mut self) -> Result<usize, String> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| "invalid embedded Mazda integer".to_string())?;
        Ok(u32::from_le_bytes(bytes) as usize)
    }

    fn f64(&mut self) -> Result<f64, String> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| "invalid embedded Mazda float".to_string())?;
        let value = f64::from_le_bytes(bytes);
        if !value.is_finite() {
            return Err("non-finite value in embedded Mazda model".to_string());
        }
        Ok(value)
    }

    fn f64s(&mut self, count: usize) -> Result<Vec<f64>, String> {
        (0..count).map(|_| self.f64()).collect()
    }

    fn range(&mut self) -> Result<(f64, f64), String> {
        let lower = self.f64()?;
        let upper = self.f64()?;
        if upper <= lower {
            return Err("invalid range in embedded Mazda model".to_string());
        }
        Ok((lower, upper))
    }

    fn ranges(&mut self, count: usize) -> Result<Vec<(f64, f64)>, String> {
        (0..count).map(|_| self.range()).collect()
    }

    fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_matches_published_reference_rows() {
        let model = model().unwrap();
        assert_eq!(model.references.len(), 3);
        for reference in &model.references {
            let actual = model.evaluate(&reference.x).unwrap();
            for (&value, &expected) in actual.objectives.iter().zip(&reference.objectives) {
                assert!((value - expected).abs() < 1.0e-7, "{value} != {expected}");
            }
            for (&value, &expected) in actual.constraints.iter().zip(&reference.constraints) {
                assert!((value - expected).abs() < 1.0e-6, "{value} != {expected}");
            }
        }
    }

    #[test]
    fn rejects_corrupt_or_truncated_models() {
        assert!(Model::parse(b"not Mazda").is_err());
        assert!(Model::parse(&MODEL_BYTES[..MODEL_BYTES.len() / 2]).is_err());
    }
}
