use std::f64::consts::PI;

pub const DIMENSION: usize = 12;
pub const OBJECTIVES: usize = 3;
pub const DETECTION_LIMIT_UG_M3: f64 = 0.02;
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [-1_800.0, -1_800.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [1_800.0, 1_800.0];

pub const DESIGN_NAMES: [&str; DIMENSION] = [
    "source_1_x_m",
    "source_1_y_m",
    "source_1_log10_emission_g_s",
    "source_1_height_m",
    "source_2_x_m",
    "source_2_y_m",
    "source_2_log10_emission_g_s",
    "source_2_height_m",
    "wind_direction_bias_deg",
    "wind_speed_scale",
    "lateral_dispersion_scale",
    "vertical_dispersion_scale",
];

const LOWER: [f64; DIMENSION] = [
    -1_800.0, -1_800.0, -1.0, 20.0, -1_800.0, -1_800.0, -1.0, 20.0, -15.0, 0.80, 0.70, 0.70,
];
const UPPER: [f64; DIMENSION] = [
    1_800.0, 1_800.0, 1.0, 120.0, 1_800.0, 1_800.0, 1.0, 120.0, 15.0, 1.20, 1.30, 1.30,
];

const TRAINING_SENSORS: [Sensor; 16] = [
    Sensor::new(0, -2_200.0, -1_600.0, 2.0),
    Sensor::new(1, -1_200.0, -1_900.0, 2.0),
    Sensor::new(2, 0.0, -2_100.0, 2.0),
    Sensor::new(3, 1_250.0, -1_850.0, 2.0),
    Sensor::new(4, 2_150.0, -1_250.0, 2.0),
    Sensor::new(5, -2_100.0, -450.0, 2.0),
    Sensor::new(6, -1_050.0, -650.0, 2.0),
    Sensor::new(7, 150.0, -750.0, 2.0),
    Sensor::new(8, 1_350.0, -500.0, 2.0),
    Sensor::new(9, 2_250.0, 150.0, 2.0),
    Sensor::new(10, -1_850.0, 850.0, 2.0),
    Sensor::new(11, -750.0, 550.0, 2.0),
    Sensor::new(12, 450.0, 650.0, 2.0),
    Sensor::new(13, 1_650.0, 800.0, 2.0),
    Sensor::new(14, -850.0, 1_850.0, 2.0),
    Sensor::new(15, 1_000.0, 1_950.0, 2.0),
];

const VALIDATION_SENSORS: [Sensor; 10] = [
    Sensor::new(100, -2_350.0, -850.0, 2.0),
    Sensor::new(101, -1_550.0, -1_250.0, 2.0),
    Sensor::new(102, -350.0, -1_450.0, 2.0),
    Sensor::new(103, 850.0, -1_300.0, 2.0),
    Sensor::new(104, 1_850.0, -850.0, 2.0),
    Sensor::new(105, -1_450.0, 150.0, 2.0),
    Sensor::new(106, -250.0, 50.0, 2.0),
    Sensor::new(107, 1_050.0, 200.0, 2.0),
    Sensor::new(108, -1_250.0, 1_350.0, 2.0),
    Sensor::new(109, 1_350.0, 1_450.0, 2.0),
];

const TRAINING_WEATHER: [Weather; 16] = [
    Weather::new(0, 3.2, 15.0, b'C'),
    Weather::new(1, 4.8, 38.0, b'D'),
    Weather::new(2, 2.7, 62.0, b'B'),
    Weather::new(3, 5.4, 88.0, b'D'),
    Weather::new(4, 3.9, 112.0, b'C'),
    Weather::new(5, 6.1, 137.0, b'D'),
    Weather::new(6, 2.9, 161.0, b'E'),
    Weather::new(7, 4.5, 184.0, b'C'),
    Weather::new(8, 5.8, 207.0, b'D'),
    Weather::new(9, 3.4, 231.0, b'B'),
    Weather::new(10, 4.1, 254.0, b'C'),
    Weather::new(11, 6.4, 278.0, b'D'),
    Weather::new(12, 2.6, 301.0, b'E'),
    Weather::new(13, 5.1, 323.0, b'C'),
    Weather::new(14, 3.7, 341.0, b'D'),
    Weather::new(15, 4.3, 354.0, b'C'),
];

const VALIDATION_WEATHER: [Weather; 10] = [
    Weather::new(100, 3.6, 26.0, b'D'),
    Weather::new(101, 5.2, 73.0, b'C'),
    Weather::new(102, 2.8, 103.0, b'E'),
    Weather::new(103, 4.6, 149.0, b'B'),
    Weather::new(104, 6.0, 196.0, b'D'),
    Weather::new(105, 3.1, 219.0, b'C'),
    Weather::new(106, 5.6, 267.0, b'D'),
    Weather::new(107, 2.5, 289.0, b'E'),
    Weather::new(108, 4.9, 316.0, b'C'),
    Weather::new(109, 3.8, 347.0, b'D'),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Source {
    pub x_m: f64,
    pub y_m: f64,
    pub emission_g_s: f64,
    pub height_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sensor {
    pub id: usize,
    pub x_m: f64,
    pub y_m: f64,
    pub height_m: f64,
}

impl Sensor {
    pub const fn new(id: usize, x_m: f64, y_m: f64, height_m: f64) -> Self {
        Self {
            id,
            x_m,
            y_m,
            height_m,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weather {
    pub id: usize,
    pub speed_m_s: f64,
    pub direction_deg: f64,
    pub stability: u8,
}

impl Weather {
    pub const fn new(id: usize, speed_m_s: f64, direction_deg: f64, stability: u8) -> Self {
        Self {
            id,
            speed_m_s,
            direction_deg,
            stability,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Training,
    Validation,
}

impl Split {
    pub fn name(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Validation => "validation",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub split: Split,
    pub sensor: Sensor,
    pub weather: Weather,
    pub measured_ug_m3: f64,
}

#[derive(Clone, Debug)]
pub struct Design {
    values: [f64; DIMENSION],
    sources: [Source; 2],
}

impl Design {
    pub fn from_slice(values: &[f64]) -> Result<Self, &'static str> {
        if values.len() != DIMENSION {
            return Err("a dispersion design must contain exactly twelve values");
        }
        if values.iter().enumerate().any(|(index, value)| {
            !value.is_finite() || *value < LOWER[index] || *value > UPPER[index]
        }) {
            return Err("dispersion parameters lie outside the supported bounds");
        }
        let mut first = Source {
            x_m: values[0],
            y_m: values[1],
            emission_g_s: 10.0_f64.powf(values[2]),
            height_m: values[3],
        };
        let mut second = Source {
            x_m: values[4],
            y_m: values[5],
            emission_g_s: 10.0_f64.powf(values[6]),
            height_m: values[7],
        };
        if (second.x_m, second.y_m) < (first.x_m, first.y_m) {
            std::mem::swap(&mut first, &mut second);
        }
        let canonical = [
            first.x_m,
            first.y_m,
            first.emission_g_s.log10(),
            first.height_m,
            second.x_m,
            second.y_m,
            second.emission_g_s.log10(),
            second.height_m,
            values[8],
            values[9],
            values[10],
            values[11],
        ];
        Ok(Self {
            values: canonical,
            sources: [first, second],
        })
    }

    pub fn baseline() -> Self {
        Self::from_slice(&[
            -600.0, -300.0, 0.0, 60.0, 600.0, 300.0, 0.0, 60.0, 0.0, 1.0, 1.0, 1.0,
        ])
        .expect("baseline lies inside the bounds")
    }

    pub fn truth() -> Self {
        Self::from_slice(&[
            -820.0,
            420.0,
            2.35_f64.log10(),
            52.0,
            930.0,
            -610.0,
            1.25_f64.log10(),
            78.0,
            3.0,
            1.04,
            1.07,
            0.94,
        ])
        .expect("truth lies inside the bounds")
    }

    pub fn values(&self) -> &[f64; DIMENSION] {
        &self.values
    }

    pub fn sources(&self) -> &[Source; 2] {
        &self.sources
    }

    pub fn wind_direction_bias_deg(&self) -> f64 {
        self.values[8]
    }

    pub fn wind_speed_scale(&self) -> f64 {
        self.values[9]
    }

    pub fn lateral_dispersion_scale(&self) -> f64 {
        self.values[10]
    }

    pub fn vertical_dispersion_scale(&self) -> f64 {
        self.values[11]
    }

    pub fn total_emission_g_s(&self) -> f64 {
        self.sources.iter().map(|source| source.emission_g_s).sum()
    }

    pub fn emission_centroid(&self) -> [f64; 2] {
        let total = self.total_emission_g_s().max(f64::MIN_POSITIVE);
        [
            self.sources
                .iter()
                .map(|source| source.x_m * source.emission_g_s)
                .sum::<f64>()
                / total,
            self.sources
                .iter()
                .map(|source| source.y_m * source.emission_g_s)
                .sum::<f64>()
                / total,
        ]
    }
}

impl Default for Design {
    fn default() -> Self {
        Self::baseline()
    }
}

pub fn lower_bounds() -> [f64; DIMENSION] {
    LOWER
}

pub fn upper_bounds() -> [f64; DIMENSION] {
    UPPER
}

#[derive(Clone, Debug)]
pub struct Dataset {
    training: Vec<Observation>,
    validation: Vec<Observation>,
    truth: Design,
}

impl Dataset {
    pub fn synthetic() -> Self {
        let truth = Design::truth();
        let training = synthesize(
            Split::Training,
            &TRAINING_SENSORS,
            &TRAINING_WEATHER,
            &truth,
        );
        let validation = synthesize(
            Split::Validation,
            &VALIDATION_SENSORS,
            &VALIDATION_WEATHER,
            &truth,
        );
        Self {
            training,
            validation,
            truth,
        }
    }

    pub fn training(&self) -> &[Observation] {
        &self.training
    }

    pub fn validation(&self) -> &[Observation] {
        &self.validation
    }

    pub fn truth(&self) -> &Design {
        &self.truth
    }

    pub fn observations(&self, split: Split) -> &[Observation] {
        match split {
            Split::Training => self.training(),
            Split::Validation => self.validation(),
        }
    }

    pub fn weather(&self, split: Split) -> &'static [Weather] {
        match split {
            Split::Training => &TRAINING_WEATHER,
            Split::Validation => &VALIDATION_WEATHER,
        }
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::synthetic()
    }
}

fn synthesize(
    split: Split,
    sensors: &[Sensor],
    weather: &[Weather],
    truth: &Design,
) -> Vec<Observation> {
    let mut observations = Vec::with_capacity(sensors.len() * weather.len());
    for (weather_index, &hour) in weather.iter().enumerate() {
        for (sensor_index, &sensor) in sensors.iter().enumerate() {
            let phase = (weather_index * 37 + sensor_index * 17 + 11) as f64;
            let lateral_mismatch = 1.0 + 0.055 * (phase * 0.73).sin();
            let vertical_mismatch = 1.0 + 0.045 * (phase * 1.13).cos();
            let mut measured = predict_with_scales(
                truth,
                sensor,
                hour,
                truth.lateral_dispersion_scale() * lateral_mismatch,
                truth.vertical_dispersion_scale() * vertical_mismatch,
            );
            let relative_noise = 0.075 * (phase * 1.618_033_988_75).sin();
            let background = 0.004 * (1.0 + (phase * 0.41).cos());
            measured = (measured * (1.0 + relative_noise) + background).max(0.0);
            if measured < DETECTION_LIMIT_UG_M3 {
                measured = 0.0;
            }
            observations.push(Observation {
                split,
                sensor,
                weather: hour,
                measured_ug_m3: measured,
            });
        }
    }
    observations
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub mean_huber_error: f64,
    pub p95_log_error: f64,
    pub detection_mismatch_fraction: f64,
    pub total_emission_g_s: f64,
    pub scalar_score: f64,
    pub source_position_error_m: f64,
    pub observations: usize,
}

impl Metrics {
    pub fn objectives(&self) -> [f64; OBJECTIVES] {
        [
            self.mean_huber_error,
            self.p95_log_error + 0.5 * self.detection_mismatch_fraction,
            self.total_emission_g_s,
        ]
    }
}

pub fn evaluate_training(values: &[f64], dataset: &Dataset) -> Result<Metrics, &'static str> {
    evaluate(values, dataset, Split::Training)
}

pub fn evaluate_validation(values: &[f64], dataset: &Dataset) -> Result<Metrics, &'static str> {
    evaluate(values, dataset, Split::Validation)
}

fn evaluate(values: &[f64], dataset: &Dataset, split: Split) -> Result<Metrics, &'static str> {
    let design = Design::from_slice(values)?;
    let observations = dataset.observations(split);
    let mut huber_sum = 0.0;
    let mut absolute_log_errors = Vec::with_capacity(observations.len());
    let mut detection_mismatches = 0usize;
    for observation in observations {
        let predicted = predict(&design, observation.sensor, observation.weather);
        let predicted_log = (predicted / DETECTION_LIMIT_UG_M3).ln_1p();
        let measured_log = (observation.measured_ug_m3 / DETECTION_LIMIT_UG_M3).ln_1p();
        let residual = predicted_log - measured_log;
        let absolute = residual.abs();
        huber_sum += if absolute <= 0.5 {
            0.5 * residual * residual
        } else {
            0.5 * (absolute - 0.25)
        };
        absolute_log_errors.push(absolute);
        let predicted_detected = predicted >= DETECTION_LIMIT_UG_M3;
        let measured_detected = observation.measured_ug_m3 >= DETECTION_LIMIT_UG_M3;
        detection_mismatches += usize::from(predicted_detected != measured_detected);
    }
    absolute_log_errors.sort_by(f64::total_cmp);
    let p95_index = ((absolute_log_errors.len() - 1) as f64 * 0.95).round() as usize;
    let count = observations.len().max(1);
    let mean_huber_error = huber_sum / count as f64;
    let p95_log_error = absolute_log_errors[p95_index];
    let detection_mismatch_fraction = detection_mismatches as f64 / count as f64;
    let total_emission_g_s = design.total_emission_g_s();
    let scalar_score = mean_huber_error
        + 0.35 * p95_log_error
        + 0.8 * detection_mismatch_fraction
        + 0.01 * total_emission_g_s;
    let source_position_error_m = design
        .sources()
        .iter()
        .zip(dataset.truth().sources())
        .map(|(actual, truth)| (actual.x_m - truth.x_m).hypot(actual.y_m - truth.y_m))
        .sum::<f64>()
        / 2.0;
    Ok(Metrics {
        mean_huber_error,
        p95_log_error,
        detection_mismatch_fraction,
        total_emission_g_s,
        scalar_score,
        source_position_error_m,
        observations: observations.len(),
    })
}

pub fn scalar_objective(values: &[f64], dataset: &Dataset) -> f64 {
    evaluate_training(values, dataset).map_or(1.0e99, |metrics| metrics.scalar_score)
}

pub fn multi_objective(values: &[f64], dataset: &Dataset) -> Vec<f64> {
    evaluate_training(values, dataset).map_or_else(
        |_| vec![1.0e99; OBJECTIVES],
        |metrics| metrics.objectives().to_vec(),
    )
}

pub fn qd_objective(values: &[f64], dataset: &Dataset) -> (f64, [f64; 2]) {
    let Ok(design) = Design::from_slice(values) else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    let Ok(metrics) = evaluate_training(design.values(), dataset) else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    (metrics.scalar_score, design.emission_centroid())
}

pub fn concentration_ug_m3(design: &Design, sensor: Sensor, weather: Weather) -> f64 {
    predict(design, sensor, weather)
}

fn predict(design: &Design, sensor: Sensor, weather: Weather) -> f64 {
    predict_with_scales(
        design,
        sensor,
        weather,
        design.lateral_dispersion_scale(),
        design.vertical_dispersion_scale(),
    )
}

fn predict_with_scales(
    design: &Design,
    sensor: Sensor,
    weather: Weather,
    lateral_scale: f64,
    vertical_scale: f64,
) -> f64 {
    design
        .sources()
        .iter()
        .map(|source| {
            source_concentration_ug_m3(
                *source,
                sensor,
                weather,
                design.wind_direction_bias_deg(),
                design.wind_speed_scale(),
                lateral_scale,
                vertical_scale,
            )
        })
        .sum()
}

fn source_concentration_ug_m3(
    source: Source,
    sensor: Sensor,
    weather: Weather,
    direction_bias_deg: f64,
    speed_scale: f64,
    lateral_scale: f64,
    vertical_scale: f64,
) -> f64 {
    let reference_speed = weather.speed_m_s * speed_scale;
    let stack_speed =
        effective_wind_speed(reference_speed, source.height_m, 10.0, weather.stability);
    if stack_speed <= 0.5 {
        return 0.0;
    }
    let direction = (weather.direction_deg + direction_bias_deg).to_radians();
    let (mut downwind_km, crosswind_m) = wind_components(
        sensor.x_m,
        sensor.y_m,
        source.x_m,
        source.y_m,
        direction.sin(),
        direction.cos(),
    );
    let (rise_m, rise_offset_m) =
        plume_rise(stack_speed, 10.0, 0.5, 333.15, 293.15, weather.stability);
    downwind_km -= rise_offset_m / 1_000.0;
    if downwind_km <= 0.0 {
        return 0.0;
    }
    let sigma_y_m = get_sigma_y(weather.stability, downwind_km) * lateral_scale;
    let sigma_z_m = get_sigma_z(weather.stability, downwind_km) * vertical_scale;
    if !sigma_y_m.is_finite() || !sigma_z_m.is_finite() || sigma_y_m <= 0.0 || sigma_z_m <= 0.0 {
        return 0.0;
    }
    gaussian_concentration(
        downwind_km,
        crosswind_m,
        sensor.height_m,
        stack_speed,
        source.emission_g_s,
        source.height_m + rise_m,
        sigma_y_m,
        sigma_z_m,
    ) * 1.0e6
}

fn sigma_y(c: f64, d: f64, x_km: f64) -> f64 {
    let theta = 0.017_453_293 * (c - d * x_km.ln());
    465.116_28 * x_km * theta.tan()
}

fn get_sigma_y(stability: u8, x_km: f64) -> f64 {
    match stability {
        b'A' => sigma_y(24.1670, 2.5334, x_km),
        b'B' => sigma_y(18.3330, 1.8096, x_km),
        b'C' => sigma_y(12.5000, 1.0857, x_km),
        b'D' => sigma_y(8.3330, 0.72382, x_km),
        b'E' => sigma_y(6.2500, 0.54287, x_km),
        b'F' => sigma_y(4.1667, 0.36191, x_km),
        _ => f64::NAN,
    }
}

fn sigma_z(a: f64, b: f64, x_km: f64) -> f64 {
    a * x_km.powf(b)
}

fn get_sigma_z(stability: u8, x: f64) -> f64 {
    let value = match stability {
        b'A' if x <= 0.10 => sigma_z(122.800, 0.94470, x),
        b'A' if x <= 0.15 => sigma_z(158.080, 1.05420, x),
        b'A' if x <= 0.20 => sigma_z(170.220, 1.09320, x),
        b'A' if x <= 0.25 => sigma_z(179.520, 1.12620, x),
        b'A' if x <= 0.30 => sigma_z(217.410, 1.26440, x),
        b'A' if x <= 0.40 => sigma_z(258.890, 1.40940, x),
        b'A' if x <= 0.50 => sigma_z(346.750, 1.72830, x),
        b'A' if x <= 3.11 => sigma_z(453.850, 2.11660, x),
        b'A' => 5_000.0,
        b'B' if x <= 0.20 => sigma_z(90.673, 0.93198, x),
        b'B' if x <= 0.40 => sigma_z(98.483, 0.98332, x),
        b'B' => sigma_z(109.300, 1.09710, x),
        b'C' => sigma_z(61.141, 0.91465, x),
        b'D' if x <= 0.30 => sigma_z(34.459, 0.86974, x),
        b'D' if x <= 1.0 => sigma_z(32.093, 0.81066, x),
        b'D' if x <= 3.0 => sigma_z(32.093, 0.64403, x),
        b'D' if x <= 10.0 => sigma_z(33.504, 0.60486, x),
        b'D' if x <= 30.0 => sigma_z(36.650, 0.56589, x),
        b'D' => sigma_z(44.053, 0.51179, x),
        b'E' if x <= 0.10 => sigma_z(24.260, 0.83660, x),
        b'E' if x <= 0.30 => sigma_z(23.331, 0.81956, x),
        b'E' if x <= 1.0 => sigma_z(21.628, 0.75660, x),
        b'E' if x <= 2.0 => sigma_z(21.628, 0.63077, x),
        b'E' if x <= 4.0 => sigma_z(22.534, 0.57154, x),
        b'E' if x <= 10.0 => sigma_z(24.703, 0.50527, x),
        b'E' if x <= 20.0 => sigma_z(26.970, 0.46713, x),
        b'E' if x <= 40.0 => sigma_z(35.420, 0.37615, x),
        b'E' => sigma_z(47.618, 0.29592, x),
        b'F' if x <= 0.20 => sigma_z(15.209, 0.81558, x),
        b'F' if x <= 0.70 => sigma_z(14.457, 0.78407, x),
        b'F' if x <= 1.0 => sigma_z(13.953, 0.68465, x),
        b'F' if x <= 2.0 => sigma_z(13.953, 0.63227, x),
        b'F' if x <= 3.0 => sigma_z(14.823, 0.54503, x),
        b'F' if x <= 7.0 => sigma_z(16.187, 0.46490, x),
        b'F' if x <= 15.0 => sigma_z(17.836, 0.41507, x),
        b'F' if x <= 30.0 => sigma_z(22.651, 0.32681, x),
        b'F' if x <= 60.0 => sigma_z(27.074, 0.27436, x),
        b'F' => sigma_z(34.219, 0.21716, x),
        _ => f64::NAN,
    };
    value.min(5_000.0)
}

fn effective_wind_speed(reference: f64, height: f64, reference_height: f64, stability: u8) -> f64 {
    let exponent = match stability {
        b'A' | b'B' => 0.15,
        b'C' => 0.20,
        b'D' => 0.25,
        b'E' | b'F' => 0.30,
        _ => return f64::NAN,
    };
    reference * (height / reference_height).powf(exponent)
}

fn wind_components(
    receptor_x: f64,
    receptor_y: f64,
    source_x: f64,
    source_y: f64,
    sin_direction: f64,
    cos_direction: f64,
) -> (f64, f64) {
    (
        (-(receptor_x - source_x) * sin_direction - (receptor_y - source_y) * cos_direction)
            / 1_000.0,
        (receptor_x - source_x) * cos_direction - (receptor_y - source_y) * sin_direction,
    )
}

fn plume_rise(
    wind_speed: f64,
    exit_velocity: f64,
    diameter: f64,
    stack_temperature_k: f64,
    ambient_temperature_k: f64,
    stability: u8,
) -> (f64, f64) {
    let gravity = 9.806_16;
    let buoyancy = gravity
        * exit_velocity
        * diameter
        * diameter
        * (stack_temperature_k - ambient_temperature_k)
        / (4.0 * stack_temperature_k);
    let momentum = exit_velocity * exit_velocity * diameter * diameter * ambient_temperature_k
        / (4.0 * stack_temperature_k);
    if matches!(stability, b'E' | b'F') {
        let eta: f64 = if stability == b'E' { 0.020 } else { 0.035 };
        let stratification = gravity * eta / ambient_temperature_k;
        let threshold = 0.019_582 * stack_temperature_k * exit_velocity * stratification.sqrt();
        if stack_temperature_k - ambient_temperature_k >= threshold {
            (
                2.6 * (buoyancy / (wind_speed * stratification)).cbrt(),
                2.0715 * wind_speed / stratification.sqrt(),
            )
        } else {
            let neutral = 3.0 * diameter * exit_velocity / wind_speed;
            let stable = 1.5 * (momentum / (wind_speed * stratification.sqrt())).cbrt();
            (neutral.min(stable), 0.0)
        }
    } else if buoyancy < 55.0 {
        let threshold =
            0.0297 * stack_temperature_k * exit_velocity.cbrt() / diameter.powf(2.0 / 3.0);
        if stack_temperature_k - ambient_temperature_k >= threshold {
            (
                21.425 * buoyancy.powf(0.75) / wind_speed,
                49.0 * buoyancy.powf(0.625),
            )
        } else {
            (3.0 * diameter * exit_velocity / wind_speed, 0.0)
        }
    } else {
        let threshold =
            0.00575 * stack_temperature_k * exit_velocity.powf(2.0 / 3.0) / diameter.cbrt();
        if stack_temperature_k - ambient_temperature_k >= threshold {
            (
                38.71 * buoyancy.powf(0.6) / wind_speed,
                119.0 * buoyancy.powf(0.4),
            )
        } else {
            (3.0 * diameter * exit_velocity / wind_speed, 0.0)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gaussian_concentration(
    downwind_km: f64,
    crosswind_m: f64,
    receptor_height_m: f64,
    wind_speed_m_s: f64,
    emission_g_s: f64,
    effective_height_m: f64,
    sigma_y_m: f64,
    sigma_z_m: f64,
) -> f64 {
    if downwind_km <= 0.0 {
        return 0.0;
    }
    let scale = emission_g_s / (2.0 * PI * wind_speed_m_s * sigma_y_m * sigma_z_m);
    let vertical_direct =
        (-(receptor_height_m - effective_height_m).powi(2) / (2.0 * sigma_z_m.powi(2))).exp();
    let vertical_reflected =
        (-(receptor_height_m + effective_height_m).powi(2) / (2.0 * sigma_z_m.powi(2))).exp();
    let lateral = (-crosswind_m.powi(2) / (2.0 * sigma_y_m.powi(2))).exp();
    let concentration = scale * (vertical_direct + vertical_reflected) * lateral;
    if concentration.is_finite() {
        concentration
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_dispersion_reference_values_match() {
        assert!((get_sigma_y(b'D', 0.5) - 36.146_193_496_038).abs() < 1.0e-9);
        assert!((get_sigma_z(b'D', 0.5) - 18.296_892_641_654).abs() < 1.0e-9);
        let (rise, offset) = plume_rise(6.0, 20.0, 5.0, 400.0, 280.0, b'D');
        assert!((rise - 223.352_113_600_373).abs() < 1.0e-9);
        assert!((offset - 1_264.034_881_130_08).abs() < 1.0e-9);
    }

    #[test]
    fn design_is_bounded_and_canonicalized() {
        let truth = Design::truth();
        assert!(truth.sources()[0].x_m < truth.sources()[1].x_m);
        let mut swapped = *truth.values();
        for index in 0..4 {
            swapped.swap(index, index + 4);
        }
        let canonical = Design::from_slice(&swapped).expect("swapped truth is valid");
        assert_eq!(canonical.values(), truth.values());
        assert!(Design::from_slice(&[0.0; DIMENSION - 1]).is_err());
        assert!(Design::from_slice(&[f64::NAN; DIMENSION]).is_err());
    }

    #[test]
    fn deterministic_dataset_has_disjoint_holdout_ids() {
        let first = Dataset::synthetic();
        let second = Dataset::synthetic();
        assert_eq!(first.training().len(), 256);
        assert_eq!(first.validation().len(), 100);
        assert_eq!(
            first.training()[42].measured_ug_m3,
            second.training()[42].measured_ug_m3
        );
        assert!(first.training().iter().all(|training| {
            first
                .validation()
                .iter()
                .all(|validation| training.sensor.id != validation.sensor.id)
        }));
    }

    #[test]
    fn truth_is_better_than_baseline_on_training_and_holdout() {
        let dataset = Dataset::synthetic();
        let truth = evaluate_training(Design::truth().values(), &dataset).unwrap();
        let baseline = evaluate_training(Design::baseline().values(), &dataset).unwrap();
        let truth_holdout = evaluate_validation(Design::truth().values(), &dataset).unwrap();
        let baseline_holdout = evaluate_validation(Design::baseline().values(), &dataset).unwrap();
        assert!(truth.scalar_score < baseline.scalar_score);
        assert!(truth_holdout.scalar_score < baseline_holdout.scalar_score);
        assert_eq!(truth.source_position_error_m, 0.0);
    }

    #[test]
    fn objective_adapters_reject_bad_dimension() {
        let dataset = Dataset::synthetic();
        assert_eq!(scalar_objective(&[0.0], &dataset), 1.0e99);
        assert_eq!(multi_objective(&[0.0], &dataset), vec![1.0e99; OBJECTIVES]);
        assert!(!qd_objective(&[0.0], &dataset).0.is_finite());
    }
}
