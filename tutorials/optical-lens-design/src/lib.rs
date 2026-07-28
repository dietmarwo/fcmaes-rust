//! Sequential geometric ray tracing and Cooke-triplet optimization.
//!
//! The physics hot path has no optical-simulator dependency: sphere
//! intersection, vector Snell refraction, Sellmeier dispersion, and paraxial
//! first-order calculations are implemented below and covered by tests.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, Mode, ModeParams, RetryBounds,
    RetryConfig, RetryImprovement, RetryRunResult, Rng, optimize_bite, parallel_batch,
    pareto_indices, retry,
};

pub const DIMENSION: usize = 11;
pub const OBJECTIVES: usize = 3;
pub const CONSTRAINTS: usize = 3;
pub const INVALID_COST: f64 = 1.0e12;
pub const APERTURE_RADIUS_MM: f64 = 5.0;
pub const PUBLICATION_GRID_RADIUS: usize = 8;
pub const TARGET_EFL_MM: f64 = 50.0;
pub const EFL_TOLERANCE_MM: f64 = 1.0;
pub const MIN_EDGE_THICKNESS_MM: f64 = 0.8;
pub const FIELDS_DEG: [f64; 3] = [0.0, 14.0, 20.0];
pub const WAVELENGTHS_UM: [f64; 3] = [0.4861, 0.5876, 0.6563];

// Six curvatures in mm^-1, three centre thicknesses and two air gaps in mm.
pub const LOWER_BOUNDS: [f64; DIMENSION] = [
    1.0 / 80.0,
    -1.0 / 20.0,
    -1.0 / 10.0,
    1.0 / 80.0,
    1.0 / 800.0,
    -1.0 / 10.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
];
pub const UPPER_BOUNDS: [f64; DIMENSION] = [
    1.0 / 15.0,
    -1.0 / 800.0,
    -1.0 / 80.0,
    1.0 / 10.0,
    1.0 / 20.0,
    -1.0 / 80.0,
    8.0,
    8.0,
    8.0,
    10.0,
    10.0,
];

/// Optiland tutorial 5c final prescription, converted from radii to curvature.
pub const REFERENCE_DESIGN: [f64; DIMENSION] = [
    1.0 / 30.0189,
    1.0 / -63.0945,
    1.0 / -18.2466,
    1.0 / 31.338,
    1.0 / 623.507,
    1.0 / -16.4225,
    4.0,
    4.0,
    4.0,
    4.21698,
    2.17393,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Option<Self> {
        let norm = self.norm();
        (norm.is_finite() && norm > 0.0).then_some(Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        })
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glass {
    Air,
    Sk16,
    F2,
}

impl Glass {
    /// Refractive index for vacuum wavelength in micrometres.
    pub fn refractive_index(self, wavelength_um: f64) -> Option<f64> {
        if !wavelength_um.is_finite() || !(0.35..=2.5).contains(&wavelength_um) {
            return None;
        }
        if self == Self::Air {
            return Some(1.0);
        }
        let (b, c) = match self {
            Self::Sk16 => (
                [1.343_177_74, 0.241_144_399, 0.994_317_969],
                [0.007_046_873_39, 0.022_900_500_2, 92.750_852_6],
            ),
            Self::F2 => (
                [1.345_333_59, 0.209_073_176, 0.937_357_162],
                [0.009_977_438_71, 0.047_045_076_7, 111.886_764],
            ),
            Self::Air => unreachable!(),
        };
        let lambda2 = wavelength_um * wavelength_um;
        let n2 = 1.0
            + (0..3)
                .map(|index| b[index] * lambda2 / (lambda2 - c[index]))
                .sum::<f64>();
        (n2.is_finite() && n2 > 1.0).then(|| n2.sqrt())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub position: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Surface {
    vertex_z: f64,
    curvature: f64,
    medium_after: Glass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Design {
    pub values: [f64; DIMENSION],
}

impl Design {
    pub fn from_slice(values: &[f64]) -> Result<Self, &'static str> {
        let values: [f64; DIMENSION] = values
            .try_into()
            .map_err(|_| "an optical design must contain eleven values")?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("all design values must be finite");
        }
        if values
            .iter()
            .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS))
            .any(|(&value, (&lower, upper))| value < lower || value > upper)
        {
            return Err("optical design lies outside the supported bounds");
        }
        Ok(Self { values })
    }

    pub fn reference() -> Self {
        Self::from_slice(&REFERENCE_DESIGN).expect("the reference prescription is in bounds")
    }

    pub fn radii_mm(&self) -> [f64; 6] {
        std::array::from_fn(|index| 1.0 / self.values[index])
    }

    pub fn centre_thicknesses_mm(&self) -> [f64; 3] {
        [self.values[6], self.values[7], self.values[8]]
    }

    pub fn air_gaps_mm(&self) -> [f64; 2] {
        [self.values[9], self.values[10]]
    }

    fn surfaces(&self) -> [Surface; 6] {
        let t = self.centre_thicknesses_mm();
        let g = self.air_gaps_mm();
        let z = [
            0.0,
            t[0],
            t[0] + g[0],
            t[0] + g[0] + t[1],
            t[0] + g[0] + t[1] + g[1],
            t[0] + g[0] + t[1] + g[1] + t[2],
        ];
        let media = [
            Glass::Sk16,
            Glass::Air,
            Glass::F2,
            Glass::Air,
            Glass::Sk16,
            Glass::Air,
        ];
        std::array::from_fn(|index| Surface {
            vertex_z: z[index],
            curvature: self.values[index],
            medium_after: media[index],
        })
    }

    pub fn last_vertex_z(&self) -> f64 {
        let t = self.centre_thicknesses_mm();
        let g = self.air_gaps_mm();
        t.iter().sum::<f64>() + g.iter().sum::<f64>()
    }

    /// Effective focal length and back focal length at one wavelength.
    pub fn paraxial_focal_lengths(&self, wavelength_um: f64) -> Option<(f64, f64)> {
        // Ray vector is [height, reduced angle n*theta]. Refraction changes
        // reduced angle by -power*height; translation changes height by t*u/n.
        let mut matrix = [[1.0, 0.0], [0.0, 1.0]];
        let surfaces = self.surfaces();
        let mut n_before = 1.0;
        for (index, surface) in surfaces.iter().enumerate() {
            let n_after = surface.medium_after.refractive_index(wavelength_um)?;
            let power = (n_after - n_before) * surface.curvature;
            matrix = multiply([[1.0, 0.0], [-power, 1.0]], matrix);
            if index + 1 < surfaces.len() {
                let thickness = surfaces[index + 1].vertex_z - surface.vertex_z;
                matrix = multiply([[1.0, thickness / n_after], [0.0, 1.0]], matrix);
            }
            n_before = n_after;
        }
        let a = matrix[0][0];
        let c = matrix[1][0];
        if c.abs() < 1.0e-12 {
            return None;
        }
        let efl = -1.0 / c;
        let bfl = -a / c;
        (efl.is_finite() && bfl.is_finite() && bfl > 0.0).then_some((efl, bfl))
    }

    pub fn minimum_edge_thickness_mm(&self) -> Option<f64> {
        let radii = self.radii_mm();
        let centre = self.centre_thicknesses_mm();
        let mut minimum = f64::INFINITY;
        for lens in 0..3 {
            let front = spherical_sag(radii[2 * lens], APERTURE_RADIUS_MM)?;
            let back = spherical_sag(radii[2 * lens + 1], APERTURE_RADIUS_MM)?;
            minimum = minimum.min(centre[lens] + back - front);
        }
        Some(minimum)
    }

    pub fn track_length_mm(&self) -> Option<f64> {
        let (_, bfl) = self.paraxial_focal_lengths(WAVELENGTHS_UM[1])?;
        Some(self.last_vertex_z() + bfl)
    }

    pub fn glass_volume_mm3(&self) -> Option<f64> {
        // Axisymmetric numerical quadrature of local thickness is transparent
        // and robust for both curvature signs.
        let radii = self.radii_mm();
        let centre = self.centre_thicknesses_mm();
        let rings = 128;
        let dr = APERTURE_RADIUS_MM / rings as f64;
        let mut total = 0.0;
        for lens in 0..3 {
            for ring in 0..rings {
                let radius = (ring as f64 + 0.5) * dr;
                let front = spherical_sag(radii[2 * lens], radius)?;
                let back = spherical_sag(radii[2 * lens + 1], radius)?;
                let local = centre[lens] + back - front;
                total += 2.0 * std::f64::consts::PI * radius * dr * local.max(0.0);
            }
        }
        total.is_finite().then_some(total)
    }
}

fn multiply(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn spherical_sag(radius: f64, height: f64) -> Option<f64> {
    let square = radius * radius - height * height;
    (square >= 0.0).then(|| radius - radius.signum() * square.sqrt())
}

fn intersect_surface(ray: Ray, surface: Surface) -> Option<(Vec3, Vec3)> {
    let radius = 1.0 / surface.curvature;
    let centre = Vec3 {
        x: 0.0,
        y: 0.0,
        z: surface.vertex_z + radius,
    };
    let offset = ray.position - centre;
    let b = offset.dot(ray.direction);
    let c = offset.dot(offset) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let mut candidates = [-b - root, -b + root]
        .into_iter()
        .filter(|time| *time > 1.0e-9)
        .map(|time| {
            let point = ray.position + ray.direction * time;
            ((point.z - surface.vertex_z).abs(), point)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    let point = candidates.first()?.1;
    let normal = (point - centre).normalized()?;
    Some((point, normal))
}

fn refract(direction: Vec3, mut normal: Vec3, n_before: f64, n_after: f64) -> Option<Vec3> {
    if direction.dot(normal) > 0.0 {
        normal = normal * -1.0;
    }
    let eta = n_before / n_after;
    let cos_incident = -normal.dot(direction);
    let discriminant = 1.0 - eta * eta * (1.0 - cos_incident * cos_incident);
    if discriminant < 0.0 {
        return None;
    }
    (direction * eta + normal * (eta * cos_incident - discriminant.sqrt())).normalized()
}

/// Trace one object-at-infinity ray to the paraxial image plane.
pub fn trace_ray(
    design: &Design,
    pupil_x_mm: f64,
    pupil_y_mm: f64,
    field_deg: f64,
    wavelength_um: f64,
) -> Option<[f64; 2]> {
    let (_, bfl) = design.paraxial_focal_lengths(WAVELENGTHS_UM[1])?;
    trace_ray_to_image_z(
        design,
        pupil_x_mm,
        pupil_y_mm,
        field_deg,
        wavelength_um,
        design.last_vertex_z() + bfl,
    )
}

fn trace_ray_to_image_z(
    design: &Design,
    pupil_x_mm: f64,
    pupil_y_mm: f64,
    field_deg: f64,
    wavelength_um: f64,
    image_z: f64,
) -> Option<[f64; 2]> {
    if pupil_x_mm.hypot(pupil_y_mm) > APERTURE_RADIUS_MM + 1.0e-12 {
        return None;
    }
    let angle = field_deg.to_radians();
    let mut ray = Ray {
        position: Vec3 {
            x: pupil_x_mm,
            y: pupil_y_mm,
            z: -1.0e-7,
        },
        direction: Vec3 {
            x: 0.0,
            y: angle.sin(),
            z: angle.cos(),
        },
    };
    let mut n_before = 1.0;
    for surface in design.surfaces() {
        let (point, normal) = intersect_surface(ray, surface)?;
        if point.x.hypot(point.y) > 2.5 * APERTURE_RADIUS_MM {
            return None;
        }
        let n_after = surface.medium_after.refractive_index(wavelength_um)?;
        let direction = refract(ray.direction, normal, n_before, n_after)?;
        ray = Ray {
            position: point + direction * 1.0e-8,
            direction,
        };
        n_before = n_after;
    }
    let distance = (image_z - ray.position.z) / ray.direction.z;
    (distance.is_finite() && distance > 0.0).then(|| {
        let point = ray.position + ray.direction * distance;
        [point.x, point.y]
    })
}

pub fn pupil_points(grid_radius: usize) -> Vec<[f64; 2]> {
    let mut points = Vec::new();
    let scale = APERTURE_RADIUS_MM / grid_radius as f64;
    let radius2 = grid_radius * grid_radius;
    for x in -(grid_radius as isize)..=grid_radius as isize {
        for y in -(grid_radius as isize)..=grid_radius as isize {
            if (x * x + y * y) as usize <= radius2 {
                points.push([x as f64 * scale, y as f64 * scale]);
            }
        }
    }
    points
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub design: Design,
    pub rms_spot_mm: f64,
    pub field_rms_mm: [f64; 3],
    pub wavelength_field_rms_mm: [[f64; 3]; 3],
    pub efl_mm: f64,
    pub bfl_mm: f64,
    pub track_length_mm: f64,
    pub glass_volume_mm3: f64,
    pub minimum_edge_thickness_mm: f64,
    pub lost_rays: usize,
    pub total_rays: usize,
    pub constraints: [f64; CONSTRAINTS],
}

impl Evaluation {
    pub fn feasible(&self) -> bool {
        self.constraints.iter().all(|value| *value <= 0.0)
    }

    pub fn objectives(&self) -> [f64; OBJECTIVES] {
        [
            self.rms_spot_mm * 1_000.0,
            self.track_length_mm,
            self.glass_volume_mm3,
        ]
    }

    pub fn scalar_score(&self) -> f64 {
        self.rms_spot_mm * 1_000.0
            + self
                .constraints
                .iter()
                .map(|value| 10_000.0 * value.max(0.0))
                .sum::<f64>()
    }
}

pub fn evaluate(values: &[f64], grid_radius: usize) -> Option<Evaluation> {
    let design = Design::from_slice(values).ok()?;
    let (efl_mm, bfl_mm) = design.paraxial_focal_lengths(WAVELENGTHS_UM[1])?;
    let track_length_mm = design.track_length_mm()?;
    let glass_volume_mm3 = design.glass_volume_mm3()?;
    let minimum_edge_thickness_mm = design.minimum_edge_thickness_mm()?;
    let pupils = pupil_points(grid_radius);
    let total_rays = pupils.len() * FIELDS_DEG.len() * WAVELENGTHS_UM.len();
    let mut lost_rays = 0;
    let mut wavelength_field_rms_mm = [[0.0; 3]; 3];
    for (field_index, field) in FIELDS_DEG.iter().enumerate() {
        for (wavelength_index, wavelength) in WAVELENGTHS_UM.iter().enumerate() {
            let hits = pupils
                .iter()
                .filter_map(|point| {
                    let hit = trace_ray(&design, point[0], point[1], *field, *wavelength);
                    if hit.is_none() {
                        lost_rays += 1;
                    }
                    hit
                })
                .collect::<Vec<_>>();
            if hits.len() != pupils.len() {
                wavelength_field_rms_mm[field_index][wavelength_index] = INVALID_COST;
                continue;
            }
            let centre = [
                hits.iter().map(|point| point[0]).sum::<f64>() / hits.len() as f64,
                hits.iter().map(|point| point[1]).sum::<f64>() / hits.len() as f64,
            ];
            wavelength_field_rms_mm[field_index][wavelength_index] = (hits
                .iter()
                .map(|point| (point[0] - centre[0]).powi(2) + (point[1] - centre[1]).powi(2))
                .sum::<f64>()
                / hits.len() as f64)
                .sqrt();
        }
    }
    let field_rms_mm = std::array::from_fn(|field| {
        let sum = wavelength_field_rms_mm[field]
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        (sum / WAVELENGTHS_UM.len() as f64).sqrt()
    });
    let field_weights = [1.0, 2.0, 3.0];
    let rms_spot_mm = (field_rms_mm
        .iter()
        .zip(field_weights)
        .map(|(value, weight)| weight * value * value)
        .sum::<f64>()
        / field_weights.iter().sum::<f64>())
    .sqrt();
    let constraints = [
        MIN_EDGE_THICKNESS_MM - minimum_edge_thickness_mm,
        (efl_mm - TARGET_EFL_MM).abs() - EFL_TOLERANCE_MM,
        lost_rays as f64,
    ];
    [
        rms_spot_mm,
        efl_mm,
        bfl_mm,
        track_length_mm,
        glass_volume_mm3,
        minimum_edge_thickness_mm,
    ]
    .iter()
    .all(|value| value.is_finite())
    .then_some(Evaluation {
        design,
        rms_spot_mm,
        field_rms_mm,
        wavelength_field_rms_mm,
        efl_mm,
        bfl_mm,
        track_length_mm,
        glass_volume_mm3,
        minimum_edge_thickness_mm,
        lost_rays,
        total_rays,
        constraints,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoOptimizer {
    Cma,
    De,
    Bite,
}

impl SoOptimizer {
    pub const ALL: [Self; 3] = [Self::Cma, Self::De, Self::Bite];
    pub fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma",
            Self::De => "de",
            Self::Bite => "bite",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SoConfig {
    pub evaluations_per_arm: u64,
    pub retries: usize,
    pub workers: usize,
    pub seed: u64,
    pub grid_radius: usize,
}

#[derive(Clone, Debug)]
pub struct SoResult {
    pub optimizer: SoOptimizer,
    pub requested_evaluations: u64,
    pub actual_evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub best: Evaluation,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_so(optimizer: SoOptimizer, config: &SoConfig) -> Result<SoResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 || config.grid_radius < 2 {
        return Err("SO budget, retries and pupil grid must be positive".into());
    }
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let objective = |x: &[f64]| {
        evaluate(x, config.grid_radius)
            .map(|evaluation| evaluation.scalar_score())
            .unwrap_or(INVALID_COST)
    };
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let started = Instant::now();
    let result = retry(
        &objective,
        &bounds,
        &RetryConfig {
            num_retries: config.retries,
            workers: config.workers,
            capacity: config.retries,
            max_evaluations: per_retry,
            seed: config.seed
                + match optimizer {
                    SoOptimizer::Cma => 0,
                    SoOptimizer::De => 10_000,
                    SoOptimizer::Bite => 20_000,
                },
            statistic_num: 200,
            ..Default::default()
        },
        |objective, context| {
            let mut rng = Rng::new(context.seed);
            let random = if context.run_id == 0 {
                REFERENCE_DESIGN.to_vec()
            } else {
                context
                    .bounds
                    .lower()
                    .iter()
                    .zip(context.bounds.upper())
                    .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
                    .collect::<Vec<_>>()
            };
            match optimizer {
                SoOptimizer::Cma => {
                    let mut fitness = Fitness::bounded(
                        DIMENSION,
                        1,
                        context.bounds.lower(),
                        context.bounds.upper(),
                    );
                    // Curvatures span intervals of order 1e-2 while air gaps
                    // span intervals of order 1e1. CMA-ES therefore operates
                    // in normalized coordinates so one scalar sigma has the
                    // same meaning in every decision dimension.
                    fitness.set_normalize(true);
                    let mut cma = Cmaes::new(
                        fitness,
                        &random,
                        &[0.2],
                        &CmaesParams {
                            max_evaluations: context.max_evaluations,
                            seed: context.seed,
                            stop_tol_hist_fun: 0.0,
                            ..Default::default()
                        },
                    );
                    let optimized = cma.optimize(objective, 1);
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
                SoOptimizer::De => {
                    let fitness = Fitness::bounded(
                        DIMENSION,
                        1,
                        context.bounds.lower(),
                        context.bounds.upper(),
                    );
                    let mut de = De::new(
                        fitness,
                        &random,
                        &[0.2; DIMENSION],
                        None,
                        &DeParams {
                            popsize: 31,
                            max_evaluations: context.max_evaluations,
                            seed: context.seed,
                            ..Default::default()
                        },
                    );
                    let optimized = de.optimize(objective);
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
                SoOptimizer::Bite => {
                    let optimized = optimize_bite(
                        objective,
                        context.bounds.lower(),
                        context.bounds.upper(),
                        Some(&random),
                        &BiteParams {
                            max_evaluations: context.max_evaluations,
                            seed: context.seed,
                            ..Default::default()
                        },
                        2,
                    );
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
            }
        },
    );
    if !result.success {
        return Err(format!("{} returned no design", optimizer.name()).into());
    }
    let optimized = evaluate(&result.x, config.grid_radius)
        .ok_or_else(|| format!("{} best design did not replay", optimizer.name()))?;
    let reference =
        evaluate(&REFERENCE_DESIGN, config.grid_radius).ok_or("reference design did not replay")?;
    // The disclosed reference is the first restart's evaluated seed and is
    // therefore part of every arm's budget. Some optimizer result structs
    // report only their final state, so retain the better evaluated point.
    let best = if reference.scalar_score() < optimized.scalar_score() {
        reference
    } else {
        optimized
    };
    Ok(SoResult {
        optimizer,
        requested_evaluations: config.evaluations_per_arm,
        actual_evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        improvements: result.improvements,
    })
}

#[derive(Clone, Debug)]
pub struct MoConfig {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: i32,
    pub seed: u64,
    pub grid_radius: usize,
}

#[derive(Clone, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub feasible_population: usize,
    pub pareto_population: usize,
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub evaluation: Evaluation,
    pub selected: bool,
}

#[derive(Clone, Debug)]
pub struct MoResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub elapsed: Duration,
    pub pareto: Vec<ParetoPoint>,
    pub progress: Vec<MoProgress>,
}

fn mo_values(x: &[f64], grid_radius: usize) -> Vec<f64> {
    evaluate(x, grid_radius)
        .map(|evaluation| {
            let mut values = evaluation.objectives().to_vec();
            values.extend(evaluation.constraints);
            values
        })
        .unwrap_or_else(|| vec![INVALID_COST; OBJECTIVES + CONSTRAINTS])
}

pub fn optimize_mo(config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.evaluations == 0
        || config.popsize < 4
        || !config.popsize.is_multiple_of(2)
        || config.grid_radius < 2
    {
        return Err("invalid MODE budget, population or pupil grid".into());
    }
    let mut mode = Mode::try_new(
        Fitness::bounded(
            DIMENSION,
            OBJECTIVES + CONSTRAINTS,
            &LOWER_BOUNDS,
            &UPPER_BOUNDS,
        ),
        OBJECTIVES,
        CONSTRAINTS,
        None,
        &ModeParams {
            popsize: config.popsize as i32,
            seed: config.seed,
            nsga_update: true,
            ..Default::default()
        },
    )?;
    let generations = config.evaluations.div_ceil(config.popsize);
    let started = Instant::now();
    let mut actual_evaluations = 0;
    let mut progress = Vec::new();
    let mut seed_rng = Rng::new(config.seed ^ 0xD1B5_4A32_D192_ED03);
    for generation in 0..generations {
        let xs = if generation == 0 {
            (0..config.popsize)
                .map(|index| {
                    (0..DIMENSION)
                        .map(|dimension| {
                            let displacement = if index == 0 {
                                0.0
                            } else {
                                (2.0 * seed_rng.uniform01() - 1.0)
                                    * 0.01
                                    * (UPPER_BOUNDS[dimension] - LOWER_BOUNDS[dimension])
                            };
                            (REFERENCE_DESIGN[dimension] + displacement)
                                .clamp(LOWER_BOUNDS[dimension], UPPER_BOUNDS[dimension])
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            mode.ask()
        };
        let ys = parallel_batch(&xs, config.workers, |x| mo_values(x, config.grid_radius));
        actual_evaluations += ys.len();
        if generation == 0 {
            mode.set_population(&xs, &ys);
        } else {
            mode.tell(&ys);
        }
        if generation == 0 || (generation + 1) % 5 == 0 || generation + 1 == generations {
            let current = mode.result();
            let feasible = current
                .y
                .iter()
                .filter(|row| row[OBJECTIVES..].iter().all(|value| *value <= 0.0))
                .count();
            let feasible_objectives = current
                .y
                .iter()
                .filter(|row| row[OBJECTIVES..].iter().all(|value| *value <= 0.0))
                .map(|row| row[..OBJECTIVES].to_vec())
                .collect::<Vec<_>>();
            let pareto = if feasible_objectives.is_empty() {
                0
            } else {
                pareto_indices(&feasible_objectives, OBJECTIVES)?.len()
            };
            let best = feasible_objectives
                .iter()
                .map(|row| row[0] + row[1] / 100.0 + row[2] / 1_000.0)
                .fold(f64::INFINITY, f64::min);
            progress.push(MoProgress {
                evaluations: actual_evaluations,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_quality: best,
            });
        }
    }
    let result = mode.result();
    let feasible = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, row)| row[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, row)| (index, row[..OBJECTIVES].to_vec()))
        .collect::<Vec<_>>();
    if feasible.is_empty() {
        return Err("MODE retained no feasible optical design".into());
    }
    let front_rows = feasible
        .iter()
        .map(|(_, row)| row.clone())
        .collect::<Vec<_>>();
    let front = pareto_indices(&front_rows, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            let population_index = feasible[*front_index].0;
            evaluate(&result.x[population_index], config.grid_radius).map(|evaluation| {
                ParetoPoint {
                    evaluation,
                    selected: false,
                }
            })
        })
        .collect::<Vec<_>>();
    for objective in 0..OBJECTIVES {
        if let Some(index) = (0..pareto.len()).min_by(|&left, &right| {
            pareto[left].evaluation.objectives()[objective]
                .total_cmp(&pareto[right].evaluation.objectives()[objective])
        }) {
            pareto[index].selected = true;
        }
    }
    if let Some(index) = (0..pareto.len()).min_by(|&left, &right| {
        let score = |point: &ParetoPoint| {
            let values = point.evaluation.objectives();
            values[0] + values[1] / 100.0 + values[2] / 1_000.0
        };
        score(&pareto[left]).total_cmp(&score(&pareto[right]))
    }) {
        pareto[index].selected = true;
    }
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations,
        elapsed: started.elapsed(),
        pareto,
        progress,
    })
}

#[derive(Clone, Debug)]
pub struct ValidationSummary {
    pub paraxial_focus_relative_residual: f64,
    pub efl_mm: f64,
    pub reference_efl_mm: f64,
    pub efl_relative_error: f64,
    pub on_axis_rms_mm: [f64; 3],
    pub reference_on_axis_rms_mm: [f64; 3],
    pub maximum_spot_relative_error: f64,
    pub convergence: Vec<(usize, usize, f64)>,
    pub paraxial_pass: bool,
    pub efl_pass: bool,
    pub spot_pass: bool,
    pub convergence_pass: bool,
}

impl ValidationSummary {
    pub fn passed(&self) -> bool {
        self.paraxial_pass && self.efl_pass && self.spot_pass && self.convergence_pass
    }
}

pub fn validate_reference() -> Result<ValidationSummary, Box<dyn Error>> {
    let reference = Design::reference();
    let published_efl = 50.002;
    let published_spot = [0.01444, 0.01076, 0.01108];
    let evaluation = evaluate(&reference.values, 4).ok_or("reference trace failed")?;
    let paraxial_hit = trace_ray(&reference, 1.0e-4, 0.0, 0.0, WAVELENGTHS_UM[1])
        .ok_or("paraxial marginal ray failed")?;
    let paraxial_focus_relative_residual = paraxial_hit[0].abs() / 1.0e-4;
    // Optiland optimized and published an explicit final image gap of
    // 43.5928 mm. Use that plane for the cross-implementation spot check;
    // optimization below deliberately solves the paraxial focus instead.
    let pupils = pupil_points(4);
    let on_axis = std::array::from_fn(|wavelength_index| {
        let hits = pupils
            .iter()
            .filter_map(|point| {
                trace_ray_to_image_z(
                    &reference,
                    point[0],
                    point[1],
                    0.0,
                    WAVELENGTHS_UM[wavelength_index],
                    reference.last_vertex_z() + 43.5928,
                )
            })
            .collect::<Vec<_>>();
        let centre = [
            hits.iter().map(|point| point[0]).sum::<f64>() / hits.len() as f64,
            hits.iter().map(|point| point[1]).sum::<f64>() / hits.len() as f64,
        ];
        (hits
            .iter()
            .map(|point| (point[0] - centre[0]).powi(2) + (point[1] - centre[1]).powi(2))
            .sum::<f64>()
            / hits.len() as f64)
            .sqrt()
    });
    let efl_relative_error = ((evaluation.efl_mm - published_efl) / published_efl).abs();
    let maximum_spot_relative_error = on_axis
        .iter()
        .zip(published_spot)
        .map(|(actual, expected)| ((actual - expected) / expected).abs())
        .fold(0.0, f64::max);
    let mut convergence = Vec::new();
    for radius in [3, 4, 5, 6, 8, 10, 12, 16] {
        let result = evaluate(&reference.values, radius).ok_or("convergence trace failed")?;
        convergence.push((
            radius,
            pupil_points(radius).len(),
            result.rms_spot_mm * 1_000.0,
        ));
    }
    let last = convergence[convergence.len() - 1].2;
    let convergence_relative = convergence
        .iter()
        .filter(|(radius, _, _)| [PUBLICATION_GRID_RADIUS, 12, 16].contains(radius))
        .map(|(_, _, value)| ((value - last) / last).abs())
        .fold(0.0, f64::max);
    Ok(ValidationSummary {
        paraxial_focus_relative_residual,
        efl_mm: evaluation.efl_mm,
        reference_efl_mm: published_efl,
        efl_relative_error,
        on_axis_rms_mm: on_axis,
        reference_on_axis_rms_mm: published_spot,
        maximum_spot_relative_error,
        convergence,
        paraxial_pass: paraxial_focus_relative_residual < 1.0e-3,
        efl_pass: efl_relative_error < 1.0e-3,
        spot_pass: maximum_spot_relative_error < 0.20,
        convergence_pass: convergence_relative < 0.03,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sellmeier_indices_match_catalogue_d_line() {
        let sk16 = Glass::Sk16.refractive_index(0.5876).unwrap();
        let f2 = Glass::F2.refractive_index(0.5876).unwrap();
        assert!((sk16 - 1.620_408).abs() < 2.0e-6);
        assert!((f2 - 1.620_037).abs() < 2.0e-6);
    }

    #[test]
    fn vector_snell_preserves_normal_incidence() {
        let direction = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let normal = Vec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        };
        assert_eq!(refract(direction, normal, 1.0, 1.5).unwrap(), direction);
    }

    #[test]
    fn reference_design_is_finite_and_replayable() {
        let first = evaluate(&REFERENCE_DESIGN, 4).unwrap();
        let second = evaluate(&REFERENCE_DESIGN, 4).unwrap();
        assert_eq!(first.rms_spot_mm, second.rms_spot_mm);
        assert_eq!(first.lost_rays, 0);
        assert!(first.efl_mm > 45.0 && first.efl_mm < 55.0);
        assert!(first.minimum_edge_thickness_mm > 0.0);
    }

    #[test]
    fn published_reference_validation_passes() {
        let validation = validate_reference().unwrap();
        assert!(validation.efl_pass);
        assert!(validation.paraxial_pass);
        assert!(validation.spot_pass);
        assert!(validation.convergence_pass);
    }

    #[test]
    fn vignetting_and_bad_shapes_do_not_get_small_costs() {
        assert!(trace_ray(&Design::reference(), 6.0, 0.0, 0.0, 0.5876).is_none());
        assert!(evaluate(&[0.0], 4).is_none());
    }

    #[test]
    fn every_scalar_arm_improves_the_reference_seed() {
        let grid_radius = 3;
        let reference = evaluate(&REFERENCE_DESIGN, grid_radius)
            .unwrap()
            .scalar_score();
        for optimizer in SoOptimizer::ALL {
            let result = optimize_so(
                optimizer,
                &SoConfig {
                    evaluations_per_arm: 5_000,
                    retries: 4,
                    workers: 4,
                    seed: 42,
                    grid_radius,
                },
            )
            .unwrap();
            assert!(
                result.best.scalar_score() < reference,
                "{} failed to improve its evaluated reference seed",
                optimizer.name()
            );
        }
    }

    #[test]
    fn tiny_mode_run_is_well_formed() {
        let result = optimize_mo(&MoConfig {
            evaluations: 64,
            popsize: 32,
            workers: 2,
            seed: 7,
            grid_radius: 2,
        });
        // Tiny global runs are allowed to miss the narrow feasible EFL band,
        // but they must either return a feasible front or an explicit error.
        if let Ok(result) = result {
            assert!(!result.pareto.is_empty());
            assert!(
                result
                    .pareto
                    .iter()
                    .all(|point| point.evaluation.feasible())
            );
        }
    }

    #[test]
    fn smoke_mode_exposes_a_tradeoff_front() {
        let result = optimize_mo(&MoConfig {
            evaluations: 16_384,
            popsize: 256,
            workers: 4,
            seed: 42 ^ 0xA076_1D64_78BD_642F,
            grid_radius: 4,
        })
        .unwrap();
        assert!(
            result.pareto.len() >= 2,
            "the smoke protocol should retain a tradeoff, not one point"
        );
    }
}
