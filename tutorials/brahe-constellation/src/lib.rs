//! Ground-contact constellation design with Brahe and fcmaes.
//!
//! The default execution mode puts parallelism at the objective boundary:
//! fcmaes evaluates independent candidates while every Brahe access search is
//! serial. This avoids multiplying an outer optimizer pool by Brahe's access
//! pool. An explicit inner-parallel comparison mode is also supported.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use brahe::traits::{DStatePropagator, Identifiable};
use brahe::utils::set_num_threads;
use brahe::{
    AccessSearchConfig, AccessWindow, AngleFormat, DNumericalOrbitPropagator, ElevationConstraint,
    Epoch, ForceModelConfig, KeplerianPropagator, PointLocation, R_EARTH, StaticEOPProvider,
    TimeSystem, location_accesses, set_global_eop_provider, state_koe_to_eci,
};
use fcmaes_core::{
    Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, RetryBounds,
    RetryConfig, RetryImprovement, RetryRunResult, Rng, map_elites_batch_with_progress,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use nalgebra::{DVector, Vector6};

pub const DIMENSION: usize = 10;
pub const OBJECTIVES: usize = 3;
pub const CONSTRAINTS: usize = 1;
pub const SATELLITES: usize = 4;
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [450.0, 0.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [900.0, 1.0];

pub const VARIABLE_NAMES: [&str; DIMENSION] = [
    "altitude_km",
    "inclination_deg",
    "raan_1_deg",
    "raan_2_deg",
    "raan_3_deg",
    "raan_4_deg",
    "mean_anomaly_1_deg",
    "mean_anomaly_2_deg",
    "mean_anomaly_3_deg",
    "mean_anomaly_4_deg",
];

pub const DEFAULT_STATIONS: [&str; 6] = [
    "Svalbard",
    "Fairbanks",
    "Singapore",
    "Hartebeesthoek",
    "Cordoba",
    "Troll",
];

pub const LOWER_BOUNDS: [f64; DIMENSION] = [450.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
pub const UPPER_BOUNDS: [f64; DIMENSION] = [
    900.0, 100.0, 360.0, 360.0, 360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
];
pub const BASELINE_DESIGN: [f64; DIMENSION] = [
    600.0, 97.8, 0.0, 90.0, 180.0, 270.0, 0.0, 90.0, 180.0, 270.0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// fcmaes owns the worker pool; each Brahe access search is sequential.
    Outer,
    /// fcmaes evaluates one candidate at a time; Brahe parallelizes its
    /// location/satellite pairs.
    Inner,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub horizon_hours: f64,
    pub minimum_elevation_deg: f64,
    pub minimum_pass_seconds: f64,
    pub access_step_seconds: f64,
    pub provider: String,
    pub station_names: Vec<String>,
    pub parallelism: Parallelism,
    pub workers: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            horizon_hours: 24.0,
            minimum_elevation_deg: 10.0,
            minimum_pass_seconds: 180.0,
            access_step_seconds: 60.0,
            provider: "ksat".to_string(),
            station_names: DEFAULT_STATIONS.iter().map(ToString::to_string).collect(),
            parallelism: Parallelism::Outer,
            workers: 0,
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.horizon_hours.is_finite() || !(0.25..=168.0).contains(&self.horizon_hours) {
            return Err("horizon must be finite and lie in 0.25..=168 hours");
        }
        if !self.minimum_elevation_deg.is_finite()
            || !(0.0..=45.0).contains(&self.minimum_elevation_deg)
        {
            return Err("minimum elevation must lie in 0..=45 degrees");
        }
        if !self.minimum_pass_seconds.is_finite()
            || self.minimum_pass_seconds <= 0.0
            || self.minimum_pass_seconds > self.horizon_hours * 3600.0
        {
            return Err("minimum pass duration must be positive and shorter than the horizon");
        }
        if !self.access_step_seconds.is_finite()
            || !(1.0..=300.0).contains(&self.access_step_seconds)
        {
            return Err("access step must lie in 1..=300 seconds");
        }
        if self.provider.trim().is_empty() || self.station_names.is_empty() {
            return Err("provider and station selection must be non-empty");
        }
        Ok(())
    }

    pub fn required_passes_per_station(&self) -> usize {
        (2.0 * self.horizon_hours / 24.0).ceil().max(1.0) as usize
    }

    pub fn resolved_workers(&self) -> usize {
        if self.workers == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            self.workers
        }
    }

    pub fn outer_workers(&self) -> usize {
        match self.parallelism {
            Parallelism::Outer => self.workers,
            Parallelism::Inner => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Station {
    pub name: String,
    pub longitude_deg: f64,
    pub latitude_deg: f64,
}

#[derive(Clone, Debug)]
pub struct AccessPass {
    pub station: String,
    pub satellite: String,
    pub open_seconds: f64,
    pub close_seconds: f64,
    pub duration_seconds: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct StationMetrics {
    pub station: String,
    pub accepted_passes: usize,
    pub rejected_short_passes: usize,
    pub contact_hours: f64,
    pub maximum_gap_hours: f64,
}

#[derive(Clone, Debug)]
pub struct AccessMetrics {
    pub maximum_gap_hours: f64,
    pub total_contact_hours: f64,
    pub minimum_passes: usize,
    pub missing_passes: usize,
    pub minimum_accepted_pass_seconds: f64,
    pub altitude_cost: f64,
    pub plane_spread: f64,
    pub launch_complexity: f64,
    pub scalar_score: f64,
    pub stations: Vec<StationMetrics>,
    pub passes: Vec<AccessPass>,
}

impl AccessMetrics {
    pub fn objectives(&self) -> [f64; OBJECTIVES] {
        [
            self.maximum_gap_hours,
            -self.total_contact_hours,
            self.launch_complexity,
        ]
    }

    pub fn constraint(&self) -> f64 {
        self.missing_passes as f64
    }

    /// A compact, higher-is-better value used only to select one Pareto
    /// representative. The optimizer still sees the three separate objectives.
    pub fn quality(&self) -> f64 {
        self.total_contact_hours / ((1.0 + self.maximum_gap_hours) * (1.0 + self.launch_complexity))
    }
}

pub struct ConstellationModel {
    config: ModelConfig,
    epoch: Epoch,
    end: Epoch,
    constraint: ElevationConstraint,
    locations: Vec<PointLocation>,
    stations: Vec<Station>,
    search: AccessSearchConfig,
}

impl ConstellationModel {
    pub fn new(config: ModelConfig) -> Result<Self, Box<dyn Error>> {
        config.validate()?;

        // Static zero EOP values keep the experiment reproducible and fully
        // offline. They are sufficient for comparing candidate geometries.
        set_global_eop_provider(StaticEOPProvider::from_zero());
        if config.parallelism == Parallelism::Inner {
            set_num_threads(config.resolved_workers())?;
        }

        let wanted: HashMap<&str, usize> = config
            .station_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut selected: Vec<Option<PointLocation>> = vec![None; wanted.len()];
        for location in brahe::datasets::groundstations::load_groundstations(&config.provider)? {
            if let Some(name) = location.get_name()
                && let Some(&index) = wanted.get(name)
            {
                selected[index] = Some(location);
            }
        }
        let missing: Vec<&str> = selected
            .iter()
            .enumerate()
            .filter_map(|(index, station)| {
                station
                    .is_none()
                    .then_some(config.station_names[index].as_str())
            })
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "stations not found in provider '{}': {}",
                config.provider,
                missing.join(", ")
            )
            .into());
        }
        let locations: Vec<PointLocation> = selected.into_iter().flatten().collect();
        let stations = locations
            .iter()
            .map(|location| Station {
                name: location.get_name().unwrap_or("unnamed").to_string(),
                longitude_deg: location.lon(),
                latitude_deg: location.lat(),
            })
            .collect();
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = epoch + config.horizon_hours * 3600.0;
        let constraint = ElevationConstraint::new(Some(config.minimum_elevation_deg), None)?;
        let search = AccessSearchConfig {
            initial_time_step: config.access_step_seconds,
            adaptive_step: false,
            parallel: config.parallelism == Parallelism::Inner,
            // `None` uses the Brahe pool configured once above. Supplying a
            // value here would rebuild a Rayon pool on every objective call.
            num_threads: None,
            time_tolerance: 0.25,
            ..Default::default()
        };
        Ok(Self {
            config,
            epoch,
            end,
            constraint,
            locations,
            stations,
            search,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn stations(&self) -> &[Station] {
        &self.stations
    }

    pub fn evaluate(&self, values: &[f64]) -> Result<AccessMetrics, Box<dyn Error>> {
        validate_design(values)?;
        let propagators = self.keplerian_propagators(values)?;
        let windows = location_accesses(
            &self.locations,
            &propagators,
            self.epoch,
            self.end,
            &self.constraint,
            None,
            Some(&self.search),
        )?;
        Ok(self.aggregate(values, &windows))
    }

    /// Validate one finalist with numerical propagation and a 20x20 EGM2008
    /// gravity field. This is intentionally outside the optimization loop.
    pub fn evaluate_numerical(&self, values: &[f64]) -> Result<AccessMetrics, Box<dyn Error>> {
        validate_design(values)?;
        let elements = orbital_elements(values);
        let mut propagators = Vec::with_capacity(SATELLITES);
        for (index, koe) in elements.into_iter().enumerate() {
            let state = state_koe_to_eci(koe, AngleFormat::Degrees);
            let mut propagator = DNumericalOrbitPropagator::builder(
                self.epoch,
                DVector::from_column_slice(state.as_slice()),
                ForceModelConfig::earth_gravity(),
            )
            .build()?
            .with_name(&format!("SAT-{}", index + 1));
            propagator.propagate_to(self.end)?;
            propagators.push(propagator);
        }
        let windows = location_accesses(
            &self.locations,
            &propagators,
            self.epoch,
            self.end,
            &self.constraint,
            None,
            Some(&self.search),
        )?;
        Ok(self.aggregate(values, &windows))
    }

    fn keplerian_propagators(
        &self,
        values: &[f64],
    ) -> Result<Vec<KeplerianPropagator>, Box<dyn Error>> {
        orbital_elements(values)
            .into_iter()
            .enumerate()
            .map(|(index, elements)| {
                Ok(KeplerianPropagator::from_keplerian(
                    self.epoch,
                    elements,
                    AngleFormat::Degrees,
                    self.config.access_step_seconds,
                )?
                .with_name(&format!("SAT-{}", index + 1)))
            })
            .collect()
    }

    fn aggregate(&self, values: &[f64], windows: &[AccessWindow]) -> AccessMetrics {
        let horizon_seconds = self.config.horizon_hours * 3600.0;
        let mut passes = Vec::with_capacity(windows.len());
        let mut by_station: HashMap<&str, Vec<(f64, f64)>> = self
            .stations
            .iter()
            .map(|station| (station.name.as_str(), Vec::new()))
            .collect();
        let mut rejected: HashMap<&str, usize> = self
            .stations
            .iter()
            .map(|station| (station.name.as_str(), 0))
            .collect();

        for window in windows {
            let station = window.location_name.as_deref().unwrap_or("unnamed");
            let satellite = window.satellite_name.as_deref().unwrap_or("unnamed");
            let duration = window.duration();
            let accepted = duration + 1.0e-9 >= self.config.minimum_pass_seconds;
            let open = (window.window_open - self.epoch).clamp(0.0, horizon_seconds);
            let close = (window.window_close - self.epoch).clamp(0.0, horizon_seconds);
            if accepted {
                if let Some(intervals) = by_station.get_mut(station) {
                    intervals.push((open, close));
                }
            } else if let Some(count) = rejected.get_mut(station) {
                *count += 1;
            }
            passes.push(AccessPass {
                station: station.to_string(),
                satellite: satellite.to_string(),
                open_seconds: open,
                close_seconds: close,
                duration_seconds: duration,
                accepted,
            });
        }

        let required = self.config.required_passes_per_station();
        let mut maximum_gap_seconds: f64 = 0.0;
        let mut total_contact_seconds = 0.0;
        let mut minimum_passes = usize::MAX;
        let mut missing_passes = 0;
        let mut station_metrics = Vec::with_capacity(self.stations.len());
        for station in &self.stations {
            let intervals = by_station
                .get_mut(station.name.as_str())
                .expect("known station");
            intervals.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
            let merged = merge_intervals(intervals);
            // Simultaneous visibility of two satellites is one continuous
            // station contact opportunity, not two independent passes.
            let accepted_passes = merged.len();
            minimum_passes = minimum_passes.min(accepted_passes);
            missing_passes += required.saturating_sub(accepted_passes);
            let contact_seconds = merged.iter().map(|(open, close)| close - open).sum::<f64>();
            let mut station_max_gap: f64 = 0.0;
            let mut cursor = 0.0;
            for &(open, close) in &merged {
                station_max_gap = station_max_gap.max(open - cursor);
                cursor = cursor.max(close);
            }
            station_max_gap = station_max_gap.max(horizon_seconds - cursor);
            maximum_gap_seconds = maximum_gap_seconds.max(station_max_gap);
            total_contact_seconds += contact_seconds;
            station_metrics.push(StationMetrics {
                station: station.name.clone(),
                accepted_passes,
                rejected_short_passes: rejected[station.name.as_str()],
                contact_hours: contact_seconds / 3600.0,
                maximum_gap_hours: station_max_gap / 3600.0,
            });
        }
        if minimum_passes == usize::MAX {
            minimum_passes = 0;
        }
        let minimum_accepted_pass_seconds = passes
            .iter()
            .filter(|pass| pass.accepted)
            .map(|pass| pass.duration_seconds)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let altitude_cost = (values[0] - LOWER_BOUNDS[0]) / (UPPER_BOUNDS[0] - LOWER_BOUNDS[0]);
        let plane_spread = circular_spread(&values[2..6]);
        let launch_complexity = altitude_cost + 0.5 * plane_spread;
        let maximum_gap_hours = maximum_gap_seconds / 3600.0;
        let total_contact_hours = total_contact_seconds / 3600.0;
        // All undesirable terms are positive. A negative missing-pass or
        // altitude term would reward constraint violations or higher orbits.
        let scalar_score =
            maximum_gap_hours + 10.0 * missing_passes as f64 + 4.5 * altitude_cost + plane_spread;
        AccessMetrics {
            maximum_gap_hours,
            total_contact_hours,
            minimum_passes,
            missing_passes,
            minimum_accepted_pass_seconds,
            altitude_cost,
            plane_spread,
            launch_complexity,
            scalar_score,
            stations: station_metrics,
            passes,
        }
    }
}

fn validate_design(values: &[f64]) -> Result<(), &'static str> {
    if values.len() != DIMENSION {
        return Err("a constellation design must contain exactly ten values");
    }
    if values.iter().enumerate().any(|(index, value)| {
        !value.is_finite() || *value < LOWER_BOUNDS[index] || *value > UPPER_BOUNDS[index]
    }) {
        return Err("constellation design lies outside the supported bounds");
    }
    Ok(())
}

fn orbital_elements(values: &[f64]) -> [Vector6<f64>; SATELLITES] {
    std::array::from_fn(|satellite| {
        Vector6::new(
            R_EARTH + values[0] * 1_000.0,
            0.001,
            values[1],
            values[2 + satellite],
            0.0,
            values[6 + satellite],
        )
    })
}

fn circular_spread(angles_deg: &[f64]) -> f64 {
    let (cosine, sine) = angles_deg.iter().fold((0.0, 0.0), |(cosine, sine), angle| {
        let radians = angle.to_radians();
        (cosine + radians.cos(), sine + radians.sin())
    });
    1.0 - cosine.hypot(sine) / angles_deg.len().max(1) as f64
}

fn merge_intervals(intervals: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(intervals.len());
    for &(open, close) in intervals {
        if let Some(last) = merged.last_mut()
            && open <= last.1
        {
            last.1 = last.1.max(close);
            continue;
        }
        merged.push((open, close));
    }
    merged
}

pub fn scalar_objective(values: &[f64], model: &ConstellationModel) -> f64 {
    model
        .evaluate(values)
        .map_or(1.0e99, |metrics| metrics.scalar_score)
}

pub fn mode_objective(values: &[f64], model: &ConstellationModel) -> Vec<f64> {
    model.evaluate(values).map_or_else(
        |_| vec![1.0e99; OBJECTIVES + CONSTRAINTS],
        |metrics| {
            let objectives = metrics.objectives();
            vec![
                objectives[0],
                objectives[1],
                objectives[2],
                metrics.constraint(),
            ]
        },
    )
}

#[derive(Clone, Debug)]
pub struct ScalarOptions {
    pub evaluations_per_retry: u64,
    pub retries: usize,
    pub workers: usize,
    pub depth: i32,
    pub seed: u64,
}

impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            evaluations_per_retry: 250,
            retries: 8,
            workers: 0,
            depth: 6,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarOutcome {
    pub design: Vec<f64>,
    pub metrics: AccessMetrics,
    pub evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    model: &ConstellationModel,
    options: &ScalarOptions,
) -> Result<ScalarOutcome, Box<dyn Error>> {
    if options.evaluations_per_retry == 0 || options.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    if !(1..=36).contains(&options.depth) {
        return Err("BiteOpt depth must lie in 1..=36".into());
    }
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let objective = |values: &[f64]| scalar_objective(values, model);
    let retry_config = RetryConfig {
        num_retries: options.retries,
        workers: options.workers,
        capacity: options.retries.min(500),
        max_evaluations: options.evaluations_per_retry,
        seed: options.seed,
        statistic_num: 1_000,
        ..Default::default()
    };
    let started = Instant::now();
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let mut rng = Rng::new(context.seed);
        let random_guess: Vec<f64> = context
            .bounds
            .lower()
            .iter()
            .zip(context.bounds.upper())
            .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
            .collect();
        let guess = context.guess.as_deref().unwrap_or(&random_guess);
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(guess),
            &BiteParams {
                max_evaluations: context.max_evaluations,
                seed: rng.next_u64(),
                runid: context.run_id as i64,
                ..Default::default()
            },
            options.depth,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("BiteOpt retry returned no finite constellation design".into());
    }
    let metrics = model.evaluate(&result.x)?;
    Ok(ScalarOutcome {
        design: result.x,
        metrics,
        evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        improvements: result.improvements,
    })
}

#[derive(Clone, Debug)]
pub struct MultiOptions {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for MultiOptions {
    fn default() -> Self {
        Self {
            evaluations: 4_096,
            popsize: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub design: Vec<f64>,
    pub objectives: [f64; OBJECTIVES],
    pub quality: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
pub struct MultiOutcome {
    pub pareto: Vec<ParetoPoint>,
    pub representative: ParetoPoint,
    pub metrics: AccessMetrics,
    pub evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub convergence: Vec<MoProgress>,
    pub quality: f64,
}

pub fn optimize_multi(
    model: &ConstellationModel,
    options: &MultiOptions,
) -> Result<MultiOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("MODE evaluations must be positive".into());
    }
    if options.popsize < 4 || options.popsize > i32::MAX as usize {
        return Err("MODE population size must lie in 4..=i32::MAX".into());
    }
    let generations = options.evaluations.div_ceil(options.popsize);
    let evaluations = generations * options.popsize;
    let fitness = Fitness::bounded(
        DIMENSION,
        OBJECTIVES + CONSTRAINTS,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
    );
    let parameters = ModeParams {
        popsize: options.popsize as i32,
        nsga_update: true,
        seed: options.seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, OBJECTIVES, CONSTRAINTS, None, &parameters)?;
    let mut convergence = Vec::with_capacity(generations);
    let mut best_quality: f64 = 0.0;
    let started = Instant::now();
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, options.workers as i32, |values| {
            mode_objective(values, model)
        });
        for output in &ys {
            if output[OBJECTIVES] <= 0.0 {
                let quality = -output[1] / ((1.0 + output[0]) * (1.0 + output[2]));
                best_quality = best_quality.max(quality);
            }
        }
        mode.tell(&ys);
        convergence.push(MoProgress {
            evaluations: (generation + 1) * options.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality,
        });
    }

    let population = mode.population();
    let evaluated = parallel_batch(&population, options.workers as i32, |candidate| {
        model.evaluate(candidate).ok()
    });
    let feasible: Vec<(usize, &AccessMetrics)> = evaluated
        .iter()
        .enumerate()
        .filter_map(|(index, result)| {
            result
                .as_ref()
                .filter(|metrics| metrics.constraint() <= 0.0)
                .map(|metrics| (index, metrics))
        })
        .collect();
    if feasible.is_empty() {
        return Err(
            "MODE found no design satisfying the required accepted passes per station".into(),
        );
    }
    let objective_values: Vec<Vec<f64>> = feasible
        .iter()
        .map(|(_, metrics)| metrics.objectives().to_vec())
        .collect();
    let indices = pareto_indices(&objective_values, OBJECTIVES)?;
    let mut pareto = Vec::with_capacity(indices.len());
    for front_index in indices {
        let (population_index, metrics) = feasible[front_index];
        pareto.push(ParetoPoint {
            design: population[population_index].clone(),
            objectives: metrics.objectives(),
            quality: metrics.quality(),
        });
    }
    pareto.sort_by(|left, right| right.quality.total_cmp(&left.quality));
    let representative = pareto[0].clone();
    let metrics = model.evaluate(&representative.design)?;
    Ok(MultiOutcome {
        quality: representative.quality,
        pareto,
        representative,
        metrics,
        evaluations,
        generations,
        elapsed: started.elapsed(),
        convergence,
    })
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for QdOptions {
    fn default() -> Self {
        Self {
            evaluations: 4_096,
            capacity: 400,
            chunk_size: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdPoint {
    pub niche_id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub design: Vec<f64>,
    pub metrics: AccessMetrics,
    pub quality: f64,
    pub descriptors: [f64; 2],
    pub visit_count: u64,
}

#[derive(Clone, Debug)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct QdOutcome {
    pub elites: Vec<QdPoint>,
    pub representative: QdPoint,
    pub evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

pub fn qd_objective(values: &[f64], model: &ConstellationModel) -> (f64, [f64; 2]) {
    let Ok(metrics) = model.evaluate(values) else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    if metrics.constraint() > 0.0
        || !metrics.scalar_score.is_finite()
        || !metrics.plane_spread.is_finite()
    {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    }
    (metrics.scalar_score, [values[0], metrics.plane_spread])
}

struct ConstellationQdBatch<'a> {
    model: &'a ConstellationModel,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for ConstellationQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers as i32, |x| qd_objective(x, self.model));
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        let mut output = Vec::with_capacity(evaluated.len());
        for (quality, descriptors) in evaluated {
            if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
                self.invalid.fetch_add(1, Ordering::Relaxed);
            } else if descriptors
                .iter()
                .zip(QD_DESCRIPTOR_LOWER.iter().zip(QD_DESCRIPTOR_UPPER))
                .any(|(&value, (&lower, upper))| value < lower || value > upper)
            {
                self.clipped.fetch_add(1, Ordering::Relaxed);
            }
            output.push((quality, descriptors.to_vec()));
        }
        output
    }
}

pub fn optimize_qd(
    model: &ConstellationModel,
    options: &QdOptions,
) -> Result<QdOutcome, Box<dyn Error>> {
    if model.config().parallelism != Parallelism::Outer {
        return Err("MAP-Elites requires --parallel outer to avoid nested pools".into());
    }
    if options.evaluations == 0 {
        return Err("QD evaluations must be positive".into());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".into());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let generations = options.evaluations.div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size;
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER_BOUNDS, &UPPER_BOUNDS, &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = ConstellationQdBatch {
        model,
        workers: options.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        clipped: Arc::clone(&clipped),
    };
    let parameters = MapElitesParams {
        generations,
        chunk_size: options.chunk_size,
        use_sbx: false,
        ..Default::default()
    };
    let started = Instant::now();
    let mut convergence = Vec::with_capacity(generations);
    map_elites_batch_with_progress(
        &mut archive,
        &mut batch,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
        &parameters,
        &mut rng,
        &mut |_, archive| {
            let count = evaluations.load(Ordering::Relaxed);
            convergence.push(QdProgress {
                evaluations: count,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / count.max(1) as f64,
            });
        },
    )?;
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);
    let candidates = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .map(|index| (index, archive.xs()[index].clone()))
        .collect::<Vec<_>>();
    let metrics = parallel_batch(
        &candidates
            .iter()
            .map(|(_, values)| values.clone())
            .collect::<Vec<_>>(),
        options.workers as i32,
        |values| model.evaluate(values).ok(),
    );
    let mut elites = Vec::with_capacity(candidates.len());
    for ((niche_id, design), metrics) in candidates.into_iter().zip(metrics) {
        let metrics = metrics.ok_or("failed to re-evaluate a QD elite")?;
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            quality: archive.ys()[niche_id],
            descriptors: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            visit_count: archive.counts()[niche_id],
            design,
            metrics,
        });
    }
    elites.sort_by(|left, right| left.quality.total_cmp(&right.quality));
    let representative = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites found no feasible constellation")?;
    Ok(QdOutcome {
        representative,
        evaluations: actual_evaluations,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        elapsed: started.elapsed(),
        convergence,
        elites,
    })
}

pub fn write_qd_artifacts(
    directory: &Path,
    model: &ConstellationModel,
    outcome: &QdOutcome,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_artifacts(
        directory,
        model,
        &outcome.representative.design,
        &outcome.representative.metrics,
        &[],
        &[],
    )?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,descriptor_altitude_km_train,descriptor_plane_spread_train,visit_count",
    );
    for name in VARIABLE_NAMES {
        let _ = write!(archive, ",decision_{name}");
    }
    archive.push('\n');
    for point in &outcome.elites {
        let _ = write!(
            archive,
            "{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            point.visit_count,
        );
        for value in &point.design {
            let _ = write!(archive, ",{value}");
        }
        archive.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive)?;

    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &outcome.convergence {
        let _ = writeln!(
            convergence,
            "{},{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction,
        );
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let side = (outcome.capacity as f64).sqrt() as usize;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "brahe-constellation",
        "formulation": "qd",
        "strategy": "outer-fcmaes",
        "command": command,
        "seed": options.seed,
        "workers": if options.workers == 0 {
            model.config().resolved_workers()
        } else {
            options.workers
        },
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "simulation": {
            "horizon_hours": model.config().horizon_hours,
            "minimum_pass_seconds": model.config().minimum_pass_seconds,
            "access_step_seconds": model.config().access_step_seconds,
            "provider": model.config().provider,
            "stations": model.config().station_names
        },
        "descriptors": [
            {
                "column": "descriptor_altitude_km",
                "label": "Shared altitude",
                "unit": "km",
                "bounds": [QD_DESCRIPTOR_LOWER[0], QD_DESCRIPTOR_UPPER[0]]
            },
            {
                "column": "descriptor_plane_spread",
                "label": "Circular RAAN spread",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            }
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_label": "Feasible constellation score (lower is better)",
            "occupied": outcome.occupied,
            "coverage": outcome.occupied as f64 / outcome.capacity as f64,
            "qd_score": outcome.qd_score,
            "best_quality": outcome.representative.quality,
            "invalid_evaluations": outcome.invalid_evaluations,
            "clipped_descriptors": outcome.clipped_descriptors
        },
        "convergence_metrics": [
            "coverage", "qd_score", "best_quality", "invalid_fraction"
        ],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv",
            "access_windows": "access_windows.csv",
            "stations": "stations.csv",
            "design": "design.csv",
            "report": "report.html"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

pub fn write_artifacts(
    directory: &Path,
    model: &ConstellationModel,
    design: &[f64],
    metrics: &AccessMetrics,
    convergence: &[MoProgress],
    pareto: &[ParetoPoint],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;

    let mut design_csv = String::from("variable,value\n");
    for (name, value) in VARIABLE_NAMES.iter().zip(design) {
        let _ = writeln!(design_csv, "{name},{value:.12}");
    }
    fs::write(directory.join("design.csv"), design_csv)?;

    let mut stations_csv = String::from(
        "station,longitude_deg,latitude_deg,accepted_passes,rejected_short_passes,contact_hours,maximum_gap_hours\n",
    );
    for (station, result) in model.stations().iter().zip(&metrics.stations) {
        let _ = writeln!(
            stations_csv,
            "{},{:.8},{:.8},{},{},{:.9},{:.9}",
            station.name,
            station.longitude_deg,
            station.latitude_deg,
            result.accepted_passes,
            result.rejected_short_passes,
            result.contact_hours,
            result.maximum_gap_hours,
        );
    }
    fs::write(directory.join("stations.csv"), stations_csv)?;

    let mut passes_csv =
        String::from("station,satellite,open_seconds,close_seconds,duration_seconds,accepted\n");
    for pass in &metrics.passes {
        let _ = writeln!(
            passes_csv,
            "{},{},{:.6},{:.6},{:.6},{}",
            pass.station,
            pass.satellite,
            pass.open_seconds,
            pass.close_seconds,
            pass.duration_seconds,
            pass.accepted,
        );
    }
    fs::write(directory.join("access_windows.csv"), passes_csv)?;

    let mut convergence_csv = String::from("evaluations,elapsed_seconds,best_quality\n");
    for sample in convergence {
        let _ = writeln!(
            convergence_csv,
            "{},{:.12},{:.12}",
            sample.evaluations, sample.elapsed_seconds, sample.best_quality
        );
    }
    fs::write(directory.join("convergence.csv"), convergence_csv)?;

    let mut pareto_csv = String::from(
        "point_id,feasible,selected,objective_maximum_gap_hours,objective_negative_contact_hours,objective_launch_complexity,quality",
    );
    for name in VARIABLE_NAMES {
        let _ = write!(pareto_csv, ",decision_{name}");
    }
    pareto_csv.push('\n');
    for (index, point) in pareto.iter().enumerate() {
        let _ = writeln!(
            pareto_csv,
            "{index},1,{},{},{},{},{},{}",
            u8::from(index == 0),
            point.objectives[0],
            point.objectives[1],
            point.objectives[2],
            point.quality,
            point
                .design
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    fs::write(directory.join("pareto.csv"), pareto_csv)?;
    write_report_html(&directory.join("report.html"), model, metrics, convergence)?;
    Ok(())
}

fn json_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn write_report_html(
    path: &Path,
    model: &ConstellationModel,
    metrics: &AccessMetrics,
    convergence: &[MoProgress],
) -> Result<(), Box<dyn Error>> {
    let mut stations = String::from("[");
    for station in model.stations() {
        let _ = write!(
            stations,
            "[{},{:.8},{:.8}],",
            json_string(&station.name),
            station.longitude_deg,
            station.latitude_deg
        );
    }
    stations.push(']');
    let mut passes = String::from("[");
    for pass in metrics.passes.iter().filter(|pass| pass.accepted) {
        let _ = write!(
            passes,
            "[{},{},{:.6},{:.6}],",
            json_string(&pass.station),
            json_string(&pass.satellite),
            pass.open_seconds / 3600.0,
            pass.close_seconds / 3600.0,
        );
    }
    passes.push(']');
    let mut convergence_data = String::from("[");
    for sample in convergence {
        let _ = write!(
            convergence_data,
            "[{},{:.12}],",
            sample.evaluations, sample.best_quality
        );
    }
    convergence_data.push(']');
    let horizon = model.config().horizon_hours;
    let html = format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fcmaes + Brahe constellation report</title>
<style>
body{{margin:0;background:#08131f;color:#e9f1f7;font:15px system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:24px}} canvas{{width:100%;height:auto;background:#10243a;border-radius:10px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}}
.card{{background:#10243a;border-radius:10px;padding:14px}}.value{{font-size:1.6rem;color:#68d9c0}}
</style></head><body><main>
<h1>Constellation ground-contact design</h1>
<div class="cards">
<div class="card">Worst station gap<div class="value">{:.3} h</div></div>
<div class="card">Network contact<div class="value">{:.3} h</div></div>
<div class="card">Minimum accepted passes<div class="value">{}</div></div>
<div class="card">Pareto quality<div class="value">{:.4}</div></div>
</div>
<h2>Selected KSAT network</h2><canvas id="map" width="1120" height="430"></canvas>
<h2>Accepted access timeline</h2><canvas id="timeline" width="1120" height="500"></canvas>
<h2>Optimization progress</h2><canvas id="progress" width="1120" height="280"></canvas>
<script>
const stations={stations},passes={passes},progress={convergence_data},horizon={horizon};
const colors=["#68d9c0","#f3b562","#e76f91","#8ab4f8"];
{{const c=document.getElementById("map"),g=c.getContext("2d");g.fillStyle="#173653";g.fillRect(35,25,c.width-70,c.height-55);
g.strokeStyle="#31546f";for(let lon=-120;lon<=120;lon+=60){{let x=35+(lon+180)/360*(c.width-70);g.beginPath();g.moveTo(x,25);g.lineTo(x,c.height-30);g.stroke()}}
for(let lat=-60;lat<=60;lat+=30){{let y=25+(90-lat)/180*(c.height-55);g.beginPath();g.moveTo(35,y);g.lineTo(c.width-35,y);g.stroke()}}
g.font="14px system-ui";stations.forEach(([name,lon,lat])=>{{const x=35+(lon+180)/360*(c.width-70),y=25+(90-lat)/180*(c.height-55);g.fillStyle="#68d9c0";g.beginPath();g.arc(x,y,5,0,Math.PI*2);g.fill();g.fillStyle="#e9f1f7";g.fillText(name,x+8,y-7)}});}}
{{const c=document.getElementById("timeline"),g=c.getContext("2d"),names=stations.map(s=>s[0]),top=35,row=(c.height-65)/names.length;
g.font="13px system-ui";names.forEach((name,i)=>{{const y=top+i*row;g.fillStyle="#dce8f1";g.fillText(name,8,y+10);g.strokeStyle="#31546f";g.beginPath();g.moveTo(120,y+5);g.lineTo(c.width-25,y+5);g.stroke()}});
passes.forEach(([station,sat,open,close])=>{{const i=names.indexOf(station),s=Number(sat.split("-")[1])-1,y=top+i*row;g.fillStyle=colors[s%colors.length];const x=120+open/horizon*(c.width-145),w=Math.max(2,(close-open)/horizon*(c.width-145));g.fillRect(x,y-6,w,22)}});
g.fillStyle="#dce8f1";g.fillText("0 h",120,c.height-12);g.fillText(horizon+" h",c.width-60,c.height-12);}}
{{const c=document.getElementById("progress"),g=c.getContext("2d");if(progress.length>1){{const xmax=progress.at(-1)[0],ys=progress.map(p=>p[1]),ymin=Math.min(...ys),ymax=Math.max(...ys),X=v=>50+v/xmax*(c.width-75),Y=v=>20+(ymax-v)/Math.max(1e-12,ymax-ymin)*(c.height-55);g.strokeStyle="#68d9c0";g.lineWidth=2;g.beginPath();progress.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();g.fillStyle="#dce8f1";g.fillText("evaluations",c.width-95,c.height-10);g.fillText("quality",8,18)}}}}
</script></main></body></html>"##,
        metrics.maximum_gap_hours,
        metrics.total_contact_hours,
        metrics.minimum_passes,
        metrics.quality(),
    );
    fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_model() -> ConstellationModel {
        ConstellationModel::new(ModelConfig {
            horizon_hours: 0.25,
            station_names: vec!["Svalbard".to_string(), "Singapore".to_string()],
            parallelism: Parallelism::Outer,
            workers: 2,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn bounds_and_baseline_are_valid() {
        validate_design(&BASELINE_DESIGN).unwrap();
        assert!(validate_design(&BASELINE_DESIGN[..9]).is_err());
        let mut invalid = BASELINE_DESIGN;
        invalid[0] = 100.0;
        assert!(validate_design(&invalid).is_err());
    }

    #[test]
    fn interval_union_handles_overlap_and_gaps() {
        assert_eq!(
            merge_intervals(&[(0.0, 2.0), (1.0, 4.0), (6.0, 7.0)]),
            vec![(0.0, 4.0), (6.0, 7.0)]
        );
    }

    #[test]
    fn plane_spread_distinguishes_one_and_four_planes() {
        assert!(circular_spread(&[0.0, 0.0, 0.0, 0.0]) < 1.0e-12);
        assert!((circular_spread(&[0.0, 90.0, 180.0, 270.0]) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn embedded_station_evaluation_is_finite_and_reproducible() {
        let model = quick_model();
        let first = model.evaluate(&BASELINE_DESIGN).unwrap();
        let second = model.evaluate(&BASELINE_DESIGN).unwrap();
        assert!(first.scalar_score.is_finite());
        assert_eq!(first.stations.len(), 2);
        assert_eq!(first.scalar_score, second.scalar_score);
        assert_eq!(first.passes.len(), second.passes.len());
    }

    #[test]
    fn mode_adapter_appends_a_nonnegative_constraint() {
        let output = mode_objective(&BASELINE_DESIGN, &quick_model());
        assert_eq!(output.len(), OBJECTIVES + CONSTRAINTS);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output[OBJECTIVES] >= 0.0);
    }

    #[test]
    fn qd_adapter_uses_feasible_architecture_descriptors() {
        let model = ConstellationModel::new(ModelConfig::default()).unwrap();
        let (quality, descriptors) = qd_objective(&BASELINE_DESIGN, &model);
        assert!(quality.is_finite());
        assert_eq!(descriptors[0], BASELINE_DESIGN[0]);
        assert!((descriptors[1] - 1.0).abs() < 1.0e-12);
        assert!(
            optimize_qd(
                &model,
                &QdOptions {
                    capacity: 15,
                    ..Default::default()
                }
            )
            .is_err()
        );

        let outcome = optimize_qd(
            &model,
            &QdOptions {
                evaluations: 32,
                capacity: 4,
                chunk_size: 8,
                workers: 2,
                seed: 9,
            },
        )
        .unwrap();
        assert_eq!(outcome.evaluations, 32);
        assert_eq!(outcome.capacity, 4);
        assert!(!outcome.elites.is_empty());
    }

    #[test]
    fn configuration_rejects_invalid_inputs_and_unknown_station() {
        assert!(
            ConstellationModel::new(ModelConfig {
                horizon_hours: 0.0,
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            ConstellationModel::new(ModelConfig {
                station_names: vec!["not-a-station".to_string()],
                ..Default::default()
            })
            .is_err()
        );
    }
}
