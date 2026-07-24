use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::model::{
    DESIGN_NAMES, Dataset, Design, Metrics, Split, concentration_ug_m3, evaluate_training,
    evaluate_validation,
};
use crate::optimize::{
    MultiOptions, MultiOutcome, QdOptions, QdOutcome, ScalarOptions, ScalarOutcome,
};

fn effective_workers(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    }
}

fn write_observations(directory: &Path, dataset: &Dataset) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "split,sensor_id,weather_id,sensor_x_m,sensor_y_m,sensor_height_m,wind_speed_m_s,wind_direction_deg,stability,measured_ug_m3\n",
    );
    for split in [Split::Training, Split::Validation] {
        for observation in dataset.observations(split) {
            writeln!(
                csv,
                "{},{},{},{},{},{},{},{},{},{}",
                split.name(),
                observation.sensor.id,
                observation.weather.id,
                observation.sensor.x_m,
                observation.sensor.y_m,
                observation.sensor.height_m,
                observation.weather.speed_m_s,
                observation.weather.direction_deg,
                char::from(observation.weather.stability),
                observation.measured_ug_m3,
            )?;
        }
    }
    fs::write(directory.join("observations.csv"), csv)?;
    Ok(())
}

fn append_summary_row(
    csv: &mut String,
    label: &str,
    split: Split,
    metrics: &Metrics,
    design: &Design,
) -> Result<(), std::fmt::Error> {
    write!(
        csv,
        "{label},{},{},{},{},{},{},{},{}",
        split.name(),
        metrics.observations,
        metrics.mean_huber_error,
        metrics.p95_log_error,
        metrics.detection_mismatch_fraction,
        metrics.total_emission_g_s,
        metrics.scalar_score,
        metrics.source_position_error_m,
    )?;
    for value in design.values() {
        write!(csv, ",{value}")?;
    }
    csv.push('\n');
    Ok(())
}

fn write_summary(
    directory: &Path,
    dataset: &Dataset,
    selected: &Design,
) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "design,split,observations,mean_huber_error,p95_log_error,detection_mismatch_fraction,total_emission_g_s,scalar_score,source_position_error_m",
    );
    for name in DESIGN_NAMES {
        write!(csv, ",decision_{name}")?;
    }
    csv.push('\n');
    for (label, design) in [
        ("baseline", Design::baseline()),
        ("truth", Design::truth()),
        ("selected", selected.clone()),
    ] {
        append_summary_row(
            &mut csv,
            label,
            Split::Training,
            &evaluate_training(design.values(), dataset)?,
            &design,
        )?;
        append_summary_row(
            &mut csv,
            label,
            Split::Validation,
            &evaluate_validation(design.values(), dataset)?,
            &design,
        )?;
    }
    fs::write(directory.join("summary.csv"), csv)?;
    Ok(())
}

fn color(value: f64) -> String {
    let bounded = value.clamp(0.0, 1.0);
    let red = (35.0 + 220.0 * bounded) as u8;
    let green = (55.0 + 160.0 * bounded.sqrt()) as u8;
    let blue = (125.0 - 105.0 * bounded) as u8;
    format!("#{red:02x}{green:02x}{blue:02x}")
}

fn write_source_map(
    directory: &Path,
    dataset: &Dataset,
    selected: &Design,
) -> Result<(), Box<dyn Error>> {
    const CELLS: usize = 48;
    const MIN: f64 = -2_500.0;
    const MAX: f64 = 2_500.0;
    const PLOT: f64 = 624.0;
    const LEFT: f64 = 64.0;
    const TOP: f64 = 42.0;
    let weather = dataset.weather(Split::Training);
    let step = (MAX - MIN) / CELLS as f64;
    let mut grid = Vec::with_capacity(CELLS * CELLS);
    for row in 0..CELLS {
        let y = MIN + (row as f64 + 0.5) * step;
        for column in 0..CELLS {
            let x = MIN + (column as f64 + 0.5) * step;
            let sensor = crate::model::Sensor::new(usize::MAX, x, y, 2.0);
            let average = weather
                .iter()
                .map(|&hour| concentration_ug_m3(selected, sensor, hour))
                .sum::<f64>()
                / weather.len() as f64;
            grid.push(average);
        }
    }
    let maximum = grid.iter().copied().fold(0.0_f64, f64::max);
    let scale = maximum.max(1.0e-12).ln_1p();
    let pixel = PLOT / CELLS as f64;
    let project_x = |x: f64| LEFT + (x - MIN) / (MAX - MIN) * PLOT;
    let project_y = |y: f64| TOP + (MAX - y) / (MAX - MIN) * PLOT;
    let mut svg = r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="780" height="730" viewBox="0 0 780 730">
<rect width="780" height="730" fill="#f8fafc"/>
<text x="390" y="25" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">Mean predicted ground concentration over training weather</text>
<g shape-rendering="crispEdges">
"##
    .to_string();
    for row in 0..CELLS {
        for column in 0..CELLS {
            let value = grid[row * CELLS + column];
            let normalized = value.ln_1p() / scale;
            writeln!(
                svg,
                r#"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}"/>"#,
                LEFT + column as f64 * pixel,
                TOP + (CELLS - 1 - row) as f64 * pixel,
                pixel + 0.1,
                pixel + 0.1,
                color(normalized),
            )?;
        }
    }
    svg.push_str("</g>\n");
    let mut sensors = BTreeSet::new();
    for observation in dataset.training() {
        if sensors.insert(observation.sensor.id) {
            writeln!(
                svg,
                r##"<circle cx="{:.3}" cy="{:.3}" r="4" fill="#ffffff" stroke="#111827" stroke-width="1.5"/>"##,
                project_x(observation.sensor.x_m),
                project_y(observation.sensor.y_m),
            )?;
        }
    }
    for source in dataset.truth().sources() {
        let x = project_x(source.x_m);
        let y = project_y(source.y_m);
        writeln!(
            svg,
            r##"<path d="M {:.3} {:.3} l 12 12 m -12 0 l 12 -12" stroke="#111827" stroke-width="3"/>"##,
            x - 6.0,
            y - 6.0,
        )?;
    }
    for source in selected.sources() {
        writeln!(
            svg,
            r##"<circle cx="{:.3}" cy="{:.3}" r="8" fill="none" stroke="#d81b60" stroke-width="3"/>"##,
            project_x(source.x_m),
            project_y(source.y_m),
        )?;
    }
    svg.push_str(
        r##"<rect x="64" y="42" width="624" height="624" fill="none" stroke="#111827" stroke-width="1.5"/>
<text x="376" y="704" text-anchor="middle" font-family="sans-serif" font-size="14">Easting [m]</text>
<text x="18" y="354" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 354)">Northing [m]</text>
<circle cx="712" cy="82" r="4" fill="#fff" stroke="#111827" stroke-width="1.5"/>
<text x="724" y="87" font-family="sans-serif" font-size="12">sensor</text>
<path d="M 706 105 l 12 12 m -12 0 l 12 -12" stroke="#111827" stroke-width="3"/>
<text x="724" y="116" font-family="sans-serif" font-size="12">truth</text>
<circle cx="712" cy="142" r="8" fill="none" stroke="#d81b60" stroke-width="3"/>
<text x="724" y="147" font-family="sans-serif" font-size="12">inferred</text>
</svg>
"##,
    );
    fs::write(directory.join("source-map.svg"), svg)?;
    Ok(())
}

fn write_common(
    directory: &Path,
    dataset: &Dataset,
    selected: &Design,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    write_observations(directory, dataset)?;
    write_summary(directory, dataset, selected)?;
    write_source_map(directory, dataset, selected)?;
    Ok(())
}

fn scalar_requested_evaluations(options: &ScalarOptions) -> usize {
    if options.retries <= 1 {
        return options.evaluations_per_retry as usize;
    }
    (0..options.retries)
        .map(|run| {
            let progress = run as f64 / (options.retries - 1) as f64;
            let factor = 1.0 + (options.max_eval_fac - 1.0) * progress;
            (options.evaluations_per_retry as f64 * factor).round() as usize
        })
        .sum()
}

pub fn write_scalar_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &ScalarOutcome,
    options: &ScalarOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_common(directory, dataset, &outcome.design)?;
    let mut convergence = String::from("evaluations,elapsed_seconds,best_quality\n");
    for sample in &outcome.improvements {
        writeln!(
            convergence,
            "{},{},{}",
            sample.evaluations, sample.elapsed_seconds, -sample.value
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "dispersion-source-localization",
        "formulation": "scalar",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": scalar_requested_evaluations(options),
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "simulation": {
            "training_observations": dataset.training().len(),
            "validation_observations": dataset.validation().len(),
            "educational_model": "ISC-3-derived Gaussian plume with Briggs plume rise",
            "non_regulatory": true
        },
        "optimizer": {
            "algorithm": "BiteOpt coordinated advanced retry",
            "retries": options.retries,
            "initial_evaluations_per_retry": options.evaluations_per_retry,
            "max_eval_fac": options.max_eval_fac,
            "depth": options.depth
        },
        "objectives": [],
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": {
            "convergence": "convergence.csv",
            "observations": "observations.csv",
            "summary": "summary.csv",
            "source_map": "source-map.svg"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

pub fn write_multi_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &MultiOutcome,
    options: &MultiOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_common(directory, dataset, &outcome.representative.design)?;
    let mut convergence = String::from("evaluations,elapsed_seconds,best_quality\n");
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{}",
            sample.evaluations, sample.elapsed_seconds, sample.best_quality
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_mean_huber_error,objective_p95_detection_error,objective_total_emission_g_s",
    );
    for name in DESIGN_NAMES {
        write!(pareto, ",decision_{name}")?;
    }
    pareto.push('\n');
    for (index, point) in outcome.pareto.iter().enumerate() {
        write!(
            pareto,
            "{index},1,{},{},{},{}",
            u8::from(index == 0),
            point.objectives[0],
            point.objectives[1],
            point.objectives[2],
        )?;
        for value in point.design.values() {
            write!(pareto, ",{value}")?;
        }
        pareto.push('\n');
    }
    fs::write(directory.join("pareto.csv"), pareto)?;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "dispersion-source-localization",
        "formulation": "mo",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "simulation": {
            "training_observations": dataset.training().len(),
            "validation_observations": dataset.validation().len(),
            "educational_model": "ISC-3-derived Gaussian plume with Briggs plume rise",
            "non_regulatory": true
        },
        "objectives": [
            {
                "column": "objective_mean_huber_error",
                "label": "Mean Huber error"
            },
            {
                "column": "objective_p95_detection_error",
                "label": "P95 + detection error"
            },
            {
                "column": "objective_total_emission_g_s",
                "label": "Total emission",
                "unit": "g/s"
            }
        ],
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": {
            "pareto": "pareto.csv",
            "convergence": "convergence.csv",
            "observations": "observations.csv",
            "summary": "summary.csv",
            "source_map": "source-map.svg"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

pub fn write_qd_artifacts(
    directory: &Path,
    dataset: &Dataset,
    outcome: &QdOutcome,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_common(directory, dataset, &outcome.representative.design)?;
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &outcome.convergence {
        writeln!(
            convergence,
            "{},{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction,
        )?;
    }
    fs::write(directory.join("convergence.csv"), convergence)?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,descriptor_centroid_x_train,descriptor_centroid_y_train,descriptor_centroid_x_validation,descriptor_centroid_y_validation,visit_count",
    );
    for name in DESIGN_NAMES {
        write!(archive, ",decision_{name}")?;
    }
    archive.push('\n');
    for point in &outcome.elites {
        write!(
            archive,
            "{},{},{},{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality_train,
            point.quality_validation,
            point.descriptors[0],
            point.descriptors[1],
            point.descriptors[0],
            point.descriptors[1],
            point.visit_count,
        )?;
        for value in point.design.values() {
            write!(archive, ",{value}")?;
        }
        archive.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive)?;
    let side = (outcome.capacity as f64).sqrt() as usize;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "dispersion-source-localization",
        "formulation": "qd",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "validation_elapsed_seconds": outcome.validation_elapsed.as_secs_f64(),
        "simulation": {
            "training_observations": dataset.training().len(),
            "validation_observations": dataset.validation().len(),
            "educational_model": "ISC-3-derived Gaussian plume with Briggs plume rise",
            "non_regulatory": true
        },
        "descriptors": [
            {
                "column": "descriptor_centroid_x",
                "label": "Source-centroid easting",
                "unit": "m",
                "bounds": [-1800.0, 1800.0]
            },
            {
                "column": "descriptor_centroid_y",
                "label": "Source-centroid northing",
                "unit": "m",
                "bounds": [-1800.0, 1800.0]
            }
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_validation_column": "quality_validation",
            "quality_label": "Robust score (lower is better)",
            "occupied": outcome.occupied,
            "coverage": outcome.occupied as f64 / outcome.capacity as f64,
            "qd_score": outcome.qd_score,
            "best_quality": outcome.representative.quality_train,
            "invalid_evaluations": outcome.invalid_evaluations,
            "clipped_descriptors": outcome.clipped_descriptors,
            "validation_evaluations": outcome.validation_evaluations
        },
        "convergence_metrics": [
            "coverage", "qd_score", "best_quality", "invalid_fraction"
        ],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv",
            "observations": "observations.csv",
            "summary": "summary.csv",
            "source_map": "source-map.svg"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::{MultiOptions, optimize_multi};

    #[test]
    fn writes_schema_artifacts() {
        let dataset = Dataset::synthetic();
        let outcome = optimize_multi(
            &dataset,
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 1,
                seed: 11,
            },
        )
        .unwrap();
        let directory =
            std::env::temp_dir().join(format!("fcmaes-dispersion-output-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        write_multi_artifacts(
            &directory,
            &dataset,
            &outcome,
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 1,
                seed: 11,
            },
            "test",
        )
        .unwrap();
        for name in [
            "run.json",
            "pareto.csv",
            "convergence.csv",
            "observations.csv",
            "summary.csv",
            "source-map.svg",
        ] {
            assert!(directory.join(name).is_file(), "missing {name}");
        }
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(directory.join("run.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["formulation"], "mo");
        fs::remove_dir_all(directory).unwrap();
    }
}
