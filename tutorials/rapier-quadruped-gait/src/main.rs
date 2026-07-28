use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use rapier_quadruped_gait::{
    DIMENSION, Gait, QdConfig, QdResult, RangeRow, Rollout, RolloutConfig, ScalarConfig,
    ScalarResult, optimize_qd, optimize_scalar, range_study, rollout,
};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Range,
    Scalar,
    Qd,
    All,
    Simulate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preset {
    Smoke,
    Publication,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    preset: Preset,
    workers: i32,
    seed: u64,
    output: PathBuf,
    write_output: bool,
    gait: Option<Gait>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            seed: 42,
            output: PathBuf::from("results"),
            write_output: true,
            gait: None,
        }
    }
}

#[derive(Clone, Copy)]
struct Protocol {
    range_samples: usize,
    scalar_evaluations: u64,
    scalar_retries: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk: usize,
}

fn protocol(preset: Preset) -> Protocol {
    match preset {
        Preset::Smoke => Protocol {
            range_samples: 64,
            scalar_evaluations: 256,
            scalar_retries: 2,
            qd_evaluations: 256,
            qd_capacity: 100,
            qd_chunk: 32,
        },
        Preset::Publication => Protocol {
            range_samples: 2_000,
            scalar_evaluations: 50_000,
            scalar_retries: 20,
            qd_evaluations: 50_000,
            qd_capacity: 400,
            qd_chunk: 128,
        },
    }
}

fn usage() {
    println!(
        "Rapier quadruped gait optimization\n\n\
         cargo run --release -- [OPTIONS]\n\n\
         --mode NAME       range, scalar, qd, all, or simulate (all)\n\
         --preset NAME     smoke or publication (smoke)\n\
         --workers N       fcmaes candidate workers; 0 uses CPUs (0)\n\
         --seed N          Optimizer root seed (42)\n\
         --output DIR      Artifact root (results)\n\
         --x CSV           25 gait values for simulate mode\n\
         --no-output       Run without writing artifacts\n\
         -h, --help        Show this help"
    );
}

fn parse_gait(text: &str) -> Result<Gait, Box<dyn Error>> {
    let values = text
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Gait::from_slice(&values)?)
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| {
            arguments
                .next()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match value("--mode")?.as_str() {
                    "range" => Mode::Range,
                    "scalar" => Mode::Scalar,
                    "qd" => Mode::Qd,
                    "all" => Mode::All,
                    "simulate" => Mode::Simulate,
                    other => return Err(format!("unknown mode: {other}").into()),
                }
            }
            "--preset" => {
                parsed.preset = match value("--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    other => return Err(format!("unknown preset: {other}").into()),
                }
            }
            "--workers" => parsed.workers = value("--workers")?.parse()?,
            "--seed" => parsed.seed = value("--seed")?.parse()?,
            "--output" => parsed.output = value("--output")?.into(),
            "--x" => parsed.gait = Some(parse_gait(&value("--x")?)?),
            "--no-output" => parsed.write_output = false,
            other => return Err(format!("unknown option: {other}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    Ok(Some(parsed))
}

fn command_line() -> String {
    let tail = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if tail.is_empty() {
        "cargo run --release".to_owned()
    } else {
        format!("cargo run --release -- {tail}")
    }
}

fn write(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

fn rollout_fields(row: &Rollout) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{}",
        usize::from(row.feasible),
        row.forward_distance_m,
        row.lateral_drift_m,
        row.mechanical_work_j,
        row.duty_factor,
        row.body_height_std_mm,
        row.minimum_torso_height_m,
        row.terrain_contact_steps,
        row.fall_constraint_m,
        row.drift_constraint_m,
        row.score
    )
}

const ROLLOUT_HEADER: &str = "feasible,forward_distance_m,lateral_drift_m,mechanical_work_j,descriptor_duty_factor,descriptor_body_height_std_mm,minimum_torso_height_m,terrain_contact_steps,constraint_fall_m,constraint_drift_m,score";

fn write_replay(path: &Path, rollout: &Rollout) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "time_s,torso_x_m,torso_y_m,torso_z_m,front_left,front_right,rear_left,rear_right\n",
    );
    for row in &rollout.replay {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{}",
            row.time_s,
            row.torso[0],
            row.torso[1],
            row.torso[2],
            usize::from(row.contacts[0]),
            usize::from(row.contacts[1]),
            usize::from(row.contacts[2]),
            usize::from(row.contacts[3])
        )?;
    }
    write(path, &csv)
}

fn write_range(path: &Path, rows: &[RangeRow]) -> Result<(), Box<dyn Error>> {
    let mut csv = format!("sample,{ROLLOUT_HEADER}");
    for index in 0..DIMENSION {
        write!(csv, ",decision_{index}")?;
    }
    csv.push('\n');
    for row in rows {
        write!(csv, "{},{}", row.sample, rollout_fields(&row.rollout))?;
        for value in row.gait.values {
            write!(csv, ",{value}")?;
        }
        csv.push('\n');
    }
    write(path, &csv)
}

fn write_scalar(
    directory: &Path,
    command: &str,
    args: &Args,
    result: &ScalarResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut best = ROLLOUT_HEADER.to_string();
    for index in 0..DIMENSION {
        write!(best, ",decision_{index}")?;
    }
    best.push('\n');
    best.push_str(&rollout_fields(&result.rollout));
    for value in result.gait.values {
        write!(best, ",{value}")?;
    }
    best.push('\n');
    let mut convergence = String::from("evaluations,elapsed_seconds,best_quality\n");
    for row in &result.improvements {
        writeln!(
            convergence,
            "{},{},{}",
            row.evaluations, row.elapsed_seconds, row.value
        )?;
    }
    write(&directory.join("best.csv"), &best)?;
    write(&directory.join("convergence.csv"), &convergence)?;
    write_replay(&directory.join("replay.csv"), &result.rollout)?;
    write(
        &directory.join("run.json"),
        &(serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "rapier-quadruped-gait",
            "formulation": "scalar-bite-retry",
            "command": command,
            "seed": args.seed,
            "simulation_seed": 17,
            "workers": args.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "score", "label": "Negative distance plus motor work", "unit": "m + 0.002 J"}],
            "descriptors": [],
            "constraints": [
                {"column": "constraint_fall_m", "label": "Fall-height deficit", "unit": "m", "feasible": "<= 0"},
                {"column": "constraint_drift_m", "label": "Lateral-drift excess", "unit": "m", "feasible": "<= 0"}
            ],
            "rollout": {"duration_s": 4.0, "settle_s": 1.0, "time_step_s": 1.0 / 240.0},
            "artifacts": {"best": "best.csv", "convergence": "convergence.csv", "replay": "replay.csv"}
        }))? + "\n"),
    )
}

fn write_qd(
    directory: &Path,
    command: &str,
    args: &Args,
    result: &QdResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let metric_names = [
        "feasible",
        "forward_distance_m",
        "lateral_drift_m",
        "mechanical_work_j",
        "descriptor_duty_factor",
        "descriptor_body_height_std_mm",
        "minimum_torso_height_m",
        "terrain_contact_steps",
        "constraint_fall_m",
        "constraint_drift_m",
        "score",
    ];
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,validation_feasible_fraction,visit_count",
    );
    for suffix in ["train", "validation"] {
        for metric in metric_names {
            write!(archive, ",{metric}_{suffix}")?;
        }
    }
    for index in 0..DIMENSION {
        write!(archive, ",decision_{index}")?;
    }
    archive.push('\n');
    for elite in &result.elites {
        let quality_validation = elite.validation.qd_quality();
        write!(
            archive,
            "{},{},{},{},{},{},{},{},{}",
            elite.niche_id,
            elite.grid_x,
            elite.grid_y,
            elite.quality,
            quality_validation,
            elite.validation_feasible_fraction,
            elite.visit_count,
            rollout_fields(&elite.train),
            rollout_fields(&elite.validation)
        )?;
        for value in elite.gait.values {
            write!(archive, ",{value}")?;
        }
        archive.push('\n');
    }
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{},{},{},{},{}",
            row.evaluations,
            row.elapsed_seconds,
            row.coverage,
            row.qd_score,
            row.best_quality,
            row.invalid_fraction
        )?;
    }
    write(&directory.join("qd_archive.csv"), &archive)?;
    write(&directory.join("convergence.csv"), &convergence)?;
    // Three contact strips spanning the discovered duty-factor range.
    if !result.elites.is_empty() {
        let mut ordered = result.elites.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.train.duty_factor.total_cmp(&right.train.duty_factor));
        for (label, index) in [
            ("low-duty", 0),
            ("mid-duty", ordered.len() / 2),
            ("high-duty", ordered.len() - 1),
        ] {
            let mut config = RolloutConfig {
                record: true,
                ..Default::default()
            };
            config.terrain_seed = 17;
            let replay = rollout(&ordered[index].gait, &config);
            write_replay(&directory.join(format!("replay-{label}.csv")), &replay)?;
        }
    }
    write(
        &directory.join("run.json"),
        &(serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "rapier-quadruped-gait",
            "formulation": "qd",
            "command": command,
            "seed": args.seed,
            "simulation_seed": 17,
            "validation_seeds": [1001, 1002, 1003, 1004, 1005],
            "workers": args.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "quality_train", "label": "Distance/work quality", "unit": "1"}],
            "descriptors": [
                {"column": "descriptor_duty_factor", "label": "Mean foot duty factor", "unit": "fraction", "bounds": [0.0, 1.0]},
                {"column": "descriptor_body_height_std_mm", "label": "Torso-height standard deviation", "unit": "mm", "bounds": [0.0, 200.0]}
            ],
            "constraints": [
                {"column": "constraint_fall_m", "label": "Fall-height deficit", "unit": "m", "feasible": "<= 0"},
                {"column": "constraint_drift_m", "label": "Lateral-drift excess", "unit": "m", "feasible": "<= 0"}
            ],
            "qd": {
                "grid_shape": [(result.capacity as f64).sqrt() as usize, (result.capacity as f64).sqrt() as usize],
                "quality_label": "Distance/work quality (minimized)"
            },
            "occupied": result.occupied,
            "coverage": result.occupied as f64 / result.capacity as f64,
            "qd_score": result.qd_score,
            "invalid_evaluations": result.invalid_evaluations,
            "rejected_out_of_bounds": result.rejected_out_of_bounds,
            "descriptor_bound_policy": "reject",
            "rollout": {"duration_s": 4.0, "settle_s": 1.0, "time_step_s": 1.0 / 240.0},
            "artifacts": {
                "qd_archive": "qd_archive.csv",
                "convergence": "convergence.csv",
                "replay_low_duty": "replay-low-duty.csv",
                "replay_mid_duty": "replay-mid-duty.csv",
                "replay_high_duty": "replay-high-duty.csv"
            }
        }))? + "\n"),
    )
}

fn print_rollout(label: &str, result: &Rollout) {
    println!(
        "{label} feasible={} distance={:.6}m work={:.6}J duty={:.4} height_std={:.3}mm drift={:.6}m contacts={} score={:.6}",
        result.feasible,
        result.forward_distance_m,
        result.mechanical_work_j,
        result.duty_factor,
        result.body_height_std_mm,
        result.lateral_drift_m,
        result.terrain_contact_steps,
        result.score
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let protocol = protocol(args.preset);
    let preset = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let root = args.output.join(preset);
    let command = command_line();
    let base_config = RolloutConfig::default();
    let mut initial_config = base_config.clone();
    initial_config.record = true;
    let initial_gait = args.gait.clone().unwrap_or_else(Gait::initial);
    let initial = rollout(&initial_gait, &initial_config);
    print_rollout("INITIAL", &initial);
    if args.mode == Mode::Simulate {
        if args.write_output {
            write_replay(&root.join("simulate/replay.csv"), &initial)?;
        }
        return Ok(());
    }

    if matches!(args.mode, Mode::Range | Mode::All | Mode::Qd) {
        let rows = range_study(
            protocol.range_samples,
            args.workers,
            args.seed ^ 0xE703_7ED1_A0B4_28DB,
            &base_config,
        );
        let feasible = rows.iter().filter(|row| row.rollout.feasible).count();
        let contacts = rows
            .iter()
            .filter(|row| row.rollout.terrain_contact_steps > 0)
            .count();
        println!(
            "RANGE samples={} feasible={} terrain_contact={} descriptor_bounds=duty[0,1],height_std_mm[0,200]",
            rows.len(),
            feasible,
            contacts
        );
        if args.write_output {
            write_range(&root.join("range-study.csv"), &rows)?;
        }
        if args.mode == Mode::Range {
            return Ok(());
        }
    }

    if matches!(args.mode, Mode::Scalar | Mode::All) {
        let result = optimize_scalar(
            &ScalarConfig {
                evaluations: protocol.scalar_evaluations,
                retries: protocol.scalar_retries,
                workers: args.workers as usize,
                seed: args.seed,
            },
            &base_config,
        )?;
        print_rollout("SCALAR", &result.rollout);
        println!(
            "SCALAR evaluations={} wall={:.3}s",
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        if args.write_output {
            write_scalar(&root.join("scalar"), &command, &args, &result)?;
        }
    }

    if matches!(args.mode, Mode::Qd | Mode::All) {
        let result = optimize_qd(
            &QdConfig {
                evaluations: protocol.qd_evaluations,
                capacity: protocol.qd_capacity,
                chunk_size: protocol.qd_chunk,
                workers: args.workers,
                seed: args.seed ^ 0xA076_1D64_78BD_642F,
                holdout_seeds: vec![1001, 1002, 1003, 1004, 1005],
            },
            &base_config,
        )?;
        println!(
            "QD evaluations={} occupied={}/{} coverage={:.2}% invalid={} rejected_out_of_bounds={} wall={:.3}s",
            result.actual_evaluations,
            result.occupied,
            result.capacity,
            100.0 * result.occupied as f64 / result.capacity as f64,
            result.invalid_evaluations,
            result.rejected_out_of_bounds,
            result.elapsed.as_secs_f64()
        );
        if let Some(best) = result.elites.first() {
            print_rollout("QD_BEST_TRAIN", &best.train);
            print_rollout("QD_BEST_HOLDOUT", &best.validation);
        }
        if args.write_output {
            write_qd(&root.join("qd"), &command, &args, &result)?;
        }
    }
    Ok(())
}
