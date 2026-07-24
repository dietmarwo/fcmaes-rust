use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use cfd_room_ventilation::{
    DIMENSION, Design, Evaluation, RoomConfig, RoomProblem, straight_channel_reference,
};

#[derive(Clone, Debug)]
struct Args {
    mode_results: PathBuf,
    qd_results: PathBuf,
    output: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode_results: "results/mode-seed-42/pareto.csv".into(),
            qd_results: "results/qd-seed-42/archive.csv".into(),
            output: "results/verification".into(),
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or_else(|| format!("missing value after {argument}"))
            };
            match argument.as_str() {
                "--mode-results" => parsed.mode_results = value()?.into(),
                "--qd-results" => parsed.qd_results = value()?.into(),
                "--output" => parsed.output = value()?.into(),
                "-h" | "--help" => {
                    println!(
                        "CFD publication verification\n\n\
                         --mode-results PATH  MODE pareto.csv\n\
                         --qd-results PATH    MAP-Elites archive.csv\n\
                         --output DIR         Verification output directory"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        Ok(parsed)
    }
}

fn selected_design(path: &Path) -> Result<Design, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty result CSV")?.split(',').collect();
    let selected_column = header
        .iter()
        .position(|column| *column == "selected")
        .ok_or("result CSV has no selected column")?;
    let decision_columns: Vec<usize> = header
        .iter()
        .enumerate()
        .filter_map(|(index, column)| column.starts_with("decision_").then_some(index))
        .collect();
    if decision_columns.len() != DIMENSION {
        return Err("result CSV has the wrong number of decision columns".into());
    }
    for line in lines {
        let columns: Vec<&str> = line.split(',').collect();
        if columns.get(selected_column) == Some(&"1") {
            let values: Vec<f64> = decision_columns
                .iter()
                .map(|&index| {
                    columns
                        .get(index)
                        .ok_or("short result CSV row")?
                        .parse()
                        .map_err(|_| "invalid decision value")
                })
                .collect::<Result<_, _>>()?;
            return Design::decode(&values).ok_or_else(|| "malformed selected design".into());
        }
    }
    Err("result CSV has no selected design".into())
}

fn append_metrics(
    output: &mut String,
    name: &str,
    source_set: &str,
    config: &RoomConfig,
    evaluation: &Evaluation,
) -> Result<(), std::fmt::Error> {
    writeln!(
        output,
        "{name},{source_set},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        config.nx,
        config.ny,
        config.flow_steps,
        config.scalar_steps,
        evaluation.scalar_objective(),
        evaluation.exposure,
        evaluation.maximum_receptor,
        evaluation.fan_power,
        evaluation.final_mass_fraction,
        evaluation.clearance_time,
        evaluation.flow_rate_m2_s,
        evaluation.low_velocity_fraction,
        evaluation.mass_imbalance,
        evaluation.flow_residual,
        u8::from(evaluation.feasible())
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let mode = selected_design(&args.mode_results)?;
    let qd = selected_design(&args.qd_results)?;
    let designs = [
        ("baseline", Design::default()),
        ("mode", mode),
        ("map_elites", qd),
    ];
    fs::create_dir_all(&args.output)?;

    let channel = straight_channel_reference(48, 20, 1_200)?;
    let channel_csv = format!(
        "nx,ny,flow_steps,iterations,symmetry_relative_l2,maximum_transverse_velocity,maximum_to_mean_axial_velocity,mass_imbalance,residual\n\
         {},{},1200,{},{},{},{},{},{}\n",
        channel.nx,
        channel.ny,
        channel.iterations,
        channel.symmetry_relative_l2,
        channel.maximum_transverse_velocity,
        channel.maximum_to_mean_axial_velocity,
        channel.mass_imbalance,
        channel.residual
    );
    fs::write(args.output.join("channel-reference.csv"), channel_csv)?;

    let mut resolution = String::from(
        "design,source_set,nx,ny,flow_steps,scalar_steps,quality,exposure,maximum_receptor,fan_power,final_mass_fraction,clearance_time,flow_rate_m2_s,low_velocity_fraction,mass_imbalance,flow_residual,feasible\n",
    );
    for (nx, ny, flow_steps, scalar_steps) in
        [(30, 18, 450, 450), (40, 24, 800, 600), (60, 36, 1_800, 900)]
    {
        let config = RoomConfig {
            nx,
            ny,
            flow_steps,
            scalar_steps,
            ..Default::default()
        };
        let training = RoomProblem::new(config.clone())?;
        let validation = training.validation_problem()?;
        for (name, design) in designs {
            append_metrics(
                &mut resolution,
                name,
                "training",
                &config,
                &training.evaluate_design(design),
            )?;
            append_metrics(
                &mut resolution,
                name,
                "held_out",
                &config,
                &validation.evaluate_design(design),
            )?;
        }
    }
    fs::write(args.output.join("resolution-study.csv"), resolution)?;

    let field_config = RoomConfig {
        flow_steps: 800,
        scalar_steps: 600,
        ..Default::default()
    };
    let field_problem = RoomProblem::new(field_config.clone())?;
    for (name, design) in designs {
        let detailed = field_problem
            .evaluate_detailed(design)
            .ok_or("failed to reproduce verification field")?;
        detailed
            .field
            .write_csv(args.output.join(format!("{name}-field.csv")), &field_config)?;
    }

    println!(
        "CHANNEL_REFERENCE symmetry_relative_l2={:.6e} maximum_transverse_velocity={:.6e} maximum_to_mean_axial_velocity={:.6} mass_imbalance={:.6} residual={:.6e} iterations={}",
        channel.symmetry_relative_l2,
        channel.maximum_transverse_velocity,
        channel.maximum_to_mean_axial_velocity,
        channel.mass_imbalance,
        channel.residual,
        channel.iterations
    );
    println!(
        "VERIFICATION_OUTPUT directory={} files=channel-reference.csv,resolution-study.csv,baseline-field.csv,mode-field.csv,map_elites-field.csv",
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_design_reader_uses_named_columns() {
        let directory =
            env::temp_dir().join(format!("cfd-verification-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("selected.csv");
        let mut csv = String::from("selected,quality");
        for index in 0..DIMENSION {
            write!(csv, ",decision_{index}").unwrap();
        }
        csv.push('\n');
        write!(csv, "1,0").unwrap();
        for value in Design::default().as_array() {
            write!(csv, ",{value}").unwrap();
        }
        csv.push('\n');
        fs::write(&path, csv).unwrap();
        assert_eq!(selected_design(&path).unwrap(), Design::default());
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
