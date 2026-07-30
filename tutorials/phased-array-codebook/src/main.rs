use std::error::Error;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use num_complex::Complex64;
use phased_array_codebook::array::{AngleGrid, Array};
use phased_array_codebook::artifacts::{
    RunMetadata, ValidationEvidence, write_geometry, write_mo, write_pilot, write_qd, write_so,
    write_staircase, write_validation,
};
use phased_array_codebook::config::{Preset, Protocol};
use phased_array_codebook::geometry::{GeometryConfig, optimize_geometry};
use phased_array_codebook::kernel::SteeringMatrix;
use phased_array_codebook::metrics::planar_directivity;
use phased_array_codebook::mo::{MoConfig, optimize_mode};
use phased_array_codebook::pilot::{QdDecision, run_pilot};
use phased_array_codebook::qd::{QdConfig, optimize_qd};
use phased_array_codebook::so::{BeamContext, SoConfig, SoOptimizer, optimize_arm};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    So,
    Pilot,
    Qd,
    Mo,
    Geometry,
    Validation,
    All,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    preset: Preset,
    workers: i32,
    seed: u64,
    output: Option<PathBuf>,
    evaluations: Option<usize>,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            seed: 42,
            output: None,
            evaluations: None,
            write_output: true,
        }
    }
}

fn usage() {
    println!(
        "Quantized phased-array codebook synthesis with fcmaes-core\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME          so, pilot, qd, mo, geometry, validation, or all\n\
         --preset NAME        smoke or publication (smoke)\n\
         --workers N          Candidate threads; 0 uses available CPUs (0)\n\
         --seed N             Root optimizer seed (42)\n\
         --evaluations N      Override each selected optimizer budget\n\
         --output DIR         Artifact root (results/<preset>)\n\
         --no-output          Execute without writing artifacts\n\
         -h, --help           Show this help\n\
         \n\
         All decision variables are normalized and decoded into 6-bit phase\n\
         and 5-bit attenuation register codes inside the Rust objective."
    );
}

fn next(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match next(&mut arguments, "--mode")?.as_str() {
                    "so" => Mode::So,
                    "pilot" => Mode::Pilot,
                    "qd" => Mode::Qd,
                    "mo" => Mode::Mo,
                    "geometry" => Mode::Geometry,
                    "validation" => Mode::Validation,
                    "all" => Mode::All,
                    value => return Err(format!("unknown mode {value}").into()),
                };
            }
            "--preset" => {
                parsed.preset = match next(&mut arguments, "--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    value => return Err(format!("unknown preset {value}").into()),
                };
            }
            "--workers" => parsed.workers = next(&mut arguments, "--workers")?.parse()?,
            "--seed" => parsed.seed = next(&mut arguments, "--seed")?.parse()?,
            "--evaluations" => {
                parsed.evaluations = Some(next(&mut arguments, "--evaluations")?.parse()?)
            }
            "--output" => parsed.output = Some(next(&mut arguments, "--output")?.into()),
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown option {value}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    Ok(Some(parsed))
}

fn command_line() -> String {
    let forwarded = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let base = if cfg!(feature = "fft") {
        "cargo run --release --locked --features fft"
    } else {
        "cargo run --release --locked"
    };
    if forwarded.is_empty() {
        base.to_owned()
    } else {
        format!("{base} -- {forwarded}")
    }
}

fn metadata<'a>(
    directory: &'a Path,
    command: &'a str,
    args: &Args,
    points: usize,
) -> RunMetadata<'a> {
    RunMetadata {
        directory,
        command,
        seed: args.seed,
        workers: args.workers,
        points,
    }
}

fn run_so(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let evaluations = args
        .evaluations
        .map_or(protocol.so_evaluations, |value| value as u64);
    let config = SoConfig {
        evaluations_per_arm: evaluations,
        retries: protocol.so_retries.min(evaluations as usize).max(1),
        workers: args.workers as usize,
        seed: args.seed,
        points: protocol.cut_points,
        requested_deg: 20.0,
    };
    let mut results = Vec::new();
    for optimizer in SoOptimizer::ALL {
        let result = optimize_arm(optimizer, &config)?;
        println!(
            "SO {:>4}: objective={:.3} peak={:.3}° nominal={:.2} dB worst={:.2} dB eval={} wall={:.3}s",
            optimizer.name(),
            result.best.objective,
            result.best.robust.nominal.peak_theta_deg,
            result.best.robust.nominal.psll_db,
            result.best.robust.worst_psll_db,
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        results.push(result);
    }
    if args.write_output {
        write_so(
            &metadata(&output.join("so"), command, args, protocol.cut_points),
            &results,
            20.0,
        )?;
    }
    Ok(())
}

fn run_descriptor_pilot(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<QdDecision, Box<dyn Error>> {
    let samples = args.evaluations.unwrap_or(protocol.pilot_samples);
    let (rows, summary) = run_pilot(samples, protocol.cut_points);
    println!(
        "PILOT {}: feasible={} correlation={:.3} coverage={:.1}% retention={:.1}%",
        summary.decision.label(),
        rows.len(),
        summary.d1_rank_correlation,
        100.0 * summary.coverage,
        100.0 * summary.holdout_niche_retention
    );
    if args.write_output {
        write_pilot(
            &metadata(&output.join("pilot"), command, args, protocol.cut_points),
            &rows,
            &summary,
        )?;
    }
    Ok(summary.decision)
}

fn run_qd(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let result = optimize_qd(&QdConfig {
        evaluations: args.evaluations.unwrap_or(protocol.qd_evaluations),
        capacity: protocol.qd_capacity,
        chunk_size: protocol.qd_chunk_size,
        workers: args.workers,
        seed: args.seed,
        points: protocol.cut_points,
    })?;
    println!(
        "QD: occupied={}/{} coverage={:.1}% invalid={} infeasible={} eval={} wall={:.3}s",
        result.entries.len(),
        result.capacity,
        100.0 * result.entries.len() as f64 / result.capacity as f64,
        result.invalid_evaluations,
        result.infeasible_evaluations,
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_qd(
            &metadata(&output.join("qd"), command, args, protocol.cut_points),
            &result,
        )?;
    }
    Ok(())
}

fn run_mo(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let result = optimize_mode(&MoConfig {
        evaluations: args.evaluations.unwrap_or(protocol.mo_evaluations),
        population: protocol.mo_population,
        workers: args.workers,
        seed: args.seed,
        points: protocol.cut_points,
    })?;
    println!(
        "MODE: pareto={} eval={} wall={:.3}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_mo(
            &metadata(&output.join("mo"), command, args, protocol.cut_points),
            &result,
        )?;
    }
    Ok(())
}

fn run_geometry(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let evaluations = args.evaluations.unwrap_or(protocol.so_evaluations as usize) as u64;
    let result = optimize_geometry(&GeometryConfig {
        evaluations,
        retries: protocol.so_retries,
        workers: args.workers as usize,
        seed: args.seed,
        points: protocol.cut_points,
    })?;
    println!(
        "GEOMETRY: psll={:.2} dB spacing={:.3}λ eval={} wall={:.3}s",
        result.best.metrics.psll_db,
        result.best.array.minimum_spacing_lambda(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_geometry(
            &metadata(&output.join("geometry"), command, args, protocol.cut_points),
            &result,
        )?;
    }
    Ok(())
}

fn average_time(mut operation: impl FnMut(), repeats: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..repeats {
        operation();
    }
    started.elapsed().as_secs_f64() / repeats as f64
}

fn run_validation(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let linear = BeamContext::stage_a(protocol.cut_points);
    let excitation = vec![Complex64::new(1.0, 0.0); 16];
    let mut linear_out = vec![Complex64::new(0.0, 0.0); linear.grid.len()];
    let direct_linear_seconds = average_time(
        || {
            linear
                .steering
                .field_direct(&excitation, black_box(&mut linear_out))
                .unwrap();
        },
        20,
    );
    let (coarse_shape, fine_shape) = match args.preset {
        Preset::Smoke => ((45, 90), (90, 180)),
        Preset::Publication => ((90, 180), (180, 360)),
    };
    let planar = Array::uniform_rectangular(8, 8, 0.5, 0.5, 1.0);
    let planar_excitation = vec![Complex64::new(1.0, 0.0); 64];
    let coarse_grid = AngleGrid::upper_hemisphere(coarse_shape.0, coarse_shape.1);
    let fine_grid = AngleGrid::upper_hemisphere(fine_shape.0, fine_shape.1);
    let coarse_matrix = SteeringMatrix::build(&planar, &coarse_grid);
    let fine_matrix = SteeringMatrix::build(&planar, &fine_grid);
    let mut coarse_field = vec![Complex64::new(0.0, 0.0); coarse_grid.len()];
    let mut fine_field = vec![Complex64::new(0.0, 0.0); fine_grid.len()];
    coarse_matrix.field_direct(&planar_excitation, &mut coarse_field)?;
    fine_matrix.field_direct(&planar_excitation, &mut fine_field)?;
    let coarse_directivity = planar_directivity(&coarse_field, &coarse_grid)
        .directivity_dbi
        .ok_or("coarse directivity failed")?;
    let fine_directivity = planar_directivity(&fine_field, &fine_grid)
        .directivity_dbi
        .ok_or("fine directivity failed")?;
    let direct_planar_seconds = average_time(
        || {
            fine_matrix
                .field_direct(&planar_excitation, black_box(&mut fine_field))
                .unwrap();
        },
        5,
    );
    #[cfg(feature = "fft")]
    let (fft_linear_us, fft_planar_us) = {
        let fft_array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let fft = phased_array_codebook::kernel::FftKernel::linear(&fft_array, 256)?;
        let linear_time = average_time(
            || {
                black_box(fft.field(&excitation).unwrap());
            },
            100,
        );
        let fft_planar_array =
            Array::uniform_rectangular(8, 8, 0.5, 0.5, 1.0).with_element_exponent(0.0);
        let fft2 = phased_array_codebook::kernel::FftKernel::planar(&fft_planar_array, 32, 32)?;
        let planar_time = average_time(
            || {
                black_box(fft2.field(&planar_excitation).unwrap());
            },
            20,
        );
        (Some(linear_time * 1.0e6), Some(planar_time * 1.0e6))
    };
    #[cfg(not(feature = "fft"))]
    let (fft_linear_us, fft_planar_us) = (None, None);
    println!(
        "VALIDATION: directivity coarse={coarse_directivity:.4} fine={fine_directivity:.4} dBi, matrix={:.2} MiB",
        fine_matrix.memory_bytes() as f64 / 1_048_576.0
    );
    if args.write_output {
        let directory = output.join("validation");
        write_staircase(&directory)?;
        write_validation(
            &metadata(&directory, command, args, protocol.cut_points),
            &ValidationEvidence {
                coarse_directivity_dbi: coarse_directivity,
                fine_directivity_dbi: fine_directivity,
                steering_memory_bytes: fine_matrix.memory_bytes(),
                direct_linear_us: direct_linear_seconds * 1.0e6,
                direct_planar_ms: direct_planar_seconds * 1.0e3,
                fft_linear_us,
                fft_planar_us,
            },
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let protocol = args.preset.protocol();
    let output = args.output.clone().unwrap_or_else(|| match args.preset {
        Preset::Smoke => "results/smoke".into(),
        Preset::Publication => "results/publication".into(),
    });
    let command = command_line();
    match args.mode {
        Mode::So => run_so(&args, protocol, &output, &command)?,
        Mode::Pilot => {
            run_descriptor_pilot(&args, protocol, &output, &command)?;
        }
        Mode::Qd => run_qd(&args, protocol, &output, &command)?,
        Mode::Mo => run_mo(&args, protocol, &output, &command)?,
        Mode::Geometry => run_geometry(&args, protocol, &output, &command)?,
        Mode::Validation => run_validation(&args, protocol, &output, &command)?,
        Mode::All => {
            run_validation(&args, protocol, &output, &command)?;
            run_so(&args, protocol, &output, &command)?;
            let decision = run_descriptor_pilot(&args, protocol, &output, &command)?;
            if decision == QdDecision::Rejected {
                println!("QD skipped because the pre-registered descriptor pilot rejected it");
            } else {
                run_qd(&args, protocol, &output, &command)?;
            }
            run_mo(&args, protocol, &output, &command)?;
            run_geometry(&args, protocol, &output, &command)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_protocol_is_nonzero_and_bounded() {
        let protocol = Preset::Smoke.protocol();
        assert!(protocol.so_evaluations > 0);
        assert!(protocol.cut_points < Preset::Publication.protocol().cut_points);
    }
}
