use std::error::Error;

use std::path::PathBuf;

use fcmaes_foundations::campaign::{self, Preset};
use fcmaes_foundations::lennard_jones_campaign::{self, LjPreset};
use fcmaes_foundations::lessons;
use fcmaes_foundations::suites;
use fcmaes_foundations::suites::Suite;
use fcmaes_foundations::suites::lennard_jones::{LennardJones, Parameterization};

fn usage() {
    println!(
        "fcmaes foundations\n\n\
         --lesson N|all       run the progressive ladder\n\
         --suite NAME         inspect one conventional suite problem\n\
         --campaign           run classic/ZDT/DTLZ evidence\n\
         --lj-campaign        run the Lennard-Jones scaling study\n\
         --preset NAME        smoke or publication (smoke)\n\
         --atoms N            Lennard-Jones atom count for --suite (13)\n\
         --parameterization P free or fixed-frame (free)\n\
         --reference-file P   audit a separately obtained LJ coordinate file\n\
         --reference-directory P audit Cambridge files named 13, 38, 55, 75, 98 during --lj-campaign\n\
         --seed N             root seed (42)\n\
         --workers N          recorded worker count (2)\n\
         --output DIR         campaign artifact root\n\
         --help               show this help"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut lesson = None;
    let mut suite = None;
    let mut run_campaign = false;
    let mut run_lj_campaign = false;
    let mut preset = Preset::Smoke;
    let mut lj_preset = LjPreset::Smoke;
    let mut seed = 42;
    let mut workers = 2;
    let mut output = None;
    let mut atoms = 13;
    let mut parameterization = Parameterization::Free;
    let mut reference_file = None;
    let mut reference_directory = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--lesson" => lesson = arguments.next(),
            "--suite" => suite = arguments.next(),
            "--campaign" => run_campaign = true,
            "--lj-campaign" => run_lj_campaign = true,
            "--preset" => {
                let value = arguments.next().ok_or("missing --preset value")?;
                preset = Preset::parse(&value).ok_or("preset must be smoke or publication")?;
                lj_preset = LjPreset::parse(&value).ok_or("preset must be smoke or publication")?;
            }
            "--atoms" => atoms = arguments.next().ok_or("missing --atoms value")?.parse()?,
            "--parameterization" => {
                parameterization = match arguments
                    .next()
                    .ok_or("missing --parameterization value")?
                    .as_str()
                {
                    "free" => Parameterization::Free,
                    "fixed-frame" => Parameterization::FixedFrame,
                    _ => return Err("parameterization must be free or fixed-frame".into()),
                }
            }
            "--reference-file" => {
                reference_file = Some(PathBuf::from(
                    arguments.next().ok_or("missing --reference-file value")?,
                ));
            }
            "--reference-directory" => {
                reference_directory = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("missing --reference-directory value")?,
                ));
            }
            "--seed" => seed = arguments.next().ok_or("missing --seed value")?.parse()?,
            "--workers" => {
                workers = arguments.next().ok_or("missing --workers value")?.parse()?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("missing --output value")?,
                ))
            }
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            value => return Err(format!("unknown option {value}").into()),
        }
    }
    let selections = usize::from(run_campaign)
        + usize::from(run_lj_campaign)
        + usize::from(lesson.is_some())
        + usize::from(suite.is_some());
    if selections > 1 {
        return Err("choose only one of --lesson, --suite, --campaign, or --lj-campaign".into());
    }
    if reference_file.is_some() && reference_directory.is_some() {
        return Err("choose only one of --reference-file or --reference-directory".into());
    }
    if reference_directory.is_some() && !run_lj_campaign {
        return Err("--reference-directory is only valid with --lj-campaign".into());
    }
    if run_campaign {
        let root = output.unwrap_or_else(|| PathBuf::from("results/smoke"));
        campaign::run(preset, seed, workers, &root)?;
        println!("foundations campaign wrote {}", root.display());
    } else if run_lj_campaign {
        let root = output.unwrap_or_else(|| PathBuf::from("results/smoke"));
        lennard_jones_campaign::run(
            lj_preset,
            seed,
            workers,
            &root,
            reference_directory.as_deref(),
        )?;
        println!("Lennard-Jones campaign wrote {}", root.display());
    } else if let Some(name) = suite {
        let problem: Box<dyn Suite> = if matches!(name.as_str(), "lennard-jones" | "lj") {
            Box::new(
                LennardJones::new(atoms, parameterization)
                    .map_err(|_| format!("invalid Lennard-Jones atom count {atoms}"))?,
            )
        } else {
            suites::by_name(&name).map_err(|_| format!("unknown suite problem {name}"))?
        };
        let (lower, upper) = problem.bounds();
        println!(
            "problem={} dimension={} objectives={} bounds={} reference_front={}",
            problem.name(),
            problem.dimension(),
            problem.objectives(),
            lower.len().min(upper.len()),
            problem.reference_front(101).map_or(0, |front| front.len())
        );
        if let Some(optimum) = problem.known_optimum() {
            println!(
                "known_decision={} known_objectives={:?} replay={:?}",
                optimum.decision.len(),
                optimum.objectives,
                problem.evaluate(&optimum.decision)?
            );
        } else if let Some(value) = problem.known_optimum_value() {
            println!("source_cited_putative_optimum={value:.9}");
        }
        if let Some(path) = reference_file {
            if !matches!(name.as_str(), "lennard-jones" | "lj") {
                return Err("--reference-file is only valid with --suite lennard-jones".into());
            }
            let lj = LennardJones::new(atoms, parameterization)?;
            let audit = lj.audit_reference(&path)?;
            println!(
                "reference_audit measured={:.9} target={:.9} absolute_error={:.3e} matches={}",
                audit.measured_energy, audit.target_energy, audit.absolute_error, audit.matches
            );
            println!("reference_audit_sha256={}", audit.coordinate_sha256);
        }
    } else {
        let selection = lesson.as_deref().unwrap_or("all");
        print!("{}", lessons::run(selection, workers)?);
    }
    Ok(())
}
