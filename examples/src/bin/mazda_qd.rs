//! Quality-diversity Mazda factory-design example using Rust CVT-MAP-Elites
//! followed optionally by the Rust Diversifier.

use std::env;
use std::error::Error;
use std::path::PathBuf;

use fcmaes_core::{Archive, DiversifierParams, MapElitesParams, Rng, diversify, map_elites};
use fcmaes_examples::mazda::{
    MAZDA_QD_LOWER, MAZDA_QD_UPPER, MazdaDecisionSpace, MazdaEvaluator, qd_value,
};

const DEFAULT_LIBRARY: &str = "mazda/mazda_cpp/Mazda_CdMOBP/src/libmazda.so";
const DEFAULT_DECISIONS: &str = "mazda/mazda_cpp/Mazda_CdMOBP/src/mazda.py";

struct Args {
    library: PathBuf,
    decisions: PathBuf,
    capacity: usize,
    samples_per_niche: usize,
    generations: usize,
    chunk_size: usize,
    diversify_evaluations: u64,
    seed: u64,
    use_sbx: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut parsed = Self {
            library: DEFAULT_LIBRARY.into(),
            decisions: DEFAULT_DECISIONS.into(),
            capacity: 1_000,
            samples_per_niche: 0,
            generations: 200,
            chunk_size: 64,
            diversify_evaluations: 0,
            seed: 42,
            use_sbx: true,
        };
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--library" => parsed.library = next_value(&mut args, "--library")?.into(),
                "--decisions" => parsed.decisions = next_value(&mut args, "--decisions")?.into(),
                "--capacity" => parsed.capacity = parse_value(&mut args, "--capacity")?,
                "--samples-per-niche" => {
                    parsed.samples_per_niche = parse_value(&mut args, "--samples-per-niche")?
                }
                "--generations" => parsed.generations = parse_value(&mut args, "--generations")?,
                "--chunk-size" => parsed.chunk_size = parse_value(&mut args, "--chunk-size")?,
                "--diversify-evaluations" => {
                    parsed.diversify_evaluations =
                        parse_value(&mut args, "--diversify-evaluations")?
                }
                "--seed" => parsed.seed = parse_value(&mut args, "--seed")?,
                "--iso-line" => parsed.use_sbx = false,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if parsed.capacity == 0 || parsed.chunk_size < 2 {
            return Err("--capacity must be positive and --chunk-size at least two".to_string());
        }
        Ok(parsed)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    next_value(args, option)?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "Mazda quality-diversity example\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin mazda-qd -- [OPTIONS]\n\
         \n  --library PATH                 Mazda libmazda.so ({DEFAULT_LIBRARY})\n\
         \n  --decisions PATH               Python decision_x sample ({DEFAULT_DECISIONS})\n\
         \n  --capacity N                   CVT niches (1000)\n\
         \n  --samples-per-niche N          CVT samples/niche; 0 selects fast grid (0)\n\
         \n  --generations N                MAP-Elites generations (200)\n\
         \n  --chunk-size N                 Candidates per generation (64)\n\
         \n  --diversify-evaluations N      Optional CMA Diversifier budget (0)\n\
         \n  --iso-line                     Use Iso+LineDD instead of SBX\n\
         \n  --seed N                       RNG seed (42)"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let space = MazdaDecisionSpace::from_python_sample(&args.decisions)?;
    let evaluator = MazdaEvaluator::load(&args.library)?;
    let lower = space.lower();
    let upper = space.upper();
    let mut rng = Rng::new(args.seed);
    let mut archive = Archive::try_new(
        space.dim(),
        &MAZDA_QD_LOWER,
        &MAZDA_QD_UPPER,
        args.capacity,
        args.samples_per_niche,
        &mut rng,
    )?;
    archive.seed_uniform(&lower, &upper, &mut rng);

    let mut qd_fitness = |indices: &[f64]| {
        evaluator
            .evaluate_indices(&space, indices)
            .and_then(|values| qd_value(&values))
            .unwrap_or((f64::INFINITY, vec![0.0; 2]))
    };

    // Evaluate the random parent pool once so parent selection starts from
    // actual occupied niches rather than unevaluated placeholders.
    let initial = archive.xs().to_vec();
    archive.update(&initial, &mut qd_fitness);
    archive.argsort();
    eprintln!(
        "initial evaluations={} occupied={} best={:.6}",
        initial.len(),
        archive.occupied(),
        archive.best_y()
    );

    let parameters = MapElitesParams {
        generations: args.generations,
        chunk_size: args.chunk_size,
        use_sbx: args.use_sbx,
        ..Default::default()
    };
    map_elites(
        &mut archive,
        &mut qd_fitness,
        &lower,
        &upper,
        &parameters,
        &mut rng,
    );

    if args.diversify_evaluations > 0 {
        let parameters = DiversifierParams {
            max_evaluations: args.diversify_evaluations,
            ..Default::default()
        };
        let (_, best) = diversify(
            &mut archive,
            &mut qd_fitness,
            &lower,
            &upper,
            &parameters,
            &mut rng,
        );
        eprintln!("Diversifier best real QD fitness={best:.6}");
    }

    println!(
        "Mazda QD: capacity={} occupied={} coverage={:.3}% best={:.8} qd_score={:.8}",
        archive.capacity(),
        archive.occupied(),
        100.0 * archive.occupied() as f64 / archive.capacity() as f64,
        archive.best_y(),
        archive.qd_score()
    );
    let mut occupied = archive.occupied_data();
    occupied.sort_by(|left, right| left.1.total_cmp(&right.1));
    for (_, fitness, descriptor) in occupied.iter().take(30) {
        println!(
            "fitness={fitness:.8} mass={:.8} common_parts={:.0}",
            descriptor[0], -descriptor[1]
        );
    }
    Ok(())
}
