use std::env;
use std::path::PathBuf;

use crate::adapters::Arm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Campaign,
    Report,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Smoke,
    Pilot,
    Publication,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub mode: Mode,
    pub preset: Preset,
    pub output: PathBuf,
    pub problem_keys: Vec<String>,
    pub arms: Vec<Arm>,
    pub costs_ns: Vec<u64>,
    pub deadlines_ms: Vec<u64>,
    pub seeds: usize,
    pub root_seed: u64,
    pub workers: usize,
    pub population: Option<usize>,
    pub resume: bool,
}

impl Config {
    pub const USAGE: &str = "cmaes-implementation-comparison\n\
      --mode campaign|report|verify\n\
      --preset smoke|pilot|publication\n\
      --output PATH\n\
      --problems sphere10,rosenbrock10,...\n\
      --arms a,b,c\n\
      --cost-ns 0,1000,100000\n\
      --deadlines-ms 10,100,1000\n\
      --seeds N --seed N --workers N --population N\n\
      --resume\n\
\n\
Presets are explicit, inspectable defaults. Any list or scalar option overrides\n\
the corresponding preset value.";

    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_args(env::args().skip(1))
    }

    pub fn from_args<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        let preset = option(&args, "--preset")
            .map(parse_preset)
            .transpose()?
            .unwrap_or(Preset::Smoke);
        let mut config = Self::for_preset(preset);
        let mut index = 0;
        while index < args.len() {
            let argument = &args[index];
            if argument == "--resume" {
                config.resume = true;
                index += 1;
                continue;
            }
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("{argument} requires a value"))?;
            match argument.as_str() {
                "--mode" => config.mode = parse_mode(value)?,
                "--preset" => {}
                "--output" => config.output = value.into(),
                "--problems" => config.problem_keys = split_list(value),
                "--arms" => {
                    config.arms = split_list(value)
                        .into_iter()
                        .map(|value| Arm::parse(&value))
                        .collect::<Result<_, _>>()?
                }
                "--cost-ns" => config.costs_ns = parse_list(value, "--cost-ns")?,
                "--deadlines-ms" => config.deadlines_ms = parse_list(value, "--deadlines-ms")?,
                "--seeds" => config.seeds = parse_positive(value, "--seeds")?,
                "--seed" => config.root_seed = parse_value(value, "--seed")?,
                "--workers" => config.workers = parse_positive(value, "--workers")?,
                "--population" => config.population = Some(parse_positive(value, "--population")?),
                _ => return Err(format!("unknown option '{argument}'")),
            }
            index += 2;
        }
        config.validate()?;
        Ok(Some(config))
    }

    fn for_preset(preset: Preset) -> Self {
        let workers = num_cpus::get_physical().max(1);
        let (problem_keys, costs_ns, deadlines_ms, seeds) = match preset {
            Preset::Smoke => (
                vec!["sphere10", "rosenbrock10"],
                vec![0, 100_000],
                vec![10, 50],
                3,
            ),
            Preset::Pilot => (
                vec!["sphere10", "sphere100", "rosenbrock10", "rastrigin10"],
                vec![0, 1_000, 100_000],
                vec![100, 1_000],
                10,
            ),
            Preset::Publication => (
                vec![
                    "sphere10",
                    "sphere100",
                    "rosenbrock10",
                    "rosenbrock40",
                    "rastrigin10",
                    "rastrigin40",
                    "ellipsoid100",
                    "cassini1",
                ],
                vec![0, 1_000, 100_000],
                vec![100, 1_000, 10_000],
                20,
            ),
        };
        let output = PathBuf::from(match preset {
            Preset::Smoke => "results/smoke",
            Preset::Pilot => "results/pilot",
            Preset::Publication => "results/implementation-v1",
        });
        Self {
            mode: Mode::Campaign,
            preset,
            output,
            problem_keys: problem_keys.into_iter().map(str::to_owned).collect(),
            arms: vec![Arm::A, Arm::B, Arm::C],
            costs_ns,
            deadlines_ms,
            seeds,
            root_seed: 42,
            workers,
            population: None,
            resume: false,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.problem_keys.is_empty() || self.arms.is_empty() || self.deadlines_ms.is_empty() {
            return Err("problems, arms, and deadlines must be non-empty".to_owned());
        }
        if self.seeds == 0 || self.workers == 0 {
            return Err("seeds and workers must be positive".to_owned());
        }
        if self.deadlines_ms.contains(&0) {
            return Err("deadlines must be positive".to_owned());
        }
        Ok(())
    }
}

fn option<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == key)
        .map(|pair| pair[1].as_str())
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value {
        "campaign" => Ok(Mode::Campaign),
        "report" => Ok(Mode::Report),
        "verify" => Ok(Mode::Verify),
        _ => Err("--mode requires campaign, report, or verify".to_owned()),
    }
}

fn parse_preset(value: &str) -> Result<Preset, String> {
    match value {
        "smoke" => Ok(Preset::Smoke),
        "pilot" => Ok(Preset::Pilot),
        "publication" => Ok(Preset::Publication),
        _ => Err("--preset requires smoke, pilot, or publication".to_owned()),
    }
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_list<T>(value: &str, option: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr,
{
    split_list(value)
        .into_iter()
        .map(|entry| parse_value(&entry, option))
        .collect()
}

fn parse_positive<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr + PartialEq + Default,
{
    let parsed = parse_value(value, option)?;
    if parsed == T::default() {
        Err(format!("{option} requires a positive integer"))
    } else {
        Ok(parsed)
    }
}

fn parse_value<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("{option} has invalid value '{value}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_overrides_preset() {
        let config = Config::from_args([
            "--preset",
            "pilot",
            "--problems",
            "sphere10",
            "--arms",
            "a,c",
            "--workers",
            "2",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(config.preset, Preset::Pilot);
        assert_eq!(config.problem_keys, ["sphere10"]);
        assert_eq!(config.arms, [Arm::A, Arm::C]);
        assert_eq!(config.workers, 2);
    }
}
