//! Round-3 runner machinery: the campaign sets, the two-total protocol-T
//! campaign (cells and resume key), the writer lock, the seed rule and the
//! `serve` protocol.

use std::io::Cursor;

use candidate_harness::campaign::{
    CampaignConfig, Protocol, RAW_HEADER_T, RunLock, TSettings, run_campaign_t,
    run_campaign_t_totals, t_run_seed,
};
use candidate_harness::problems::{self, Problem};
use candidate_harness::round3::{
    self, CATEGORICAL_PROBLEMS, ENTERED, Kind, MASK_SOURCES, PROBLEMS, SOLVERS, de_mask,
    declared_settings, type_table,
};
use candidate_harness::seeds::run_seed_from;
use candidate_harness::serve::{self, decode, encode};
use candidate_harness::solvers;
use fcmaes_core::Rng;
use solver_api::Solver;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let directory =
        std::env::temp_dir().join(format!("candidate-harness-r3-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn toy_problem() -> Problem {
    Problem {
        name: "toy",
        domain: "test",
        lower: vec![-5.0; 4],
        upper: vec![5.0; 4],
        objective: Box::new(|x: &[f64]| x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum()),
    }
}

// ------------------------------------------------------------------- sets

#[test]
fn round_three_sets_follow_the_declaration() {
    // DEVIATIONS.md, round 3, entry 2: the entered problems without the two
    // network-coverage problems.
    let expected: Vec<&str> = ENTERED
        .iter()
        .copied()
        .filter(|p| !p.starts_with("coverage_"))
        .collect();
    assert_eq!(PROBLEMS.to_vec(), expected);
    for name in PROBLEMS {
        let problem = problems::by_name(name).unwrap();
        assert!(problem.dim() <= 200, "{name}");
        let table = type_table(name).unwrap();
        let categorical = table.count(Kind::Categorical) > 0;
        assert_eq!(categorical, CATEGORICAL_PROBLEMS.contains(&name), "{name}");
    }
    // Seven incumbents as in round 2, with the harness jSO→CMA arm.
    let registry: Vec<&str> = solvers::registry().iter().map(|s| s.name()).collect();
    for name in SOLVERS {
        assert!(
            registry.contains(&name),
            "{name} needs --features candidates"
        );
    }
    assert_eq!(
        SOLVERS.to_vec(),
        ["bite", "de", "cma", "crfmnes", "da", "de_cma", "jso_cma"]
    );
    assert_eq!(
        declared_settings(),
        vec![
            TSettings {
                root_seed: 3,
                total: 2_000,
                run_cap: 2_000,
                min_run: 100
            },
            TSettings {
                root_seed: 3,
                total: 20_000,
                run_cap: 20_000,
                min_run: 100
            },
        ]
    );
}

#[test]
fn no_round_three_problem_passes_a_de_mask() {
    let listed: Vec<&str> = MASK_SOURCES.iter().map(|(p, _, _)| *p).collect();
    assert_eq!(listed, PROBLEMS.to_vec());
    for (name, _, mask) in MASK_SOURCES {
        assert_eq!(mask, "none");
        assert!(de_mask(name).is_none(), "{name}");
    }
}

#[test]
fn run_seeds_ignore_the_total_and_match_the_round_two_derivation() {
    for name in PROBLEMS {
        for rep in 0..3 {
            for restart in 0..4 {
                let small = round3::settings_for(3, 2_000, 100);
                let large = round3::settings_for(3, 20_000, 100);
                let seed = t_run_seed(&small, name, rep, restart);
                assert_eq!(seed, t_run_seed(&large, name, rep, restart));
                assert_eq!(seed, run_seed_from(3, name, 'T', rep, restart));
            }
        }
    }
    // Frozen: a change here changes every round-3 seed (and the Python copy).
    assert_eq!(
        t_run_seed(&round3::settings_for(3, 2_000, 100), "jobshop", 0, 0),
        8_837_266_590_916_294_613
    );
}

// --------------------------------------------------------------- campaign

fn config<'a>(
    problem: &'a Problem,
    registry: &'a [Box<dyn Solver>],
    reps: u32,
) -> CampaignConfig<'a> {
    CampaignConfig {
        problems: vec![(problem, 0)],
        solvers: registry
            .iter()
            .filter(|s| ["bite", "cma", "de_cma"].contains(&s.name()))
            .map(|s| s.as_ref())
            .collect(),
        protocols: vec![Protocol::T],
        repetitions: reps,
        threads: 2,
        budget_scale: 1.0,
    }
}

fn two_totals() -> Vec<TSettings> {
    vec![
        round3::settings_for(3, 300, 30),
        round3::settings_for(3, 900, 30),
    ]
}

/// (solver, rep, total, evaluations, over, out, runs, root, total, cap, min).
fn rows(path: &std::path::Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(RAW_HEADER_T));
    let mut rows: Vec<Vec<String>> = lines
        .map(|line| {
            let mut fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
            assert_eq!(fields.len(), 19);
            fields.remove(13); // wall time
            fields
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn two_totals_are_separate_cells_with_their_own_resume_key() {
    let directory = temp_dir("totals");
    let out = directory.join("raw.tsv");
    let problem = toy_problem();
    let registry = solvers::registry();
    assert_eq!(
        run_campaign_t_totals(&config(&problem, &registry, 2), &two_totals(), &out).unwrap(),
        12
    );
    let first = rows(&out);
    // Every (solver, rep) appears once per total, with run cap = total.
    for row in &first {
        let total: u64 = row[7].parse().unwrap();
        assert!(total == 300 || total == 900);
        assert_eq!(
            row[14..],
            [
                "3".to_owned(),
                total.to_string(),
                total.to_string(),
                "30".to_owned()
            ]
        );
        let evaluations: u64 = row[9].parse().unwrap();
        assert!(evaluations <= total && total - evaluations < 30, "{row:?}");
        assert_eq!((row[11].as_str(), row[12].as_str()), ("0", "0"));
        // The run-0 seed is shared by both totals.
        let rep: u32 = row[5].parse().unwrap();
        assert_eq!(row[6], run_seed_from(3, "toy", 'T', rep, 0).to_string());
    }
    let mut keys: Vec<(String, String, String)> = first
        .iter()
        .map(|r| (r[0].clone(), r[5].clone(), r[7].clone()))
        .collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 12);
    // A rerun writes nothing; a third repetition adds one row per cell.
    assert_eq!(
        run_campaign_t_totals(&config(&problem, &registry, 2), &two_totals(), &out).unwrap(),
        0
    );
    // An interrupted write is dropped and recomputed.
    let text = std::fs::read_to_string(&out).unwrap();
    let cut = text.trim_end_matches('\n').rfind('\n').unwrap() + 1;
    std::fs::write(&out, format!("{}cma\ttoy\t", &text[..cut])).unwrap();
    assert_eq!(
        run_campaign_t_totals(&config(&problem, &registry, 3), &two_totals(), &out).unwrap(),
        7
    );
    assert_eq!(rows(&out).len(), 18);
    // Rows reproduce on rerun into a fresh file (same seeds, same rows).
    let again = directory.join("again.tsv");
    run_campaign_t_totals(&config(&problem, &registry, 3), &two_totals(), &again).unwrap();
    assert_eq!(rows(&out), rows(&again));
    // Only one total of the file: the other total's rows are foreign settings.
    assert!(
        run_campaign_t_totals(&config(&problem, &registry, 3), &two_totals()[..1], &out).is_err()
    );
    // Other root seed: refused; repeated totals: refused.
    let other: Vec<TSettings> = two_totals()
        .into_iter()
        .map(|s| TSettings { root_seed: 4, ..s })
        .collect();
    assert!(run_campaign_t_totals(&config(&problem, &registry, 3), &other, &out).is_err());
    let repeated = vec![two_totals()[0], two_totals()[0]];
    assert!(
        run_campaign_t_totals(
            &config(&problem, &registry, 3),
            &repeated,
            &directory.join("r.tsv")
        )
        .is_err()
    );
    // A single-settings (round-2) campaign refuses the two-total file.
    assert!(run_campaign_t(&config(&problem, &registry, 3), &two_totals()[0], &out).is_err());
    assert!(!RunLock::path_for(&out).exists());
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_held_lock_refuses_a_second_writer_and_a_stale_lock_is_replaced() {
    let directory = temp_dir("lock");
    let out = directory.join("raw.tsv");
    let problem = toy_problem();
    let registry = solvers::registry();
    {
        let _held = RunLock::acquire(&out).unwrap();
        assert!(RunLock::acquire(&out).is_err());
        let error = run_campaign_t_totals(&config(&problem, &registry, 1), &two_totals(), &out)
            .unwrap_err();
        assert!(error.contains("locked"), "{error}");
        assert!(!out.exists(), "the file is not touched while locked");
    }
    assert!(!RunLock::path_for(&out).exists());
    // A lock of a process that does not exist (pid above the kernel limit).
    std::fs::write(RunLock::path_for(&out), "4194999\n").unwrap();
    assert_eq!(
        run_campaign_t_totals(&config(&problem, &registry, 1), &two_totals(), &out).unwrap(),
        6
    );
    assert!(!RunLock::path_for(&out).exists());
    std::fs::remove_dir_all(&directory).unwrap();
}

// ------------------------------------------------------------------ serve

fn request(input: &str) -> (Result<(), String>, String) {
    let mut output = Vec::new();
    let result = serve::serve(Cursor::new(input.as_bytes().to_vec()), &mut output);
    (result, String::from_utf8(output).unwrap())
}

fn line_of(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| encode(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn serve_values_are_bit_identical_to_in_process_evaluation() {
    for name in PROBLEMS {
        let problem = problems::by_name(name).unwrap();
        let mut rng = Rng::new(0x5E4E_0003 ^ problem.dim() as u64);
        let mut points: Vec<Vec<f64>> = (0..60)
            .map(|_| solvers::uniform_point(&problem.lower, &problem.upper, &mut rng))
            .collect();
        points.push(problem.lower.clone());
        points.push(problem.upper.clone());
        let mut input = format!("info {name}\neval {name} {}\n", points.len());
        for x in &points {
            input.push_str(&line_of(x));
            input.push('\n');
        }
        let (result, output) = request(&input);
        result.unwrap();
        let mut lines = output.lines();
        assert_eq!(
            lines.next().unwrap(),
            format!("ok {} {}", problem.dim(), problem.domain)
        );
        for bounds in [&problem.lower, &problem.upper] {
            let decoded: Vec<f64> = lines
                .next()
                .unwrap()
                .split(' ')
                .map(|t| decode(t).unwrap())
                .collect();
            assert_eq!(
                decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bounds.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let header: Vec<&str> = lines.next().unwrap().split(' ').collect();
        assert_eq!(header[..2], ["ok", &points.len().to_string()]);
        let mut check = format!("check {name} {}\n", points.len());
        for x in &points {
            let y = decode(lines.next().unwrap()).unwrap();
            assert_eq!(y.to_bits(), (problem.objective)(x).to_bits(), "{name}");
            let mut row = x.clone();
            row.push(y);
            check.push_str(&line_of(&row));
            check.push('\n');
        }
        assert!(lines.next().is_none());
        let (result, output) = request(&check);
        result.unwrap();
        assert_eq!(output, "ok 0\n");
    }
}

#[test]
fn serve_check_detects_a_wrong_value() {
    let problem = problems::by_name("jobshop").unwrap();
    let x = vec![0.5; problem.dim()];
    let mut row = x.clone();
    row.push(f64::from_bits((problem.objective)(&x).to_bits() ^ 1));
    let (result, output) = request(&format!("check jobshop 1\n{}\n", line_of(&row)));
    result.unwrap();
    assert_eq!(output, "ok 1\n");
}

#[test]
fn serve_levels_report_the_level_of_every_center() {
    for name in PROBLEMS {
        let table = type_table(name).unwrap();
        // Vector k puts every discrete coordinate at level k (mod its count).
        let mut input = format!("levels {name} 3\n");
        let mut expected = Vec::new();
        for k in 0..3usize {
            let (x, levels): (Vec<f64>, Vec<String>) = table
                .coordinates
                .iter()
                .map(|c| match c.kind {
                    Kind::Continuous => (0.5 * (c.lower + c.upper), "-".to_owned()),
                    _ => {
                        let level = (k * 7) % c.levels.len();
                        (c.levels[level].center, level.to_string())
                    }
                })
                .unzip();
            input.push_str(&line_of(&x));
            input.push('\n');
            expected.push(levels.join(" "));
        }
        let (result, output) = request(&input);
        result.unwrap();
        let mut lines = output.lines();
        assert_eq!(lines.next(), Some("ok 3"));
        for line in expected {
            assert_eq!(lines.next().unwrap(), line, "{name}");
        }
    }
}

#[test]
fn serve_echo_and_fmt_round_trip() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5e-7,
        1e12,
        1e99,
        f64::MIN_POSITIVE,
        5e-324,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.1 + 0.2,
    ];
    let mut input = format!("echo 1\n{}\nfmt {}\n", line_of(&values), values.len());
    for v in values {
        input.push_str(&encode(v));
        input.push('\n');
    }
    let (result, output) = request(&input);
    result.unwrap();
    let mut lines = output.lines();
    assert_eq!(lines.next(), Some("ok 1"));
    assert_eq!(lines.next().unwrap(), line_of(&values));
    assert_eq!(lines.next().unwrap(), format!("ok {}", values.len()));
    for v in values {
        assert_eq!(lines.next().unwrap(), format!("{v:e}"));
    }
}

#[test]
fn serve_refuses_points_outside_the_box_and_bad_requests() {
    let problem = problems::by_name("material_flow").unwrap();
    let outside = line_of(&[0.5, 10.0]);
    let (result, output) = request(&format!("eval material_flow 1\n{outside}\n"));
    assert!(result.is_err());
    assert!(output.starts_with("err "), "{output}");
    let (result, _) = request(&format!("eval material_flow 1\n{}\n", line_of(&[2.0])));
    assert!(result.is_err());
    let (result, _) = request("eval nowhere 0\n");
    assert!(result.is_err());
    let (result, _) = request("eval material_flow 2\n4000000000000000 4000000000000000\n");
    assert!(result.is_err(), "missing data line");
    let (result, output) = request("quit\neval material_flow 1\nxx\n");
    assert!(result.is_ok() && output.is_empty());
    assert_eq!(problem.dim(), 2);
}
