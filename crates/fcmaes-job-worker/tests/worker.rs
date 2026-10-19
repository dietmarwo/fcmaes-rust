//! Black-box checks of the fcmaes adapters: validation, resolved defaults,
//! budgets, agreement with direct `fcmaes-core` runs from the same seeds, and
//! cooperative stops.
use fcmaes_core::RetryRunResult;
use fcmaes_core::{
    Archive, Fitness, MapElitesParams, Mode, ModeParams, ReferencePoint, RetryBounds, Rng,
    hypervolume, map_elites_batch, pareto_indices, retry_run_seed,
};
use fcmaes_job_worker::{
    problems,
    retry::{cmaes_retry, judge},
};
use fcmaes_native_worker::{objective_by_id, objective_for, solve};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command, thread, time::Duration};

const WORKER: &str = env!("CARGO_BIN_EXE_fcmaes-job-worker");

fn job(optimizer: &str, problem: &str, dim: usize) -> Value {
    let (population, mode) = if optimizer == "de" {
        (json!(12), "soft")
    } else {
        (Value::Null, "hard")
    };
    let bounds = if problem == "gtop.cassini1" {
        json!({"lower": [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
               "upper": [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0], "broadcast": false})
    } else {
        json!({"lower": [-5.0], "upper": [5.0], "broadcast": true})
    };
    json!({
        "schema_version": "optimization.job/0.1",
        "request_key": "worker-test",
        "worker": "fcmaes",
        "payload_schema": "fcmaes.retry/1",
        "payload": {
            "problem": {"id": problem, "version": "1", "config": {"dimension": dim}, "bounds": bounds},
            "optimizer": {"kind": optimizer, "parameters": {"population": population}},
            "seed": 11, "retries": 5, "evaluations_per_retry": 997,
            "evaluation_limit_mode": mode
        },
        "resources": {"cpu_slots": 3, "outer_workers": 3, "inner_threads": 1},
        "limits": {"wall_time_seconds": 120.0, "grace_seconds": 1.0},
        "inputs": []
    })
}

fn worker(args: &[&str]) -> std::process::Output {
    Command::new(WORKER).args(args).output().unwrap()
}

/// `validate` output for `spec`, or the worker's error message.
fn validate(dir: &Path, spec: &Value) -> Result<Value, String> {
    let path = dir.join("spec.json");
    fs::write(&path, spec.to_string()).unwrap();
    let out = worker(&["validate", path.to_str().unwrap()]);
    if out.status.success() {
        Ok(serde_json::from_slice::<Value>(&out.stdout).unwrap()["resolved"].clone())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// Validates and executes `spec` in a fresh job directory; returns run.json.
fn execute(spec: &Value) -> Value {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), spec).unwrap();
    fs::write(dir.path().join("resolved.json"), resolved.to_string()).unwrap();
    let out = worker(&["execute", dir.path().to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap()
}

#[test]
fn describe_reports_profile_and_workspace_engine_version() {
    let out = worker(&["describe"]);
    assert!(out.status.success());
    let d: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(d["protocol"], "optimization.worker/0.2");
    assert_eq!(
        d["profiles"],
        json!([
            "fcmaes.retry/1",
            "fcmaes.mode/1",
            "fcmaes.map-elites/1",
            "fcmaes.evaluate/1"
        ])
    );
    assert_eq!(d["evaluation"]["profile"], "fcmaes.evaluate/1");
    assert_eq!(d["resources"]["inner_threads_max"], 1);
    let problems = d["problems"].as_array().unwrap();
    for (profile, id) in [
        ("fcmaes.retry/1", "gtop.cassini1"),
        ("fcmaes.mode/1", "benchmark.binh-korn"),
        ("fcmaes.map-elites/1", "benchmark.rastrigin-projection"),
    ] {
        assert!(
            problems
                .iter()
                .any(|p| p["id"] == id && p["profile"] == profile),
            "{profile} {id}"
        );
    }
    for kind in ["biteopt", "de", "cmaes", "mode", "map_elites"] {
        assert!(d["optimizers"][kind]["profile"].is_string(), "{kind}");
    }
    // fcmaes-core and this worker both inherit [workspace.package] version.
    let manifest = include_str!("../../../Cargo.toml");
    let workspace = &manifest[manifest.find("[workspace.package]").unwrap()..];
    let version = workspace
        .lines()
        .find_map(|l| l.strip_prefix("version = "))
        .unwrap()
        .trim_matches('"');
    assert_eq!(d["build"]["engine_version"], version);
}

#[test]
fn validation_resolves_defaults_and_rejects_unsupported_requests() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &job("biteopt", "benchmark.sphere", 4)).unwrap();
    assert_eq!(resolved["task_count"], 5);
    assert_eq!(
        resolved["payload"]["problem"]["bounds"]["lower"],
        json!(vec![-5.0; 4])
    );
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"]["population"],
        9 + 3 * 4
    );

    let mut threads = job("biteopt", "benchmark.sphere", 4);
    threads["resources"] = json!({"cpu_slots": 4, "outer_workers": 2, "inner_threads": 2});
    assert!(
        validate(dir.path(), &threads)
            .unwrap_err()
            .contains("inner_threads=1")
    );

    let mut hard_de = job("de", "benchmark.sphere", 4);
    hard_de["payload"]["evaluation_limit_mode"] = json!("hard");
    assert!(validate(dir.path(), &hard_de).is_err());

    let mut unknown = job("biteopt", "benchmark.sphere", 4);
    unknown["payload"]["stop_fitness"] = json!(0.0);
    assert!(validate(dir.path(), &unknown).is_err());

    let mut cassini = job("biteopt", "gtop.cassini1", 6);
    cassini["payload"]["problem"]["bounds"] =
        json!({"lower": [-1e6], "upper": [1e6], "broadcast": true});
    assert!(validate(dir.path(), &cassini).is_err());
}

#[test]
fn retries_match_the_shared_native_solver_and_honour_budgets() {
    for (problem, dim) in [("benchmark.rosenbrock", 6), ("gtop.cassini1", 6)] {
        for optimizer in ["biteopt", "de"] {
            let spec = job(optimizer, problem, dim);
            let run = execute(&spec);
            assert_eq!(run["status"], "completed");
            assert_eq!(run["runner"]["accounting_complete"], true);
            let results = run["results"].as_array().unwrap();
            assert_eq!(results.len(), 5);

            // Re-solve each retry in-process with the legacy definitions the
            // native worker uses, from the same retry seed.
            let dir = tempfile::tempdir().unwrap();
            let resolved = validate(dir.path(), &spec).unwrap();
            let legacy: fcmaes_job_api::JobSpec = serde_json::from_value(json!({
                "schema_version": "fcmaes.job/0.1", "request_key": "worker-test",
                "problem": resolved["payload"]["problem"],
                "optimizer": resolved["payload"]["optimizer"],
                "execution": {"strategy": "independent_retry", "seed": 11, "workers": 1,
                              "retry_wave_size": 1},
                "budget": {"retries": 5, "evaluations_per_retry": 997,
                           "evaluation_limit_mode": resolved["payload"]["evaluation_limit_mode"],
                           "wall_time_seconds": 120.0},
                "outputs": {"progress": "completed_retries"}
            }))
            .unwrap();
            let legacy = legacy.resolve().unwrap();
            let bounds = RetryBounds::new(
                legacy.problem.bounds.lower.clone(),
                legacy.problem.bounds.upper.clone(),
            )
            .unwrap();
            let objective = objective_for(&legacy);
            let mut total = 0;
            for r in results {
                let task = r["task_id"].as_u64().unwrap() as usize;
                let seed = retry_run_seed(11, task);
                let native = solve(&legacy, &bounds, seed, &objective);
                assert_eq!(r["outcome"]["details"]["run_seed"], seed);
                assert_eq!(
                    r["outcome"]["objective"],
                    json!(native.y),
                    "{problem}/{optimizer}"
                );
                assert_eq!(r["outcome"]["x"], json!(native.x));
                assert_eq!(r["outcome"]["claim"], "best_found");
                assert_eq!(
                    r["outcome"]["evidence"],
                    json!({"level": "rechecked", "checks": ["objective", "bounds"]})
                );
                assert_eq!(r["work"]["evaluations"], native.evaluations);
                let evaluations = native.evaluations;
                if optimizer == "biteopt" {
                    assert_eq!(evaluations, 997);
                } else {
                    assert!((997..=997 + 2 * 12 - 1).contains(&evaluations));
                }
                total += evaluations;
            }
            assert_eq!(run["work"]["evaluations"], total);
        }
    }
}

fn envelope(profile: &str, payload: Value, outer: usize, wall: f64) -> Value {
    json!({
        "schema_version": "optimization.job/0.1",
        "request_key": "worker-test",
        "worker": "fcmaes",
        "payload_schema": profile,
        "payload": payload,
        "resources": {"cpu_slots": outer, "outer_workers": outer, "inner_threads": 1},
        "limits": {"wall_time_seconds": wall, "grace_seconds": 1.0},
        "inputs": []
    })
}

fn problem(id: &str, dim: usize, lower: Value, upper: Value) -> Value {
    let broadcast = lower.as_array().is_some_and(|l| l.len() == 1) && dim > 1;
    json!({"id": id, "version": "1", "config": {"dimension": dim},
           "bounds": {"lower": lower, "upper": upper, "broadcast": broadcast}})
}

fn cmaes_job(parameters: Value) -> Value {
    envelope(
        "fcmaes.retry/1",
        json!({
            "problem": problem("benchmark.rosenbrock", 6, json!([-5.0]), json!([5.0])),
            "optimizer": {"kind": "cmaes", "parameters": parameters},
            "seed": 23, "retries": 5, "evaluations_per_retry": 3000,
            "evaluation_limit_mode": "soft"
        }),
        3,
        120.0,
    )
}

#[test]
fn cmaes_retries_match_the_library_run_and_keep_soft_budgets() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &cmaes_job(json!({}))).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"population": 31, "sigma": 0.3})
    );
    // The CMA-ES addition leaves BiteOpt/DE payloads, and so their digests, unchanged.
    let bite = validate(dir.path(), &job("biteopt", "benchmark.sphere", 4)).unwrap();
    assert_eq!(
        bite["payload"]["optimizer"],
        json!({"kind": "biteopt", "parameters": {"population": 21}})
    );
    for (edit, message) in [
        (json!({"sigma": 0.0}), "sigma"),
        (json!({"sigma": 2.5}), "sigma"),
        (json!({"population": 3}), "population"),
        (json!({"step": 1.0}), "unknown field"),
    ] {
        let err = validate(dir.path(), &cmaes_job(edit.clone())).unwrap_err();
        assert!(err.contains(message), "{edit}: {err}");
    }
    // An optimum on a bound: the reported point must be exactly the evaluated
    // one, inside the box (CMA-ES stores an unclamped decode).
    let mut edge = cmaes_job(json!({}));
    edge["payload"]["problem"] = problem("benchmark.sphere", 2, json!([0.1]), json!([0.5]));
    edge["payload"]["retries"] = json!(3);
    let run = execute(&edge);
    assert_eq!(run["status"], "completed");
    for r in run["results"].as_array().unwrap() {
        let x: Vec<f64> = serde_json::from_value(r["outcome"]["x"].clone()).unwrap();
        assert!(x.iter().all(|v| (0.1..=0.5).contains(v)), "{x:?}");
        assert_eq!(x, vec![0.1, 0.1]);
    }
    let mut overflow = cmaes_job(json!({}));
    overflow["payload"]["evaluations_per_retry"] = json!(u64::MAX);
    assert!(
        validate(dir.path(), &overflow)
            .unwrap_err()
            .contains("overflow")
    );
    let mut hard = cmaes_job(json!({}));
    hard["payload"]["evaluation_limit_mode"] = json!("hard");
    assert!(validate(dir.path(), &hard).unwrap_err().contains("soft"));
    let mut sigma = job("biteopt", "benchmark.sphere", 4);
    sigma["payload"]["optimizer"]["parameters"]["sigma"] = json!(0.2);
    assert!(
        validate(dir.path(), &sigma)
            .unwrap_err()
            .contains("only to cmaes")
    );

    let spec = cmaes_job(json!({"population": 12, "sigma": 0.25}));
    let run = execute(&spec);
    assert_eq!(run["status"], "completed");
    let bounds = RetryBounds::new(vec![-5.0; 6], vec![5.0; 6]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    let mut total = 0;
    for r in run["results"].as_array().unwrap() {
        let seed = retry_run_seed(23, r["task_id"].as_u64().unwrap() as usize);
        let (native, stop) = cmaes_retry(&bounds, seed, 12, 0.25, 3000, objective);
        assert_eq!(r["outcome"]["objective"], json!(native.y));
        assert_eq!(r["outcome"]["x"], json!(native.x));
        assert_eq!(r["outcome"]["details"]["stop_code"], stop);
        assert_eq!(r["work"]["evaluations"], native.evaluations);
        assert!(native.evaluations <= 3000 + 11);
        let expected = if native.evaluations >= 3000 {
            "evaluation_limit"
        } else {
            "optimizer_stop"
        };
        assert_eq!(r["outcome"]["termination"], expected);
        assert_eq!(r["outcome"]["claim"], "best_found");
        total += native.evaluations;
    }
    assert_eq!(run["work"]["evaluations"], total);
}

fn mode_job(id: &str, dim: usize, lower: Value, upper: Value, parameters: Value) -> Value {
    envelope(
        "fcmaes.mode/1",
        json!({
            "problem": problem(id, dim, lower, upper),
            "optimizer": {"kind": "mode", "parameters": parameters},
            "seed": 31, "runs": 3, "generations": 30
        }),
        3,
        120.0,
    )
}

fn binh_korn_job(parameters: Value) -> Value {
    mode_job(
        "benchmark.binh-korn",
        2,
        json!([0.0, 0.0]),
        json!([5.0, 3.0]),
        parameters,
    )
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

#[test]
fn mode_fronts_match_core_mode_and_are_feasible_and_non_dominated() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &binh_korn_job(json!({"population": 16}))).unwrap();
    let p = &resolved["payload"];
    assert_eq!(resolved["task_count"], 3);
    assert_eq!(p["reference_point"], json!([140.0, 50.0]));
    assert_eq!(p["optimizer"]["parameters"]["population"], 16);
    assert_eq!(p["optimizer"]["parameters"]["nsga_update"], true);
    assert_eq!(p["optimizer"]["parameters"]["dis_m"], 20.0);

    let run = execute(&binh_korn_job(json!({"population": 16})));
    assert_eq!(run["status"], "completed");
    assert_eq!(run["work"]["evaluations"], 3 * 30 * 16);
    for r in run["results"].as_array().unwrap() {
        let o = &r["outcome"];
        assert_eq!(o["claim"], "non_dominated_set");
        assert_eq!(o["feasible"], true);
        assert!(o["x"].is_null() && o["objective"].is_null());
        assert_eq!(o["termination"], "evaluation_limit");
        assert_eq!(r["work"]["validation_evaluations"], 16);
        assert_eq!(
            o["evidence"],
            json!({"level": "rechecked",
                   "checks": ["objective", "bounds", "constraints", "non_dominance"]})
        );
        assert_eq!(o["details"]["front_source"], "final_population");
        assert_eq!(o["details"]["hypervolume"]["normalization"], "none");

        // The same run, driven directly through fcmaes-core.
        let seed = retry_run_seed(31, r["task_id"].as_u64().unwrap() as usize);
        let fit = Fitness::bounded(2, 4, &[0.0, 0.0], &[5.0, 3.0]);
        let params = ModeParams {
            popsize: 16,
            seed,
            ..Default::default()
        };
        let mut mode = Mode::try_new(fit, 2, 2, None, &params).unwrap();
        for _ in 0..30 {
            let xs = mode.ask();
            let ys: Vec<Vec<f64>> = xs.iter().map(|x| problems::binh_korn(x)).collect();
            mode.tell(&ys);
        }
        let result = mode.result();
        let mut feasible: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        for (x, y) in result.x.iter().zip(&result.y) {
            if y[2] <= 0.0 && y[3] <= 0.0 && !feasible.iter().any(|(other, _)| other == x) {
                feasible.push((x.clone(), y.clone()));
            }
        }
        let objectives: Vec<Vec<f64>> = feasible.iter().map(|(_, y)| y[..2].to_vec()).collect();
        let front = pareto_indices(&objectives, 2).unwrap();
        let reported = o["details"]["front"].as_array().unwrap();
        assert_eq!(reported.len(), front.len());
        assert_eq!(o["details"]["front_size"], front.len());
        for (point, &i) in reported.iter().zip(&front) {
            assert_eq!(point["x"], json!(feasible[i].0));
            assert_eq!(point["objectives"], json!(objectives[i]));
            assert_eq!(point["constraints"], json!(feasible[i].1[2..]));
        }
        let values: Vec<Vec<f64>> = reported
            .iter()
            .map(|p| serde_json::from_value(p["objectives"].clone()).unwrap())
            .collect();
        for a in &values {
            assert!(!values.iter().any(|b| dominates(b, a)));
        }
        let h = hypervolume(&values, &ReferencePoint::new(vec![140.0, 50.0]).unwrap()).unwrap();
        assert_eq!(o["details"]["hypervolume"]["value"], h.estimate.value());
        assert_eq!(o["details"]["hypervolume"]["exact"], true);
    }

    let zdt = mode_job("benchmark.zdt1", 5, json!([0.0]), json!([1.0]), json!({}));
    let zdt = execute(&zdt);
    assert_eq!(zdt["work"]["evaluations"], 3 * 30 * 64);
    assert!(zdt["results"][0]["outcome"]["details"]["hypervolume"]["value"].as_f64() > Some(0.0));

    for (spec, message) in [
        (
            mode_job("benchmark.zdt7", 5, json!([0.0]), json!([1.0]), json!({})),
            "unknown multi-objective",
        ),
        (
            mode_job("benchmark.zdt1", 1, json!([0.0]), json!([1.0]), json!({})),
            "invalid dimension",
        ),
        (
            mode_job("benchmark.dtlz2", 4, json!([0.0]), json!([2.0]), json!({})),
            "canonical box",
        ),
        (binh_korn_job(json!({"population": 3})), "population"),
        (
            mode_job(
                "benchmark.dtlz2",
                4,
                json!([0.0]),
                json!([1.0]),
                json!({"population": 1001}),
            ),
            "4..1000",
        ),
        (binh_korn_job(json!({"f": 0.0})), "f = 0"),
        (binh_korn_job(json!({"pro_c": 1.5})), "pro_c"),
        (binh_korn_job(json!({"seed": 1})), "unknown field"),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
    let mut reference = binh_korn_job(json!({}));
    reference["payload"]["reference_point"] = json!([1.0, 2.0, 3.0]);
    assert!(
        validate(dir.path(), &reference)
            .unwrap_err()
            .contains("2 coordinates")
    );
    let mut kind = binh_korn_job(json!({}));
    kind["payload"]["optimizer"]["kind"] = json!("nsga2");
    assert!(
        validate(dir.path(), &kind)
            .unwrap_err()
            .contains("kind mode")
    );
    let mut generations = binh_korn_job(json!({}));
    generations["payload"]["generations"] = json!(0);
    assert!(validate(dir.path(), &generations).is_err());
}

fn qd_job(id: &str, dim: usize, capacity: usize, parameters: Value) -> Value {
    envelope(
        "fcmaes.map-elites/1",
        json!({
            "problem": problem(id, dim, json!([-5.12]), json!([5.12])),
            "archive": {"capacity": capacity},
            "optimizer": {"kind": "map_elites", "parameters": parameters},
            "seed": 41, "runs": 2, "generations": 40
        }),
        2,
        120.0,
    )
}

#[test]
fn map_elites_archives_match_core_map_elites() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(
        dir.path(),
        &qd_job("benchmark.sphere-projection", 4, 36, json!({})),
    )
    .unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"chunk_size": 20, "variation": "sbx", "dis_c": 20.0, "dis_m": 20.0})
    );
    for (parameters, use_sbx) in [
        (json!({"chunk_size": 10}), true),
        (json!({"chunk_size": 10, "variation": "iso_dd"}), false),
    ] {
        let spec = qd_job("benchmark.sphere-projection", 4, 36, parameters);
        let run = execute(&spec);
        assert_eq!(run["status"], "completed");
        assert_eq!(run["work"]["evaluations"], 2 * 40 * 10);
        for r in run["results"].as_array().unwrap() {
            let seed = retry_run_seed(41, r["task_id"].as_u64().unwrap() as usize);
            let (lower, upper) = (vec![-5.12; 4], vec![5.12; 4]);
            let mut rng = Rng::new(seed);
            let mut archive =
                Archive::try_new(4, &[-10.24; 2], &[10.24; 2], 36, 0, &mut rng).unwrap();
            archive.seed_uniform(&lower, &upper, &mut rng);
            let mut evaluate = |xs: &[Vec<f64>]| {
                xs.iter()
                    .map(|x| problems::sphere_projection(x))
                    .collect::<Vec<_>>()
            };
            let params = MapElitesParams {
                generations: 40,
                chunk_size: 10,
                use_sbx,
                ..Default::default()
            };
            map_elites_batch(
                &mut archive,
                &mut evaluate,
                &lower,
                &upper,
                &params,
                &mut rng,
            )
            .unwrap();
            let o = &r["outcome"];
            let elites = o["details"]["elites"].as_array().unwrap();
            assert_eq!(elites.len(), archive.occupied());
            assert_eq!(o["details"]["occupied"], archive.occupied());
            assert_eq!(o["details"]["qd_score"], archive.qd_score());
            for e in elites {
                let niche = e["niche"].as_u64().unwrap() as usize;
                assert_eq!(e["x"], json!(archive.xs()[niche]));
                assert_eq!(e["fitness"], archive.ys()[niche]);
                assert_eq!(e["descriptor"], json!(archive.descriptors()[niche]));
                assert_eq!(e["visits"], archive.counts()[niche]);
            }
            assert_eq!(o["claim"], "elite_archive");
            assert_eq!(
                o["evidence"],
                json!({"level": "rechecked", "checks": ["objective", "bounds", "niche"]})
            );
            assert_eq!(o["objective"], archive.best_y());
            assert_eq!(r["work"]["validation_evaluations"], archive.occupied());
        }
    }

    for (spec, message) in [
        (
            qd_job(
                "benchmark.sphere-projection",
                4,
                36,
                json!({"chunk_size": 9}),
            ),
            "even",
        ),
        (
            qd_job("benchmark.sphere-projection", 3, 36, json!({})),
            "even dimension",
        ),
        (
            qd_job("benchmark.sphere-projection", 4, 0, json!({})),
            "capacity",
        ),
        (
            qd_job("benchmark.sphere-projection", 4, 10_001, json!({})),
            "capacity",
        ),
        (
            qd_job(
                "benchmark.sphere-projection",
                4,
                36,
                json!({"variation": "iso_dd", "dis_c": 5.0}),
            ),
            "only to variation sbx",
        ),
        (
            qd_job(
                "benchmark.sphere-projection",
                4,
                36,
                json!({"iso_sigma": 0.1}),
            ),
            "only to variation iso_dd",
        ),
        (
            qd_job(
                "benchmark.sphere-projection",
                4,
                36,
                json!({"variation": "cma"}),
            ),
            "sbx or iso_dd",
        ),
        (
            qd_job("benchmark.ackley-projection", 4, 36, json!({})),
            "unknown quality-diversity",
        ),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
    let mut wide = qd_job("benchmark.rastrigin-projection", 4, 36, json!({}));
    wide["payload"]["problem"]["bounds"] =
        json!({"lower": [-6.0], "upper": [6.0], "broadcast": true});
    assert!(
        validate(dir.path(), &wide)
            .unwrap_err()
            .contains("[-5.12, 5.12]")
    );
}

/// Validates `spec` into a fresh directory and starts `execute` there.
fn start(spec: &Value) -> (tempfile::TempDir, std::process::Child) {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), spec).unwrap();
    fs::write(dir.path().join("resolved.json"), resolved.to_string()).unwrap();
    let child = Command::new(WORKER)
        .args(["execute", dir.path().to_str().unwrap()])
        .spawn()
        .unwrap();
    (dir, child)
}

#[test]
fn generation_runs_stop_cooperatively_at_the_deadline_and_on_cancel() {
    let mut endless = binh_korn_job(json!({"population": 16}));
    endless["payload"]["generations"] = json!(1_000_000_000u64);
    endless["limits"]["wall_time_seconds"] = json!(0.5);
    let (dir, mut child) = start(&endless);
    assert!(child.wait().unwrap().success());
    let run: Value =
        serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["status"], "timed_out");
    assert_eq!(run["runner"]["accounting_complete"], true);
    let results = run["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    let mut evaluations = 0;
    for r in results {
        let o = &r["outcome"];
        // The SDK stops runs through the same signal at the deadline; the job
        // state, not the task, says it was the deadline.
        assert_eq!(o["termination"], "cancelled");
        assert_eq!(o["claim"], "non_dominated_set");
        let generations = o["details"]["generations"].as_u64().unwrap();
        assert!(generations > 0 && generations < 1_000_000_000);
        assert_eq!(r["work"]["evaluations"], generations * 16);
        evaluations += generations * 16;
    }
    assert_eq!(run["work"]["evaluations"], evaluations);

    let mut archive = qd_job("benchmark.rastrigin-projection", 4, 36, json!({}));
    archive["payload"]["generations"] = json!(1_000_000_000u64);
    let (dir, mut child) = start(&archive);
    thread::sleep(Duration::from_millis(400));
    fs::write(
        dir.path().join("cancel.request"),
        r#"{"reason":"cancelled"}"#,
    )
    .unwrap();
    assert!(child.wait().unwrap().success());
    let run: Value =
        serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["status"], "cancelled");
    for r in run["results"].as_array().unwrap() {
        assert_eq!(r["outcome"]["termination"], "cancelled");
        assert_eq!(r["outcome"]["claim"], "elite_archive");
        let generations = r["outcome"]["details"]["generations"].as_u64().unwrap();
        assert_eq!(r["work"]["evaluations"], generations * 20);
    }
}

#[test]
fn a_retry_answer_that_fails_its_recheck_is_refuted_not_reported() {
    let bounds = RetryBounds::new(vec![-1.0; 2], vec![1.0; 2]).unwrap();
    let sphere = objective_by_id("benchmark.sphere");
    let run = |x: Vec<f64>, y: f64| {
        let (o, validations) = judge(
            RetryRunResult {
                x,
                y,
                evaluations: 10,
            },
            &bounds,
            sphere,
        )
        .unwrap();
        // Termination is filled in by the caller.
        let o = optimization_job_api::result::Outcome {
            termination: "evaluation_limit".into(),
            ..o
        };
        o.check().unwrap();
        (serde_json::to_value(&o).unwrap(), validations)
    };
    let (ok, n) = run(vec![0.5, 0.5], 0.5);
    assert_eq!((ok["claim"].as_str(), n), (Some("best_found"), 1));
    // Outside the box: no claim survives, and the model is not evaluated
    // where it is undefined.
    let (outside, n) = run(vec![1.5, 0.0], 2.25);
    assert_eq!((outside["claim"].as_str(), n), (Some("none"), 0));
    assert_eq!(outside["feasible"], false);
    assert!(outside["objective"].is_null());
    assert_eq!(outside["evidence"]["checks"], json!(["bounds"]));
    assert_eq!(outside["evidence"]["refuted"]["claim"], "best_found");
    assert_eq!(
        outside["evidence"]["refuted"]["verdict"],
        "infeasible_point"
    );
    assert!(
        outside["evidence"]["refuted"]["message"]
            .as_str()
            .unwrap()
            .contains("x[0] = 1.5")
    );
    // A wrong value: the point is feasible with its recomputed value.
    let (mismatch, _) = run(vec![0.5, 0.5], 0.25);
    assert_eq!(mismatch["claim"], "feasible");
    assert_eq!(mismatch["objective"], 0.5);
    assert_eq!(mismatch["details"]["reported_objective"], 0.25);
    assert_eq!(
        mismatch["evidence"]["refuted"]["verdict"],
        "objective_mismatch"
    );
    // A mismatch whose recomputed value is the penalty leaves no claim.
    let unusable = |_: &[f64]| 1e99;
    let (o, _) = judge(
        RetryRunResult {
            x: vec![0.5, 0.5],
            y: 0.5,
            evaluations: 10,
        },
        &bounds,
        unusable,
    )
    .unwrap();
    assert_eq!(o.claim, "none");
    assert_eq!(o.feasible, Some(false));
    assert_eq!(
        o.evidence.unwrap().refuted.unwrap().verdict,
        "objective_mismatch"
    );
    // The fcmaes penalty is no answer at all.
    let (penalty, n) = run(vec![0.5, 0.5], 1e99);
    assert_eq!((penalty["claim"].as_str(), n), (Some("none"), 0));
    assert_eq!(
        penalty["evidence"],
        json!({"level": "claimed", "checks": []})
    );
    assert!(
        judge(
            RetryRunResult {
                x: vec![0.0],
                y: 0.0,
                evaluations: 1
            },
            &bounds,
            sphere
        )
        .is_err()
    );
}

fn evaluate_job(problem: Value, points: Value) -> Value {
    envelope(
        "fcmaes.evaluate/1",
        json!({"problem": problem, "points": points}),
        2,
        60.0,
    )
}

#[test]
fn evaluate_checks_given_points_on_every_model_family() {
    let sphere = problem("benchmark.sphere", 3, json!([-5.0]), json!([5.0]));
    let run = execute(&evaluate_job(
        sphere.clone(),
        json!([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
    ));
    assert_eq!(run["status"], "completed");
    assert_eq!(run["work"]["evaluations"], 2);
    let first = &run["results"][0]["outcome"];
    assert_eq!(first["termination"], "evaluated");
    assert_eq!(first["claim"], "feasible");
    assert_eq!(first["objective"], 9.0);
    assert_eq!(first["x"], json!([1.0, 2.0, 2.0]));
    assert_eq!(first["evidence"]["level"], "rechecked");
    assert_eq!(run["results"][1]["outcome"]["objective"], 0.0);

    // Binh-Korn: (1, 1) satisfies both constraints, (0, 3) violates the first.
    let binh = problem(
        "benchmark.binh-korn",
        2,
        json!([0.0, 0.0]),
        json!([5.0, 3.0]),
    );
    let run = execute(&evaluate_job(binh, json!([[1.0, 1.0], [0.0, 3.0]])));
    let (good, bad) = (&run["results"][0]["outcome"], &run["results"][1]["outcome"]);
    assert_eq!(
        (good["claim"].as_str(), good["feasible"].as_bool()),
        (Some("feasible"), Some(true))
    );
    assert!(good["objective"].is_null());
    assert_eq!(
        good["details"]["objectives"],
        json!(problems::binh_korn(&[1.0, 1.0])[..2])
    );
    assert_eq!(
        (bad["claim"].as_str(), bad["feasible"].as_bool()),
        (Some("none"), Some(false))
    );
    assert_eq!(
        bad["details"]["constraints"],
        json!(problems::binh_korn(&[0.0, 3.0])[2..])
    );
    assert_eq!(
        good["evidence"]["checks"],
        json!(["objective", "bounds", "constraints"])
    );

    let qd = problem(
        "benchmark.sphere-projection",
        2,
        json!([-5.12]),
        json!([5.12]),
    );
    let run = execute(&evaluate_job(qd, json!([[1.0, -2.0]])));
    let o = &run["results"][0]["outcome"];
    let (fitness, descriptor) = problems::sphere_projection(&[1.0, -2.0]);
    assert_eq!(o["objective"], fitness);
    assert_eq!(o["details"]["descriptor"], json!(descriptor));

    let dir = tempfile::tempdir().unwrap();
    for (spec, message) in [
        (evaluate_job(sphere.clone(), json!([])), "1..1000000"),
        (
            evaluate_job(sphere.clone(), json!([[1.0, 2.0]])),
            "dimension 3",
        ),
        (
            evaluate_job(sphere.clone(), json!([[1.0, 2.0, 6.0]])),
            "entry 2",
        ),
        (
            evaluate_job(
                problem("benchmark.ackley", 2, json!([-1.0]), json!([1.0])),
                json!([[0.0, 0.0]]),
            ),
            "unknown problem",
        ),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
}
