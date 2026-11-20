//! jSO, CR-FM-NES, PGPE and CR-FM-NES → BiteOpt retries, CMA-ES restarts,
//! CVT archives and CMA-ES emitters, and the independence of every
//! population-based run from the number of evaluation threads.
mod common;
use common::{envelope, execute, problem, results, validate};
use fcmaes_core::{
    Archive, CrfmnesBiteParams, Fitness, Jso, JsoParams, MapElitesParams, RetryBounds, Rng,
    crfmnes_bite_until, map_elites_batch, retry_run_seed,
};
use fcmaes_job_worker::{
    problems,
    retry::{EsKind, EsSettings, Restarts, batch_retry, es_retry, jso_population, jso_retry},
    threads::Threads,
};
use fcmaes_native_worker::objective_by_id;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn retry(kind: &str, parameters: Value, dim: usize, evaluations: u64, inner: usize) -> Value {
    envelope(
        "fcmaes.retry/1",
        json!({"problem": problem("benchmark.rosenbrock", dim, -5.0, 5.0),
            "optimizer": {"kind": kind, "parameters": parameters}, "seed": 5, "retries": 4,
            "evaluations_per_retry": evaluations, "evaluation_limit_mode": "soft"}),
        2,
        inner,
    )
}

fn qd(archive: Value, parameters: Value, generations: u64, inner: usize) -> Value {
    envelope(
        "fcmaes.map-elites/1",
        json!({"problem": problem("benchmark.sphere-projection", 6, -5.12, 5.12),
            "archive": archive,
            "optimizer": {"kind": "map_elites", "parameters": parameters},
            "seed": 41, "runs": 2, "generations": generations}),
        2,
        inner,
    )
}

/// Compares every retry with a direct run of `settings` from its seed.
fn assert_matches_library(run: &Value, settings: &EsSettings, dim: usize, evaluations: u64) {
    assert_eq!(run["status"], "completed");
    let bounds = RetryBounds::new(vec![-5.0; dim], vec![5.0; dim]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    for r in results(run) {
        let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
        let direct = es_retry(
            settings,
            &bounds,
            seed,
            evaluations,
            &objective,
            &Threads::new(1).unwrap(),
        );
        let o = &r["outcome"];
        assert_eq!(o["objective"], json!(direct.result.y), "{settings:?}");
        assert_eq!(o["x"], json!(direct.result.x));
        assert_eq!(o["claim"], "best_found", "{o}");
        assert_eq!(
            o["evidence"],
            json!({"level": "rechecked", "checks": ["objective", "bounds"]})
        );
        assert_eq!(o["details"]["stop_code"], direct.stop);
        assert_eq!(r["work"]["evaluations"], direct.result.evaluations);
        // A soft limit: the last generation may overshoot by population - 1.
        assert!(direct.result.evaluations < evaluations + settings.max_population as u64);
    }
}

#[test]
fn crfmnes_and_pgpe_retries_report_their_best_evaluated_points() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &retry("crfmnes", json!({}), 6, 4000, 1)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"population": 32, "sigma": 0.3})
    );
    let resolved = validate(dir.path(), &retry("pgpe", json!({}), 6, 4000, 1)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"population": 32, "sigma": 0.1, "center_learning_rate": 0.15,
            "stdev_learning_rate": 0.1})
    );
    for (kind, settings) in [
        (
            "crfmnes",
            EsSettings {
                kind: EsKind::Crfmnes,
                ..EsSettings::cmaes(16, 0.3)
            },
        ),
        (
            "pgpe",
            EsSettings {
                kind: EsKind::Pgpe,
                ..EsSettings::cmaes(16, 0.1)
            },
        ),
    ] {
        let run = execute(&retry(kind, json!({"population": 16}), 6, 4000, 1));
        assert_matches_library(&run, &settings, 6, 4000);
    }

    for (spec, message) in [
        (
            retry("crfmnes", json!({"population": 15}), 6, 100, 1),
            "even",
        ),
        (
            retry("pgpe", json!({"restarts": "ipop"}), 6, 100, 1),
            "restarts applies only to cmaes",
        ),
        (
            retry("cmaes", json!({"center_learning_rate": 0.1}), 6, 100, 1),
            "applies only to pgpe",
        ),
        (
            retry("cmaes", json!({"max_population": 64}), 6, 100, 1),
            "needs restarts",
        ),
        (
            retry("cmaes", json!({"restarts": "lpop"}), 6, 100, 1),
            "none, ipop or bipop",
        ),
        (
            retry(
                "cmaes",
                json!({"population": 16, "restarts": "ipop", "max_population": 8}),
                6,
                100,
                1,
            ),
            "max_population",
        ),
        (retry("crfmnes", json!({"sigma": 3.0}), 6, 100, 1), "sigma"),
        (retry("de", json!({}), 6, 100, 2), "inner_threads=1"),
        (
            retry("pgpe", json!({"stdev_learning_rate": 0.0}), 6, 100, 1),
            "stdev_learning_rate",
        ),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
    let mut hard = retry("crfmnes", json!({}), 6, 100, 1);
    hard["payload"]["evaluation_limit_mode"] = json!("hard");
    assert!(validate(dir.path(), &hard).unwrap_err().contains("soft"));
}

/// A jSO retry job: a hard limit.
fn jso(parameters: Value, dim: usize, evaluations: u64, inner: usize) -> Value {
    let mut spec = retry("jso", parameters, dim, evaluations, inner);
    spec["payload"]["evaluation_limit_mode"] = json!("hard");
    spec
}

#[test]
fn jso_default_population_is_the_published_one_of_fcmaes_core() {
    for dim in (1..=64).chain([100, 1000, 1445, 1446, 10_000]) {
        let core = Jso::new(
            Fitness::new(dim, 1, vec![-1.0; dim], vec![1.0; dim]),
            &JsoParams::default(),
        );
        assert_eq!(jso_population(dim), core.popsize(), "dimension {dim}");
    }
    assert_eq!(jso_population(1), 4);
    assert_eq!(jso_population(6), 110);
}

#[test]
fn jso_retries_match_the_library_and_spend_exactly_their_budget() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &jso(json!({}), 6, 4000, 1)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"],
        json!({"kind": "jso", "parameters": {"population": 110}})
    );
    assert_eq!(resolved["payload"]["evaluation_limit_mode"], "hard");
    assert!(resolved["payload"].get("guess").is_none());
    let bounds = RetryBounds::new(vec![-5.0; 6], vec![5.0; 6]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    // 50 evaluations end the run inside the initial population.
    for (parameters, population, evaluations) in [
        (json!({}), 110, 4000),
        (json!({"population": 20}), 20, 3001),
        (json!({}), 110, 50),
    ] {
        let run = execute(&jso(parameters, 6, evaluations, 1));
        assert_eq!(run["status"], "completed");
        let mut total = 0;
        for r in results(&run) {
            let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
            let direct = jso_retry(
                population,
                &bounds,
                seed,
                evaluations,
                &objective,
                &Threads::new(1).unwrap(),
            );
            let o = &r["outcome"];
            assert_eq!(o["objective"], json!(direct.result.y));
            assert_eq!(o["x"], json!(direct.result.x));
            assert_eq!(o["claim"], "best_found", "{o}");
            assert_eq!(
                o["evidence"],
                json!({"level": "rechecked", "checks": ["objective", "bounds"]})
            );
            assert_eq!(o["termination"], "evaluation_limit");
            assert_eq!(o["details"]["stop_code"], direct.stop);
            assert_eq!(o["details"]["generations"], direct.generations);
            assert_eq!(o["details"]["run_seed"], seed);
            // A hard limit: exactly the budget, never more.
            assert_eq!(direct.result.evaluations, evaluations);
            assert_eq!(r["work"]["evaluations"], evaluations);
            assert_eq!(r["work"]["validation_evaluations"], 1);
            total += evaluations;
        }
        assert_eq!(run["work"]["evaluations"], total);
    }
    let mut soft = jso(json!({}), 6, 100, 1);
    soft["payload"]["evaluation_limit_mode"] = json!("soft");
    for (spec, message) in [
        (soft, "use evaluation_limit_mode hard"),
        (
            jso(json!({"sigma": 0.3}), 6, 100, 1),
            "sigma does not apply to jso",
        ),
        (
            jso(json!({"batch": 8}), 6, 100, 1),
            "batch applies only to biteopt and de",
        ),
        (
            jso(json!({"restarts": "ipop"}), 6, 100, 1),
            "restarts does not apply to jso",
        ),
        (
            jso(json!({"max_population": 64}), 6, 100, 1),
            "max_population does not apply to jso",
        ),
        (
            jso(json!({"center_learning_rate": 0.1}), 6, 100, 1),
            "center_learning_rate does not apply to jso",
        ),
        (
            jso(json!({"stdev_learning_rate": 0.1}), 6, 100, 1),
            "stdev_learning_rate does not apply to jso",
        ),
        (jso(json!({"population": 3}), 6, 100, 1), "population"),
        (
            jso(json!({"population": 1_000_001}), 6, 100, 1),
            "population",
        ),
        (
            retry("lshade", json!({}), 6, 100, 1),
            "biteopt, de, jso, cmaes, crfmnes, pgpe or crfmnes_bite",
        ),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
}

/// A CR-FM-NES → BiteOpt retry job: a hard limit, serial evaluation.
fn crfmnes_bite(parameters: Value, dim: usize, evaluations: u64) -> Value {
    let mut spec = retry("crfmnes_bite", parameters, dim, evaluations, 1);
    spec["payload"]["evaluation_limit_mode"] = json!("hard");
    spec
}

#[test]
fn crfmnes_bite_retries_match_the_library_and_spend_exactly_their_budget() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &crfmnes_bite(json!({}), 6, 4000)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"],
        json!({"kind": "crfmnes_bite",
            "parameters": {"population": 500, "crfmnes_share": 0.3, "bite_depth": 4}})
    );
    let bounds = RetryBounds::new(vec![-5.0; 6], vec![5.0; 6]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    let custom = json!({"population": 31, "sigma": 0.3, "crfmnes_share": 0.5, "bite_depth": 2});
    let resolved = validate(dir.path(), &crfmnes_bite(custom.clone(), 6, 3001)).unwrap();
    assert_eq!(resolved["payload"]["optimizer"]["parameters"], custom);
    for (parameters, settings, evaluations) in [
        (json!({}), CrfmnesBiteParams::default(), 4000),
        // The payload's sigma is in the normalized box, the core's a fraction
        // of each range; an odd population is rounded up for CR-FM-NES.
        (
            custom,
            CrfmnesBiteParams {
                popsize: 31,
                sigma: Some(0.15),
                crfmnes_share: 0.5,
                bite_depth: 2,
                ..Default::default()
            },
            3001,
        ),
        // Less than one CR-FM-NES population: BiteOpt spends everything.
        (json!({}), CrfmnesBiteParams::default(), 300),
    ] {
        let run = execute(&crfmnes_bite(parameters, 6, evaluations));
        assert_eq!(run["status"], "completed");
        let mut total = 0;
        for r in results(&run) {
            let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
            let direct = crfmnes_bite_until(
                &|x: &[f64]| objective(x),
                &bounds,
                None,
                seed,
                evaluations,
                &settings,
                &|| false,
            )
            .unwrap();
            let o = &r["outcome"];
            assert_eq!(o["objective"], json!(direct.y));
            assert_eq!(o["x"], json!(direct.x));
            assert_eq!(o["claim"], "best_found", "{o}");
            assert_eq!(
                o["evidence"],
                json!({"level": "rechecked", "checks": ["objective", "bounds"]})
            );
            assert_eq!(o["termination"], "evaluation_limit");
            assert_eq!(o["details"]["run_seed"], seed);
            // A hard limit: exactly the budget, never more.
            assert_eq!(direct.evaluations, evaluations);
            assert_eq!(r["work"]["evaluations"], evaluations);
            assert_eq!(r["work"]["validation_evaluations"], 1);
            total += evaluations;
        }
        assert_eq!(run["work"]["evaluations"], total);
    }
    let mut soft = crfmnes_bite(json!({}), 6, 100);
    soft["payload"]["evaluation_limit_mode"] = json!("soft");
    let threaded = retry("crfmnes_bite", json!({}), 6, 100, 2);
    let mut on_cmaes = retry("cmaes", json!({"crfmnes_share": 0.3}), 6, 100, 1);
    on_cmaes["payload"]["evaluation_limit_mode"] = json!("soft");
    let mut on_jso = retry("jso", json!({"bite_depth": 2}), 6, 100, 1);
    on_jso["payload"]["evaluation_limit_mode"] = json!("hard");
    let mut on_biteopt = retry("biteopt", json!({"bite_depth": 2}), 6, 100, 1);
    on_biteopt["payload"]["evaluation_limit_mode"] = json!("hard");
    for (spec, message) in [
        (soft, "use evaluation_limit_mode hard"),
        (threaded, "use inner_threads=1"),
        (
            crfmnes_bite(json!({"batch": 8}), 6, 100),
            "batch applies only to biteopt and de",
        ),
        (
            crfmnes_bite(json!({"restarts": "ipop"}), 6, 100),
            "restarts does not apply to crfmnes_bite",
        ),
        (
            crfmnes_bite(json!({"max_population": 64}), 6, 100),
            "max_population does not apply to crfmnes_bite",
        ),
        (
            crfmnes_bite(json!({"center_learning_rate": 0.1}), 6, 100),
            "center_learning_rate does not apply to crfmnes_bite",
        ),
        (
            crfmnes_bite(json!({"stdev_learning_rate": 0.1}), 6, 100),
            "stdev_learning_rate does not apply to crfmnes_bite",
        ),
        (crfmnes_bite(json!({"population": 3}), 6, 100), "population"),
        (
            crfmnes_bite(json!({"population": 1_000_001}), 6, 100),
            "population",
        ),
        (
            crfmnes_bite(json!({"crfmnes_share": 1.5}), 6, 100),
            "crfmnes_share must lie in [0, 1]",
        ),
        (
            crfmnes_bite(json!({"bite_depth": 0}), 6, 100),
            "bite_depth must be 1..36",
        ),
        (
            crfmnes_bite(json!({"bite_depth": 37}), 6, 100),
            "bite_depth must be 1..36",
        ),
        (crfmnes_bite(json!({"sigma": 0.0}), 6, 100), "sigma"),
        (crfmnes_bite(json!({"sigma": 2.5}), 6, 100), "sigma"),
        (on_cmaes, "crfmnes_share applies only to crfmnes_bite"),
        (on_jso, "bite_depth does not apply to jso"),
        (on_biteopt, "bite_depth does not apply to biteopt"),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
    // Both stages can be the whole run.
    for share in [0.0, 1.0] {
        let run = execute(&crfmnes_bite(
            json!({"population": 8, "crfmnes_share": share}),
            6,
            501,
        ));
        for r in results(&run) {
            assert_eq!(r["work"]["evaluations"], 501, "share {share}");
            assert_eq!(r["outcome"]["claim"], "best_found");
        }
    }
}

#[test]
fn biteopt_and_de_batches_match_the_library_and_keep_their_budgets() {
    let dir = tempfile::tempdir().unwrap();
    let bounds = RetryBounds::new(vec![-5.0; 6], vec![5.0; 6]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    for (kind, mode, population) in [("biteopt", "hard", 27), ("de", "soft", 31)] {
        let mut spec = retry(kind, json!({"batch": 10}), 6, 3000, 3);
        spec["payload"]["evaluation_limit_mode"] = json!(mode);
        let resolved = validate(dir.path(), &spec).unwrap();
        assert_eq!(
            resolved["payload"]["optimizer"]["parameters"],
            json!({"population": population, "batch": 10})
        );
        let legacy: optimization_job_api::legacy::JobSpec = serde_json::from_value(json!({
            "schema_version": "fcmaes.job/0.1", "request_key": "k",
            "problem": resolved["payload"]["problem"],
            "optimizer": {"kind": kind, "parameters": {"population": population}},
            "execution": {"strategy": "independent_retry", "seed": 5, "workers": 1,
                          "retry_wave_size": 1},
            "budget": {"retries": 4, "evaluations_per_retry": 3000,
                       "evaluation_limit_mode": mode, "wall_time_seconds": 60.0},
            "outputs": {"progress": "completed_retries"}
        }))
        .unwrap();
        let legacy = legacy.resolve().unwrap();
        let run = execute(&spec);
        assert_eq!(run["status"], "completed");
        for r in results(&run) {
            let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
            let direct = batch_retry(
                &legacy,
                10,
                &bounds,
                seed,
                &objective,
                &Threads::new(1).unwrap(),
            );
            let o = &r["outcome"];
            assert_eq!(o["objective"], json!(direct.y), "{kind}");
            assert_eq!(o["x"], json!(direct.x));
            assert_eq!(o["claim"], "best_found");
            assert_eq!(r["work"]["evaluations"], direct.evaluations);
            // The last batch is cut: never more than the budget.
            assert!(direct.evaluations <= 3000);
        }
    }
    for (spec, message) in [
        (
            retry("biteopt", json!({}), 6, 100, 2),
            "set optimizer.parameters.batch",
        ),
        (retry("de", json!({"batch": 0}), 6, 100, 1), "batch must be"),
        (
            retry("cmaes", json!({"batch": 8}), 6, 100, 1),
            "batch applies only to biteopt and de",
        ),
    ] {
        let err = validate(dir.path(), &spec).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
}

#[test]
fn cmaes_restarts_spend_the_budget_and_match_the_library() {
    let dir = tempfile::tempdir().unwrap();
    for (name, restarts) in [("ipop", Restarts::Ipop), ("bipop", Restarts::Bipop)] {
        let spec = retry(
            "cmaes",
            json!({"population": 8, "restarts": name}),
            4,
            20_000,
            1,
        );
        let resolved = validate(dir.path(), &spec).unwrap();
        assert_eq!(
            resolved["payload"]["optimizer"]["parameters"],
            json!({"population": 8, "sigma": 0.3, "restarts": name, "max_population": 4096})
        );
        let settings = EsSettings {
            restarts,
            max_population: 4096,
            ..EsSettings::cmaes(8, 0.3)
        };
        let run = execute(&spec);
        assert_matches_library(&run, &settings, 4, 20_000);
        let bounds = RetryBounds::new(vec![-5.0; 4], vec![5.0; 4]).unwrap();
        let objective = objective_by_id("benchmark.rosenbrock");
        for r in results(&run) {
            let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
            let direct = es_retry(
                &settings,
                &bounds,
                seed,
                20_000,
                &objective,
                &Threads::new(1).unwrap(),
            );
            let d = &r["outcome"]["details"];
            assert!(direct.restarts >= 1, "{name}: CMA-ES stops on Rosenbrock");
            assert_eq!(d["restarts"], direct.restarts);
            assert_eq!(d["largest_population"], direct.largest_population);
            assert_eq!(r["outcome"]["termination"], "evaluation_limit");
            assert!(r["work"]["evaluations"].as_u64().unwrap() >= 20_000);
            if restarts == Restarts::Ipop {
                let doubled = 8usize << direct.restarts.min(20);
                assert_eq!(direct.largest_population, doubled.min(4096));
            }
        }
    }
}

/// `repair_z` reaches `CmaesParams`: on a box that excludes Rosenbrock's
/// optimum (1, ..., 1) the samples keep reaching the upper bound, so the
/// setting changes the result; each run matches the library with it.
#[test]
fn cmaes_repair_z_reaches_the_library_and_changes_bounded_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (dim, evaluations) = (4, 3000);
    let spec = |parameters: Value| {
        envelope(
            "fcmaes.retry/1",
            json!({"problem": problem("benchmark.rosenbrock", dim, -0.5, 0.5),
                "optimizer": {"kind": "cmaes", "parameters": parameters}, "seed": 5,
                "retries": 4, "evaluations_per_retry": evaluations,
                "evaluation_limit_mode": "soft"}),
            2,
            1,
        )
    };
    let resolved = validate(dir.path(), &spec(json!({"repair_z": true}))).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"population": 31, "sigma": 0.3, "repair_z": true})
    );
    // The default, false, is absent when resolved, as in payloads without it.
    let resolved = validate(dir.path(), &spec(json!({"repair_z": false}))).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"],
        json!({"population": 31, "sigma": 0.3})
    );
    let bounds = RetryBounds::new(vec![-0.5; dim], vec![0.5; dim]).unwrap();
    let objective = objective_by_id("benchmark.rosenbrock");
    let mut objectives = Vec::new();
    for repair_z in [false, true] {
        let run = execute(&spec(json!({"repair_z": repair_z})));
        assert_eq!(run["status"], "completed");
        let settings = EsSettings {
            repair_z,
            ..EsSettings::cmaes(31, 0.3)
        };
        let mut values = Vec::new();
        for r in results(&run) {
            let seed = retry_run_seed(5, r["task_id"].as_u64().unwrap() as usize);
            let direct = es_retry(
                &settings,
                &bounds,
                seed,
                evaluations,
                &objective,
                &Threads::new(1).unwrap(),
            );
            let o = &r["outcome"];
            assert_eq!(o["objective"], json!(direct.result.y), "{repair_z}");
            assert_eq!(o["x"], json!(direct.result.x));
            assert_eq!(r["work"]["evaluations"], direct.result.evaluations);
            values.push(direct.result.y);
        }
        objectives.push(values);
    }
    assert_ne!(
        objectives[0], objectives[1],
        "repair_z did not change the runs"
    );
    // Other kinds reject it.
    for (kind, mode) in [
        ("biteopt", "hard"),
        ("de", "soft"),
        ("jso", "hard"),
        ("crfmnes", "soft"),
        ("pgpe", "soft"),
        ("crfmnes_bite", "hard"),
    ] {
        let mut other = spec(json!({"repair_z": true}));
        other["payload"]["optimizer"]["kind"] = json!(kind);
        other["payload"]["evaluation_limit_mode"] = json!(mode);
        if kind == "crfmnes_bite" {
            other["resources"] = json!({"cpu_slots": 1, "outer_workers": 1, "inner_threads": 1});
        }
        let error = validate(dir.path(), &other).unwrap_err();
        assert!(
            error.contains("optimizer.parameters.repair_z"),
            "{kind}: {error}"
        );
        other["payload"]["optimizer"]["parameters"] = json!({});
        validate(dir.path(), &other).unwrap();
    }
}

#[test]
fn population_runs_do_not_depend_on_inner_threads() {
    let mode = |inner| {
        envelope(
            "fcmaes.mode/1",
            json!({"problem": problem("benchmark.zdt1", 6, 0.0, 1.0),
                "optimizer": {"kind": "mode", "parameters": {"population": 32}},
                "seed": 3, "runs": 2, "generations": 20}),
            2,
            inner,
        )
    };
    type Job = Box<dyn Fn(usize) -> Value>;
    let jobs: Vec<(&str, Job)> = vec![
        (
            "cmaes bipop",
            Box::new(|inner| {
                retry(
                    "cmaes",
                    json!({"population": 8, "restarts": "bipop"}),
                    4,
                    6000,
                    inner,
                )
            }),
        ),
        (
            "biteopt batch",
            Box::new(|inner| {
                let mut s = retry("biteopt", json!({"batch": 12}), 6, 3000, inner);
                s["payload"]["evaluation_limit_mode"] = json!("hard");
                s
            }),
        ),
        (
            "de batch",
            Box::new(|inner| retry("de", json!({"batch": 16}), 6, 3000, inner)),
        ),
        (
            "crfmnes",
            Box::new(|inner| retry("crfmnes", json!({}), 6, 3000, inner)),
        ),
        (
            "pgpe",
            Box::new(|inner| retry("pgpe", json!({}), 6, 3000, inner)),
        ),
        ("jso", Box::new(|inner| jso(json!({}), 6, 3000, inner))),
        (
            "jso cut generation",
            Box::new(|inner| jso(json!({"population": 17}), 6, 2999, inner)),
        ),
        ("mode", Box::new(mode)),
        (
            "map-elites cvt cma_mae",
            Box::new(|inner| {
                qd(
                    json!({"capacity": 64, "kind": "cvt", "samples_per_niche": 40}),
                    json!({"chunk_size": 20, "cma_generations": 10, "emitter": "cma_mae",
                        "emitters": 2, "population": 8, "threshold_init": 50.0}),
                    10,
                    inner,
                )
            }),
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    for (name, job) in jobs {
        let (serial, parallel) = (job(1), job(3));
        let mut resolved = validate(dir.path(), &parallel).unwrap();
        resolved["resources"] = serial["resources"].clone();
        assert_eq!(resolved, validate(dir.path(), &serial).unwrap(), "{name}");
        let (a, b) = (execute(&serial), execute(&parallel));
        assert_eq!(b["status"], "completed", "{name}");
        assert_eq!(results(&a), results(&b), "{name}");
    }
}

/// Elites of a run by niche: `(fitness, x)`.
fn elites(outcome: &Value) -> BTreeMap<u64, (f64, Value)> {
    outcome["details"]["elites"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["niche"].as_u64().unwrap(),
                (e["fitness"].as_f64().unwrap(), e["x"].clone()),
            )
        })
        .collect()
}

#[test]
fn cvt_archives_share_their_niches_and_match_core_map_elites() {
    let dir = tempfile::tempdir().unwrap();
    let spec = qd(
        json!({"capacity": 40, "kind": "cvt", "samples_per_niche": 10}),
        json!({"chunk_size": 10}),
        30,
        1,
    );
    let resolved = validate(dir.path(), &spec).unwrap();
    assert_eq!(
        resolved["payload"]["archive"],
        json!({"capacity": 40, "kind": "cvt", "samples_per_niche": 10, "seed": 41})
    );
    let run = execute(&spec);
    assert_eq!(run["status"], "completed");
    let (lower, upper) = (vec![-5.12; 6], vec![5.12; 6]);
    let reach = 5.12 * 3.0;
    for r in results(&run) {
        let seed = retry_run_seed(41, r["task_id"].as_u64().unwrap() as usize);
        // The tessellation comes from the archive seed, shared by all runs.
        let mut archive =
            Archive::try_new(6, &[-reach; 2], &[reach; 2], 40, 10, &mut Rng::new(41)).unwrap();
        let mut rng = Rng::new(seed);
        archive.seed_uniform(&lower, &upper, &mut rng);
        let mut evaluate = |xs: &[Vec<f64>]| {
            xs.iter()
                .map(|x| problems::sphere_projection(x))
                .collect::<Vec<_>>()
        };
        let params = MapElitesParams {
            generations: 30,
            chunk_size: 10,
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
        assert_eq!(o["details"]["archive_kind"], "cvt");
        assert!(o["details"]["grid"].is_null());
        assert_eq!(o["details"]["occupied"], archive.occupied());
        for (niche, (fitness, x)) in elites(o) {
            assert_eq!(fitness, archive.ys()[niche as usize]);
            assert_eq!(x, json!(archive.xs()[niche as usize]));
        }
    }
    for (archive, message) in [
        (
            json!({"capacity": 40, "kind": "grid", "samples_per_niche": 5}),
            "only to cvt",
        ),
        (json!({"capacity": 40, "kind": "hex"}), "grid or cvt"),
        (
            json!({"capacity": 10_000, "kind": "cvt", "samples_per_niche": 20}),
            "1e9",
        ),
        (
            json!({"capacity": 1, "kind": "cvt", "samples_per_niche": 1_000_000_000u64}),
            "capacity * samples_per_niche <= 1e6",
        ),
    ] {
        let err = validate(dir.path(), &qd(archive, json!({}), 5, 1)).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
}

#[test]
fn cma_emitters_spend_exact_budgets_and_only_improve_the_archive() {
    let dir = tempfile::tempdir().unwrap();
    let grid = json!({"capacity": 100});
    let plain = execute(&qd(grid.clone(), json!({"chunk_size": 20}), 10, 1));
    for (emitter, extra, resolved_extra) in [
        ("cma_me", json!({}), json!({})),
        (
            "cma_mae",
            json!({"threshold_init": 300.0, "archive_learning_rate": 0.1}),
            json!({"threshold_init": 300.0, "archive_learning_rate": 0.1}),
        ),
    ] {
        let mut parameters = json!({"chunk_size": 20, "cma_generations": 15, "emitter": emitter,
            "emitters": 2, "population": 8});
        let mut expected = json!({"chunk_size": 20, "variation": "sbx", "dis_c": 20.0,
            "dis_m": 20.0, "cma_generations": 15, "emitter": emitter, "emitters": 2,
            "population": 8, "sigma": 0.1});
        for (k, v) in extra.as_object().unwrap() {
            parameters[k] = v.clone();
        }
        for (k, v) in resolved_extra.as_object().unwrap() {
            expected[k] = v.clone();
        }
        let spec = qd(grid.clone(), parameters, 10, 1);
        let resolved = validate(dir.path(), &spec).unwrap();
        assert_eq!(resolved["payload"]["optimizer"]["parameters"], expected);
        let run = execute(&spec);
        assert_eq!(run["status"], "completed");
        assert_eq!(run["work"]["evaluations"], 2 * (10 * 20 + 15 * 2 * 8));
        for (r, p) in results(&run).iter().zip(results(&plain)) {
            let o = &r["outcome"];
            assert_eq!(o["claim"], "elite_archive");
            assert_eq!(o["details"]["cma_generations"], 15);
            assert!(o["details"]["emitter_restarts"].is_u64());
            // The first phase is the plain run; emitters only add or improve.
            let (full, first) = (elites(o), elites(&p["outcome"]));
            assert!(full.len() >= first.len(), "{emitter}");
            for (niche, (fitness, _)) in first {
                assert!(full[&niche].0 <= fitness, "{emitter} niche {niche}");
            }
        }
    }
    let only_cma = qd(
        grid.clone(),
        json!({"cma_generations": 5, "emitters": 2, "population": 8}),
        0,
        1,
    );
    assert_eq!(execute(&only_cma)["work"]["evaluations"], 2 * 5 * 2 * 8);
    for (parameters, message) in [
        (
            json!({"cma_generations": 5, "emitter": "cma_mae"}),
            "requires threshold_init",
        ),
        (
            json!({"cma_generations": 5, "archive_learning_rate": 0.5}),
            "only to emitter cma_mae",
        ),
        (json!({"emitters": 3}), "need cma_generations"),
        (
            json!({"cma_generations": 5, "emitter": "cma_es"}),
            "cma_me or cma_mae",
        ),
        (json!({"cma_generations": 5, "sigma": 0.0}), "sigma"),
        (
            json!({"cma_generations": 5, "threshold_init": 1.0}),
            "only to emitter cma_mae",
        ),
        (
            json!({"cma_generations": 5, "emitter": "cma_mae", "threshold_init": 1.0,
                "archive_learning_rate": 1.5}),
            "[0, 1]",
        ),
    ] {
        let err = validate(dir.path(), &qd(grid.clone(), parameters, 5, 1)).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    }
    let err = validate(dir.path(), &qd(grid, json!({}), 0, 1)).unwrap_err();
    assert!(err.contains("cma_generations"), "{err}");
    // Each emitter is a CMA-ES with dimension x dimension matrices.
    let mut wide = qd(
        json!({"capacity": 10}),
        json!({"cma_generations": 1, "emitters": 1000, "population": 10}),
        0,
        1,
    );
    wide["payload"]["problem"] = problem("benchmark.sphere-projection", 1000, -5.12, 5.12);
    let err = validate(dir.path(), &wide).unwrap_err();
    assert!(err.contains("emitters * dimension^2"), "{err}");
}

#[test]
fn population_runs_respect_memory_limits() {
    let dir = tempfile::tempdir().unwrap();
    let err = validate(dir.path(), &retry("cmaes", json!({}), 4000, 100, 1)).unwrap_err();
    assert!(err.contains("dimension <= 3162"), "{err}");
    let resolved = validate(dir.path(), &retry("crfmnes", json!({}), 4000, 100, 1)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"]["population"],
        32
    );
    let err = validate(
        dir.path(),
        &retry("pgpe", json!({"population": 4000}), 4000, 100, 1),
    )
    .unwrap_err();
    assert!(err.contains("population * dimension <= 1e7"), "{err}");
    // The default largest restart population respects the same limit.
    let resolved = validate(
        dir.path(),
        &retry(
            "cmaes",
            json!({"population": 100, "restarts": "ipop"}),
            2000,
            100,
            1,
        ),
    )
    .unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"]["max_population"],
        5000
    );
    // jSO's default population, round(25 ln(D) sqrt(D)), fits up to 1445.
    let resolved = validate(dir.path(), &jso(json!({}), 1445, 100, 1)).unwrap();
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"]["population"],
        6914
    );
    let err = validate(dir.path(), &jso(json!({}), 1446, 100, 1)).unwrap_err();
    assert!(
        err.contains("population * dimension must be at most 1e7")
            && err.contains("smaller optimizer.parameters.population"),
        "{err}"
    );
    assert!(
        validate(
            dir.path(),
            &jso(json!({"population": 1000}), 10_000, 100, 1)
        )
        .is_ok()
    );
    let err = validate(
        dir.path(),
        &jso(json!({"population": 1001}), 10_000, 100, 1),
    )
    .unwrap_err();
    assert!(err.contains("population * dimension"), "{err}");
    // CR-FM-NES → BiteOpt holds bite_depth BiteOpt populations: the defaults
    // (500 and 4) fit up to dimension 5000.
    assert!(validate(dir.path(), &crfmnes_bite(json!({}), 5000, 100)).is_ok());
    let err = validate(dir.path(), &crfmnes_bite(json!({}), 5001, 100)).unwrap_err();
    assert!(
        err.contains("population * bite_depth * dimension must be at most 1e7"),
        "{err}"
    );
    let shallow = json!({"population": 1000, "bite_depth": 1});
    assert!(validate(dir.path(), &crfmnes_bite(shallow, 10_000, 100)).is_ok());
    let deeper = json!({"population": 1000, "bite_depth": 2});
    let err = validate(dir.path(), &crfmnes_bite(deeper, 10_000, 100)).unwrap_err();
    assert!(
        err.contains("smaller optimizer.parameters.population or bite_depth"),
        "{err}"
    );
}
