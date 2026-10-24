//! Native worker: runs one fcmaes experiment job in its job directory.
#![forbid(unsafe_code)]
use fcmaes_core::{RetryBounds, RetryConfig, retry, retry_run_seed};
use fcmaes_native_worker::{objective_for, solve};
use optimization_job_api::{
    Result,
    legacy::{result::*, *},
    storage::*,
};
use std::{fs, path::Path, sync::Mutex, time::Instant};
fn execute(dir: &Path) -> Result<()> {
    let _owner = lock(&dir.join("worker.lock"))?;
    let spec: JobSpec = serde_json::from_slice(&fs::read(dir.join("resolved.json"))?)?;
    let spec = spec.resolve()?;
    if dir.join("retries.jsonl").exists() {
        return Err("worker refuses to restart an existing attempt".into());
    }
    // Finish start-up work first, so an existing journal means the first wave
    // is about to start.
    let digest = spec.digest()?;
    let exe_digest = digest_file(&std::env::current_exe()?)?;
    // Create and directory-sync before any work, including a zero-completion kill.
    fs::File::create(dir.join("retries.jsonl"))?.sync_all()?;
    fs::File::open(dir)?.sync_all()?;
    let start = Instant::now();
    let errors = Mutex::new(None::<String>);
    let writer = Mutex::new(());
    let progress = Mutex::new((0usize, 0u64));
    let bounds = RetryBounds::new(
        spec.problem.bounds.lower.clone(),
        spec.problem.bounds.upper.clone(),
    )?;
    let f = objective_for(&spec);
    let objective = |x: &[f64]| f(x);
    let mut status = "completed";
    for offset in (0..spec.budget.retries).step_by(spec.execution.retry_wave_size) {
        if dir.join("cancel.request").exists() {
            status = "cancelled";
            break;
        }
        if start.elapsed().as_secs_f64() >= spec.budget.wall_time_seconds {
            status = "timed_out";
            break;
        }
        let config = RetryConfig {
            num_retries: spec
                .execution
                .retry_wave_size
                .min(spec.budget.retries - offset),
            workers: spec.execution.workers,
            max_evaluations: spec.budget.evaluations_per_retry,
            seed: spec.execution.seed,
            capacity: 1,
            ..Default::default()
        };
        retry(&objective, &bounds, &config, |obj, context| {
            let id = offset + context.run_id;
            let seed = retry_run_seed(spec.execution.seed, id);
            let t = Instant::now();
            let result = solve(&spec, &bounds, seed, obj);
            let valid = result.y.is_finite() && result.y < 1e99;
            let record = RetryRecord {
                retry_id: id,
                run_seed: seed,
                spec_digest: digest.clone(),
                initialization: "uniform;runid=0".into(),
                x: result.x.clone(),
                y: valid.then_some(result.y),
                actual_evaluations: result.evaluations,
                elapsed_seconds: t.elapsed().as_secs_f64(),
                valid,
                feasible: valid,
            };
            let _guard = writer.lock().unwrap();
            let mut error = errors.lock().unwrap();
            if error.is_none() {
                match append_checked(&dir.join("retries.jsonl"), 1, &record) {
                    Ok(()) => {
                        let mut p = progress.lock().unwrap();
                        p.0 += 1;
                        p.1 += record.actual_evaluations;
                    }
                    Err(e) => *error = Some(e.to_string()),
                }
            }
            result
        });
        if let Some(e) = errors.lock().unwrap().as_ref() {
            return Err(format!("journal write failure: {e}").into());
        }
        let p = progress.lock().unwrap();
        write_atomic_json(
            &dir.join("progress.json"),
            &serde_json::json!({"completed_retries":p.0,"known_evaluations_lower_bound":p.1,"accounting_complete":false,"elapsed_seconds":start.elapsed().as_secs_f64(),"spec_digest":digest}),
        )?;
    }
    write_atomic_json(
        &dir.join("run.json"),
        &manifest(
            dir,
            &spec,
            status,
            true,
            start.elapsed().as_secs_f64(),
            &exe_digest,
        )?,
    )?;
    Ok(())
}

/// Fixed-work diagnostic harness; timings are measurements, not scientific run manifests.
fn benchmark(path: &Path, dir: &Path, mode: &str) -> Result<()> {
    if !["none", "shared_atomic", "journal"].contains(&mode) {
        return Err("unknown benchmark mode".into());
    }
    let spec: JobSpec = serde_json::from_slice(&fs::read(path)?)?;
    let spec = spec.resolve()?;
    fs::create_dir(dir)?;
    let bounds = RetryBounds::new(
        spec.problem.bounds.lower.clone(),
        spec.problem.bounds.upper.clone(),
    )?;
    let count = std::sync::atomic::AtomicU64::new(0);
    let writer = Mutex::new(());
    let error = Mutex::new(None::<String>);
    let digest = spec.digest()?;
    let start = Instant::now();
    let mut evaluations = 0u64;
    // Choose the wrapper before entering any solver; no branch in the uninstrumented objective.
    let f = objective_for(&spec);
    let plain = |x: &[f64]| f(x);
    let atomic = |x: &[f64]| {
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        f(x)
    };
    #[allow(clippy::too_many_arguments)]
    fn campaign<O: Fn(&[f64]) -> f64 + Sync>(
        spec: &JobSpec,
        bounds: &RetryBounds,
        obj: &O,
        dir: &Path,
        mode: &str,
        digest: &str,
        writer: &Mutex<()>,
        error: &Mutex<Option<String>>,
    ) -> u64 {
        let mut evaluations = 0;
        for offset in (0..spec.budget.retries).step_by(spec.execution.retry_wave_size) {
            let config = RetryConfig {
                num_retries: spec
                    .execution
                    .retry_wave_size
                    .min(spec.budget.retries - offset),
                workers: spec.execution.workers,
                max_evaluations: spec.budget.evaluations_per_retry,
                seed: spec.execution.seed,
                capacity: 1,
                ..Default::default()
            };
            let r = retry(obj, bounds, &config, |obj, c| {
                let seed = retry_run_seed(spec.execution.seed, offset + c.run_id);
                let t = Instant::now();
                let r = solve(spec, bounds, seed, obj);
                if mode == "journal" {
                    let valid = r.y.is_finite() && r.y < 1e99;
                    let record = RetryRecord {
                        retry_id: offset + c.run_id,
                        run_seed: seed,
                        spec_digest: digest.into(),
                        initialization: "uniform;runid=0".into(),
                        x: r.x.clone(),
                        y: valid.then_some(r.y),
                        actual_evaluations: r.evaluations,
                        elapsed_seconds: t.elapsed().as_secs_f64(),
                        valid,
                        feasible: valid,
                    };
                    let _g = writer.lock().unwrap();
                    if let Err(e) = append_checked(&dir.join("retries.jsonl"), 1, &record) {
                        *error.lock().unwrap() = Some(e.to_string());
                    }
                }
                r
            });
            evaluations += r.evaluations;
        }
        evaluations
    }
    evaluations += if mode == "shared_atomic" {
        campaign(&spec, &bounds, &atomic, dir, mode, &digest, &writer, &error)
    } else {
        campaign(&spec, &bounds, &plain, dir, mode, &digest, &writer, &error)
    };
    let elapsed = start.elapsed().as_secs_f64();
    if let Some(e) = error.lock().unwrap().as_ref() {
        return Err(e.clone().into());
    }
    let report = serde_json::json!({"measurement":"fixed-work telemetry diagnostic","mode":mode,"workers":spec.execution.workers,"requested_evaluations":spec.requested()?,"actual_evaluations":evaluations,"wall_seconds":elapsed,"evaluations_per_second":evaluations as f64/elapsed,"shared_counter":count.load(std::sync::atomic::Ordering::Relaxed),"spec":spec,"note":"shared_atomic measures a counter only, not the complete examples ProgressState incumbent CAS path"});
    write_atomic_json(&dir.join("benchmark.json"), &report)?;
    println!("{report}");
    Ok(())
}

fn digest_file(path: &Path) -> Result<String> {
    Ok(digest(&fs::read(path)?))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("describe") => {
            let mut d = discovery();
            d["build"] = serde_json::json!({"source_base_commit":env!("FCMAES_SOURCE_COMMIT"),"rustc":env!("FCMAES_RUSTC"),"target":env!("FCMAES_TARGET"),"source_note":"base commit plus experiment-runner patch; binary digest is authoritative"});
            println!("{d}");
            Ok(())
        }
        Some("execute") if args.len() == 3 => execute(Path::new(&args[2])),
        Some("benchmark") if args.len() == 5 => {
            benchmark(Path::new(&args[2]), Path::new(&args[3]), &args[4])
        }
        _ => Err("usage: fcmaes-native-worker describe | execute JOB_DIRECTORY | benchmark SPEC OUTPUT_DIRECTORY none|shared_atomic|journal".into()),
    };
    if let Err(e) = result {
        eprintln!("{}", serde_json::json!({"error":e.to_string()}));
        std::process::exit(1);
    }
}
