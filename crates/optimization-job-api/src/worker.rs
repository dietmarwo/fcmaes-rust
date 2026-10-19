//! Small blocking worker SDK. Solver state is private to each task; results are
//! committed before a completed task is reported. No solver checkpoint is implied.
use crate::{
    JobSpec, PROTOCOL, Request, Result,
    result::{self, Outcome, Record},
    storage,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

pub struct TaskResult {
    pub outcome: Outcome,
    pub work: BTreeMap<String, u64>,
}
pub trait Prepared: Send + Sync {
    fn tasks(&self) -> usize;
    fn solve(&self, task: usize, remaining_seconds: f64) -> Result<TaskResult>;
    /// A cooperative stop. Implementations without solver tokens may finish
    /// their active task; the runner enforces the process deadline.
    fn cancel(&self);
}
pub trait Worker {
    fn describe(&self) -> Value;
    /// Validate all model data, bounds, starts, budgets and thread support.
    /// Resolve defaults into payload; no optimization is performed here.
    fn prepare(&self, spec: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>>;
}
pub fn description(
    engine: &str,
    version: &str,
    profiles: Value,
    problems: Value,
    inner_max: usize,
) -> Value {
    json!({"protocol":PROTOCOL,"engine":engine,"build":{"engine_version":version,"job_api_version":env!("CARGO_PKG_VERSION")},"profiles":profiles,"problems":problems,"resources":{"inner_threads_max":inner_max,"cpu_slots":"outer_workers * inner_threads <= cpu_slots; no hidden nested parallelism"},"recovery":"journal_only_no_resume","execution":"native registered models; no deserialized callbacks"})
}
fn prepared(worker: &dyn Worker, path: &Path) -> Result<(JobSpec, Box<dyn Prepared>)> {
    let mut s: JobSpec = serde_json::from_slice(&fs::read(path)?)?;
    s.validate()?;
    let expected = s.task_count;
    let p = worker.prepare(&mut s, path.parent().ok_or("spec parent missing")?)?;
    if expected.is_some_and(|n| n != p.tasks()) {
        return Err("task_count disagrees with payload".into());
    }
    s.task_count = Some(p.tasks());
    s.validate()?;
    Ok((s, p))
}
pub fn execute(worker: &dyn Worker, dir: &Path) -> Result<()> {
    let _owner = storage::lock(&dir.join("worker.lock"))?;
    if dir.join("results.jsonl").exists() || dir.join("run.json").exists() {
        return Err("existing attempt: journal recovery is not solver resume".into());
    }
    let start = Instant::now();
    let (s, p) = prepared(worker, &dir.join("resolved.json"))?;
    let persisted: JobSpec = serde_json::from_slice(&fs::read(dir.join("resolved.json"))?)?;
    if s != persisted {
        return Err("resolved job changed during worker validation".into());
    }
    let digest = s.digest()?;
    let exe_digest = storage::digest(&fs::read(std::env::current_exe()?)?);
    let journal = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join("results.jsonl"))?;
    journal.sync_all()?;
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let output = Mutex::new(());
    let error = Mutex::new(None::<String>);
    thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                if dir.join("cancel.request").exists()
                    || start.elapsed().as_secs_f64() >= s.limits.wall_time_seconds
                {
                    stop.store(true, Ordering::Release);
                    p.cancel();
                }
                thread::sleep(Duration::from_millis(5));
            }
        });
        let mut handles = Vec::new();
        for _ in 0..s.resources.outer_workers.min(p.tasks()) {
            handles.push(scope.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    // Check locally too: a pre-existing cancel or tiny deadline
                    // must never rely on watcher scheduling.
                    if dir.join("cancel.request").exists()
                        || start.elapsed().as_secs_f64() >= s.limits.wall_time_seconds
                    {
                        stop.store(true, Ordering::Release);
                        p.cancel();
                        break;
                    }
                    let task = next.fetch_add(1, Ordering::Relaxed);
                    if task >= p.tasks() {
                        break;
                    }
                    let began = Instant::now();
                    let attempt = (|| -> Result<()> {
                        let r = p.solve(
                            task,
                            (s.limits.wall_time_seconds - start.elapsed().as_secs_f64()).max(0.0),
                        )?;
                        let record = Record {
                            task_id: task,
                            spec_digest: digest.clone(),
                            elapsed_seconds: began.elapsed().as_secs_f64(),
                            outcome: r.outcome,
                            work: r.work,
                        };
                        let _guard = output.lock().map_err(|_| "journal lock poisoned")?;
                        storage::append_checked(&dir.join("results.jsonl"), 1, &record)?;
                        Ok(())
                    })();
                    if let Err(e) = attempt {
                        *error.lock().unwrap() = Some(e.to_string());
                        stop.store(true, Ordering::Release);
                        p.cancel();
                        break;
                    }
                }
            }));
        }
        for handle in handles {
            if handle.join().is_err() {
                *error.lock().unwrap() = Some("task panicked".into());
                stop.store(true, Ordering::Release);
                p.cancel();
            }
        }
        done.store(true, Ordering::Release);
        let _ = watcher.join();
    });
    if let Some(e) = error.into_inner().map_err(|_| "task error lock poisoned")? {
        return Err(e.into());
    }
    let marker = fs::read(dir.join("cancel.request"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let timed_out = start.elapsed().as_secs_f64() >= s.limits.wall_time_seconds
        || marker.as_ref().is_some_and(|m| m["reason"] == "timed_out");
    let status = if timed_out {
        "timed_out"
    } else if stop.load(Ordering::Acquire) || dir.join("cancel.request").exists() {
        "cancelled"
    } else {
        "completed"
    };
    let run = result::manifest(
        dir,
        &Request::Job(s),
        status,
        true,
        start.elapsed().as_secs_f64(),
        &exe_digest,
    )?;
    storage::write_atomic_json(&dir.join("run.json"), &run)?;
    Ok(())
}
pub fn run(worker: impl Worker) {
    let run = (|| -> Result<()> {
        let args = std::env::args().collect::<Vec<_>>();
        let value = match args.get(1).map(String::as_str) {
            Some("describe") if args.len() == 2 => worker.describe(),
            Some("validate") if args.len() == 3 => {
                let (s, _) = prepared(&worker, Path::new(&args[2]))?;
                json!({"resolved":s})
            }
            Some("execute") if args.len() == 3 => {
                execute(&worker, Path::new(&args[2]))?;
                json!({"executed":true})
            }
            _ => {
                return Err(
                    "usage: WORKER describe | validate SPEC | execute JOB_DIRECTORY".into(),
                );
            }
        };
        println!("{}", serde_json::to_string(&value)?);
        Ok(())
    })();
    if let Err(e) = run {
        eprintln!(
            "{}",
            json!({"error":{"code":"worker_error","message":e.to_string()}})
        );
        std::process::exit(1);
    }
}
