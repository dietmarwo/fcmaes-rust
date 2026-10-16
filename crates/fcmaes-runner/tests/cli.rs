//! End-to-end checks of the foreground runner: idempotency, comparison, slot
//! admission, cancellation modes, the wall-time grace and launcher recovery.
//!
//! Needs the worker binary next to `fcmaes-run`; `cargo test --workspace`
//! builds it because the worker crate has integration tests of its own.
#![cfg(target_os = "linux")]
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

const RUN: &str = env!("CARGO_BIN_EXE_fcmaes-run");
/// About one to two seconds per retry in a debug build.
const LONG_RETRY: u64 = 150_000;
/// Never finishes within a test; the job must be stopped.
const ENDLESS_RETRY: u64 = 50_000_000;
const TIMEOUT: Duration = Duration::from_secs(120);

fn worker_binary() -> PathBuf {
    let path = Path::new(RUN).with_file_name("fcmaes-native-worker");
    assert!(
        path.exists(),
        "{} missing: run `cargo test --workspace` or build fcmaes-native-worker first",
        path.display()
    );
    path
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn job_id(key: &str) -> String {
    fcmaes_job_api::storage::digest(key.as_bytes())[..32].to_owned()
}

fn base(key: &str) -> Value {
    json!({
        "schema_version": "fcmaes.job/0.1",
        "request_key": key,
        "problem": {"id": "benchmark.rosenbrock", "version": "1", "config": {"dimension": 20},
                    "bounds": {"lower": [-5.0], "upper": [5.0], "broadcast": true}},
        "optimizer": {"kind": "biteopt", "parameters": {}},
        "execution": {"strategy": "independent_retry", "seed": 3, "workers": 2,
                      "retry_wave_size": 2},
        "budget": {"retries": 4, "evaluations_per_retry": 2000, "evaluation_limit_mode": "hard",
                   "wall_time_seconds": 120.0},
        "outputs": {"progress": "completed_retries"}
    })
}

struct Node {
    dir: tempfile::TempDir,
    specs: AtomicUsize,
}

impl Node {
    fn new(capacity: usize) -> Self {
        let node = Node {
            dir: tempfile::tempdir().unwrap(),
            specs: AtomicUsize::new(0),
        };
        node.ok(&["node", "init", "--capacity", &capacity.to_string()]);
        let worker = worker_binary();
        node.ok(&[
            "workers",
            "add",
            "--id",
            "native",
            "--executable",
            worker.to_str().unwrap(),
        ]);
        node
    }
    fn root(&self) -> PathBuf {
        self.dir.path().join("root")
    }
    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(RUN);
        command.arg("--root").arg(self.root()).args(args);
        command
    }
    fn output(&self, args: &[&str]) -> Output {
        self.command(args).output().unwrap()
    }
    fn ok(&self, args: &[&str]) -> Value {
        let out = self.output(args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn err(&self, args: &[&str]) -> String {
        let out = self.output(args);
        assert!(!out.status.success(), "{args:?} unexpectedly succeeded");
        String::from_utf8_lossy(&out.stderr).into_owned()
    }
    /// Writes a spec derived from `base(key)` and returns its path.
    fn spec(&self, key: &str, edit: impl FnOnce(&mut Value)) -> String {
        let mut spec = base(key);
        edit(&mut spec);
        let n = self.specs.fetch_add(1, Ordering::Relaxed);
        let path = self.dir.path().join(format!("spec-{n}.json"));
        fs::write(&path, spec.to_string()).unwrap();
        path.to_str().unwrap().to_owned()
    }
    fn spawn_run(&self, spec: &str) -> Child {
        let log = fs::File::create(self.dir.path().join(format!("{}.log", spec.len()))).unwrap();
        self.command(&["run", "--spec", spec])
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap()
    }
    fn get(&self, id: &str) -> Value {
        self.ok(&["jobs", "get", id])
    }
    fn wait_state(&self, id: &str, state: &str) -> Value {
        let start = Instant::now();
        loop {
            let out = self.output(&["jobs", "get", id]);
            if out.status.success() {
                let v: Value = serde_json::from_slice(&out.stdout).unwrap();
                if v["state"] == state {
                    return v;
                }
            }
            assert!(start.elapsed() < TIMEOUT, "job {id} never reached {state}");
            thread::sleep(Duration::from_millis(50));
        }
    }
    fn ledger(&self) -> Value {
        read(&self.root().join("slots.json"))
    }
}

/// Waits until the worker has created its journal, i.e. is inside its first wave.
fn wait_for_journal(dir: &Path) {
    let start = Instant::now();
    while !dir.join("retries.jsonl").exists() {
        assert!(start.elapsed() < TIMEOUT, "worker never started");
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_millis(200));
}

fn wait_child(child: &mut Child) {
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        assert!(start.elapsed() < TIMEOUT, "runner did not exit");
        thread::sleep(Duration::from_millis(50));
    }
}

fn journal_total(run: &Value) -> u64 {
    run["runner"]["known_evaluations_lower_bound"]
        .as_u64()
        .unwrap()
}

#[test]
fn foreground_run_is_idempotent_and_comparable() {
    let node = Node::new(2);
    let bite = node.spec("bite", |_| {});
    let first = node.ok(&["run", "--spec", &bite]);
    assert_eq!(first["state"], "completed");
    let dir = PathBuf::from(first["directory"].as_str().unwrap());
    let run = read(&dir.join("run.json"));
    assert_eq!(run["actual_evaluations"], 8000);
    assert_eq!(run["requested_evaluations"], 8000);
    let journal = fs::read(dir.join("retries.jsonl")).unwrap();

    // Same key and content: the existing job is returned, nothing executes.
    let again = node.ok(&["run", "--spec", &bite]);
    assert_eq!(again["id"], first["id"]);
    assert_eq!(again["finished_at"], first["finished_at"]);
    assert_eq!(fs::read(dir.join("retries.jsonl")).unwrap(), journal);

    // Same key, different experiment: refused.
    let conflict = node.spec("bite", |s| s["execution"]["seed"] = json!(4));
    assert!(
        node.err(&["run", "--spec", &conflict])
            .contains("request_key conflict")
    );

    let de = node.spec("de", |s| {
        s["optimizer"] = json!({"kind": "de", "parameters": {"population": 12}});
        s["budget"]["evaluation_limit_mode"] = json!("soft");
    });
    let de = node.ok(&["run", "--spec", &de]);
    let de_dir = de["directory"].as_str().unwrap();
    let cmp = node.ok(&["compare", dir.to_str().unwrap(), de_dir]);
    assert_eq!(cmp["summary"].as_array().unwrap().len(), 2);
    assert_eq!(cmp["summary"][1]["optimizer"], "de");
    assert_eq!(cmp["equal_requested_work"], true);
    assert!(cmp["warnings"].to_string().contains("limit modes differ"));

    let sphere = node.spec("sphere", |s| {
        s["problem"]["id"] = json!("benchmark.sphere");
    });
    let sphere = node.ok(&["run", "--spec", &sphere]);
    let sphere_dir = sphere["directory"].as_str().unwrap();
    assert!(
        node.err(&["compare", dir.to_str().unwrap(), sphere_dir])
            .contains("incompatible")
    );
}

#[test]
fn queued_job_waits_and_both_cancel_modes_work() {
    let node = Node::new(2);
    let long = node.spec("long", |s| {
        s["budget"]["evaluations_per_retry"] = json!(ENDLESS_RETRY);
    });
    let small = node.spec("small", |s| {
        s["execution"]["workers"] = json!(1);
        s["execution"]["retry_wave_size"] = json!(1);
    });
    let (long_id, small_id) = (job_id("long"), job_id("small"));

    let mut long_child = node.spawn_run(&long);
    let long_dir = PathBuf::from(
        node.wait_state(&long_id, "running")["directory"]
            .as_str()
            .unwrap(),
    );
    // The worker must be inside its endless wave, or `--now` legitimately
    // stops it cleanly at the first wave boundary.
    wait_for_journal(&long_dir);
    let mut small_child = node.spawn_run(&small);
    node.wait_state(&small_id, "waiting");
    thread::sleep(Duration::from_millis(500));
    assert_eq!(
        node.get(&small_id)["state"],
        "waiting",
        "admitted over capacity"
    );

    // Cancelling a waiting job finishes it without any work or artifacts.
    node.ok(&["jobs", "cancel", &small_id]);
    wait_child(&mut small_child);
    let small_dir = PathBuf::from(node.get(&small_id)["directory"].as_str().unwrap());
    let small_run = read(&small_dir.join("run.json"));
    assert_eq!(small_run["status"], "cancelled");
    assert_eq!(small_run["actual_evaluations"], 0);
    assert_eq!(small_run["artifacts"], json!({}));

    // --now kills the active wave after about one second.
    let started = Instant::now();
    node.ok(&["jobs", "cancel", &long_id, "--now"]);
    wait_child(&mut long_child);
    assert!(started.elapsed() < Duration::from_secs(15));
    let long_run = read(&long_dir.join("run.json"));
    assert_eq!(long_run["status"], "cancelled");
    assert_eq!(long_run["actual_evaluations"], Value::Null);
    assert_eq!(long_run["runner"]["accounting_complete"], false);

    // Its slots are free again.
    assert_eq!(node.ledger(), json!({}));
    let next = node.spec("next", |_| {});
    assert_eq!(node.ok(&["run", "--spec", &next])["state"], "completed");
}

#[test]
fn graceful_cancel_keeps_the_active_wave() {
    let node = Node::new(2);
    let spec = node.spec("graceful", |s| {
        s["budget"]["retries"] = json!(8);
        s["budget"]["evaluations_per_retry"] = json!(LONG_RETRY);
    });
    let id = job_id("graceful");
    let mut child = node.spawn_run(&spec);
    let dir = PathBuf::from(
        node.wait_state(&id, "running")["directory"]
            .as_str()
            .unwrap(),
    );
    wait_for_journal(&dir);
    node.ok(&["jobs", "cancel", &id]);
    wait_child(&mut child);
    let run = read(&dir.join("run.json"));
    assert_eq!(run["status"], "cancelled");
    assert_eq!(run["runner"]["accounting_complete"], true);
    let completed = run["runner"]["completed_retries"].as_u64().unwrap();
    assert!(
        completed > 0 && completed.is_multiple_of(2) && completed < 8,
        "completed {completed} retries"
    );
    assert_eq!(run["actual_evaluations"], completed * LONG_RETRY);
    assert_eq!(
        run["actual_evaluations"].as_u64(),
        Some(journal_total(&run))
    );
}

#[test]
fn wall_time_kills_after_the_declared_grace() {
    let node = Node::new(2);
    let spec = node.spec("timeout", |s| {
        s["budget"]["evaluations_per_retry"] = json!(ENDLESS_RETRY);
        s["budget"]["wall_time_seconds"] = json!(6.0);
        s["budget"]["grace_seconds"] = json!(0.5);
    });
    let id = job_id("timeout");
    let started = Instant::now();
    let mut child = node.spawn_run(&spec);
    let dir = PathBuf::from(
        node.wait_state(&id, "running")["directory"]
            .as_str()
            .unwrap(),
    );
    wait_for_journal(&dir);
    // Only a worker that is inside its wave at the deadline exercises the kill.
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "worker start-up too slow"
    );
    wait_child(&mut child);
    let elapsed = started.elapsed();
    let run = read(&dir.join("run.json"));
    assert_eq!(run["status"], "timed_out");
    assert_eq!(run["actual_evaluations"], Value::Null);
    assert_eq!(run["runner"]["accounting_complete"], false);
    assert!(
        elapsed >= Duration::from_millis(6400) && elapsed < Duration::from_secs(30),
        "{elapsed:?}"
    );
}

#[test]
fn killed_launcher_releases_slots_and_repair_keeps_the_worker_result() {
    let node = Node::new(2);
    let spec = node.spec("killed", |s| {
        s["budget"]["retries"] = json!(2);
        s["budget"]["evaluations_per_retry"] = json!(4 * LONG_RETRY);
    });
    let id = job_id("killed");
    let mut child = node.spawn_run(&spec);
    let dir = PathBuf::from(
        node.wait_state(&id, "running")["directory"]
            .as_str()
            .unwrap(),
    );
    wait_for_journal(&dir);
    child.kill().unwrap(); // SIGKILL: no cleanup in the launcher
    child.wait().unwrap();

    assert_eq!(node.get(&id)["observed_state"], "launcher_lost");
    assert!(
        node.err(&["jobs", "repair", &id])
            .contains("worker still owns")
    );

    // The orphaned worker finishes its wave; repair then keeps its exact result.
    let started = Instant::now();
    let repaired = loop {
        let out = node.output(&["jobs", "repair", &id]);
        if out.status.success() {
            break serde_json::from_slice::<Value>(&out.stdout).unwrap();
        }
        assert!(started.elapsed() < TIMEOUT, "repair never succeeded");
        thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(repaired["state"], "completed");
    let run = read(&dir.join("run.json"));
    assert_eq!(run["actual_evaluations"], 2 * 4 * LONG_RETRY);
    assert_eq!(run["runner"]["accounting_complete"], true);
    assert_eq!(node.ledger(), json!({}));

    let next = node.spec("after-kill", |_| {});
    assert_eq!(node.ok(&["run", "--spec", &next])["state"], "completed");
}

#[test]
fn sighup_stops_a_foreground_job_gracefully() {
    let node = Node::new(2);
    let spec = node.spec("hangup", |s| {
        s["budget"]["retries"] = json!(6);
        s["budget"]["evaluations_per_retry"] = json!(LONG_RETRY);
    });
    let id = job_id("hangup");
    let mut child = node.spawn_run(&spec);
    let dir = PathBuf::from(
        node.wait_state(&id, "running")["directory"]
            .as_str()
            .unwrap(),
    );
    wait_for_journal(&dir);
    let rc = unsafe { libc::kill(child.id() as i32, libc::SIGHUP) };
    assert_eq!(rc, 0);
    wait_child(&mut child);
    let run = read(&dir.join("run.json"));
    assert_eq!(run["status"], "cancelled");
    assert_eq!(run["runner"]["accounting_complete"], true);
    let completed = run["runner"]["completed_retries"].as_u64().unwrap();
    assert!(
        completed > 0 && completed < 6,
        "completed {completed} retries"
    );
    assert_eq!(node.ledger(), json!({}));
}
