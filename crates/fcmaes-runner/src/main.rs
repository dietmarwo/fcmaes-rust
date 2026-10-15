use clap::{Parser, Subcommand};
use fcmaes_job_api::{process::*, result, storage::*, *};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Parser)]
#[command(
    version,
    about = "Durable native fcmaes experiments (Linux foreground/systemd user services)"
)]
struct Cli {
    #[arg(
        long,
        global = true,
        env = "FCMAES_RUN_ROOT",
        default_value = ".fcmaes-jobs"
    )]
    root: PathBuf,
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    Node {
        #[command(subcommand)]
        command: Node,
    },
    Workers {
        #[command(subcommand)]
        command: Workers,
    },
    Problems {
        #[command(subcommand)]
        command: Problems,
    },
    Validate {
        #[arg(long)]
        spec: PathBuf,
    },
    Run {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Submit {
        #[arg(long)]
        spec: PathBuf,
    },
    Jobs {
        #[command(subcommand)]
        command: Jobs,
    },
    Result {
        path: PathBuf,
    },
    Compare {
        #[arg(required=true,num_args=2..)]
        paths: Vec<PathBuf>,
    },
    #[command(hide = true)]
    Launch {
        id: String,
    },
}
#[derive(Subcommand)]
enum Node {
    Init {
        #[arg(long)]
        capacity: usize,
    },
    Doctor,
}
#[derive(Subcommand)]
enum Workers {
    Add {
        #[arg(long)]
        id: String,
        #[arg(long)]
        executable: PathBuf,
    },
    List,
}
#[derive(Subcommand)]
enum Problems {
    List,
    Describe { id: String },
}
#[derive(Subcommand)]
enum Jobs {
    List,
    Get {
        id: String,
    },
    Wait {
        id: String,
        #[arg(long, default_value_t = 60.0)]
        timeout: f64,
    },
    /// Stop a job. By default the active retry wave finishes and is kept;
    /// `--now` kills the worker after one second and loses the active wave.
    Cancel {
        id: String,
        #[arg(long)]
        now: bool,
    },
    Reconcile,
    Repair {
        id: String,
    },
    Artifact {
        id: String,
        name: String,
    },
}
/// Number of SIGINT/SIGTERM/SIGHUP received: one requests a graceful stop,
/// two or more an immediate one.
static SIGNALS: AtomicUsize = AtomicUsize::new(0);
extern "C" fn stop_handler(_: i32) {
    SIGNALS.fetch_add(1, Ordering::Relaxed);
}
/// Kill delay after an immediate stop request.
const IMMEDIATE_GRACE: Duration = Duration::from_secs(1);
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
enum Stop {
    None,
    Graceful,
    Immediate,
}
/// Pending stop request from signals or `cancel.request`. The launcher's own
/// timeout marker (reason `timed_out`) is not a cancellation.
fn stop_request(dir: &Path) -> Stop {
    let from_signals = match SIGNALS.load(Ordering::Relaxed) {
        0 => Stop::None,
        1 => Stop::Graceful,
        _ => Stop::Immediate,
    };
    let path = dir.join("cancel.request");
    let from_file = if !path.exists() {
        Stop::None
    } else {
        match read(&path) {
            Ok(v) if v["reason"] == "timed_out" => Stop::None,
            Ok(v) if v["mode"] == "graceful" => Stop::Graceful,
            // An unreadable or unknown request is treated as the stronger one.
            _ => Stop::Immediate,
        }
    };
    if from_file > from_signals {
        from_file
    } else {
        from_signals
    }
}
fn read(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?)
}
fn optional(path: &Path) -> Result<Value> {
    if path.exists() {
        read(path)
    } else {
        Ok(json!({}))
    }
}
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn boot() -> Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_owned())
}
fn output(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
fn spec(path: &Path) -> Result<JobSpec> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice::<JobSpec>(&bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .resolve()
}
/// True when neither the launcher (`job.lock`) nor the worker (`worker.lock`)
/// of an attempt is alive. Both hold their lock for their whole lifetime and
/// the native worker starts no processes of its own. Errors count as alive.
fn attempt_processes_gone(dir: &Path) -> bool {
    matches!(lock_is_free(&dir.join("job.lock")), Ok(true))
        && matches!(lock_is_free(&dir.join("worker.lock")), Ok(true))
}
fn capacity(root: &Path) -> Result<u64> {
    read(&root.join("node.json"))?["capacity"]
        .as_u64()
        .ok_or_else(|| "node init --capacity required".into())
}
fn worker(root: &Path) -> Result<Value> {
    let w = read(&root.join("workers.json"))?["native"].clone();
    if w.is_null() {
        return Err("register --id native first".into());
    }
    Ok(w)
}
fn directory(root: &Path, id: &str) -> Result<PathBuf> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid job id".into());
    }
    let entry = read(&root.join("requests.json"))?[id].clone();
    Ok(PathBuf::from(
        entry["directory"].as_str().ok_or("unknown job")?,
    ))
}
fn check_systemd() -> Result<()> {
    let (s, out, _) = bounded_command(
        &[
            "systemctl".into(),
            "--user".into(),
            "show".into(),
            "--property=Version".into(),
        ],
        vec![],
        Duration::from_secs(10),
        65536,
    )?;
    if !s.success()
        || !String::from_utf8_lossy(&out)
            .lines()
            .any(|x| x.starts_with("Version=") && x.len() > 8)
    {
        return Err("systemd user manager unavailable; use foreground run".into());
    }
    Ok(())
}
fn prepare(
    root: &Path,
    input: &Path,
    dest: Option<PathBuf>,
    detached: bool,
) -> Result<(String, bool)> {
    let resolved = spec(input)?;
    if resolved.execution.workers as u64 > capacity(root)? {
        return Err("workers exceed node capacity".into());
    }
    let w = worker(root)?;
    let executable = Path::new(
        w["executable"]
            .as_str()
            .ok_or("invalid worker registration")?,
    );
    let bytes = fs::read(executable)?;
    let hash = digest(&bytes);
    if w["digest"] != hash {
        return Err("worker changed since registration; register again explicitly".into());
    }
    let id = digest(resolved.request_key.as_bytes())[..32].to_owned();
    let identity = digest(&serde_json::to_vec(
        &json!({"spec":resolved,"worker_digest":hash}),
    )?);
    let _guard = lock(&root.join("registry.lock"))?;
    let mut requests = optional(&root.join("requests.json"))?;
    if !requests[&id].is_null() {
        if requests[&id]["identity"] != identity {
            return Err("request_key conflict: normalized spec or worker differs".into());
        }
        return Ok((id, false));
    }
    if detached {
        check_systemd()?;
    }
    let dir = dest.unwrap_or_else(|| root.join("jobs").join(&id));
    let dir = if dir.is_absolute() {
        dir
    } else {
        std::env::current_dir()?.join(dir)
    };
    if dir.exists() {
        return Err("output already exists; refusing overwrite".into());
    }
    fs::create_dir_all(dir.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&dir)?;
    let dir = fs::canonicalize(dir)?;
    write_atomic_json(&dir.join("submitted.json"), &read(input)?)?;
    write_atomic_json(&dir.join("resolved.json"), &resolved)?;
    fs::write(dir.join("worker"), bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir.join("worker"), fs::Permissions::from_mode(0o700))?;
    }
    fs::File::open(dir.join("worker"))?.sync_all()?;
    let unit = if detached {
        Some(format!("fcmaes-{id}.service"))
    } else {
        None
    };
    write_atomic_json(
        &dir.join("provenance.json"),
        &json!({"worker_digest":hash,"registered_executable":executable,"identity":identity,"build":w["description"]["build"],"unit":unit,"boot_id":boot()?,"attempt_id":id}),
    )?;
    write_atomic_json(
        &dir.join("job.json"),
        &json!({"id":id,"state":"submitted","submitted_at":now(),"unit":unit,"boot_id":boot()?}),
    )?;
    requests[&id] = json!({"identity":identity,"directory":dir});
    write_atomic_json(&root.join("requests.json"), &requests)?;
    Ok((id, true))
}
fn submit(root: &Path, id: &str) -> Result<()> {
    let dir = directory(root, id)?;
    let mut state = read(&dir.join("job.json"))?;
    state["state"] = json!("starting");
    write_atomic_json(&dir.join("job.json"), &state)?;
    let unit = state["unit"].as_str().ok_or("missing unit")?;
    let exe = std::env::current_exe()?;
    let args = vec![
        "systemd-run".into(),
        "--user".into(),
        "--quiet".into(),
        format!("--unit={unit}"),
        "--property=KillMode=control-group".into(),
        "--property=Restart=no".into(),
        "--property=TimeoutStopSec=5s".into(),
        format!("--working-directory={}", dir.display()),
        exe.to_string_lossy().into_owned(),
        "--root".into(),
        root.to_string_lossy().into_owned(),
        "launch".into(),
        id.into(),
    ];
    match bounded_command(&args,vec![],Duration::from_secs(15),65536) {
        Ok((s,_,_)) if s.success()=>Ok(()),
        _=>Err(format!("start acknowledgement unknown for {unit}; inspect recorded unit; repeating submit will NOT launch again").into()),
    }
}
/// Drop reservations whose holders are provably gone: a previous boot, an
/// inactive systemd unit with an empty cgroup, or an attempt whose launcher
/// and worker locks are both free. Unknown liveness keeps the reservation.
fn reconcile_ledger(root: &Path, ledger: &mut Value) -> Result<()> {
    let current_boot = boot()?;
    let keys: Vec<String> = ledger
        .as_object()
        .ok_or("invalid ledger")?
        .keys()
        .cloned()
        .collect();
    for id in keys {
        let r = &ledger[&id];
        let old_boot = r["boot_id"].as_str().is_some_and(|b| b != current_boot);
        // A running service owns its reservation. A missing/unknown manager state is not proof of death.
        let stopped = if let Some(unit) = r["unit"].as_str() {
            let args = vec![
                "systemctl".into(),
                "--user".into(),
                "show".into(),
                unit.into(),
                "--property=ActiveState,ControlGroup".into(),
            ];
            match bounded_command(&args, vec![], Duration::from_secs(5), 16384) {
                Ok((s, out, _)) if s.success() => {
                    let v = String::from_utf8_lossy(&out);
                    let inactive = v
                        .lines()
                        .any(|x| x == "ActiveState=inactive" || x == "ActiveState=failed");
                    let cg = v.lines().find_map(|x| x.strip_prefix("ControlGroup="));
                    // Only accept manager-empty cgroup, or an explicitly unpopulated cgroup.events.
                    let empty = match cg {
                        Some("") => true,
                        Some(c) if c.starts_with('/') && !c.contains("..") => {
                            fs::read_to_string(format!("/sys/fs/cgroup{c}/cgroup.events"))
                                .map(|s| s.lines().any(|x| x == "populated 0"))
                                .unwrap_or(false)
                        }
                        _ => false,
                    };
                    inactive && empty
                }
                _ => false,
            }
        } else {
            false
        };
        // Foreground jobs have no unit; a launcher killed by SIGKILL (or a
        // closed terminal) would otherwise hold its slots until reboot.
        let gone = directory(root, &id)
            .map(|dir| attempt_processes_gone(&dir))
            .unwrap_or(false);
        if old_boot || stopped || gone {
            ledger.as_object_mut().unwrap().remove(&id);
        }
    }
    Ok(())
}
fn admit(root: &Path, id: &str, dir: &Path, slots: usize) -> Result<bool> {
    if stop_request(dir) != Stop::None {
        return Ok(false);
    }
    let _guard = lock(&root.join("slots.lock"))?;
    let mut ledger = optional(&root.join("slots.json"))?;
    reconcile_ledger(root, &mut ledger)?;
    let used = ledger
        .as_object()
        .ok_or("bad ledger")?
        .values()
        .try_fold(0u64, |s, v| {
            s.checked_add(v["slots"].as_u64().ok_or("bad slots")?)
                .ok_or("slots overflow")
        })?;
    if ledger.get(id).is_some() {
        return Err("attempt already reserved; refusing duplicate launch".into());
    }
    let accepted = used
        .checked_add(slots as u64)
        .is_some_and(|n| n <= capacity(root).unwrap_or(0));
    if accepted {
        let p = read(&dir.join("provenance.json"))?;
        ledger[id] = json!({"slots":slots,"boot_id":boot()?,"unit":p["unit"],"launcher_pid":std::process::id()});
    }
    write_atomic_json(&root.join("slots.json"), &ledger)?;
    Ok(accepted)
}
fn release(root: &Path, id: &str) -> Result<()> {
    let _g = lock(&root.join("slots.lock"))?;
    let mut l = optional(&root.join("slots.json"))?;
    l.as_object_mut().ok_or("invalid ledger")?.remove(id);
    write_atomic_json(&root.join("slots.json"), &l)?;
    Ok(())
}
fn finish(dir: &Path, spec: &JobSpec, status: &str, exact: bool, elapsed: f64) -> Result<()> {
    let p = read(&dir.join("provenance.json"))?;
    let mut run = result::manifest(
        dir,
        spec,
        status,
        exact,
        elapsed,
        p["worker_digest"].as_str().ok_or("missing worker digest")?,
    )?;
    run["runner"]["attempt_id"] = p["attempt_id"].clone();
    run["runner"]["build"] = p["build"].clone();
    write_atomic_json(&dir.join("run.json"), &run)?;
    let mut state = read(&dir.join("job.json"))?;
    state["state"] = json!(status);
    state["finished_at"] = json!(now());
    write_atomic_json(&dir.join("job.json"), &state)?;
    Ok(())
}
fn launch(root: &Path, id: &str) -> Result<()> {
    let dir = directory(root, id)?;
    let _owner = lock(&dir.join("job.lock"))?;
    let mut state = read(&dir.join("job.json"))?;
    if !matches!(state["state"].as_str(), Some("submitted" | "starting")) {
        return Err("attempt already started; no implicit restart".into());
    }
    let s = spec(&dir.join("resolved.json"))?;
    // SIGHUP matters for foreground runs: a closed terminal or agent shell
    // must stop the job instead of orphaning the worker.
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        unsafe {
            libc::signal(signal, stop_handler as *const () as libc::sighandler_t);
        }
    }
    state["state"] = json!("waiting");
    write_atomic_json(&dir.join("job.json"), &state)?;
    let queue = Instant::now();
    loop {
        if stop_request(&dir) != Stop::None {
            finish(&dir, &s, "cancelled", true, 0.0)?;
            return Ok(());
        }
        if admit(root, id, &dir, s.execution.workers)? {
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }
    let start = Instant::now();
    state["state"] = json!("running");
    state["started_at"] = json!(now());
    state["queue_seconds"] = json!(queue.elapsed().as_secs_f64());
    if let Err(e) = write_atomic_json(&dir.join("job.json"), &state) {
        release(root, id)?;
        return Err(e.into());
    }
    let mut safely_stopped = true;
    let outcome = execute_worker(&dir, &s, start, &mut safely_stopped);
    // execute_worker always reaps/cleans the group once spawn succeeds.
    if safely_stopped {
        release(root, id)?;
    }
    match outcome {
        Ok((status, exact)) => finish(&dir, &s, &status, exact, start.elapsed().as_secs_f64()),
        Err(e) => {
            let _ = finish(&dir, &s, "failed", false, start.elapsed().as_secs_f64());
            Err(e)
        }
    }
}
fn execute_worker(
    dir: &Path,
    s: &JobSpec,
    start: Instant,
    safely_stopped: &mut bool,
) -> Result<(String, bool)> {
    let p = read(&dir.join("provenance.json"))?;
    if json!(digest(&fs::read(dir.join("worker"))?)) != p["worker_digest"] {
        return Err("staged worker digest mismatch".into());
    }
    let stdout = fs::File::create(dir.join("stdout.log"))?;
    let stderr = fs::File::create(dir.join("stderr.log"))?;
    let mut cmd = Command::new(dir.join("worker"));
    cmd.arg("execute")
        .arg(dir)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1");
    group(&mut cmd);
    let mut child = cmd.spawn()?;
    let pgid = child.id();
    *safely_stopped = false;
    let a = child.stdout.take().unwrap();
    let b = child.stderr.take().unwrap();
    let ta = thread::spawn(move || drain_file(a, stdout, 1_048_576));
    let tb = thread::spawn(move || drain_file(b, stderr, 1_048_576));
    // The worker checks `cancel.request` between retry waves. A graceful stop
    // lets the active wave finish; the wall-time deadline plus the declared
    // grace, or an immediate stop, kills the worker group.
    let wall = Duration::from_secs_f64(s.budget.wall_time_seconds);
    let hard_deadline = start + wall + Duration::from_secs_f64(s.budget.grace_seconds);
    let mut reason = None;
    let mut immediate_deadline: Option<Instant> = None;
    let polled = (|| -> Result<std::process::ExitStatus> {
        loop {
            if let Some(exit) = child.try_wait()? {
                return Ok(exit);
            }
            let request = stop_request(dir);
            if reason.is_none() {
                if request != Stop::None {
                    reason = Some("cancelled");
                } else if start.elapsed() >= wall {
                    reason = Some("timed_out");
                    write_atomic_json(
                        &dir.join("cancel.request"),
                        &json!({"mode":"graceful","reason":"timed_out"}),
                    )?;
                }
            }
            // A signal-only request must still reach the worker.
            if request != Stop::None && !dir.join("cancel.request").exists() {
                let mode = if request == Stop::Immediate {
                    "immediate"
                } else {
                    "graceful"
                };
                write_atomic_json(
                    &dir.join("cancel.request"),
                    &json!({"mode":mode,"reason":"signal","requested_at":now()}),
                )?;
            }
            if request == Stop::Immediate && immediate_deadline.is_none() {
                immediate_deadline = Some(Instant::now() + IMMEDIATE_GRACE);
            }
            let deadline = immediate_deadline.map_or(hard_deadline, |d| d.min(hard_deadline));
            if Instant::now() >= deadline {
                terminate_process_group(&mut child);
                return Ok(child.wait()?);
            }
            thread::sleep(Duration::from_millis(10));
        }
    })();
    terminate_process_group(&mut child);
    let _ = child.wait();
    let verify = Instant::now();
    loop {
        match group_stopped(pgid) {
            Ok(true) => break,
            _ if verify.elapsed() < Duration::from_secs(2) => {
                thread::sleep(Duration::from_millis(10))
            }
            _ => {
                write_atomic_json(
                    &dir.join("containment.unknown"),
                    &json!({"pgid":pgid,"reason":"group liveness unknown; slots retained"}),
                )?;
                return Err("worker group termination unverified; reservation retained".into());
            }
        }
    }
    *safely_stopped = true;
    ta.join().map_err(|_| "stdout drainer panic")??;
    tb.join().map_err(|_| "stderr drainer panic")??;
    let exit = polled?;
    let worker = if exit.success() {
        worker_outcome(dir, s)?
    } else {
        None
    };
    let normal = worker.is_some();
    let status = match (reason, worker) {
        (Some(reason), _) => reason.to_string(),
        (None, Some((status, _))) => status,
        (None, None) => "failed".into(),
    };
    Ok((status, normal))
}
/// The worker's own `run.json`, if it is a complete manifest consistent with
/// the journal: returns its status and elapsed time.
fn worker_outcome(dir: &Path, s: &JobSpec) -> Result<Option<(String, f64)>> {
    let path = dir.join("run.json");
    if !path.exists() {
        return Ok(None);
    }
    let records = result::replay(dir, s)?;
    let r = read(&path)?;
    let status = r["status"].as_str().unwrap_or_default().to_owned();
    let consistent = r["runner"]["spec_digest"] == s.digest()?
        && r["runner"]["accounting_complete"] == true
        && matches!(status.as_str(), "completed" | "cancelled" | "timed_out")
        && r["actual_evaluations"].as_u64()
            == Some(records.iter().map(|r| r.actual_evaluations).sum());
    if !consistent {
        return Ok(None);
    }
    if status == "completed" && records.len() != s.budget.retries {
        return Err("worker completed without all retries".into());
    }
    Ok(Some((status, r["elapsed_seconds"].as_f64().unwrap_or(0.0))))
}
fn get(root: &Path, id: &str) -> Result<Value> {
    let dir = directory(root, id)?;
    let mut v = read(&dir.join("job.json"))?;
    v["directory"] = json!(dir);
    let s = spec(&dir.join("resolved.json"))?;
    let r = result::replay(&dir, &s)?;
    v["completed_retries"] = json!(r.len());
    v["known_evaluations_lower_bound"] = json!(r.iter().map(|r| r.actual_evaluations).sum::<u64>());
    if v["boot_id"].as_str() != Some(&boot()?) {
        v["observed_state"] = json!("interrupted_after_reboot");
    }
    // A launcher holds job.lock for its whole lifetime, so a free lock in an
    // active state means it died (SIGKILL, closed terminal) without finishing.
    if matches!(v["state"].as_str(), Some("waiting" | "running"))
        && v["observed_state"].is_null()
        && matches!(lock_is_free(&dir.join("job.lock")), Ok(true))
    {
        v["observed_state"] = json!("launcher_lost");
        v["recovery"] = json!("run `jobs repair` after the worker has exited");
    }
    if matches!(
        v["state"].as_str(),
        Some("waiting" | "starting" | "running")
    ) && let Some(unit) = v["unit"].as_str()
    {
        let args = vec![
            "systemctl".into(),
            "--user".into(),
            "show".into(),
            unit.into(),
            "--property=ActiveState".into(),
        ];
        v["service_observation"] =
            match bounded_command(&args, vec![], Duration::from_secs(3), 4096) {
                Ok((s, out, _)) if s.success() => json!(String::from_utf8_lossy(&out).trim()),
                _ => json!("unknown; reservation retained"),
            };
    }
    if dir.join("cancel.request").exists() {
        v["cancel_requested"] = json!(true);
    }
    Ok(v)
}
fn main_result() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.root)?;
    let root = fs::canonicalize(&cli.root)?;
    match cli.command {
        Cmd::Node {
            command: Node::Init { capacity: c },
        } => {
            if c == 0 {
                return Err("capacity must be positive".into());
            }
            let _g = lock(&root.join("slots.lock"))?;
            if root.join("node.json").exists() {
                return Err("node already initialized; keep its capacity stable".into());
            }
            write_atomic_json(&root.join("node.json"), &json!({"capacity":c}))?;
            output(json!({"capacity":c}))
        }
        Cmd::Node {
            command: Node::Doctor,
        } => output(
            json!({"systemd_user_manager":check_systemd().is_ok(),"boot_id":boot()?,"capacity":capacity(&root).ok(),"logout_survival":"requires user manager lingering; verify on host"}),
        ),
        Cmd::Workers {
            command: Workers::Add { id, executable },
        } => {
            if id != "native" {
                return Err("initial release supports only the native worker protocol".into());
            }
            let executable = fs::canonicalize(executable)?;
            let (status, out, _) = bounded_command(
                &[executable.to_string_lossy().into_owned(), "describe".into()],
                vec![],
                Duration::from_secs(10),
                65536,
            )?;
            let description: Value = serde_json::from_slice(&out)?;
            if !status.success()
                || ["protocol", "problems", "optimizers", "parallelism"]
                    .iter()
                    .any(|k| description[*k] != discovery()[*k])
            {
                return Err("worker capability profile mismatch".into());
            }
            let _g = lock(&root.join("registry.lock"))?;
            let mut w = optional(&root.join("workers.json"))?;
            w[&id] = json!({"executable":executable,"digest":digest(&fs::read(&executable)?),"description":description});
            write_atomic_json(&root.join("workers.json"), &w)?;
            output(w)
        }
        Cmd::Workers {
            command: Workers::List,
        } => output(optional(&root.join("workers.json"))?),
        Cmd::Problems {
            command: Problems::List,
        } => output(discovery()),
        Cmd::Problems {
            command: Problems::Describe { id },
        } => {
            let d = discovery();
            output(
                d["problems"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| p["id"] == id)
                    .ok_or("unknown problem")?
                    .clone(),
            )
        }
        Cmd::Validate { spec: p } => {
            let s = spec(&p)?;
            output(
                json!({"valid":true,"requested_evaluations":s.requested()?,"maximum_actual_evaluations":s.maximum_evaluations()?,"resolved":s}),
            )
        }
        Cmd::Run {
            spec: p,
            output: dest,
        } => {
            let (id, new) = prepare(&root, &p, dest, false)?;
            if new {
                launch(&root, &id)?;
            }
            output(get(&root, &id)?)
        }
        Cmd::Submit { spec: p } => {
            let (id, new) = prepare(&root, &p, None, true)?;
            if new {
                submit(&root, &id)?;
            }
            output(get(&root, &id)?)
        }
        Cmd::Launch { id } => launch(&root, &id),
        Cmd::Result { path } => output(read(&if path.is_dir() {
            path.join("run.json")
        } else {
            path
        })?),
        Cmd::Compare { paths } => {
            let runs = paths
                .iter()
                .map(|p| {
                    read(&if p.is_dir() {
                        p.join("run.json")
                    } else {
                        p.clone()
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            output(result::compare(&runs)?)
        }
        Cmd::Jobs { command } => match command {
            Jobs::List => {
                let requests = optional(&root.join("requests.json"))?;
                let mut jobs = vec![];
                for id in requests.as_object().ok_or("bad requests")?.keys() {
                    jobs.push(get(&root, id)?);
                }
                output(json!(jobs))
            }
            Jobs::Get { id } => output(get(&root, &id)?),
            Jobs::Wait { id, timeout } => {
                if !timeout.is_finite() || timeout <= 0.0 {
                    return Err("positive finite timeout required".into());
                }
                let t = Instant::now();
                loop {
                    let v = get(&root, &id)?;
                    if matches!(
                        v["state"].as_str(),
                        Some("completed" | "failed" | "cancelled" | "timed_out" | "interrupted")
                    ) || !v["observed_state"].is_null()
                        || t.elapsed().as_secs_f64() >= timeout
                    {
                        return output(v);
                    }
                    thread::sleep(Duration::from_millis(250));
                }
            }
            Jobs::Cancel { id, now: immediate } => {
                let dir = directory(&root, &id)?;
                let path = dir.join("cancel.request");
                // Never downgrade an immediate request to a graceful one.
                let existing = optional(&path)?;
                if immediate || existing["mode"] != "immediate" || existing["reason"] == "timed_out"
                {
                    let mode = if immediate { "immediate" } else { "graceful" };
                    write_atomic_json(&path, &json!({"mode":mode,"requested_at":now()}))?;
                }
                output(get(&root, &id)?)
            }
            Jobs::Reconcile => {
                let _g = lock(&root.join("slots.lock"))?;
                let mut l = optional(&root.join("slots.json"))?;
                reconcile_ledger(&root, &mut l)?;
                write_atomic_json(&root.join("slots.json"), &l)?;
                output(l)
            }
            Jobs::Repair { id } => {
                let dir = directory(&root, &id)?;
                let owner = fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join("job.lock"))?;
                fs2::FileExt::try_lock_exclusive(&owner)
                    .map_err(|_| "launcher still owns this attempt")?;
                let worker = fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join("worker.lock"))?;
                fs2::FileExt::try_lock_exclusive(&worker)
                    .map_err(|_| "worker still owns this attempt")?;
                let state = read(&dir.join("job.json"))?;
                if matches!(
                    state["state"].as_str(),
                    Some("completed" | "failed" | "cancelled" | "timed_out" | "interrupted")
                ) {
                    return Err(
                        "attempt already finalized; repair is for interrupted attempts".into(),
                    );
                }
                // Holding both locks proves the launcher and worker are gone, so
                // this attempt's reservation can be released.
                {
                    let _g = lock(&root.join("slots.lock"))?;
                    let mut ledger = optional(&root.join("slots.json"))?;
                    reconcile_ledger(&root, &mut ledger)?;
                    ledger.as_object_mut().ok_or("invalid ledger")?.remove(&id);
                    write_atomic_json(&root.join("slots.json"), &ledger)?;
                }
                let s = spec(&dir.join("resolved.json"))?;
                result::replay(&dir, &s)?;
                if dir.join("retries.jsonl").exists() {
                    repair_tail::<result::RetryRecord>(&dir.join("retries.jsonl"), 1)?;
                }
                // An orphaned worker may have finished on its own after the
                // launcher died; keep its exact result.
                if let Some((status, elapsed)) = worker_outcome(&dir, &s)? {
                    finish(&dir, &s, &status, true, elapsed)?;
                } else {
                    finish(&dir, &s, "interrupted", false, 0.0)?;
                    // The execution duration is unknown after launcher loss.
                    let mut r = read(&dir.join("run.json"))?;
                    r["elapsed_seconds"] = Value::Null;
                    write_atomic_json(&dir.join("run.json"), &r)?;
                }
                output(get(&root, &id)?)
            }
            Jobs::Artifact { id, name } => {
                let dir = directory(&root, &id)?;
                let run = read(&dir.join("run.json"))?;
                let rel = run["artifacts"][&name]
                    .as_str()
                    .ok_or("unknown artifact id")?;
                let path = fs::canonicalize(dir.join(rel))?;
                if !path.starts_with(fs::canonicalize(&dir)?) {
                    return Err("artifact escapes job directory".into());
                }
                if fs::metadata(&path)?.len() > 1_048_576 {
                    return Err("artifact exceeds inline cap; use job directory".into());
                }
                output(json!({"name":name,"content":fs::read_to_string(path)?}))
            }
        },
    }
}
fn main() {
    if let Err(e) = main_result() {
        eprintln!(
            "{}",
            json!({"error":{"code":"runner_error","message":e.to_string()}})
        );
        std::process::exit(2);
    }
}
