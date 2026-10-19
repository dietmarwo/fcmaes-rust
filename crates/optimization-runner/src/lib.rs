use clap::{Parser, Subcommand};
use optimization_job_api::{legacy::discovery, process::*, result, storage::*, *};
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
    about = "Durable optimization jobs: fcmaes, LP-IP and SQP (Linux)"
)]
struct Cli {
    #[arg(long, global = true, env = "OPTIMIZATION_RUN_ROOT")]
    root: Option<PathBuf>,
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
    Events {
        id: String,
    },
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
/// SIGHUP matters for foreground runs: a closed terminal or agent shell must
/// stop the job instead of orphaning the worker.
fn install_stop_handlers() {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        unsafe {
            libc::signal(signal, stop_handler as *const () as libc::sighandler_t);
        }
    }
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
fn state_write(path: &Path, state: &Value) -> Result<()> {
    append_checked(
        &path.with_file_name("events.jsonl"),
        1,
        &json!({"event":"state","state":state["state"],"at":now(),"id":state["id"]}),
    )?;
    write_atomic_json(path, state)?;
    Ok(())
}
fn terminal(state: &Value) -> bool {
    matches!(
        state.as_str(),
        Some("completed" | "failed" | "cancelled" | "timed_out" | "interrupted")
    )
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
fn spec(path: &Path) -> Result<Request> {
    Request::parse(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
}
/// An error of the submitted job itself: `invalid_job`, unless it already
/// has a code.
fn invalid(e: Error) -> Error {
    if e.downcast_ref::<Coded>().is_some() {
        e
    } else {
        coded("invalid_job", e.to_string())
    }
}
fn staged(root: &Path, input: &Path) -> Result<(Request, Value, Vec<u8>, tempfile::TempDir)> {
    let mut s = spec(input).map_err(invalid)?;
    let w = worker(root, s.worker())?;
    let exe = Path::new(w["executable"].as_str().ok_or("invalid executable")?);
    let bytes = fs::read(exe)?;
    if w["digest"] != digest(&bytes) {
        return Err(coded(
            "worker_changed",
            "worker changed since registration; register again explicitly",
        ));
    }
    // The node root must allow execution anyway (attempts run their staged
    // worker there); TMPDIR may be mounted noexec.
    let tmp = tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(root)?;
    let executable = tmp.path().join("worker");
    fs::write(&executable, &bytes)?;
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))?;
    }
    if let Request::Job(ref mut job) = s {
        if !protocol_supported(&w["description"]["protocol"])
            || !w["description"]["profiles"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(&job.payload_schema)))
        {
            return Err(coded(
                "unsupported_profile",
                format!(
                    "worker {} does not support payload schema {}",
                    job.worker, job.payload_schema
                ),
            ));
        }
        stage_inputs(
            job,
            input
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
            tmp.path(),
        )
        .map_err(|e| coded("invalid_job", format!("input: {e}")))?;
        write_atomic_json(&tmp.path().join("request.json"), job)?;
        // A signal during validation stops the validator too: it runs in its
        // own process group, which a terminal's SIGINT does not reach.
        install_stop_handlers();
        let (status, out, err) = bounded_command_until(
            &[
                executable.to_string_lossy().into_owned(),
                "validate".into(),
                tmp.path()
                    .join("request.json")
                    .to_string_lossy()
                    .into_owned(),
            ],
            vec![],
            Duration::from_secs(30),
            4 * 1024 * 1024,
            &|| SIGNALS.load(Ordering::Relaxed) > 0,
        )
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::TimedOut => {
                coded("invalid_job", "worker validation took longer than 30 s")
            }
            _ if e.to_string().contains("exceeds configured cap") => coded(
                "invalid_job",
                "the resolved job exceeds the 4 MiB validation output; keep inline starts or points well below 4 MiB",
            ),
            _ => e.into(),
        })?;
        if !status.success() {
            return Err(coded(
                "invalid_job",
                format!(
                    "worker validation failed: {}",
                    String::from_utf8_lossy(&err)
                ),
            ));
        }
        let answer: Value = serde_json::from_slice(&out)?;
        let resolved: JobSpec = serde_json::from_value(answer["resolved"].clone())?;
        resolved.validate()?;
        let mut envelope = resolved.clone();
        envelope.payload = job.payload.clone();
        envelope.task_count = job.task_count;
        if envelope != *job || resolved.task_count.is_none() {
            return Err("worker changed shared envelope or omitted task count".into());
        }
        *job = resolved;
    }
    Ok((s, w, bytes, tmp))
}
/// True when neither the launcher (`job.lock`) nor the worker (`worker.lock`)
/// of an attempt is alive. Both hold their lock for their whole lifetime and
/// the native worker starts no processes of its own. Errors count as alive.
fn attempt_processes_gone(dir: &Path) -> bool {
    let locks = matches!(lock_is_free(&dir.join("job.lock")), Ok(true))
        && matches!(lock_is_free(&dir.join("worker.lock")), Ok(true));
    if !locks {
        return false;
    }
    // Nothing of an attempt survives a reboot: its processes and transient
    // unit ended with the boot that recorded them, and a recorded process
    // group ID may since belong to an unrelated process.
    let current = boot().ok();
    let earlier_boot = |v: &Value| current.is_some() && v["boot_id"].as_str() != current.as_deref();
    if let Ok(p) = read(&dir.join("worker-process.json")) {
        return earlier_boot(&p)
            || p["pgid"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .is_some_and(|pid| group_stopped(pid).unwrap_or(false));
    }
    read(&dir.join("job.json"))
        .is_ok_and(|s| terminal(&s["state"]) || never_started(&s) || earlier_boot(&s))
}
/// Without a recorded process group, a worker can exist only if the attempt
/// reached `running`, which is recorded before the worker is spawned. A
/// queued attempt (or a foreground one not yet launched) whose launcher is
/// gone therefore has no worker. Detached `submitted`/`starting` attempts may
/// still be started by their unit, so they do not count.
fn never_started(job: &Value) -> bool {
    job["state"] == "waiting"
        || (matches!(job["state"].as_str(), Some("submitted" | "starting"))
            && job["unit"].is_null())
}
/// True when an attempt's launcher died before admission: it never started a
/// worker, holds no lock, has no process record and no slot reservation. Such
/// an attempt did provably zero work and can be sealed with complete
/// (empty) accounting.
fn never_admitted(root: &Path, id: &str, dir: &Path) -> Result<bool> {
    if dir.join("worker-process.json").exists()
        || !never_started(&read(&dir.join("job.json"))?)
        || !matches!(lock_is_free(&dir.join("job.lock")), Ok(true))
        || !matches!(lock_is_free(&dir.join("worker.lock")), Ok(true))
    {
        return Ok(false);
    }
    let _g = lock(&root.join("slots.lock"))?;
    Ok(optional(&root.join("slots.json"))?[id].is_null())
}
fn capacity(root: &Path) -> Result<u64> {
    let path = root.join("node.json");
    if !path.exists() {
        return Err(coded(
            "node_not_initialized",
            "node init --capacity required",
        ));
    }
    read(&path)?["capacity"]
        .as_u64()
        .ok_or_else(|| "node.json has no valid capacity".into())
}
fn worker(root: &Path, id: &str) -> Result<Value> {
    let w = optional(&root.join("workers.json"))?[id].clone();
    if w.is_null() {
        return Err(coded(
            "unknown_worker",
            format!("register worker {id} first"),
        ));
    }
    Ok(w)
}
fn directory(root: &Path, id: &str) -> Result<PathBuf> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(coded("unknown_job", format!("invalid job id {id:?}")));
    }
    let entry = optional(&root.join("requests.json"))?[id].clone();
    Ok(PathBuf::from(entry["directory"].as_str().ok_or_else(
        || coded("unknown_job", format!("no job {id} on this node")),
    )?))
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
        return Err(coded(
            "systemd_unavailable",
            "systemd user manager unavailable; use foreground run",
        ));
    }
    Ok(())
}
fn prepare(
    root: &Path,
    input: &Path,
    dest: Option<PathBuf>,
    detached: bool,
) -> Result<(String, bool)> {
    let (mut resolved, w, bytes, tmp) = staged(root, input)?;
    let node = capacity(root)?;
    if resolved.slots() as u64 > node {
        return Err(coded(
            "capacity_exceeded",
            format!(
                "the job requests {} CPU slots but the node capacity is {node}; it can never be admitted here",
                resolved.slots()
            ),
        ));
    }
    let executable = Path::new(
        w["executable"]
            .as_str()
            .ok_or("invalid worker registration")?,
    );
    let hash = digest(&bytes);
    let id = digest(resolved.request_key().as_bytes())[..32].to_owned();
    let identity = digest(&serde_json::to_vec(
        &json!({"spec":resolved,"worker_digest":hash}),
    )?);
    let _guard = lock(&root.join("registry.lock"))?;
    let mut requests = optional(&root.join("requests.json"))?;
    if !requests[&id].is_null() {
        if requests[&id]["identity"] != identity {
            return Err(coded(
                "request_key_conflict",
                format!(
                    "request_key conflict: {:?} already names job {id} with a different normalized spec or worker; use a new key",
                    resolved.request_key()
                ),
            ));
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
    let parent = dir.parent().ok_or("output has no parent")?;
    fs::create_dir_all(parent)?;
    fs::create_dir(&dir)?;
    // requests.json will point here; make the directory entry durable first.
    fs::File::open(parent)?.sync_all()?;
    let dir = fs::canonicalize(dir)?;
    if let Request::Job(ref mut job) = resolved {
        stage_inputs(job, tmp.path(), &dir)?;
    }
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
        Some(match resolved {
            Request::Legacy(_) => format!("fcmaes-{id}.service"),
            Request::Job(_) => format!("optimization-job-{id}.service"),
        })
    } else {
        None
    };
    write_atomic_json(
        &dir.join("provenance.json"),
        &json!({"worker_digest":hash,"registered_executable":executable,"identity":identity,"build":w["description"]["build"],"unit":unit,"boot_id":boot()?,"attempt_id":id}),
    )?;
    state_write(
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
    state_write(&dir.join("job.json"), &state)?;
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
    match bounded_command(&args, vec![], Duration::from_secs(15), 65536) {
        Ok((s, _, _)) if s.success() => Ok(()),
        _ => Err(coded(
            "start_unknown",
            format!(
                "start acknowledgement unknown for {unit}; inspect recorded unit; repeating submit will NOT launch again"
            ),
        )),
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
fn finish(dir: &Path, spec: &Request, status: &str, exact: bool, elapsed: f64) -> Result<()> {
    let p = read(&dir.join("provenance.json"))?;
    let worker_digest = p["worker_digest"].as_str().ok_or("missing worker digest")?;
    // A journal that cannot be replayed (for example a worker that wrote an
    // invalid record) still ends the attempt: `failed`, with no trusted
    // results or work totals, instead of a job left `running` forever.
    let (status, mut run) = match result::manifest(dir, spec, status, exact, elapsed, worker_digest)
    {
        Ok(run) => (status, run),
        Err(e) => (
            "failed",
            result::unreadable_manifest(spec, elapsed, worker_digest, &e.to_string()),
        ),
    };
    run["runner"]["attempt_id"] = p["attempt_id"].clone();
    run["runner"]["build"] = p["build"].clone();
    write_atomic_json(&dir.join("run.json"), &run)?;
    let mut state = read(&dir.join("job.json"))?;
    state["state"] = json!(status);
    state["finished_at"] = json!(now());
    state_write(&dir.join("job.json"), &state)?;
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
    install_stop_handlers();
    state["state"] = json!("waiting");
    state_write(&dir.join("job.json"), &state)?;
    let queue = Instant::now();
    loop {
        if stop_request(&dir) != Stop::None {
            finish(&dir, &s, "cancelled", true, 0.0)?;
            return Ok(());
        }
        if admit(root, id, &dir, s.slots())? {
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }
    let start = Instant::now();
    state["state"] = json!("running");
    state["started_at"] = json!(now());
    state["queue_seconds"] = json!(queue.elapsed().as_secs_f64());
    if let Err(e) = state_write(&dir.join("job.json"), &state) {
        release(root, id)?;
        return Err(e);
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
    s: &Request,
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
    if let Err(e) = write_atomic_json(
        &dir.join("worker-process.json"),
        &json!({"pgid":pgid,"boot_id":boot()?}),
    ) {
        terminate_process_group(&mut child);
        let _ = child.wait();
        *safely_stopped = group_stopped(pgid).unwrap_or(false);
        return Err(e.into());
    }
    let a = child.stdout.take().unwrap();
    let b = child.stderr.take().unwrap();
    let ta = thread::spawn(move || drain_file(a, stdout, 1_048_576));
    let tb = thread::spawn(move || drain_file(b, stderr, 1_048_576));
    // The worker checks `cancel.request` between retry waves. A graceful stop
    // lets the active wave finish; the wall-time deadline plus the declared
    // grace, or an immediate stop, kills the worker group.
    let wall = Duration::from_secs_f64(s.wall());
    let hard_deadline = start + wall + Duration::from_secs_f64(s.grace());
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
fn worker_outcome(dir: &Path, s: &Request) -> Result<Option<(String, f64)>> {
    result::worker_outcome(dir, s)
}
fn get(root: &Path, id: &str) -> Result<Value> {
    let dir = directory(root, id)?;
    let mut v = read(&dir.join("job.json"))?;
    v["directory"] = json!(dir);
    let s = spec(&dir.join("resolved.json"))?;
    // A corrupt journal is an error here; `jobs list` reports it per job.
    for (k, value) in result::progress(&dir, &s)?
        .as_object()
        .ok_or("bad progress")?
    {
        v[k] = value.clone();
    }
    if !terminal(&v["state"]) && v["boot_id"].as_str() != Some(&boot()?) {
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
    // Usage errors are coded like every other error; help and version print
    // normally.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp
                    | clap::error::ErrorKind::DisplayVersion
                    | clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
            ) =>
        {
            e.exit()
        }
        Err(e) => return Err(coded("invalid_command", e.render().to_string())),
    };
    let selected = cli
        .root
        .or_else(|| std::env::var_os("FCMAES_RUN_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| {
            let legacy = std::env::current_exe()
                .ok()
                .is_some_and(|p| p.file_stem().is_some_and(|s| s == "fcmaes-run"));
            PathBuf::from(if legacy {
                ".fcmaes-jobs"
            } else {
                ".optimization-jobs"
            })
        });
    fs::create_dir_all(&selected)?;
    let root = fs::canonicalize(&selected)?;
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
            if !safe_name(&id) {
                return Err("invalid worker id".into());
            }
            let executable = fs::canonicalize(executable)?;
            let registered_hash = digest(&fs::read(&executable)?);
            let (status, out, _) = bounded_command(
                &[executable.to_string_lossy().into_owned(), "describe".into()],
                vec![],
                Duration::from_secs(10),
                65536,
            )?;
            let description: Value = serde_json::from_slice(&out)?;
            let generic = protocol_supported(&description["protocol"])
                && description["profiles"]
                    .as_array()
                    .is_some_and(|p| !p.is_empty() && p.iter().all(Value::is_string));
            let compatible = id == "native"
                && ["protocol", "problems", "optimizers", "parallelism"]
                    .iter()
                    .all(|k| description[*k] == discovery()[*k]);
            if !status.success()
                || (!generic && !compatible)
                || registered_hash != digest(&fs::read(&executable)?)
            {
                return Err("worker capability profile mismatch or executable changed".into());
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
        } => {
            if std::env::current_exe()?
                .file_stem()
                .is_some_and(|name| name == "fcmaes-run")
            {
                output(discovery())
            } else {
                output(
                    json!({"legacy":discovery(),"registered":optional(&root.join("workers.json"))?}),
                )
            }
        }
        Cmd::Problems {
            command: Problems::Describe { id },
        } => {
            let registry = optional(&root.join("workers.json"))?;
            if !registry[&id].is_null() {
                output(registry[&id]["description"].clone())
            } else {
                let d = discovery();
                output(
                    d["problems"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|p| p["id"] == id)
                        .ok_or("unknown worker/problem")?
                        .clone(),
                )
            }
        }
        Cmd::Validate { spec: p } => match spec(&p).map_err(invalid)? {
            Request::Legacy(s) => output(
                json!({"valid":true,"requested_evaluations":s.requested()?,"maximum_actual_evaluations":s.maximum_evaluations()?,"resolved":s}),
            ),
            Request::Job(_) => {
                let (s, _, _, _) = staged(&root, &p)?;
                output(json!({"valid":true,"resolved":s}))
            }
        },
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
            Jobs::Events { id } => {
                let dir = directory(&root, &id)?;
                output(json!(read_checked::<Value>(&dir.join("events.jsonl"), 1)?))
            }
            Jobs::List => {
                let requests = optional(&root.join("requests.json"))?;
                let mut jobs = vec![];
                for id in requests.as_object().ok_or("bad requests")?.keys() {
                    jobs.push(
                        get(&root, id).unwrap_or_else(|e| json!({"id":id,"error":e.to_string()})),
                    );
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
                // Like a 409: a finished job has nothing to stop, and a late
                // request must not look as if it had stopped anything.
                let state = read(&dir.join("job.json"))?["state"].clone();
                if terminal(&state) {
                    return Err(coded(
                        "already_terminal",
                        format!(
                            "job {id} already ended as {}; nothing to cancel",
                            state.as_str().unwrap_or_default()
                        ),
                    ));
                }
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
                if terminal(&read(&dir.join("job.json"))?["state"]) {
                    return Err(coded(
                        "already_terminal",
                        "attempt already finalized; repair is for interrupted attempts",
                    ));
                }
                // Decided before the reservation is touched below.
                let unadmitted = never_admitted(&root, &id, &dir)?;
                if !attempt_processes_gone(&dir) {
                    // A held lock is a live launcher or worker: waiting helps.
                    // Without one, liveness stays unknown (a process group
                    // that still exists, or a detached unit that may yet
                    // start); waiting alone may never settle it.
                    if matches!(lock_is_free(&dir.join("worker.lock")), Ok(false)) {
                        return Err(coded("attempt_active", "worker still owns this attempt"));
                    }
                    if matches!(lock_is_free(&dir.join("job.lock")), Ok(false)) {
                        return Err(coded("attempt_active", "launcher still owns this attempt"));
                    }
                    return Err(coded(
                        "liveness_unknown",
                        "no process holds this attempt, but its process group still exists or its detached unit may still start; inspect it (jobs get, systemctl --user) before repairing",
                    ));
                }
                let owner = fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join("job.lock"))?;
                fs2::FileExt::try_lock_exclusive(&owner)
                    .map_err(|_| coded("attempt_active", "launcher still owns this attempt"))?;
                let worker = fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join("worker.lock"))?;
                fs2::FileExt::try_lock_exclusive(&worker)
                    .map_err(|_| coded("attempt_active", "worker still owns this attempt"))?;
                let state = read(&dir.join("job.json"))?;
                if matches!(
                    state["state"].as_str(),
                    Some("completed" | "failed" | "cancelled" | "timed_out" | "interrupted")
                ) {
                    return Err(coded(
                        "already_terminal",
                        "attempt already finalized; repair is for interrupted attempts",
                    ));
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
                // A journal corrupt before its final line is not repaired; no
                // record of it is trusted and `finish` seals the attempt failed.
                let replayable = result::repair(&dir, &s).is_ok();
                // An orphaned worker may have finished on its own after the
                // launcher died; keep its exact result.
                if unadmitted {
                    // No worker ever ran: the attempt did provably zero work.
                    finish(&dir, &s, "interrupted", true, 0.0)?;
                } else if let Some((status, elapsed)) = replayable
                    .then(|| worker_outcome(&dir, &s))
                    .transpose()?
                    .flatten()
                {
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
pub fn main() {
    if !cfg!(target_os = "linux") {
        eprintln!("Linux is required for process supervision");
        std::process::exit(2);
    }
    if let Err(e) = main_result() {
        let code = error_code(&e);
        eprintln!(
            "{}",
            json!({"error":{"code":code,"message":e.to_string(),"retryable":retryable(code)}})
        );
        std::process::exit(2);
    }
}
