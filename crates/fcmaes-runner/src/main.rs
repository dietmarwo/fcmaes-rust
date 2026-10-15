use clap::{Parser, Subcommand};
use fcmaes_job_api::{process::*, result, storage::*, *};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
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
    Cancel {
        id: String,
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
static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop_handler(_: i32) {
    STOP.store(true, Ordering::Relaxed);
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
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
    serde_json::from_slice::<JobSpec>(&fs::read(path)?)?.resolve()
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
fn reconcile_ledger(_root: &Path, ledger: &mut Value) -> Result<()> {
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
        if old_boot || stopped {
            ledger.as_object_mut().unwrap().remove(&id);
        }
    }
    Ok(())
}
fn admit(root: &Path, id: &str, dir: &Path, slots: usize) -> Result<bool> {
    if dir.join("cancel.request").exists() || STOP.load(Ordering::Relaxed) {
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
    unsafe {
        libc::signal(
            libc::SIGINT,
            stop_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            stop_handler as *const () as libc::sighandler_t,
        );
    }
    state["state"] = json!("waiting");
    write_atomic_json(&dir.join("job.json"), &state)?;
    let queue = Instant::now();
    loop {
        if dir.join("cancel.request").exists() || STOP.load(Ordering::Relaxed) {
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
    let mut reason = None;
    let mut stop_time = None;
    let polled = (|| -> Result<std::process::ExitStatus> {
        loop {
            if let Some(exit) = child.try_wait()? {
                return Ok(exit);
            }
            if reason.is_none() {
                if dir.join("cancel.request").exists() || STOP.load(Ordering::Relaxed) {
                    reason = Some("cancelled");
                } else if start.elapsed().as_secs_f64() >= s.budget.wall_time_seconds {
                    reason = Some("timed_out");
                }
                if reason.is_some() {
                    write_atomic_json(&dir.join("cancel.request"), &json!({"reason":reason}))?;
                    stop_time = Some(Instant::now());
                }
            }
            if stop_time.is_some_and(|t| t.elapsed() >= Duration::from_secs(1)) {
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
    let records = result::replay(dir, s)?;
    let normal = if exit.success() && dir.join("run.json").exists() {
        let r = read(&dir.join("run.json"))?;
        r["runner"]["spec_digest"] == s.digest()?
            && r["runner"]["accounting_complete"] == true
            && matches!(
                r["status"].as_str(),
                Some("completed" | "cancelled" | "timed_out")
            )
            && r["actual_evaluations"].as_u64()
                == Some(records.iter().map(|r| r.actual_evaluations).sum())
    } else {
        false
    };
    let status = if let Some(reason) = reason {
        reason.to_string()
    } else if normal {
        read(&dir.join("run.json"))?["status"]
            .as_str()
            .unwrap()
            .to_owned()
    } else {
        "failed".into()
    };
    if status == "completed" && records.len() != s.budget.retries {
        return Err("worker completed without all retries".into());
    }
    Ok((status, normal))
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
                    ) || t.elapsed().as_secs_f64() >= timeout
                    {
                        return output(v);
                    }
                    thread::sleep(Duration::from_millis(250));
                }
            }
            Jobs::Cancel { id } => {
                let dir = directory(&root, &id)?;
                write_atomic_json(&dir.join("cancel.request"), &json!({"requested_at":now()}))?;
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
                let _g = lock(&root.join("slots.lock"))?;
                let mut ledger = optional(&root.join("slots.json"))?;
                reconcile_ledger(&root, &mut ledger)?;
                write_atomic_json(&root.join("slots.json"), &ledger)?;
                if !ledger[&id].is_null() {
                    return Err("liveness unresolved; reservation retained, repair refused".into());
                }
                let state = read(&dir.join("job.json"))?;
                if matches!(
                    state["state"].as_str(),
                    Some("completed" | "failed" | "cancelled" | "timed_out")
                ) {
                    return Err(
                        "attempt already finalized; repair is for interrupted attempts".into(),
                    );
                }
                let s = spec(&dir.join("resolved.json"))?;
                result::replay(&dir, &s)?;
                if dir.join("retries.jsonl").exists() {
                    repair_tail::<result::RetryRecord>(&dir.join("retries.jsonl"), 1)?;
                }
                finish(&dir, &s, "interrupted", false, 0.0)?;
                // The execution duration is unknown after launcher loss.
                let mut r = read(&dir.join("run.json"))?;
                r["elapsed_seconds"] = Value::Null;
                write_atomic_json(&dir.join("run.json"), &r)?;
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
