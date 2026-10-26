#![forbid(unsafe_code)]
use cap_search::{
    algorithms::{self, Method},
    bridge::{Bridge, Failure, Result},
};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    about = "Bounded floating-point proof-parameter search; certification is a separate command"
)]
struct Args {
    #[arg(long)]
    julia: PathBuf,
    #[arg(long)]
    project: PathBuf,
    #[arg(long)]
    server: PathBuf,
    #[arg(long)]
    problem: PathBuf,
    #[arg(long)]
    protocol: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 300.0)]
    wall_seconds: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema_version: String,
    evaluation_budget: usize,
    arms: Vec<Arm>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    id: String,
    method: Method,
    seed: u64,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if fs::metadata(path).map_err(Failure::io)?.len() > limit {
        return Err(Failure::new("invalid_input", "input exceeds byte limit"));
    }
    fs::read(path).map_err(Failure::io)
}
fn create(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(Failure::io)?;
    file.write_all(bytes).map_err(Failure::io)
}
fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value).map_err(Failure::io)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct Ledger {
    file: File,
    search_attempted: usize,
    search_completed: usize,
    rechecks_attempted: usize,
    rechecks_completed: usize,
}
impl Ledger {
    fn event(&mut self, value: Value) -> Result<()> {
        self.file
            .write_all(&json_bytes(&value)?)
            .map_err(Failure::io)?;
        self.file.flush().map_err(Failure::io)
    }
    fn evaluate(&mut self, bridge: &mut Bridge, arm: &Arm, phase: &str, x: &[f64]) -> Result<f64> {
        // An attempt is a request Julia was sent: a cancellation or deadline
        // that stops the search before sending leaves no attempt behind.
        let started = Instant::now();
        let id = bridge.send("evaluate", Some(x))?;
        if phase == "search" {
            self.search_attempted += 1;
        } else {
            self.rechecks_attempted += 1;
        }
        self.event(
            json!({"schema_version":"cap.search.event/0.1","event":"evaluation_started",
            "arm":arm.id,"phase":phase,"parameters":x}),
        )?;
        let response = bridge.receive(id, "estimated")?;
        let estimate = response["estimate"]
            .as_object()
            .ok_or_else(|| Failure::new("protocol_error", "missing estimate object"))?;
        let expected = [
            "score",
            "estimated_Y",
            "estimated_Z1",
            "estimated_Z2",
            "estimated_polynomial_normalized",
            "estimated_contraction_minus_one",
            "physical_epsilon",
        ];
        if estimate.len() != expected.len()
            || expected.iter().any(|key| {
                estimate
                    .get(*key)
                    .and_then(Value::as_f64)
                    .is_none_or(|x| !x.is_finite())
            })
        {
            return Err(Failure::new(
                "protocol_error",
                "incomplete or nonfinite estimate",
            ));
        }
        let score = response["estimate"]["score"]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| Failure::new("protocol_error", "missing or nonfinite score"))?;
        if response["parameters"] != json!(x) {
            return Err(Failure::new("protocol_error", "point identity mismatch"));
        }
        if phase == "search" {
            self.search_completed += 1;
        } else {
            self.rechecks_completed += 1;
        }
        self.event(
            json!({"schema_version":"cap.search.event/0.1","event":"evaluation_finished",
            "arm":arm.id,"phase":phase,"parameters":x,"estimate":response["estimate"],
            "wall_seconds":started.elapsed().as_secs_f64()}),
        )?;
        Ok(score)
    }
    fn counts(&self, warmup: usize) -> Value {
        json!({"warmup_evaluations":warmup,"search_attempted":self.search_attempted,
            "startup_accounting_complete":warmup==1,
            "search_completed":self.search_completed,"rechecks_attempted":self.rechecks_attempted,
            "rechecks_completed":self.rechecks_completed,
            "total_attempted_including_warmup":warmup+self.search_attempted+self.rechecks_attempted})
    }
}

fn study(args: &Args, cancelled: Arc<AtomicBool>) -> Result<()> {
    let started = Instant::now();
    if !args.wall_seconds.is_finite() || !(0.05..=3600.0).contains(&args.wall_seconds) {
        return Err(Failure::new(
            "invalid_input",
            "wall seconds must be finite in 0.05..3600",
        ));
    }
    let protocol_bytes = read(&args.protocol, 65536)?;
    let protocol: Protocol =
        serde_json::from_slice(&protocol_bytes).map_err(|e| Failure::new("invalid_protocol", e))?;
    let problem_bytes = read(&args.problem, 1 << 20)?;
    let problem: Value = serde_json::from_slice(&problem_bytes).map_err(Failure::io)?;
    let dim = match problem["id"].as_str() {
        Some("scalar-cuberoot") => 1,
        Some("cahn-hilliard") => 2,
        _ => {
            return Err(Failure::new(
                "invalid_input",
                "unsupported reference problem",
            ));
        }
    };
    if protocol.schema_version != "cap.search.protocol/0.1"
        || !(9..=256).contains(&protocol.evaluation_budget)
        || !(6..=16).contains(&protocol.arms.len())
    {
        return Err(Failure::new(
            "invalid_protocol",
            "expected schema 0.1, budget 9..256 and 6..16 arms",
        ));
    }
    let mut ids = std::collections::BTreeSet::new();
    for arm in &protocol.arms {
        if arm.id.is_empty()
            || arm.id.len() > 64
            || !arm
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            || !ids.insert(&arm.id)
        {
            return Err(Failure::new(
                "invalid_protocol",
                "invalid or repeated arm ID",
            ));
        }
    }
    for method in [
        Method::Reference,
        Method::Grid,
        Method::Random,
        Method::Local,
        Method::Cmaes,
        Method::Biteopt,
    ] {
        if !protocol.arms.iter().any(|a| a.method == method) {
            return Err(Failure::new(
                "invalid_protocol",
                "reference, grid, random, local, cmaes and biteopt are mandatory",
            ));
        }
    }
    let levels = if dim == 1 {
        protocol.evaluation_budget
    } else {
        (protocol.evaluation_budget as f64).sqrt() as usize
    };
    if levels.pow(dim as u32) != protocol.evaluation_budget || levels % 2 == 0 {
        return Err(Failure::new(
            "invalid_protocol",
            "budget must give a complete odd grid for this dimension",
        ));
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).map_err(Failure::io)?;
    }
    match fs::create_dir(&args.output) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && args.output.is_dir() => {
            if fs::read_dir(&args.output)
                .map_err(Failure::io)?
                .next()
                .is_some()
            {
                return Err(Failure::new(
                    "invalid_input",
                    "output directory is not empty",
                ));
            }
        }
        Err(error) => return Err(Failure::io(error)),
    }
    let mut ledger = Ledger {
        file: OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(args.output.join("evaluations.jsonl"))
            .map_err(Failure::io)?,
        search_attempted: 0,
        search_completed: 0,
        rechecks_attempted: 0,
        rechecks_completed: 0,
    };
    let mut record = json!({"schema_version":"cap.search.study/0.1","outcome":"running",
        "certification_performed":false,"problem_id":problem["id"],"problem_sha256":digest(&problem_bytes),
        "protocol_sha256":digest(&protocol_bytes),"protocol":serde_json::from_slice::<Value>(&protocol_bytes).map_err(Failure::io)?,
        "search_binary_sha256":digest(&fs::read(std::env::current_exe().map_err(Failure::io)?).map_err(Failure::io)?),
        "fcmaes_core_version":fcmaes_core::CORE_VERSION,"parameterization":"finite-inverse-row-scales/1",
        "parameter_bounds":[0.5,1.5],"nominal_parameters":vec![1.0;dim],
        "method_settings":{"cmaes":{"population":4,"sigma":0.2},"biteopt":{"population":8,"batch":1,"start":"nominal point, told its known score"},
            "local":{"initial_step":0.25,"rule":"coordinate pattern; halve after a non-improving cycle"},
            "grid":{"levels_per_axis":levels,"complete_cartesian":true}},
        "estimated_score_definition":"max(P(epsilon)/epsilon,Z1+Z2*epsilon-1); floating estimates only",
        "scheduling":"sequential_one_julia_compute_thread_one_blas_thread","arms":[],"selection":null});
    let mut arms = Vec::new();
    let mut warmup = 0;
    let mut chosen: Option<(String, Vec<f64>, f64, Vec<u8>)> = None;
    let mut candidates = serde_json::Map::new();
    let mut command = Command::new(&args.julia);
    command
        .args(["--startup-file=no", "--history-file=no"])
        .arg(format!("--project={}", args.project.display()))
        .arg(&args.server)
        .arg(&args.problem)
        .env("JULIA_NUM_THREADS", "1")
        .env("JULIA_LOAD_PATH", "@:@stdlib")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1");
    let outcome = (|| -> Result<()> {
        let mut bridge = Bridge::spawn(
            command,
            started + Duration::from_secs_f64(args.wall_seconds),
            cancelled,
        )?;
        record["server_pid"] = json!(bridge.pid());
        ledger.event(json!({"schema_version":"cap.search.event/0.1","event":"server_started","pid":bridge.pid()}))?;
        let ready = bridge.ready()?;
        if ready["dimension"].as_u64() != Some(dim as u64)
            || ready["problem_sha256"] != record["problem_sha256"]
            || ready["problem_id"] != problem["id"]
        {
            return Err(Failure::new(
                "protocol_error",
                "server reference identity mismatch",
            ));
        }
        if ready["server_sha256"] != digest(&read(&args.server, 1 << 20)?)
            || ready["search_module_sha256"]
                != digest(&read(&args.project.join("src/CAPSearch.jl"), 1 << 20)?)
            || ready["environment"]["verifier_sha256"]
                != digest(&read(&args.project.join("src/CAPReference.jl"), 1 << 20)?)
        {
            return Err(Failure::new(
                "protocol_error",
                "server source identity mismatch",
            ));
        }
        warmup = 1;
        record["server"] = ready;
        record["startup_wall_seconds"] = json!(started.elapsed().as_secs_f64());
        ledger.event(json!({"schema_version":"cap.search.event/0.1","event":"server_ready","warmup_evaluations":1}))?;
        for arm in &protocol.arms {
            let arm_started = Instant::now();
            let best =
                algorithms::run(arm.method, dim, protocol.evaluation_budget, arm.seed, |x| {
                    ledger.evaluate(&mut bridge, arm, "search", x)
                })?;
            let rechecked = ledger.evaluate(&mut bridge, arm, "recheck", &best.parameters)?;
            if rechecked.to_bits() != best.score.to_bits() {
                return Err(Failure::new(
                    "objective_mismatch",
                    "same point did not reproduce its floating score",
                ));
            }
            let exported = bridge.request("candidate", Some(&best.parameters), "candidate")?;
            if exported["parameters"] != json!(best.parameters)
                || exported["candidate"]["problem_sha256"] != record["problem_sha256"]
            {
                return Err(Failure::new(
                    "protocol_error",
                    "exported candidate identity mismatch",
                ));
            }
            let bytes = json_bytes(&exported["candidate"])?;
            let hash = digest(&bytes);
            candidates.insert(arm.id.clone(),json!({"candidate_sha256":hash,"candidate_json":String::from_utf8(bytes.clone()).map_err(Failure::io)?}));
            if chosen
                .as_ref()
                .is_none_or(|(_, _, score, _)| best.score < *score)
            {
                chosen = Some((arm.id.clone(), best.parameters.clone(), best.score, bytes));
            }
            arms.push(
                json!({"id":arm.id,"method":arm.method,"seed":arm.seed,"best":best,
                "recheck_evaluations":1,"rechecked_score":rechecked,"candidate_sha256":hash,
                "warm_wall_seconds":arm_started.elapsed().as_secs_f64()}),
            );
        }
        bridge.finish()?;
        let (id, parameters, score, bytes) = chosen.as_ref().expect("mandatory arms");
        create(&args.output.join("candidate.json"), bytes)?;
        create(
            &args.output.join("candidates.json"),
            &json_bytes(&json!({"schema_version":"cap.search.candidates/0.1","arms":candidates}))?,
        )?;
        record["selection"] = json!({"arm":id,"parameters":parameters,"estimated_score":score,"candidate_sha256":digest(bytes)});
        Ok(())
    })();
    record["arms"] = json!(arms);
    record["counts"] = ledger.counts(warmup);
    record["total_wall_seconds"] = json!(started.elapsed().as_secs_f64());
    record["outcome"] = json!(outcome.as_ref().map_or_else(|e| e.code, |_| "completed"));
    if let Err(error) = &outcome {
        record["error"] = json!({"code":error.code,"message":error.message});
    }
    ledger.event(json!({"schema_version":"cap.search.event/0.1","event":"study_finished","outcome":record["outcome"],"counts":record["counts"]}))?;
    record["evaluation_log_sha256"] = json!(digest(
        &fs::read(args.output.join("evaluations.jsonl")).map_err(Failure::io)?
    ));
    create(&args.output.join("study.json"), &json_bytes(&record)?)?;
    outcome
}

fn main() {
    let args = Args::parse();
    let cancelled = Arc::new(AtomicBool::new(false));
    let token = cancelled.clone();
    let outcome = ctrlc::set_handler(move || token.store(true, Ordering::Release))
        .map_err(Failure::io)
        .and_then(|_| study(&args, cancelled));
    match outcome {
        Ok(()) => println!(
            "{}",
            json!({"outcome":"completed","certification_performed":false,"output":args.output})
        ),
        Err(error) => {
            eprintln!(
                "{}",
                json!({"error":{"code":error.code,"message":error.message}})
            );
            std::process::exit(match error.code {
                "timed_out" => 124,
                "cancelled" => 130,
                _ => 2,
            });
        }
    }
}
