use cap_search::{
    algorithms::{self, Method},
    bridge::{Bridge, FRAME_LIMIT},
};
use std::{
    process::Command,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

const METHODS: [Method; 6] = [
    Method::Reference,
    Method::Grid,
    Method::Random,
    Method::Local,
    Method::Cmaes,
    Method::Biteopt,
];

#[test]
fn all_methods_obey_the_work_cap_and_bounds() {
    for dim in [1, 2] {
        for budget in [9, 25] {
            for method in METHODS {
                let mut calls = 0;
                let best = algorithms::run(method, dim, budget, 17, |x| {
                    if calls == 0 {
                        assert_eq!(x, vec![1.0; dim]);
                    }
                    assert!(x.iter().all(|x| (0.5..=1.5).contains(x)));
                    calls += 1;
                    Ok(x.iter().map(|x| (x - 0.75).powi(2)).sum())
                })
                .unwrap();
                assert_eq!(
                    calls,
                    if method == Method::Reference {
                        1
                    } else {
                        budget
                    }
                );
                assert_eq!(calls, best.evaluations);
            }
        }
    }
}

#[test]
fn seeded_methods_reproduce_the_whole_trajectory() {
    for dim in [1, 2] {
        for method in [Method::Random, Method::Cmaes, Method::Biteopt] {
            let trace = |seed| {
                let mut points = Vec::new();
                algorithms::run(method, dim, 25, seed, |x| {
                    points.push(x.to_vec());
                    Ok(x.iter().map(|x| (x - 0.8).powi(2)).sum())
                })
                .unwrap();
                points
            };
            assert_eq!(trace(17), trace(17));
            assert_ne!(trace(17), trace(29));
        }
    }
}

#[test]
fn grid_is_complete_and_has_no_duplicated_nominal_point() {
    let mut points = Vec::new();
    let best = algorithms::run(Method::Grid, 2, 25, 0, |x| {
        points.push(x.to_vec());
        Ok((x[0] - 0.75).powi(2) + (x[1] - 1.25).powi(2))
    })
    .unwrap();
    assert_eq!(best.score, 0.0);
    for a in [0.5, 0.75, 1.0, 1.25, 1.5] {
        for b in [0.5, 0.75, 1.0, 1.25, 1.5] {
            assert_eq!(points.iter().filter(|p| **p == vec![a, b]).count(), 1);
        }
    }
}

#[test]
fn local_pattern_moves_only_to_an_improving_centre() {
    let best = algorithms::run(Method::Local, 2, 25, 0, |x| {
        Ok((x[0] - 0.75).powi(2) + (x[1] - 1.25).powi(2))
    })
    .unwrap();
    assert_eq!(best.parameters, vec![0.75, 1.25]);
    assert_eq!(best.score, 0.0);
}

#[test]
fn bad_budget_and_nonfinite_scores_are_explicit_errors() {
    assert!(algorithms::run(Method::Grid, 2, 10, 0, |_| Ok(0.0)).is_err());
    assert!(algorithms::run(Method::Random, 0, 25, 0, |_| Ok(0.0)).is_err());
    assert_eq!(
        algorithms::run(Method::Random, 1, 25, 0, |_| Ok(f64::NAN))
            .unwrap_err()
            .code,
        "protocol_error"
    );
}

fn bridge(code: &str, timeout: Duration) -> Bridge {
    let mut command = Command::new("python3");
    command.args(["-u", "-c", code]);
    Bridge::spawn(
        command,
        Instant::now() + timeout,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap()
}

fn reaped(pid: u32) {
    // /proc can be mounted from an outer PID namespace. waitpid speaks the
    // caller's namespace and proves the owned child has already been reaped.
    assert_eq!(
        nix::sys::wait::waitpid(
            nix::unistd::Pid::from_raw(pid as i32),
            Some(nix::sys::wait::WaitPidFlag::WNOHANG)
        ),
        Err(nix::errno::Errno::ECHILD)
    );
}

const READY: &str = "import json,sys\nprint(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=0,status='ready')),flush=True)\n";

#[test]
fn a_silent_child_times_out_and_is_reaped() {
    let b = bridge("import time;time.sleep(20)", Duration::from_millis(100));
    let pid = b.pid();
    assert_eq!(b.ready().unwrap_err().code, "timed_out");
    drop(b);
    reaped(pid);
}

#[test]
fn cancellation_reaps_a_live_child() {
    let token = Arc::new(AtomicBool::new(true));
    let mut command = Command::new("python3");
    command.args(["-c", "import time;time.sleep(20)"]);
    let b = Bridge::spawn(command, Instant::now() + Duration::from_secs(5), token).unwrap();
    let pid = b.pid();
    assert_eq!(b.ready().unwrap_err().code, "cancelled");
    drop(b);
    reaped(pid);
}

#[test]
fn oversized_and_truncated_frames_are_rejected() {
    for code in [
        format!(
            "import sys;sys.stdout.write('x'*{}+'\\n');sys.stdout.flush()",
            FRAME_LIMIT + 10
        ),
        "print('{broken')".to_string(),
        "import sys;sys.stdout.write('{}');sys.stdout.flush()".to_string(),
    ] {
        let b = bridge(&code, Duration::from_secs(5));
        let pid = b.pid();
        assert_eq!(b.ready().unwrap_err().code, "protocol_error");
        drop(b);
        reaped(pid);
    }
}

/// A server frame of `size` JSON bytes plus the newline terminator.
fn padded_ready(size: usize) -> String {
    format!(
        "import json,sys\nbase=json.dumps(dict(schema_version='cap.search.response/0.1',request_id=0,status='ready'))\nframe=base[:-1]+' '*({size}-len(base))+'}}'\nassert len(frame)=={size}\nsys.stdout.write(frame+'\\n');sys.stdout.flush()\nsys.stdin.read()\n"
    )
}

#[test]
fn the_frame_limit_counts_json_bytes_like_the_server() {
    // The Julia server accepts and emits FRAME_LIMIT JSON bytes plus a newline.
    let b = bridge(&padded_ready(FRAME_LIMIT), Duration::from_secs(5));
    let pid = b.pid();
    b.ready().unwrap();
    drop(b);
    reaped(pid);
    let b = bridge(&padded_ready(FRAME_LIMIT + 1), Duration::from_secs(5));
    let pid = b.pid();
    assert_eq!(b.ready().unwrap_err().code, "protocol_error");
    drop(b);
    reaped(pid);
}

#[test]
fn a_server_ended_by_a_group_signal_reports_the_cancellation() {
    // A Ctrl-C reaches the server too; it may exit before this process's
    // handler has recorded the cancellation.
    let token = Arc::new(AtomicBool::new(false));
    let mut command = Command::new("python3");
    command.args([
        "-u",
        "-c",
        &format!("{READY}sys.stdin.readline()\nimport os\nos._exit(1)\n"),
    ]);
    let mut b = Bridge::spawn(
        command,
        Instant::now() + Duration::from_secs(5),
        token.clone(),
    )
    .unwrap();
    let pid = b.pid();
    b.ready().unwrap();
    let signal = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(40));
        token.store(true, std::sync::atomic::Ordering::Release);
    });
    assert_eq!(
        b.request("evaluate", Some(&[1.0]), "estimated")
            .unwrap_err()
            .code,
        "cancelled"
    );
    signal.join().unwrap();
    drop(b);
    reaped(pid);
}

#[test]
fn cleanup_does_not_wait_for_descendants_holding_the_pipes() {
    // Like a Julia precompilation worker, the grandchild inherits stdout/stderr.
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("grandchild.pid");
    let code = format!(
        "import subprocess,time\np=subprocess.Popen(['sleep','30'])\nopen({file:?},'w').write(str(p.pid))\ntime.sleep(30)\n"
    );
    let b = bridge(&code, Duration::from_secs(2));
    let pid = b.pid();
    let waiting = Instant::now();
    while !file.exists() && waiting.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(b.ready().unwrap_err().code, "timed_out");
    let started = Instant::now();
    drop(b);
    let cleanup = started.elapsed();
    reaped(pid);
    if let Ok(text) = std::fs::read_to_string(&file) {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(text.trim().parse().unwrap()),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    assert!(cleanup < Duration::from_secs(5), "cleanup took {cleanup:?}");
}

#[test]
fn wrong_reply_identity_is_rejected() {
    let code = format!(
        "{READY}r=json.loads(sys.stdin.readline())\nprint(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=99,status='estimated')),flush=True)\n"
    );
    let mut b = bridge(&code, Duration::from_secs(5));
    let pid = b.pid();
    b.ready().unwrap();
    assert_eq!(
        b.request("evaluate", Some(&[1.0]), "estimated")
            .unwrap_err()
            .code,
        "protocol_error"
    );
    drop(b);
    reaped(pid);
}

#[test]
fn a_child_exit_is_observed_without_waiting_for_the_deadline() {
    let mut b = bridge(READY, Duration::from_secs(5));
    let pid = b.pid();
    b.ready().unwrap();
    assert!(b.request("evaluate", Some(&[1.0]), "estimated").is_err());
    drop(b);
    reaped(pid);
}

#[test]
fn graceful_shutdown_drains_stderr_and_reaps_child() {
    let code = format!(
        "{READY}sys.stderr.write('diagnostic\\n');sys.stderr.flush()\nr=json.loads(sys.stdin.readline())\nprint(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=r['request_id'],status='stopped')),flush=True)\n"
    );
    let mut b = bridge(&code, Duration::from_secs(5));
    let pid = b.pid();
    b.ready().unwrap();
    b.finish().unwrap();
    drop(b);
    reaped(pid);
}
