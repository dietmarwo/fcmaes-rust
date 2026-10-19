use optimization_job_api::{process::*, storage::*};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    time::{Duration, Instant},
};
#[test]
fn tail_repair_preserves_valid_records_and_rejects_checksum_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("journal");
    append_checked(&p, 1, &json!({"id":1})).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&p)
        .unwrap()
        .write_all(b"{truncated")
        .unwrap();
    assert_eq!(repair_tail::<Value>(&p, 1).unwrap().len(), 1);
    append_checked(&p, 1, &json!({"id":2})).unwrap();
    let data = fs::read(&p).unwrap();
    fs::write(&p, &data[..data.len() - 1]).unwrap();
    assert_eq!(repair_tail::<Value>(&p, 1).unwrap().len(), 2);
    assert!(fs::read(&p).unwrap().ends_with(b"\n"));
    let text = fs::read_to_string(&p)
        .unwrap()
        .replace("\\\"id\\\":1", "\\\"id\\\":9");
    fs::write(&p, text).unwrap();
    assert!(read_checked::<Value>(&p, 1).is_err());
}
#[test]
fn complete_corruption_is_not_truncation() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("j");
    fs::write(&p, b"{bad}\n").unwrap();
    assert!(read_checked::<Value>(&p, 1).is_err());
    append_checked(&d.path().join("ok"), 1, &json!(1)).unwrap();
    let mut e: Value = serde_json::from_slice(&fs::read(d.path().join("ok")).unwrap()).unwrap();
    e["checksum"] = json!("bad");
    fs::write(&p, serde_json::to_vec(&e).unwrap()).unwrap();
    assert!(repair_tail::<Value>(&p, 1).is_err());
}
#[test]
fn concurrent_completions_and_atomic_snapshots() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("j");
    let m = std::sync::Mutex::new(());
    std::thread::scope(|s| {
        for t in 0..8 {
            let p = &p;
            let m = &m;
            s.spawn(move || {
                for i in 0..20 {
                    let _g = m.lock().unwrap();
                    append_checked(p, 1, &json!([t, i])).unwrap();
                }
            });
        }
    });
    assert_eq!(read_checked::<Value>(&p, 1).unwrap().len(), 160);
    let p = d.path().join("snapshot");
    write_atomic_json(&p, &json!({"old":true})).unwrap();
    write_atomic_json(&p, &json!({"new":true})).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(p).unwrap()).unwrap(),
        json!({"new":true})
    );
}
#[test]
#[cfg(target_os = "linux")]
fn disk_write_error_propagates() {
    assert!(append_checked(std::path::Path::new("/dev/full"), 1, &json!(1)).is_err());
}
fn python(
    code: &str,
    timeout: Duration,
    cap: usize,
) -> std::io::Result<(std::process::ExitStatus, Vec<u8>, Vec<u8>)> {
    bounded_command(
        &["/usr/bin/python3".into(), "-c".into(), code.into()],
        vec![],
        timeout,
        cap,
    )
}
#[test]
fn argv_fidelity_and_stream_overflow() {
    let argv = vec![
        "/usr/bin/python3".into(),
        "-c".into(),
        "import sys; print(sys.argv[1])".into(),
        "a b;$(false)'\"".into(),
    ];
    let (_, out, _) = bounded_command(&argv, vec![], Duration::from_secs(2), 100).unwrap();
    assert_eq!(String::from_utf8(out).unwrap().trim(), argv[3]);
    for stream in ["stdout", "stderr"] {
        assert!(
            python(
                &format!("import sys;sys.{stream}.write('x'*200000)"),
                Duration::from_secs(2),
                1024
            )
            .is_err()
        );
    }
}
#[test]
#[cfg(target_os = "linux")]
fn deadline_kills_descendants_and_closes_inherited_pipes() {
    let d = tempfile::tempdir().unwrap();
    let marker = d.path().join("child_finished");
    let code = format!(
        "import subprocess,time;subprocess.Popen(['/usr/bin/python3','-c',\"import time,pathlib;time.sleep(0.5);pathlib.Path('{}').touch()\"]);time.sleep(20)",
        marker.display()
    );
    let t = Instant::now();
    assert!(python(&code, Duration::from_millis(100), 1024).is_err());
    assert!(t.elapsed() < Duration::from_secs(2));
    std::thread::sleep(Duration::from_millis(600));
    assert!(!marker.exists());
    // Parent success must also close pipes held by descendants.
    let t = Instant::now();
    assert!(
        python(
            "import subprocess;subprocess.Popen(['/usr/bin/sleep','20'])",
            Duration::from_secs(1),
            100
        )
        .unwrap()
        .0
        .success()
    );
    assert!(t.elapsed() < Duration::from_secs(2));
}
#[test]
#[cfg(target_os = "linux")]
fn process_group_liveness_uses_current_pid_namespace() {
    let mut cmd = std::process::Command::new("/usr/bin/sleep");
    cmd.arg("20");
    group(&mut cmd);
    let mut child = cmd.spawn().unwrap();
    assert!(!group_stopped(child.id()).unwrap());
    terminate_process_group(&mut child);
    child.wait().unwrap();
    assert!(group_stopped(child.id()).unwrap());
}
