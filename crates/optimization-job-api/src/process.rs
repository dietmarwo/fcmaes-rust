//! Shell-free process groups and bounded I/O, extracted/adapted from route_agent.
use std::{
    io::{self, Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};
pub fn group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}
pub fn terminate_process_group(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(child.id()) {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
}
/// Keep draining beyond the limit to avoid pipe deadlocks; never retain excess bytes.
pub fn drain(mut reader: impl Read, maximum: usize) -> io::Result<(Vec<u8>, bool)> {
    let mut out = Vec::new();
    let mut overflow = false;
    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let keep = n.min(maximum.saturating_sub(out.len()));
        out.extend_from_slice(&buf[..keep]);
        overflow |= keep < n;
    }
    Ok((out, overflow))
}
pub fn bounded_command(
    argv: &[String],
    input: Vec<u8>,
    timeout: Duration,
    maximum: usize,
) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    bounded_command_until(argv, input, timeout, maximum, &|| false)
}
/// [`bounded_command`] that also stops the command's process group, with an
/// `Interrupted` error, as soon as `abort` returns true. The command runs in
/// its own group, so a terminal's SIGINT to the caller does not reach it.
pub fn bounded_command_until(
    argv: &[String],
    input: Vec<u8>,
    timeout: Duration,
    maximum: usize,
    abort: &dyn Fn() -> bool,
) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    let exe = argv.first().ok_or_else(|| io::Error::other("empty argv"))?;
    let mut command = Command::new(exe);
    command
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    group(&mut command);
    let mut child = command.spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    let writer = thread::spawn(move || stdin.write_all(&input));
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let a = thread::spawn(move || drain(stdout, maximum));
    let b = thread::spawn(move || drain(stderr, maximum));
    let start = Instant::now();
    let mut aborted = false;
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if abort() {
            aborted = true;
            break None;
        }
        if start.elapsed() >= timeout {
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    // A parent may exit leaving descendants holding the pipes. Clean the group
    // even on success before joining bounded pipe readers.
    terminate_process_group(&mut child);
    let _ = child.wait();
    let write_result = writer
        .join()
        .map_err(|_| io::Error::other("stdin thread panicked"))?;
    let (out, oa) = a
        .join()
        .map_err(|_| io::Error::other("stdout thread panicked"))??;
    let (err, ob) = b
        .join()
        .map_err(|_| io::Error::other("stderr thread panicked"))??;
    if aborted {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "command interrupted",
        ));
    }
    let status = status
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "command deadline exceeded"))?;
    write_result?;
    if oa || ob {
        return Err(io::Error::other("command stream exceeds configured cap"));
    }
    Ok((status, out, err))
}
/// Bounded on-disk diagnostics, with pipes drained after cap is reached.
pub fn drain_file(
    mut reader: impl Read,
    mut file: std::fs::File,
    maximum: usize,
) -> io::Result<()> {
    let mut kept = 0;
    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let k = n.min(maximum.saturating_sub(kept));
        file.write_all(&buf[..k])?;
        kept += k;
    }
    file.sync_data()
}
/// Check Linux process-group membership, counting zombies as already stopped.
/// Returns an error when liveness cannot be established. Escaped sessions are
/// outside the foreground backend's contract; use systemd for containment.
#[cfg(target_os = "linux")]
pub fn group_stopped(pgid: u32) -> io::Result<bool> {
    // kill(0) uses the caller's PID namespace and avoids unrelated /proc access.
    let rc = unsafe { libc::kill(-(pgid as i32), 0) };
    if rc < 0 {
        let e = io::Error::last_os_error();
        return if e.raw_os_error() == Some(libc::ESRCH) {
            Ok(true)
        } else {
            Err(e)
        };
    }
    let namespace = std::fs::read_link("/proc/self/ns/pid")?;
    let own_status = std::fs::read_to_string("/proc/self/status")?;
    let depth = own_status
        .lines()
        .find_map(|l| l.strip_prefix("NSpgid:"))
        .ok_or_else(|| io::Error::other("NSpgid unavailable"))?
        .split_whitespace()
        .count();
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().parse::<u32>().is_err() {
            continue;
        }
        // A process that exits during the scan is gone, not unknown.
        let vanished = |e: &io::Error| {
            e.kind() == io::ErrorKind::NotFound || e.raw_os_error() == Some(libc::ESRCH)
        };
        let status = match std::fs::read_to_string(entry.path().join("status")) {
            Ok(s) => s,
            Err(e) if vanished(&e) => continue,
            Err(e) => return Err(e),
        };
        let groups: Vec<_> = status
            .lines()
            .find_map(|l| l.strip_prefix("NSpgid:"))
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if groups.len() != depth || groups.last().and_then(|s| s.parse::<u32>().ok()) != Some(pgid)
        {
            continue;
        }
        let other = match std::fs::read_link(entry.path().join("ns/pid")) {
            Ok(n) => n,
            Err(e) if vanished(&e) => continue,
            Err(e) => return Err(e),
        };
        if other != namespace {
            continue;
        }
        let state = status
            .lines()
            .find_map(|l| l.strip_prefix("State:"))
            .and_then(|s| s.split_whitespace().next());
        if state.is_none() {
            return Err(io::Error::other("process state unavailable"));
        }
        if !matches!(state, Some("Z" | "X")) {
            return Ok(false);
        }
    }
    Ok(true)
}
#[cfg(not(target_os = "linux"))]
pub fn group_stopped(_: u32) -> io::Result<bool> {
    Err(io::Error::other("Linux process verification required"))
}
