//! A single sequential evaluation lane owned by one admitted search attempt.
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Largest response frame in JSON bytes, excluding the newline terminator;
/// the Julia server applies the same limit to the frames it emits.
pub const FRAME_LIMIT: usize = 1 << 20;

/// How long a failed server may take to be explained by a pending cancellation.
const CANCEL_GRACE: Duration = Duration::from_millis(200);

#[derive(Debug, Clone)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl Failure {
    pub fn new(code: &'static str, message: impl ToString) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }
    pub fn io(error: impl ToString) -> Self {
        Self::new("io_error", error)
    }
}

pub type Result<T> = std::result::Result<T, Failure>;

pub struct Bridge {
    child: Child,
    input: Option<ChildStdin>,
    responses: Option<Receiver<Result<Value>>>,
    reader: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<String>>,
    deadline: Instant,
    cancelled: Arc<AtomicBool>,
    sequence: u64,
}

impl Bridge {
    pub fn spawn(
        mut command: Command,
        deadline: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(Failure::io)?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");
        // One frame ahead at most; a noisy/broken child cannot grow an unbounded queue.
        let (sender, receiver) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = Vec::new();
                let read = reader
                    .by_ref()
                    .take((FRAME_LIMIT + 2) as u64)
                    .read_until(b'\n', &mut line);
                let value = match read {
                    Ok(0) => Err(Failure::new("protocol_error", "server stdout closed")),
                    Ok(n) if n > FRAME_LIMIT + 1 || line.last() != Some(&b'\n') => Err(
                        Failure::new("protocol_error", "oversized or truncated response frame"),
                    ),
                    Ok(_) => {
                        serde_json::from_slice(&line).map_err(|e| Failure::new("protocol_error", e))
                    }
                    Err(e) => Err(Failure::io(e)),
                };
                let failed = value.is_err();
                if sender.send(value).is_err() || failed {
                    break;
                }
            }
        });
        let stderr = thread::spawn(move || {
            let mut kept = Vec::new();
            let mut buffer = [0_u8; 4096];
            while let Ok(n) = stderr.read(&mut buffer) {
                if n == 0 {
                    break;
                }
                let retain = n.min((64 * 1024_usize).saturating_sub(kept.len()));
                kept.extend_from_slice(&buffer[..retain]);
            }
            String::from_utf8_lossy(&kept).into_owned()
        });
        Ok(Self {
            child,
            input,
            responses: Some(receiver),
            reader: Some(reader),
            stderr: Some(stderr),
            deadline,
            cancelled,
            sequence: 0,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn checkpoint(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(Failure::new("cancelled", "search cancelled"))
        } else if Instant::now() >= self.deadline {
            Err(Failure::new("timed_out", "search wall deadline reached"))
        } else {
            Ok(())
        }
    }

    /// Report a server failure as the cancellation or timeout that caused it.
    ///
    /// A signal sent to the whole process group (a terminal Ctrl-C, or a
    /// runtime stopping the attempt) also ends the server, and its closed pipe
    /// can arrive before this process's own handler has recorded the
    /// cancellation. Such a failure is a cancellation, not a protocol error.
    fn attribute(&self, failure: Failure) -> Failure {
        if let Err(termination) = self.checkpoint() {
            return termination;
        }
        let until = Instant::now() + CANCEL_GRACE;
        while Instant::now() < until {
            thread::sleep(Duration::from_millis(5));
            if self.cancelled.load(Ordering::Acquire) {
                return Failure::new("cancelled", "search cancelled");
            }
        }
        failure
    }

    fn response(&self, id: u64, status: &str) -> Result<Value> {
        loop {
            self.checkpoint()?;
            let wait = self
                .deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(20));
            match self
                .responses
                .as_ref()
                .expect("open bridge")
                .recv_timeout(wait)
            {
                Ok(value) => {
                    let value = value.map_err(|failure| self.attribute(failure))?;
                    if value["schema_version"] != "cap.search.response/0.1"
                        || value["request_id"].as_u64() != Some(id)
                        || value["status"] != status
                    {
                        return Err(Failure::new(
                            "protocol_error",
                            "response schema, sequence or status mismatch",
                        ));
                    }
                    return Ok(value);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.attribute(Failure::new(
                        "protocol_error",
                        "server response channel closed",
                    )));
                }
            }
        }
    }

    pub fn ready(&self) -> Result<Value> {
        self.response(0, "ready")
    }

    pub fn request(&mut self, op: &str, parameters: Option<&[f64]>, status: &str) -> Result<Value> {
        self.checkpoint()?;
        self.sequence += 1;
        let mut value =
            json!({"schema_version":"cap.search.request/0.1","request_id":self.sequence,"op":op});
        if let Some(parameters) = parameters {
            value["parameters"] = json!(parameters);
        }
        let mut bytes = serde_json::to_vec(&value).map_err(Failure::io)?;
        if bytes.len() > 4095 {
            return Err(Failure::new(
                "protocol_error",
                "request frame exceeds 4096 bytes",
            ));
        }
        bytes.push(b'\n');
        let input = self.input.as_mut().expect("open bridge stdin");
        if let Err(error) = input.write_all(&bytes).and_then(|()| input.flush()) {
            return Err(self.attribute(Failure::io(error)));
        }
        self.response(self.sequence, status)
    }

    pub fn finish(&mut self) -> Result<()> {
        self.request("shutdown", None, "stopped")?;
        loop {
            self.checkpoint()?;
            if let Some(status) = self.child.try_wait().map_err(Failure::io)? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(self.attribute(Failure::new("server_failed", status)))
                };
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        drop(self.input.take());
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        // Release the bounded sender before joining the stdout reader.
        drop(self.responses.take());
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        if let Some(stderr) = self.stderr.take()
            && let Ok(log) = stderr.join()
            && !log.is_empty()
        {
            eprintln!("Julia server stderr (first 64 KiB):\n{log}");
        }
    }
}
