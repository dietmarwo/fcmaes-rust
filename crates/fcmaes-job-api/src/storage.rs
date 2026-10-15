//! Durable primitives extracted from route_search and route_archive (3ef2673).
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
static TEMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<(), io::Error> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_data()?;
    Ok(())
}

/// Reads JSONL while ignoring only an incomplete final record.
///
/// A malformed newline-terminated record or corruption before the final line
/// remains fatal.
///
/// # Errors
///
/// Returns an error for I/O or non-final JSON corruption.
pub fn read_jsonl_resilient<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, io::Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let bytes = reader.read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        let terminated = line.last() == Some(&b'\n');
        if terminated {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice(&line) {
            Ok(record) => records.push(record),
            Err(_) if !terminated => break,
            Err(error) => return Err(io::Error::other(error)),
        }
    }
    Ok(records)
}

/// Writes a JSON snapshot to a sibling temporary file and atomically renames
/// it over the target.
///
/// # Errors
///
/// Returns an I/O or serialization error.
pub fn write_atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), io::Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::other("snapshot path has no file name".to_owned()))?;
    let temporary = temporary_path(parent, filename);
    let result = (|| {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok::<(), io::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn temporary_path(parent: &Path, filename: &str) -> PathBuf {
    parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// Route archive v2 framing, kept byte compatible.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub schema_version: u32,
    pub checksum: String,
    pub payload_json: String,
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn append_checked<T: Serialize>(path: &Path, version: u32, value: &T) -> io::Result<()> {
    let payload_json = serde_json::to_string(value)?;
    append_jsonl(
        path,
        &Envelope {
            schema_version: version,
            checksum: digest(payload_json.as_bytes()),
            payload_json,
        },
    )
}
pub fn read_checked<T: DeserializeOwned>(path: &Path, version: u32) -> io::Result<Vec<T>> {
    let lines: Vec<Envelope> = read_jsonl_resilient(path)?;
    lines
        .into_iter()
        .map(|line| {
            if line.schema_version != version
                || line.checksum != digest(line.payload_json.as_bytes())
            {
                return Err(io::Error::other("checksum or schema mismatch"));
            }
            Ok(serde_json::from_str(&line.payload_json)?)
        })
        .collect()
}
/// Repair only after verified replay, under exclusive job ownership.
/// A valid unterminated record is retained and receives its newline.
pub fn repair_tail<T: DeserializeOwned>(path: &Path, version: u32) -> io::Result<Vec<T>> {
    let records = read_checked(path, version)?;
    let data = fs::read(path)?;
    if !data.is_empty() && !data.ends_with(b"\n") {
        let start = data.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let mut file = OpenOptions::new().append(true).open(path)?;
        if serde_json::from_slice::<Envelope>(&data[start..]).is_ok() {
            file.write_all(b"\n")?;
        } else {
            file.set_len(start as u64)?;
        }
        file.sync_all()?;
    }
    Ok(records)
}
/// Liveness probe for a lock that a process holds for its whole lifetime.
///
/// The kernel releases `flock` locks when their holder exits, however it
/// exits, so a free lock proves the holder is gone. Returns `Ok(true)` when
/// the lock is free or the file does not exist, `Ok(false)` when another open
/// file description holds it. The probe takes the lock only momentarily.
///
/// # Errors
///
/// Returns an error when the lock state cannot be determined.
pub fn lock_is_free(path: &Path) -> io::Result<bool> {
    let file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(true),
        Err(e) => return Err(e),
    };
    match fs2::FileExt::try_lock_exclusive(&file) {
        Ok(()) => Ok(true),
        Err(e) if e.raw_os_error() == fs2::lock_contended_error().raw_os_error() => Ok(false),
        Err(e) => Err(e),
    }
}
/// Stable inode lock. Never unlink it; atomic replacement is for data only.
pub fn lock(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    fs2::FileExt::lock_exclusive(&file)?;
    Ok(file)
}
