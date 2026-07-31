//! Versioned publication-artifact helpers.

use std::fs;
use std::io;
use std::path::Path;

use serde::Serialize;

/// Write pretty, newline-terminated JSON after creating its parent directory.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<(), io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    bytes.push(b'\n');
    fs::write(path, bytes)
}

/// Write UTF-8 text after creating its parent directory.
pub fn write_text(path: &Path, value: &str) -> Result<(), io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, value)
}
