//! Shared pieces of the CMA-ES bound-handling comparison
//! (`benchmarks/cmaes-bound-handling/PROTOCOL.md`): the three modes and
//! resumable TSV output.
#![forbid(unsafe_code)]

use fcmaes_core::CmaesParams;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// How CMA-ES handles samples that leave the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `repair_z: false`, `bound_transform: false` (the default).
    Clip,
    /// `repair_z: true`.
    Repair,
    /// `bound_transform: true` (experimental).
    Transform,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Clip, Mode::Repair, Mode::Transform];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Clip => "clip",
            Mode::Repair => "repair",
            Mode::Transform => "transform",
        }
    }

    pub fn parse(name: &str) -> Result<Mode, String> {
        Mode::ALL
            .into_iter()
            .find(|m| m.name() == name)
            .ok_or_else(|| format!("unknown mode '{name}' (clip, repair, transform)"))
    }

    /// Sets the mode's fields of `params`.
    pub fn apply(self, params: CmaesParams) -> CmaesParams {
        CmaesParams {
            repair_z: self == Mode::Repair,
            bound_transform: self == Mode::Transform,
            ..params
        }
    }
}

/// `a,b` or `a-b` (inclusive) or a mix: `1-3,7`.
pub fn parse_seeds(text: &str) -> Result<Vec<u64>, String> {
    let mut seeds = Vec::new();
    for part in text.split(',') {
        let bad = || format!("bad seed list '{text}'");
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (u64, u64) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
            seeds.extend(a..=b);
        } else {
            seeds.push(part.parse().map_err(|_| bad())?);
        }
    }
    Ok(seeds)
}

/// A TSV file of rows keyed by their first `key_columns` fields: rows already
/// present are skipped, new rows are appended and flushed one by one.
pub struct Table {
    out: Option<std::fs::File>,
    done: HashSet<Vec<String>>,
    key_columns: usize,
}

impl Table {
    /// Opens `path` (or stdout only when `None`), writing `header` to a new file.
    pub fn open(path: Option<&Path>, header: &str, key_columns: usize) -> Self {
        let mut done = HashSet::new();
        let out = path.map(|path| {
            let exists = path.exists();
            if exists {
                let text = std::fs::read_to_string(path).expect("read the output file");
                for line in text.lines().skip(1) {
                    let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
                    if fields.len() == header.split('\t').count() {
                        done.insert(fields[..key_columns].to_vec());
                    }
                }
            }
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open the output file");
            if !exists {
                writeln!(file, "{header}").expect("write the header");
            }
            file
        });
        if path.is_none() {
            println!("{header}");
        }
        Table {
            out,
            done,
            key_columns,
        }
    }

    pub fn is_done(&self, key: &[String]) -> bool {
        self.done.contains(&key[..self.key_columns])
    }

    pub fn write(&mut self, fields: &[String]) {
        let line = fields.join("\t");
        println!("{line}");
        if let Some(file) = &mut self.out {
            writeln!(file, "{line}").expect("write a row");
            file.flush().expect("flush");
        }
        self.done.insert(fields[..self.key_columns].to_vec());
    }
}

/// The value of `--name` in `args`, if given.
pub fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}
