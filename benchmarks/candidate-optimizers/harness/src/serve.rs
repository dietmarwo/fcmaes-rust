//! `serve`: the line protocol through which the round-3 candidate driver
//! (`run_catcmawm.py`) evaluates the harness objectives.
//!
//! The driver starts `candidate-harness serve` once and keeps it running.
//! Every request is one header line of space-separated tokens, followed by
//! the number of data lines the header announces; every response starts with
//! `ok` and is flushed as a whole. Each `f64` travels as its IEEE-754 bit
//! pattern in 16 lowercase hexadecimal digits ([`encode`]), so values
//! round-trip exactly, including `-0.0`, infinities and NaN payloads.
//!
//! | Request | Data lines | Response |
//! |---|---|---|
//! | `info P` | none | `ok D domain` (the domain is the rest of the line), then the lower and the upper bounds (one line each, D values) |
//! | `eval P N` | N vectors of D values | `ok N NS`, then N lines with one value each; NS is the objective time of the batch in nanoseconds |
//! | `check P N` | N lines of D values and one expected value | `ok M`: M vectors whose in-process evaluation by a separately built instance of P differs bitwise from the expected value |
//! | `levels P N` | N vectors of D values | `ok N`, then per vector D tokens: the level index each discrete coordinate decodes to with the problem's own decoder ([`crate::round3::level_probe`]), `-` for a continuous coordinate |
//! | `echo N` | N lines of values | `ok N`, then the same values re-encoded |
//! | `fmt N` | N lines of one value | `ok N`, then each value in the raw-file float format (`{:e}`) |
//! | `quit` | none | none; the process exits |
//!
//! `eval` refuses (with an error) a vector of the wrong length or outside the
//! problem's box: the driver's guard must have refused it before. Any error
//! is answered with one line `err <message>`, after which the process exits
//! with a non-zero status; end of input ends it normally.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::time::Instant;

use crate::problems::{self, Problem};
use crate::round3::{Kind, Probe, TypeTable, level_probe, type_table};

/// Wire form of a value: its bit pattern in 16 lowercase hex digits.
pub fn encode(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

/// Inverse of [`encode`].
///
/// # Errors
/// Returns a message unless `token` is exactly 16 hex digits.
pub fn decode(token: &str) -> Result<f64, String> {
    if token.len() != 16 || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "bad value token '{token}' (16 hex digits expected)"
        ));
    }
    u64::from_str_radix(token, 16)
        .map(f64::from_bits)
        .map_err(|e| format!("bad value token '{token}': {e}"))
}

fn decode_line(line: &str) -> Result<Vec<f64>, String> {
    line.split_ascii_whitespace().map(decode).collect()
}

struct Entry {
    problem: Problem,
    table: Option<TypeTable>,
    probe: Option<Probe>,
}

/// Problems built on first use.
#[derive(Default)]
struct Cache {
    entries: HashMap<String, Entry>,
}

impl Cache {
    fn get(&mut self, name: &str) -> Result<&Entry, String> {
        if !self.entries.contains_key(name) {
            let problem =
                problems::by_name(name).ok_or_else(|| format!("unknown problem '{name}'"))?;
            self.entries.insert(
                name.to_owned(),
                Entry {
                    problem,
                    table: None,
                    probe: None,
                },
            );
        }
        Ok(&self.entries[name])
    }

    fn with_levels(&mut self, name: &str) -> Result<&Entry, String> {
        self.get(name)?;
        let entry = self.entries.get_mut(name).expect("inserted above");
        if entry.probe.is_none() {
            entry.table =
                Some(type_table(name).ok_or_else(|| format!("no type table for '{name}'"))?);
            entry.probe =
                Some(level_probe(name).ok_or_else(|| format!("no level probe for '{name}'"))?);
        }
        Ok(&self.entries[name])
    }
}

fn read_line<R: BufRead>(input: &mut R) -> Result<Option<String>, String> {
    let mut line = String::new();
    let read = input.read_line(&mut line).map_err(|e| e.to_string())?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_owned()))
}

fn data_lines<R: BufRead>(input: &mut R, count: usize) -> Result<Vec<Vec<f64>>, String> {
    (0..count)
        .map(|i| {
            let line = read_line(input)?
                .ok_or_else(|| format!("input ended after {i} of {count} data lines"))?;
            decode_line(&line)
        })
        .collect()
}

fn parse_count(token: Option<&str>) -> Result<usize, String> {
    token
        .ok_or("missing count")?
        .parse()
        .map_err(|_| "bad count".to_owned())
}

fn check_vector(problem: &Problem, x: &[f64]) -> Result<(), String> {
    if x.len() != problem.dim() {
        return Err(format!(
            "{}: vector of length {}, expected {}",
            problem.name,
            x.len(),
            problem.dim()
        ));
    }
    let inside = x
        .iter()
        .zip(problem.lower.iter().zip(&problem.upper))
        .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi);
    if !inside {
        return Err(format!("{}: vector outside the box", problem.name));
    }
    Ok(())
}

/// Answer one request. Returns `Ok(false)` on `quit`.
fn handle<R: BufRead>(
    header: &str,
    input: &mut R,
    cache: &mut Cache,
    out: &mut String,
) -> Result<bool, String> {
    let mut tokens = header.split_ascii_whitespace();
    let command = tokens.next().ok_or("empty request")?;
    match command {
        "quit" => return Ok(false),
        "info" => {
            let name = tokens.next().ok_or("info needs a problem")?;
            let problem = &cache.get(name)?.problem;
            out.push_str(&format!("ok {} {}\n", problem.dim(), problem.domain));
            for bounds in [&problem.lower, &problem.upper] {
                let line: Vec<String> = bounds.iter().map(|v| encode(*v)).collect();
                out.push_str(&line.join(" "));
                out.push('\n');
            }
        }
        "eval" => {
            let name = tokens.next().ok_or("eval needs a problem")?;
            let count = parse_count(tokens.next())?;
            let rows = data_lines(input, count)?;
            let problem = &cache.get(name)?.problem;
            for x in &rows {
                check_vector(problem, x)?;
            }
            let started = Instant::now();
            let values: Vec<f64> = rows.iter().map(|x| (problem.objective)(x)).collect();
            let nanos = started.elapsed().as_nanos();
            out.push_str(&format!("ok {count} {nanos}\n"));
            for value in values {
                out.push_str(&encode(value));
                out.push('\n');
            }
        }
        "check" => {
            let name = tokens.next().ok_or("check needs a problem")?;
            let count = parse_count(tokens.next())?;
            let rows = data_lines(input, count)?;
            // A separately built instance: the reference evaluation does not
            // share state with the one `eval` uses.
            let fresh =
                problems::by_name(name).ok_or_else(|| format!("unknown problem '{name}'"))?;
            let mut mismatches = 0usize;
            for row in &rows {
                let (expected, x) = row.split_last().ok_or("check line without values")?;
                check_vector(&fresh, x)?;
                if (fresh.objective)(x).to_bits() != expected.to_bits() {
                    mismatches += 1;
                }
            }
            out.push_str(&format!("ok {mismatches}\n"));
        }
        "levels" => {
            let name = tokens.next().ok_or("levels needs a problem")?;
            let count = parse_count(tokens.next())?;
            let rows = data_lines(input, count)?;
            let entry = cache.with_levels(name)?;
            let table = entry.table.as_ref().expect("built with the probe");
            let probe = entry.probe.as_ref().expect("built");
            out.push_str(&format!("ok {count}\n"));
            for x in &rows {
                check_vector(&entry.problem, x)?;
                let mut tokens = Vec::with_capacity(x.len());
                for coordinate in &table.coordinates {
                    tokens.push(if coordinate.kind == Kind::Continuous {
                        "-".to_owned()
                    } else {
                        probe(coordinate.index, x[coordinate.index])?.to_string()
                    });
                }
                out.push_str(&tokens.join(" "));
                out.push('\n');
            }
        }
        "echo" => {
            let count = parse_count(tokens.next())?;
            let rows = data_lines(input, count)?;
            out.push_str(&format!("ok {count}\n"));
            for row in rows {
                let line: Vec<String> = row.iter().map(|v| encode(*v)).collect();
                out.push_str(&line.join(" "));
                out.push('\n');
            }
        }
        "fmt" => {
            let count = parse_count(tokens.next())?;
            let rows = data_lines(input, count)?;
            out.push_str(&format!("ok {count}\n"));
            for row in rows {
                let [value] = row[..] else {
                    return Err("fmt takes one value per line".into());
                };
                out.push_str(&format!("{value:e}\n"));
            }
        }
        other => return Err(format!("unknown request '{other}'")),
    }
    Ok(true)
}

/// Serve requests from `input` until `quit` or end of input, flushing
/// `output` after every response.
///
/// # Errors
/// Returns the message that was also sent as `err <message>`, or an output
/// error.
pub fn serve<R: BufRead, W: Write>(mut input: R, mut output: W) -> Result<(), String> {
    let mut cache = Cache::default();
    while let Some(header) = read_line(&mut input)? {
        if header.trim().is_empty() {
            continue;
        }
        let mut response = String::new();
        match handle(&header, &mut input, &mut cache, &mut response) {
            Ok(true) => {
                output
                    .write_all(response.as_bytes())
                    .and_then(|()| output.flush())
                    .map_err(|e| e.to_string())?;
            }
            Ok(false) => return Ok(()),
            Err(message) => {
                let one_line = message.replace('\n', " ");
                let _ = output
                    .write_all(format!("err {one_line}\n").as_bytes())
                    .and_then(|()| output.flush());
                return Err(message);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip_bit_for_bit() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.5e-300,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
            1e12,
            0.1 + 0.2,
        ];
        for v in specials {
            assert_eq!(decode(&encode(v)).unwrap().to_bits(), v.to_bits());
        }
        let mut rng = fcmaes_core::Rng::new(7);
        for _ in 0..10_000 {
            let v = f64::from_bits(rng.next_u64());
            assert_eq!(decode(&encode(v)).unwrap().to_bits(), v.to_bits());
        }
        assert!(decode("3ff").is_err());
        assert!(decode("3ff000000000000g").is_err());
    }
}
