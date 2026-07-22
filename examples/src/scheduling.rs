//! Dyson-ring transfer scheduler ported from `examples/scheduling.py`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use xz2::read::XzDecoder;

pub const STATION_COUNT: usize = 12;
pub const MAX_TIME: f64 = 20.0;
pub const WAIT_TIME: f64 = 90.0 / 365.25;
pub const ALPHA: f64 = 6.0e-9;
pub const YEAR: f64 = 24.0 * 3600.0 * 365.25;
pub const A_DYSON: f64 = 1.1;

#[derive(Clone, Copy, Debug)]
pub struct Transfer {
    pub asteroid: usize,
    pub station: usize,
    pub trajectory: usize,
    pub mass: f64,
    pub delta_v: f64,
    pub start: f64,
    pub duration: f64,
}

#[derive(Clone, Debug)]
pub struct Scheduler {
    transfers: Vec<Transfer>,
    pub trajectory_count: usize,
    pub asteroid_count: usize,
    trajectory_delta_v: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ScheduleResult {
    /// Python's optimizer-shaped minimum (weighted sum of sorted slots).
    pub shaped_mass: f64,
    pub slot_mass: [f64; STATION_COUNT],
    pub selected_delta_v: [f64; 10],
}

impl Scheduler {
    pub fn new(transfers: Vec<Transfer>) -> Result<Self, String> {
        if transfers.is_empty() {
            return Err("at least one transfer is required".into());
        }
        let trajectory_count = transfers.iter().map(|row| row.trajectory).max().unwrap() + 1;
        let asteroid_count = transfers.iter().map(|row| row.asteroid).max().unwrap() + 1;
        if trajectory_count < 10 {
            return Err("the scheduler requires at least ten trajectories".into());
        }
        let mut asteroid_delta_v = vec![0.0; trajectory_count * asteroid_count];
        for row in &transfers {
            asteroid_delta_v[row.trajectory * asteroid_count + row.asteroid] = row.delta_v;
        }
        let trajectory_delta_v = asteroid_delta_v
            .chunks(asteroid_count)
            .map(|values| values.iter().sum())
            .collect();
        Ok(Self {
            transfers,
            trajectory_count,
            asteroid_count,
            trajectory_delta_v,
        })
    }

    pub fn sample() -> Self {
        let mut rows = Vec::new();
        for trajectory in 0..10 {
            for slot in 0..STATION_COUNT {
                rows.push(Transfer {
                    asteroid: trajectory * STATION_COUNT + slot,
                    station: slot + 1,
                    trajectory,
                    mass: 1.0e15 + (trajectory * 100 + slot) as f64 * 1.0e12,
                    delta_v: 4.0 + trajectory as f64 * 0.2,
                    start: slot as f64 * MAX_TIME / STATION_COUNT as f64,
                    duration: 0.2,
                });
            }
        }
        Self::new(rows).unwrap()
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| error.to_string())?;
        let reader: Box<dyn Read> = if path.extension().is_some_and(|extension| extension == "xz") {
            Box::new(XzDecoder::new(file))
        } else {
            Box::new(file)
        };
        Self::from_reader(BufReader::new(reader))
    }

    pub fn from_reader(reader: impl BufRead) -> Result<Self, String> {
        let mut transfers = Vec::new();
        for (line_number, line) in reader.lines().enumerate() {
            let line = line.map_err(|error| error.to_string())?;
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let values: Vec<&str> = line.split_whitespace().collect();
            let offset = values
                .len()
                .checked_sub(7)
                .ok_or_else(|| format!("line {} has fewer than seven columns", line_number + 1))?;
            let parse_usize = |index: usize| -> Result<usize, String> {
                values[offset + index]
                    .parse()
                    .map_err(|_| format!("invalid integer on line {}", line_number + 1))
            };
            let parse_float = |index: usize| -> Result<f64, String> {
                values[offset + index]
                    .parse()
                    .map_err(|_| format!("invalid float on line {}", line_number + 1))
            };
            transfers.push(Transfer {
                asteroid: parse_usize(0)?,
                station: parse_usize(1)?,
                trajectory: parse_usize(2)?,
                mass: parse_float(3)?,
                delta_v: parse_float(4)?,
                start: parse_float(5)?,
                duration: parse_float(6)?,
            });
        }
        Self::new(transfers)
    }

    pub fn dimension(&self) -> usize {
        10 + 2 * STATION_COUNT - 1
    }

    pub fn transfer_count(&self) -> usize {
        self.transfers.len()
    }

    pub fn evaluate(&self, x: &[f64]) -> ScheduleResult {
        assert_eq!(x.len(), self.dimension());
        let (trajectory_ids, used_trajectories) = trajectory_selection(x, self.trajectory_count);
        let stations = station_order(x);
        let times = timings(x);
        let mut slot_mass = [0.0_f64; STATION_COUNT];
        let mut asteroid_mass = vec![0.0_f64; self.asteroid_count];
        let mut asteroid_slot = vec![0usize; self.asteroid_count];
        for row in &self.transfers {
            if !used_trajectories[row.trajectory] {
                continue;
            }
            let arrival = row.start + row.duration;
            for slot in 0..STATION_COUNT {
                let minimum = times[slot] + WAIT_TIME;
                if minimum >= MAX_TIME
                    || arrival < times[slot]
                    || arrival > times[slot + 1]
                    || row.station != stations[slot]
                {
                    continue;
                }
                let mut flight_time = row.duration;
                if arrival < minimum {
                    let add = minimum - arrival;
                    flight_time += add * (1.0 + add / WAIT_TIME).sqrt();
                }
                let mut value = (1.0 - YEAR * flight_time * ALPHA) * row.mass;
                if asteroid_mass[row.asteroid] > 0.0 {
                    let old_slot = asteroid_slot[row.asteroid];
                    let minimum_mass = slot_mass.iter().copied().fold(f64::INFINITY, f64::min);
                    let old_mass = slot_mass[old_slot];
                    if (old_slot == slot || minimum_mass < 0.99 * old_mass)
                        && asteroid_mass[row.asteroid] < value
                    {
                        slot_mass[old_slot] -= asteroid_mass[row.asteroid];
                    } else {
                        value = 0.0;
                    }
                }
                if value > 0.0 {
                    slot_mass[slot] += value;
                    asteroid_mass[row.asteroid] = value;
                    asteroid_slot[row.asteroid] = slot;
                }
            }
        }
        slot_mass.sort_by(f64::total_cmp);
        let mut shaped_mass = slot_mass[0];
        let mut factor = 1.0;
        for mass in slot_mass {
            shaped_mass += factor * mass;
            factor *= 0.5;
        }
        let selected_delta_v = std::array::from_fn(|i| self.trajectory_delta_v[trajectory_ids[i]]);
        ScheduleResult {
            shaped_mass,
            slot_mass,
            selected_delta_v,
        }
    }

    pub fn score(&self, x: &[f64]) -> f64 {
        let result = self.evaluate(x);
        score(result.slot_mass[0], &result.selected_delta_v)
    }

    pub fn fitness(&self, x: &[f64]) -> f64 {
        let result = self.evaluate(x);
        -score(result.shaped_mass, &result.selected_delta_v)
    }

    pub fn objectives(&self, x: &[f64]) -> [f64; 2] {
        let result = self.evaluate(x);
        let dv_value = result
            .selected_delta_v
            .iter()
            .map(|dv| (1.0 + dv / 50.0).powi(2))
            .sum();
        [-result.shaped_mass * 1e-10, dv_value]
    }
}

pub fn score(minimum_mass: f64, delta_v: &[f64]) -> f64 {
    let dv_value: f64 = delta_v.iter().map(|dv| (1.0 + dv / 50.0).powi(2)).sum();
    minimum_mass * 1e-10 / (A_DYSON * A_DYSON * dv_value)
}

pub fn trajectory_selection(x: &[f64], count: usize) -> ([usize; 10], Vec<bool>) {
    let mut used = vec![false; count];
    let mut result = [0usize; 10];
    for i in 0..10 {
        let mut candidate = (x[i] as usize).min(count - 1);
        while used[candidate] {
            candidate = (candidate + 1) % count;
        }
        used[candidate] = true;
        result[i] = candidate;
    }
    (result, used)
}

pub fn station_order(x: &[f64]) -> [usize; STATION_COUNT] {
    let mut result: [usize; STATION_COUNT] = std::array::from_fn(|i| i);
    result.sort_by(|&left, &right| x[10 + left].total_cmp(&x[10 + right]));
    result.map(|station| station + 1)
}

pub fn timings(x: &[f64]) -> [f64; STATION_COUNT + 1] {
    let mut times = [0.0; STATION_COUNT + 1];
    for i in 0..STATION_COUNT - 1 {
        times[i] = MAX_TIME * x[10 + STATION_COUNT + i];
    }
    times[STATION_COUNT - 1] = 0.0;
    times[STATION_COUNT] = MAX_TIME;
    times.sort_by(f64::total_cmp);
    times
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> Vec<f64> {
        let mut x = vec![0.0; 10 + 2 * STATION_COUNT - 1];
        for (i, value) in x.iter_mut().take(10).enumerate() {
            *value = i as f64;
        }
        for i in 0..STATION_COUNT {
            x[10 + i] = i as f64 / STATION_COUNT as f64;
        }
        for i in 0..STATION_COUNT - 1 {
            x[10 + STATION_COUNT + i] = (i + 1) as f64 / STATION_COUNT as f64;
        }
        x
    }

    #[test]
    fn selection_disjoins_duplicate_trajectories() {
        let (selected, used) = trajectory_selection(&[0.0; 10], 10);
        assert_eq!(selected, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(used.into_iter().all(|value| value));
    }

    #[test]
    fn sample_is_finite_and_deterministic() {
        let scheduler = Scheduler::sample();
        let x = candidate();
        let result = scheduler.evaluate(&x);
        assert!(result.shaped_mass.is_finite());
        assert!(result.slot_mass.iter().all(|mass| *mass > 0.0));
        assert!((scheduler.fitness(&x) + 72_054.8653059838).abs() < 1e-8);
        let objectives = scheduler.objectives(&x);
        assert!((scheduler.score(&x) - 24_011.191916344935).abs() < 1e-9);
        assert!(objectives[0] < 0.0 && objectives[1] > 0.0);
    }

    #[test]
    fn parser_accepts_indexed_rows() {
        let mut text = String::new();
        for i in 0..10 {
            text.push_str(&format!("{i} {i} 1 {i} 1e15 5 0 1\n"));
        }
        let scheduler = Scheduler::from_reader(text.as_bytes()).unwrap();
        assert_eq!(scheduler.trajectory_count, 10);
        assert_eq!(scheduler.asteroid_count, 10);
    }
}
