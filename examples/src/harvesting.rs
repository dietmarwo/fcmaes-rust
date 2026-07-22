//! Resource-constrained job-shop variant from `examples/harvesting.py`.

use crate::jobshop::JobShop;

pub const TIMING_ERROR: [f64; 4] = [0.0, 0.0, 0.0, 10_000.0];

pub fn adjust_timing(
    starts: &[f64],
    durations: &[f64],
    max_active: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = starts.len();
    if n == 0 || n != durations.len() || max_active == 0 || max_active > n {
        return None;
    }
    let mut start = starts.to_vec();
    let mut start_order = sorted_indices(&start);
    for &index in start_order.iter().take(max_active) {
        start[index] = 0.0;
    }
    let mut stop: Vec<f64> = start
        .iter()
        .zip(durations)
        .map(|(start, duration)| start + duration)
        .collect();
    let mut stop_order = sorted_indices(&stop);
    for &index in stop_order.iter().rev().take(max_active) {
        stop[index] = 10_000_000.0;
    }

    loop {
        let (mut active, mut i, mut j, mut moved) = (0isize, 0usize, 0usize, false);
        while i < n && j < n {
            let current_start = start_order[i];
            let current_stop = stop_order[j];
            if start[current_start] >= stop[current_stop] {
                active -= 1;
                j += 1;
            } else if active < max_active as isize {
                active += 1;
                i += 1;
            } else if current_start != current_stop {
                // Preserve the Python guard (including its `stop[current_start]`
                // index) so existing parameter vectors retain their meaning.
                if start.iter().any(|value| *value == stop[current_start]) {
                    return None;
                }
                start[current_start] = stop[current_stop];
                stop[current_start] = start[current_start] + durations[current_start];
                moved = true;
                break;
            } else {
                j += 1;
            }
        }
        if !moved {
            return Some((start, stop));
        }
        start_order = sorted_indices(&start);
        stop_order = sorted_indices(&stop);
    }
}

pub fn evaluate(problem: &JobShop, x: &[f64], max_active: usize) -> [f64; 4] {
    let operations = problem.operations.len();
    let machines = problem.machines;
    assert_eq!(x.len(), 2 * operations + 2 * machines + 1);
    let tasks = problem.decode(&x[..2 * operations]);
    let max_time = x[x.len() - 1];
    let duration_offset = 2 * operations;
    let start_offset = duration_offset + machines;
    let durations: Vec<f64> = x[duration_offset..start_offset]
        .iter()
        .map(|value| value * max_time)
        .collect();
    let starts: Vec<f64> = x[start_offset..start_offset + machines]
        .iter()
        .map(|value| value * max_time)
        .collect();
    let Some((starts, stops)) = adjust_timing(&starts, &durations, max_active) else {
        return TIMING_ERROR;
    };
    let mut machine_time = starts;
    let mut machine_work = vec![0.0_f64; machines];
    let mut job_time = vec![0.0_f64; problem.jobs];
    let mut failures = 0usize;
    for task in tasks {
        let end = machine_time[task.machine].max(job_time[task.job]) + task.time;
        if end > stops[task.machine] {
            failures += 1;
            continue;
        }
        machine_time[task.machine] = end;
        job_time[task.job] = end;
        machine_work[task.machine] += task.time;
    }
    [
        machine_time.iter().copied().fold(0.0, f64::max),
        machine_work.iter().sum(),
        machine_work.iter().copied().fold(0.0, f64::max),
        failures as f64,
    ]
}

pub fn fitness(problem: &JobShop, x: &[f64], max_active: usize) -> f64 {
    let [span, work, max_work, failures] = evaluate(problem, x, max_active);
    span + 0.02 * work + 0.001 * max_work + 1000.0 * failures
}

fn sorted_indices(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_enforces_active_limit() {
        let (start, stop) = adjust_timing(&[2.0, 4.0, 6.0], &[5.0, 5.0, 5.0], 1).unwrap();
        assert_eq!(start, [0.0, 5.0, 10.0]);
        assert_eq!(stop, [5.0, 10.0, 15.0]);
        assert!(adjust_timing(&[], &[], 1).is_none());
        assert!(adjust_timing(&[0.0], &[1.0], 2).is_none());
    }

    #[test]
    fn objective_matches_python_reference() {
        let problem = JobShop::mini();
        let x = [
            0.1, 0.2, 0.3, 0.4, 0.8, 0.1, 0.7, 0.2, // job-shop keys
            0.5, 0.5, 0.5, // durations
            0.1, 0.2, 0.3, // starts
            20.0,
        ];
        assert_eq!(evaluate(&problem, &x, 2), [22.0, 14.0, 9.0, 0.0]);
        assert!((fitness(&problem, &x, 2) - 22.289).abs() < 1e-12);
    }
}
