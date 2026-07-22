//! Flexible job-shop objective ported from `examples/jobshop.py`.

use std::fs;
use std::path::Path;

pub const MINI_FJS: &str = "2 3\n2 2 1 3 2 2 2 2 4 3 6\n2 2 1 2 3 5 1 2 3\n";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Task {
    pub job: usize,
    pub operation: usize,
    pub machine: usize,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct JobShop {
    pub jobs: usize,
    pub machines: usize,
    pub operations: Vec<Vec<Task>>,
    pub sum_time: f64,
    operation_jobs: Vec<usize>,
    job_starts: Vec<usize>,
}

impl JobShop {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
        Self::parse(&text)
    }

    pub fn mini() -> Self {
        Self::parse(MINI_FJS).expect("the embedded FJS instance is valid")
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let header = numbers(lines.next().ok_or("missing FJS header")?)?;
        if header.len() < 2 {
            return Err("FJS header must contain job and machine counts".into());
        }
        let jobs = as_usize(header[0], "job count")?;
        let machines = as_usize(header[1], "machine count")?;
        if jobs == 0 || machines == 0 {
            return Err("job and machine counts must be positive".into());
        }

        let mut operations = Vec::new();
        let mut operation_jobs = Vec::new();
        let mut job_starts = Vec::with_capacity(jobs);
        let mut sum_time = 0.0;
        for job in 0..jobs {
            job_starts.push(operations.len());
            let values = numbers(lines.next().ok_or("missing FJS job row")?)?;
            let operation_count = values
                .first()
                .copied()
                .ok_or_else(|| "empty FJS job row".to_string())
                .and_then(|value| as_usize(value, "operation count"))?;
            let mut cursor = 1;
            for _ in 0..operation_count {
                let alternatives = values
                    .get(cursor)
                    .copied()
                    .ok_or_else(|| "truncated FJS operation".to_string())
                    .and_then(|value| as_usize(value, "alternative count"))?;
                cursor += 1;
                if alternatives == 0 {
                    return Err("an operation must have a machine alternative".into());
                }
                let operation = operations.len();
                let mut choices = Vec::with_capacity(alternatives);
                for _ in 0..alternatives {
                    let machine = values
                        .get(cursor)
                        .copied()
                        .ok_or_else(|| "truncated FJS machine".to_string())
                        .and_then(|value| as_usize(value, "machine"))?;
                    let time = *values.get(cursor + 1).ok_or("truncated FJS time")?;
                    cursor += 2;
                    if machine == 0 || machine > machines || time < 0.0 {
                        return Err("invalid FJS machine or processing time".into());
                    }
                    sum_time += time;
                    choices.push(Task {
                        job,
                        operation,
                        machine: machine - 1,
                        time,
                    });
                }
                operations.push(choices);
                operation_jobs.push(job);
            }
            if cursor != values.len() {
                return Err(format!("unexpected values at end of FJS job {job}"));
            }
        }
        Ok(Self {
            jobs,
            machines,
            operations,
            sum_time,
            operation_jobs,
            job_starts,
        })
    }

    pub fn dimension(&self) -> usize {
        2 * self.operations.len()
    }

    /// Decode machine choices and random keys exactly as the Python example.
    pub fn decode(&self, x: &[f64]) -> Vec<Task> {
        let n = self.operations.len();
        assert_eq!(x.len(), 2 * n);
        let selected: Vec<Task> = self
            .operations
            .iter()
            .enumerate()
            .map(|(i, alternatives)| {
                let index = ((x[i] * 10.0 * self.machines as f64) as usize) % alternatives.len();
                alternatives[index]
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| x[n + left].total_cmp(&x[n + right]));
        let mut next_in_job = vec![0usize; self.jobs];
        order
            .into_iter()
            .map(|operation| {
                let job = self.operation_jobs[operation];
                let index = self.job_starts[job] + next_in_job[job];
                next_in_job[job] += 1;
                selected[index]
            })
            .collect()
    }

    pub fn evaluate(&self, x: &[f64]) -> [f64; 3] {
        self.evaluate_tasks(&self.decode(x))
    }

    pub fn evaluate_tasks(&self, tasks: &[Task]) -> [f64; 3] {
        let mut machine_time = vec![0.0_f64; self.machines];
        let mut machine_work = vec![0.0_f64; self.machines];
        let mut job_time = vec![0.0_f64; self.jobs];
        for task in tasks {
            let end = machine_time[task.machine].max(job_time[task.job]) + task.time;
            machine_time[task.machine] = end;
            job_time[task.job] = end;
            machine_work[task.machine] += task.time;
        }
        [
            machine_time.iter().copied().fold(0.0, f64::max),
            machine_work.iter().sum(),
            machine_work.iter().copied().fold(0.0, f64::max),
        ]
    }

    pub fn fitness(&self, x: &[f64]) -> f64 {
        let [span, work, max_work] = self.evaluate(x);
        span + 0.02 * work + 0.001 * max_work
    }
}

fn numbers(line: &str) -> Result<Vec<f64>, String> {
    line.split_whitespace()
        .map(|token| token.parse::<f64>().map_err(|error| error.to_string()))
        .collect()
}

fn as_usize(value: f64, name: &str) -> Result<usize, String> {
    if value >= 0.0 && value.fract() == 0.0 {
        Ok(value as usize)
    } else {
        Err(format!("invalid {name}: {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_instance_matches_python_reference() {
        let problem = JobShop::mini();
        assert_eq!(
            (problem.jobs, problem.machines, problem.operations.len()),
            (2, 3, 4)
        );
        assert_eq!(problem.sum_time, 25.0);
        let x = [0.1, 0.2, 0.3, 0.4, 0.8, 0.1, 0.7, 0.2];
        assert_eq!(problem.evaluate(&x), [12.0, 14.0, 9.0]);
        assert!((problem.fitness(&x) - 12.289).abs() < 1e-12);
    }

    #[test]
    fn parser_rejects_bad_input() {
        assert!(JobShop::parse("").is_err());
        assert!(JobShop::parse("1 1\n1 1 2 3\n").is_err());
        assert!(JobShop::parse("1 1\n1 0\n").is_err());
    }
}
