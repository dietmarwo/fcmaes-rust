//! Native material-flow-planning objective.
//!
//! Two machines process batches of different product types with setup times
//! and an eight-part intermediate FIFO. The deliberately fine-grained model
//! advances a full 24-hour production day one second at a time. This makes it
//! a useful demonstration of the benefit of implementing a hot objective in
//! Rust rather than calling a Python object graph for every simulated second.

use std::collections::VecDeque;

pub const DAY_SECONDS: i64 = 60 * 60 * 24;
pub const BUFFER_CAPACITY: usize = 8;
pub const SETUP_TIME: i64 = 10;
pub const ORIGINAL_DURATIONS: [[i64; 2]; 2] = [[10, 20], [25, 15]];

#[derive(Clone, Debug)]
struct Machine {
    product: Option<usize>,
    last_product: Option<usize>,
    time: i64,
}

impl Machine {
    fn new() -> Self {
        Self {
            product: None,
            last_product: None,
            time: 0,
        }
    }

    fn process(&mut self, now: i64, product: usize, duration: i64) {
        let mut start = now.max(self.time);
        if self.last_product != Some(product) {
            start += SETUP_TIME;
        }
        self.product = Some(product);
        self.time = start + duration;
    }

    fn take(&mut self) -> usize {
        let product = self
            .product
            .take()
            .expect("only finished machines are taken");
        self.last_product = Some(product);
        product
    }

    fn finished(&self, now: i64) -> bool {
        now >= self.time && self.product.is_some()
    }

    fn available(&self, now: i64) -> bool {
        now >= self.time && self.product.is_none()
    }
}

#[derive(Clone, Debug)]
pub struct Plant {
    durations: Vec<Vec<i64>>,
    machine1: Machine,
    machine2: Machine,
    buffer: VecDeque<usize>,
    count_finished: i64,
    time: i64,
    product: Option<usize>,
    to_process: usize,
    batch_size: Vec<usize>,
}

impl Plant {
    pub fn try_new(durations: Vec<Vec<i64>>) -> Result<Self, &'static str> {
        if durations.len() != 2
            || durations[0].is_empty()
            || durations[0].len() != durations[1].len()
            || durations.iter().flatten().any(|duration| *duration <= 0)
        {
            return Err("durations must be two equal, non-empty rows of positive values");
        }
        let dim = durations[0].len();
        Ok(Self {
            durations,
            machine1: Machine::new(),
            machine2: Machine::new(),
            buffer: VecDeque::with_capacity(BUFFER_CAPACITY),
            count_finished: 0,
            time: 0,
            product: None,
            to_process: 0,
            batch_size: vec![1; dim],
        })
    }

    pub fn original() -> Self {
        Self::try_new(ORIGINAL_DURATIONS.iter().map(|row| row.to_vec()).collect())
            .expect("valid original Siemens durations")
    }

    pub fn dim(&self) -> usize {
        self.durations[0].len()
    }

    fn reset(&mut self, batch_size: &[usize]) {
        assert_eq!(batch_size.len(), self.dim());
        self.machine1 = Machine::new();
        self.machine2 = Machine::new();
        self.buffer.clear();
        self.count_finished = 0;
        self.time = 0;
        self.product = None;
        self.to_process = 0;
        self.batch_size.copy_from_slice(batch_size);
    }

    fn next_product(&mut self) -> usize {
        if self.to_process > 0 {
            self.to_process -= 1;
        } else {
            let next = self.product.map_or(0, |product| (product + 1) % self.dim());
            self.product = Some(next);
            // Preserve the Python example's semantics: the switching tick plus
            // `batch_size` following ticks produce this product.
            self.to_process = self.batch_size[next];
        }
        self.product.expect("next_product initializes the product")
    }

    fn tick(&mut self) {
        if self.machine1.finished(self.time) && self.buffer.len() < BUFFER_CAPACITY {
            self.buffer.push_back(self.machine1.take());
        }

        if self.machine1.available(self.time) {
            let product = self.next_product();
            self.machine1
                .process(self.time, product, self.durations[0][product]);
        }

        if self.machine2.finished(self.time) {
            self.machine2.take();
            self.count_finished += 1;
        }

        if self.machine2.available(self.time)
            && let Some(product) = self.buffer.pop_front()
        {
            self.machine2
                .process(self.time, product, self.durations[1][product]);
        }

        self.time += 1;
    }

    pub fn simulate(&mut self, seconds: i64, batch_size: &[usize]) -> i64 {
        self.reset(batch_size);
        while self.time < seconds {
            self.tick();
        }
        self.count_finished
    }

    /// Thread-safe scalar objective. Continuous optimizer values are truncated
    /// to integers exactly like `numpy.astype(int)` for these positive bounds.
    pub fn fitness(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim());
        let batches: Vec<usize> = x
            .iter()
            .map(|value| value.clamp(1.0, 50.0) as usize)
            .collect();
        let mut simulation = self.clone();
        -(simulation.simulate(DAY_SECONDS, &batches) as f64)
    }
}

/// Deterministic benchmark candidates shared by the Python implementation.
pub fn benchmark_candidate(index: usize) -> [f64; 2] {
    [
        1.0 + ((17 * index + 3) % 50) as f64,
        1.0 + ((31 * index + 7) % 50) as f64,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_reference_values_match_python() {
        let mut plant = Plant::original();
        assert_eq!(plant.simulate(DAY_SECONDS, &[10, 32]), 4802);
        assert_eq!(plant.simulate(DAY_SECONDS, &[1, 1]), 3454);
        assert_eq!(plant.simulate(DAY_SECONDS, &[50, 50]), 4093);
        assert_eq!(plant.simulate(DAY_SECONDS, &[20, 20]), 4214);
        assert_eq!(plant.simulate(DAY_SECONDS, &[7, 13]), 4418);
    }

    #[test]
    fn validates_durations_and_objective_conversion() {
        assert!(Plant::try_new(vec![]).is_err());
        assert!(Plant::try_new(vec![vec![1], vec![0]]).is_err());
        let plant = Plant::original();
        assert_eq!(plant.dim(), 2);
        assert_eq!(plant.fitness(&[10.9, 32.9]), -4802.0);
        assert_eq!(plant.fitness(&[-2.0, 80.0]), -4319.0);
    }

    #[test]
    fn benchmark_candidates_are_bounded_and_repeatable() {
        assert_eq!(benchmark_candidate(0), [4.0, 8.0]);
        assert_eq!(benchmark_candidate(50), benchmark_candidate(0));
        for i in 0..100 {
            assert!(
                benchmark_candidate(i)
                    .iter()
                    .all(|value| (1.0..=50.0).contains(value))
            );
        }
    }
}
