//! Population evaluation on the task's own threads (`inner_threads`).
//!
//! Each outer worker thread keeps one pool of `inner_threads` threads for
//! the tasks it runs, so `outer_workers` concurrent tasks use at most
//! `outer_workers * inner_threads` evaluation threads, the job's reserved
//! `cpu_slots`, and no task shares threads with another. Every
//! population is evaluated in parallel and collected in input order; the
//! optimizer's own state stays serial, so a run gives the same result with
//! any number of threads.
use optimization_job_api::Result;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    /// The pool of the tasks this (outer worker) thread runs, kept between
    /// its tasks; it ends with the thread.
    static POOL: RefCell<Option<(usize, Arc<ThreadPool>)>> = const { RefCell::new(None) };
}

/// The evaluation threads of one task: none for `inner_threads = 1`.
pub struct Threads(Option<Arc<ThreadPool>>);

impl Threads {
    /// `count` threads for the task running on the calling thread. Tasks of
    /// the same outer worker thread share one pool, one task at a time.
    pub fn new(count: usize) -> Result<Self> {
        if count <= 1 {
            return Ok(Self(None));
        }
        POOL.with(|cached| {
            let mut cached = cached.borrow_mut();
            if let Some((n, pool)) = cached.as_ref()
                && *n == count
            {
                return Ok(Self(Some(Arc::clone(pool))));
            }
            let pool = Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(count)
                    .thread_name(|i| format!("fcmaes-task-{i}"))
                    .build()?,
            );
            *cached = Some((count, Arc::clone(&pool)));
            Ok(Self(Some(pool)))
        })
    }
    /// `f` applied to every item, in order.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        match &self.0 {
            None => items.iter().map(f).collect(),
            Some(pool) => pool.install(|| items.par_iter().map(f).collect()),
        }
    }
    /// Runs `f` inside the pool, where fcmaes-core code given
    /// [`Threads::core_workers`] evaluates on these threads.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.0 {
            None => f(),
            Some(pool) => pool.install(f),
        }
    }
    /// The worker count for fcmaes-core inside [`Threads::install`]: `1`
    /// evaluates serially, `0` on the current (this task's) pool.
    pub fn core_workers(&self) -> i32 {
        if self.0.is_some() { 0 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_keeps_order_and_runs_on_the_task_pool() {
        let items: Vec<u64> = (0..1000).collect();
        let threads = Threads::new(3).unwrap();
        let names = threads.map(&items, |_| std::thread::current().name().map(String::from));
        assert!(
            names
                .iter()
                .all(|n| n.as_deref().is_some_and(|n| n.starts_with("fcmaes-task-")))
        );
        assert_eq!(
            threads.map(&items, |i| i * 2),
            Threads::new(1).unwrap().map(&items, |i| i * 2)
        );
        assert_eq!(threads.install(rayon::current_num_threads), 3);
        assert_eq!(threads.core_workers(), 0);
        assert_eq!(Threads::new(1).unwrap().core_workers(), 1);
    }
}
