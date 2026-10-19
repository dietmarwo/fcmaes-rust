//! The task's own evaluation threads (`inner_threads`).
//!
//! Each outer worker thread keeps one pool of `inner_threads` threads for
//! the tasks it runs, one task at a time. A task runs entirely inside that
//! pool, so every Rayon computation of the task (population evaluation, the
//! CVT construction, a model that uses Rayon itself) stays on it: with
//! `outer_workers` concurrent tasks, at most `outer_workers * inner_threads`
//! threads compute, the job's reserved `cpu_slots`, and Rayon's global pool
//! is never used. Populations are evaluated in parallel and collected in
//! input order; the optimizer's own state stays serial, so a run gives the
//! same result with any number of threads.
use optimization_job_api::Result;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    /// The pool of the tasks this (outer worker) thread runs, kept between
    /// its tasks; it ends with the thread.
    static POOL: RefCell<Option<(usize, Arc<ThreadPool>)>> = const { RefCell::new(None) };
}

/// The evaluation threads of one task.
pub struct Threads {
    pool: Arc<ThreadPool>,
    count: usize,
}

impl Threads {
    /// `count` threads (at least one) for the task running on the calling
    /// thread. Tasks of the same outer worker thread share one pool, one
    /// task at a time.
    pub fn new(count: usize) -> Result<Self> {
        let count = count.max(1);
        POOL.with(|cached| {
            let mut cached = cached.borrow_mut();
            if let Some((n, pool)) = cached.as_ref()
                && *n == count
            {
                return Ok(Self {
                    pool: Arc::clone(pool),
                    count,
                });
            }
            let pool = Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(count)
                    .thread_name(|i| format!("fcmaes-task-{i}"))
                    .build()?,
            );
            *cached = Some((count, Arc::clone(&pool)));
            Ok(Self { pool, count })
        })
    }
    /// `f` applied to every item, in order: in parallel with more than one
    /// thread, serially otherwise, always on the task's pool.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        if self.count > 1 {
            self.pool.install(|| items.par_iter().map(f).collect())
        } else {
            self.pool.install(|| items.iter().map(f).collect())
        }
    }
    /// Runs `f` inside the task's pool: Rayon code in `f`, and fcmaes-core
    /// code given [`Threads::core_workers`], computes on these threads.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pool.install(f)
    }
    /// The worker count for fcmaes-core inside [`Threads::install`]: `1`
    /// evaluates serially, `0` on the current (this task's) pool.
    pub fn core_workers(&self) -> i32 {
        if self.count > 1 { 0 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_keeps_order_and_runs_on_the_task_pool() {
        let items: Vec<u64> = (0..1000).collect();
        for count in [1, 3] {
            let threads = Threads::new(count).unwrap();
            let names = threads.map(&items, |_| std::thread::current().name().map(String::from));
            assert!(
                names
                    .iter()
                    .all(|n| n.as_deref().is_some_and(|n| n.starts_with("fcmaes-task-")))
            );
            // Rayon code of the task, the CVT construction for one, sees
            // only the task's threads, never the global pool.
            assert_eq!(threads.install(rayon::current_num_threads), count);
            assert_eq!(
                threads.map(&[()], |_| rayon::current_num_threads()),
                [count]
            );
            assert_eq!(threads.map(&items, |i| i * 2)[999], 1998);
        }
        assert_eq!(Threads::new(3).unwrap().core_workers(), 0);
        assert_eq!(Threads::new(1).unwrap().core_workers(), 1);
    }
}
