//! The work schedulers of one call, both from one `std::thread::scope` and within
//! `max_threads` threads, the calling thread included: [`run_parallel`] runs `n` independent
//! items (the AVIF grid, `avif.rs`); [`with_team`] keeps a team of threads for a sequence of
//! stages, each a set of independent items (the tiles and the in-loop filters of a frame,
//! `decoder.rs`).
//!
//! With the `rayon` feature the threads of [`run_parallel`] come from a shared rayon pool and the
//! frame decode does not use [`with_team`] (`decoder/pool_wrap.rs`).
#![cfg_attr(feature = "rayon", allow(dead_code))]

use crate::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

/// Runs `f( 0 )` .. `f( n - 1 )` on `min( max_threads, n )` threads (0 counts as 1) and returns
/// the results in item order. With one thread (`max_threads == 1` or `n <= 1`) nothing is
/// spawned and the items run in order on the calling thread.
///
/// Items are taken in increasing order from a shared counter. After a failure at index `k` no item
/// above `k` is started, but every item below it still runs, so the error returned is always the
/// one of the lowest failing index, whatever the timing.
///
/// # Errors
/// The error of the lowest failing item.
pub(crate) fn run_parallel<T: Send>(
    n: usize,
    max_threads: usize,
    f: impl Fn(usize) -> Result<T, Error> + Sync,
) -> Result<Vec<T>, Error> {
    let workers = max_threads.max(1).min(n);
    if workers <= 1 {
        return (0..n).map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    // The lowest index that failed so far.
    let failed = AtomicUsize::new(usize::MAX);
    let slots: Vec<Mutex<Option<Result<T, Error>>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n || i > failed.load(Ordering::Relaxed) {
                break;
            }
            let r = f(i);
            if r.is_err() {
                failed.fetch_min(i, Ordering::Relaxed);
            }
            if let Some(slot) = slots.get(i) {
                *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            }
        }
    };
    spread(workers - 1, &work);
    let mut out = Vec::with_capacity(n);
    for slot in slots {
        match slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some(Ok(v)) => out.push(v),
            Some(Err(e)) => return Err(e),
            None => return Err(Error::Invalid("work item skipped")),
        }
    }
    Ok(out)
}

/// Runs `work` on the calling thread and on up to `extra` more, and returns when all are done.
/// With the `rayon` feature the others are jobs of a pool: the caller's own when the call is
/// already on a rayon thread (so a caller's pool wins, and so does ours when calls nest), else
/// the lazy shared pool. A job that never starts is not an error: `work` takes items from a
/// shared counter, so the others take its items. A call never holds a lock across this.
fn spread(extra: usize, work: &(impl Fn() + Sync)) {
    #[cfg(feature = "rayon")]
    {
        if rayon_core::current_thread_index().is_some() {
            let extra = extra.min(rayon_core::current_num_threads());
            rayon_core::in_place_scope(|s| {
                for _ in 0..extra {
                    s.spawn(move |_| work());
                }
                work();
            });
            return;
        }
        if let Some(pool) = pool() {
            let extra = extra.min(pool.current_num_threads());
            pool.in_place_scope(|s| {
                for _ in 0..extra {
                    s.spawn(move |_| work());
                }
                work();
            });
            return;
        }
    }
    std::thread::scope(|s| {
        for _ in 0..extra {
            // A thread that cannot start is not an error: the others take its items.
            let _ = std::thread::Builder::new().spawn_scoped(s, work);
        }
        work();
    });
}

/// The lazy shared pool: `available_parallelism` - 1 threads (the calling thread works too),
/// made on first use and never grown. `None` when it cannot be built; the caller then spawns
/// scoped threads.
#[cfg(feature = "rayon")]
fn pool() -> Option<&'static rayon_core::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon_core::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        rayon_core::ThreadPoolBuilder::new()
            .num_threads(n.saturating_sub(1).max(1))
            .thread_name(|i| format!("av1-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// The threads of one call, shared by its stages (docs/threading.md): made by [`with_team`],
/// which spawns them once in one `std::thread::scope`; each [`Team::run`] hands them the items
/// of one stage and returns when the stage is done.
pub(crate) struct Team<'a, J> {
    /// `item( job, i )` runs item `i` of the stage `job`.
    item: &'a (dyn Fn(J, usize) -> Result<(), Error> + Sync),
    /// Threads of the team, the calling thread included.
    threads: usize,
    state: Mutex<TeamState<J>>,
    /// Signalled when a stage starts or the team stops.
    wake: Condvar,
    /// Signalled when the last thread leaves a stage.
    idle: Condvar,
    /// The next item of the current stage.
    next: AtomicUsize,
    /// The lowest item of the current stage that failed so far.
    failed: AtomicUsize,
}

struct TeamState<J> {
    /// The current stage and its item count; `None` between stages.
    job: Option<(J, usize)>,
    /// Counts the stages, so a worker joins each stage at most once.
    stage: u64,
    /// Threads working on the current stage.
    busy: usize,
    /// The lowest failing item of the current stage and its error.
    error: Option<(usize, Error)>,
    quit: bool,
}

/// Runs `lead( team )` with a team of `threads` threads (0 counts as 1), the calling thread
/// included: the others are spawned once, in one `std::thread::scope`, and wait for the stages
/// `lead` runs with [`Team::run`]. With one thread nothing is spawned and every stage runs on
/// the calling thread. A thread that cannot be spawned is not an error: the others take its
/// items.
pub(crate) fn with_team<J: Copy + Send, R>(
    threads: usize,
    item: &(dyn Fn(J, usize) -> Result<(), Error> + Sync),
    lead: impl FnOnce(&Team<'_, J>) -> R,
) -> R {
    let team = Team {
        item,
        threads: threads.max(1),
        state: Mutex::new(TeamState {
            job: None,
            stage: 0,
            busy: 0,
            error: None,
            quit: false,
        }),
        wake: Condvar::new(),
        idle: Condvar::new(),
        next: AtomicUsize::new(0),
        failed: AtomicUsize::new(usize::MAX),
    };
    if team.threads == 1 {
        return lead(&team);
    }
    std::thread::scope(|s| {
        for _ in 1..team.threads {
            let _ = std::thread::Builder::new().spawn_scoped(s, || team.serve());
        }
        // Stops the workers when `lead` returns (or unwinds), so the scope can join them.
        let stop = Stop(&team);
        let r = lead(&team);
        drop(stop);
        r
    })
}

impl<J: Copy> Team<'_, J> {
    /// Runs `item( job, 0 )` .. `item( job, n - 1 )` on the team and returns when all have
    /// finished, with the same error rule as [`run_parallel`]: items are taken in increasing
    /// order, none above a failed one starts, and the error of the lowest failing item is
    /// returned. With one thread, or one item, they run in order on the calling thread.
    ///
    /// # Errors
    /// The error of the lowest failing item.
    pub(crate) fn run(&self, job: J, n: usize) -> Result<(), Error> {
        if self.threads == 1 || n <= 1 {
            for i in 0..n {
                (self.item)(job, i)?;
            }
            return Ok(());
        }
        {
            // No thread is inside `work` here: the last run waited for `busy == 0`.
            let mut st = lock(&self.state);
            self.next.store(0, Ordering::Relaxed);
            self.failed.store(usize::MAX, Ordering::Relaxed);
            st.error = None;
            st.job = Some((job, n));
            st.stage = st.stage.wrapping_add(1);
            st.busy += 1;
        }
        self.wake.notify_all();
        {
            let _busy = Busy(self);
            self.work(job, n);
        }
        let mut st = lock(&self.state);
        while st.busy > 0 {
            st = self.idle.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = None;
        match st.error.take() {
            Some((_, e)) => Err(e),
            None => Ok(()),
        }
    }

    /// The loop of a spawned thread: joins every stage it sees until the team stops.
    fn serve(&self) {
        let mut seen = 0;
        loop {
            let (job, n) = {
                let mut st = lock(&self.state);
                loop {
                    if st.quit {
                        return;
                    }
                    if st.stage != seen {
                        seen = st.stage;
                        if let Some(j) = st.job {
                            st.busy += 1;
                            break j;
                        }
                    }
                    st = self.wake.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            };
            let _busy = Busy(self);
            self.work(job, n);
        }
    }

    /// Takes items of the current stage until none is left (or one below failed).
    fn work(&self, job: J, n: usize) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= n || i > self.failed.load(Ordering::Relaxed) {
                break;
            }
            if let Err(e) = (self.item)(job, i) {
                self.failed.fetch_min(i, Ordering::Relaxed);
                let mut st = lock(&self.state);
                if st.error.as_ref().is_none_or(|(k, _)| i < *k) {
                    st.error = Some((i, e));
                }
            }
        }
    }
}

/// A thread's place in the current stage, given back on drop (also when an item panics, so
/// [`Team::run`] never waits for a thread that is gone).
struct Busy<'t, 'a, J>(&'t Team<'a, J>);

impl<J> Drop for Busy<'_, '_, J> {
    fn drop(&mut self) {
        let mut st = lock(&self.0.state);
        st.busy -= 1;
        if st.busy == 0 {
            self.0.idle.notify_all();
        }
    }
}

/// Stops the spawned threads of a team when dropped.
struct Stop<'t, 'a, J>(&'t Team<'a, J>);

impl<J> Drop for Stop<'_, '_, J> {
    fn drop(&mut self) {
        lock(&self.0.state).quit = true;
        self.0.wake.notify_all();
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::thread::ThreadId;
    use std::time::Duration;

    /// Runs `n` items that each hold a "live" count for a moment; returns (results, peak live,
    /// distinct threads).
    fn run(n: usize, max_threads: usize) -> (Vec<usize>, usize, usize) {
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let ids: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
        let out = run_parallel(n, max_threads, |i| {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            ids.lock().unwrap().insert(std::thread::current().id());
            std::thread::sleep(Duration::from_millis(5));
            live.fetch_sub(1, Ordering::SeqCst);
            Ok(i * 10)
        })
        .unwrap();
        let threads = ids.lock().unwrap().len();
        (out, peak.load(Ordering::SeqCst), threads)
    }

    #[test]
    fn peak_threads_stay_within_max_threads_and_results_are_in_order() {
        for max_threads in [1, 2, 3] {
            let (out, peak, threads) = run(8, max_threads);
            assert_eq!(out, (0..8).map(|i| i * 10).collect::<Vec<_>>());
            assert!(peak <= max_threads, "peak {peak} > {max_threads}");
            assert!(threads <= max_threads);
        }
    }

    #[test]
    fn one_thread_spawns_nothing() {
        let me = std::thread::current().id();
        let out = run_parallel(5, 1, |i| {
            assert_eq!(std::thread::current().id(), me);
            Ok(i)
        })
        .unwrap();
        assert_eq!(out, [0, 1, 2, 3, 4]);
        // max_threads 0 counts as 1, and a single item never spawns.
        run_parallel(3, 0, |_| {
            assert_eq!(std::thread::current().id(), me);
            Ok(())
        })
        .unwrap();
        run_parallel(1, 8, |_| {
            assert_eq!(std::thread::current().id(), me);
            Ok(())
        })
        .unwrap();
    }

    /// A call inside a call, as the AVIF grid and the frames of a batch do: the outer call gives
    /// each inner one `k / workers` threads, and the threads busy at any moment stay within `k`
    /// (with a pool, a thread waiting in the inner call may run a job of another, so count the
    /// threads that run a leaf, not the ones that exist).
    #[test]
    fn nested_calls_peak_within_max_threads() {
        for k in [1, 2, 3, 8] {
            let live: Mutex<HashMap<ThreadId, usize>> = Mutex::new(HashMap::new());
            let peak = AtomicUsize::new(0);
            let me = std::thread::current().id();
            let outer = k.min(2);
            run_parallel(2, k, |_| {
                run_parallel(6, (k / outer).max(1), |_| {
                    let id = std::thread::current().id();
                    if k == 1 {
                        assert_eq!(id, me);
                        #[cfg(feature = "rayon")]
                        assert!(rayon_core::current_thread_index().is_none());
                    }
                    {
                        let mut m = live.lock().unwrap();
                        *m.entry(id).or_insert(0) += 1;
                        peak.fetch_max(m.len(), Ordering::SeqCst);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                    let mut m = live.lock().unwrap();
                    if let Some(c) = m.get_mut(&id) {
                        *c -= 1;
                        if *c == 0 {
                            m.remove(&id);
                        }
                    }
                    Ok(())
                })
            })
            .unwrap();
            assert!(peak.load(Ordering::SeqCst) <= k, "k {k}");
        }
    }

    #[test]
    fn no_items_is_empty() {
        assert!(run_parallel(0, 4, |_| Ok(())).unwrap().is_empty());
    }

    #[test]
    fn lowest_failing_index_wins() {
        for max_threads in [1, 2, 3, 8] {
            let r = run_parallel(8, max_threads, |i| match i {
                // The later failure comes first in time.
                2 => {
                    std::thread::sleep(Duration::from_millis(30));
                    Err(Error::Invalid("two"))
                }
                5 => Err(Error::Truncated),
                _ => Ok(i),
            });
            assert_eq!(r.unwrap_err(), Error::Invalid("two"));
        }
    }

    #[test]
    fn charged_allocations_stay_within_the_ledger() {
        use crate::ledger::{Ledger, SMALL};
        // Each of 4 items books 100 bytes while it runs; at most 2 run at once.
        let book = |max: usize| {
            let ledger = Ledger::default();
            ledger.reset(max, 0).unwrap();
            let r = run_parallel(4, 2, |_| {
                let held = ledger.charge(100)?;
                std::thread::sleep(Duration::from_millis(10));
                drop(held);
                Ok(())
            });
            assert!(ledger.peak() <= max as u64);
            assert_eq!(ledger.left(), max - SMALL);
            r
        };
        assert!(book(SMALL + 200).is_ok());
        assert!(book(SMALL + 1000).is_ok());
        assert_eq!(book(SMALL + 50).unwrap_err(), Error::AllocLimit);
        assert_eq!(book(SMALL).unwrap_err(), Error::AllocLimit);
    }

    #[test]
    fn team_stages_run_every_item_once_within_max_threads() {
        for threads in [1, 2, 3, 8] {
            let live = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let ids: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
            let done: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
            let item: &(dyn Fn(usize, usize) -> Result<(), Error> + Sync) = &|stage, i| {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                ids.lock().unwrap().insert(std::thread::current().id());
                std::thread::sleep(Duration::from_millis(2));
                done.lock().unwrap().push((stage, i));
                live.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            };
            with_team(threads, item, |team| {
                for stage in 0..4 {
                    team.run(stage, 7).unwrap();
                    // A stage is over when `run` returns: all its items, none of the next.
                    let mut d = done.lock().unwrap().clone();
                    d.sort_unstable();
                    let want: Vec<_> = (0..=stage)
                        .flat_map(|s| (0..7).map(move |i| (s, i)))
                        .collect();
                    assert_eq!(d, want);
                }
            });
            assert!(peak.load(Ordering::SeqCst) <= threads);
            assert!(ids.lock().unwrap().len() <= threads);
        }
    }

    #[test]
    fn team_of_one_runs_on_the_calling_thread() {
        let me = std::thread::current().id();
        let item: &(dyn Fn((), usize) -> Result<(), Error> + Sync) = &|_, _| {
            assert_eq!(std::thread::current().id(), me);
            Ok(())
        };
        for threads in [0, 1] {
            with_team(threads, item, |team| team.run((), 5)).unwrap();
        }
    }

    #[test]
    fn team_returns_the_lowest_failing_item_and_goes_on() {
        for threads in [1, 2, 3, 8] {
            let item: &(dyn Fn(usize, usize) -> Result<(), Error> + Sync) = &|stage, i| {
                match (stage, i) {
                    // The later failure comes first in time.
                    (0, 2) => {
                        std::thread::sleep(Duration::from_millis(30));
                        Err(Error::Invalid("two"))
                    }
                    (0, 5) => Err(Error::Truncated),
                    _ => Ok(()),
                }
            };
            with_team(threads, item, |team| {
                assert_eq!(team.run(0, 8), Err(Error::Invalid("two")));
                // The next stage starts clean.
                assert_eq!(team.run(1, 8), Ok(()));
                assert_eq!(team.run(1, 0), Ok(()));
            });
        }
    }
}
