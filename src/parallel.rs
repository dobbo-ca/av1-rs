//! The work scheduler of one call: `n` independent items run on at most `max_threads` threads,
//! the calling thread included, from one `std::thread::scope`. Used by the tile decode of a
//! frame (`decoder.rs`) and by the AVIF grid (`avif.rs`).

use crate::error::Error;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    std::thread::scope(|s| {
        for _ in 1..workers {
            // A thread that cannot start is not an error: the others take its items.
            let _ = std::thread::Builder::new().spawn_scoped(s, work);
        }
        work();
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
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
}
