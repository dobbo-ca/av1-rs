#![no_main]
//! `simd_threads`: the frame wrapup on a frame built from the input, with 1, 2 and 3 threads;
//! the outputs must be the same (`av1::fuzzing::threads`).

use libfuzzer_sys::fuzz_target;

// DEBUG (temporary): RSS every 250 execs; AV1_DBG=spawn spawns 1 + 2 empty scoped threads
// and skips av1, AV1_DBG=serial runs the wrapup with 1 thread only (no spawn).
static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fuzz_target!(|data: &[u8]| {
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n % 250 == 0 {
        let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        let rss: u64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        eprintln!("DBG mode={:?} exec={n} rss_kb={}", std::env::var("AV1_DBG").ok(), rss * 4);
    }
    match std::env::var("AV1_DBG").as_deref() {
        Ok("spawn") => {
            for t in [2, 3] {
                std::thread::scope(|s| {
                    for _ in 1..t {
                        s.spawn(|| std::hint::black_box(data.len()));
                    }
                });
            }
        }
        Ok("serial") => av1::fuzzing::threads_dbg_serial(data),
        _ => av1::fuzzing::threads(data),
    }
});
