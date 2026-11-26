#![no_main]
//! `frames_threads`: `Decoder::decode_temporal_units` with 1, 2, 3 and 8 threads against a loop of
//! `decode_temporal_unit` calls on one thread: the same frames (those before an error too) and the
//! same error, unless the ledger refused either side (`AllocLimit`), where only the frames up to
//! the refusal must agree. Option bits: 0 = Annex B (non-IVF input), 2 = no film grain.

use av1::Error;
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

#[path = "../common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some((stream, opt)) = common::split(data) else {
        return;
    };
    let limits = av1::Limits {
        max_alloc: 256 << 20,
        max_width: 4096,
        max_height: 4096,
        max_threads: 1,
        ..av1::Limits::default()
    };
    let options = av1::Options {
        film_grain: opt & 4 == 0,
        ..av1::Options::default()
    };
    let cancel = AtomicBool::new(false);
    let joined;
    let (units, format, whole): (Vec<&[u8]>, _, &[u8]) = match common::ivf_units(stream) {
        Some(u) => {
            joined = u.concat();
            (u, av1::Format::Section5, &joined)
        }
        None => {
            let format = common::format(opt);
            match av1::split_temporal_units(stream, format, &limits) {
                Ok(u) => (u, format, stream),
                Err(_) => return,
            }
        }
    };
    // The work budget of the `decode` target, divided by the five decodes of one input.
    let Ok(p) = av1::probe(whole, format, &limits) else {
        return;
    };
    let work: u64 = p
        .frames
        .iter()
        .map(|f| u64::from(f.width) * u64::from(f.height))
        .sum();
    if work > 12 << 20 {
        return;
    }
    let mut dec = av1::Decoder::new(options.clone(), limits);
    let mut want = Vec::new();
    let mut want_err = None;
    for u in &units {
        match dec.decode_temporal_unit(u, format, &cancel) {
            Ok(f) => want.extend(f),
            Err(e) => {
                want.extend(dec.take_frames());
                want_err = Some(e);
                break;
            }
        }
    }
    // A failed system allocation is the one error that may differ with the number of frames in
    // flight.
    let limit = |e: &Option<Error>| matches!(e, Some(Error::OutOfMemory));
    if limit(&want_err) {
        return;
    }
    for threads in [1, 2, 3, 8] {
        let limits = av1::Limits {
            max_threads: threads,
            ..limits
        };
        let mut dec = av1::Decoder::new(options.clone(), limits);
        let (got, got_err) = match dec.decode_temporal_units(&units, format, &cancel) {
            Ok(f) => (f, None),
            Err(e) => (dec.take_frames(), Some(e)),
        };
        if limit(&got_err) {
            continue;
        }
        // The batch keeps the frames of every unit booked until it returns, where the loop hands
        // each unit's frames back, and it decodes with more threads, which book more: it can run
        // out of `max_alloc` at an earlier unit than the loop does, or where the loop does not.
        // So when either side is refused by the ledger, the frames output up to there agree and
        // nothing else is required; any other difference is a bug.
        let refused = |e: &Option<Error>| matches!(e, Some(Error::AllocLimit));
        // Another batch error can only come where the loop got to: one before the loop's refusal
        // is a batch-only decode error.
        if refused(&want_err) && got_err.is_some() && !refused(&got_err) {
            assert!(
                got.len() >= want.len(),
                "{threads} threads: {got_err:?} before the loop's refusal"
            );
        }
        if refused(&got_err) || refused(&want_err) {
            let n = got.len().min(want.len());
            assert!(
                got[..n] == want[..n],
                "{threads} threads: the frames differ"
            );
            continue;
        }
        assert_eq!(got_err, want_err, "{threads} threads");
        assert!(got == want, "{threads} threads: the frames differ");
    }
});
