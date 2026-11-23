#![no_main]
//! `decode`: the streaming decoder on every temporal unit. Option bits: 0 = Annex B (non-IVF
//! input), 1 = two threads, 2 = no film grain.

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
        max_threads: if opt & 2 != 0 { 2 } else { 1 },
        max_width: 4096,
        max_height: 4096,
        ..av1::Limits::default()
    };
    let options = av1::Options {
        film_grain: opt & 4 == 0,
        ..av1::Options::default()
    };
    let cancel = AtomicBool::new(false);
    let mut dec = av1::Decoder::new(options, limits);
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
    // Work budget: tile data past the end reads as zeros, so a few bytes can decode a whole
    // frame. The probe records every frame header (shown, hidden, every layer), so the budget
    // covers all of them: give up above 64M luma samples so one input stays under the timeout.
    // Broken framing gives no budget: skip the input rather than decode it unbounded.
    let Ok(p) = av1::probe(whole, format, &limits) else {
        return;
    };
    {
        let work: u64 = p
            .frames
            .iter()
            .map(|f| u64::from(f.width) * u64::from(f.height))
            .sum();
        if work > 64 << 20 {
            return;
        }
    }
    for u in units {
        if dec.decode_temporal_unit(u, format, &cancel).is_err() {
            break;
        }
    }
});
