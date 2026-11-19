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
    let (units, format) = match common::ivf_units(stream) {
        Some(u) => (u, av1::Format::Section5),
        None => {
            let format = common::format(opt);
            match av1::split_temporal_units(stream, format) {
                Ok(u) => (u, format),
                Err(_) => return,
            }
        }
    };
    for u in units {
        if dec.decode_temporal_unit(u, format, &cancel).is_err() {
            break;
        }
    }
});
