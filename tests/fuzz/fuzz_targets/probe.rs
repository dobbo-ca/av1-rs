#![no_main]
//! `probe`: OBU framing, sequence and frame headers, tile group headers.

use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some((stream, opt)) = common::split(data) else {
        return;
    };
    let limits = av1::Limits {
        max_alloc: 256 << 20,
        max_threads: 1,
        ..av1::Limits::default()
    };
    match common::ivf_units(stream) {
        Some(units) => {
            for u in units {
                let _ = av1::probe(u, av1::Format::Section5, &limits);
            }
        }
        None => {
            let _ = av1::probe(stream, common::format(opt), &limits);
        }
    }
});
