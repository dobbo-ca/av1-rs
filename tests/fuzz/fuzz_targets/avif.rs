#![no_main]
//! `avif`: `decode_avif` on a whole file (isobmff-rs item graph, av1C, grid assembly, alpha).
//! The last byte holds the options (bit 1: two threads, bit 3: `apply_transforms`).

use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

#[path = "../common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some((file, opt)) = common::split(data) else {
        return;
    };
    let limits = av1::Limits {
        max_alloc: 256 << 20,
        max_threads: if opt & 2 != 0 { 2 } else { 1 },
        max_width: 4096,
        max_height: 4096,
        ..av1::Limits::default()
    };
    let _ = av1::decode_avif(
        file,
        &av1::Options {
            apply_transforms: opt & 8 != 0,
            ..av1::Options::default()
        },
        &limits,
        &AtomicBool::new(false),
    );
});
