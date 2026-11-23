#![no_main]
//! `simd_entropy`: the symbol reader against the spec's bit by bit reader, and the CDF
//! adaptation of every SIMD tier against scalar (`av1::fuzzing::entropy`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::entropy(data);
});
