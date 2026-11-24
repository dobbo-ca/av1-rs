#![no_main]
//! `simd_ipred`: the ipred kernels (intra prediction, palette, CFL, intra edge filter) on every
//! SIMD tier against scalar (`av1::fuzzing::ipred`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::ipred(data);
});
