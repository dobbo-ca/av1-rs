#![no_main]
//! `simd_grain`: the film grain blend kernels, every SIMD tier against scalar.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::grain(data);
});
