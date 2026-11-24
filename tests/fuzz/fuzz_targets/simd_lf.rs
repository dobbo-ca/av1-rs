#![no_main]
//! `simd_lf`: the loop filter kernels of every SIMD tier against the scalar oracle.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::lf(data);
});
