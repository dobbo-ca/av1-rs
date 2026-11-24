#![no_main]
//! `simd_itx`: the `itx` kernels (inverse transform + add) on every SIMD tier against scalar.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::itx(data);
});
