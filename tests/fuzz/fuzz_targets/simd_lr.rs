#![no_main]
//! `simd_lr`: the loop restoration kernels, every SIMD tier against scalar.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| av1::fuzzing::lr(data));
