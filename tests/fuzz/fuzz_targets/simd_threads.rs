#![no_main]
//! `simd_threads`: the frame wrapup on a frame built from the input, with 1, 2 and 3 threads;
//! the outputs must be the same (`av1::fuzzing::threads`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::threads(data);
});
