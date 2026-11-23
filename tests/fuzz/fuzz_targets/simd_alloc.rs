#![no_main]
//! `simd_alloc`: the decoder that reuses its buffers and recycled frames against one that
//! reuses nothing (`av1::fuzzing::alloc`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::alloc(data);
});
