#![no_main]
//! `simd_cdef`: the CDEF direction and filter kernels of every SIMD tier against the scalar
//! oracle (`av1::fuzzing::cdef`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    av1::fuzzing::cdef(data);
});
