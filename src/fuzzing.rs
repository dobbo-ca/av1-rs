//! Cross-tier checks for the fuzz targets in `tests/fuzz` (feature `fuzzing`, not for users).
//!
//! Each kernel family adds one `pub fn <family>(data: &[u8])`: it builds valid kernel inputs
//! from the bytes (any bytes give some input), runs the kernel on every tier of
//! `simd::available()` and asserts that each output equals the scalar output.

/// The number of SIMD tiers this CPU has, scalar included (a fuzz target can report it).
pub fn tiers() -> usize {
    crate::simd::available().len()
}

/// Loop filter (`simd::lf`): one edge run (bit depth, plane, filter size, direction, length,
/// level, sharpness, start and sample pattern from the bytes) on every tier against scalar.
pub fn lf(data: &[u8]) {
    crate::simd::lf::check(data);
}
