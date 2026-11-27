//! Kernel family `codegen`: the coefficient reader (`TileDecoder::coeffs`, with `read_levels`
//! and the symbol decoder inlined) and `intra_frame_mode_info` built a second time for
//! x86-64-v3 (AVX2, BMI1, BMI2, LZCNT, POPCNT, FMA). The code is the same safe Rust; only the
//! instructions differ: the renormalisation's `leading_zeros` becomes `lzcnt` and its shifts by
//! a variable count `shlx` / `shrx` (one micro-op each instead of `bsr` + `xor` and `shl %cl`).
//! The baseline build is the oracle and runs on the scalar tier (`AV1_SIMD=scalar`), on CPUs
//! without the AVX2 tier and on other targets.

use super::Simd;
#[cfg(target_arch = "x86_64")]
use super::Tier;
use crate::error::Error;
use crate::tile::TileDecoder;

/// True when `simd` is a tier whose detection includes every feature of `v3` (AVX-512
/// implies AVX2).
#[cfg(target_arch = "x86_64")]
fn is_v3(simd: Simd) -> bool {
    matches!(simd.tier(), Tier::Avx2 | Tier::Avx512)
}

/// `TileDecoder::coeffs` in the build of tier `simd`.
#[inline]
#[allow(unsafe_code)]
pub(crate) fn coeffs(
    simd: Simd,
    td: &mut TileDecoder<'_>,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_sz: usize,
) -> Result<(usize, usize, (usize, usize)), Error> {
    #[cfg(target_arch = "x86_64")]
    if is_v3(simd) {
        // SAFETY: `simd` is the proof (simd::available / simd::best) that the CPU has this
        // tier, and `is_v3` holds only for tiers whose detection includes every feature of
        // `v3::coeffs`.
        return unsafe { v3::coeffs(td, plane, start_x, start_y, tx_sz) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = simd;
    td.coeffs_body(plane, start_x, start_y, tx_sz)
}

/// `TileDecoder::intra_frame_mode_info` in the build of tier `simd`.
#[inline]
#[allow(unsafe_code)]
pub(crate) fn intra_frame_mode_info(simd: Simd, td: &mut TileDecoder<'_>) -> Result<(), Error> {
    #[cfg(target_arch = "x86_64")]
    if is_v3(simd) {
        // SAFETY: as in `coeffs`.
        return unsafe { v3::intra_frame_mode_info(td) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = simd;
    td.intra_frame_mode_info_body()
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use crate::error::Error;
    use crate::tile::TileDecoder;

    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt,fma")]
    pub(super) fn coeffs(
        td: &mut TileDecoder<'_>,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> Result<(usize, usize, (usize, usize)), Error> {
        td.coeffs_body(plane, start_x, start_y, tx_sz)
    }

    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt,fma")]
    pub(super) fn intra_frame_mode_info(td: &mut TileDecoder<'_>) -> Result<(), Error> {
        td.intra_frame_mode_info_body()
    }
}
