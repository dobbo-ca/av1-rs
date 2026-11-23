//! AVX2 `itx` kernels: 8 `i32` lanes (see `itx_kernels!` in mod.rs).

use crate::simd::x86::{load128_u16, load256_i32, store128_u16, store256_i32};
use std::arch::x86_64::*;

type V = __m256i;
const L: usize = 8;

#[inline]
#[target_feature(enable = "avx2")]
fn vsplat(x: i32) -> V {
    _mm256_set1_epi32(x)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vadd(a: V, b: V) -> V {
    _mm256_add_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vsub(a: V, b: V) -> V {
    _mm256_sub_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vmul(a: V, b: V) -> V {
    _mm256_mullo_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vand(a: V, b: V) -> V {
    _mm256_and_si256(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vmin(a: V, b: V) -> V {
    _mm256_min_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vmax(a: V, b: V) -> V {
    _mm256_max_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vsra(a: V, n: u32) -> V {
    _mm256_sra_epi32(a, _mm_cvtsi32_si128(n as i32))
}
#[inline]
#[target_feature(enable = "avx2")]
fn vload(a: &[i32; L]) -> V {
    load256_i32(a)
}
#[inline]
#[target_feature(enable = "avx2")]
fn vstore(a: &mut [i32; L], v: V) {
    store256_i32(a, v)
}
#[inline]
#[target_feature(enable = "avx2")]
fn px_load(a: &[u16; L]) -> V {
    _mm256_cvtepu16_epi32(load128_u16(a))
}
/// `v` holds values in `0..=65535`: pack with unsigned saturation (per 128-bit half), then
/// move the two low quadwords together.
#[inline]
#[target_feature(enable = "avx2")]
fn px_store(a: &mut [u16; L], v: V) {
    let p = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi32(v, v));
    store128_u16(a, _mm256_castsi256_si128(p))
}

itx_kernels!("avx2");
