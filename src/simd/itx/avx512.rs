//! AVX-512 `itx` kernels: 16 `i32` lanes (see `itx_kernels!` in mod.rs). The dispatch uses
//! them for blocks of at least 16x16 only.

use crate::simd::x86::{load256_u16, load512_i32, store256_u16, store512_i32};
use std::arch::x86_64::*;

type V = __m512i;
const L: usize = 16;

#[inline]
#[target_feature(enable = "avx512f")]
fn vsplat(x: i32) -> V {
    _mm512_set1_epi32(x)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vadd(a: V, b: V) -> V {
    _mm512_add_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vsub(a: V, b: V) -> V {
    _mm512_sub_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vmul(a: V, b: V) -> V {
    _mm512_mullo_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vand(a: V, b: V) -> V {
    _mm512_and_si512(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vmin(a: V, b: V) -> V {
    _mm512_min_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vmax(a: V, b: V) -> V {
    _mm512_max_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vsra(a: V, n: u32) -> V {
    _mm512_sra_epi32(a, _mm_cvtsi32_si128(n as i32))
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vload(a: &[i32; L]) -> V {
    load512_i32(a)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn vstore(a: &mut [i32; L], v: V) {
    store512_i32(a, v)
}
#[inline]
#[target_feature(enable = "avx512f")]
fn px_load(a: &[u16; L]) -> V {
    _mm512_cvtepu16_epi32(load256_u16(a))
}
/// `v` holds values in `0..=65535`, so the truncating narrow is exact.
#[inline]
#[target_feature(enable = "avx512f")]
fn px_store(a: &mut [u16; L], v: V) {
    store256_u16(a, _mm512_cvtepi32_epi16(v))
}

itx_kernels!("avx512f");
