//! AVX2 CDF adaptation (see `super::adapt_cdf`). It needs only SSE4.1 instructions on 128-bit
//! vectors; the AVX2 target feature gives them the VEX encoding.

use crate::simd::x86::{load64_u16, load128_u16, store64_u16, store128_u16};
use std::arch::x86_64::*;

/// The update of spec 8.2.6 on the lanes of `c`, whose first lane is value `first` of the
/// row: lanes at or above `sym` move up towards 32768, the others down towards 0, by `rate`
/// (a shift count vector). The arithmetic wraps as the scalar code does.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(c: __m128i, first: i16, sym: i16, rate: __m128i) -> __m128i {
    let idx = _mm_add_epi16(
        _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7),
        _mm_set1_epi16(first),
    );
    let up = _mm_cmpgt_epi16(idx, _mm_set1_epi16(sym - 1));
    let down = _mm_sub_epi16(c, _mm_srl_epi16(c, rate));
    // 0x8000 - c is 32768 - c, wrapped.
    let gap = _mm_sub_epi16(_mm_set1_epi16(i16::MIN), c);
    let up_value = _mm_add_epi16(c, _mm_srl_epi16(gap, rate));
    _mm_blendv_epi8(down, up_value, up)
}

/// Adapts the 4 to 16 values of `cdf` for `sym` (below the length) at `rate`.
#[target_feature(enable = "avx2")]
pub(super) fn adapt(cdf: &mut [u16], sym: usize, rate: u32) {
    let n = cdf.len();
    let sym = sym as i16;
    let rate = _mm_cvtsi32_si128(rate as i32);
    if n >= 8 {
        let (Some(a), Some(b)) = (cdf.first_chunk::<8>(), cdf.last_chunk::<8>()) else {
            return;
        };
        let a = lanes(load128_u16(a), 0, sym, rate);
        let b = lanes(load128_u16(b), (n - 8) as i16, sym, rate);
        if let Some(d) = cdf.first_chunk_mut::<8>() {
            store128_u16(d, a);
        }
        if let Some(d) = cdf.last_chunk_mut::<8>() {
            store128_u16(d, b);
        }
    } else {
        let (Some(a), Some(b)) = (cdf.first_chunk::<4>(), cdf.last_chunk::<4>()) else {
            return;
        };
        let a = lanes(load64_u16(a), 0, sym, rate);
        let b = lanes(load64_u16(b), (n - 4) as i16, sym, rate);
        if let Some(d) = cdf.first_chunk_mut::<4>() {
            store64_u16(d, a);
        }
        if let Some(d) = cdf.last_chunk_mut::<4>() {
            store64_u16(d, b);
        }
    }
}
