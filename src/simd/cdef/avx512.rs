//! AVX-512 CDEF filter: the AVX2 filter (`super::avx2::filter`, same padded-block design) on
//! 32 samples per vector, four rows of 8 or eight rows of 4. The arithmetic is AVX-512 BW; the
//! loads are the AVX2 kernel's (`avx2::ld`, two per vector). The dispatch uses it for blocks of
//! at least 32 samples (8x8, 8x4, 4x8).

use super::{Filt, avx2};
use crate::simd::x86::store512_u16;
use std::arch::x86_64::*;

/// 32 samples of the window that starts at `s` in `t` (row stride `st`): four rows of 8
/// (`w == 8`) or eight rows of 4.
#[inline]
#[target_feature(enable = "avx512bw")]
fn ld(t: &[u16], s: usize, w: usize, st: usize) -> __m512i {
    let a = avx2::ld(t, s, w, st);
    let b = avx2::ld(t, s + 16 / w * st, w, st);
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
}

/// `constrain(p - px, thr, damping)` (spec 7.15.3) with `sh` = `damping_adj`; a sentinel `p`
/// gives 0. AVX-512 has no `sign`: the magnitude is negated where `diff < 0` (where
/// `diff == 0` the magnitude is already 0).
#[inline]
#[target_feature(enable = "avx512bw")]
fn constrain(p: __m512i, px: __m512i, thr: __m512i, sh: __m128i) -> __m512i {
    let diff = _mm512_subs_epi16(p, px);
    let mag = _mm512_abs_epi16(diff);
    let v = _mm512_min_epu16(mag, _mm512_subs_epu16(thr, _mm512_srl_epi16(mag, sh)));
    _mm512_mask_sub_epi16(v, _mm512_movepi16_mask(diff), _mm512_setzero_si512(), v)
}

/// The filtered block, `f.w` x `f.h` samples packed in `out` (row stride `f.w`), from `t` (the
/// padded block or the source plane); `f.w * f.h >= 32`. True when every sample it loaded is
/// below 4096.
#[target_feature(enable = "avx512bw")]
pub(super) fn filter(t: &[u16], out: &mut [u16; 64], f: &Filt) -> bool {
    let zero = _mm512_setzero_si512();
    let mut or = zero;
    let pri_thr = _mm512_set1_epi16(f.pri_str);
    let sec_thr = _mm512_set1_epi16(f.sec_str);
    let pri_sh = _mm_cvtsi32_si128(f.pri_shift);
    let sec_sh = _mm_cvtsi32_si128(f.sec_shift);
    let pri_w = [
        _mm512_set1_epi16(f.pri_tap[0]),
        _mm512_set1_epi16(f.pri_tap[1]),
    ];
    let rows = 32 / f.w;
    for (n, dst) in out
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .take(f.w * f.h / 32)
        .enumerate()
    {
        let o = n * rows * f.stride;
        let px = ld(t, f.center + o, f.w, f.stride);
        or = _mm512_or_si512(or, px);
        let (mut sum, mut mn, mut mx) = (zero, px, px);
        for (q, &s) in f.pri.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = _mm512_or_si512(or, p);
            let c = constrain(p, px, pri_thr, pri_sh);
            sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(c, pri_w[q / 2]));
            mn = _mm512_min_epu16(mn, p);
            mx = _mm512_max_epi16(mx, p);
        }
        for (q, &s) in f.sec.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = _mm512_or_si512(or, p);
            let c = constrain(p, px, sec_thr, sec_sh);
            // Weight 2 for k = 0, 1 for k = 1.
            let c = if q < 4 { _mm512_add_epi16(c, c) } else { c };
            sum = _mm512_add_epi16(sum, c);
            mn = _mm512_min_epu16(mn, p);
            mx = _mm512_max_epi16(mx, p);
        }
        // (8 + sum - (sum < 0)) >> 4
        let neg = _mm512_movepi16_mask(sum);
        let sum = _mm512_mask_sub_epi16(sum, neg, sum, _mm512_set1_epi16(1));
        let sum = _mm512_add_epi16(sum, _mm512_set1_epi16(8));
        let v = _mm512_add_epi16(px, _mm512_srai_epi16::<4>(sum));
        store512_u16(dst, _mm512_min_epi16(_mm512_max_epi16(v, mn), mx));
    }
    // No sample has a bit at or above bit 12 (0xf000 is -4096 as i16).
    _mm512_test_epi16_mask(or, _mm512_set1_epi16(-4096)) == 0
}
