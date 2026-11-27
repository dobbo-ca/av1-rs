//! AVX2 CDEF kernels (also the AVX-512 tier, except its filter of blocks of 32 or more samples,
//! see `avx512.rs`). The filter follows dav1d's
//! padded-block design (BSD-2-Clause, Copyright © 2018, VideoLAN and dav1d authors, Copyright
//! © 2018, Two Orioles, LLC; see NOTICE and `mod.rs`).

use super::{Filt, at4, at8};
use crate::simd::x86::{load64_u16, load128_u16, store256_i32, store256_u16};
use std::arch::x86_64::*;

/// 16 samples of the window that starts at `s` in `t` (row stride `st`): two rows of 8
/// (`w == 8`) or four rows of 4.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld(t: &[u16], s: usize, w: usize, st: usize) -> __m256i {
    let (a, b) = if w == 8 {
        (load128_u16(at8(t, s)), load128_u16(at8(t, s + st)))
    } else {
        (
            _mm_unpacklo_epi64(load64_u16(at4(t, s)), load64_u16(at4(t, s + st))),
            _mm_unpacklo_epi64(
                load64_u16(at4(t, s + 2 * st)),
                load64_u16(at4(t, s + 3 * st)),
            ),
        )
    };
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(a), b)
}

/// `constrain(p - px, thr, damping)` (spec 7.15.3) with `sh` = `damping_adj`; a sentinel `p`
/// gives 0.
#[inline]
#[target_feature(enable = "avx2")]
fn constrain(p: __m256i, px: __m256i, thr: __m256i, sh: __m128i) -> __m256i {
    let diff = _mm256_subs_epi16(p, px);
    let mag = _mm256_abs_epi16(diff);
    let v = _mm256_min_epu16(mag, _mm256_subs_epu16(thr, _mm256_srl_epi16(mag, sh)));
    _mm256_sign_epi16(v, diff)
}

/// The filtered block, `f.w` x `f.h` samples packed in `out` (row stride `f.w`), from `t` (the
/// padded block or the source plane); true when every sample it loaded is below 4096.
#[target_feature(enable = "avx2")]
pub(super) fn filter(t: &[u16], out: &mut [u16; 64], f: &Filt) -> bool {
    let zero = _mm256_setzero_si256();
    let mut or = zero;
    let pri_thr = _mm256_set1_epi16(f.pri_str);
    let sec_thr = _mm256_set1_epi16(f.sec_str);
    let pri_sh = _mm_cvtsi32_si128(f.pri_shift);
    let sec_sh = _mm_cvtsi32_si128(f.sec_shift);
    let pri_w = [
        _mm256_set1_epi16(f.pri_tap[0]),
        _mm256_set1_epi16(f.pri_tap[1]),
    ];
    let rows = 16 / f.w;
    for (n, dst) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .take(f.w * f.h / 16)
        .enumerate()
    {
        let o = n * rows * f.stride;
        let px = ld(t, f.center + o, f.w, f.stride);
        or = _mm256_or_si256(or, px);
        let (mut sum, mut mn, mut mx) = (zero, px, px);
        for (q, &s) in f.pri.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = _mm256_or_si256(or, p);
            let c = constrain(p, px, pri_thr, pri_sh);
            sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(c, pri_w[q / 2]));
            mn = _mm256_min_epu16(mn, p);
            mx = _mm256_max_epi16(mx, p);
        }
        for (q, &s) in f.sec.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = _mm256_or_si256(or, p);
            let c = constrain(p, px, sec_thr, sec_sh);
            // Weight 2 for k = 0, 1 for k = 1.
            let c = if q < 4 { _mm256_add_epi16(c, c) } else { c };
            sum = _mm256_add_epi16(sum, c);
            mn = _mm256_min_epu16(mn, p);
            mx = _mm256_max_epi16(mx, p);
        }
        // (8 + sum - (sum < 0)) >> 4
        let neg = _mm256_cmpgt_epi16(zero, sum);
        let sum = _mm256_add_epi16(_mm256_add_epi16(sum, neg), _mm256_set1_epi16(8));
        let v = _mm256_add_epi16(px, _mm256_srai_epi16::<4>(sum));
        store256_u16(dst, _mm256_min_epi16(_mm256_max_epi16(v, mn), mx));
    }
    // No sample has a bit at or above bit 12 (0xf000 is -4096 as i16).
    _mm256_testz_si256(or, _mm256_set1_epi16(-4096)) != 0
}

/// Adds the 8 rows of `v`, row `n` moved up by `n` lanes, into a 16-lane (lo, hi) partial.
#[inline]
#[target_feature(enable = "avx2")]
fn diag(v: [__m128i; 8]) -> (__m128i, __m128i) {
    let (mut lo, mut hi) = (v[0], _mm_setzero_si128());
    macro_rules! up {
        ($($n:literal)*) => {$(
            lo = _mm_add_epi16(lo, _mm_slli_si128::<{ 2 * $n }>(v[$n]));
            hi = _mm_add_epi16(hi, _mm_srli_si128::<{ 16 - 2 * $n }>(v[$n]));
        )*};
    }
    up!(1 2 3 4 5 6 7);
    (lo, hi)
}

/// `sum(v[i]^2 * w[i])` per lane, as 8 x i32.
#[inline]
#[target_feature(enable = "avx2")]
fn sq(v: __m128i, w: __m256i) -> __m256i {
    let a = _mm256_cvtepi16_epi32(v);
    _mm256_mullo_epi32(_mm256_mullo_epi32(a, a), w)
}

/// [`sq`] of a 16-lane (lo, hi) partial.
#[inline]
#[target_feature(enable = "avx2")]
fn sq2((lo, hi): (__m128i, __m128i), wlo: __m256i, whi: __m256i) -> __m256i {
    _mm256_add_epi32(sq(lo, wlo), sq(hi, whi))
}

/// The eight direction costs of spec 7.15.2 (`cost[ d ]`), for the 8x8 block `rows`.
#[target_feature(enable = "avx2")]
pub(super) fn dir_costs(rows: &[[u16; 8]; 8], shift: u32) -> [i32; 8] {
    let z = _mm_setzero_si128();
    let sh = _mm_cvtsi32_si128(shift as i32);
    let k128 = _mm_set1_epi16(128);
    let rev = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    let mut x = [z; 8];
    let mut rx = [z; 8];
    for i in 0..8 {
        x[i] = _mm_sub_epi16(_mm_srl_epi16(load128_u16(&rows[i]), sh), k128);
        rx[i] = _mm_shuffle_epi8(x[i], rev);
    }
    // partial[2][i]: row sums; partial[6][j]: column sums.
    let p2 = _mm_hadd_epi16(
        _mm_hadd_epi16(_mm_hadd_epi16(x[0], x[1]), _mm_hadd_epi16(x[2], x[3])),
        _mm_hadd_epi16(_mm_hadd_epi16(x[4], x[5]), _mm_hadd_epi16(x[6], x[7])),
    );
    let mut p6 = x[0];
    for i in 1..8 {
        p6 = _mm_add_epi16(p6, x[i]);
    }
    // partial[0][i + j], partial[4][7 + i - j] (row reversed: i + 7 - j).
    let p0 = diag(x);
    let p4 = diag(rx);
    // partial[1][i + j / 2], partial[3][3 + i - j / 2]: pair sums of each row (reversed).
    let mut ps = [z; 8];
    let mut rps = [z; 8];
    for i in 0..8 {
        ps[i] = _mm_hadd_epi16(x[i], z);
        rps[i] = _mm_hadd_epi16(rx[i], z);
    }
    let p1 = diag(ps);
    let p3 = diag(rps);
    // partial[7][i / 2 + j], partial[5][3 - i / 2 + j]: row pair sums moved by i / 2.
    let rp: [__m128i; 4] = [
        _mm_add_epi16(x[0], x[1]),
        _mm_add_epi16(x[2], x[3]),
        _mm_add_epi16(x[4], x[5]),
        _mm_add_epi16(x[6], x[7]),
    ];
    let p7 = diag([rp[0], rp[1], rp[2], rp[3], z, z, z, z]);
    let p5 = diag([rp[3], rp[2], rp[1], rp[0], z, z, z, z]);
    // Div_Table weights per partial index (lanes 0..8 in lo, 8..16 in hi).
    let we_lo = _mm256_setr_epi32(840, 420, 280, 210, 168, 140, 120, 105);
    let we_hi = _mm256_setr_epi32(120, 140, 168, 210, 280, 420, 840, 0);
    let wo_lo = _mm256_setr_epi32(420, 210, 140, 105, 105, 105, 105, 105);
    let wo_hi = _mm256_setr_epi32(140, 210, 420, 0, 0, 0, 0, 0);
    let w105 = _mm256_set1_epi32(105);
    let c = [
        sq2(p0, we_lo, we_hi),
        sq2(p1, wo_lo, wo_hi),
        sq(p2, w105),
        sq2(p3, wo_lo, wo_hi),
        sq2(p4, we_lo, we_hi),
        sq2(p5, wo_lo, wo_hi),
        sq(p6, w105),
        sq2(p7, wo_lo, wo_hi),
    ];
    // Horizontal sums: [c0, c1, c2, c3 | c4, c5, c6, c7].
    let h0 = _mm256_hadd_epi32(_mm256_hadd_epi32(c[0], c[1]), _mm256_hadd_epi32(c[2], c[3]));
    let h1 = _mm256_hadd_epi32(_mm256_hadd_epi32(c[4], c[5]), _mm256_hadd_epi32(c[6], c[7]));
    let s0 = _mm_add_epi32(
        _mm256_castsi256_si128(h0),
        _mm256_extracti128_si256::<1>(h0),
    );
    let s1 = _mm_add_epi32(
        _mm256_castsi256_si128(h1),
        _mm256_extracti128_si256::<1>(h1),
    );
    let mut cost = [0i32; 8];
    store256_i32(&mut cost, _mm256_set_m128i(s1, s0));
    cost
}
