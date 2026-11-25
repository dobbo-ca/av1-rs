//! AVX2 loop restoration kernels (see `super` for the arithmetic): 8 samples per vector, in
//! 32-bit lanes. No `unsafe`: memory goes through `simd::x86` and safe slices.

use super::{BLOCK_H, BoxK, Mem, SgrK, WienerK, starts};
use crate::consts::{
    SGRPROJ_MTABLE_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS, SGRPROJ_SGR_BITS,
};
use crate::lr::BLOCK_W;
use crate::simd::x86::{load256_i32, store128_u16, store256_i32};
use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 8;
/// Row stride of the A and B arrays (spec `A[ i ][ j ]`, `j` in `-1..w+1`, stored at `j + 1`).
const AS: usize = BLOCK_W + 2;

#[inline]
#[target_feature(enable = "avx2")]
fn ld(s: &[i32], i: usize) -> __m256i {
    load256_i32(s[i..i + L].try_into().unwrap_or(&[0; L]))
}

#[inline]
#[target_feature(enable = "avx2")]
fn st(s: &mut [i32], i: usize, v: __m256i) {
    if let Ok(a) = <&mut [i32; L]>::try_from(&mut s[i..i + L]) {
        store256_i32(a, v);
    }
}

/// Stores 8 values in `0..=0xffff` as samples.
#[inline]
#[target_feature(enable = "avx2")]
fn st_px(s: &mut [u16], i: usize, v: __m256i) {
    let p = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    if let Ok(a) = <&mut [u16; L]>::try_from(&mut s[i..i + L]) {
        store128_u16(a, p);
    }
}

/// The rounding term and the shift count of `Round2( x, n )`, `n >= 0`.
#[inline]
#[target_feature(enable = "avx2")]
fn rounding(n: i32) -> (__m256i, __m128i) {
    (_mm256_set1_epi32((1 << n) >> 1), _mm_cvtsi32_si128(n))
}

/// `Round2( v, n )` with `rounding(n)`.
#[inline]
#[target_feature(enable = "avx2")]
fn round(v: __m256i, (rnd, cnt): (__m256i, __m128i)) -> __m256i {
    _mm256_sra_epi32(_mm256_add_epi32(v, rnd), cnt)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip(v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(v, lo), hi)
}

/// The 7-tap filter `f` (taps 0..=3, symmetric) on the vectors at `s[i + t * step]`, `t` in
/// `0..7`: `f[t] * (x[t] + x[6 - t])` is `f[t] * x[t] + f[6 - t] * x[6 - t]`.
#[inline]
#[target_feature(enable = "avx2")]
fn taps7(s: &[i32], i: usize, step: usize, f: &[__m256i; 4]) -> __m256i {
    let mut acc = _mm256_mullo_epi32(f[3], ld(s, i + 3 * step));
    for t in 0..3 {
        let pair = _mm256_add_epi32(ld(s, i + t * step), ld(s, i + (6 - t) * step));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(f[t], pair));
    }
    acc
}

/// The Wiener filter of a `w` x `h` block (`8 <= w <= BLOCK_W`, `h <= BLOCK_H`) from its window
/// `win` (`w + 6` wide) into `out` (row `r` at `r * stride`).
#[target_feature(enable = "avx2")]
pub(super) fn wiener(
    win: &[i32],
    w: usize,
    h: usize,
    k: &WienerK,
    out: &mut [u16],
    stride: usize,
    inter: &mut [i32],
) {
    let ww = w + 6;
    let mut hf = [_mm256_setzero_si256(); 4];
    let mut vf = hf;
    for t in 0..4 {
        hf[t] = _mm256_set1_epi32(k.hf[t]);
        vf[t] = _mm256_set1_epi32(k.vf[t]);
    }
    let (r0, r1) = (rounding(k.round0), rounding(k.round1));
    let (lo, hi) = (_mm256_set1_epi32(k.lo), _mm256_set1_epi32(k.hi));
    let (zero, max) = (_mm256_setzero_si256(), _mm256_set1_epi32(k.max));
    // The horizontal pass of window rows 0..h + 6, `BLOCK_W` values per row.
    for r in 0..h + 6 {
        for c in starts(w, L) {
            let v = round(taps7(win, r * ww + c, 1, &hf), r0);
            st(inter, r * BLOCK_W + c, clip(v, lo, hi));
        }
    }
    for r in 0..h {
        for c in starts(w, L) {
            let v = round(taps7(inter, r * BLOCK_W + c, BLOCK_W, &vf), r1);
            st_px(out, r * stride + c, clip(v, zero, max));
        }
    }
}

/// `e * (s[i] + s[i + 2]) + m * s[i + 1]` on vectors: one row of the 3x3 weights.
#[inline]
#[target_feature(enable = "avx2")]
fn row3(s: &[i32], i: usize, e: __m256i, m: __m256i) -> __m256i {
    let ends = _mm256_add_epi32(ld(s, i), ld(s, i + 2));
    _mm256_add_epi32(
        _mm256_mullo_epi32(e, ends),
        _mm256_mullo_epi32(m, ld(s, i + 1)),
    )
}

/// The weighted 3x3 sum of `s` (A or B) for output `(i, j)` at `o = i * AS + j` (spec 7.17.3,
/// the `F` loop).
#[inline]
#[target_feature(enable = "avx2")]
fn weigh(s: &[i32], o: usize, i: usize, pass0: bool) -> __m256i {
    if !pass0 {
        let (c3, c4) = (_mm256_set1_epi32(3), _mm256_set1_epi32(4));
        let a = _mm256_add_epi32(row3(s, o, c3, c4), row3(s, o + AS, c4, c4));
        _mm256_add_epi32(a, row3(s, o + 2 * AS, c3, c4))
    } else {
        let (c5, c6) = (_mm256_set1_epi32(5), _mm256_set1_epi32(6));
        if i & 1 == 0 {
            _mm256_add_epi32(row3(s, o, c5, c6), row3(s, o + 2 * AS, c5, c6))
        } else {
            row3(s, o + AS, c5, c6)
        }
    }
}

/// One box filter pass (spec 7.17.3) of a `w` x `h` block: `F` into `f`, `BLOCK_W` per row.
#[target_feature(enable = "avx2")]
fn box_pass(
    win: &[i32],
    w: usize,
    h: usize,
    p: &BoxK,
    f: &mut [i32],
    a_arr: &mut [i32],
    b_arr: &mut [i32],
) {
    let (ww, r) = (w + 6, p.r);
    // Column sums and sums of squares of the current box rows.
    let mut sum = [0i32; BLOCK_W + 6];
    let mut sq = [0i32; BLOCK_W + 6];
    let (ra, rb) = (rounding(p.sh_a), rounding(p.sh_b));
    let (n, s) = (_mm256_set1_epi32(p.n), _mm256_set1_epi32(p.s));
    let (pmax, one_over_n) = (_mm256_set1_epi32(p.pmax), _mm256_set1_epi32(p.one_over_n));
    let (zero, one) = (_mm256_setzero_si256(), _mm256_set1_epi32(1));
    let (c255, c256) = (
        _mm256_set1_epi32(255),
        _mm256_set1_epi32(1 << SGRPROJ_SGR_BITS),
    );
    let rnd_z = _mm256_set1_epi32(1 << (SGRPROJ_MTABLE_BITS - 1));
    let rnd_b = _mm256_set1_epi32(1 << (SGRPROJ_RECIP_BITS - 1));
    for i in 0..h + 2 {
        // Spec row i - 1: pass 0 uses only the odd ones.
        if p.pass0 && i & 1 == 1 {
            continue;
        }
        // Window rows i + 2 - r ..= i + 2 + r hold the box rows of spec row i - 1.
        for c in starts(ww, L) {
            let (mut b, mut a) = (zero, zero);
            for dy in 0..=2 * r {
                let v = ld(win, (i + 2 - r + dy) * ww + c);
                b = _mm256_add_epi32(b, v);
                a = _mm256_add_epi32(a, _mm256_mullo_epi32(v, v));
            }
            st(&mut sum, c, b);
            st(&mut sq, c, a);
        }
        for j in starts(w + 2, L) {
            let (mut b, mut a) = (zero, zero);
            for dx in 0..=2 * r {
                b = _mm256_add_epi32(b, ld(&sum, j + 2 - r + dx));
                a = _mm256_add_epi32(a, ld(&sq, j + 2 - r + dx));
            }
            let a = round(a, ra);
            let d = round(b, rb);
            let pp = _mm256_sub_epi32(_mm256_mullo_epi32(a, n), _mm256_mullo_epi32(d, d));
            let pp = clip(pp, zero, pmax);
            let z = _mm256_srai_epi32::<{ SGRPROJ_MTABLE_BITS as i32 }>(_mm256_add_epi32(
                _mm256_mullo_epi32(pp, s),
                rnd_z,
            ));
            let z = _mm256_min_epi32(z, c255);
            // a2: 1 for z == 0 (the quotient is 0), 256 for z == 255 (the quotient is 255).
            let num = _mm256_add_epi32(
                _mm256_slli_epi32::<{ SGRPROJ_SGR_BITS as i32 }>(z),
                _mm256_srai_epi32::<1>(z),
            );
            let den = _mm256_add_epi32(z, one);
            let q = _mm256_cvttps_epi32(_mm256_div_ps(
                _mm256_cvtepi32_ps(num),
                _mm256_cvtepi32_ps(den),
            ));
            let a2 = _mm256_sub_epi32(_mm256_max_epi32(q, one), _mm256_cmpeq_epi32(z, c255));
            // (256 - a2) * oneOverN * b is below 2^32: wrapping multiplies, logical shift.
            let m = _mm256_mullo_epi32(_mm256_sub_epi32(c256, a2), one_over_n);
            let b2 = _mm256_add_epi32(_mm256_mullo_epi32(m, b), rnd_b);
            st(a_arr, i * AS + j, a2);
            st(
                b_arr,
                i * AS + j,
                _mm256_srli_epi32::<{ SGRPROJ_RECIP_BITS as i32 }>(b2),
            );
        }
    }
    // Round2( v, SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS ) for shift 5 and 4.
    let r5 = rounding((SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as i32);
    let r4 = rounding((SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS) as i32);
    for i in 0..h {
        let rr = if p.pass0 && i & 1 == 1 { r4 } else { r5 };
        for j in starts(w, L) {
            let o = i * AS + j;
            let a = weigh(a_arr, o, i, p.pass0);
            let b = weigh(b_arr, o, i, p.pass0);
            let v = _mm256_add_epi32(_mm256_mullo_epi32(a, ld(win, (i + 3) * ww + j + 3)), b);
            st(f, i * BLOCK_W + j, round(v, rr));
        }
    }
}

/// The self guided filter of a `w` x `h` block (`8 <= w <= BLOCK_W`, `h <= BLOCK_H`) from its
/// window `win` (`w + 6` wide) into `out` (row `r` at `r * stride`).
#[target_feature(enable = "avx2")]
pub(super) fn self_guided(
    win: &[i32],
    w: usize,
    h: usize,
    k: &SgrK,
    out: &mut [u16],
    stride: usize,
    m: &mut Mem<'_>,
) {
    let ww = w + 6;
    let (f0, f1) = m.flt.split_at_mut(BLOCK_H * BLOCK_W);
    if let Some(p) = &k.pass[0] {
        box_pass(win, w, h, p, f0, &mut m.a, &mut m.b);
    }
    if let Some(p) = &k.pass[1] {
        box_pass(win, w, h, p, f1, &mut m.a, &mut m.b);
    }
    let (w0, w1, w2) = (
        _mm256_set1_epi32(k.w0),
        _mm256_set1_epi32(k.w1),
        _mm256_set1_epi32(k.w2),
    );
    let rnd = _mm256_set1_epi32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1));
    let (zero, max) = (_mm256_setzero_si256(), _mm256_set1_epi32(k.max));
    for i in 0..h {
        for j in starts(w, L) {
            let u = _mm256_slli_epi32::<{ SGRPROJ_RST_BITS as i32 }>(ld(win, (i + 3) * ww + j + 3));
            let o = i * BLOCK_W + j;
            let x0 = if k.pass[0].is_some() { ld(f0, o) } else { u };
            let x1 = if k.pass[1].is_some() { ld(f1, o) } else { u };
            let v = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(w1, u), _mm256_mullo_epi32(w0, x0)),
                _mm256_mullo_epi32(w2, x1),
            );
            let v = _mm256_srai_epi32::<{ (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as i32 }>(
                _mm256_add_epi32(v, rnd),
            );
            st_px(out, i * stride + j, clip(v, zero, max));
        }
    }
}
