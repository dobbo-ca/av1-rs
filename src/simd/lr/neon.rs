//! NEON loop restoration kernels (see `super` for the arithmetic): 4 samples per vector, in
//! 32-bit lanes. The same steps as `avx2.rs`. No `unsafe`: memory goes through `simd::arm` and
//! safe slices.

use super::{BLOCK_H, BoxK, SgrK, WienerK, starts};
use crate::consts::{
    SGRPROJ_MTABLE_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS, SGRPROJ_SGR_BITS,
};
use crate::lr::BLOCK_W;
use crate::simd::arm::{load_i32x4, store_i32x4, store_u16x4};
use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;
/// Row stride of the A and B arrays (spec `A[ i ][ j ]`, `j` in `-1..w+1`, stored at `j + 1`).
const AS: usize = BLOCK_W + 2;

#[inline]
#[target_feature(enable = "neon")]
fn ld(s: &[i32], i: usize) -> int32x4_t {
    load_i32x4(s[i..i + L].try_into().unwrap_or(&[0; L]))
}

#[inline]
#[target_feature(enable = "neon")]
fn st(s: &mut [i32], i: usize, v: int32x4_t) {
    if let Ok(a) = <&mut [i32; L]>::try_from(&mut s[i..i + L]) {
        store_i32x4(a, v);
    }
}

/// Stores 4 values in `0..=0xffff` as samples.
#[inline]
#[target_feature(enable = "neon")]
fn st_px(s: &mut [u16], i: usize, v: int32x4_t) {
    if let Ok(a) = <&mut [u16; L]>::try_from(&mut s[i..i + L]) {
        store_u16x4(a, vqmovun_s32(v));
    }
}

/// The rounding term and the (negative) shift count of `Round2( x, n )`, `n >= 0`.
#[inline]
#[target_feature(enable = "neon")]
fn rounding(n: i32) -> (int32x4_t, int32x4_t) {
    (vdupq_n_s32((1 << n) >> 1), vdupq_n_s32(-n))
}

/// `Round2( v, n )` with `rounding(n)`: a negative `vshlq_s32` count is an arithmetic right shift.
#[inline]
#[target_feature(enable = "neon")]
fn round(v: int32x4_t, (rnd, cnt): (int32x4_t, int32x4_t)) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, rnd), cnt)
}

#[inline]
#[target_feature(enable = "neon")]
fn clip(v: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, lo), hi)
}

/// The 7-tap filter `f` (taps 0..=3, symmetric) on the vectors at `s[i + t * step]`, `t` in
/// `0..7`: `f[t] * (x[t] + x[6 - t])` is `f[t] * x[t] + f[6 - t] * x[6 - t]`.
#[inline]
#[target_feature(enable = "neon")]
fn taps7(s: &[i32], i: usize, step: usize, f: &[int32x4_t; 4]) -> int32x4_t {
    let mut acc = vmulq_s32(f[3], ld(s, i + 3 * step));
    for t in 0..3 {
        let pair = vaddq_s32(ld(s, i + t * step), ld(s, i + (6 - t) * step));
        acc = vaddq_s32(acc, vmulq_s32(f[t], pair));
    }
    acc
}

/// The Wiener filter of a `w` x `h` block (`4 <= w <= BLOCK_W`, `h <= BLOCK_H`) from its window
/// `win` (`w + 6` wide) into `out` (row `r` at `r * stride`).
#[target_feature(enable = "neon")]
pub(super) fn wiener(win: &[i32], w: usize, h: usize, k: &WienerK, out: &mut [u16], stride: usize) {
    let ww = w + 6;
    let mut hf = [vdupq_n_s32(0); 4];
    let mut vf = hf;
    for t in 0..4 {
        hf[t] = vdupq_n_s32(k.hf[t]);
        vf[t] = vdupq_n_s32(k.vf[t]);
    }
    let (r0, r1) = (rounding(k.round0), rounding(k.round1));
    let (lo, hi) = (vdupq_n_s32(k.lo), vdupq_n_s32(k.hi));
    let (zero, max) = (vdupq_n_s32(0), vdupq_n_s32(k.max));
    // The horizontal pass of window rows 0..h + 6, `BLOCK_W` values per row.
    let mut inter = [0i32; (BLOCK_H + 6) * BLOCK_W];
    for r in 0..h + 6 {
        for c in starts(w, L) {
            let v = round(taps7(win, r * ww + c, 1, &hf), r0);
            st(&mut inter, r * BLOCK_W + c, clip(v, lo, hi));
        }
    }
    for r in 0..h {
        for c in starts(w, L) {
            let v = round(taps7(&inter, r * BLOCK_W + c, BLOCK_W, &vf), r1);
            st_px(out, r * stride + c, clip(v, zero, max));
        }
    }
}

/// `e * (s[i] + s[i + 2]) + m * s[i + 1]` on vectors: one row of the 3x3 weights.
#[inline]
#[target_feature(enable = "neon")]
fn row3(s: &[i32], i: usize, e: int32x4_t, m: int32x4_t) -> int32x4_t {
    let ends = vaddq_s32(ld(s, i), ld(s, i + 2));
    vaddq_s32(vmulq_s32(e, ends), vmulq_s32(m, ld(s, i + 1)))
}

/// The weighted 3x3 sum of `s` (A or B) for output `(i, j)` at `o = i * AS + j` (spec 7.17.3,
/// the `F` loop).
#[inline]
#[target_feature(enable = "neon")]
fn weigh(s: &[i32], o: usize, i: usize, pass0: bool) -> int32x4_t {
    if !pass0 {
        let (c3, c4) = (vdupq_n_s32(3), vdupq_n_s32(4));
        let a = vaddq_s32(row3(s, o, c3, c4), row3(s, o + AS, c4, c4));
        vaddq_s32(a, row3(s, o + 2 * AS, c3, c4))
    } else {
        let (c5, c6) = (vdupq_n_s32(5), vdupq_n_s32(6));
        if i & 1 == 0 {
            vaddq_s32(row3(s, o, c5, c6), row3(s, o + 2 * AS, c5, c6))
        } else {
            row3(s, o + AS, c5, c6)
        }
    }
}

/// One box filter pass (spec 7.17.3) of a `w` x `h` block: `F` into `f`, `BLOCK_W` per row.
#[target_feature(enable = "neon")]
fn box_pass(win: &[i32], w: usize, h: usize, p: &BoxK, f: &mut [i32]) {
    let (ww, r) = (w + 6, p.r);
    let mut a_arr = [0i32; (BLOCK_H + 2) * AS];
    let mut b_arr = [0i32; (BLOCK_H + 2) * AS];
    // Column sums and sums of squares of the current box rows.
    let mut sum = [0i32; BLOCK_W + 6];
    let mut sq = [0i32; BLOCK_W + 6];
    let (ra, rb) = (rounding(p.sh_a), rounding(p.sh_b));
    let (n, s) = (vdupq_n_s32(p.n), vdupq_n_s32(p.s));
    let (pmax, one_over_n) = (vdupq_n_s32(p.pmax), vdupq_n_s32(p.one_over_n));
    let (zero, one) = (vdupq_n_s32(0), vdupq_n_s32(1));
    let (c255, c256) = (vdupq_n_s32(255), vdupq_n_s32(1 << SGRPROJ_SGR_BITS));
    let rnd_z = vdupq_n_s32(1 << (SGRPROJ_MTABLE_BITS - 1));
    let rnd_b = vdupq_n_s32(1 << (SGRPROJ_RECIP_BITS - 1));
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
                b = vaddq_s32(b, v);
                a = vaddq_s32(a, vmulq_s32(v, v));
            }
            st(&mut sum, c, b);
            st(&mut sq, c, a);
        }
        for j in starts(w + 2, L) {
            let (mut b, mut a) = (zero, zero);
            for dx in 0..=2 * r {
                b = vaddq_s32(b, ld(&sum, j + 2 - r + dx));
                a = vaddq_s32(a, ld(&sq, j + 2 - r + dx));
            }
            let a = round(a, ra);
            let d = round(b, rb);
            let pp = clip(vsubq_s32(vmulq_s32(a, n), vmulq_s32(d, d)), zero, pmax);
            let z =
                vshrq_n_s32::<{ SGRPROJ_MTABLE_BITS as i32 }>(vaddq_s32(vmulq_s32(pp, s), rnd_z));
            let z = vminq_s32(z, c255);
            // a2: 1 for z == 0 (the quotient is 0), 256 for z == 255 (the quotient is 255).
            let num = vaddq_s32(
                vshlq_n_s32::<{ SGRPROJ_SGR_BITS as i32 }>(z),
                vshrq_n_s32::<1>(z),
            );
            let den = vaddq_s32(z, one);
            let q = vcvtq_s32_f32(vdivq_f32(vcvtq_f32_s32(num), vcvtq_f32_s32(den)));
            let sat = vreinterpretq_s32_u32(vceqq_s32(z, c255));
            let a2 = vsubq_s32(vmaxq_s32(q, one), sat);
            // (256 - a2) * oneOverN * b is below 2^32: wrapping multiplies, logical shift.
            let m = vmulq_s32(vsubq_s32(c256, a2), one_over_n);
            let b2 = vreinterpretq_u32_s32(vaddq_s32(vmulq_s32(m, b), rnd_b));
            st(&mut a_arr, i * AS + j, a2);
            st(
                &mut b_arr,
                i * AS + j,
                vreinterpretq_s32_u32(vshrq_n_u32::<{ SGRPROJ_RECIP_BITS as i32 }>(b2)),
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
            let a = weigh(&a_arr, o, i, p.pass0);
            let b = weigh(&b_arr, o, i, p.pass0);
            let v = vaddq_s32(vmulq_s32(a, ld(win, (i + 3) * ww + j + 3)), b);
            st(f, i * BLOCK_W + j, round(v, rr));
        }
    }
}

/// The self guided filter of a `w` x `h` block (`4 <= w <= BLOCK_W`, `h <= BLOCK_H`) from its
/// window `win` (`w + 6` wide) into `out` (row `r` at `r * stride`).
#[target_feature(enable = "neon")]
pub(super) fn self_guided(
    win: &[i32],
    w: usize,
    h: usize,
    k: &SgrK,
    out: &mut [u16],
    stride: usize,
) {
    let ww = w + 6;
    let mut flt = [[0i32; BLOCK_H * BLOCK_W]; 2];
    let [f0, f1] = &mut flt;
    if let Some(p) = &k.pass[0] {
        box_pass(win, w, h, p, f0);
    }
    if let Some(p) = &k.pass[1] {
        box_pass(win, w, h, p, f1);
    }
    let (w0, w1, w2) = (vdupq_n_s32(k.w0), vdupq_n_s32(k.w1), vdupq_n_s32(k.w2));
    let rnd = vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1));
    let (zero, max) = (vdupq_n_s32(0), vdupq_n_s32(k.max));
    for i in 0..h {
        for j in starts(w, L) {
            let u = vshlq_n_s32::<{ SGRPROJ_RST_BITS as i32 }>(ld(win, (i + 3) * ww + j + 3));
            let o = i * BLOCK_W + j;
            let x0 = if k.pass[0].is_some() {
                ld(&flt[0], o)
            } else {
                u
            };
            let x1 = if k.pass[1].is_some() {
                ld(&flt[1], o)
            } else {
                u
            };
            let v = vaddq_s32(
                vaddq_s32(vmulq_s32(w1, u), vmulq_s32(w0, x0)),
                vmulq_s32(w2, x1),
            );
            let v =
                vshrq_n_s32::<{ (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as i32 }>(vaddq_s32(v, rnd));
            st_px(out, i * stride + j, clip(v, zero, max));
        }
    }
}
