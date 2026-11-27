//! NEON CDEF kernels. The filter follows dav1d's padded-block design (BSD-2-Clause, Copyright
//! © 2018, VideoLAN and dav1d authors, Copyright © 2018, Two Orioles, LLC; see NOTICE and
//! `mod.rs`).

use super::{Filt, at4, at8};
use crate::simd::arm::{load_i32x4, load_u16x4, load_u16x8, store_u16x8};
use std::arch::aarch64::*;

/// 8 samples of the window that starts at `s` in `t` (row stride `st`): one row of 8
/// (`w == 8`) or two rows of 4.
#[inline]
#[target_feature(enable = "neon")]
fn ld(t: &[u16], s: usize, w: usize, st: usize) -> uint16x8_t {
    if w == 8 {
        load_u16x8(at8(t, s))
    } else {
        vcombine_u16(load_u16x4(at4(t, s)), load_u16x4(at4(t, s + st)))
    }
}

/// `constrain(p - px, thr, damping)` (spec 7.15.3) with `sh` = `-damping_adj`; a sentinel `p`
/// gives 0.
#[inline]
#[target_feature(enable = "neon")]
fn constrain(p: uint16x8_t, px: int16x8_t, thr: uint16x8_t, sh: int16x8_t) -> int16x8_t {
    let diff = vqsubq_s16(vreinterpretq_s16_u16(p), px);
    // Wrapping abs: the saturated sentinel difference -32768 stays 0x8000 (32768 unsigned).
    let mag = vreinterpretq_u16_s16(vabsq_s16(diff));
    let v = vreinterpretq_s16_u16(vminq_u16(mag, vqsubq_u16(thr, vshlq_u16(mag, sh))));
    vbslq_s16(vcltzq_s16(diff), vnegq_s16(v), v)
}

/// The filtered block, `f.w` x `f.h` samples packed in `out` (row stride `f.w`), from `t` (the
/// padded block or the source plane); true when every sample it loaded is below 4096.
#[target_feature(enable = "neon")]
pub(super) fn filter(t: &[u16], out: &mut [u16; 64], f: &Filt) -> bool {
    let mut or = vdupq_n_u16(0);
    let pri_thr = vdupq_n_u16(f.pri_str as u16);
    let sec_thr = vdupq_n_u16(f.sec_str as u16);
    let pri_sh = vdupq_n_s16(-(f.pri_shift as i16));
    let sec_sh = vdupq_n_s16(-(f.sec_shift as i16));
    let rows = 8 / f.w;
    for (n, dst) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .take(f.w * f.h / 8)
        .enumerate()
    {
        let o = n * rows * f.stride;
        let pxu = ld(t, f.center + o, f.w, f.stride);
        or = vorrq_u16(or, pxu);
        let px = vreinterpretq_s16_u16(pxu);
        let (mut sum, mut mn, mut mx) = (vdupq_n_s16(0), pxu, px);
        for (q, &s) in f.pri.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = vorrq_u16(or, p);
            let c = constrain(p, px, pri_thr, pri_sh);
            sum = vmlaq_n_s16(sum, c, f.pri_tap[q / 2]);
            mn = vminq_u16(mn, p);
            mx = vmaxq_s16(mx, vreinterpretq_s16_u16(p));
        }
        for (q, &s) in f.sec.iter().enumerate() {
            let p = ld(t, s + o, f.w, f.stride);
            or = vorrq_u16(or, p);
            let c = constrain(p, px, sec_thr, sec_sh);
            // Weight 2 for k = 0, 1 for k = 1.
            let c = if q < 4 { vaddq_s16(c, c) } else { c };
            sum = vaddq_s16(sum, c);
            mn = vminq_u16(mn, p);
            mx = vmaxq_s16(mx, vreinterpretq_s16_u16(p));
        }
        // (8 + sum - (sum < 0)) >> 4
        let neg = vreinterpretq_s16_u16(vcltzq_s16(sum));
        let sum = vaddq_s16(vaddq_s16(sum, neg), vdupq_n_s16(8));
        let v = vaddq_s16(px, vshrq_n_s16::<4>(sum));
        let v = vminq_s16(vmaxq_s16(v, vreinterpretq_s16_u16(mn)), mx);
        store_u16x8(dst, vreinterpretq_u16_s16(v));
    }
    // The largest lane of the OR is below 4096 exactly when no sample has a bit at or above 12.
    vmaxvq_u16(or) < 4096
}

/// Adds the 8 rows of `v`, row `n` moved up by `n` lanes, into a 16-lane (lo, hi) partial.
#[inline]
#[target_feature(enable = "neon")]
fn diag(v: [int16x8_t; 8]) -> (int16x8_t, int16x8_t) {
    let z = vdupq_n_s16(0);
    let (mut lo, mut hi) = (v[0], z);
    macro_rules! up {
        ($($n:literal)*) => {$(
            lo = vaddq_s16(lo, vextq_s16::<{ 8 - $n }>(z, v[$n]));
            hi = vaddq_s16(hi, vextq_s16::<{ 8 - $n }>(v[$n], z));
        )*};
    }
    up!(1 2 3 4 5 6 7);
    (lo, hi)
}

/// `sum(v[i]^2 * w[i])` over the 16 lanes of a (lo, hi) partial.
#[inline]
#[target_feature(enable = "neon")]
fn sq2((lo, hi): (int16x8_t, int16x8_t), w: &[[i32; 4]; 4]) -> i32 {
    let mut acc = vdupq_n_s32(0);
    for (v, w) in [(lo, &w[..2]), (hi, &w[2..])] {
        let a = vmull_s16(vget_low_s16(v), vget_low_s16(v));
        let b = vmull_high_s16(v, v);
        acc = vmlaq_s32(acc, a, load_i32x4(&w[0]));
        acc = vmlaq_s32(acc, b, load_i32x4(&w[1]));
    }
    vaddvq_s32(acc)
}

/// `Div_Table` weights per partial index of the even directions (0, 4), the odd ones and 2, 6.
const W_EVEN: [[i32; 4]; 4] = [
    [840, 420, 280, 210],
    [168, 140, 120, 105],
    [120, 140, 168, 210],
    [280, 420, 840, 0],
];
const W_ODD: [[i32; 4]; 4] = [
    [420, 210, 140, 105],
    [105, 105, 105, 105],
    [140, 210, 420, 0],
    [0, 0, 0, 0],
];
const W_HV: [[i32; 4]; 4] = [[105; 4], [105; 4], [0; 4], [0; 4]];

/// The eight direction costs of spec 7.15.2 (`cost[ d ]`), for the 8x8 block `rows`.
#[target_feature(enable = "neon")]
pub(super) fn dir_costs(rows: &[[u16; 8]; 8], shift: u32) -> [i32; 8] {
    let z = vdupq_n_s16(0);
    let sh = vdupq_n_s16(-(shift as i16));
    let k128 = vdupq_n_s16(128);
    let mut x = [z; 8];
    let mut rx = [z; 8];
    for i in 0..8 {
        x[i] = vsubq_s16(
            vreinterpretq_s16_u16(vshlq_u16(load_u16x8(&rows[i]), sh)),
            k128,
        );
        let r = vrev64q_s16(x[i]);
        rx[i] = vextq_s16::<4>(r, r);
    }
    // partial[2][i]: row sums; partial[6][j]: column sums.
    let p2 = vpaddq_s16(
        vpaddq_s16(vpaddq_s16(x[0], x[1]), vpaddq_s16(x[2], x[3])),
        vpaddq_s16(vpaddq_s16(x[4], x[5]), vpaddq_s16(x[6], x[7])),
    );
    let mut p6 = x[0];
    for i in 1..8 {
        p6 = vaddq_s16(p6, x[i]);
    }
    // partial[0][i + j], partial[4][7 + i - j] (row reversed: i + 7 - j).
    let p0 = diag(x);
    let p4 = diag(rx);
    // partial[1][i + j / 2], partial[3][3 + i - j / 2]: pair sums of each row (reversed).
    let mut ps = [z; 8];
    let mut rps = [z; 8];
    for i in 0..8 {
        ps[i] = vpaddq_s16(x[i], z);
        rps[i] = vpaddq_s16(rx[i], z);
    }
    let p1 = diag(ps);
    let p3 = diag(rps);
    // partial[7][i / 2 + j], partial[5][3 - i / 2 + j]: row pair sums moved by i / 2.
    let rp = [
        vaddq_s16(x[0], x[1]),
        vaddq_s16(x[2], x[3]),
        vaddq_s16(x[4], x[5]),
        vaddq_s16(x[6], x[7]),
    ];
    let p7 = diag([rp[0], rp[1], rp[2], rp[3], z, z, z, z]);
    let p5 = diag([rp[3], rp[2], rp[1], rp[0], z, z, z, z]);
    [
        sq2(p0, &W_EVEN),
        sq2(p1, &W_ODD),
        sq2((p2, z), &W_HV),
        sq2(p3, &W_ODD),
        sq2(p4, &W_EVEN),
        sq2(p5, &W_ODD),
        sq2((p6, z), &W_HV),
        sq2(p7, &W_ODD),
    ]
}
