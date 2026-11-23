//! Inverse transforms (spec 7.13).
//!
//! The 1D transforms work in place on an array `T` of up to 64 values and follow the spec's
//! butterfly steps one to one. The 2D process (7.13.3) runs the row transforms, clamps, then the
//! column transforms. The output is the spec's `Residual` array *before* the flips of 7.12.3,
//! which `recon::reconstruct` applies when it writes samples.
//!
//! Out-of-range values: the spec only requires intermediate values to fit a bit width for
//! conforming streams. For non-conforming streams this code behaves like dav1d (the CI oracle):
//! the Hadamard steps `H` clamp to `r` bits, the row output is clamped to the column range
//! between the passes, and butterfly rotations `B` do not clamp. `B`, ADST4 and the identity
//! transforms compute in `i64` and wrap on narrowing, so hostile input can never overflow-panic.

#![allow(dead_code)]

use crate::tables::{TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

/// Spec 7.13.2 transform array `T`; only the first `1 << n` entries are used.
type T = [i32; 64];

/// Spec 7.13.3 `Transform_Row_Shift[ TX_SIZES_ALL ]`.
static TRANSFORM_ROW_SHIFT: [u32; 19] = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

/// Spec 7.12.3 `flipUD`, indexed by `PlaneTxType`: FLIPADST_DCT, FLIPADST_ADST, V_FLIPADST,
/// FLIPADST_FLIPADST.
pub(crate) static FLIP_UD: [bool; 16] = [
    false, false, false, false, true, false, true, false, true, false, false, false, false, false,
    true, false,
];

/// Spec 7.12.3 `flipLR`, indexed by `PlaneTxType`: DCT_FLIPADST, ADST_FLIPADST, H_FLIPADST,
/// FLIPADST_FLIPADST.
pub(crate) static FLIP_LR: [bool; 16] = [
    false, false, false, false, false, true, true, true, false, false, false, false, false, false,
    false, true,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dct,
    Adst,
    Identity,
}

use Kind::{Adst as A, Dct as D, Identity as I};

/// 1D transform used for the rows, indexed by `PlaneTxType` (spec 7.13.3).
static ROW_KIND: [Kind; 16] = [D, D, A, A, D, A, A, A, A, I, I, D, I, A, I, A];

/// 1D transform used for the columns, indexed by `PlaneTxType` (spec 7.13.3).
static COL_KIND: [Kind; 16] = [D, A, D, A, A, D, A, A, A, I, D, I, A, I, A, I];

/// Spec `Cos128_Lookup[ 65 ]`.
static COS128_LOOKUP: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

const SINPI_1_9: i64 = 1321;
const SINPI_2_9: i64 = 2482;
const SINPI_3_9: i64 = 3344;
const SINPI_4_9: i64 = 3803;

/// Spec `Round2( x, n )` for signed values.
fn round2(x: i32, n: u32) -> i32 {
    ((i64::from(x) + ((1i64 << n) >> 1)) >> n) as i32
}

/// Spec `brev( numBits, x )` for `1 <= num_bits <= 6`.
fn brev(num_bits: u32, x: i32) -> i32 {
    ((x as u32).reverse_bits() >> (32 - num_bits)) as i32
}

/// Spec 7.13.2.1 `cos128( angle )`.
fn cos128(angle: i32) -> i32 {
    let a = angle & 255;
    match a {
        0..=64 => COS128_LOOKUP[a as usize],
        65..=128 => -COS128_LOOKUP[(128 - a) as usize],
        129..=192 => -COS128_LOOKUP[(a - 128) as usize],
        _ => COS128_LOOKUP[(256 - a) as usize],
    }
}

/// Spec 7.13.2.1 `sin128( angle )`.
fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

/// Spec 7.13.2.1 butterfly `B( a, b, angle, flip, r )`. The `r`-bit range check is a
/// conformance requirement only, so it is not applied. Indices are in `0..64`.
fn b(t: &mut T, a: i32, b: i32, angle: i32, flip: bool) {
    let (a, b) = (a as usize, b as usize);
    let (c, s) = (i64::from(cos128(angle)), i64::from(sin128(angle)));
    let (ta, tb) = (i64::from(t[a]), i64::from(t[b]));
    let x = ((ta * c - tb * s + 2048) >> 12) as i32;
    let y = ((ta * s + tb * c + 2048) >> 12) as i32;
    if flip {
        t[a] = y;
        t[b] = x;
    } else {
        t[a] = x;
        t[b] = y;
    }
}

/// Spec 7.13.2.1 Hadamard rotation `H( a, b, flip, r )`. Indices are in `0..64`.
fn h(t: &mut T, a: i32, b: i32, flip: bool, r: u32) {
    let (a, b) = if flip {
        (b as usize, a as usize)
    } else {
        (a as usize, b as usize)
    };
    let lo = -(1i32 << (r - 1));
    let hi = (1i32 << (r - 1)) - 1;
    let (x, y) = (t[a], t[b]);
    t[a] = x.saturating_add(y).clamp(lo, hi);
    t[b] = x.saturating_sub(y).clamp(lo, hi);
}

/// Spec 7.13.2.2 inverse DCT array permutation.
fn dct_permute(t: &mut T, n: u32) {
    let copy = *t;
    for i in 0..(1i32 << n) {
        t[i as usize] = copy[brev(n, i) as usize];
    }
}

/// Spec 7.13.2.3 inverse DCT of length `1 << n`, `2 <= n <= 6`, with clamping range `r`.
fn inverse_dct(t: &mut T, n: u32, r: u32) {
    dct_permute(t, n);
    if n == 6 {
        for i in 0..16 {
            b(t, 32 + i, 63 - i, 63 - 4 * brev(4, i), false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            b(t, 16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), false);
        }
    }
    if n == 6 {
        for i in 0..16 {
            h(t, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            b(t, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            h(t, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                b(
                    t,
                    62 - i * 4 - j,
                    33 + i * 4 + j,
                    60 - 16 * brev(2, i) + 64 * j,
                    true,
                );
            }
        }
    }
    if n >= 3 {
        for i in 0..2 {
            b(t, 4 + i, 7 - i, 56 - 32 * i, false);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            h(t, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                b(
                    t,
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + (j << 6) + ((1 - i) << 5),
                    true,
                );
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                h(t, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
            }
        }
    }
    for i in 0..2 {
        b(t, 2 * i, 2 * i + 1, 32 + 16 * i, i == 0);
    }
    if n >= 3 {
        for i in 0..2 {
            h(t, 4 + 2 * i, 5 + 2 * i, i == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            b(t, 14 - i, 9 + i, 48 + 64 * i, true);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                h(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                b(
                    t,
                    61 - i * 8 - j,
                    34 + i * 8 + j,
                    56 - i * 32 + (j >> 1) * 64,
                    true,
                );
            }
        }
    }
    for i in 0..2 {
        h(t, i, 3 - i, false, r);
    }
    if n >= 3 {
        b(t, 6, 5, 32, true);
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                h(t, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            b(t, 29 - i, 18 + i, 48 + (i >> 1) * 64, true);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                h(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            h(t, i, 7 - i, false, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            b(t, 13 - i, 10 + i, 32, true);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                h(t, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            b(t, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
        }
    }
    if n >= 4 {
        for i in 0..8 {
            h(t, i, 15 - i, false, r);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            b(t, 27 - i, 20 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..8 {
            h(t, 32 + i, 47 - i, false, r);
            h(t, 48 + i, 63 - i, true, r);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            h(t, i, 31 - i, false, r);
        }
    }
    if n == 6 {
        for i in 0..8 {
            b(t, 55 - i, 40 + i, 32, true);
        }
        for i in 0..32 {
            h(t, i, 63 - i, false, r);
        }
    }
}

/// Spec 7.13.2.4 inverse ADST input array permutation, `n` is 3 or 4.
fn adst_in_permute(t: &mut T, n: u32) {
    let n0 = 1usize << n;
    let copy = *t;
    for i in 0..n0 {
        let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
        t[i] = copy[idx];
    }
}

/// Spec 7.13.2.5 inverse ADST output array permutation, `n` is 3 or 4.
fn adst_out_permute(t: &mut T, n: u32) {
    let n0 = 1usize << n;
    let copy = *t;
    for i in 0..n0 {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        let d = (i & 1) ^ ((i >> 1) & 1);
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        t[i] = if i & 1 == 1 {
            copy[idx].wrapping_neg()
        } else {
            copy[idx]
        };
    }
}

/// Spec 7.13.2.6 inverse ADST4.
fn inverse_adst4(t: &mut T) {
    let (t0, t1, t2, t3) = (
        i64::from(t[0]),
        i64::from(t[1]),
        i64::from(t[2]),
        i64::from(t[3]),
    );
    let mut s0 = SINPI_1_9 * t0;
    let mut s1 = SINPI_2_9 * t0;
    let mut s2 = SINPI_3_9 * t1;
    let mut s3 = SINPI_4_9 * t2;
    let s4 = SINPI_1_9 * t2;
    let s5 = SINPI_2_9 * t3;
    let s6 = SINPI_4_9 * t3;
    let a7 = t0 - t2;
    let b7 = a7 + t3;

    s0 += s3;
    s1 -= s4;
    s3 = s2;
    s2 = SINPI_3_9 * b7;

    s0 += s5;
    s1 -= s6;

    let x0 = s0 + s3;
    let x1 = s1 + s3;
    let x2 = s2;
    let x3 = s0 + s1 - s3;

    t[0] = ((x0 + 2048) >> 12) as i32;
    t[1] = ((x1 + 2048) >> 12) as i32;
    t[2] = ((x2 + 2048) >> 12) as i32;
    t[3] = ((x3 + 2048) >> 12) as i32;
}

/// Spec 7.13.2.7 inverse ADST8.
fn inverse_adst8(t: &mut T, r: u32) {
    adst_in_permute(t, 3);
    for i in 0..4 {
        b(t, 2 * i, 2 * i + 1, 60 - 16 * i, true);
    }
    for i in 0..4 {
        h(t, i, 4 + i, false, r);
    }
    for i in 0..2 {
        b(t, 4 + 3 * i, 5 + i, 48 - 32 * i, true);
    }
    for i in 0..2 {
        for j in 0..2 {
            h(t, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..2 {
        b(t, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_out_permute(t, 3);
}

/// Spec 7.13.2.8 inverse ADST16.
fn inverse_adst16(t: &mut T, r: u32) {
    adst_in_permute(t, 4);
    for i in 0..8 {
        b(t, 2 * i, 2 * i + 1, 62 - 8 * i, true);
    }
    for i in 0..8 {
        h(t, i, 8 + i, false, r);
    }
    for i in 0..2 {
        b(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true);
        b(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true);
    }
    for i in 0..4 {
        for j in 0..2 {
            h(t, 8 * j + i, 4 + 8 * j + i, false, r);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            b(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true);
        }
    }
    for i in 0..2 {
        for j in 0..4 {
            h(t, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..4 {
        b(t, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_out_permute(t, 4);
}

/// Spec 7.13.2.9 inverse ADST of length `1 << n`, `2 <= n <= 4`. Other lengths are not valid
/// ADST sizes; the array is left unchanged.
fn inverse_adst(t: &mut T, n: u32, r: u32) {
    match n {
        2 => inverse_adst4(t),
        3 => inverse_adst8(t, r),
        4 => inverse_adst16(t, r),
        _ => {}
    }
}

/// Spec 7.13.2.10 inverse Walsh-Hadamard transform on `T[ 0..4 ]`.
fn inverse_walsh_hadamard(t: &mut T, shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a = a.wrapping_add(c);
    d = d.wrapping_sub(b);
    let e = a.wrapping_sub(d) >> 1;
    b = e.wrapping_sub(b);
    c = e.wrapping_sub(c);
    a = a.wrapping_sub(b);
    d = d.wrapping_add(c);
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// Spec 7.13.2.11 to 7.13.2.15 inverse identity transform of length `1 << n`, `2 <= n <= 5`.
/// Other lengths are not valid identity sizes; the array is left unchanged.
fn inverse_identity(t: &mut T, n: u32) {
    let len = 1usize << n;
    match n {
        2 => {
            for v in &mut t[..len] {
                *v = ((i64::from(*v) * 5793 + 2048) >> 12) as i32;
            }
        }
        3 => {
            for v in &mut t[..len] {
                *v = v.wrapping_mul(2);
            }
        }
        4 => {
            for v in &mut t[..len] {
                *v = ((i64::from(*v) * 11586 + 2048) >> 12) as i32;
            }
        }
        5 => {
            for v in &mut t[..len] {
                *v = v.wrapping_mul(4);
            }
        }
        _ => {}
    }
}

/// Runs the 1D transform `kind` of length `1 << n` on `t` (the dispatch of 7.13.3).
fn transform_1d(t: &mut T, kind: Kind, n: u32, r: u32) {
    match kind {
        Kind::Dct => inverse_dct(t, n, r),
        Kind::Adst => inverse_adst(t, n, r),
        Kind::Identity => inverse_identity(t, n),
    }
}

/// Spec 7.13.3 2D inverse transform process.
///
/// `dequant` and `residual` are `h * w` row-major with `w = 1 << Tx_Width_Log2[ tx_sz ]` and
/// `h = 1 << Tx_Height_Log2[ tx_sz ]`; entries of `dequant` outside the top-left
/// `min(32, w) x min(32, h)` must be zero. The caller guarantees `tx_sz < 19`, `tx_type < 16`,
/// `bit_depth` in 8..=12 and both slices at least `w * h` long (otherwise nothing is written).
/// `residual` is the spec's `Residual` before the flips of 7.12.3.
pub(crate) fn inverse_transform_2d(
    dequant: &[i32],
    residual: &mut [i32],
    tx_sz: usize,
    tx_type: usize,
    lossless: bool,
    bit_depth: u32,
) {
    let (Some(&log2_w), Some(&log2_h)) = (TX_WIDTH_LOG2.get(tx_sz), TX_HEIGHT_LOG2.get(tx_sz))
    else {
        return;
    };
    let (log2_w, log2_h) = (log2_w as u32, log2_h as u32);
    let (Some(&row_kind), Some(&col_kind)) = (ROW_KIND.get(tx_type), COL_KIND.get(tx_type)) else {
        return;
    };
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    if dequant.len() < w * h || residual.len() < w * h {
        return;
    }
    let tw = w.min(32);
    let row_shift = if lossless {
        0
    } else {
        TRANSFORM_ROW_SHIFT[tx_sz]
    };
    let col_shift = if lossless { 0 } else { 4 };
    let row_clamp_range = bit_depth + 8;
    let col_clamp_range = (bit_depth + 6).max(16);
    let col_lo = -(1i32 << (col_clamp_range - 1));
    let col_hi = (1i32 << (col_clamp_range - 1)) - 1;
    let rect = log2_w.abs_diff(log2_h) == 1;

    for i in 0..h {
        let out = &mut residual[i * w..(i + 1) * w];
        let mut t: T = [0; 64];
        if i < 32 {
            t[..tw].copy_from_slice(&dequant[i * w..i * w + tw]);
        }
        // Every 1D transform maps an all-zero row to zeros.
        if t[..w].iter().all(|&v| v == 0) {
            out.fill(0);
            continue;
        }
        if rect {
            for v in &mut t[..w] {
                *v = ((i64::from(*v) * 2896 + 2048) >> 12) as i32;
            }
        }
        if lossless {
            inverse_walsh_hadamard(&mut t, 2);
        } else {
            transform_1d(&mut t, row_kind, log2_w, row_clamp_range);
        }
        for j in 0..w {
            out[j] = round2(t[j], row_shift).clamp(col_lo, col_hi);
        }
    }

    for j in 0..w {
        let mut t: T = [0; 64];
        for i in 0..h {
            t[i] = residual[i * w + j];
        }
        if lossless {
            inverse_walsh_hadamard(&mut t, 0);
        } else {
            transform_1d(&mut t, col_kind, log2_h, col_clamp_range);
        }
        for i in 0..h {
            residual[i * w + j] = round2(t[i], col_shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::IDTX;

    fn run(tx_sz: usize, tx_type: usize, lossless: bool, bd: u32, dc: i32) -> Vec<i32> {
        let n = (1usize << TX_WIDTH_LOG2[tx_sz]) * (1usize << TX_HEIGHT_LOG2[tx_sz]);
        let mut dq = vec![0; n];
        dq[0] = dc;
        let mut res = vec![7; n];
        inverse_transform_2d(&dq, &mut res, tx_sz, tx_type, lossless, bd);
        res
    }

    /// A DC-only DCT_DCT block is flat; the value follows from the spec's rounding steps
    /// (each 1D DC pass multiplies by cos128(32) = 2896 with Round2 12).
    #[test]
    fn dc_only_dct_is_flat_for_every_size() {
        for tx_sz in 0..19 {
            let (lw, lh) = (TX_WIDTH_LOG2[tx_sz], TX_HEIGHT_LOG2[tx_sz]);
            let mut a = 1000i64;
            if lw.abs_diff(lh) == 1 {
                a = (a * 2896 + 2048) >> 12;
            }
            let v = (a * 2896 + 2048) >> 12;
            let v = (v + ((1i64 << TRANSFORM_ROW_SHIFT[tx_sz]) >> 1)) >> TRANSFORM_ROW_SHIFT[tx_sz];
            let c = (v * 2896 + 2048) >> 12;
            let expect = ((c + 8) >> 4) as i32;
            let res = run(tx_sz, 0, false, 8, 1000);
            assert!(
                res.iter().all(|&r| r == expect),
                "tx_sz {tx_sz}: {res:?} vs {expect}"
            );
        }
    }

    #[test]
    fn dc_only_dct_4x4_value() {
        // 1000 -> 707 -> 500 -> 31.
        assert!(run(0, 0, false, 8, 1000).iter().all(|&r| r == 31));
    }

    #[test]
    fn zero_input_gives_zero() {
        for tx_sz in 0..19 {
            assert!(run(tx_sz, 0, false, 10, 0).iter().all(|&r| r == 0));
        }
    }

    #[test]
    fn wht_dc_known_answer() {
        // Row: a = 16 >> 2 = 4, e = 2 -> [2, 2, 2, 2]; column: a = 2, e = 1 -> [1, 1, 1, 1].
        assert!(run(0, 0, true, 8, 16).iter().all(|&r| r == 1));
        assert!(run(0, 0, true, 8, 32).iter().all(|&r| r == 2));
    }

    #[test]
    fn wht_single_ac_coefficient() {
        // Row 0: T = [0, 16, 0, 0] >> 2 gives a = 0, c = 4; a = 4, d = 0, e = 2, b = 2, c = -2,
        // a = 2, d = -2 -> [2, 2, -2, -2]. Each column [r, 0, 0, 0] gives a flat r / 2.
        let mut dq = vec![0; 16];
        dq[1] = 16;
        let mut res = vec![0; 16];
        inverse_transform_2d(&dq, &mut res, 0, 0, true, 8);
        for row in res.chunks(4) {
            assert_eq!(row, [1, 1, -1, -1]);
        }
    }

    #[test]
    fn identity_scaling() {
        // IDTX 4x4: 1000 -> 1414 (x 5793 / 4096) -> 2000, then Round2( 2000, 4 ) = 125.
        let res = run(0, IDTX, false, 8, 1000);
        assert_eq!(res[0], 125);
        assert!(res[1..].iter().all(|&r| r == 0));
        // IDTX 8x8: x2 in the row, row shift 1, x2 in the column, Round2( 2000, 4 ) = 125.
        let res = run(1, IDTX, false, 8, 1000);
        assert_eq!(res[0], 125);
        assert!(res[1..].iter().all(|&r| r == 0));
        // IDTX 32x32: x4, row shift 2, x4, Round2( 4000, 4 ) = 250.
        let res = run(3, IDTX, false, 8, 1000);
        assert_eq!(res[0], 250);
    }

    #[test]
    fn adst4_dc_matches_hand_computation() {
        // ADST_ADST (3) 4x4, DC 1000: row: x = [1321, 2482, 3344, 3803] * 1000 / 4096.
        let res = run(0, 3, false, 8, 1000);
        let row: Vec<i64> = [1321i64, 2482, 3344, 3803]
            .iter()
            .map(|k| (k * 1000 + 2048) >> 12)
            .collect();
        // The first column pass input is row[j] at row 0; check column 0 against ADST4 by hand.
        let k = row[0];
        let col: Vec<i64> = [1321i64, 2482, 3344, 3803]
            .iter()
            .map(|s| (s * k + 2048) >> 12)
            .collect();
        for i in 0..4 {
            assert_eq!(res[i * 4] as i64, (col[i] + 8) >> 4);
        }
    }

    #[test]
    fn size_limits_do_not_panic() {
        // Hostile extremes: every coefficient at the clamp limits for a 12-bit stream.
        for tx_sz in 0..19 {
            for tx_type in 0..16 {
                let n = (1usize << TX_WIDTH_LOG2[tx_sz]) * (1usize << TX_HEIGHT_LOG2[tx_sz]);
                let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                let mut dq = vec![0; n];
                for i in 0..n {
                    let (r, c) = (i / w, i % w);
                    if r < 32 && c < 32 {
                        dq[i] = if i & 1 == 0 {
                            (1 << 19) - 1
                        } else {
                            -(1 << 19)
                        };
                    }
                }
                let mut res = vec![0; n];
                inverse_transform_2d(&dq, &mut res, tx_sz, tx_type, false, 12);
            }
        }
    }
}
