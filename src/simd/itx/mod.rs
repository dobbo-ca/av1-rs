//! Kernel family `itx`: the 2D inverse transforms (DCT 4..64, ADST 4/8/16, identity 4..32, the
//! flips) and the add-residual-and-clip into the samples (spec 7.13.3, 7.12.3). The scalar
//! oracle is `transform::inverse_transform_add`; lossless blocks (Walsh-Hadamard) always take it.
//!
//! The kernels run many 1D transforms at once, one per `i32` lane: the row pass takes `L` rows
//! (lanes = rows, read from the column-major `Dequant`), the column pass `L` columns (lanes =
//! columns, read from the row-major row output). Each 1D transform is the scalar code's
//! butterfly sequence, step for step, on vectors (`itx_kernels!`), so the clamps of the
//! Hadamard steps and the rounding of every rotation are the spec's.
//!
//! Exactness of the products: a rotation computes `( a * c + b * s + 2048 ) >> 12` with
//! `|c|, |s| <= 4096` (11586 for identity16), which the scalar code does in `i64` and narrows.
//! - Bit depth 8 and 10 (`HBD = false`): the inputs are at most `2^( BitDepth + 7 )` (the
//!   dequantizer clamp of `recon`), every Hadamard output fits the row range `BitDepth + 8` or the
//!   column range 16, and every rotation input is a transform input or a Hadamard output. So
//!   `|sum| <= 2^17 * 5793` (rotations), `2^17 * 11586` (identity16) or `2^17 * 10950` (ADST4),
//!   all below `2^31`: the wrapping `i32` sum is the true sum.
//! - Bit depth 12 (`HBD = true`): each input is split as `hi * 4096 + lo` with `0 <= lo < 4096`;
//!   then `( sum c * x + 2048 ) >> 12 = sum c * hi + ( ( sum c * lo + 2048 ) >> 12 )`, where the
//!   low sum is below `2^26` and the high sum wraps exactly like the scalar narrowing. This is
//!   exact for every `i32` input.
//!
//! The Hadamard sums and the `Round2` steps cannot overflow with these inputs either, so the
//! kernels equal the scalar code bit for bit.

use crate::simd::{Simd, Tier};
use crate::transform::{self, DQ_LEN, MID_LEN, Plan};

/// The 1D transforms and the 2D driver over a vector type `V` of `L` `i32` lanes, for one tier.
/// The tier module defines `V`, `L` and the `#[target_feature]` primitives `vsplat`, `vadd`,
/// `vsub`, `vmul` (low 32 bits), `vand`, `vmin`, `vmax`, `vsra` (arithmetic shift right),
/// `vload`/`vstore` (`[i32; L]`) and `px_load`/`px_store` (`[u16; L]`, values in `0..=65535`),
/// then invokes this macro with its feature string.
macro_rules! itx_kernels {
    ($feat:literal) => {
        use crate::transform::{
            DQ_LEN, DQ_STRIDE, Kind, MID_LEN, MID_STRIDE, Plan, SINPI_1_9, SINPI_2_9, SINPI_3_9,
            SINPI_4_9, brev, cos128, sin128,
        };

        /// `L` lanes of `s` from `o`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn ld(s: &[i32], o: usize) -> V {
            match s[o..].first_chunk::<L>() {
                Some(a) => vload(a),
                None => vsplat(0),
            }
        }

        #[inline]
        #[target_feature(enable = $feat)]
        fn clamp(v: V, lo: V, hi: V) -> V {
            vmax(vmin(v, hi), lo)
        }

        /// `( sum v[ k ] * c[ k ] + 2048 ) >> 12`, narrowed to `i32` like the scalar code (see
        /// the module doc for why this is exact).
        #[inline]
        #[target_feature(enable = $feat)]
        fn lin<const HBD: bool, const K: usize>(v: [V; K], c: [i32; K]) -> V {
            if HBD {
                let mask = vsplat(4095);
                let mut hi = vsplat(0);
                let mut lo = vsplat(2048);
                for k in 0..K {
                    let ck = vsplat(c[k]);
                    hi = vadd(hi, vmul(vsra(v[k], 12), ck));
                    lo = vadd(lo, vmul(vand(v[k], mask), ck));
                }
                vadd(hi, vsra(lo, 12))
            } else {
                let mut s = vsplat(2048);
                for k in 0..K {
                    s = vadd(s, vmul(v[k], vsplat(c[k])));
                }
                vsra(s, 12)
            }
        }

        /// Spec 7.13.2.1 butterfly `B( a, b, angle, flip )`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn bf<const HBD: bool, const N: usize>(
            t: &mut [V; N],
            a: i32,
            b: i32,
            angle: i32,
            flip: bool,
        ) {
            let (a, b) = (a as usize, b as usize);
            let (c, s) = (cos128(angle), sin128(angle));
            let (ta, tb) = (t[a], t[b]);
            let x = lin::<HBD, 2>([ta, tb], [c, -s]);
            let y = lin::<HBD, 2>([ta, tb], [s, c]);
            if flip {
                t[a] = y;
                t[b] = x;
            } else {
                t[a] = x;
                t[b] = y;
            }
        }

        /// Spec 7.13.2.1 Hadamard rotation `H( a, b, flip, r )`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn hd<const N: usize>(t: &mut [V; N], a: i32, b: i32, flip: bool, r: u32) {
            let (a, b) = if flip {
                (b as usize, a as usize)
            } else {
                (a as usize, b as usize)
            };
            let lo = vsplat(-(1i32 << (r - 1)));
            let hi = vsplat((1i32 << (r - 1)) - 1);
            let (x, y) = (t[a], t[b]);
            t[a] = clamp(vadd(x, y), lo, hi);
            t[b] = clamp(vsub(x, y), lo, hi);
        }

        /// Spec 7.13.2.2 inverse DCT array permutation.
        #[inline]
        #[target_feature(enable = $feat)]
        fn dct_permute<const N: usize>(t: &mut [V; N], n: u32) {
            let copy = *t;
            for i in 0..(1i32 << n) {
                t[i as usize] = copy[brev(n, i) as usize];
            }
        }

        /// Spec 7.13.2.4 inverse ADST input array permutation, `n` is 3 or 4.
        #[inline]
        #[target_feature(enable = $feat)]
        fn adst_in_permute<const N: usize>(t: &mut [V; N], n: u32) {
            let n0 = 1usize << n;
            let copy = *t;
            for i in 0..n0 {
                let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
                t[i] = copy[idx];
            }
        }

        /// Spec 7.13.2.5 inverse ADST output array permutation, `n` is 3 or 4.
        #[inline]
        #[target_feature(enable = $feat)]
        fn adst_out_permute<const N: usize>(t: &mut [V; N], n: u32) {
            let n0 = 1usize << n;
            let copy = *t;
            for i in 0..n0 {
                let a = (i >> 3) & 1;
                let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
                let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
                let d = (i & 1) ^ ((i >> 1) & 1);
                let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
                t[i] = if i & 1 == 1 {
                    vsub(vsplat(0), copy[idx])
                } else {
                    copy[idx]
                };
            }
        }

        /// Spec 7.13.2.6 inverse ADST4, each output as one linear combination of the inputs
        /// (the scalar code's `i64` steps expanded).
        #[inline]
        #[target_feature(enable = $feat)]
        fn adst4<const HBD: bool, const N: usize>(t: &mut [V; N]) {
            let (s1, s2, s3, s4) = (
                SINPI_1_9 as i32,
                SINPI_2_9 as i32,
                SINPI_3_9 as i32,
                SINPI_4_9 as i32,
            );
            let x = [t[0], t[1], t[2], t[3]];
            t[0] = lin::<HBD, 4>(x, [s1, s3, s4, s2]);
            t[1] = lin::<HBD, 4>(x, [s2, s3, -s1, -s4]);
            t[2] = lin::<HBD, 3>([x[0], x[2], x[3]], [s3, -s3, s3]);
            t[3] = lin::<HBD, 4>(x, [s1 + s2, -s3, s4 - s1, s2 - s4]);
        }

        /// Spec 7.13.2.11 to 7.13.2.15 inverse identity transform, `2 <= n <= 5`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn identity<const HBD: bool, const N: usize>(t: &mut [V; N], n: u32) {
            let len = 1usize << n;
            match n {
                2 => {
                    for v in &mut t[..len] {
                        *v = lin::<HBD, 1>([*v], [5793]);
                    }
                }
                3 => {
                    for v in &mut t[..len] {
                        *v = vadd(*v, *v);
                    }
                }
                4 => {
                    for v in &mut t[..len] {
                        *v = lin::<HBD, 1>([*v], [11586]);
                    }
                }
                5 => {
                    for v in &mut t[..len] {
                        let d = vadd(*v, *v);
                        *v = vadd(d, d);
                    }
                }
                _ => {}
            }
        }

        /// Spec 7.13.2.3 inverse DCT of length `1 << n`, `2 <= n <= 6`, with clamping range `r`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn dct<const HBD: bool, const N: usize>(t: &mut [V; N], n: u32, r: u32) {
            dct_permute(t, n);
            if n == 6 {
                for i in 0..16 {
                    bf::<HBD, N>(t, 32 + i, 63 - i, 63 - 4 * brev(4, i), false);
                }
            }
            if n >= 5 {
                for i in 0..8 {
                    bf::<HBD, N>(t, 16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), false);
                }
            }
            if n == 6 {
                for i in 0..16 {
                    hd(t, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
                }
            }
            if n >= 4 {
                for i in 0..4 {
                    bf::<HBD, N>(t, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false);
                }
            }
            if n >= 5 {
                for i in 0..8 {
                    hd(t, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
                }
            }
            if n == 6 {
                for i in 0..4 {
                    for j in 0..2 {
                        bf::<HBD, N>(
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
                    bf::<HBD, N>(t, 4 + i, 7 - i, 56 - 32 * i, false);
                }
            }
            if n >= 4 {
                for i in 0..4 {
                    hd(t, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
                }
            }
            if n >= 5 {
                for i in 0..2 {
                    for j in 0..2 {
                        bf::<HBD, N>(
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
                        hd(t, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
                    }
                }
            }
            for i in 0..2 {
                bf::<HBD, N>(t, 2 * i, 2 * i + 1, 32 + 16 * i, i == 0);
            }
            if n >= 3 {
                for i in 0..2 {
                    hd(t, 4 + 2 * i, 5 + 2 * i, i == 1, r);
                }
            }
            if n >= 4 {
                for i in 0..2 {
                    bf::<HBD, N>(t, 14 - i, 9 + i, 48 + 64 * i, true);
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    for j in 0..2 {
                        hd(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
                    }
                }
            }
            if n == 6 {
                for i in 0..2 {
                    for j in 0..4 {
                        bf::<HBD, N>(
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
                hd(t, i, 3 - i, false, r);
            }
            if n >= 3 {
                bf::<HBD, N>(t, 6, 5, 32, true);
            }
            if n >= 4 {
                for i in 0..2 {
                    for j in 0..2 {
                        hd(t, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
                    }
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    bf::<HBD, N>(t, 29 - i, 18 + i, 48 + (i >> 1) * 64, true);
                }
            }
            if n == 6 {
                for i in 0..4 {
                    for j in 0..4 {
                        hd(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
                    }
                }
            }
            if n >= 3 {
                for i in 0..4 {
                    hd(t, i, 7 - i, false, r);
                }
            }
            if n >= 4 {
                for i in 0..2 {
                    bf::<HBD, N>(t, 13 - i, 10 + i, 32, true);
                }
            }
            if n >= 5 {
                for i in 0..2 {
                    for j in 0..4 {
                        hd(t, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
                    }
                }
            }
            if n == 6 {
                for i in 0..8 {
                    bf::<HBD, N>(t, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
                }
            }
            if n >= 4 {
                for i in 0..8 {
                    hd(t, i, 15 - i, false, r);
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    bf::<HBD, N>(t, 27 - i, 20 + i, 32, true);
                }
            }
            if n == 6 {
                for i in 0..8 {
                    hd(t, 32 + i, 47 - i, false, r);
                    hd(t, 48 + i, 63 - i, true, r);
                }
            }
            if n >= 5 {
                for i in 0..16 {
                    hd(t, i, 31 - i, false, r);
                }
            }
            if n == 6 {
                for i in 0..8 {
                    bf::<HBD, N>(t, 55 - i, 40 + i, 32, true);
                }
                for i in 0..32 {
                    hd(t, i, 63 - i, false, r);
                }
            }
        }

        /// Spec 7.13.2.7 inverse ADST8.
        #[inline]
        #[target_feature(enable = $feat)]
        fn adst8<const HBD: bool, const N: usize>(t: &mut [V; N], r: u32) {
            adst_in_permute(t, 3);
            for i in 0..4 {
                bf::<HBD, N>(t, 2 * i, 2 * i + 1, 60 - 16 * i, true);
            }
            for i in 0..4 {
                hd(t, i, 4 + i, false, r);
            }
            for i in 0..2 {
                bf::<HBD, N>(t, 4 + 3 * i, 5 + i, 48 - 32 * i, true);
            }
            for i in 0..2 {
                for j in 0..2 {
                    hd(t, 4 * j + i, 2 + 4 * j + i, false, r);
                }
            }
            for i in 0..2 {
                bf::<HBD, N>(t, 2 + 4 * i, 3 + 4 * i, 32, true);
            }
            adst_out_permute(t, 3);
        }

        /// Spec 7.13.2.8 inverse ADST16.
        #[inline]
        #[target_feature(enable = $feat)]
        fn adst16<const HBD: bool, const N: usize>(t: &mut [V; N], r: u32) {
            adst_in_permute(t, 4);
            for i in 0..8 {
                bf::<HBD, N>(t, 2 * i, 2 * i + 1, 62 - 8 * i, true);
            }
            for i in 0..8 {
                hd(t, i, 8 + i, false, r);
            }
            for i in 0..2 {
                bf::<HBD, N>(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true);
                bf::<HBD, N>(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true);
            }
            for i in 0..4 {
                for j in 0..2 {
                    hd(t, 8 * j + i, 4 + 8 * j + i, false, r);
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    bf::<HBD, N>(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true);
                }
            }
            for i in 0..2 {
                for j in 0..4 {
                    hd(t, 4 * j + i, 2 + 4 * j + i, false, r);
                }
            }
            for i in 0..4 {
                bf::<HBD, N>(t, 2 + 4 * i, 3 + 4 * i, 32, true);
            }
            adst_out_permute(t, 4);
        }

        /// The 1D transform `kind` of length `1 << n` (the dispatch of 7.13.3).
        #[inline]
        #[target_feature(enable = $feat)]
        fn transform_1d<const HBD: bool, const N: usize>(
            t: &mut [V; N],
            kind: Kind,
            n: u32,
            r: u32,
        ) {
            match kind {
                Kind::Dct => dct::<HBD, N>(t, n, r),
                Kind::Adst => match n {
                    2 => adst4::<HBD, N>(t),
                    3 => adst8::<HBD, N>(t, r),
                    4 => adst16::<HBD, N>(t, r),
                    _ => {}
                },
                Kind::Identity => identity::<HBD, N>(t, n),
            }
        }

        /// Adds `res` to the first `row.len() <= L` samples of `row`, clipped to `0..=max`.
        #[inline]
        #[target_feature(enable = $feat)]
        fn add_px(row: &mut [u16], res: V, max: V) {
            let n = row.len();
            if let Some(a) = row.first_chunk_mut::<L>() {
                let v = clamp(vadd(px_load(a), res), vsplat(0), max);
                px_store(a, v);
            } else {
                let mut a = [0u16; L];
                a[..n].copy_from_slice(row);
                let v = clamp(vadd(px_load(&a), res), vsplat(0), max);
                px_store(&mut a, v);
                row.copy_from_slice(&a[..n]);
            }
        }

        /// `transform::inverse_transform_add` for a non-lossless block; the dispatch has
        /// checked `px.len() >= ( h - 1 ) * stride + w`.
        #[target_feature(enable = $feat)]
        pub(super) fn inverse_transform_add(
            p: &Plan,
            dq: &[i32; DQ_LEN],
            mid: &mut [i32; MID_LEN],
            px: &mut [u16],
            stride: usize,
        ) {
            let n = p.w().max(p.h());
            if p.bit_depth > 10 {
                run_n::<true>(n, p, dq, mid, px, stride);
            } else {
                run_n::<false>(n, p, dq, mid, px, stride);
            }
        }

        /// `run` with the transform array sized to the longest side `n` (4 to 64), so a small
        /// block does not zero a full 64-entry array.
        #[inline]
        #[target_feature(enable = $feat)]
        fn run_n<const HBD: bool>(
            n: usize,
            p: &Plan,
            dq: &[i32; DQ_LEN],
            mid: &mut [i32; MID_LEN],
            px: &mut [u16],
            stride: usize,
        ) {
            match n {
                0..=8 => run::<HBD, 8>(p, dq, mid, px, stride),
                9..=16 => run::<HBD, 16>(p, dq, mid, px, stride),
                17..=32 => run::<HBD, 32>(p, dq, mid, px, stride),
                _ => run::<HBD, 64>(p, dq, mid, px, stride),
            }
        }

        #[inline]
        #[target_feature(enable = $feat)]
        fn run<const HBD: bool, const N: usize>(
            p: &Plan,
            dq: &[i32; DQ_LEN],
            mid: &mut [i32; MID_LEN],
            px: &mut [u16],
            stride: usize,
        ) {
            let (w, h) = (p.w(), p.h());
            let zero = vsplat(0);
            let mut t = [zero; N];

            // Row pass, lanes = rows r0..r0 + L. 32 is a multiple of L, so the loads stay in
            // the coded rows' 32-entry columns of `dq`; the lanes past `p.rows` are not stored.
            let col_lo = vsplat(-(1i32 << (p.col_r() - 1)));
            let col_hi = vsplat((1i32 << (p.col_r() - 1)) - 1);
            let bias = vsplat((1i32 << p.row_shift) >> 1);
            let mut r0 = 0;
            while r0 < p.rows {
                for k in 0..w {
                    t[k] = if k < p.cols {
                        ld(dq, k * DQ_STRIDE + r0)
                    } else {
                        zero
                    };
                }
                if p.rect {
                    for v in &mut t[..w] {
                        *v = lin::<HBD, 1>([*v], [2896]);
                    }
                }
                transform_1d::<HBD, N>(&mut t, p.row_kind, p.log2_w, p.row_r());
                let n = (p.rows - r0).min(L);
                for k in 0..w {
                    let v = clamp(vsra(vadd(t[k], bias), p.row_shift), col_lo, col_hi);
                    let mut lanes = [0i32; L];
                    vstore(&mut lanes, v);
                    for l in 0..n {
                        mid[(r0 + l) * MID_STRIDE + k] = lanes[l];
                    }
                }
                r0 += L;
            }
            // Flipping the columns before the column pass flips its output.
            if p.flip_lr {
                for i in 0..p.rows {
                    mid[i * MID_STRIDE..i * MID_STRIDE + w].reverse();
                }
            }

            // Column pass, lanes = columns j0..j0 + L (64 is a multiple of L); the rows below
            // the coded ones are zero.
            let max = vsplat((1i32 << p.bit_depth) - 1);
            let eight = vsplat(8);
            let mut j0 = 0;
            while j0 < w {
                for i in 0..h {
                    t[i] = if i < p.rows {
                        ld(mid, i * MID_STRIDE + j0)
                    } else {
                        zero
                    };
                }
                transform_1d::<HBD, N>(&mut t, p.col_kind, p.log2_h, p.col_r());
                let n = (w - j0).min(L);
                for i in 0..h {
                    let y = if p.flip_ud { h - i - 1 } else { i };
                    let o = y * stride + j0;
                    add_px(&mut px[o..o + n], vsra(vadd(t[i], eight), 4), max);
                }
                j0 += L;
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// `transform::inverse_transform_add` on the tier `simd` (see there for the arguments). Does
/// nothing if `px` is shorter than the block.
#[allow(unsafe_code)]
pub(crate) fn inverse_transform_add(
    simd: Simd,
    p: &Plan,
    dq: &[i32; DQ_LEN],
    mid: &mut [i32; MID_LEN],
    px: &mut [u16],
    stride: usize,
) {
    let (w, h) = (p.w(), p.h());
    if px.len() < (h - 1) * stride + w {
        return;
    }
    if p.lossless {
        transform::inverse_transform_add(p, dq, mid, px, stride);
        return;
    }
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: `simd` is a detected tier (only `simd::available`/`best` make one), and
            // AVX-512 implies AVX2 (the kernel's feature).
            unsafe { avx2::inverse_transform_add(p, dq, mid, px, stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `simd` is a `Simd` of tier NEON, which only `simd::available()` and
            // `simd::best()` make after detecting NEON (the kernel's feature).
            unsafe { neon::inverse_transform_add(p, dq, mid, px, stride) }
        }
        _ => transform::inverse_transform_add(p, dq, mid, px, stride),
    }
}

/// Checks every available tier against the scalar oracle on one block (tests and the fuzz
/// target). `rows` and `cols` are clamped to the valid coded region; the coefficients come from
/// `next` shaped by `vmode` (0 small, 1 uniform, 2 extremes, 3 sparse) within the dequantizer
/// clamp `2^( bd + 7 )`, the samples by `pmode` (0 zero, 1 the maximum, else uniform). The block
/// starts `pad` samples into the buffer and has stride `w + pad`; the row pass buffer starts
/// full of garbage.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn check(
    tx_sz: usize,
    tx_type: usize,
    lossless: bool,
    bd: u32,
    rows: usize,
    cols: usize,
    pad: usize,
    vmode: u32,
    pmode: u32,
    next: &mut dyn FnMut() -> u32,
) {
    use crate::tables::{TX_HEIGHT_LOG2, TX_WIDTH_LOG2};
    use crate::transform::DQ_STRIDE;

    let (w, h) = (
        1usize << TX_WIDTH_LOG2[tx_sz],
        1usize << TX_HEIGHT_LOG2[tx_sz],
    );
    let (rows, cols) = (rows.clamp(1, h.min(32)), cols.clamp(1, w.min(32)));
    let Some(plan) = Plan::new(tx_sz, tx_type, lossless, bd, rows, cols) else {
        return;
    };
    let lim = 1i32 << (bd + 7);
    let span = 2 * lim as u32;
    let mut dq = [0i32; DQ_LEN];
    for j in 0..cols {
        for i in 0..rows {
            let r = next();
            dq[j * DQ_STRIDE + i] = match vmode {
                0 => (r % 65) as i32 - 32,
                1 => (r % span) as i32 - lim,
                2 => {
                    if r & 1 == 0 {
                        lim - 1
                    } else {
                        -lim
                    }
                }
                _ => {
                    if r.is_multiple_of(8) {
                        ((r >> 3) % span) as i32 - lim
                    } else {
                        0
                    }
                }
            };
        }
    }
    let max = (1u32 << bd) - 1;
    let stride = w + pad;
    let start: Vec<u16> = (0..pad + (h - 1) * stride + w)
        .map(|_| match pmode {
            0 => 0,
            1 => max as u16,
            _ => (next() % (max + 1)) as u16,
        })
        .collect();
    let mut want = start.clone();
    let mut mid = [0x5a5a_5a5a; MID_LEN];
    transform::inverse_transform_add(&plan, &dq, &mut mid, &mut want[pad..], stride);
    for simd in crate::simd::available() {
        let mut got = start.clone();
        let mut mid = [i32::MIN; MID_LEN];
        inverse_transform_add(simd, &plan, &dq, &mut mid, &mut got[pad..], stride);
        assert_eq!(got, want, "{:?} {plan:?}", simd.tier());
    }
}

/// [`check`] with every parameter taken from `next`.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn check_random(next: &mut dyn FnMut() -> u32) {
    let tx_sz = next() as usize % 19;
    let tx_type = next() as usize % 16;
    let bd = [8, 10, 12][next() as usize % 3];
    let lossless = tx_sz == 0 && next().is_multiple_of(8);
    let (rows, cols) = (next() as usize % 33, next() as usize % 33);
    let pad = next() as usize % 4;
    let (vmode, pmode) = (next() % 4, next() % 4);
    check(
        tx_sz, tx_type, lossless, bd, rows, cols, pad, vmode, pmode, next,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: u32) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }

    #[test]
    fn random_blocks_match_scalar() {
        let mut next = xorshift(0x1234_5678);
        for _ in 0..1500 {
            check_random(&mut next);
        }
    }

    /// Every size, type and bit depth; coded regions of one coefficient, the full 32x32 and
    /// sizes around the lane counts 4, 8 and 16; extreme and random coefficients and samples;
    /// unaligned starts and strides.
    #[test]
    fn every_size_type_and_depth() {
        let mut next = xorshift(99);
        for tx_sz in 0..19 {
            for tx_type in 0..16 {
                for bd in [8, 10, 12] {
                    for (rows, cols, vmode, pmode) in [
                        (1, 1, 2, 1),
                        (32, 32, 2, 0),
                        (32, 32, 1, 2),
                        (3, 8, 1, 2),
                        (9, 17, 3, 2),
                        (16, 15, 0, 2),
                    ] {
                        let pad = tx_type & 3;
                        check(
                            tx_sz, tx_type, false, bd, rows, cols, pad, vmode, pmode, &mut next,
                        );
                    }
                }
            }
        }
    }
}
