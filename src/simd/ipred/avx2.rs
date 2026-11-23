//! AVX2 kernels of the `ipred` family (docs/safety.md). Each step computes 8 samples as i32
//! lanes with the integer arithmetic of the scalar oracle (src/predict.rs), so the output is
//! the same bit for bit. A 4-wide row uses the low 4 lanes. The AVX-512 tier runs these.

use std::arch::x86_64::*;

use crate::consts::{SMOOTH_PRED, SMOOTH_V_PRED};
use crate::predict::{Edge, OFF};
use crate::simd::x86::{
    load64_u8, load128_u8, load128_u16, load256_i32, store128_u16, store256_i32,
};

/// `e[ k..k + 8 ]` (zero if not all in `e`; the callers stay inside).
#[inline]
#[target_feature(enable = "avx2")]
fn ld(e: &[i32], k: usize) -> __m256i {
    match e.get(k..).and_then(|s| s.first_chunk::<8>()) {
        Some(a) => load256_i32(a),
        None => _mm256_setzero_si256(),
    }
}

/// Writes the lanes of `v` (each in 0..=65535) to `d`: all 8, or the first `d.len()`.
#[inline]
#[target_feature(enable = "avx2")]
fn st(d: &mut [u16], v: __m256i) {
    // packus gives [v0..v3 v0..v3 | v4..v7 v4..v7]; 64-bit lanes 0 and 2 are v0..v7.
    let p = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(v, v));
    let p = _mm256_castsi256_si128(p);
    if let Some(a) = d.first_chunk_mut::<8>() {
        store128_u16(a, p);
    } else {
        let mut t = [0u16; 8];
        store128_u16(&mut t, p);
        let n = d.len().min(8);
        d[..n].copy_from_slice(&t[..n]);
    }
}

/// Up to 8 samples of `d`, zero-extended to i32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn ld_u16(d: &[u16]) -> __m256i {
    let v = match d.first_chunk::<8>() {
        Some(a) => load128_u16(a),
        None => {
            let mut t = [0u16; 8];
            let n = d.len().min(8);
            t[..n].copy_from_slice(&d[..n]);
            load128_u16(&t)
        }
    };
    _mm256_cvtepu16_epi32(v)
}

/// Paeth: `predict::predict_paeth`.
#[target_feature(enable = "avx2")]
pub(super) fn paeth(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    left: &Edge,
) {
    let tl = _mm256_set1_epi32(above[OFF - 1]);
    for i in 0..h {
        let l = _mm256_set1_epi32(left[OFF + i]);
        // pTop = | base - AboveRow[ j ] | = | LeftCol[ i ] - AboveRow[ -1 ] |.
        let p_top = _mm256_abs_epi32(_mm256_sub_epi32(l, tl));
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(8).enumerate() {
            let a = ld(above, OFF + 8 * k);
            let base = _mm256_sub_epi32(_mm256_add_epi32(a, l), tl);
            let p_left = _mm256_abs_epi32(_mm256_sub_epi32(base, l));
            let p_tl = _mm256_abs_epi32(_mm256_sub_epi32(base, tl));
            let not_left = _mm256_or_si256(
                _mm256_cmpgt_epi32(p_left, p_top),
                _mm256_cmpgt_epi32(p_left, p_tl),
            );
            let top_or_tl = _mm256_blendv_epi8(a, tl, _mm256_cmpgt_epi32(p_top, p_tl));
            st(d, _mm256_blendv_epi8(l, top_or_tl, not_left));
        }
    }
}

/// SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED: `predict::predict_smooth`. `wx` / `wy` are the
/// weights of the width / height (`w` / `h` entries).
#[target_feature(enable = "avx2")]
pub(super) fn smooth(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: usize,
    wx: &[u8],
    wy: &[u8],
    above: &Edge,
    left: &Edge,
) {
    let mut wxs = [0i32; 64];
    for (d, &s) in wxs.iter_mut().zip(wx) {
        *d = i32::from(s);
    }
    let tr = _mm256_set1_epi32(above[OFF + w - 1]);
    let bl = left[OFF + h - 1];
    let c256 = _mm256_set1_epi32(256);
    let r128 = _mm256_set1_epi32(128);
    for (i, &sy) in wy.iter().enumerate().take(h) {
        let sy = i32::from(sy);
        let vsy = _mm256_set1_epi32(sy);
        let l = _mm256_set1_epi32(left[OFF + i]);
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(8).enumerate() {
            let a = ld(above, OFF + 8 * k);
            let sx = ld(&wxs, 8 * k);
            // sx * LeftCol[ i ] + ( 256 - sx ) * AboveRow[ w - 1 ]
            let hor = _mm256_add_epi32(
                _mm256_mullo_epi32(sx, l),
                _mm256_mullo_epi32(_mm256_sub_epi32(c256, sx), tr),
            );
            let v = if mode == SMOOTH_PRED {
                // + sy * AboveRow[ j ] + ( 256 - sy ) * LeftCol[ h - 1 ], Round2( , 9 )
                let ver = _mm256_add_epi32(
                    _mm256_mullo_epi32(vsy, a),
                    _mm256_set1_epi32((256 - sy) * bl + 256),
                );
                _mm256_srai_epi32::<9>(_mm256_add_epi32(ver, hor))
            } else if mode == SMOOTH_V_PRED {
                _mm256_srai_epi32::<8>(_mm256_add_epi32(
                    _mm256_mullo_epi32(vsy, a),
                    _mm256_set1_epi32((256 - sy) * bl + 128),
                ))
            } else {
                _mm256_srai_epi32::<8>(_mm256_add_epi32(hor, r128))
            };
            st(d, v);
        }
    }
}

/// Directional zone 1 (pAngle < 90) without upsampling: the `p_angle < 90` arm of
/// `predict::directional_ref` with `upsample_above` 0.
#[target_feature(enable = "avx2")]
pub(super) fn dr_z1(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    dx: i32,
) {
    let max_base_x = (w + h - 1) as i32;
    let fill = above[OFF + w + h - 1];
    let vfill = _mm256_set1_epi32(fill);
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let last = _mm256_set1_epi32(max_base_x - 1);
    let r16 = _mm256_set1_epi32(16);
    for i in 0..h {
        let idx = (i as i32 + 1) * dx;
        let base0 = idx >> 6;
        let shift = (idx >> 1) & 0x1f;
        let s1 = _mm256_set1_epi32(shift);
        let s0 = _mm256_set1_epi32(32 - shift);
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(8).enumerate() {
            let base = base0 + 8 * k as i32;
            if base >= max_base_x {
                d.fill(fill as u16);
                continue;
            }
            // base < max_base_x <= 127, so the reads end at OFF + 135 < EDGE_LEN.
            let b = OFF + base as usize;
            let v = _mm256_add_epi32(
                _mm256_mullo_epi32(ld(above, b), s0),
                _mm256_mullo_epi32(ld(above, b + 1), s1),
            );
            let v = _mm256_srai_epi32::<5>(_mm256_add_epi32(v, r16));
            // Lanes with base + lane >= maxBaseX take AboveRow[ maxBaseX ].
            let over = _mm256_cmpgt_epi32(_mm256_add_epi32(_mm256_set1_epi32(base), lanes), last);
            st(d, _mm256_blendv_epi8(v, vfill, over));
        }
    }
}

/// The intra edge filter: `buf[ OFF + i - 1 ] = Round2( sum k[ j ] * p[ i + j ], 4 )` for `i`
/// in 1..sz, where `p` is the padded edge of `super::pad_edge`.
#[target_feature(enable = "avx2")]
pub(super) fn filter_edge(buf: &mut Edge, p: &[i32], sz: usize, k: &[i32; 5]) {
    let k0 = _mm256_set1_epi32(k[0]);
    let k1 = _mm256_set1_epi32(k[1]);
    let k2 = _mm256_set1_epi32(k[2]);
    let k3 = _mm256_set1_epi32(k[3]);
    let k4 = _mm256_set1_epi32(k[4]);
    let r8 = _mm256_set1_epi32(8);
    let mut i = 1;
    while i < sz {
        let mut s = _mm256_mullo_epi32(ld(p, i), k0);
        s = _mm256_add_epi32(s, _mm256_mullo_epi32(ld(p, i + 1), k1));
        s = _mm256_add_epi32(s, _mm256_mullo_epi32(ld(p, i + 2), k2));
        s = _mm256_add_epi32(s, _mm256_mullo_epi32(ld(p, i + 3), k3));
        s = _mm256_add_epi32(s, _mm256_mullo_epi32(ld(p, i + 4), k4));
        let v = _mm256_srai_epi32::<4>(_mm256_add_epi32(s, r8));
        let n = (sz - i).min(8);
        let d = &mut buf[OFF + i - 1..][..n];
        if let Some(a) = d.first_chunk_mut::<8>() {
            store256_i32(a, v);
        } else {
            let mut t = [0i32; 8];
            store256_i32(&mut t, v);
            d.copy_from_slice(&t[..n]);
        }
        i += 8;
    }
}

/// Palette: `predict::palette_ref`, 8 samples per step as two byte table lookups (low and high
/// byte of each colour). `palette` has at most 16 colours; a 4-wide row is scalar.
#[target_feature(enable = "avx2")]
pub(super) fn palette(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    palette: &[u16],
    map: &[u8],
    map_stride: usize,
) {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    for ((l, u), &c) in lo.iter_mut().zip(hi.iter_mut()).zip(palette) {
        *l = c as u8;
        *u = (c >> 8) as u8;
    }
    let (tlo, thi) = (load128_u8(&lo), load128_u8(&hi));
    for i in 0..h {
        let dst = &mut px[(y + i) * stride + x..][..w];
        let src = &map[i * map_stride..][..w];
        let mut dc = dst.chunks_exact_mut(8);
        let mut sc = src.chunks_exact(8);
        for (d, s) in (&mut dc).zip(&mut sc) {
            if let (Some(d), Some(s)) = (d.first_chunk_mut::<8>(), s.first_chunk::<8>()) {
                let ix = load64_u8(s);
                let v = _mm_unpacklo_epi8(_mm_shuffle_epi8(tlo, ix), _mm_shuffle_epi8(thi, ix));
                store128_u16(d, v);
            }
        }
        for (d, &m) in dc.into_remainder().iter_mut().zip(sc.remainder()) {
            *d = palette[usize::from(m)];
        }
    }
}

/// The CFL alpha step: `CurrFrame = Clip1( dc + Round2Signed( alpha * ac, 6 ) )` with
/// `ac[ i * w + j ]` from `super::cfl_ac` and `max` = ( 1 << BitDepth ) - 1.
#[target_feature(enable = "avx2")]
pub(super) fn cfl(
    chroma: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ac: &[i32],
    alpha: i32,
    max: i32,
) {
    let va = _mm256_set1_epi32(alpha);
    let r32 = _mm256_set1_epi32(32);
    let vmax = _mm256_set1_epi32(max);
    let zero = _mm256_setzero_si256();
    for i in 0..h {
        let row = &mut chroma[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(8).enumerate() {
            let v = _mm256_mullo_epi32(va, ld(ac, i * w + 8 * k));
            // Round2Signed: Round2 of | v |, with the sign of v (0 stays 0).
            let r = _mm256_srai_epi32::<6>(_mm256_add_epi32(_mm256_abs_epi32(v), r32));
            let r = _mm256_sign_epi32(r, v);
            let o = _mm256_add_epi32(ld_u16(d), r);
            st(d, _mm256_min_epi32(_mm256_max_epi32(o, zero), vmax));
        }
    }
}
