//! NEON kernels of the `ipred` family (docs/safety.md). Each step computes 4 samples as i32
//! lanes with the integer arithmetic of the scalar oracle (src/predict.rs), so the output is
//! the same bit for bit. Block widths are multiples of 4 (checked by the dispatch).

use std::arch::aarch64::*;

use crate::consts::{SMOOTH_PRED, SMOOTH_V_PRED};
use crate::predict::{Edge, OFF};
use crate::simd::arm::{
    load_i32x4, load_u8x8, load_u8x16, load_u16x4, store_i32x4, store_u16x4, store_u16x8,
};

/// `e[ k..k + 4 ]` (zero if not all in `e`; the callers stay inside).
#[inline]
#[target_feature(enable = "neon")]
fn ld(e: &[i32], k: usize) -> int32x4_t {
    match e.get(k..).and_then(|s| s.first_chunk::<4>()) {
        Some(a) => load_i32x4(a),
        None => vdupq_n_s32(0),
    }
}

/// Writes the 4 lanes of `v` (each in 0..=65535) to `d` (4 samples).
#[inline]
#[target_feature(enable = "neon")]
fn st(d: &mut [u16], v: int32x4_t) {
    if let Some(a) = d.first_chunk_mut::<4>() {
        store_u16x4(a, vqmovun_s32(v));
    }
}

/// Paeth: `predict::predict_paeth`.
#[target_feature(enable = "neon")]
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
    let tl = vdupq_n_s32(above[OFF - 1]);
    for i in 0..h {
        let l = vdupq_n_s32(left[OFF + i]);
        // pTop = | base - AboveRow[ j ] | = | LeftCol[ i ] - AboveRow[ -1 ] |.
        let p_top = vabsq_s32(vsubq_s32(l, tl));
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(4).enumerate() {
            let a = ld(above, OFF + 4 * k);
            let base = vsubq_s32(vaddq_s32(a, l), tl);
            let p_left = vabsq_s32(vsubq_s32(base, l));
            let p_tl = vabsq_s32(vsubq_s32(base, tl));
            let not_left = vorrq_u32(vcgtq_s32(p_left, p_top), vcgtq_s32(p_left, p_tl));
            let top_or_tl = vbslq_s32(vcgtq_s32(p_top, p_tl), tl, a);
            st(d, vbslq_s32(not_left, top_or_tl, l));
        }
    }
}

/// SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED: `predict::predict_smooth`. `wx` / `wy` are the
/// weights of the width / height (`w` / `h` entries).
#[target_feature(enable = "neon")]
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
    let tr = vdupq_n_s32(above[OFF + w - 1]);
    let bl = left[OFF + h - 1];
    let c256 = vdupq_n_s32(256);
    let r128 = vdupq_n_s32(128);
    for (i, &sy) in wy.iter().enumerate().take(h) {
        let sy = i32::from(sy);
        let vsy = vdupq_n_s32(sy);
        let l = vdupq_n_s32(left[OFF + i]);
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(4).enumerate() {
            let a = ld(above, OFF + 4 * k);
            let sx = ld(&wxs, 4 * k);
            // sx * LeftCol[ i ] + ( 256 - sx ) * AboveRow[ w - 1 ]
            let hor = vaddq_s32(vmulq_s32(sx, l), vmulq_s32(vsubq_s32(c256, sx), tr));
            let v = if mode == SMOOTH_PRED {
                // + sy * AboveRow[ j ] + ( 256 - sy ) * LeftCol[ h - 1 ], Round2( , 9 )
                let ver = vaddq_s32(vmulq_s32(vsy, a), vdupq_n_s32((256 - sy) * bl + 256));
                vshrq_n_s32::<9>(vaddq_s32(ver, hor))
            } else if mode == SMOOTH_V_PRED {
                vshrq_n_s32::<8>(vaddq_s32(
                    vmulq_s32(vsy, a),
                    vdupq_n_s32((256 - sy) * bl + 128),
                ))
            } else {
                vshrq_n_s32::<8>(vaddq_s32(hor, r128))
            };
            st(d, v);
        }
    }
}

/// Directional zone 1 (pAngle < 90) without upsampling: the `p_angle < 90` arm of
/// `predict::directional_ref` with `upsample_above` 0.
#[target_feature(enable = "neon")]
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
    let vfill = vdupq_n_s32(fill);
    let lanes = load_i32x4(&[0, 1, 2, 3]);
    let last = vdupq_n_s32(max_base_x - 1);
    let r16 = vdupq_n_s32(16);
    for i in 0..h {
        let idx = (i as i32 + 1) * dx;
        let base0 = idx >> 6;
        let shift = (idx >> 1) & 0x1f;
        let s1 = vdupq_n_s32(shift);
        let s0 = vdupq_n_s32(32 - shift);
        let row = &mut px[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(4).enumerate() {
            let base = base0 + 4 * k as i32;
            if base >= max_base_x {
                d.fill(fill as u16);
                continue;
            }
            // base < max_base_x <= 127, so the reads end at OFF + 131 < EDGE_LEN.
            let b = OFF + base as usize;
            let v = vaddq_s32(vmulq_s32(ld(above, b), s0), vmulq_s32(ld(above, b + 1), s1));
            let v = vshrq_n_s32::<5>(vaddq_s32(v, r16));
            // Lanes with base + lane >= maxBaseX take AboveRow[ maxBaseX ].
            let over = vcgtq_s32(vaddq_s32(vdupq_n_s32(base), lanes), last);
            st(d, vbslq_s32(over, vfill, v));
        }
    }
}

/// The intra edge filter: `buf[ OFF + i - 1 ] = Round2( sum k[ j ] * p[ i + j ], 4 )` for `i`
/// in 1..sz, where `p` is the padded edge of `super::pad_edge`.
#[target_feature(enable = "neon")]
pub(super) fn filter_edge(buf: &mut Edge, p: &[i32], sz: usize, k: &[i32; 5]) {
    let r8 = vdupq_n_s32(8);
    let mut i = 1;
    while i < sz {
        let mut s = vmulq_n_s32(ld(p, i), k[0]);
        s = vmlaq_n_s32(s, ld(p, i + 1), k[1]);
        s = vmlaq_n_s32(s, ld(p, i + 2), k[2]);
        s = vmlaq_n_s32(s, ld(p, i + 3), k[3]);
        s = vmlaq_n_s32(s, ld(p, i + 4), k[4]);
        let v = vshrq_n_s32::<4>(vaddq_s32(s, r8));
        let n = (sz - i).min(4);
        let d = &mut buf[OFF + i - 1..][..n];
        if let Some(a) = d.first_chunk_mut::<4>() {
            store_i32x4(a, v);
        } else {
            let mut t = [0i32; 4];
            store_i32x4(&mut t, v);
            d.copy_from_slice(&t[..n]);
        }
        i += 4;
    }
}

/// Palette: `predict::palette_ref`, 8 samples per step as two byte table lookups (low and high
/// byte of each colour). `palette` has at most 16 colours; a 4-wide row is scalar.
#[target_feature(enable = "neon")]
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
    let (tlo, thi) = (load_u8x16(&lo), load_u8x16(&hi));
    for i in 0..h {
        let dst = &mut px[(y + i) * stride + x..][..w];
        let src = &map[i * map_stride..][..w];
        let (dc, dr) = dst.as_chunks_mut::<8>();
        let (sc, sr) = src.as_chunks::<8>();
        for (d, s) in dc.iter_mut().zip(sc) {
            let ix = load_u8x8(s);
            let (l8, h8) = (vqtbl1_u8(tlo, ix), vqtbl1_u8(thi, ix));
            // Interleaved low, high bytes: little-endian u16 lanes.
            let v = vcombine_u8(vzip1_u8(l8, h8), vzip2_u8(l8, h8));
            store_u16x8(d, vreinterpretq_u16_u8(v));
        }
        for (d, &m) in dr.iter_mut().zip(sr) {
            *d = palette[usize::from(m)];
        }
    }
}

/// The CFL alpha step: `CurrFrame = Clip1( dc + Round2Signed( alpha * ac, 6 ) )` with
/// `ac[ i * w + j ]` from `super::cfl_ac` and `max` = ( 1 << BitDepth ) - 1.
#[target_feature(enable = "neon")]
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
    let r32 = vdupq_n_s32(32);
    let vmax = vdupq_n_s32(max);
    let zero = vdupq_n_s32(0);
    for i in 0..h {
        let row = &mut chroma[(y + i) * stride + x..][..w];
        for (k, d) in row.chunks_mut(4).enumerate() {
            let Some(d) = d.first_chunk_mut::<4>() else {
                continue;
            };
            let v = vmulq_n_s32(ld(ac, i * w + 4 * k), alpha);
            // Round2Signed: Round2 of | v |, negated where v < 0.
            let r = vshrq_n_s32::<6>(vaddq_s32(vabsq_s32(v), r32));
            let r = vbslq_s32(vcltzq_s32(v), vnegq_s32(r), r);
            let dc = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(d)));
            let o = vminq_s32(vmaxq_s32(vaddq_s32(dc, r), zero), vmax);
            store_u16x4(d, vqmovun_s32(o));
        }
    }
}
