//! NEON kernels of the film grain blend (`simd::grain`), 4 samples per step in 32-bit lanes.
//! NEON has no gather: the 4 scales are looked up one by one. The rest of a row (fewer than 4
//! samples, or the chroma samples whose luma pair runs past the luma row) goes to the scalar
//! oracle.

use crate::grain::{Blend, blend_chroma_row, blend_luma_row};
use crate::simd::arm::{load_i32x4, load_u16x4, load_u16x8, store_u16x4};
use std::arch::aarch64::*;

/// `table[idx & 4095]` for each lane.
#[inline]
#[target_feature(enable = "neon")]
fn lookup(table: &[i32; 4096], idx: int32x4_t) -> int32x4_t {
    let at = |i: i32| table[i as usize & 4095];
    load_i32x4(&[
        at(vgetq_lane_s32::<0>(idx)),
        at(vgetq_lane_s32::<1>(idx)),
        at(vgetq_lane_s32::<2>(idx)),
        at(vgetq_lane_s32::<3>(idx)),
    ])
}

/// `clip( orig + round2( scale * noise, ScalingShift ) )` for 4 lanes, narrowed to `u16`.
#[inline]
#[target_feature(enable = "neon")]
fn finish(orig: int32x4_t, scale: int32x4_t, noise: int32x4_t, b: &Blend) -> uint16x4_t {
    // A rounding shift right: `(x + (1 << (s - 1))) >> s`, arithmetic.
    let n = vrshlq_s32(
        vmulq_s32(scale, noise),
        vdupq_n_s32(-(b.scaling_shift as i32)),
    );
    let v = vminq_s32(
        vmaxq_s32(vaddq_s32(orig, n), vdupq_n_s32(b.min)),
        vdupq_n_s32(b.max),
    );
    // The values are in `min..=max`, inside `u16`: the saturating narrow is exact.
    vqmovun_s32(v)
}

/// `grain::blend_luma_row`; the slices have the same length.
#[target_feature(enable = "neon")]
pub(super) fn blend_luma(
    out: &mut [u16],
    src: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    let x = out.len() / 4 * 4;
    let (oc, _) = out.as_chunks_mut::<4>();
    let (sc, _) = src.as_chunks::<4>();
    let (gc, _) = noise.as_chunks::<4>();
    for ((o, s), g) in oc.iter_mut().zip(sc).zip(gc) {
        let orig = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(s)));
        let scale = lookup(table, orig);
        store_u16x4(o, finish(orig, scale, load_i32x4(g), b));
    }
    blend_luma_row(&mut out[x..], &src[x..], &noise[x..], lut, b);
}

/// `grain::blend_chroma_row`; `out`, `src` and `noise` have the same length, at most
/// `(luma.len() + b.sx) >> b.sx`.
#[target_feature(enable = "neon")]
pub(super) fn blend_chroma(
    out: &mut [u16],
    src: &[u16],
    luma: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    // Whole steps only while the 4 (or 8) luma samples are all inside the row, so the
    // `lumaNextX` clamp never applies here.
    let steps = (out.len() / 4).min(luma.len() / (4 << b.sx));
    let (oc, _) = out.as_chunks_mut::<4>();
    let (sc, _) = src.as_chunks::<4>();
    let (gc, _) = noise.as_chunks::<4>();
    let (l4, _) = luma.as_chunks::<4>();
    let (l8, _) = luma.as_chunks::<8>();
    for (i, ((o, s), g)) in oc.iter_mut().zip(sc).zip(gc).take(steps).enumerate() {
        let average_luma = if b.sx == 1 {
            vreinterpretq_s32_u32(vrshrq_n_u32::<1>(vpaddlq_u16(load_u16x8(&l8[i]))))
        } else {
            vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&l4[i])))
        };
        let orig = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(s)));
        let merged = if b.cfl {
            average_luma
        } else {
            let combined = vmlaq_s32(
                vmulq_s32(average_luma, vdupq_n_s32(b.luma_mult)),
                orig,
                vdupq_n_s32(b.mult),
            );
            let m = vaddq_s32(vshrq_n_s32::<6>(combined), vdupq_n_s32(b.offset));
            vminq_s32(vmaxq_s32(m, vdupq_n_s32(0)), vdupq_n_s32(b.pixel_max))
        };
        let scale = lookup(table, merged);
        store_u16x4(o, finish(orig, scale, load_i32x4(g), b));
    }
    let x = steps * 4;
    blend_chroma_row(
        &mut out[x..],
        &src[x..],
        &luma[x << b.sx..],
        &noise[x..],
        lut,
        b,
    );
}
