//! AVX2 kernels of the film grain blend (`simd::grain`), 8 samples per step in 32-bit lanes.
//! The rest of a row (fewer than 8 samples, or the chroma samples whose luma pair runs past
//! the luma row) goes to the scalar oracle.

use crate::grain::{Blend, blend_chroma_row, blend_luma_row};
use crate::simd::x86::{gather256_i32, load128_u16, load256_i32, load256_u16, store128_u16};
use std::arch::x86_64::*;

/// `clip( orig + round2( scale * noise, ScalingShift ) )` for 8 lanes, packed to `u16`.
#[inline]
#[target_feature(enable = "avx2")]
fn finish(orig: __m256i, scale: __m256i, noise: __m256i, b: &Blend) -> __m128i {
    let rnd = _mm256_set1_epi32(1 << (b.scaling_shift - 1));
    let shift = _mm_cvtsi32_si128(b.scaling_shift as i32);
    let n = _mm256_sra_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(scale, noise), rnd),
        shift,
    );
    let v = _mm256_add_epi32(orig, n);
    let v = _mm256_min_epi32(
        _mm256_max_epi32(v, _mm256_set1_epi32(b.min)),
        _mm256_set1_epi32(b.max),
    );
    // The values are in `min..=max`, inside `u16`: the saturating pack is exact.
    _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
}

/// `grain::blend_luma_row`; the slices have the same length.
#[target_feature(enable = "avx2")]
pub(super) fn blend_luma(
    out: &mut [u16],
    src: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    let x = out.len() / 8 * 8;
    let (oc, _) = out.as_chunks_mut::<8>();
    let (sc, _) = src.as_chunks::<8>();
    let (gc, _) = noise.as_chunks::<8>();
    for ((o, s), g) in oc.iter_mut().zip(sc).zip(gc) {
        let orig = _mm256_cvtepu16_epi32(load128_u16(s));
        let scale = gather256_i32(table, orig);
        store128_u16(o, finish(orig, scale, load256_i32(g), b));
    }
    blend_luma_row(&mut out[x..], &src[x..], &noise[x..], lut, b);
}

/// `grain::blend_chroma_row`; `out`, `src` and `noise` have the same length, at most
/// `(luma.len() + b.sx) >> b.sx`.
#[target_feature(enable = "avx2")]
pub(super) fn blend_chroma(
    out: &mut [u16],
    src: &[u16],
    luma: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    // Whole steps only while the 8 (or 16) luma samples are all inside the row, so the
    // `lumaNextX` clamp never applies here.
    let steps = (out.len() / 8).min(luma.len() / (8 << b.sx));
    let (oc, _) = out.as_chunks_mut::<8>();
    let (sc, _) = src.as_chunks::<8>();
    let (gc, _) = noise.as_chunks::<8>();
    let (l8, _) = luma.as_chunks::<8>();
    let (l16, _) = luma.as_chunks::<16>();
    let ones = _mm256_set1_epi16(1);
    for (i, ((o, s), g)) in oc.iter_mut().zip(sc).zip(gc).take(steps).enumerate() {
        let average_luma = if b.sx == 1 {
            // Samples are below 4096, so the signed pair sums of `madd` are exact.
            let sum = _mm256_madd_epi16(load256_u16(&l16[i]), ones);
            _mm256_srli_epi32::<1>(_mm256_add_epi32(sum, _mm256_set1_epi32(1)))
        } else {
            _mm256_cvtepu16_epi32(load128_u16(&l8[i]))
        };
        let orig = _mm256_cvtepu16_epi32(load128_u16(s));
        let merged = if b.cfl {
            average_luma
        } else {
            let combined = _mm256_add_epi32(
                _mm256_mullo_epi32(average_luma, _mm256_set1_epi32(b.luma_mult)),
                _mm256_mullo_epi32(orig, _mm256_set1_epi32(b.mult)),
            );
            let m = _mm256_add_epi32(
                _mm256_srai_epi32::<6>(combined),
                _mm256_set1_epi32(b.offset),
            );
            _mm256_min_epi32(
                _mm256_max_epi32(m, _mm256_setzero_si256()),
                _mm256_set1_epi32(b.pixel_max),
            )
        };
        let scale = gather256_i32(table, merged);
        store128_u16(o, finish(orig, scale, load256_i32(g), b));
    }
    let x = steps * 8;
    blend_chroma_row(
        &mut out[x..],
        &src[x..],
        &luma[x << b.sx..],
        &noise[x..],
        lut,
        b,
    );
}
