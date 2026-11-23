//! Kernel family `grain`: the film grain noise blend of one row (spec 7.18.3.5, last step):
//! the scaling lookup, `round2( scale * noise, ScalingShift )`, the clip, and for chroma the
//! luma average and the `luma_mult`/`mult`/`offset` merge or chroma scaling from luma.
//! The scalar oracle is `grain::blend_luma_row` / `grain::blend_chroma_row`.
//!
//! The SIMD kernels look the scale up in `grain::scale_table` (one entry per sample value,
//! built from the spec's `scale_lut`), so the high bit depth interpolation is exact by
//! construction. AVX2 gathers 8 lanes; AVX-512 runs the AVX2 kernel (the gather bounds it, so
//! wider vectors do not win); NEON has no gather and looks the 4 lanes up one by one.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use super::{Simd, Tier};
use crate::grain::{Blend, blend_chroma_row, blend_luma_row};

/// [`blend_luma_row`] on the `simd` tier, over the common length of `out`, `src` and `noise`.
/// Samples must be below `1 << b.bit_depth`; `table` is `scale_table( lut, b.bit_depth )`.
#[allow(unsafe_code)]
pub(crate) fn blend_luma(
    simd: Simd,
    out: &mut [u16],
    src: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    let n = out.len().min(src.len()).min(noise.len());
    let (out, src, noise) = (&mut out[..n], &src[..n], &noise[..n]);
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `simd` is the proof that AVX2 was detected (`simd::available`); the AVX-512
        // tier is only detected together with AVX2.
        Tier::Avx2 | Tier::Avx512 => unsafe { avx2::blend_luma(out, src, noise, lut, table, b) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `simd` is the proof that NEON was detected (`simd::available`).
        Tier::Neon => unsafe { neon::blend_luma(out, src, noise, lut, table, b) },
        _ => blend_luma_row(out, src, noise, lut, b),
    }
}

/// [`blend_chroma_row`] on the `simd` tier, over the common length of `out`, `src` and
/// `noise`, capped at `(luma.len() + b.sx) >> b.sx`. Samples must be below
/// `1 << b.bit_depth`; `table` is `scale_table( lut, b.bit_depth )`.
#[allow(unsafe_code)]
pub(crate) fn blend_chroma(
    simd: Simd,
    out: &mut [u16],
    src: &[u16],
    luma: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    table: &[i32; 4096],
    b: &Blend,
) {
    let n = out
        .len()
        .min(src.len())
        .min(noise.len())
        .min((luma.len() + b.sx) >> b.sx);
    let (out, src, noise) = (&mut out[..n], &src[..n], &noise[..n]);
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `simd` is the proof that AVX2 was detected (`simd::available`); the AVX-512
        // tier is only detected together with AVX2.
        Tier::Avx2 | Tier::Avx512 => unsafe {
            avx2::blend_chroma(out, src, luma, noise, lut, table, b)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `simd` is the proof that NEON was detected (`simd::available`).
        Tier::Neon => unsafe { neon::blend_chroma(out, src, luma, noise, lut, table, b) },
        _ => blend_chroma_row(out, src, luma, noise, lut, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grain::scale_table;
    use crate::simd::available;

    struct XorShift(u64);

    impl XorShift {
        fn below(&mut self, n: u32) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % u64::from(n)) as u32
        }
    }

    /// Random blend parameters for `bit_depth` and chroma subsampling `sx`.
    fn params(r: &mut XorShift, bit_depth: u32, sx: usize) -> Blend {
        let sh = bit_depth - 8;
        let pixel_max = (1i32 << bit_depth) - 1;
        let (min, max) = match r.below(3) {
            0 => (16 << sh, 235 << sh),
            1 => (16 << sh, 240 << sh),
            _ => (0, pixel_max),
        };
        Blend {
            bit_depth,
            pixel_max,
            scaling_shift: 8 + r.below(4),
            min,
            max,
            sx,
            cfl: r.below(3) == 0,
            luma_mult: r.below(256) as i32 - 128,
            mult: r.below(256) as i32 - 128,
            offset: (r.below(512) as i32 - 256) << sh,
        }
    }

    /// A sample: mostly random, often 0 or the maximum.
    fn sample(r: &mut XorShift, bit_depth: u32) -> u16 {
        let max = (1u32 << bit_depth) - 1;
        match r.below(6) {
            0 => 0,
            1 => max as u16,
            _ => r.below(max + 1) as u16,
        }
    }

    /// Noise in the grain range of `bit_depth`, often at its ends.
    fn noise(r: &mut XorShift, bit_depth: u32) -> i32 {
        let c = 128i32 << (bit_depth - 8);
        match r.below(6) {
            0 => -c,
            1 => c - 1,
            _ => r.below(2 * c as u32) as i32 - c,
        }
    }

    /// Runs both kernels on every tier for each length (and each start offset 0..4) and
    /// compares with the scalar oracle, output buffers included past the row.
    #[test]
    fn every_tier_matches_scalar() {
        let mut r = XorShift(0x2545_f491_4f6c_dd1d);
        let lens = [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 67];
        for bit_depth in [8u32, 10, 12] {
            for &n in &lens {
                for start in 0..4 {
                    for sx in [0usize, 1] {
                        let b = params(&mut r, bit_depth, sx);
                        let mut lut = [0i32; 256];
                        for v in lut.iter_mut() {
                            *v = r.below(256) as i32;
                        }
                        if r.below(4) == 0 {
                            lut = [255; 256];
                        }
                        let table = scale_table(&lut, bit_depth);
                        let total = start + n + 3;
                        let src: Vec<u16> = (0..total).map(|_| sample(&mut r, bit_depth)).collect();
                        let nz: Vec<i32> = (0..total).map(|_| noise(&mut r, bit_depth)).collect();
                        // An odd luma width when it can be (the last pair reads one sample twice).
                        let luma_len = (n << sx).saturating_sub(sx * r.below(2) as usize);
                        let luma: Vec<u16> = (0..start + luma_len)
                            .map(|_| sample(&mut r, bit_depth))
                            .collect();
                        let fill: Vec<u16> = (0..total).map(|_| r.below(65536) as u16).collect();
                        let (src, nz, luma) = (&src[start..], &nz[start..], &luma[start..]);

                        let mut want_y = fill.clone();
                        blend_luma_row(&mut want_y[start..start + n], src, nz, &lut, &b);
                        let mut want_c = fill.clone();
                        blend_chroma_row(&mut want_c[start..start + n], src, luma, nz, &lut, &b);
                        for simd in available() {
                            let mut got = fill.clone();
                            let row = &mut got[start..start + n];
                            blend_luma(simd, row, src, nz, &lut, &table, &b);
                            assert_eq!(got, want_y, "luma {simd:?} bd {bit_depth} n {n} {b:?}");
                            let mut got = fill.clone();
                            let row = &mut got[start..start + n];
                            blend_chroma(simd, row, src, luma, nz, &lut, &table, &b);
                            assert_eq!(got, want_c, "chroma {simd:?} bd {bit_depth} n {n} {b:?}");
                        }
                    }
                }
            }
        }
    }

    /// `scale_table` is `scale_lut` at every sample value (the interpolation at 10 and 12 bits).
    #[test]
    fn scale_table_matches_scale_lut() {
        let mut r = XorShift(7);
        let mut lut = [0i32; 256];
        for v in lut.iter_mut() {
            *v = r.below(256) as i32;
        }
        for bit_depth in [8u32, 10, 12] {
            let table = scale_table(&lut, bit_depth);
            let mut want = vec![0u16; 1 << bit_depth];
            let mut got = want.clone();
            let src: Vec<u16> = (0..1u32 << bit_depth).map(|v| v as u16).collect();
            let nz = vec![1i32 << 11; src.len()];
            let b = Blend {
                bit_depth,
                pixel_max: (1 << bit_depth) - 1,
                scaling_shift: 8,
                min: 0,
                max: i32::MAX,
                sx: 0,
                cfl: false,
                luma_mult: 0,
                mult: 0,
                offset: 0,
            };
            blend_luma_row(&mut want, &src, &nz, &lut, &b);
            for simd in available() {
                blend_luma(simd, &mut got, &src, &nz, &lut, &table, &b);
                assert_eq!(got, want, "{simd:?} bd {bit_depth}");
            }
        }
    }
}
