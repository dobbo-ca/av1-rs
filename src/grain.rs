//! Film grain synthesis (spec 7.18.3), applied to a copy of the output frame.
//!
//! The code follows the spec text one to one; the noise stripes are built one at a time and
//! blended into a full noise image per plane, which gives the same result as the spec's
//! whole-frame `noiseStripe` arrays.
#![allow(dead_code)]

use crate::consts::MC_IDENTITY;
use crate::error::Error;
use crate::header::FilmGrainParams;
use crate::ledger::Ledger;
use crate::tables::GAUSSIAN_SEQUENCE;

/// Frame data the film grain process reads (spec 7.18.3 inputs).
///
/// `width` and `height` are the output (upscaled) luma size. Each present plane holds at least
/// `(rows - 1) * stride + cols` samples, where chroma `cols`/`rows` are
/// `(width + subx) >> subx` and `(height + suby) >> suby`.
pub(crate) struct GrainInput<'a> {
    pub bit_depth: u32,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub matrix_coefficients: u8,
    pub width: usize,
    pub height: usize,
    pub planes: [&'a [u16]; 3],
    pub strides: [usize; 3],
}

const LUMA_W: usize = 82;
const LUMA_H: usize = 73;

/// Spec Round2 for signed values (arithmetic shift); `n == 0` returns `x`.
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// Spec 7.18.3.2: the 16-bit LFSR behind `get_random_number`.
struct Rng(u32);

impl Rng {
    /// Spec 7.18.3.2 `get_random_number( bits )`, `bits` in 1..=16.
    fn get(&mut self, bits: u32) -> u32 {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r;
        (r >> (16 - bits)) & ((1 << bits) - 1)
    }
}

/// The three grain templates (spec LumaGrain, CbGrain, CrGrain) with their row strides.
struct Grain {
    data: [Vec<i32>; 3],
    stride: [usize; 3],
}

/// Spec 7.18.3.3: white noise from the Gaussian table, then the auto-regressive filter.
fn generate_grain(p: &FilmGrainParams, inp: &GrainInput, grain_min: i32, grain_max: i32) -> Grain {
    let seed = u32::from(p.grain_seed);
    let sub_x = usize::from(inp.subsampling_x);
    let sub_y = usize::from(inp.subsampling_y);
    let white_shift = 12u32.saturating_sub(inp.bit_depth) + u32::from(p.grain_scale_shift);
    let ar_shift = u32::from(p.ar_coeff_shift_minus_6) + 6;
    let lag = usize::from(p.ar_coeff_lag).min(3);
    let chroma_w = if inp.subsampling_x { 44 } else { LUMA_W };
    let chroma_h = if inp.subsampling_y { 38 } else { LUMA_H };

    let mut rng = Rng(seed);
    let mut luma = vec![0i32; LUMA_W * LUMA_H];
    if p.num_y_points > 0 {
        for v in luma.iter_mut() {
            *v = round2(
                i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize]),
                white_shift,
            );
        }
    }
    // Spec: the filter runs even when the luma noise is all zero; the result is then unused.
    for y in 3..LUMA_H {
        for x in 3..LUMA_W - 3 {
            let mut s = 0i32;
            let mut pos = 0;
            for dr in 0..=lag {
                for dc in 0..=2 * lag {
                    if dr == lag && dc == lag {
                        break;
                    }
                    let c = i32::from(p.ar_coeffs_y_plus_128[pos]) - 128;
                    s += luma[(y + dr - lag) * LUMA_W + x + dc - lag] * c;
                    pos += 1;
                }
            }
            let i = y * LUMA_W + x;
            luma[i] = (luma[i] + round2(s, ar_shift)).clamp(grain_min, grain_max);
        }
    }

    let mut cb = Vec::new();
    let mut cr = Vec::new();
    if !inp.mono_chrome {
        let make = |seed_xor: u32, wanted: bool| {
            let mut rng = Rng(seed ^ seed_xor);
            let mut v = vec![0i32; chroma_w * chroma_h];
            if wanted {
                for g in v.iter_mut() {
                    *g = round2(
                        i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize]),
                        white_shift,
                    );
                }
            }
            v
        };
        cb = make(0xb524, p.num_cb_points > 0 || p.chroma_scaling_from_luma);
        cr = make(0x49d8, p.num_cr_points > 0 || p.chroma_scaling_from_luma);
        for y in 3..chroma_h {
            for x in 3..chroma_w - 3 {
                let mut s0 = 0i32;
                let mut s1 = 0i32;
                let mut pos = 0;
                for dr in 0..=lag {
                    for dc in 0..=2 * lag {
                        let c0 = i32::from(p.ar_coeffs_cb_plus_128[pos]) - 128;
                        let c1 = i32::from(p.ar_coeffs_cr_plus_128[pos]) - 128;
                        if dr == lag && dc == lag {
                            if p.num_y_points > 0 {
                                let luma_x = ((x - 3) << sub_x) + 3;
                                let luma_y = ((y - 3) << sub_y) + 3;
                                let mut l = 0;
                                for i in 0..=sub_y {
                                    for j in 0..=sub_x {
                                        l += luma[(luma_y + i) * LUMA_W + luma_x + j];
                                    }
                                }
                                let l = round2(l, (sub_x + sub_y) as u32);
                                s0 += l * c0;
                                s1 += l * c1;
                            }
                            break;
                        }
                        let idx = (y + dr - lag) * chroma_w + x + dc - lag;
                        s0 += cb[idx] * c0;
                        s1 += cr[idx] * c1;
                        pos += 1;
                    }
                }
                let i = y * chroma_w + x;
                cb[i] = (cb[i] + round2(s0, ar_shift)).clamp(grain_min, grain_max);
                cr[i] = (cr[i] + round2(s1, ar_shift)).clamp(grain_min, grain_max);
            }
        }
    }
    Grain {
        data: [luma, cb, cr],
        stride: [LUMA_W, chroma_w, chroma_w],
    }
}

/// Spec 7.18.3.4: piecewise linear scaling table for `plane` (0 = Y, 1 = Cb, 2 = Cr).
fn scaling_lut(p: &FilmGrainParams, plane: usize) -> [i32; 256] {
    let (num, xs, ys) = if plane == 0 || p.chroma_scaling_from_luma {
        (p.num_y_points, &p.point_y_value, &p.point_y_scaling)
    } else if plane == 1 {
        (p.num_cb_points, &p.point_cb_value, &p.point_cb_scaling)
    } else {
        (p.num_cr_points, &p.point_cr_value, &p.point_cr_scaling)
    };
    let num = usize::from(num).min(16);
    let mut lut = [0i32; 256];
    if num == 0 {
        return lut;
    }
    let x_at = |i: usize| usize::from(xs[i]);
    let y_at = |i: usize| i32::from(ys[i]);
    for v in lut.iter_mut().take(x_at(0)) {
        *v = y_at(0);
    }
    for i in 0..num - 1 {
        let delta_y = y_at(i + 1) - y_at(i);
        let delta_x = x_at(i + 1) as i32 - x_at(i) as i32;
        // Point values must increase; a zero or negative step (hostile header) adds nothing.
        if delta_x <= 0 {
            continue;
        }
        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            lut[x_at(i) + x as usize] = y_at(i) + ((x * delta + 32768) >> 16);
        }
    }
    for v in lut.iter_mut().skip(x_at(num - 1)) {
        *v = y_at(num - 1);
    }
    lut
}

/// Spec `scale_lut( plane, index )`, `index` in 0..(1 << bit_depth).
fn scale_lut(lut: &[i32; 256], bit_depth: u32, index: i32) -> i32 {
    let shift = bit_depth - 8;
    let x = (index >> shift) as usize;
    let rem = index - ((x as i32) << shift);
    if bit_depth == 8 || x >= 255 {
        lut[x.min(255)]
    } else {
        let start = lut[x];
        let end = lut[x + 1];
        start + round2((end - start) * rem, shift)
    }
}

/// Spec 7.18.3.
///
/// Writes the frame with grain into `out` (same layout as the input planes, strides as in
/// `input`); every sample of the `width` x `height` area of each present plane is written.
/// Planes 1 and 2 are ignored for monochrome. The caller guarantees `bit_depth` in 8..=12 and
/// that every plane slice (input and output) covers its area as described on [`GrainInput`].
///
/// Note on dav1d: the spec's order is followed (chroma noise uses the luma samples before luma
/// noise is added; odd sizes read the last luma column twice through `lumaNextX`). No place was
/// found where it differs from dav1d's `filmgrain_tmpl.c` output.
///
/// The noise images and stripe buffers are charged to `ledger` for the duration of the call and
/// allocated fallibly.
pub(crate) fn apply_film_grain(
    p: &FilmGrainParams,
    input: &GrainInput,
    out: [&mut [u16]; 3],
    ledger: &Ledger,
) -> Result<(), Error> {
    let bd = input.bit_depth;
    let grain_center = 128i32 << (bd - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bd - 8)) - 1 - grain_center;
    let num_planes = if input.mono_chrome { 1 } else { 3 };
    let (w, h) = (input.width, input.height);
    let seed = u32::from(p.grain_seed);

    let grain = generate_grain(p, input, grain_min, grain_max);
    let luts = [scaling_lut(p, 0), scaling_lut(p, 1), scaling_lut(p, 2)];

    // Spec 7.18.3.5, first two steps: noise stripes blended into one noise image per plane.
    let sub = |plane: usize| -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (
                usize::from(input.subsampling_x),
                usize::from(input.subsampling_y),
            )
        }
    };
    let dims = |plane: usize| -> (usize, usize) {
        let (sx, sy) = sub(plane);
        ((w + sx) >> sx, (h + sy) >> sy)
    };
    let sw = (w + 1).div_ceil(2).div_ceil(16) * 32 + 34;
    // Charge the noise images and two stripe buffers (`cur`, `prev`) per plane up front.
    let mut samples = 2 * 34 * sw * num_planes;
    for plane in 0..num_planes {
        let (pw, ph) = dims(plane);
        samples += pw * ph;
    }
    let _held = ledger.charge(
        samples
            .checked_mul(size_of::<i32>())
            .ok_or(Error::AllocLimit)?,
    )?;
    let mut noise: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for plane in 0..num_planes {
        let (pw, ph) = dims(plane);
        noise[plane] = zeros(pw * ph)?;
    }
    let mut prev: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (luma_num, _) in (0..h.div_ceil(2)).step_by(16).enumerate() {
        let mut cur: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for plane in 0..num_planes {
            cur[plane] = zeros(34 * sw)?;
        }
        let mut rng = Rng(seed);
        rng.0 ^= (((luma_num * 37 + 178) & 255) << 8) as u32;
        rng.0 ^= ((luma_num * 173 + 105) & 255) as u32;
        for x in (0..w.div_ceil(2)).step_by(16) {
            let rand = rng.get(8) as usize;
            let (offset_x, offset_y) = (rand >> 4, rand & 15);
            for plane in 0..num_planes {
                let (psx, psy) = sub(plane);
                let plane_off_x = if psx == 1 {
                    6 + offset_x
                } else {
                    9 + offset_x * 2
                };
                let plane_off_y = if psy == 1 {
                    6 + offset_y
                } else {
                    9 + offset_y * 2
                };
                let gs = grain.stride[plane];
                for i in 0..34 >> psy {
                    for j in 0..34 >> psx {
                        let mut g = grain.data[plane][(plane_off_y + i) * gs + plane_off_x + j];
                        let col = if psx == 0 { x * 2 + j } else { x + j };
                        let slot = i * sw + col;
                        if p.overlap_flag && x > 0 {
                            let old = cur[plane][slot];
                            if psx == 0 && j < 2 {
                                g = if j == 0 {
                                    old * 27 + g * 17
                                } else {
                                    old * 17 + g * 27
                                };
                                g = round2(g, 5).clamp(grain_min, grain_max);
                            } else if psx == 1 && j == 0 {
                                g = round2(old * 23 + g * 22, 5).clamp(grain_min, grain_max);
                            }
                        }
                        cur[plane][slot] = g;
                    }
                }
            }
        }
        for plane in 0..num_planes {
            let (_, psy) = sub(plane);
            let (pw, ph) = dims(plane);
            let first = luma_num << (5 - psy);
            let last = ph.min((luma_num + 1) << (5 - psy));
            for y in first..last {
                let i = y - first;
                for x in 0..pw {
                    let mut g = cur[plane][i * sw + x];
                    if p.overlap_flag && luma_num > 0 {
                        if psy == 0 && i < 2 {
                            let old = prev[plane][(i + 32) * sw + x];
                            g = if i == 0 {
                                old * 27 + g * 17
                            } else {
                                old * 17 + g * 27
                            };
                            g = round2(g, 5).clamp(grain_min, grain_max);
                        } else if psy == 1 && i < 1 {
                            let old = prev[plane][(i + 16) * sw + x];
                            g = round2(old * 23 + g * 22, 5).clamp(grain_min, grain_max);
                        }
                    }
                    noise[plane][y * pw + x] = g;
                }
            }
        }
        prev = cur;
    }

    // Spec 7.18.3.5, last step: blend the noise with the image.
    let (min_value, max_luma, max_chroma) = if p.clip_to_restricted_range {
        let max_luma = 235i32 << (bd - 8);
        let max_chroma = if input.matrix_coefficients == MC_IDENTITY {
            max_luma
        } else {
            240i32 << (bd - 8)
        };
        (16i32 << (bd - 8), max_luma, max_chroma)
    } else {
        let m = (256i32 << (bd - 8)) - 1;
        (0, m, m)
    };
    let pixel_max = (1i32 << bd) - 1;
    let scaling_shift = u32::from(p.grain_scaling_minus_8) + 8;

    if !input.mono_chrome {
        let (sx, sy) = sub(1);
        let (cw, ch) = dims(1);
        let params = [
            (p.num_cb_points, p.cb_luma_mult, p.cb_mult, p.cb_offset),
            (p.num_cr_points, p.cr_luma_mult, p.cr_mult, p.cr_offset),
        ];
        for (k, &(num_points, luma_mult, mult, offset)) in params.iter().enumerate() {
            let plane = k + 1;
            let stride = input.strides[plane];
            let apply = num_points > 0 || p.chroma_scaling_from_luma;
            for y in 0..ch {
                for x in 0..cw {
                    let orig = i32::from(input.planes[plane][y * stride + x]);
                    let mut value = orig;
                    if apply {
                        let luma_x = x << sx;
                        let luma_y = y << sy;
                        let luma_next_x = (luma_x + 1).min(w - 1);
                        let row = &input.planes[0][luma_y * input.strides[0]..];
                        let average_luma = if sx == 1 {
                            round2(i32::from(row[luma_x]) + i32::from(row[luma_next_x]), 1)
                        } else {
                            i32::from(row[luma_x])
                        };
                        let merged = if p.chroma_scaling_from_luma {
                            average_luma
                        } else {
                            let combined = average_luma * (i32::from(luma_mult) - 128)
                                + orig * (i32::from(mult) - 128);
                            ((combined >> 6) + ((i32::from(offset) - 256) << (bd - 8)))
                                .clamp(0, pixel_max)
                        };
                        let n = noise[plane][y * cw + x];
                        let n = round2(scale_lut(&luts[plane], bd, merged) * n, scaling_shift);
                        value = (orig + n).clamp(min_value, max_chroma);
                    }
                    out[plane][y * stride + x] = value as u16;
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let orig = i32::from(input.planes[0][y * input.strides[0] + x]);
            let mut value = orig;
            if p.num_y_points > 0 {
                let n = round2(
                    scale_lut(&luts[0], bd, orig) * noise[0][y * w + x],
                    scaling_shift,
                );
                value = (orig + n).clamp(min_value, max_luma);
            }
            out[0][y * input.strides[0] + x] = value as u16;
        }
    }
    Ok(())
}

/// `n` zeros, allocated fallibly (the bytes are charged by the caller).
fn zeros(n: usize) -> Result<Vec<i32>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, 0);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_known_steps() {
        // Seed 1: feedback bit is 1, so the register becomes 0x8000.
        let mut r = Rng(1);
        assert_eq!(r.get(8), 0x80);
        assert_eq!(r.0, 0x8000);
        // Bits 0, 1, 3, 12 are all zero: shift only.
        assert_eq!(r.get(11), 0x4000 >> 5);
        assert_eq!(r.0, 0x4000);
    }

    #[test]
    fn scaling_lut_two_points() {
        let mut point_y_value = [0u8; 16];
        let mut point_y_scaling = [0u8; 16];
        point_y_value[1] = 100;
        point_y_scaling[1] = 100;
        let p = FilmGrainParams {
            num_y_points: 2,
            point_y_value,
            point_y_scaling,
            ..Default::default()
        };
        let lut = scaling_lut(&p, 0);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[1], 1);
        assert_eq!(lut[50], 50);
        assert_eq!(lut[99], 99);
        assert_eq!(lut[100], 100);
        assert_eq!(lut[255], 100);
        // 10-bit lookup interpolates between neighbouring entries.
        assert_eq!(scale_lut(&lut, 10, 4 * 50 + 2), 51);
    }

    #[test]
    fn no_points_leaves_planes_unchanged() {
        let (w, h) = (17usize, 13usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y: Vec<u16> = (0..w * h).map(|i| (i * 7 % 256) as u16).collect();
        let u: Vec<u16> = (0..cw * ch).map(|i| (i * 5 % 256) as u16).collect();
        let v: Vec<u16> = (0..cw * ch).map(|i| (i * 3 % 256) as u16).collect();
        let input = GrainInput {
            bit_depth: 8,
            mono_chrome: false,
            subsampling_x: true,
            subsampling_y: true,
            matrix_coefficients: 2,
            width: w,
            height: h,
            planes: [&y, &u, &v],
            strides: [w, cw, cw],
        };
        let p = FilmGrainParams {
            apply_grain: true,
            grain_seed: 1234,
            ..Default::default()
        };
        let (mut oy, mut ou, mut ov) = (
            vec![0u16; y.len()],
            vec![0u16; u.len()],
            vec![0u16; v.len()],
        );
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        apply_film_grain(&p, &input, [&mut oy, &mut ou, &mut ov], &ledger).unwrap();
        let tiny = Ledger::default();
        tiny.reset(crate::ledger::SMALL, 0).unwrap();
        let (mut ty, mut tu, mut tv) = (oy.clone(), ou.clone(), ov.clone());
        assert_eq!(
            apply_film_grain(&p, &input, [&mut ty, &mut tu, &mut tv], &tiny),
            Err(Error::AllocLimit)
        );
        assert_eq!(oy, y);
        assert_eq!(ou, u);
        assert_eq!(ov, v);
    }
}
