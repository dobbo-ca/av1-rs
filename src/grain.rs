//! Film grain synthesis (spec 7.18.3), applied to a copy of the output frame.
//!
//! The code follows the spec text one to one. Each band of 32 luma rows builds its noise stripe
//! (and, with overlap, the stripe above it) and blends it into its rows of the image, which
//! gives the same result as the spec's whole-frame `noiseStripe` and `noiseImage` arrays.
//! The blend of a row is the `grain` SIMD kernel family (`simd::grain`); the scalar row fns
//! here are its oracle.
#![allow(dead_code)]

use crate::consts::MC_IDENTITY;
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::header::FilmGrainParams;
use crate::ledger::{Buf, Ledger};
use crate::parallel::run_each;
use crate::simd::Simd;
use crate::simd::grain::{blend_chroma, blend_luma};
use crate::tables::GAUSSIAN_SEQUENCE;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

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

/// The three grain templates (spec LumaGrain, CbGrain, CrGrain) with their row strides (on the
/// stack: 72 KB; a chroma template uses the start of its array).
struct Grain {
    data: [[i32; LUMA_W * LUMA_H]; 3],
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
    let mut luma = [0i32; LUMA_W * LUMA_H];
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

    let mut cb = [0i32; LUMA_W * LUMA_H];
    let mut cr = [0i32; LUMA_W * LUMA_H];
    if !inp.mono_chrome {
        let make = |v: &mut [i32], seed_xor: u32, wanted: bool| {
            let mut rng = Rng(seed ^ seed_xor);
            if wanted {
                for g in v[..chroma_w * chroma_h].iter_mut() {
                    *g = round2(
                        i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize]),
                        white_shift,
                    );
                }
            }
        };
        make(
            &mut cb[..],
            0xb524,
            p.num_cb_points > 0 || p.chroma_scaling_from_luma,
        );
        make(
            &mut cr[..],
            0x49d8,
            p.num_cr_points > 0 || p.chroma_scaling_from_luma,
        );
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

/// `scale_lut( plane, v )` for every `v` below `1 << bit_depth` (`bit_depth` in 8..=12), for
/// the SIMD kernels, which look the scale up by sample value.
pub(crate) fn scale_table(lut: &[i32; 256], bit_depth: u32) -> [i32; 4096] {
    let mut t = [0i32; 4096];
    for (v, s) in t.iter_mut().enumerate().take(1 << bit_depth) {
        *s = scale_lut(lut, bit_depth, v as i32);
    }
    t
}

/// The parameters of the noise blend of one plane (spec 7.18.3.5, last step).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Blend {
    /// 8..=12.
    pub bit_depth: u32,
    /// `(1 << bit_depth) - 1`.
    pub pixel_max: i32,
    /// `ScalingShift`, 8..=11.
    pub scaling_shift: u32,
    /// The clip range of the result (`minValue`, `maxLuma` or `maxChroma`).
    pub min: i32,
    pub max: i32,
    /// Chroma only: the horizontal subsampling (0 or 1), `chroma_scaling_from_luma`,
    /// `luma_mult - 128`, `mult - 128` and `(offset - 256) << (bit_depth - 8)`.
    pub sx: usize,
    pub cfl: bool,
    pub luma_mult: i32,
    pub mult: i32,
    pub offset: i32,
}

/// Spec 7.18.3.5, last step, one luma row: `out[x]` from `src[x]` and `noise[x]`. The scalar
/// oracle of `simd::grain::blend_luma`; the slices have the same length, samples are below
/// `1 << bit_depth`.
pub(crate) fn blend_luma_row(
    out: &mut [u16],
    src: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    b: &Blend,
) {
    for ((o, &s), &g) in out.iter_mut().zip(src).zip(noise) {
        let orig = i32::from(s);
        let n = round2(scale_lut(lut, b.bit_depth, orig) * g, b.scaling_shift);
        *o = (orig + n).clamp(b.min, b.max) as u16;
    }
}

/// Spec 7.18.3.5, last step, one chroma row: `out[x]` from `src[x]`, `noise[x]` and the luma
/// row `luma` (the whole row of the frame width, or its tail from `x0 << sx` for an `out`
/// starting at `x0`). The scalar oracle of `simd::grain::blend_chroma`; `out`, `src` and
/// `noise` have the same length, at most `(luma.len() + sx) >> sx`.
pub(crate) fn blend_chroma_row(
    out: &mut [u16],
    src: &[u16],
    luma: &[u16],
    noise: &[i32],
    lut: &[i32; 256],
    b: &Blend,
) {
    let last = luma.len().saturating_sub(1);
    for x in 0..out.len() {
        let orig = i32::from(src[x]);
        let luma_x = x << b.sx;
        let average_luma = if b.sx == 1 {
            round2(
                i32::from(luma[luma_x]) + i32::from(luma[(luma_x + 1).min(last)]),
                1,
            )
        } else {
            i32::from(luma[luma_x])
        };
        let merged = if b.cfl {
            average_luma
        } else {
            let combined = average_luma * b.luma_mult + orig * b.mult;
            ((combined >> 6) + b.offset).clamp(0, b.pixel_max)
        };
        let n = round2(
            scale_lut(lut, b.bit_depth, merged) * noise[x],
            b.scaling_shift,
        );
        out[x] = (orig + n).clamp(b.min, b.max) as u16;
    }
}

/// The blend parameters of the three planes and whether each plane gets noise.
fn blend_params(p: &FilmGrainParams, input: &GrainInput) -> ([Blend; 3], [bool; 3]) {
    let bd = input.bit_depth;
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
    let luma = Blend {
        bit_depth: bd,
        pixel_max: (1i32 << bd) - 1,
        scaling_shift: u32::from(p.grain_scaling_minus_8) + 8,
        min: min_value,
        max: max_luma,
        sx: 0,
        cfl: false,
        luma_mult: 0,
        mult: 0,
        offset: 0,
    };
    let chroma = |luma_mult: u8, mult: u8, offset: u16| Blend {
        max: max_chroma,
        sx: usize::from(input.subsampling_x),
        cfl: p.chroma_scaling_from_luma,
        luma_mult: i32::from(luma_mult) - 128,
        mult: i32::from(mult) - 128,
        offset: (i32::from(offset) - 256) << (bd - 8),
        ..luma
    };
    (
        [
            luma,
            chroma(p.cb_luma_mult, p.cb_mult, p.cb_offset),
            chroma(p.cr_luma_mult, p.cr_mult, p.cr_offset),
        ],
        [
            p.num_y_points > 0,
            p.num_cb_points > 0 || p.chroma_scaling_from_luma,
            p.num_cr_points > 0 || p.chroma_scaling_from_luma,
        ],
    )
}

/// Spec 7.18.3.5, first step: the noise stripe of luma block row `luma_num` for each plane
/// below `num_planes`, into `stripe` (34 rows of `sw` samples per plane, plane subsampling
/// `subs`). The generator is seeded per stripe, so stripes are independent of each other.
fn build_stripe(
    luma_num: usize,
    p: &FilmGrainParams,
    grain: &Grain,
    subs: [(usize, usize); 3],
    num_planes: usize,
    w: usize,
    sw: usize,
    grain_min: i32,
    grain_max: i32,
    stripe: &mut [impl std::ops::DerefMut<Target = [i32]>; 3],
) {
    let mut rng = Rng(u32::from(p.grain_seed));
    rng.0 ^= (((luma_num * 37 + 178) & 255) << 8) as u32;
    rng.0 ^= ((luma_num * 173 + 105) & 255) as u32;
    for x in (0..w.div_ceil(2)).step_by(16) {
        let rand = rng.get(8) as usize;
        let (offset_x, offset_y) = (rand >> 4, rand & 15);
        for plane in 0..num_planes {
            let (psx, psy) = subs[plane];
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
                        let old = stripe[plane][slot];
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
                    stripe[plane][slot] = g;
                }
            }
        }
    }
}

/// Spec 7.18.3.
///
/// Writes the frame with grain into `out` (same layout as the input planes, strides as in
/// `input`); every sample of the `width` x `height` area of each present plane is written.
/// Planes 1 and 2 are ignored for monochrome. The caller guarantees `bit_depth` in 8..=12 and
/// samples below `1 << bit_depth`; a plane slice (input or output) that does not cover its
/// area as described on [`GrainInput`] is `Error::Invalid`.
///
/// Note on dav1d: the spec's order is followed (chroma noise uses the luma samples before luma
/// noise is added; odd sizes read the last luma column twice through `lumaNextX`). No place was
/// found where it differs from dav1d's `filmgrain_tmpl.c` output.
///
/// The frame is done in bands of 32 luma rows (one noise stripe each) on up to `max_threads`
/// threads (`parallel::run_each`, one scope). A band builds its own stripe and, with
/// overlap, the stripe above it (each stripe seeds its own generator), so the bands are
/// independent and the output is the same for every thread count. The stripe buffers of a band
/// come from the ledger's pool (or are allocated fallibly) and are charged to `ledger` while
/// they live; `cancel` is checked before each band. `simd` picks the blend kernels.
///
/// How many bands hold their stripes at once depends on timing, so a refusal (`AllocLimit`,
/// `OutOfMemory`) with more than one thread runs the bands again on one thread, after the pool
/// is emptied: the result is the one of `max_threads = 1`, whose refusal is final (what the
/// pool holds does not change whether a charge fits, see [`Ledger::take`]). Every output sample
/// is written again, so nothing of the refused run is left.
pub(crate) fn apply_film_grain(
    p: &FilmGrainParams,
    input: &GrainInput,
    mut out: [&mut [u16]; 3],
    ledger: &Ledger,
    cancel: &AtomicBool,
    max_threads: usize,
    simd: Simd,
) -> Result<(), Error> {
    let r = grain_bands(
        p,
        input,
        out.each_mut().map(|o| &mut **o),
        ledger,
        cancel,
        max_threads,
        simd,
    );
    match r {
        // One band (32 rows or fewer) already ran serially: its refusal is final.
        Err(Error::AllocLimit | Error::OutOfMemory) if max_threads > 1 && input.height > 32 => {
            ledger.evict();
            grain_bands(p, input, out, ledger, cancel, 1, simd)
        }
        r => r,
    }
}

/// [`apply_film_grain`] on `max_threads` threads, without the retry.
fn grain_bands(
    p: &FilmGrainParams,
    input: &GrainInput,
    out: [&mut [u16]; 3],
    ledger: &Ledger,
    cancel: &AtomicBool,
    max_threads: usize,
    simd: Simd,
) -> Result<(), Error> {
    let bd = input.bit_depth;
    let grain_center = 128i32 << (bd - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bd - 8)) - 1 - grain_center;
    let num_planes = if input.mono_chrome { 1 } else { 3 };
    let (w, h) = (input.width, input.height);
    if w == 0 || h == 0 {
        return Ok(());
    }
    let cs = (
        usize::from(input.subsampling_x),
        usize::from(input.subsampling_y),
    );
    let subs = [(0, 0), cs, cs];
    let dims = |plane: usize| -> (usize, usize) {
        let (sx, sy) = subs[plane];
        ((w + sx) >> sx, (h + sy) >> sy)
    };
    // Every row read or written below lies inside these sizes.
    for plane in 0..num_planes {
        let (pw, ph) = dims(plane);
        let stride = input.strides[plane];
        let need = (ph - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(pw))
            .ok_or(Error::Invalid("film grain plane size"))?;
        if stride < pw || input.planes[plane].len() < need || out[plane].len() < need {
            return Err(Error::Invalid("film grain plane size"));
        }
    }

    let grain = generate_grain(p, input, grain_min, grain_max);
    let luts = [scaling_lut(p, 0), scaling_lut(p, 1), scaling_lut(p, 2)];
    let tables = luts.each_ref().map(|lut| scale_table(lut, bd));
    let (blends, applies) = blend_params(p, input);

    let sw = (w + 1).div_ceil(2).div_ceil(16) * 32 + 34;
    let bands = h.div_ceil(32);
    // Band `b` of a plane is its rows from `b << (5 - sy)`: one chunk of the output per band.
    let [o0, o1, o2] = out;
    let rows = |plane: usize| (input.strides[plane] << (5 - subs[plane].1)).max(1);
    let mut chunks = [
        o0.chunks_mut(rows(0)),
        o1.chunks_mut(rows(1)),
        o2.chunks_mut(rows(2)),
    ];
    // The band list grows with the frame height, so it is booked while it lives.
    let _outs_held = ledger.charge(bands.saturating_mul(size_of::<Mutex<[&mut [u16]; 3]>>()))?;
    let outs: Vec<Mutex<[&mut [u16]; 3]>> = (0..bands)
        .map(|_| Mutex::new(chunks.each_mut().map(|c| c.next().unwrap_or_default())))
        .collect();

    run_each(bands, max_threads, |b| {
        check_cancel(cancel)?;
        let stripes = if p.overlap_flag && b > 0 { 2 } else { 1 };
        // The stripe buffers from the ledger's pool, charged while they live (empty for a plane
        // the frame does not have).
        let len = |plane: usize, n: usize| if plane < num_planes { n } else { 0 };
        let n = 34 * sw;
        let mut cur: [Buf<i32>; 3] = [
            ledger.zeros(len(0, n))?,
            ledger.zeros(len(1, n))?,
            ledger.zeros(len(2, n))?,
        ];
        let p_n = if stripes == 2 { n } else { 0 };
        let mut prev: [Buf<i32>; 3] = [
            ledger.zeros(len(0, p_n))?,
            ledger.zeros(len(1, p_n))?,
            ledger.zeros(len(2, p_n))?,
        ];
        let mut row = ledger.zeros::<i32>(sw)?;
        build_stripe(
            b, p, &grain, subs, num_planes, w, sw, grain_min, grain_max, &mut cur,
        );
        if stripes == 2 {
            build_stripe(
                b - 1,
                p,
                &grain,
                subs,
                num_planes,
                w,
                sw,
                grain_min,
                grain_max,
                &mut prev,
            );
        }
        let mut dst_band = outs[b].lock().unwrap_or_else(|e| e.into_inner());
        for plane in 0..num_planes {
            let (_, psy) = subs[plane];
            let (pw, ph) = dims(plane);
            let stride = input.strides[plane];
            let first = b << (5 - psy);
            let last = ph.min((b + 1) << (5 - psy));
            for y in first..last {
                let i = y - first;
                // Spec 7.18.3.5, second step: the top rows blend with the stripe above.
                let noise: &[i32] = if stripes == 2 && i < 2 >> psy {
                    let (wo, wn) = if psy == 1 {
                        (23, 22)
                    } else if i == 0 {
                        (27, 17)
                    } else {
                        (17, 27)
                    };
                    let old = &prev[plane][(i + (32 >> psy)) * sw..][..pw];
                    let new = &cur[plane][i * sw..][..pw];
                    for ((r, &o), &g) in row.iter_mut().zip(old).zip(new) {
                        *r = round2(o * wo + g * wn, 5).clamp(grain_min, grain_max);
                    }
                    &row[..pw]
                } else {
                    &cur[plane][i * sw..][..pw]
                };
                let src = &input.planes[plane][y * stride..][..pw];
                let dst = &mut dst_band[plane][i * stride..][..pw];
                if !applies[plane] {
                    dst.copy_from_slice(src);
                } else if plane == 0 {
                    blend_luma(simd, dst, src, noise, &luts[0], &tables[0], &blends[0]);
                } else {
                    let luma = &input.planes[0][(y << psy) * input.strides[0]..][..w];
                    blend_chroma(
                        simd,
                        dst,
                        src,
                        luma,
                        noise,
                        &luts[plane],
                        &tables[plane],
                        &blends[plane],
                    );
                }
            }
        }
        Ok(())
    })?;
    Ok(())
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
        let go = AtomicBool::new(false);
        apply_film_grain(
            &p,
            &input,
            [&mut oy, &mut ou, &mut ov],
            &ledger,
            &go,
            1,
            Simd::scalar(),
        )
        .unwrap();
        let tiny = Ledger::default();
        tiny.reset(crate::ledger::SMALL, 0).unwrap();
        let (mut ty, mut tu, mut tv) = (oy.clone(), ou.clone(), ov.clone());
        assert_eq!(
            apply_film_grain(
                &p,
                &input,
                [&mut ty, &mut tu, &mut tv],
                &tiny,
                &go,
                1,
                Simd::scalar()
            ),
            Err(Error::AllocLimit)
        );
        // A set cancel flag stops the grain before it writes anything.
        let stop = AtomicBool::new(true);
        assert_eq!(
            apply_film_grain(
                &p,
                &input,
                [&mut ty, &mut tu, &mut tv],
                &ledger,
                &stop,
                1,
                Simd::scalar()
            ),
            Err(Error::Cancelled)
        );
        assert_eq!(oy, y);
        assert_eq!(ou, u);
        assert_eq!(ov, v);
    }

    /// The film grain as it was before the bands: whole noise images built stripe after stripe,
    /// then the scalar blend over every row.
    fn reference(p: &FilmGrainParams, input: &GrainInput) -> [Vec<u16>; 3] {
        let bd = input.bit_depth;
        let grain_min = -(128i32 << (bd - 8));
        let grain_max = (128i32 << (bd - 8)) - 1;
        let num_planes = if input.mono_chrome { 1 } else { 3 };
        let (w, h) = (input.width, input.height);
        let cs = (
            usize::from(input.subsampling_x),
            usize::from(input.subsampling_y),
        );
        let subs = [(0, 0), cs, cs];
        let dims = |plane: usize| {
            let (sx, sy) = subs[plane];
            ((w + sx) >> sx, (h + sy) >> sy)
        };
        let grain = generate_grain(p, input, grain_min, grain_max);
        let sw = (w + 1).div_ceil(2).div_ceil(16) * 32 + 34;
        let mut noise: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for plane in 0..num_planes {
            let (pw, ph) = dims(plane);
            noise[plane] = vec![0; pw * ph];
        }
        let mut prev: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for luma_num in 0..h.div_ceil(32) {
            let mut cur: [Vec<i32>; 3] = [vec![0; 34 * sw], vec![0; 34 * sw], vec![0; 34 * sw]];
            build_stripe(
                luma_num, p, &grain, subs, num_planes, w, sw, grain_min, grain_max, &mut cur,
            );
            for plane in 0..num_planes {
                let (_, psy) = subs[plane];
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
        let luts = [scaling_lut(p, 0), scaling_lut(p, 1), scaling_lut(p, 2)];
        let (blends, applies) = blend_params(p, input);
        let mut out: [Vec<u16>; 3] = [
            input.planes[0].to_vec(),
            input.planes[1].to_vec(),
            input.planes[2].to_vec(),
        ];
        for plane in 0..num_planes {
            let (psy, (pw, ph)) = (subs[plane].1, dims(plane));
            let stride = input.strides[plane];
            for y in 0..ph {
                let src = &input.planes[plane][y * stride..][..pw];
                let dst = &mut out[plane][y * stride..][..pw];
                let n = &noise[plane][y * pw..][..pw];
                if !applies[plane] {
                    dst.copy_from_slice(src);
                } else if plane == 0 {
                    blend_luma_row(dst, src, n, &luts[0], &blends[0]);
                } else {
                    let luma = &input.planes[0][(y << psy) * input.strides[0]..][..w];
                    blend_chroma_row(dst, src, luma, n, &luts[plane], &blends[plane]);
                }
            }
        }
        out
    }

    #[test]
    fn every_tier_and_thread_count_matches_the_reference() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(n)) as u32
        };
        // (bit depth, subsampling x, y, mono, width, height): odd sizes, 1 to 3 bands.
        let cases = [
            (8, true, true, false, 37usize, 67usize),
            (10, true, true, false, 64, 70),
            (12, true, false, false, 33, 65),
            (8, false, false, false, 40, 33),
            (10, false, false, true, 35, 66),
            (12, true, true, false, 1, 1),
            (8, true, true, false, 2, 64),
        ];
        for (k, &(bd, sx, sy, mono, w, h)) in cases.iter().enumerate() {
            let mut p = FilmGrainParams {
                apply_grain: true,
                grain_seed: rnd(65536) as u16,
                num_y_points: (k % 3) as u8 + 1,
                chroma_scaling_from_luma: k % 4 == 1,
                num_cb_points: (k % 2) as u8 + 1,
                num_cr_points: (k % 3) as u8,
                grain_scaling_minus_8: rnd(4) as u8,
                ar_coeff_lag: rnd(4) as u8,
                ar_coeff_shift_minus_6: rnd(4) as u8,
                grain_scale_shift: rnd(2) as u8,
                cb_mult: rnd(256) as u8,
                cb_luma_mult: rnd(256) as u8,
                cb_offset: rnd(512) as u16,
                cr_mult: rnd(256) as u8,
                cr_luma_mult: rnd(256) as u8,
                cr_offset: rnd(512) as u16,
                overlap_flag: k % 3 != 2,
                clip_to_restricted_range: k % 2 == 1,
                ..Default::default()
            };
            for i in 0..16 {
                // Increasing point values; the scalings are anything.
                p.point_y_value[i] = (i * 16 + rnd(16) as usize) as u8;
                p.point_cb_value[i] = (i * 16 + rnd(16) as usize) as u8;
                p.point_cr_value[i] = (i * 16 + rnd(16) as usize) as u8;
                p.point_y_scaling[i] = rnd(256) as u8;
                p.point_cb_scaling[i] = rnd(256) as u8;
                p.point_cr_scaling[i] = rnd(256) as u8;
            }
            for c in p.ar_coeffs_y_plus_128.iter_mut() {
                *c = 96 + rnd(64) as u8;
            }
            for c in p
                .ar_coeffs_cb_plus_128
                .iter_mut()
                .chain(p.ar_coeffs_cr_plus_128.iter_mut())
            {
                *c = 96 + rnd(64) as u8;
            }
            let (cw, ch) = if mono {
                (0, 0)
            } else {
                (
                    (w + usize::from(sx)) >> usize::from(sx),
                    (h + usize::from(sy)) >> usize::from(sy),
                )
            };
            // Strides wider than the planes; samples anywhere in range, extremes included.
            let (ys, cs) = (w + 3, cw + 5);
            let max = (1u32 << bd) - 1;
            let mut plane = |stride: usize, rows: usize| -> Vec<u16> {
                (0..stride * rows)
                    .map(|_| match rnd(8) {
                        0 => 0,
                        1 => max as u16,
                        _ => rnd(max + 1) as u16,
                    })
                    .collect()
            };
            let (y, u, v) = (plane(ys, h), plane(cs, ch), plane(cs, ch));
            let input = GrainInput {
                bit_depth: bd,
                mono_chrome: mono,
                subsampling_x: sx,
                subsampling_y: sy,
                matrix_coefficients: if k % 4 == 3 { MC_IDENTITY } else { 1 },
                width: w,
                height: h,
                planes: [&y, &u, &v],
                strides: [ys, cs, cs],
            };
            let want = reference(&p, &input);
            let ledger = Ledger::default();
            ledger.reset(1 << 24, 0).unwrap();
            let go = AtomicBool::new(false);
            for simd in crate::simd::available() {
                for threads in [1, 2, 3] {
                    let mut got = [y.clone(), u.clone(), v.clone()];
                    let [g0, g1, g2] = &mut got;
                    apply_film_grain(&p, &input, [g0, g1, g2], &ledger, &go, threads, simd)
                        .unwrap();
                    assert_eq!(got, want, "case {k} {simd:?} threads {threads}");
                }
            }
        }
    }

    /// At every `max_alloc` from too small for one band to room for several, 2 and 4 threads
    /// give what 1 thread gives (the samples, or the same error), every time: a refusal that
    /// depends on how many bands ran at once is retried on one thread.
    #[test]
    fn every_max_alloc_gives_the_one_thread_result() {
        let (w, h) = (64usize, 200usize);
        let (cw, ch) = (w / 2, h / 2);
        let y: Vec<u16> = (0..w * h).map(|i| (i * 7 % 256) as u16).collect();
        let u: Vec<u16> = (0..cw * ch).map(|i| (i * 5 % 256) as u16).collect();
        let v: Vec<u16> = (0..cw * ch).map(|i| (i * 3 % 256) as u16).collect();
        let input = GrainInput {
            bit_depth: 8,
            mono_chrome: false,
            subsampling_x: true,
            subsampling_y: true,
            matrix_coefficients: 1,
            width: w,
            height: h,
            planes: [&y, &u, &v],
            strides: [w, cw, cw],
        };
        let mut p = FilmGrainParams {
            apply_grain: true,
            grain_seed: 4321,
            num_y_points: 1,
            num_cb_points: 1,
            num_cr_points: 1,
            overlap_flag: true,
            ..Default::default()
        };
        p.point_y_value[0] = 128;
        p.point_y_scaling[0] = 64;
        p.point_cb_value[0] = 128;
        p.point_cb_scaling[0] = 64;
        p.point_cr_value[0] = 128;
        p.point_cr_scaling[0] = 64;
        let go = AtomicBool::new(false);
        // `seed` elements are pooled first (when they fit), as a buffer a filter before grain
        // left behind.
        let run = |max_alloc: usize, threads: usize, seed: usize| {
            let ledger = Ledger::default();
            ledger.reset(max_alloc, 0).unwrap();
            if seed > 0 {
                drop(ledger.zeros::<i32>(seed));
            }
            let mut got = [vec![0; y.len()], vec![0; u.len()], vec![0; v.len()]];
            let [g0, g1, g2] = &mut got;
            let r = apply_film_grain(
                &p,
                &input,
                [g0, g1, g2],
                &ledger,
                &go,
                threads,
                Simd::scalar(),
            );
            r.map(|()| got)
        };
        let (mut fitted, mut refused) = (0, 0);
        for max_alloc in (crate::ledger::SMALL..400 << 10).step_by(4 << 10) {
            let want = run(max_alloc, 1, 0);
            fitted += usize::from(want.is_ok());
            refused += usize::from(want.is_err());
            for _ in 0..4 {
                for threads in [2, 4] {
                    assert!(
                        run(max_alloc, threads, 0) == want,
                        "max_alloc {max_alloc}: {threads} threads differ from 1"
                    );
                }
            }
        }
        // Both sides of the limit were checked.
        assert!(
            fitted > 0 && refused > 0,
            "{fitted} fitted, {refused} refused"
        );
        // A pooled buffer larger than a stripe (a 70 x 70 loop restoration window; a stripe is
        // 34 * 130 elements here) changes nothing either, in 1 KiB steps so a gap of the 1920
        // bytes between the two sizes is not stepped over.
        let (mut fitted, mut refused) = (0, 0);
        for max_alloc in (crate::ledger::SMALL..400 << 10).step_by(1 << 10) {
            let want = run(max_alloc, 1, 70 * 70);
            assert!(want == run(max_alloc, 1, 0), "max_alloc {max_alloc}: pool");
            fitted += usize::from(want.is_ok());
            refused += usize::from(want.is_err());
            for threads in [2, 4] {
                assert!(
                    run(max_alloc, threads, 70 * 70) == want,
                    "max_alloc {max_alloc}: {threads} threads differ from 1, pool seeded"
                );
            }
        }
        assert!(
            fitted > 0 && refused > 0,
            "{fitted} fitted, {refused} refused"
        );
    }
}
