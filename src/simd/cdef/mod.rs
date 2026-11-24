//! CDEF kernels (spec 7.15): the direction search of one 8x8 luma block and the filter of one
//! 8x8 luma / 4x4, 4x8, 8x4 chroma block, for every bit depth. The scalar oracle is
//! `crate::cdef::{cdef_direction, cdef_filter}`.
//!
//! The filter follows the design of dav1d's `src/cdef_tmpl.c` and `src/x86/cdef*.asm`
//! (BSD-2-Clause, Copyright © 2018, VideoLAN and dav1d authors, Copyright © 2018, Two Orioles,
//! LLC; see NOTICE): the block and a 2-sample border are copied into a padded 16-bit buffer in
//! which every unavailable tap (outside `MiRows` x `MiCols` or the plane) holds the sentinel
//! `0x8000`. A tap then needs no availability check: its constrained difference is 0 (the
//! saturated difference is so large that `constrain` cuts it to 0), the signed max ignores it
//! (-32768) and the unsigned min ignores it (32768).
//!
//! Exactness: samples below 4096 (12 bits) keep every intermediate in 16 bits. The dispatch
//! checks that, and every other premise of the kernels (block inside the frame, strengths,
//! shifts, buffer bounds); a block that fails a check takes the scalar oracle.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use super::{Simd, Tier};
use crate::bits::floor_log2;
use crate::cdef::{CDEF_DIRECTIONS, CDEF_PRI_TAPS, FilterCtx, cdef_direction, cdef_filter};
use crate::consts::MI_SIZE;
use crate::frame::{Plane, View};

/// Row stride of the padded block.
const TS: usize = 16;
/// The padded block: up to 8 + 4 rows of `TS` samples, the block at row 2, column 2.
const TMP: usize = TS * 12;
/// Index of the block's top-left sample in the padded block.
const CENTER: usize = 2 * TS + 2;
/// An unavailable tap.
const SENTINEL: u16 = 0x8000;

static Z8: [u16; 8] = [0; 8];
static Z4: [u16; 4] = [0; 4];

/// `t[s..s + 8]` as an array (the kernels' offsets are always in bounds; zeros otherwise).
#[inline]
fn at8(t: &[u16], s: usize) -> &[u16; 8] {
    t.get(s..s + 8)
        .and_then(|a| a.try_into().ok())
        .unwrap_or(&Z8)
}

/// `t[s..s + 4]` as an array, like [`at8`].
#[inline]
fn at4(t: &[u16], s: usize) -> &[u16; 4] {
    t.get(s..s + 4)
        .and_then(|a| a.try_into().ok())
        .unwrap_or(&Z4)
}

/// The parameters of one filter call, in padded-block terms.
struct Filt {
    /// Block width and height: 8 or 4.
    w: usize,
    h: usize,
    /// Start index in the padded block of each primary tap's block-sized window:
    /// `[k0 +, k0 -, k1 +, k1 -]`, weights `pri_tap[k]`.
    pri: [usize; 4],
    /// The secondary taps (directions `dir + 2` and `dir - 2`): four for k = 0 (weight 2), then
    /// four for k = 1 (weight 1).
    sec: [usize; 8],
    pri_tap: [i16; 2],
    pri_str: i16,
    sec_str: i16,
    /// `damping_adj` of `constrain` for each strength (0..=15 here).
    pri_shift: i32,
    sec_shift: i32,
}

/// CDEF direction process (spec 7.15.2) of the 8x8 luma block at 4x4 position (`r`, `c`):
/// (`yDir`, `var`), bit-exact with [`cdef_direction`].
#[allow(unsafe_code)]
pub(crate) fn direction(
    simd: Simd,
    luma: &Plane,
    r: usize,
    c: usize,
    bit_depth: u32,
) -> (usize, i64) {
    if simd.tier() == Tier::Scalar {
        return cdef_direction(luma, r, c, bit_depth);
    }
    let (x0, y0) = (c * MI_SIZE, r * MI_SIZE);
    let shift = bit_depth.saturating_sub(8);
    let mut rows = [[0u16; 8]; 8];
    let mut or = 0u16;
    for (i, row) in rows.iter_mut().enumerate() {
        let a = (y0 + i) * luma.stride + x0;
        let Some(s) = luma.px.get(a..a + 8) else {
            return cdef_direction(luma, r, c, bit_depth);
        };
        row.copy_from_slice(s);
        or |= s.iter().fold(0, |o, &v| o | v);
    }
    // The kernels hold `(v >> shift) - 128` in 16 bits and the costs in 32 bits.
    if shift > 4 || (or >> shift) >= 256 {
        return cdef_direction(luma, r, c, bit_depth);
    }
    let cost: [i32; 8] = match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: `simd` is a detected tier (only `simd::available`/`best` make one), and
            // AVX-512 implies AVX2.
            unsafe { avx2::dir_costs(&rows, shift) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `simd` is a detected tier (only `simd::available`/`best` make one).
            unsafe { neon::dir_costs(&rows, shift) }
        }
        _ => return cdef_direction(luma, r, c, bit_depth),
    };
    let (mut best_cost, mut y_dir) = (0, 0);
    for (i, &v) in cost.iter().enumerate() {
        if v > best_cost {
            best_cost = v;
            y_dir = i;
        }
    }
    (
        y_dir,
        (i64::from(best_cost) - i64::from(cost[(y_dir + 4) & 7])) >> 10,
    )
}

/// CDEF filter process (spec 7.15.3) of one plane of the 8x8 luma block at (`r`, `c`): the
/// arguments and the result are those of [`cdef_filter`], bit for bit.
#[allow(unsafe_code)]
pub(crate) fn filter(
    simd: Simd,
    ctx: &FilterCtx,
    src: &Plane,
    dst: &mut View<'_>,
    plane: usize,
    sub: (usize, usize),
    r: usize,
    c: usize,
    strengths: (i32, i32),
    damping: i32,
    dir: usize,
) {
    let prep = if simd.tier() == Tier::Scalar {
        None
    } else {
        prepare(ctx, src, dst, plane, sub, r, c, strengths, damping, dir)
    };
    let Some((tmp, f, x0, y0)) = prep else {
        return cdef_filter(ctx, src, dst, plane, sub, r, c, strengths, damping, dir);
    };
    let mut out = [0u16; 64];
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: `simd` is a detected tier (only `simd::available`/`best` make one), and
            // AVX-512 implies AVX2.
            unsafe { avx2::filter(&tmp, &mut out, &f) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `simd` is a detected tier (only `simd::available`/`best` make one).
            unsafe { neon::filter(&tmp, &mut out, &f) }
        }
        _ => return cdef_filter(ctx, src, dst, plane, sub, r, c, strengths, damping, dir),
    }
    // `prepare` checked that every row fits in `dst`.
    for (i, row) in out.chunks_exact(f.w).take(f.h).enumerate() {
        let a = (y0 - dst.y0 + i) * dst.stride + x0;
        if let Some(d) = dst.px.get_mut(a..a + f.w) {
            d.copy_from_slice(row);
        }
    }
}

/// The padded block and the kernel parameters, with the block's sample position (`x0`, `y0`);
/// `None` when a premise of the kernels does not hold (the caller then runs the oracle).
fn prepare(
    ctx: &FilterCtx,
    src: &Plane,
    dst: &View<'_>,
    plane: usize,
    sub: (usize, usize),
    r: usize,
    c: usize,
    strengths: (i32, i32),
    damping: i32,
    dir: usize,
) -> Option<([u16; TMP], Filt, usize, usize)> {
    let (pri_str, sec_str) = strengths;
    let (sub_x, sub_y) = if plane > 0 { sub } else { (0, 0) };
    if sub_x > 1 || sub_y > 1 || ctx.coeff_shift > 4 {
        return None;
    }
    let (x0, y0) = ((c * MI_SIZE) >> sub_x, (r * MI_SIZE) >> sub_y);
    let (w, h) = (8usize >> sub_x, 8usize >> sub_y);
    let rows = (ctx.mi_rows * MI_SIZE) >> sub_y;
    let cols = (ctx.mi_cols * MI_SIZE) >> sub_x;
    // The oracle reads 0 for a block sample outside the frame; the kernels need it inside.
    if x0 + w > cols || y0 + h > rows {
        return None;
    }
    // Every output row must fit in `dst`, which starts at plane row `dst.y0` (the oracle drops
    // a sample that does not).
    let dy = y0.checked_sub(dst.y0)?;
    if (dy + h - 1) * dst.stride + x0 + w > dst.px.len() {
        return None;
    }
    // `constrain` of `cdef_filter`; a sentinel tap must give 0: `32768 >> shift >= threshold`.
    // Strengths up to 255 keep the tap sum in 16 bits.
    let shift = |t: i32| {
        if t == 0 {
            0
        } else {
            (damping - floor_log2(t as u32) as i32).clamp(0, 31)
        }
    };
    let (pri_shift, sec_shift) = (shift(pri_str), shift(sec_str));
    for (t, s) in [(pri_str, pri_shift), (sec_str, sec_shift)] {
        if !(0..=255).contains(&t) || (32768 >> s) < t {
            return None;
        }
    }
    let tap_set = ((pri_str >> ctx.coeff_shift) & 1) as usize;
    // The padded block.
    let mut tmp = [SENTINEL; TMP];
    let mut or = 0u16;
    let xa = x0.saturating_sub(2);
    let xb = (x0 + w + 2).min(cols);
    for yy in 0..h + 4 {
        let Some(y) = (y0 + yy).checked_sub(2) else {
            continue;
        };
        if y >= rows {
            continue;
        }
        let a = y * src.stride;
        let row = src.px.get(a + xa..a + xb)?;
        let d = yy * TS + xa + 2 - x0;
        tmp.get_mut(d..d + row.len())?.copy_from_slice(row);
        or |= row.iter().fold(0, |o, &v| o | v);
    }
    if or >= 4096 {
        return None;
    }
    // A tap at (dy, dx) from a sample reads the window that starts at `CENTER + dy * TS + dx`.
    if dir >= 8 {
        return None;
    }
    let at = |d: usize, k: usize, sign: isize| {
        let [dy, dx] = CDEF_DIRECTIONS[d][k];
        CENTER.wrapping_add_signed(sign * (dy as isize * TS as isize + dx as isize))
    };
    let (s0, s1) = ((dir + 2) & 7, (dir + 6) & 7);
    let f = Filt {
        w,
        h,
        pri: [at(dir, 0, 1), at(dir, 0, -1), at(dir, 1, 1), at(dir, 1, -1)],
        sec: [
            at(s0, 0, 1),
            at(s0, 0, -1),
            at(s1, 0, 1),
            at(s1, 0, -1),
            at(s0, 1, 1),
            at(s0, 1, -1),
            at(s1, 1, 1),
            at(s1, 1, -1),
        ],
        pri_tap: [
            CDEF_PRI_TAPS[tap_set][0] as i16,
            CDEF_PRI_TAPS[tap_set][1] as i16,
        ],
        pri_str: pri_str as i16,
        sec_str: sec_str as i16,
        pri_shift,
        sec_shift,
    };
    Some((tmp, f, x0, y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Ledger;
    use crate::simd::available;
    use std::sync::Arc;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn plane(w: usize, h: usize, stride: usize) -> Plane {
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 22, 0).unwrap();
        let mut p = Plane::new(stride, h, &ledger).unwrap();
        p.width = w;
        p
    }

    /// Fills `p` by `kind`: random, all 0, all max, a checkerboard of 0 and max, a random edge.
    fn fill(p: &mut Plane, rng: &mut Rng, bd: u32, kind: u64) {
        let max = (1u64 << bd) - 1;
        let (t, sh) = (rng.below(max + 1), rng.below(4));
        for (i, v) in p.px.iter_mut().enumerate() {
            let (y, x) = (i / p.stride, i % p.stride);
            *v = (match kind {
                0 => rng.below(max + 1),
                1 => 0,
                2 => max,
                3 => max * ((x + y) as u64 & 1),
                4 => (t + rng.below(1 << sh) * u64::from(x > y)).min(max),
                _ => (t + rng.below(64)).min(max),
            }) as u16;
        }
    }

    #[test]
    fn direction_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..600 {
            let bd = [8, 10, 12][case % 3];
            let (mi_rows, mi_cols) = (2 + 2 * rng.below(3) as usize, 2 + rng.below(5) as usize);
            let (w, h) = (mi_cols * 4, mi_rows * 4);
            let mut p = plane(w, h, w + rng.below(3) as usize);
            fill(&mut p, &mut rng, bd, (case / 3) as u64 % 6);
            for r in (0..mi_rows).step_by(2) {
                for c in (0..mi_cols).step_by(2) {
                    let want = cdef_direction(&p, r, c, bd);
                    for s in available() {
                        assert_eq!(
                            direction(s, &p, r, c, bd),
                            want,
                            "{s:?} bd {bd} case {case}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn filter_matches_scalar() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for case in 0..400 {
            let bd = [8u32, 10, 12][case % 3];
            let cs = bd - 8;
            let sub = [(0, 0), (1, 1), (1, 0), (0, 1)][(case / 3) % 4];
            // Odd mi_cols: the last block column is half outside the frame (oracle path).
            let (mi_rows, mi_cols) = (2 + 2 * rng.below(3) as usize, 2 + rng.below(5) as usize);
            let ctx = FilterCtx {
                mi_rows,
                mi_cols,
                coeff_shift: cs,
            };
            for plane_i in 0..3 {
                let (sx, sy) = if plane_i == 0 { (0, 0) } else { sub };
                let (w, h) = ((mi_cols * 4) >> sx, (mi_rows * 4) >> sy);
                let mut s = plane(w, h, w + rng.below(5) as usize);
                fill(&mut s, &mut rng, bd, (case / 12) as u64 % 6);
                for r in (0..mi_rows).step_by(2) {
                    for c in (0..mi_cols).step_by(2) {
                        let pri = match rng.below(4) {
                            0 => 0,
                            // The largest luma strength after the variance adjustment.
                            1 => ((15 << cs) * (4 + 12) + 8) >> 4,
                            _ => (rng.below(16) as i32) << cs,
                        };
                        let sec = [0, 1, 2, 4][rng.below(4) as usize] << cs;
                        let damping = 3 + rng.below(4) as i32 + cs as i32 - i32::from(plane_i > 0);
                        let dir = rng.below(8) as usize;
                        let st = (pri, sec);
                        let mut want = plane(w, h, s.stride);
                        want.px.copy_from_slice(&s.px);
                        cdef_filter(
                            &ctx,
                            &s,
                            &mut View::of(&mut want),
                            plane_i,
                            sub,
                            r,
                            c,
                            st,
                            damping,
                            dir,
                        );
                        for t in available() {
                            let mut got = plane(w, h, s.stride);
                            got.px.copy_from_slice(&s.px);
                            filter(
                                t,
                                &ctx,
                                &s,
                                &mut View::of(&mut got),
                                plane_i,
                                sub,
                                r,
                                c,
                                st,
                                damping,
                                dir,
                            );
                            assert_eq!(
                                got.px, want.px,
                                "{t:?} bd {bd} sub {sub:?} plane {plane_i} r {r} c {c} \
                                 pri {pri} sec {sec} damping {damping} dir {dir}"
                            );
                        }
                    }
                }
            }
        }
    }
}
