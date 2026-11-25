//! Loop restoration process (spec 7.17): Wiener and self guided filtering per restoration unit
//! and 64-row stripe.
//!
//! The spec runs the filters on 4x4 blocks; its note in 7.17.1 allows larger blocks inside one
//! restoration unit and one stripe, so each (stripe, unit column) region is cut into blocks of
//! at most 64 samples wide and one stripe high. Every block first gathers its source samples
//! (the get source sample process, 7.17.6) into a window with a 3 sample border.

use crate::consts::{
    FILTER_BITS, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_WIENER, SGRPROJ_MTABLE_BITS,
    SGRPROJ_PRJ_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS, SGRPROJ_SGR_BITS,
};
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::frame::{FrameState, Plane, View};
use crate::header::FrameHeader;
use crate::ledger::{Buf, Ledger};
use crate::seq::SequenceHeader;
use crate::simd::lr as simd_lr;
use std::sync::atomic::AtomicBool;

/// `Sgr_Params` (spec 7.17.3): `{ r0, e0, r1, e1 }` per `lr_sgr_set`.
pub(crate) const SGR_PARAMS: [[u8; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

/// Widest block filtered at once.
pub(crate) const BLOCK_W: usize = 64;

/// Tallest block: a stripe is 64 luma rows (`64 >> subsampling_y` chroma rows).
const STRIPE_H: usize = 64;

/// Samples of the largest source window (a block with a 3 sample border).
const WIN: usize = (BLOCK_W + 6) * (STRIPE_H + 6);

/// The buffers of loop restoration, made once per [`lr_frame`] call or per band item: the source
/// window and the kernels' working arrays. They are charged to the ledger while they live, come
/// from its pool (so a repeated call allocates nothing) and are not cleared between blocks
/// (every entry read is written first).
pub(crate) struct LrScratch<'a> {
    win: Buf<'a, i32>,
    mem: simd_lr::Mem<'a>,
}

impl<'a> LrScratch<'a> {
    pub(crate) fn new(ledger: &'a Ledger) -> Result<LrScratch<'a>, Error> {
        Ok(LrScratch {
            win: ledger.zeros(WIN)?,
            mem: simd_lr::Mem::new(ledger)?,
        })
    }
}

/// `Round2( x, n )` for signed `x` (arithmetic shift), `n` may be 0.
fn round2(x: i64, n: usize) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// What the get source sample process (7.17.6) needs besides the planes.
struct Stripe {
    /// `StripeStartY`.
    start: isize,
    /// `StripeEndY`.
    end: isize,
    /// `PlaneEndX`.
    end_x: isize,
    /// `PlaneEndY`.
    end_y: isize,
}

/// Get source sample process (spec 7.17.6) for the `(w + 6) x (h + 6)` samples around the block
/// at (`x`, `y`) into `win` (`w <= BLOCK_W`, `h <= STRIPE_H`): sample `(r, c)` is
/// `win[ r * (w + 6) + c ]`, `get_source_sample( plane, x + c - 3, y + r - 3 )`.
fn source_window(
    cdef: &Plane,
    curr: &Plane,
    st: &Stripe,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    win: &mut [i32],
) {
    let ww = w + 6;
    // Window column `c` is plane column `first + c`, clamped to `0..=PlaneEndX`: columns
    // `lo..hi` of the window need no clamp and are copied straight.
    let first = x as isize - 3;
    let lo = (-first).clamp(0, ww as isize) as usize;
    let hi = (st.end_x + 1 - first).clamp(lo as isize, ww as isize) as usize;
    for r in 0..h + 6 {
        let mut yy = (y as isize + r as isize - 3).clamp(0, st.end_y);
        let src = if yy < st.start {
            yy = yy.max(st.start - 2);
            curr
        } else if yy > st.end {
            yy = yy.min(st.end + 2);
            curr
        } else {
            cdef
        };
        let row = &src.px[yy as usize * src.stride..];
        let out = &mut win[r * ww..(r + 1) * ww];
        let x_lo = (first + lo as isize) as usize;
        for (o, &v) in out[lo..hi].iter_mut().zip(&row[x_lo..x_lo + (hi - lo)]) {
            *o = i32::from(v);
        }
        for c in (0..lo).chain(hi..ww) {
            let xx = (first + c as isize).clamp(0, st.end_x) as usize;
            out[c] = i32::from(row[xx]);
        }
    }
}

/// Wiener coefficient process (spec 7.17.5).
pub(crate) fn wiener_coefficients(coeff: &[i8; 3]) -> [i64; 7] {
    let mut filter = [0i64; 7];
    filter[3] = 128;
    for i in 0..3 {
        let c = i64::from(coeff[i]);
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

/// Wiener filter process (spec 7.17.4) on the `w` x `h` block at (`x`, `y`) of `dst`; `win` is the
/// source window of that block, `lr` is `LrWiener[ plane ][ unitRow ][ unitCol ]`. `y` counts
/// from `dst.y0`.
pub(crate) fn wiener_filter(
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    lr: &[[i8; 3]; 2],
    dst: &mut View<'_>,
    x: usize,
    y: usize,
) {
    let vfilter = wiener_coefficients(&lr[0]);
    let hfilter = wiener_coefficients(&lr[1]);
    // Rounding variables derivation process (7.11.3.2) with isCompound equal to 0.
    let (round0, round1) = if bd == 12 { (5, 9) } else { (3, 11) };
    let offset = 1i64 << (bd + FILTER_BITS - round0 - 1);
    let limit = (1i64 << (bd + 1 + FILTER_BITS - round0)) - 1;
    let max_val = (1i64 << bd) - 1;
    let ww = w + 6;
    let mut intermediate = [0i64; (STRIPE_H + 6) * BLOCK_W];
    for r in 0..h + 6 {
        for c in 0..w {
            let mut s = 0i64;
            for t in 0..7 {
                s += hfilter[t] * i64::from(win[r * ww + c + t]);
            }
            let v = round2(s, round0);
            intermediate[r * w + c] = v.clamp(-offset, limit - offset);
        }
    }
    for r in 0..h {
        for c in 0..w {
            let mut s = 0i64;
            for t in 0..7 {
                s += vfilter[t] * intermediate[(r + t) * w + c];
            }
            let v = round2(s, round1);
            dst.px[(y + r) * dst.stride + x + c] = v.clamp(0, max_val) as u16;
        }
    }
}

/// Box filter process (spec 7.17.3) for `r` (the nonzero `Sgr_Params[ set ][ pass * 2 ]`) and
/// `eps`; writes `F` as `h * w` values to `f`. `pass0` is `pass == 0`.
fn box_filter(
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    r: usize,
    eps: i64,
    pass0: bool,
    f: &mut [i32; STRIPE_H * BLOCK_W],
) {
    let ww = w + 6;
    let aw = w + 2;
    let n = ((2 * r + 1) * (2 * r + 1)) as i64;
    let n2e = n * n * eps;
    let s = ((1i64 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let one_over_n = ((1i64 << SGRPROJ_RECIP_BITS) + n / 2) / n;
    // A[ i ][ j ] and B[ i ][ j ] for i in -1..h+1, j in -1..w+1 are stored at (i + 1, j + 1).
    let mut a_arr = [0i32; (STRIPE_H + 2) * (BLOCK_W + 2)];
    let mut b_arr = [0i32; (STRIPE_H + 2) * (BLOCK_W + 2)];
    for i in 0..h + 2 {
        // Spec i is i - 1: with pass 0 only odd rows (even i here) are used.
        if pass0 && (i & 1) == 1 {
            continue;
        }
        for j in 0..w + 2 {
            let mut a = 0i64;
            let mut b = 0i64;
            for dy in 0..=2 * r {
                for dx in 0..=2 * r {
                    // Window row of spec sample (i - 1 + dy - r) is i + 2 - r + dy.
                    let c = i64::from(win[(i + 2 - r + dy) * ww + (j + 2 - r + dx)]);
                    a += c * c;
                    b += c;
                }
            }
            a = round2(a, 2 * (bd - 8));
            let d = round2(b, bd - 8);
            let p = (a * n - d * d).max(0);
            let z = round2(p * s, SGRPROJ_MTABLE_BITS);
            let a2 = if z >= 255 {
                256
            } else if z == 0 {
                1
            } else {
                ((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1)
            };
            let b2 = ((1 << SGRPROJ_SGR_BITS) - a2) * b * one_over_n;
            a_arr[i * aw + j] = a2 as i32;
            b_arr[i * aw + j] = round2(b2, SGRPROJ_RECIP_BITS) as i32;
        }
    }
    for i in 0..h {
        let shift = if pass0 && (i & 1) == 1 { 4 } else { 5 };
        for j in 0..w {
            let mut a = 0i64;
            let mut b = 0i64;
            for dy in 0..3 {
                for dx in 0..3 {
                    // dy, dx are the spec's dy + 1, dx + 1.
                    let weight = if pass0 {
                        if (i + dy) & 1 == 0 {
                            if dx == 1 { 6 } else { 5 }
                        } else {
                            0
                        }
                    } else if dx == 1 || dy == 1 {
                        4
                    } else {
                        3
                    };
                    a += weight * i64::from(a_arr[(i + dy) * aw + j + dx]);
                    b += weight * i64::from(b_arr[(i + dy) * aw + j + dx]);
                }
            }
            let v = a * i64::from(win[(i + 3) * ww + j + 3]) + b;
            f[i * w + j] = round2(v, SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS) as i32;
        }
    }
}

/// Self guided filter process (spec 7.17.2) on the `w` x `h` block at (`x`, `y`) of `dst` (`y`
/// counts from `dst.y0`).
pub(crate) fn self_guided_filter(
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    set: usize,
    xqd: &[i8; 2],
    dst: &mut View<'_>,
    x: usize,
    y: usize,
) {
    let Some(params) = SGR_PARAMS.get(set) else {
        return;
    };
    let (r0, r1) = (usize::from(params[0]), usize::from(params[2]));
    let mut flt0 = [0i32; STRIPE_H * BLOCK_W];
    let mut flt1 = [0i32; STRIPE_H * BLOCK_W];
    if r0 != 0 {
        box_filter(win, w, h, bd, r0, i64::from(params[1]), true, &mut flt0);
    }
    if r1 != 0 {
        box_filter(win, w, h, bd, r1, i64::from(params[3]), false, &mut flt1);
    }
    let flt0 = (r0 != 0).then_some(&flt0);
    let flt1 = (r1 != 0).then_some(&flt1);
    let w0 = i64::from(xqd[0]);
    let w1 = i64::from(xqd[1]);
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let max_val = (1i64 << bd) - 1;
    let ww = w + 6;
    for i in 0..h {
        for j in 0..w {
            let u = i64::from(win[(i + 3) * ww + j + 3]) << SGRPROJ_RST_BITS;
            let mut v = w1 * u;
            v += w0 * flt0.as_ref().map_or(u, |f| i64::from(f[i * w + j]));
            v += w2 * flt1.as_ref().map_or(u, |f| i64::from(f[i * w + j]));
            let s = round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
            dst.px[(y + i) * dst.stride + x + j] = s.clamp(0, max_val) as u16;
        }
    }
}

/// Loop restoration process (spec 7.17), with the loop restore block process (7.17.1) applied per
/// stripe and unit column. `out` must already hold a copy of `upscaled_cdef` (`LrFrame` starts
/// as `UpscaledCdefFrame`); restored units are written over it. Planes with
/// `FrameRestorationType` equal to `RESTORE_NONE` are left alone. Plane dimensions below the
/// frame size shrink the filtered area instead of panicking.
pub(crate) fn lr_frame(
    fs: &FrameState,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    upscaled_cdef: &[Plane],
    upscaled_curr: &[Plane],
    out: &mut [Plane],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let planes = fs
        .num_planes
        .min(3)
        .min(upscaled_cdef.len())
        .min(upscaled_curr.len())
        .min(out.len());
    let mut scratch = LrScratch::new(&fs.ledger)?;
    for plane in 0..planes {
        let (cdef, curr) = (&upscaled_cdef[plane], &upscaled_curr[plane]);
        let dst = &mut out[plane];
        let Some(g) = lr_plane(fs, seq, fh, plane, [cdef, curr], (dst.width, dst.height)) else {
            continue;
        };
        let mut dst = View::of(dst);
        let mut stripe_num = 0usize;
        loop {
            check_cancel(cancel)?;
            if g.stripe(stripe_num).is_none() {
                break;
            }
            lr_stripe(fs, seq, &g, stripe_num, cdef, curr, &mut dst, &mut scratch);
            stripe_num += 1;
        }
    }
    Ok(())
}

/// Where loop restoration works in one plane.
pub(crate) struct LrPlane {
    plane: usize,
    sub_y: usize,
    unit_size: usize,
    unit_rows: usize,
    unit_cols: usize,
    /// `PlaneEndX`, `PlaneEndY`.
    end_x: usize,
    end_y: usize,
}

/// The loop restoration of plane `plane` (0 to 2) with inputs `[ UpscaledCdefFrame,
/// UpscaledCurrFrame ]` and an output plane of `dst_size` (width, height) samples; `None` when
/// the plane is not restored or has no area.
pub(crate) fn lr_plane(
    fs: &FrameState,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    plane: usize,
    [cdef, curr]: [&Plane; 2],
    (dst_w, dst_h): (usize, usize),
) -> Option<LrPlane> {
    if fh.frame_restoration_type[plane] == RESTORE_NONE {
        return None;
    }
    let (sub_x, sub_y) = if plane > 0 {
        (
            usize::from(seq.subsampling_x),
            usize::from(seq.subsampling_y),
        )
    } else {
        (0, 0)
    };
    let unit_size = usize::from(fh.loop_restoration_size[plane]);
    let (unit_rows, unit_cols) = (fs.lr_unit_rows[plane], fs.lr_unit_cols[plane]);
    if unit_size == 0 || unit_rows == 0 || unit_cols == 0 {
        return None;
    }
    let round2u = |v: usize, n: usize| (v + ((1 << n) >> 1)) >> n;
    let plane_w = round2u(fs.upscaled_width, sub_x)
        .min(cdef.width)
        .min(curr.width)
        .min(dst_w);
    let plane_h = round2u(fs.frame_height, sub_y)
        .min(cdef.height)
        .min(curr.height)
        .min(dst_h);
    if plane_w == 0 || plane_h == 0 {
        return None;
    }
    Some(LrPlane {
        plane,
        sub_y,
        unit_size,
        unit_rows,
        unit_cols,
        end_x: plane_w - 1,
        end_y: plane_h - 1,
    })
}

impl LrPlane {
    /// Stripe `n` (64 luma rows, offset up by 8); `None` after the last one. The stripes of a
    /// plane cover its rows once each, in order.
    fn stripe(&self, n: usize) -> Option<Stripe> {
        let start = ((n * 64) as isize - 8) >> self.sub_y;
        if start > self.end_y as isize {
            return None;
        }
        Some(Stripe {
            start,
            end: start + (64 >> self.sub_y) as isize - 1,
            end_x: self.end_x as isize,
            end_y: self.end_y as isize,
        })
    }

    /// The rows `y0..=y1` of the plane that stripe `n` filters.
    pub(crate) fn stripe_rows(&self, n: usize) -> Option<(usize, usize)> {
        let st = self.stripe(n)?;
        Some((st.start.max(0) as usize, st.end.min(st.end_y) as usize))
    }
}

/// Loop restoration of stripe `n` of plane `g`, every unit column: reads `cdef` and `curr`
/// (that plane of `UpscaledCdefFrame` and `UpscaledCurrFrame`), writes the restored units
/// into `dst`, whole rows of the `LrFrame` plane (from row `dst.y0`, the stripe's rows among
/// them).
pub(crate) fn lr_stripe(
    fs: &FrameState,
    seq: &SequenceHeader,
    g: &LrPlane,
    n: usize,
    cdef: &Plane,
    curr: &Plane,
    dst: &mut View<'_>,
    scratch: &mut LrScratch<'_>,
) {
    let (Some(st), Some((y0, y1))) = (g.stripe(n), g.stripe_rows(n)) else {
        return;
    };
    // The stripe's first row in `dst`.
    let Some(dst_y) = y0.checked_sub(dst.y0) else {
        return;
    };
    let bd = usize::from(seq.bit_depth);
    let simd = crate::simd::best();
    let LrScratch { win, mem } = scratch;
    let plane = g.plane;
    let end_x = g.end_x;
    let luma_y = if n == 0 { 0 } else { n * 64 - 8 };
    let unit_row = (g.unit_rows - 1).min(((luma_y + 8) >> g.sub_y) / g.unit_size);
    for unit_col in 0..g.unit_cols {
        let x0 = unit_col * g.unit_size;
        if x0 > end_x {
            break;
        }
        let x1 = if unit_col + 1 == g.unit_cols {
            end_x
        } else {
            ((unit_col + 1) * g.unit_size - 1).min(end_x)
        };
        let idx = unit_row * g.unit_cols + unit_col;
        let r_type = fs.lr_type[plane].get(idx).copied().unwrap_or(RESTORE_NONE);
        if r_type != RESTORE_WIENER && r_type != RESTORE_SGRPROJ {
            continue;
        }
        let h = y1 - y0 + 1;
        for x in (x0..=x1).step_by(BLOCK_W) {
            let w = BLOCK_W.min(x1 - x + 1);
            source_window(cdef, curr, &st, x, y0, w, h, win);
            if r_type == RESTORE_WIENER {
                if let Some(lr) = fs.lr_wiener[plane].get(idx) {
                    simd_lr::wiener(simd, win, w, h, bd, lr, dst, x, dst_y, mem);
                }
            } else if let (Some(&set), Some(xqd)) =
                (fs.lr_sgr_set[plane].get(idx), fs.lr_sgr_xqd[plane].get(idx))
            {
                let set = usize::from(set);
                simd_lr::self_guided(simd, win, w, h, bd, set, xqd, dst, x, dst_y, mem);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Ledger;
    use std::sync::Arc;

    fn plane(w: usize, h: usize) -> Plane {
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 20, 0).unwrap();
        Plane::new(w, h, &ledger).unwrap()
    }

    #[test]
    fn wiener_with_centre_tap_only_is_identity() {
        let (w, h) = (5, 4);
        let win: Vec<i32> = (0..(w + 6) * (h + 6))
            .map(|i| (i as i32 * 37) % 256)
            .collect();
        let mut dst = plane(w, h);
        wiener_filter(&win, w, h, 8, &[[0; 3]; 2], &mut View::of(&mut dst), 0, 0);
        for r in 0..h {
            for c in 0..w {
                assert_eq!(i32::from(dst.px[r * w + c]), win[(r + 3) * (w + 6) + c + 3]);
            }
        }
    }

    #[test]
    fn wiener_coefficients_have_unit_gain() {
        let f = wiener_coefficients(&[3, -7, 20]);
        assert_eq!(f, [3, -7, 20, 128 - 2 * 16, 20, -7, 3]);
        assert_eq!(f.iter().sum::<i64>(), 128);
    }

    #[test]
    fn box_filter_on_constant_image() {
        let (w, h) = (6, 5);
        let win = vec![100i32; (w + 6) * (h + 6)];
        // r = 1, pass 1: variance 0 -> a2 = 1, B = Round2( 255 * 900 * 455, 12 ) = 25494,
        // F = Round2( 32 * 100 + 32 * 25494, 9 ) = 1600 = 100 << SGRPROJ_RST_BITS.
        let mut f = [0; STRIPE_H * BLOCK_W];
        box_filter(&win, w, h, 8, 1, 4, false, &mut f);
        assert!(f[..w * h].iter().all(|&v| v == 1600));
        // r = 2, pass 0: B = Round2( 255 * 2500 * 164, 12 ) = 25525, every row gives
        // Round2( 32 * 100 + 32 * 25525, 9 ) or Round2( 16 * 100 + 16 * 25525, 8 ) = 1602.
        box_filter(&win, w, h, 8, 2, 12, true, &mut f);
        assert!(f[..w * h].iter().all(|&v| v == 1602));
    }

    #[test]
    fn source_window_takes_rows_outside_the_stripe_from_curr() {
        let mut cdef = plane(8, 16);
        let mut curr = plane(8, 16);
        cdef.px.fill(1);
        curr.px.fill(2);
        // Stripe is rows 4..=9.
        let st = Stripe {
            start: 4,
            end: 9,
            end_x: 7,
            end_y: 15,
        };
        let mut win = [0; WIN];
        source_window(&cdef, &curr, &st, 0, 4, 2, 6, &mut win);
        let col = |r: usize| win[r * 8 + 3];
        // Window rows are source rows 1..=12: rows 1..=3 curr (row 1 clamped to start - 2 = 2),
        // rows 4..=9 cdef, rows 10..=12 curr.
        let expect = [2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2];
        for (r, &e) in expect.iter().enumerate() {
            assert_eq!(col(r), e, "row {r}");
        }
    }

    #[test]
    fn source_window_matches_the_clamped_reads() {
        let (pw, ph) = (80, 48);
        let mut cdef = plane(pw, ph);
        let mut curr = plane(pw, ph);
        for (i, (a, b)) in cdef.px.iter_mut().zip(curr.px.iter_mut()).enumerate() {
            *a = i as u16;
            *b = 0x8000 | i as u16;
        }
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: usize| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % n as u64) as usize
        };
        let mut win = [0; WIN];
        for _ in 0..500 {
            let end_x = next(pw);
            let end_y = next(ph);
            let start = next(end_y + 9) as isize - 8;
            let st = Stripe {
                start,
                // `StripeEndY` as the spec sets it: 32 or 64 rows after `StripeStartY`.
                end: start + if next(2) == 0 { 31 } else { 63 },
                end_x: end_x as isize,
                end_y: end_y as isize,
            };
            // As in `lr_stripe`: the block lies inside the plane.
            let x = next(end_x + 1);
            let y = next(end_y + 1);
            let w = 1 + next(BLOCK_W.min(end_x + 1 - x));
            let h = 1 + next(STRIPE_H);
            source_window(&cdef, &curr, &st, x, y, w, h, &mut win);
            for r in 0..h + 6 {
                let mut yy = (y as isize + r as isize - 3).clamp(0, st.end_y);
                let src = if yy < st.start {
                    yy = yy.max(st.start - 2);
                    &curr
                } else if yy > st.end {
                    yy = yy.min(st.end + 2);
                    &curr
                } else {
                    &cdef
                };
                for c in 0..w + 6 {
                    let xx = (x as isize + c as isize - 3).clamp(0, st.end_x) as usize;
                    let want = i32::from(src.px[yy as usize * src.stride + xx]);
                    assert_eq!(win[r * (w + 6) + c], want, "x {x} y {y} w {w} at {r}, {c}");
                }
            }
        }
    }
}
