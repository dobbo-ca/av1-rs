//! CDEF, the constrained directional enhancement filter (spec 7.15): per 8x8 block a direction
//! and variance search on luma (7.15.2) and the directional primary / secondary tap filter on
//! every plane (7.15.3). Reads `CurrFrame` (after the loop filter), writes `CdefFrame`.
//! Written from the spec text; this scalar code is the oracle of the SIMD kernels in
//! `src/simd/cdef` (which follow the dav1d design, see there).

use crate::bits::floor_log2;
use crate::consts::{BLOCK_8X8, BLOCK_64X64, MI_SIZE};
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::frame::{FrameState, Plane, View};
use crate::header::FrameHeader;
use crate::ledger::Ledger;
use crate::seq::SequenceHeader;
use crate::simd;
use crate::tables::NUM_4X4_BLOCKS_WIDE;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// `Cdef_Uv_Dir[ subsampling_x ][ subsampling_y ][ yDir ]` (spec 7.15.1).
const CDEF_UV_DIR: [[[u8; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

/// `Div_Table` (spec 7.15.2).
const DIV_TABLE: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// `Cdef_Pri_Taps` (spec 7.15.3).
pub(crate) const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];

/// `Cdef_Sec_Taps` (spec 7.15.3).
const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// `Cdef_Directions[ dir ][ k ] = [ dy, dx ]` (spec 7.15.3).
pub(crate) const CDEF_DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];

/// The CDEF process (spec 7.15): reads `curr` (`CurrFrame`), writes `CdefFrame` into `out`,
/// which must have the same planes, widths, heights and strides as `curr` (a plane that does
/// not match is left untouched). `out` is first a copy of the input; with `enable_cdef` off,
/// `coded_lossless` or `allow_intrabc` it stays that copy.
pub(crate) fn cdef_frame(
    fs: &FrameState,
    curr: &[Plane],
    seq: &SequenceHeader,
    fh: &FrameHeader,
    out: &mut [Plane],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let num_planes = usize::from(seq.num_planes).min(curr.len()).min(out.len());
    for plane in 0..num_planes {
        let (src, dst) = (&curr[plane], &mut out[plane]);
        if src.stride == dst.stride && src.px.len() == dst.px.len() {
            dst.px.copy_from_slice(&src.px);
        }
    }
    if !cdef_on(fs, curr, seq, fh, out) {
        return Ok(());
    }
    // At most 3 planes: the views live on the stack, so the call allocates nothing.
    let mut planes = out.iter_mut().take(num_planes).map(View::of);
    let mut views: [View<'_>; 3] = std::array::from_fn(|_| {
        planes.next().unwrap_or(View {
            px: &mut [],
            stride: 0,
            x0: 0,
            y0: 0,
        })
    });
    let n = num_planes.min(3);
    cdef_rows(fs, curr, seq, fh, 0..fs.mi_rows, &mut views[..n], cancel)
}

/// True when [`cdef_frame`] filters (it copies otherwise): CDEF is on for the frame, and every
/// plane of `curr` and `out` holds the 8x8 blocks of the mode info grid with equal strides.
pub(crate) fn cdef_on(
    fs: &FrameState,
    curr: &[Plane],
    seq: &SequenceHeader,
    fh: &FrameHeader,
    out: &[Plane],
) -> bool {
    if !seq.enable_cdef || fh.coded_lossless || fh.allow_intrabc {
        return false;
    }
    let num_planes = usize::from(seq.num_planes).min(curr.len()).min(out.len());
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
    let (sub_x, sub_y) = (
        usize::from(seq.subsampling_x),
        usize::from(seq.subsampling_y),
    );
    // Every plane must hold the 8x8 blocks of the mi grid, in both buffers.
    for plane in 0..num_planes {
        let (sx, sy) = if plane == 0 { (0, 0) } else { (sub_x, sub_y) };
        let (cols, rows) = ((mi_cols * MI_SIZE) >> sx, (mi_rows * MI_SIZE) >> sy);
        for p in [&curr[plane], &out[plane]] {
            if p.stride < cols || p.px.len() < p.stride * rows {
                return false;
            }
        }
        if curr[plane].stride != out[plane].stride {
            return false;
        }
    }
    mi_rows.is_multiple_of(2)
        && fs.cdef_idx.len() >= mi_rows * mi_cols
        && fs.skips.len() >= mi_rows * mi_cols
}

/// The CDEF of the 8x8 blocks at mode info rows `rows` (`rows.start` a multiple of 16, or 0),
/// for a frame where [`cdef_on`] holds: reads `curr`, writes plane `p` of the blocks into
/// `out[ p ]`, whole rows of that `CdefFrame` plane (same stride) holding at least the blocks'
/// rows and already holding the `curr` samples. The cancel flag is checked every 16 rows.
pub(crate) fn cdef_rows(
    fs: &FrameState,
    curr: &[Plane],
    seq: &SequenceHeader,
    fh: &FrameHeader,
    rows: Range<usize>,
    out: &mut [View<'_>],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let num_planes = usize::from(seq.num_planes).min(curr.len()).min(out.len());
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
    let (sub_x, sub_y) = (
        usize::from(seq.subsampling_x),
        usize::from(seq.subsampling_y),
    );
    let step4 = NUM_4X4_BLOCKS_WIDE[BLOCK_8X8];
    let cdef_mask4 = !(NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] - 1);
    let coeff_shift = u32::from(seq.bit_depth).saturating_sub(8);
    let best = simd::best();
    for r in rows.step_by(step4) {
        if r.is_multiple_of(16) {
            check_cancel(cancel)?;
        }
        for c in (0..mi_cols).step_by(step4) {
            let idx = fs.cdef_idx[(r & cdef_mask4) * mi_cols + (c & cdef_mask4)];
            // idx -1 (not coded): the block stays the copy.
            let Ok(idx) = usize::try_from(idx) else {
                continue;
            };
            if idx >= 8 {
                continue;
            }
            let sk =
                |rr: usize, cc: usize| fs.skips.get(rr * mi_cols + cc).copied().unwrap_or(true);
            if sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1) {
                continue;
            }
            let (y_dir, var) =
                simd::cdef::direction(best, &curr[0], r, c, u32::from(seq.bit_depth));
            let mut pri_str = i32::from(fh.cdef_y_pri_strength[idx]) << coeff_shift;
            let sec_str = i32::from(fh.cdef_y_sec_strength[idx]) << coeff_shift;
            let dir = if pri_str == 0 { 0 } else { y_dir };
            let var_str = if (var >> 6) != 0 {
                floor_log2((var >> 6) as u32).min(12) as i32
            } else {
                0
            };
            pri_str = if var != 0 {
                (pri_str * (4 + var_str) + 8) >> 4
            } else {
                0
            };
            let damping = i32::from(fh.cdef_damping) + coeff_shift as i32;
            let ctx = FilterCtx {
                mi_rows,
                mi_cols,
                coeff_shift,
            };
            simd::cdef::filter(
                best,
                &ctx,
                &curr[0],
                &mut out[0],
                0,
                (sub_x, sub_y),
                r,
                c,
                (pri_str, sec_str),
                damping,
                dir,
            );
            if num_planes == 1 {
                continue;
            }
            let pri_str = i32::from(fh.cdef_uv_pri_strength[idx]) << coeff_shift;
            let sec_str = i32::from(fh.cdef_uv_sec_strength[idx]) << coeff_shift;
            let dir = if pri_str == 0 {
                0
            } else {
                usize::from(CDEF_UV_DIR[sub_x & 1][sub_y & 1][y_dir])
            };
            let damping = i32::from(fh.cdef_damping) + coeff_shift as i32 - 1;
            for plane in 1..num_planes.min(3) {
                simd::cdef::filter(
                    best,
                    &ctx,
                    &curr[plane],
                    &mut out[plane],
                    plane,
                    (sub_x, sub_y),
                    r,
                    c,
                    (pri_str, sec_str),
                    damping,
                    dir,
                );
            }
        }
    }
    Ok(())
}

/// `CdefFrame` as new planes: allocates planes shaped like `fs.curr_frame` (charged to the
/// frame's ledger), then runs [`cdef_frame`] into them.
///
/// # Errors
/// `AllocLimit` or `OutOfMemory` from the plane allocations.
pub(crate) fn cdef_alloc(
    fs: &FrameState,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    cancel: &AtomicBool,
) -> Result<Vec<Plane>, Error> {
    let mut out = cdef_planes(&fs.curr_frame, &fs.ledger)?;
    cdef_frame(fs, &fs.curr_frame, seq, fh, &mut out, cancel)?;
    Ok(out)
}

/// Zeroed planes shaped like `curr` (sizes and strides), charged to `ledger`.
///
/// # Errors
/// `AllocLimit` or `OutOfMemory` from the plane allocations.
pub(crate) fn cdef_planes(curr: &[Plane], ledger: &Arc<Ledger>) -> Result<Vec<Plane>, Error> {
    let mut out = ledger.planes(curr.len())?;
    for p in curr {
        let mut q = Plane::new(p.width, p.height, ledger)?;
        q.stride = p.stride;
        out.push(q);
    }
    Ok(out)
}

/// The frame facts the filter needs.
pub(crate) struct FilterCtx {
    pub(crate) mi_rows: usize,
    pub(crate) mi_cols: usize,
    pub(crate) coeff_shift: u32,
}

/// CDEF direction process (spec 7.15.2) for the 8x8 luma block at 4x4 position (`r`, `c`):
/// returns (`yDir`, `var`).
pub(crate) fn cdef_direction(luma: &Plane, r: usize, c: usize, bit_depth: u32) -> (usize, i64) {
    let mut cost = [0i64; 8];
    let mut partial = [[0i64; 15]; 8];
    let (x0, y0) = (c * MI_SIZE, r * MI_SIZE);
    let shift = bit_depth.saturating_sub(8);
    for i in 0..8 {
        for j in 0..8 {
            let v = luma
                .px
                .get((y0 + i) * luma.stride + x0 + j)
                .copied()
                .unwrap_or(0);
            let x = i64::from(v >> shift) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
    }
    let mut best_cost = 0;
    let mut y_dir = 0;
    for i in 0..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            y_dir = i;
        }
    }
    (y_dir, (best_cost - cost[(y_dir + 4) & 7]) >> 10)
}

/// `constrain( diff, threshold, damping )` (spec 7.15.3).
fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let damping_adj = (damping - floor_log2(threshold as u32) as i32).clamp(0, 31);
    let mag = diff.abs();
    let v = (threshold - (mag >> damping_adj)).clamp(0, mag);
    if diff < 0 { -v } else { v }
}

/// CDEF filter process (spec 7.15.3) for one plane of the 8x8 luma block at (`r`, `c`): reads
/// `src` (`CurrFrame`), writes `dst` (whole rows of `CdefFrame`, the block's rows among them).
/// `sub` is `(subsampling_x, subsampling_y)`, `strengths` is `(priStr, secStr)`.
/// `is_inside_filter_region` is the whole frame, so a tap is
/// available when its sample lies inside the `MiRows` x `MiCols` area.
pub(crate) fn cdef_filter(
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
    let (pri_str, sec_str) = strengths;
    let (sub_x, sub_y) = if plane > 0 { sub } else { (0, 0) };
    let x0 = ((c * MI_SIZE) >> sub_x) as i32;
    let y0 = ((r * MI_SIZE) >> sub_y) as i32;
    let (w, h) = (8usize >> sub_x, 8usize >> sub_y);
    // A sample (y, x) is available when its mi position is inside the frame (y, x >= 0 too).
    let rows = ((ctx.mi_rows * MI_SIZE) >> sub_y) as i32;
    let cols = ((ctx.mi_cols * MI_SIZE) >> sub_x) as i32;
    let tap_set = ((pri_str >> ctx.coeff_shift) & 1) as usize;
    let at = |y: i32, x: i32| -> Option<i32> {
        if y < 0 || x < 0 || y >= rows || x >= cols {
            return None;
        }
        src.px
            .get(y as usize * src.stride + x as usize)
            .map(|&v| i32::from(v))
    };
    for i in 0..h as i32 {
        // The block row in `dst`, which starts at plane row `dst.y0`.
        let Some(row) = ((y0 + i) as usize).checked_sub(dst.y0) else {
            continue;
        };
        for j in 0..w as i32 {
            let x = at(y0 + i, x0 + j).unwrap_or(0);
            let (mut sum, mut max, mut min) = (0, x, x);
            for k in 0..2 {
                for sign in [-1, 1] {
                    let tap = |d: usize| {
                        let [dy, dx] = CDEF_DIRECTIONS[d][k];
                        at(y0 + i + sign * dy, x0 + j + sign * dx)
                    };
                    if let Some(p) = tap(dir) {
                        sum += CDEF_PRI_TAPS[tap_set][k] * constrain(p - x, pri_str, damping);
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for d in [(dir + 6) & 7, (dir + 2) & 7] {
                        if let Some(s) = tap(d) {
                            sum += CDEF_SEC_TAPS[tap_set][k] * constrain(s - x, sec_str, damping);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = (x + ((8 + sum - i32::from(sum < 0)) >> 4)).clamp(min, max);
            let idx = row * dst.stride + (x0 + j) as usize;
            if let Some(d) = dst.px.get_mut(idx) {
                *d = v as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Plane {
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 20, 0).unwrap();
        let mut p = Plane::new(w, h, &ledger).unwrap();
        for y in 0..h {
            for x in 0..w {
                p.px[y * w + x] = f(y, x);
            }
        }
        p
    }

    #[test]
    fn constrain_values() {
        // threshold 0 gives 0
        assert_eq!(constrain(5, 0, 6), 0);
        // damping_adj = max(0, 6 - floor_log2(4)) = 4: |5| >> 4 = 0, min(5, 4 - 0) = 4
        assert_eq!(constrain(5, 4, 6), 4);
        assert_eq!(constrain(-5, 4, 6), -4);
        // small diff passes: min(1, 4 - 0) = 1
        assert_eq!(constrain(1, 4, 6), 1);
        // damping 2: adj = 0, 4 - 3 = 1
        assert_eq!(constrain(3, 4, 2), 1);
        // diff beyond threshold is cut to 0: 4 - 8 < 0
        assert_eq!(constrain(8, 4, 2), 0);
        assert_eq!(constrain(0, 4, 2), 0);
    }

    #[test]
    fn vertical_edge_direction() {
        // Vertical edge: partial[6][j] sums column j, so cost[6] is the largest.
        let p = plane(8, 8, |_, x| if x < 4 { 40 } else { 200 });
        let (dir, var) = cdef_direction(&p, 0, 0, 8);
        assert_eq!(dir, 6);
        assert!(var > 0);
        // The transposed pattern (a horizontal edge) gives direction 2.
        let p = plane(8, 8, |y, _| if y < 4 { 40 } else { 200 });
        assert_eq!(cdef_direction(&p, 0, 0, 8).0, 2);
    }

    #[test]
    fn flat_image_is_unchanged() {
        let src = plane(16, 16, |_, _| 100);
        let mut dst = plane(16, 16, |_, _| 100);
        let ctx = FilterCtx {
            mi_rows: 4,
            mi_cols: 4,
            coeff_shift: 0,
        };
        for dir in 0..8 {
            cdef_filter(
                &ctx,
                &src,
                &mut View::of(&mut dst),
                0,
                (1, 1),
                0,
                0,
                (15, 4),
                6,
                dir,
            );
            cdef_filter(
                &ctx,
                &src,
                &mut View::of(&mut dst),
                1,
                (1, 1),
                0,
                0,
                (7, 2),
                5,
                dir,
            );
            assert!(dst.px.iter().all(|&v| v == 100));
        }
    }
}
