//! Deblocking loop filter, spec 7.14. Filters `CurrFrame` in place: for each plane all vertical
//! edges of the frame, then all horizontal edges.
//!
//! Only intra frames are decoded, so `RefFrames[ row ][ col ][ 0 ]` is always `INTRA_FRAME` and the
//! `loop_filter_mode_deltas` / `modeType` branch of 7.14.5 never applies.
//!
//! The filter size and level are worked out once per 4x4 edge (7.14.2 to 7.14.5), and runs of
//! adjacent edges with the same size and level along one edge line go to the `simd::lf` kernels
//! in one call. Filtering a vertical edge reads and writes one sample row, and a horizontal edge
//! one sample column, so taking vertical edges column by column within a strip of 16 mode info
//! rows (instead of in raster order) keeps the order of the filters on every sample row: the
//! output is the same.
//!
//! For the same reason the threaded wrapup (docs/threading.md) filters the vertical edges of a
//! plane in row bands and then its horizontal edges in column strips, each with
//! [`loop_filter_rows`], and gets the samples of the whole-frame order.
//!
//! An edge line is walked by transform block segments, not by every 4x4 position: from a 4x4
//! edge to the end of its transform block along the line, every position has the same transform
//! size and the same mode info on the current side of the edge, so the transform edge test, the
//! block edge test and the current side's level are worked out once per segment, and segments
//! that are not on a transform edge are skipped whole. Only the previous side (its transform
//! size, and its level when the current side's is 0) is read per position. This rests on an
//! invariant of the decoder: `transform_block` writes a transform block's size over its whole
//! rectangle of `loopfilter_tx_sizes` (clipped to the mode info grid), transform blocks are
//! aligned to their size and lie in one block, chroma reads the mode info at
//! (`row | subsampling_y`, `col | subsampling_x`), which is in the block that coded the chroma,
//! and the arrays are cleared between frames. A layout that broke it would not panic (every
//! array read is checked) but would be filtered differently from 7.14.2.
//!
//! The levels of a frame without loop filter deltas (`delta_lf_present` 0, so every `DeltaLF`
//! is 0) depend only on the segment, and come from a table made once per [`loop_filter_rows`]
//! call, as do the limits of each level.

use crate::consts::{INTRA_FRAME, MAX_LOOP_FILTER, MAX_SEGMENTS, MI_SIZE, SEG_LVL_ALT_LF_Y_V};
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::frame::{FrameState, Plane, View};
use crate::header::FrameHeader;
use crate::seq::SequenceHeader;
use crate::simd::{self, Simd};
use crate::tables::{BLOCK_HEIGHT, BLOCK_WIDTH, SUBSAMPLED_SIZE, TX_HEIGHT, TX_WIDTH};
use std::ops::Range;
use std::sync::atomic::AtomicBool;

/// Spec 7.14.1 (loop filter process), invoked by `decode_frame_wrapup` (spec 7.4) when
/// `loop_filter_level[ 0 ]` or `loop_filter_level[ 1 ]` is non-zero; otherwise this returns
/// without touching the frame.
pub(crate) fn loop_filter_frame(
    fs: &mut FrameState,
    _seq: &SequenceHeader,
    fh: &FrameHeader,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    if !lf_on(fh) {
        return Ok(());
    }
    // The planes are filtered through views, apart from the mode info they read.
    let mut curr = std::mem::take(&mut fs.curr_frame);
    let r = filter_planes(fs, fh, &mut curr, cancel);
    fs.curr_frame = curr;
    r
}

/// True when the frame is loop filtered: `loop_filter_level[ 0 ]` or `[ 1 ]` is non-zero.
pub(crate) fn lf_on(fh: &FrameHeader) -> bool {
    fh.loop_filter_level[0] != 0 || fh.loop_filter_level[1] != 0
}

/// True when plane `plane` (0 to 2) of a loop filtered frame is filtered.
pub(crate) fn lf_plane_on(fh: &FrameHeader, plane: usize) -> bool {
    plane == 0 || fh.loop_filter_level[1 + plane] != 0
}

/// The loop filter of the whole frame on `curr` (`CurrFrame`), plane by plane.
fn filter_planes(
    fs: &FrameState,
    fh: &FrameHeader,
    curr: &mut [Plane],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    for (plane, p) in curr.iter_mut().enumerate().take(fs.num_planes) {
        if lf_plane_on(fh, plane) {
            for pass in 0..2 {
                let (rows, cols) = (0..fs.mi_rows, 0..fs.mi_cols);
                loop_filter_rows(fs, fh, plane, pass, rows, cols, &mut View::of(p), cancel)?;
            }
        }
    }
    Ok(())
}

/// The edges of pass `pass` (0: vertical edges, 1: horizontal edges) of plane `plane` at mode
/// info rows `rows` and columns `cols`, on the samples in `v`. `rows.start` and `cols.start` are
/// multiples of 16 (or 0). The edges' samples must all be in `v`: the rows of `rows` across the
/// full plane width for pass 0, the columns of `cols` down the full plane height for pass 1. The
/// cancel flag is checked every 16 rows.
pub(crate) fn loop_filter_rows(
    fs: &FrameState,
    fh: &FrameHeader,
    plane: usize,
    pass: usize,
    rows: Range<usize>,
    cols: Range<usize>,
    v: &mut View<'_>,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let l = Lines::new(fs, fh, plane, pass);
    let (row_step, col_step) = (1 << l.sub_y, 1 << l.sub_x);
    // 7.14.2 filters no edge at x >= FrameWidth or y >= FrameHeight, and no vertical edge at
    // x == 0 or horizontal edge at y == 0: those positions and lines are not visited.
    let row_end = rows.end.min(fs.frame_height.div_ceil(MI_SIZE));
    let col_end = cols.end.min(fs.frame_width.div_ceil(MI_SIZE));
    // Strips of 16 mode info rows (a 64x64 superblock row).
    for strip in (rows.start..row_end).step_by(16) {
        check_cancel(cancel)?;
        let end = row_end.min(strip + 16);
        if pass == 0 {
            // Vertical edges: a run goes down one column of the strip.
            for col in (cols.start.max(col_step)..col_end).step_by(col_step) {
                l.line(col, strip..end, v);
            }
        } else {
            // Horizontal edges: a run goes along one row.
            for row in (strip.max(row_step)..end).step_by(row_step) {
                l.line(row, cols.start..col_end, v);
            }
        }
    }
    Ok(())
}

/// What the edges of one [`loop_filter_rows`] call share: the plane and pass, and the level and
/// limits tables.
struct Lines<'a> {
    fs: &'a FrameState,
    fh: &'a FrameHeader,
    simd: Simd,
    plane: usize,
    /// 0: vertical edges, 1: horizontal edges.
    pass: usize,
    /// The plane's subsampling (0 for luma).
    sub_x: usize,
    sub_y: usize,
    /// `loopfilter_tx_sizes[ plane ]`.
    tx_sizes: &'a [u8],
    /// The largest filter size of the plane (7.14.3): 16 for luma, 8 for chroma.
    cap: usize,
    /// The level of each segment when the frame has no loop filter deltas.
    lvls: Option<[i32; MAX_SEGMENTS]>,
    /// `limits( lvl, loop_filter_sharpness )` for each level `lvl`.
    lims: [[i32; 3]; MAX_LOOP_FILTER + 1],
}

/// The current side of the 4x4 edges of a transform block segment that 7.14.2 may filter.
#[derive(Clone, Copy)]
struct Seg {
    /// The current transform size across the edge, in samples.
    cross: usize,
    /// The current block's level; 0 when each edge takes the level of its previous block.
    lvl: i32,
}

impl<'a> Lines<'a> {
    /// The edges of pass `pass` of plane `plane`, with the tables of the frame of `fh`.
    fn new(fs: &'a FrameState, fh: &'a FrameHeader, plane: usize, pass: usize) -> Self {
        let (sub_x, sub_y) = if plane == 0 {
            (0, 0)
        } else {
            (fs.subsampling_x, fs.subsampling_y)
        };
        let lvls = (!fh.delta_lf_present)
            .then(|| std::array::from_fn(|s| seg_level(fh, s, [0; 4], plane, pass)));
        Lines {
            fs,
            fh,
            simd: simd::best(),
            plane,
            pass,
            sub_x,
            sub_y,
            tx_sizes: &fs.loopfilter_tx_sizes[plane],
            cap: if plane == 0 { 16 } else { 8 },
            lvls,
            lims: std::array::from_fn(|l| limits(l as i32, fh.loop_filter_sharpness)),
        }
    }

    /// Filters the 4x4 edges of one edge line on the samples in `v`: column `at` at mode info
    /// rows `along` (pass 0) or row `at` at mode info columns `along` (pass 1). `along` starts
    /// at a multiple of the subsampling step and is walked by it.
    fn line(&self, at: usize, along: Range<usize>, v: &mut View<'_>) {
        let sub = if self.pass == 0 {
            self.sub_y
        } else {
            self.sub_x
        };
        let pos = |a: usize| if self.pass == 0 { (a, at) } else { (at, a) };
        let mut run = None;
        let mut a = along.start;
        while a < along.end {
            let (row, col) = pos(a);
            let (span, seg) = self.segment(row, col);
            let next = a + (span << sub);
            match seg {
                Some(s) => {
                    for b in (a..next.min(along.end)).step_by(1 << sub) {
                        let (row, col) = pos(b);
                        self.push(&mut run, self.edge(s, row, col), v);
                    }
                }
                None => self.flush(run.take(), v),
            }
            a = next;
        }
        self.flush(run, v);
    }

    /// For the 4x4 edge at mode info (`row`, `col`): the number of edges from it to the end of
    /// its transform block along the edge line (at least 1), and their current side when they
    /// may be filtered (7.14.2 up to the filter size), else `None`.
    fn segment(&self, row: usize, col: usize) -> (usize, Option<Seg>) {
        let fs = self.fs;
        let (x_p, y_p) = ((col * MI_SIZE) >> self.sub_x, (row * MI_SIZE) >> self.sub_y);
        let (row, col) = (row | self.sub_y, col | self.sub_x);
        let tx_idx = (row >> self.sub_y) * fs.mi_cols + (col >> self.sub_x);
        let Some(&tx_sz) = self.tx_sizes.get(tx_idx) else {
            return (1, None);
        };
        let tx_sz = usize::from(tx_sz);
        // The transform size and plane position along the edge line, and across it.
        let ((along, at), (cross, across)) = if self.pass == 0 {
            ((TX_HEIGHT[tx_sz], y_p), (TX_WIDTH[tx_sz], x_p))
        } else {
            ((TX_WIDTH[tx_sz], x_p), (TX_HEIGHT[tx_sz], y_p))
        };
        // Block and transform sizes are powers of two, so a mask test is the spec's modulo.
        let span = (along - (at & (along - 1))) / MI_SIZE;
        if (across & (cross - 1)) != 0 {
            return (span, None);
        }
        let mi_idx = row * fs.mi_cols + col;
        let (Some(&mi_size), Some(&skip), Some(&ref0)) = (
            fs.mi_sizes.get(mi_idx),
            fs.skips.get(mi_idx),
            fs.ref_frames0.get(mi_idx),
        ) else {
            return (span, None);
        };
        let plane_size = SUBSAMPLED_SIZE[usize::from(mi_size)][self.sub_x][self.sub_y];
        if plane_size >= BLOCK_WIDTH.len() {
            return (span, None);
        }
        let block = if self.pass == 0 {
            BLOCK_WIDTH[plane_size]
        } else {
            BLOCK_HEIGHT[plane_size]
        };
        let is_block_edge = (across & (block - 1)) == 0;
        let is_intra = ref0 <= INTRA_FRAME as i8;
        if !is_block_edge && skip && !is_intra {
            return (span, None);
        }
        let lvl = self.level(mi_idx);
        (span, Some(Seg { cross, lvl }))
    }

    /// The rest of 7.14.2 for the 4x4 edge at mode info (`row`, `col`) of a segment with
    /// current side `s`: the filter size (7.14.3) and the level with its fallback to the
    /// previous block. `None` when the edge is not filtered.
    fn edge(&self, s: Seg, row: usize, col: usize) -> Option<EdgeParams> {
        let mi_cols = self.fs.mi_cols;
        let (x_p, y_p) = ((col * MI_SIZE) >> self.sub_x, (row * MI_SIZE) >> self.sub_y);
        let (row, col) = (row | self.sub_y, col | self.sub_x);
        // The walk never visits x == 0 (pass 0) or y == 0 (pass 1), so these do not wrap.
        let (prev_row, prev_col) = if self.pass == 0 {
            (row, col - (1 << self.sub_x))
        } else {
            (row - (1 << self.sub_y), col)
        };
        let prev_tx_idx = (prev_row >> self.sub_y) * mi_cols + (prev_col >> self.sub_x);
        let prev_tx_sz = usize::from(*self.tx_sizes.get(prev_tx_idx)?);
        let prev = if self.pass == 0 {
            TX_WIDTH[prev_tx_sz]
        } else {
            TX_HEIGHT[prev_tx_sz]
        };
        let lvl = if s.lvl != 0 {
            s.lvl
        } else {
            self.level(prev_row * mi_cols + prev_col)
        };
        (lvl != 0).then_some(EdgeParams {
            x: x_p,
            y: y_p,
            size: prev.min(s.cross).min(self.cap),
            lvl,
        })
    }

    /// `filter_level` of the mode info at index `mi_idx`, from the segment table when there is
    /// one.
    fn level(&self, mi_idx: usize) -> i32 {
        let segment = self
            .fs
            .segment_ids
            .get(mi_idx)
            .map_or(0, |&s| usize::from(s));
        if let Some(t) = &self.lvls
            && let Some(&lvl) = t.get(segment)
        {
            return lvl;
        }
        filter_level(self.fs, self.fh, mi_idx, self.plane, self.pass)
    }

    /// Adds the next 4x4 edge `e` of an edge line (`None`: not filtered) to `run` (an edge and
    /// the number of edges from it): `e` extends the run when it has the run's size and level,
    /// else the run is filtered on the samples in `v` and `e` starts the next one.
    fn push(&self, run: &mut Option<(EdgeParams, usize)>, e: Option<EdgeParams>, v: &mut View<'_>) {
        if let (Some((r, count)), Some(e)) = (run.as_mut(), e)
            && r.size == e.size
            && r.lvl == e.lvl
        {
            *count += 1;
            return;
        }
        self.flush(std::mem::replace(run, e.map(|e| (e, 1))), v);
    }

    /// Filters the `count` adjacent 4x4 edges of `run` (`(r, count)`), all with `r`'s size and
    /// level, on the samples in `v`.
    fn flush(&self, run: Option<(EdgeParams, usize)>, v: &mut View<'_>) {
        let Some((r, count)) = run else {
            return;
        };
        let (Some(x), Some(y)) = (r.x.checked_sub(v.x0), r.y.checked_sub(v.y0)) else {
            return;
        };
        let stride = v.stride;
        // Across the edge, and from one sample to the next along it.
        let (step, along) = if self.pass == 0 {
            (1, stride)
        } else {
            (stride, 1)
        };
        let lims = usize::try_from(r.lvl)
            .ok()
            .and_then(|l| self.lims.get(l))
            .copied()
            .unwrap_or_else(|| limits(r.lvl, self.fh.loop_filter_sharpness));
        let edge = simd::lf::Edge {
            pos: y * stride + x,
            step,
            along,
            n: count * MI_SIZE,
            plane: self.plane,
            bd: u32::from(self.fs.bit_depth),
            lims,
            filter_size: r.size,
        };
        simd::lf::filter_edge(self.simd, v.px, &edge);
    }
}

/// The 4x4 edge at plane sample (`x`, `y`) that 7.14.2 filters, with its filter size (7.14.3)
/// and level (7.14.4).
#[derive(Clone, Copy)]
struct EdgeParams {
    x: usize,
    y: usize,
    size: usize,
    lvl: i32,
}

/// Spec 7.14.2 (edge loop filter process) up to the sample loop: `None` when the 4x4 edge at
/// mode info (`row`, `col`) is not filtered. The per-position oracle of the segment walk.
#[cfg(test)]
fn edge_params(
    fs: &FrameState,
    fh: &FrameHeader,
    plane: usize,
    pass: usize,
    row: usize,
    col: usize,
) -> Option<EdgeParams> {
    let (sub_x, sub_y) = if plane == 0 {
        (0, 0)
    } else {
        (fs.subsampling_x, fs.subsampling_y)
    };
    let (dx, dy) = if pass == 0 { (1, 0) } else { (0, 1) };
    let x = col * MI_SIZE;
    let y = row * MI_SIZE;
    let row = row | sub_y;
    let col = col | sub_x;
    if x >= fs.frame_width || y >= fs.frame_height || (pass == 0 && x == 0) || (pass == 1 && y == 0)
    {
        return None;
    }
    let x_p = x >> sub_x;
    let y_p = y >> sub_y;
    let prev_row = row - (dy << sub_y);
    let prev_col = col - (dx << sub_x);
    let mi_idx = row * fs.mi_cols + col;
    let prev_mi_idx = prev_row * fs.mi_cols + prev_col;
    let tx_idx = (row >> sub_y) * fs.mi_cols + (col >> sub_x);
    let prev_tx_idx = (prev_row >> sub_y) * fs.mi_cols + (prev_col >> sub_x);
    // Every `None` below means "not filtered", so the order of the checks does not change the
    // result: the transform edge test, which rejects most positions, comes first. Block and
    // transform sizes are powers of two, so a mask test is the spec's modulo.
    let tx_sz = usize::from(*fs.loopfilter_tx_sizes[plane].get(tx_idx)?);
    let is_tx_edge = if pass == 0 {
        (x_p & (TX_WIDTH[tx_sz] - 1)) == 0
    } else {
        (y_p & (TX_HEIGHT[tx_sz] - 1)) == 0
    };
    if !is_tx_edge {
        return None;
    }
    let (Some(&mi_size), Some(&prev_tx_sz), Some(&skip), Some(&ref0)) = (
        fs.mi_sizes.get(mi_idx),
        fs.loopfilter_tx_sizes[plane].get(prev_tx_idx),
        fs.skips.get(mi_idx),
        fs.ref_frames0.get(mi_idx),
    ) else {
        return None;
    };
    let prev_tx_sz = usize::from(prev_tx_sz);
    let plane_size = SUBSAMPLED_SIZE[usize::from(mi_size)][sub_x][sub_y];
    if plane_size >= BLOCK_WIDTH.len() {
        return None;
    }
    let is_intra = ref0 <= INTRA_FRAME as i8;
    let is_block_edge = if pass == 0 {
        (x_p & (BLOCK_WIDTH[plane_size] - 1)) == 0
    } else {
        (y_p & (BLOCK_HEIGHT[plane_size] - 1)) == 0
    };
    let apply_filter = is_block_edge || !skip || is_intra;
    if !apply_filter {
        return None;
    }
    // 7.14.3 filter size process.
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz].min(TX_WIDTH[tx_sz])
    } else {
        TX_HEIGHT[prev_tx_sz].min(TX_HEIGHT[tx_sz])
    };
    let filter_size = if plane == 0 {
        base_size.min(16)
    } else {
        base_size.min(8)
    };
    let mut lvl = filter_level(fs, fh, mi_idx, plane, pass);
    if lvl == 0 {
        lvl = filter_level(fs, fh, prev_mi_idx, plane, pass);
    }
    if lvl == 0 {
        return None;
    }
    Some(EdgeParams {
        x: x_p,
        y: y_p,
        size: filter_size,
        lvl,
    })
}

/// Spec 7.14.4: `[ limit, blimit, thresh ]` for level `lvl` (1 to 63) and sharpness `sharp`.
pub(crate) fn limits(lvl: i32, sharp: u8) -> [i32; 3] {
    let sharp = i32::from(sharp);
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        (lvl >> shift).max(1)
    };
    let blimit = 2 * (lvl + 2) + limit;
    let thresh = lvl >> 4;
    [limit, blimit, thresh]
}

/// Spec 7.14.4 (adaptive filter strength, `lvl` output) with 7.14.5 (selection) for the mode
/// info at index `mi_idx`.
fn filter_level(
    fs: &FrameState,
    fh: &FrameHeader,
    mi_idx: usize,
    plane: usize,
    pass: usize,
) -> i32 {
    let segment = fs.segment_ids.get(mi_idx).map_or(0, |&s| usize::from(s));
    let delta_lfs = fs.delta_lfs.get(mi_idx).copied().unwrap_or([0; 4]);
    seg_level(fh, segment, delta_lfs, plane, pass)
}

/// [`filter_level`] of mode info with segment `segment` and `DeltaLFs` `delta_lfs`.
fn seg_level(
    fh: &FrameHeader,
    segment: usize,
    delta_lfs: [i8; 4],
    plane: usize,
    pass: usize,
) -> i32 {
    let i = if plane == 0 { pass } else { plane + 1 };
    let delta_lf = if fh.delta_lf_multi {
        delta_lfs[i]
    } else {
        delta_lfs[0]
    };
    let max = MAX_LOOP_FILTER as i32;
    let mut lvl_seg = (i32::from(delta_lf) + i32::from(fh.loop_filter_level[i])).clamp(0, max);
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if fh.segmentation_enabled && fh.feature_enabled[segment][feature] {
        lvl_seg = (lvl_seg + i32::from(fh.feature_data[segment][feature])).clamp(0, max);
    }
    if fh.loop_filter_delta_enabled {
        let n_shift = lvl_seg >> 5;
        lvl_seg += i32::from(fh.loop_filter_ref_deltas[INTRA_FRAME]) << n_shift;
        lvl_seg = lvl_seg.clamp(0, max);
    }
    lvl_seg
}

/// Spec 7.14.6 (sample filtering process) for the sample at `px[ pos ]` (`q0`); `step` is the
/// distance between samples across the edge (1 for vertical edges, the stride for horizontal).
/// `lims` is `[ limit, blimit, thresh ]`. Samples outside the buffer are never touched.
pub(crate) fn sample_filter(
    px: &mut [u16],
    pos: usize,
    step: usize,
    plane: usize,
    bd: u32,
    lims: [i32; 3],
    filter_size: usize,
) {
    let [limit, blimit, thresh] = lims;
    // 7.14.6.2: filterLen, and the samples each side of the edge it needs.
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };
    let m = match filter_len {
        4 => 2,
        6 => 3,
        8 => 4,
        _ => 7,
    };
    if pos < m * step || pos + (m - 1) * step >= px.len() {
        return;
    }
    // s[ 7 + k ] is the sample k steps from q0: p_i = s[ 6 - i ], q_i = s[ 7 + i ].
    let mut s = [0i32; 14];
    for i in 0..m {
        s[6 - i] = i32::from(px[pos - (i + 1) * step]);
        s[7 + i] = i32::from(px[pos + i * step]);
    }
    let p = |i: usize| s[6 - i];
    let q = |i: usize| s[7 + i];
    let sh = bd - 8;
    let hev_mask = (p(1) - p(0)).abs() > (thresh << sh) || (q(1) - q(0)).abs() > (thresh << sh);
    let (limit_bd, blimit_bd) = (limit << sh, blimit << sh);
    let mut bad = (p(1) - p(0)).abs() > limit_bd
        || (q(1) - q(0)).abs() > limit_bd
        || (p(0) - q(0)).abs() * 2 + (p(1) - q(1)).abs() / 2 > blimit_bd;
    if filter_len >= 6 {
        bad |= (p(2) - p(1)).abs() > limit_bd || (q(2) - q(1)).abs() > limit_bd;
    }
    if filter_len >= 8 {
        bad |= (p(3) - p(2)).abs() > limit_bd || (q(3) - q(2)).abs() > limit_bd;
    }
    if bad {
        return;
    }
    let one = 1 << sh;
    let mut flat = false;
    if filter_size >= 8 {
        let mut f = (p(1) - p(0)).abs() <= one
            && (q(1) - q(0)).abs() <= one
            && (p(2) - p(0)).abs() <= one
            && (q(2) - q(0)).abs() <= one;
        if filter_len >= 8 {
            f = f && (p(3) - p(0)).abs() <= one && (q(3) - q(0)).abs() <= one;
        }
        flat = f;
    }
    let mut flat2 = false;
    if filter_size >= 16 {
        flat2 = (4..=6).all(|i| (p(i) - p(0)).abs() <= one && (q(i) - q(0)).abs() <= one);
    }
    if filter_size == 4 || !flat {
        // 7.14.6.3 narrow filter.
        let lo = -(1 << (bd - 1));
        let hi = (1 << (bd - 1)) - 1;
        let c = |v: i32| v.clamp(lo, hi);
        let off = 0x80 << sh;
        let (ps1, ps0, qs0, qs1) = (p(1) - off, p(0) - off, q(0) - off, q(1) - off);
        let mut filter = if hev_mask { c(ps1 - qs1) } else { 0 };
        filter = c(filter + 3 * (qs0 - ps0));
        let filter1 = c(filter + 4) >> 3;
        let filter2 = c(filter + 3) >> 3;
        px[pos] = (c(qs0 - filter1) + off) as u16;
        px[pos - step] = (c(ps0 + filter2) + off) as u16;
        if !hev_mask {
            let filter = (filter1 + 1) >> 1;
            px[pos + step] = (c(qs1 - filter) + off) as u16;
            px[pos - 2 * step] = (c(ps1 + filter) + off) as u16;
        }
    } else {
        // 7.14.6.4 wide filter.
        let log2_size = if filter_size == 8 || !flat2 { 3 } else { 4 };
        let n: i32 = if log2_size == 4 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let n2: i32 = if log2_size == 3 && plane == 0 { 0 } else { 1 };
        let mut f = [0i32; 12];
        for i in -n..n {
            let mut t = 0;
            for j in -n..=n {
                let k = (i + j).clamp(-(n + 1), n);
                let tap = if j.abs() <= n2 { 2 } else { 1 };
                t += s[(7 + k) as usize] * tap;
            }
            f[(i + n) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
        }
        for i in -n..n {
            px[(pos as isize + i as isize * step as isize) as usize] = f[(i + n) as usize] as u16;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ledger::Ledger;
    use std::sync::Arc;

    /// Asserts the invariant of the segment walk (module doc) on `fs`: at every position that
    /// 7.14.2 visits, the transform block of the position's size, aligned to that size, has that
    /// size and the same mode info (read at the chroma position for chroma) over all of it that
    /// is in the frame.
    pub(crate) fn assert_transform_blocks_whole(fs: &FrameState) {
        let (rows, cols) = (
            fs.frame_height.div_ceil(MI_SIZE),
            fs.frame_width.div_ceil(MI_SIZE),
        );
        let mi = |row: usize, col: usize| {
            let i = row * fs.mi_cols + col;
            let (size, skip, ref0) = (fs.mi_sizes[i], fs.skips[i], fs.ref_frames0[i]);
            (size, skip, ref0, fs.segment_ids[i], fs.delta_lfs[i])
        };
        for plane in 0..fs.num_planes {
            let (sx, sy) = if plane == 0 {
                (0, 0)
            } else {
                (fs.subsampling_x, fs.subsampling_y)
            };
            let tx = &fs.loopfilter_tx_sizes[plane];
            for row in (0..rows).step_by(1 << sy) {
                for col in (0..cols).step_by(1 << sx) {
                    let (pr, pc) = (row >> sy, col >> sx);
                    let t = tx[pr * fs.mi_cols + pc];
                    let h = TX_HEIGHT[usize::from(t)] / MI_SIZE;
                    let w = TX_WIDTH[usize::from(t)] / MI_SIZE;
                    let (r0, c0) = (pr & !(h - 1), pc & !(w - 1));
                    for qr in r0..r0 + h {
                        for qc in c0..c0 + w {
                            let (r, c) = (qr << sy, qc << sx);
                            if r < rows && c < cols {
                                let (got, want) = (
                                    (tx[qr * fs.mi_cols + qc], mi(r | sy, c | sx)),
                                    (t, mi(row | sy, col | sx)),
                                );
                                assert_eq!(got, want, "plane {plane} at {row}, {col}: {r}, {c}");
                            }
                        }
                    }
                }
            }
        }
    }

    fn setup(left: u16, right: u16) -> (FrameState, FrameHeader, SequenceHeader) {
        let seq = SequenceHeader::parse(&crate::seq::tests::still_sequence_header(), 0).unwrap();
        let fh = FrameHeader {
            frame_width: 64,
            frame_height: 32,
            upscaled_width: 64,
            mi_cols: 16,
            mi_rows: 8,
            loop_filter_level: [32, 32, 0, 0],
            ..FrameHeader::default()
        };
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 20, 0).unwrap();
        let mut fs = FrameState::new(&fh, &seq, &ledger).unwrap();
        // All blocks BLOCK_4X4 with TX_4X4 (both 0 after allocation), intra, not skipped.
        let stride = fs.curr_frame[0].stride;
        for (i, v) in fs.curr_frame[0].px.iter_mut().enumerate() {
            *v = if i % stride < 32 { left } else { right };
        }
        (fs, fh, seq)
    }

    #[test]
    fn flat_block_is_unchanged() {
        let (mut fs, fh, seq) = setup(100, 100);
        let before = fs.curr_frame[0].px.clone();
        loop_filter_frame(&mut fs, &seq, &fh, &AtomicBool::new(false)).unwrap();
        assert_eq!(fs.curr_frame[0].px, before);
    }

    #[test]
    fn step_edge_uses_narrow_filter() {
        // Level 32, sharpness 0: limit 32, blimit 100, thresh 2. For p = 100, q = 108 and no hev:
        // filter = 3 * 8 = 24, filter1 = filter2 = 3, outer taps move by Round2( 3, 1 ) = 2.
        let (mut fs, fh, seq) = setup(100, 108);
        loop_filter_frame(&mut fs, &seq, &fh, &AtomicBool::new(false)).unwrap();
        let stride = fs.curr_frame[0].stride;
        for y in 0..32 {
            let r = &fs.curr_frame[0].px[y * stride + 28..y * stride + 36];
            assert_eq!(r, [100, 100, 102, 103, 105, 106, 108, 108]);
        }
    }

    /// `edge_params` in the spec's order (7.14.2), with `is_multiple_of` for the edge tests: the
    /// code before the transform edge test moved first. Returns `(x, y, size, lvl)`.
    fn edge_params_spec(
        fs: &FrameState,
        fh: &FrameHeader,
        plane: usize,
        pass: usize,
        row: usize,
        col: usize,
    ) -> Option<(usize, usize, usize, i32)> {
        let (sub_x, sub_y) = if plane == 0 {
            (0, 0)
        } else {
            (fs.subsampling_x, fs.subsampling_y)
        };
        let (dx, dy) = if pass == 0 { (1, 0) } else { (0, 1) };
        let (x, y) = (col * MI_SIZE, row * MI_SIZE);
        let (row, col) = (row | sub_y, col | sub_x);
        if x >= fs.frame_width
            || y >= fs.frame_height
            || (pass == 0 && x == 0)
            || (pass == 1 && y == 0)
        {
            return None;
        }
        let (x_p, y_p) = (x >> sub_x, y >> sub_y);
        let (prev_row, prev_col) = (row - (dy << sub_y), col - (dx << sub_x));
        let mi_idx = row * fs.mi_cols + col;
        let prev_mi_idx = prev_row * fs.mi_cols + prev_col;
        let tx_idx = (row >> sub_y) * fs.mi_cols + (col >> sub_x);
        let prev_tx_idx = (prev_row >> sub_y) * fs.mi_cols + (prev_col >> sub_x);
        let mi_size = usize::from(*fs.mi_sizes.get(mi_idx)?);
        let tx_sz = usize::from(*fs.loopfilter_tx_sizes[plane].get(tx_idx)?);
        let prev_tx_sz = usize::from(*fs.loopfilter_tx_sizes[plane].get(prev_tx_idx)?);
        let skip = *fs.skips.get(mi_idx)?;
        let is_intra = *fs.ref_frames0.get(mi_idx)? <= INTRA_FRAME as i8;
        let plane_size = SUBSAMPLED_SIZE[mi_size][sub_x][sub_y];
        if plane_size >= BLOCK_WIDTH.len() {
            return None;
        }
        let (is_block_edge, is_tx_edge) = if pass == 0 {
            (
                x_p.is_multiple_of(BLOCK_WIDTH[plane_size]),
                x_p.is_multiple_of(TX_WIDTH[tx_sz]),
            )
        } else {
            (
                y_p.is_multiple_of(BLOCK_HEIGHT[plane_size]),
                y_p.is_multiple_of(TX_HEIGHT[tx_sz]),
            )
        };
        let apply_filter = is_tx_edge && (is_block_edge || !skip || is_intra);
        if !apply_filter {
            return None;
        }
        let base_size = if pass == 0 {
            TX_WIDTH[prev_tx_sz].min(TX_WIDTH[tx_sz])
        } else {
            TX_HEIGHT[prev_tx_sz].min(TX_HEIGHT[tx_sz])
        };
        let size = base_size.min(if plane == 0 { 16 } else { 8 });
        let mut lvl = filter_level(fs, fh, mi_idx, plane, pass);
        if lvl == 0 {
            lvl = filter_level(fs, fh, prev_mi_idx, plane, pass);
        }
        (lvl != 0).then_some((x_p, y_p, size, lvl))
    }

    #[test]
    fn edge_params_match_the_spec_order() {
        use crate::consts::{BLOCK_SIZES, TX_SIZES_ALL};
        let (mut fs, mut fh, _) = setup(0, 0);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let n = fs.mi_rows * fs.mi_cols;
        for round in 0..24 {
            (fs.subsampling_x, fs.subsampling_y) = [(0, 0), (1, 0), (1, 1)][round % 3];
            for l in fh.loop_filter_level.iter_mut() {
                *l = next(64) as u8;
            }
            fh.delta_lf_multi = next(2) == 1;
            fh.loop_filter_delta_enabled = next(2) == 1;
            for i in 0..n {
                fs.mi_sizes[i] = next(BLOCK_SIZES as u64) as u8;
                fs.skips[i] = next(2) == 1;
                fs.ref_frames0[i] = next(3) as i8 - 1;
                for d in fs.delta_lfs[i].iter_mut() {
                    *d = next(33) as i8 - 16;
                }
                for plane in 0..3 {
                    // Mostly small transforms, so that many positions are edges.
                    let t = if next(2) == 0 {
                        0
                    } else {
                        next(TX_SIZES_ALL as u64)
                    };
                    fs.loopfilter_tx_sizes[plane][i] = t as u8;
                }
            }
            for plane in 0..3 {
                // As in `loop_filter_rows`: chroma positions step by the subsampling.
                let (rs, cs) = if plane == 0 {
                    (1, 1)
                } else {
                    (1 << fs.subsampling_y, 1 << fs.subsampling_x)
                };
                for pass in 0..2 {
                    for row in (0..fs.mi_rows).step_by(rs) {
                        for col in (0..fs.mi_cols).step_by(cs) {
                            let got = edge_params(&fs, &fh, plane, pass, row, col)
                                .map(|e| (e.x, e.y, e.size, e.lvl));
                            let want = edge_params_spec(&fs, &fh, plane, pass, row, col);
                            assert_eq!(got, want, "plane {plane} pass {pass} at {row}, {col}");
                        }
                    }
                }
            }
        }
    }

    /// A 196x132 frame with subsampling `(sx, sy)` and random loop filter parameters, whose
    /// blocks and transform blocks are laid out as the decoder lays them out (the invariant of
    /// the module doc): one random block size per 64x64 region tiling it, and in each block one
    /// random transform size per plane that fits, tiling the block's part of the plane (for
    /// chroma at least one 4x4 chroma unit, as for the chroma of 4xN blocks). The samples are
    /// near flat 8x8 cells, so that most edges filter.
    fn random_layout(
        next: &mut impl FnMut(u64) -> u64,
        (sx, sy): (u8, u8),
    ) -> (FrameState, FrameHeader) {
        use crate::consts::{BLOCK_SIZES, TX_SIZES_ALL};
        use crate::tile::fill_rows;
        let mut seq =
            SequenceHeader::parse(&crate::seq::tests::still_sequence_header(), 0).unwrap();
        (seq.subsampling_x, seq.subsampling_y) = (sx, sy);
        let mut fh = FrameHeader {
            frame_width: 196,
            frame_height: 132,
            upscaled_width: 196,
            mi_cols: 50,
            mi_rows: 34,
            ..FrameHeader::default()
        };
        for l in fh.loop_filter_level.iter_mut() {
            *l = next(64) as u8;
        }
        fh.loop_filter_sharpness = next(8) as u8;
        fh.delta_lf_present = next(2) == 1;
        fh.delta_lf_multi = next(2) == 1;
        fh.loop_filter_delta_enabled = next(2) == 1;
        fh.loop_filter_ref_deltas[INTRA_FRAME] = next(127) as i8 - 63;
        fh.segmentation_enabled = next(2) == 1;
        for s in 0..MAX_SEGMENTS {
            for f in SEG_LVL_ALT_LF_Y_V..SEG_LVL_ALT_LF_Y_V + 4 {
                fh.feature_enabled[s][f] = next(2) == 1;
                // Large negative data takes some levels to 0 (the previous block's level).
                fh.feature_data[s][f] = next(127) as i16 - 63;
            }
        }
        let ledger = Arc::new(Ledger::default());
        ledger.reset(16 << 20, 0).unwrap();
        let mut fs = FrameState::new(&fh, &seq, &ledger).unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        for r0 in (0..mi_rows).step_by(16) {
            for c0 in (0..mi_cols).step_by(16) {
                let b = loop {
                    let b = next(BLOCK_SIZES as u64) as usize;
                    if BLOCK_WIDTH[b] <= 64 && BLOCK_HEIGHT[b] <= 64 {
                        break b;
                    }
                };
                let (bw, bh) = (BLOCK_WIDTH[b] / MI_SIZE, BLOCK_HEIGHT[b] / MI_SIZE);
                let (r1, c1) = ((r0 + 16).min(mi_rows), (c0 + 16).min(mi_cols));
                for r in (r0..r1).step_by(bh) {
                    for c in (c0..c1).step_by(bw) {
                        let (rows, cols) = (bh.min(mi_rows - r), bw.min(mi_cols - c));
                        // Without loop filter deltas every DeltaLF is 0, as in the decoder.
                        let delta = if fh.delta_lf_present {
                            [0i8; 4].map(|_| next(33) as i8 - 16)
                        } else {
                            [0; 4]
                        };
                        fill_rows(&mut fs.mi_sizes, mi_cols, r, c, rows, cols, b as u8);
                        fill_rows(&mut fs.skips, mi_cols, r, c, rows, cols, next(2) == 1);
                        let ref0 = next(3) as i8 - 1;
                        fill_rows(&mut fs.ref_frames0, mi_cols, r, c, rows, cols, ref0);
                        let seg = next(MAX_SEGMENTS as u64) as u8;
                        fill_rows(&mut fs.segment_ids, mi_cols, r, c, rows, cols, seg);
                        fill_rows(&mut fs.delta_lfs, mi_cols, r, c, rows, cols, delta);
                    }
                }
                for plane in 0..3 {
                    let (px, py) = if plane == 0 {
                        (0, 0)
                    } else {
                        (usize::from(sx), usize::from(sy))
                    };
                    // The block's part of the plane, in luma and in plane mode info units.
                    let (uw, uh) = (bw.max(1 << px), bh.max(1 << py));
                    let (pw, ph) = (uw >> px, uh >> py);
                    for r in (r0..r1).step_by(uh) {
                        for c in (c0..c1).step_by(uw) {
                            let t = loop {
                                let t = next(TX_SIZES_ALL as u64) as usize;
                                if TX_WIDTH[t] / MI_SIZE <= pw && TX_HEIGHT[t] / MI_SIZE <= ph {
                                    break t;
                                }
                            };
                            let (tw, th) = (TX_WIDTH[t] / MI_SIZE, TX_HEIGHT[t] / MI_SIZE);
                            for y in (0..ph).step_by(th) {
                                for x in (0..pw).step_by(tw) {
                                    // Clipped to the grid as `transform_block` clips it.
                                    let (pr, pc) = ((r >> py) + y, (c >> px) + x);
                                    let rows = th.min(mi_rows.saturating_sub(pr));
                                    let cols = tw.min(mi_cols.saturating_sub(pc));
                                    let txs = &mut fs.loopfilter_tx_sizes[plane];
                                    fill_rows(txs, mi_cols, pr, pc, rows, cols, t as u8);
                                }
                            }
                        }
                    }
                }
            }
        }
        for p in fs.curr_frame.iter_mut() {
            let mut levels = vec![0u16; p.stride.div_ceil(8)];
            for (y, row) in p.px.chunks_mut(p.stride).enumerate() {
                if y.is_multiple_of(8) {
                    levels.iter_mut().for_each(|l| *l = 100 + next(40) as u16);
                }
                for (x, v) in row.iter_mut().enumerate() {
                    *v = levels[x / 8] + next(4) as u16;
                }
            }
        }
        (fs, fh)
    }

    /// Pass `pass` of plane `plane` over the whole frame as before the segment walk: `edge_params`
    /// at every 4x4 position, merged into runs in the same order.
    fn filter_every_position(
        fs: &FrameState,
        fh: &FrameHeader,
        plane: usize,
        pass: usize,
        v: &mut View<'_>,
    ) {
        let l = Lines::new(fs, fh, plane, pass);
        let (rs, cs) = (1 << l.sub_y, 1 << l.sub_x);
        for strip in (0..fs.mi_rows).step_by(16) {
            let strip_rows = (strip..fs.mi_rows.min(strip + 16)).step_by(rs);
            let lines: Vec<Vec<(usize, usize)>> = if pass == 0 {
                (0..fs.mi_cols)
                    .step_by(cs)
                    .map(|col| strip_rows.clone().map(|row| (row, col)).collect())
                    .collect()
            } else {
                strip_rows
                    .map(|row| (0..fs.mi_cols).step_by(cs).map(|col| (row, col)).collect())
                    .collect()
            };
            for line in lines {
                let mut run = None;
                for (row, col) in line {
                    l.push(&mut run, edge_params(fs, fh, plane, pass, row, col), v);
                }
                l.flush(run, v);
            }
        }
    }

    #[test]
    fn segment_walk_matches_every_position() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let mut changed = 0;
        for round in 0..18 {
            let (fs, fh) = random_layout(&mut next, [(0, 0), (1, 0), (1, 1)][round % 3]);
            assert_transform_blocks_whole(&fs);
            for plane in 0..3 {
                let stride = fs.curr_frame[plane].stride;
                let mut want = fs.curr_frame[plane].px.clone();
                let mut got = want.clone();
                for pass in 0..2 {
                    let mut v = View {
                        px: &mut want,
                        stride,
                        x0: 0,
                        y0: 0,
                    };
                    filter_every_position(&fs, &fh, plane, pass, &mut v);
                    // Odd rounds as the threaded wrapup: row bands, then column strips.
                    let (rows, cols) = (0..fs.mi_rows, 0..fs.mi_cols);
                    let ranges: Vec<(Range<usize>, Range<usize>)> = if round % 2 == 0 {
                        vec![(rows, cols)]
                    } else if pass == 0 {
                        let band = |r: usize| r..(r + 16).min(rows.end);
                        rows.clone()
                            .step_by(16)
                            .map(|r| (band(r), cols.clone()))
                            .collect()
                    } else {
                        let strip = |c: usize| c..(c + 16).min(cols.end);
                        cols.clone()
                            .step_by(16)
                            .map(|c| (rows.clone(), strip(c)))
                            .collect()
                    };
                    let mut v = View {
                        px: &mut got,
                        stride,
                        x0: 0,
                        y0: 0,
                    };
                    for (rows, cols) in ranges {
                        let cancel = AtomicBool::new(false);
                        loop_filter_rows(&fs, &fh, plane, pass, rows, cols, &mut v, &cancel)
                            .unwrap();
                    }
                    assert_eq!(got, want, "round {round} plane {plane} pass {pass}");
                }
                changed += usize::from(got != fs.curr_frame[plane].px);
            }
        }
        // Guards against layouts or samples that filter nothing.
        assert!(changed > 0);
    }
}
