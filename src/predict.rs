//! Intra prediction (spec 7.11.2), palette prediction (7.11.4) and chroma from luma (7.11.5).
//!
//! The functions work on one plane stored as `&mut [u16]` with a row stride, reading the
//! neighbouring samples from the same buffer exactly as the spec reads `CurrFrame`.

#![allow(dead_code)]

use crate::consts::{
    ANGLE_STEP, D67_PRED, DC_PRED, INTRA_FILTER_SCALE_BITS, SMOOTH_H_PRED, SMOOTH_PRED,
    SMOOTH_V_PRED, V_PRED,
};
use crate::tables::{
    DR_INTRA_DERIVATIVE, INTRA_FILTER_TAPS, MODE_TO_ANGLE, SM_WEIGHTS_TX_4X4, SM_WEIGHTS_TX_8X8,
    SM_WEIGHTS_TX_16X16, SM_WEIGHTS_TX_32X32, SM_WEIGHTS_TX_64X64,
};

/// `Intra_Edge_Kernel` (spec 7.11.2.12).
const INTRA_EDGE_KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

/// Index of `AboveRow[ 0 ]` / `LeftCol[ 0 ]` in the edge arrays; room below it holds the
/// entries -1 and -2 (the latter only after upsampling).
const OFF: usize = 16;
/// Edge array length: `OFF`, up to `w + h` = 128 entries, and slack for the interpolation reads.
const EDGE_LEN: usize = OFF + 128 + 16;
/// Largest `numPx` of the intra edge filter (spec 7.11.2.12: sz <= 129), plus slack.
const EDGE_FILTER_MAX: usize = 136;

type Edge = [i32; EDGE_LEN];

/// Which neighbours of the transform block are available (spec 7.11.2 inputs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntraEdges {
    pub have_left: bool,
    pub have_above: bool,
    pub have_above_right: bool,
    pub have_below_left: bool,
}

/// The mode and block parameters of one intra prediction call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntraParams {
    /// `YMode` or `UVMode`. `UV_CFL_PRED` is handled by the caller: it predicts `DC_PRED`
    /// and then calls [`predict_cfl`].
    pub mode: usize,
    /// `AngleDeltaY` or `AngleDeltaUV`, -3..=3.
    pub angle_delta: i32,
    pub log2w: usize,
    pub log2h: usize,
    /// `use_filter_intra`; the caller passes false for chroma planes.
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub enable_intra_edge_filter: bool,
    /// Result of `get_filter_type( plane )` (spec 7.11.2.8), computed by the caller.
    pub smooth_neighbor: bool,
    /// The spec's `maxX` / `maxY` for this plane.
    pub max_x: usize,
    pub max_y: usize,
    pub bit_depth: u32,
}

/// Index into an edge array for the signed position `i` (valid down to -`OFF`).
fn ix(i: i32) -> usize {
    (i + OFF as i32) as usize
}

fn round2(x: i32, n: u32) -> i32 {
    (x + ((1 << n) >> 1)) >> n
}

fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

/// Spec 7.11.2: predicts the `(1 << log2w) x (1 << log2h)` block at `(x, y)` of the plane `px`
/// (row `stride`), reading its neighbours from `px`.
///
/// The caller guarantees `x <= max_x`, `y <= max_y`, that `px` holds rows up to `max_y` and
/// the columns up to the written block, that `have_above` implies `y > 0`, `have_left`
/// implies `x > 0`, `mode` is below 13, `log2w` / `log2h` are in 2..=6 and `bit_depth` is
/// 8, 10 or 12.
pub(crate) fn predict_intra(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    e: IntraEdges,
    p: &IntraParams,
) {
    let w = 1usize << p.log2w;
    let h = 1usize << p.log2h;
    let mid = 1i32 << (p.bit_depth - 1);

    let mut above: Edge = [0; EDGE_LEN];
    let mut left: Edge = [0; EDGE_LEN];

    // AboveRow[ 0..w+h-1 ]
    if !e.have_above && e.have_left {
        let v = i32::from(px[y * stride + x - 1]);
        above[OFF..OFF + w + h].fill(v);
    } else if !e.have_above {
        above[OFF..OFF + w + h].fill(mid - 1);
    } else {
        let reach = if e.have_above_right { 2 * w } else { w };
        let above_limit = p.max_x.min(x + reach - 1);
        let row = (y - 1) * stride;
        for i in 0..w + h {
            above[OFF + i] = i32::from(px[row + above_limit.min(x + i)]);
        }
    }

    // LeftCol[ 0..w+h-1 ]
    if !e.have_left && e.have_above {
        let v = i32::from(px[(y - 1) * stride + x]);
        left[OFF..OFF + w + h].fill(v);
    } else if !e.have_left {
        left[OFF..OFF + w + h].fill(mid + 1);
    } else {
        let reach = if e.have_below_left { 2 * h } else { h };
        let left_limit = p.max_y.min(y + reach - 1);
        for i in 0..w + h {
            left[OFF + i] = i32::from(px[left_limit.min(y + i) * stride + x - 1]);
        }
    }

    // AboveRow[ -1 ] and LeftCol[ -1 ]
    let corner = if e.have_above && e.have_left {
        i32::from(px[(y - 1) * stride + x - 1])
    } else if e.have_above {
        i32::from(px[(y - 1) * stride + x])
    } else if e.have_left {
        i32::from(px[y * stride + x - 1])
    } else {
        mid
    };
    above[OFF - 1] = corner;
    left[OFF - 1] = corner;

    if p.use_filter_intra {
        predict_recursive(px, stride, x, y, w, h, &above, &left, p);
    } else if (V_PRED..=D67_PRED).contains(&p.mode) {
        predict_directional(px, stride, x, y, e, w, h, &mut above, &mut left, p);
    } else if matches!(p.mode, SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED) {
        predict_smooth(px, stride, x, y, &above, &left, p);
    } else if p.mode == DC_PRED {
        predict_dc(px, stride, x, y, e, &above, &left, p);
    } else {
        predict_paeth(px, stride, x, y, w, h, &above, &left);
    }
}

/// Spec 7.11.2.2 (basic intra prediction, PAETH_PRED).
fn predict_paeth(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    left: &Edge,
) {
    let top_left = above[OFF - 1];
    for i in 0..h {
        for j in 0..w {
            let a = above[OFF + j];
            let l = left[OFF + i];
            let base = a + l - top_left;
            let p_left = (base - l).abs();
            let p_top = (base - a).abs();
            let p_top_left = (base - top_left).abs();
            let v = if p_left <= p_top && p_left <= p_top_left {
                l
            } else if p_top <= p_top_left {
                a
            } else {
                top_left
            };
            px[(y + i) * stride + x + j] = v as u16;
        }
    }
}

/// Spec 7.11.2.3 (recursive intra prediction, filter intra). Each 4x2 cell is predicted from
/// 7 neighbours, taken from the edges or from cells written earlier in the block.
fn predict_recursive(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    left: &Edge,
    p: &IntraParams,
) {
    let max = (1i32 << p.bit_depth) - 1;
    for i2 in 0..h >> 1 {
        for j4 in 0..w >> 2 {
            let mut nb = [0i32; 7];
            for (i, n) in nb.iter_mut().enumerate() {
                *n = if i < 5 {
                    if i2 == 0 {
                        above[OFF + (j4 << 2) + i - 1]
                    } else if j4 == 0 && i == 0 {
                        left[OFF + (i2 << 1) - 1]
                    } else {
                        i32::from(px[(y + (i2 << 1) - 1) * stride + x + (j4 << 2) + i - 1])
                    }
                } else if j4 == 0 {
                    left[OFF + (i2 << 1) + i - 5]
                } else {
                    i32::from(px[(y + (i2 << 1) + i - 5) * stride + x + (j4 << 2) - 1])
                };
            }
            for i1 in 0..2 {
                for j1 in 0..4 {
                    let taps = &INTRA_FILTER_TAPS[p.filter_intra_mode][(i1 << 2) + j1];
                    let pr: i32 = taps.iter().zip(&nb).map(|(&t, &n)| i32::from(t) * n).sum();
                    let v = round2_signed(pr, INTRA_FILTER_SCALE_BITS as u32).clamp(0, max);
                    px[(y + (i2 << 1) + i1) * stride + x + (j4 << 2) + j1] = v as u16;
                }
            }
        }
    }
}

/// Spec 7.11.2.7: three tap filter for the top-left corner sample.
fn filter_corner(above: &Edge, left: &Edge) -> i32 {
    round2(left[OFF] * 5 + above[OFF - 1] * 6 + above[OFF] * 5, 4)
}

/// Spec 7.11.2.9: edge filter strength (0..=3) from the block size, the filter type and the
/// angle difference `delta`.
fn intra_edge_filter_strength(w: usize, h: usize, smooth: bool, delta: i32) -> usize {
    let d = delta.abs();
    let blk_wh = w + h;
    if !smooth {
        if blk_wh <= 8 {
            usize::from(d >= 56)
        } else if blk_wh <= 16 {
            usize::from(d >= 40)
        } else if blk_wh <= 24 {
            if d >= 32 {
                3
            } else if d >= 16 {
                2
            } else {
                usize::from(d >= 8)
            }
        } else if blk_wh <= 32 {
            if d >= 32 {
                3
            } else if d >= 4 {
                2
            } else {
                1
            }
        } else {
            3
        }
    } else if blk_wh <= 8 {
        if d >= 64 { 2 } else { usize::from(d >= 40) }
    } else if blk_wh <= 16 {
        if d >= 48 { 2 } else { usize::from(d >= 20) }
    } else if blk_wh <= 24 {
        if d >= 4 { 3 } else { 0 }
    } else {
        3
    }
}

/// Spec 7.11.2.10: whether the edge is upsampled.
fn use_upsample(w: usize, h: usize, smooth: bool, delta: i32) -> bool {
    let d = delta.abs();
    let blk_wh = w + h;
    if !(1..40).contains(&d) {
        false
    } else if !smooth {
        blk_wh <= 16
    } else {
        blk_wh <= 8
    }
}

/// Spec 7.11.2.12: smooths `sz` entries of an edge (starting at index -1) with the kernel
/// chosen by `strength`. `sz` is at most 129.
fn filter_edge(buf: &mut Edge, sz: usize, strength: usize) {
    if strength == 0 || sz == 0 {
        return;
    }
    let mut edge = [0i32; EDGE_FILTER_MAX];
    for i in 0..sz {
        edge[i] = buf[OFF + i - 1];
    }
    let kernel = &INTRA_EDGE_KERNEL[strength - 1];
    for i in 1..sz {
        let mut s = 0;
        for (j, &k) in kernel.iter().enumerate() {
            let idx = (i + j).saturating_sub(2).min(sz - 1);
            s += k * edge[idx];
        }
        buf[OFF + i - 1] = (s + 8) >> 4;
    }
}

/// Spec 7.11.2.11: doubles the resolution of an edge. Entries -1..num_px-1 go in, entries
/// -2..2*num_px-2 come out. `num_px` is at most 16.
fn upsample_edge(buf: &mut Edge, num_px: usize, max: i32) {
    let mut dup = [0i32; 24];
    dup[0] = buf[OFF - 1];
    for i in 0..num_px + 1 {
        dup[i + 1] = buf[OFF + i - 1];
    }
    dup[num_px + 2] = buf[OFF + num_px - 1];
    buf[OFF - 2] = dup[0];
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        buf[OFF + 2 * i - 1] = round2(s, 4).clamp(0, max);
        buf[OFF + 2 * i] = dup[i + 2];
    }
}

/// Interpolates `edge[ base ]` and `edge[ base + 1 ]` with the 5 bit weight taken from `idx`.
fn interp(edge: &Edge, idx: i32, base: i32, up: u32) -> u16 {
    let shift = ((idx << up) >> 1) & 0x1f;
    let v = edge[ix(base)] * (32 - shift) + edge[ix(base + 1)] * shift;
    round2(v, 5) as u16
}

/// Spec 7.11.2.4 (directional intra prediction), including the edge filter and upsampling.
fn predict_directional(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    e: IntraEdges,
    w: usize,
    h: usize,
    above: &mut Edge,
    left: &mut Edge,
    p: &IntraParams,
) {
    let max = (1i32 << p.bit_depth) - 1;
    let angle_delta = p.angle_delta.clamp(-3, 3);
    let p_angle = MODE_TO_ANGLE[p.mode] + angle_delta * ANGLE_STEP as i32;
    let mut upsample_above = 0u32;
    let mut upsample_left = 0u32;

    if p.enable_intra_edge_filter {
        let smooth = p.smooth_neighbor;
        if p_angle != 90 && p_angle != 180 {
            if (91..180).contains(&p_angle) && w + h >= 24 {
                let c = filter_corner(above, left);
                above[OFF - 1] = c;
                left[OFF - 1] = c;
            }
            if e.have_above {
                let strength = intra_edge_filter_strength(w, h, smooth, p_angle - 90);
                let num_px = w.min(p.max_x - x + 1) + if p_angle < 90 { h } else { 0 } + 1;
                filter_edge(above, num_px, strength);
            }
            if e.have_left {
                let strength = intra_edge_filter_strength(w, h, smooth, p_angle - 180);
                let num_px = h.min(p.max_y - y + 1) + if p_angle > 180 { w } else { 0 } + 1;
                filter_edge(left, num_px, strength);
            }
        }
        if use_upsample(w, h, smooth, p_angle - 90) {
            upsample_above = 1;
            let num_px = w + if p_angle < 90 { h } else { 0 };
            upsample_edge(above, num_px, max);
        }
        if use_upsample(w, h, smooth, p_angle - 180) {
            upsample_left = 1;
            let num_px = h + if p_angle > 180 { w } else { 0 };
            upsample_edge(left, num_px, max);
        }
    }

    let dx = if p_angle < 90 {
        i32::from(DR_INTRA_DERIVATIVE[p_angle as usize])
    } else if (91..180).contains(&p_angle) {
        i32::from(DR_INTRA_DERIVATIVE[(180 - p_angle) as usize])
    } else {
        0
    };
    let dy = if (91..180).contains(&p_angle) {
        i32::from(DR_INTRA_DERIVATIVE[(p_angle - 90) as usize])
    } else if p_angle > 180 {
        i32::from(DR_INTRA_DERIVATIVE[(270 - p_angle) as usize])
    } else {
        0
    };

    for i in 0..h {
        for j in 0..w {
            let (ii, jj) = (i as i32, j as i32);
            let v = if p_angle < 90 {
                let idx = (ii + 1) * dx;
                let base = (idx >> (6 - upsample_above)) + (jj << upsample_above);
                let max_base_x = ((w + h - 1) as i32) << upsample_above;
                if base < max_base_x {
                    interp(above, idx, base, upsample_above)
                } else {
                    above[ix(max_base_x)] as u16
                }
            } else if (91..180).contains(&p_angle) {
                let idx = (jj << 6) - (ii + 1) * dx;
                let base = idx >> (6 - upsample_above);
                if base >= -(1 << upsample_above) {
                    interp(above, idx, base, upsample_above)
                } else {
                    let idx = (ii << 6) - (jj + 1) * dy;
                    let base = idx >> (6 - upsample_left);
                    interp(left, idx, base, upsample_left)
                }
            } else if p_angle > 180 {
                let idx = (jj + 1) * dy;
                let base = (idx >> (6 - upsample_left)) + (ii << upsample_left);
                interp(left, idx, base, upsample_left)
            } else if p_angle == 90 {
                above[OFF + j] as u16
            } else {
                left[OFF + i] as u16
            };
            px[(y + i) * stride + x + j] = v;
        }
    }
}

/// Smooth weight table for a block side of `1 << log2` samples (log2 in 2..=6).
fn sm_weights(log2: usize) -> &'static [u8] {
    match log2 {
        2 => &SM_WEIGHTS_TX_4X4,
        3 => &SM_WEIGHTS_TX_8X8,
        4 => &SM_WEIGHTS_TX_16X16,
        5 => &SM_WEIGHTS_TX_32X32,
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

/// Spec 7.11.2.6 (SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED).
fn predict_smooth(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    above: &Edge,
    left: &Edge,
    p: &IntraParams,
) {
    let w = 1usize << p.log2w;
    let h = 1usize << p.log2h;
    let wx = sm_weights(p.log2w);
    let wy = sm_weights(p.log2h);
    let bottom_left = left[OFF + h - 1];
    let top_right = above[OFF + w - 1];
    for i in 0..h {
        for j in 0..w {
            let a = above[OFF + j];
            let l = left[OFF + i];
            let v = match p.mode {
                SMOOTH_PRED => {
                    let (sy, sx) = (i32::from(wy[i]), i32::from(wx[j]));
                    let s = sy * a + (256 - sy) * bottom_left + sx * l + (256 - sx) * top_right;
                    round2(s, 9)
                }
                SMOOTH_V_PRED => {
                    let sy = i32::from(wy[i]);
                    round2(sy * a + (256 - sy) * bottom_left, 8)
                }
                _ => {
                    let sx = i32::from(wx[j]);
                    round2(sx * l + (256 - sx) * top_right, 8)
                }
            };
            px[(y + i) * stride + x + j] = v as u16;
        }
    }
}

/// Spec 7.11.2.5 (DC_PRED).
fn predict_dc(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    e: IntraEdges,
    above: &Edge,
    left: &Edge,
    p: &IntraParams,
) {
    let w = 1usize << p.log2w;
    let h = 1usize << p.log2h;
    let sum_left: i32 = left[OFF..OFF + h].iter().sum();
    let sum_above: i32 = above[OFF..OFF + w].iter().sum();
    let v = match (e.have_left, e.have_above) {
        (true, true) => (sum_left + sum_above + ((w + h) >> 1) as i32) / (w + h) as i32,
        (true, false) => (sum_left + (h >> 1) as i32) >> p.log2h,
        (false, true) => (sum_above + (w >> 1) as i32) >> p.log2w,
        (false, false) => 1 << (p.bit_depth - 1),
    };
    for i in 0..h {
        px[(y + i) * stride + x..][..w].fill(v as u16);
    }
}

/// Spec 7.11.4: `px[ (y + i) * stride + x + j ] = palette[ map[ i * map_stride + j ] ]` for
/// the `w x h` block. The caller passes `map` starting at the block's first map entry and
/// guarantees every map value is below `palette.len()`.
pub(crate) fn predict_palette(
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
    for i in 0..h {
        let dst = &mut px[(y + i) * stride + x..][..w];
        let src = &map[i * map_stride..][..w];
        for (d, &m) in dst.iter_mut().zip(src) {
            *d = palette[usize::from(m)];
        }
    }
}

/// `Subpel_Filters[ BILINEAR ]` (spec 7.11.3.4): the only filter an intra block copy block uses.
const SUBPEL_FILTERS_BILINEAR: [[i32; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

/// Largest block `predict_intrabc` handles (the largest plane block of a block).
const IBC_MAX: usize = 128;

/// Spec 7.11.3.4 block inter prediction process for an intra block copy block (`refIdx` -1:
/// the reference is the current frame, `px` itself, before any loop filter), followed by the
/// final step of 7.11.3.1: the prediction is not compound, so
/// `CurrFrame = Clip1( pred )` (`InterPostRound` is 0 for the rounding of 7.11.3.2).
///
/// The `w x h` block is written at `(x, y)` of the plane `px` (row `stride`). `start_x` and
/// `start_y` are the outputs of the motion vector scaling process (7.11.3.3) in 1/1024 sample,
/// with `xStep` and `yStep` 1024 (an intra block copy reference is never scaled). `last_x` and
/// `last_y` are the spec's `lastX` and `lastY`: reference positions are clamped to them.
/// `round0` and `round1` are `InterRound0` and `InterRound1`.
///
/// Nothing is written when the block does not fit in `px`, when `w` or `h` exceeds 128 or when
/// `last_x` / `last_y` are outside `px`. The reads are in `px` too (the clamp), and all of them
/// are done before the first write, so the block may overlap its own reference.
pub(crate) fn predict_intrabc(
    px: &mut [u16],
    stride: usize,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    (start_x, start_y): (i64, i64),
    (last_x, last_y): (usize, usize),
    (round0, round1): (u32, u32),
    bit_depth: u32,
) {
    if w == 0
        || h == 0
        || w > IBC_MAX
        || h > IBC_MAX
        || x + w > stride
        || last_x >= stride
        || (y + h) * stride > px.len()
        || (last_y + 1) * stride > px.len()
    {
        return;
    }
    // intermediateHeight = ( ( ( h - 1 ) * yStep + ( 1 << SCALE_SUBPEL_BITS ) - 1 ) >>
    // SCALE_SUBPEL_BITS ) + 8, with yStep 1024.
    let inter_h = h + 7;
    let mut intermediate = [0i32; (IBC_MAX + 7) * IBC_MAX];
    for r in 0..inter_h {
        let ref_y = ((start_y >> 10) + r as i64 - 3).clamp(0, last_y as i64) as usize;
        let row = &px[ref_y * stride..(ref_y + 1) * stride];
        for c in 0..w {
            let p = start_x + 1024 * c as i64;
            let filter = &SUBPEL_FILTERS_BILINEAR[((p >> 6) & 15) as usize];
            let mut s = 0i32;
            for (t, &f) in filter.iter().enumerate() {
                let ref_x = ((p >> 10) + t as i64 - 3).clamp(0, last_x as i64) as usize;
                s += f * i32::from(row[ref_x]);
            }
            intermediate[r * w + c] = round2(s, round0);
        }
    }
    let max = (1i32 << bit_depth) - 1;
    for r in 0..h {
        let p = (start_y & 1023) + 1024 * r as i64;
        let filter = &SUBPEL_FILTERS_BILINEAR[((p >> 6) & 15) as usize];
        let base = (p >> 10) as usize;
        let dst = &mut px[(y + r) * stride + x..][..w];
        for (c, d) in dst.iter_mut().enumerate() {
            let mut s = 0i32;
            for (t, &f) in filter.iter().enumerate() {
                s += f * intermediate[(base + t) * w + c];
            }
            *d = round2(s, round1).clamp(0, max) as u16;
        }
    }
}

/// Spec 7.11.5, run after the DC prediction is in place. `(start_x, start_y)` is the chroma
/// block position, `max_luma_w` / `max_luma_h` the spec's `MaxLumaW` / `MaxLumaH`. The caller
/// guarantees the luma samples the spec reads (up to `max_luma_w` x `max_luma_h`) are in `luma`.
pub(crate) fn predict_cfl(
    chroma: &mut [u16],
    c_stride: usize,
    luma: &[u16],
    l_stride: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    max_luma_w: usize,
    max_luma_h: usize,
    alpha: i32,
    bit_depth: u32,
) {
    let sub_x = usize::from(subsampling_x);
    let sub_y = usize::from(subsampling_y);
    // Subsampled luma sample (i, j) with 3 fractional bits.
    let l = |i: usize, j: usize| -> i32 {
        let luma_y = ((start_y + i) << sub_y).min(max_luma_h.saturating_sub(1 << sub_y));
        let luma_x = ((start_x + j) << sub_x).min(max_luma_w.saturating_sub(1 << sub_x));
        let mut t = 0i32;
        for dy in 0..=sub_y {
            for dx in 0..=sub_x {
                t += i32::from(luma[(luma_y + dy) * l_stride + luma_x + dx]);
            }
        }
        t << (3 - sub_x - sub_y)
    };

    let mut luma_avg = 0i32;
    for i in 0..tx_h {
        for j in 0..tx_w {
            luma_avg += l(i, j);
        }
    }
    luma_avg = round2(luma_avg, tx_w.trailing_zeros() + tx_h.trailing_zeros());

    let max = (1i32 << bit_depth) - 1;
    for i in 0..tx_h {
        for j in 0..tx_w {
            let dst = &mut chroma[(start_y + i) * c_stride + start_x + j];
            let scaled_luma = round2_signed(alpha * (l(i, j) - luma_avg), 6);
            *dst = (i32::from(*dst) + scaled_luma).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::H_PRED;

    const NONE: IntraEdges = IntraEdges {
        have_left: false,
        have_above: false,
        have_above_right: false,
        have_below_left: false,
    };

    fn params(mode: usize, log2: usize) -> IntraParams {
        IntraParams {
            mode,
            angle_delta: 0,
            log2w: log2,
            log2h: log2,
            use_filter_intra: false,
            filter_intra_mode: 0,
            enable_intra_edge_filter: false,
            smooth_neighbor: false,
            max_x: 7,
            max_y: 7,
            bit_depth: 8,
        }
    }

    #[test]
    fn dc_without_neighbours_is_mid_grey() {
        for bd in [8u32, 10, 12] {
            let mut px = vec![0u16; 64];
            let mut p = params(DC_PRED, 2);
            p.bit_depth = bd;
            predict_intra(&mut px, 8, 0, 0, NONE, &p);
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(px[i * 8 + j], 1 << (bd - 1));
                }
            }
            assert_eq!(px[4], 0);
        }
    }

    #[test]
    fn dc_averages_both_edges() {
        let mut px = vec![0u16; 64];
        px[..8].fill(10); // row above the block at y = 1
        for r in 1..5 {
            px[r * 8] = 30; // column to the left
        }
        let e = IntraEdges {
            have_left: true,
            have_above: true,
            ..NONE
        };
        predict_intra(&mut px, 8, 1, 1, e, &params(DC_PRED, 2));
        assert_eq!(px[8 + 1], 20);
        assert_eq!(px[4 * 8 + 4], 20);
    }

    #[test]
    fn vertical_and_horizontal_copy_the_edge() {
        let mut px = vec![0u16; 64];
        for (k, v) in px[..8].iter_mut().enumerate() {
            *v = 10 * (k as u16 + 1);
        }
        let e = IntraEdges {
            have_above: true,
            ..NONE
        };
        predict_intra(&mut px, 8, 0, 1, e, &params(V_PRED, 2));
        for i in 1..5 {
            assert_eq!(&px[i * 8..i * 8 + 4], &[10, 20, 30, 40]);
        }

        let mut px = vec![0u16; 64];
        for r in 0..8 {
            px[r * 8] = 5 * (r as u16 + 1);
        }
        let e = IntraEdges {
            have_left: true,
            ..NONE
        };
        predict_intra(&mut px, 8, 1, 0, e, &params(H_PRED, 2));
        for i in 0..4 {
            assert_eq!(&px[i * 8 + 1..i * 8 + 5], &[5 * (i as u16 + 1); 4]);
        }
    }

    #[test]
    fn paeth_hand_cases() {
        let e = IntraEdges {
            have_left: true,
            have_above: true,
            ..NONE
        };
        // top-left 30, above 20, left 40: base = 30, so the top-left sample is closest.
        let mut px = vec![0u16; 64];
        px[0] = 30;
        px[1..5].fill(20);
        for r in 1..5 {
            px[r * 8] = 40;
        }
        predict_intra(&mut px, 8, 1, 1, e, &params(12, 2));
        assert_eq!(px[8 + 1], 30);
        // top-left 30, above 30, left 22: base = 22 equals the left sample.
        let mut px = vec![0u16; 64];
        px[0] = 30;
        px[1..5].fill(30);
        for r in 1..5 {
            px[r * 8] = 22;
        }
        predict_intra(&mut px, 8, 1, 1, e, &params(12, 2));
        assert_eq!(px[8 + 1], 22);
        assert_eq!(px[4 * 8 + 4], 22);
    }

    #[test]
    fn d45_reads_the_above_row_diagonally() {
        // pAngle 45 gives dx = 64: pred[ i ][ j ] = AboveRow[ i + j + 1 ], clamped at w + h - 1.
        let mut px = vec![0u16; 64];
        for (k, v) in px[..8].iter_mut().enumerate() {
            *v = 10 * (k as u16 + 1);
        }
        let e = IntraEdges {
            have_above: true,
            have_above_right: true,
            ..NONE
        };
        predict_intra(&mut px, 8, 0, 1, e, &params(3, 2));
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(px[(1 + i) * 8 + j], 10 * (i + j + 2) as u16);
            }
        }
    }

    #[test]
    fn smooth_of_flat_edges_is_flat() {
        let mut px = vec![0u16; 64];
        px[..8].fill(77);
        for r in 0..8 {
            px[r * 8] = 77;
        }
        let e = IntraEdges {
            have_left: true,
            have_above: true,
            ..NONE
        };
        for mode in [SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
            predict_intra(&mut px, 8, 1, 1, e, &params(mode, 2));
            assert_eq!(px[8 + 1], 77);
            assert_eq!(px[4 * 8 + 4], 77);
        }
    }

    #[test]
    fn all_modes_run_on_all_sizes() {
        let stride = 256;
        let mut px = vec![0u16; stride * 256];
        for (k, v) in px.iter_mut().enumerate() {
            *v = (k * 7 % 251) as u16;
        }
        let e = IntraEdges {
            have_left: true,
            have_above: true,
            have_above_right: true,
            have_below_left: true,
        };
        for mode in 0..13 {
            for delta in -3..=3 {
                for log2w in 2usize..=6 {
                    for log2h in 2..=6 {
                        if log2w.abs_diff(log2h) > 2 {
                            continue;
                        }
                        let mut p = params(mode, log2w);
                        p.log2h = log2h;
                        p.angle_delta = delta;
                        p.enable_intra_edge_filter = true;
                        p.smooth_neighbor = delta > 0;
                        p.max_x = 255;
                        p.max_y = 255;
                        predict_intra(&mut px, stride, 64, 64, e, &p);
                        p.use_filter_intra = true;
                        p.filter_intra_mode = (delta + 3) as usize % 5;
                        predict_intra(&mut px, stride, 64, 64, e, &p);
                    }
                }
            }
        }
    }

    #[test]
    fn palette_maps_indices_to_colours() {
        let mut px = vec![0u16; 6 * 4];
        let palette = [100u16, 200, 300];
        let map = [0u8, 1, 2, 9, 2, 1, 0, 9];
        predict_palette(&mut px, 6, 1, 1, 3, 2, &palette, &map, 4);
        assert_eq!(&px[6 + 1..6 + 4], &[100, 200, 300]);
        assert_eq!(&px[12 + 1..12 + 4], &[300, 200, 100]);
        assert_eq!(px[0], 0);
        assert_eq!(px[6 + 4], 0);
    }

    #[test]
    fn cfl_with_zero_alpha_keeps_dc() {
        let luma: Vec<u16> = (0..64).map(|v| v * 3).collect();
        let mut chroma = vec![50u16; 16];
        predict_cfl(&mut chroma, 4, &luma, 8, 0, 0, 4, 4, true, true, 8, 8, 0, 8);
        assert!(chroma.iter().all(|&c| c == 50));
    }

    #[test]
    fn cfl_adds_scaled_luma_ac() {
        // 4:4:4, luma 0 or 100 alternating by column: avg 50, AC = +-50 << 3 = +-400.
        let mut luma = vec![0u16; 16];
        for (k, v) in luma.iter_mut().enumerate() {
            *v = if k.is_multiple_of(2) { 0 } else { 100 };
        }
        let mut chroma = vec![128u16; 16];
        // alpha 4: Round2Signed( 4 * 400, 6 ) = 25.
        predict_cfl(
            &mut chroma,
            4,
            &luma,
            4,
            0,
            0,
            4,
            4,
            false,
            false,
            4,
            4,
            4,
            8,
        );
        assert_eq!(&chroma[..4], &[103, 153, 103, 153]);
    }

    #[test]
    fn intrabc_integer_vector_is_a_copy() {
        let mut px: Vec<u16> = (0..256u16).collect();
        let src = px.clone();
        // Block at (8, 8) from (4, 0): mv = (-64, -32) in 1/8 sample, so
        // start = ( ( (8 << 4) + 2 * mv ) * 64 + 32 ) per axis.
        predict_intrabc(
            &mut px,
            16,
            (8, 8),
            (4, 4),
            (4128, 32),
            (15, 15),
            (3, 11),
            8,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(px[(8 + i) * 16 + 8 + j], src[i * 16 + 4 + j]);
            }
        }
    }

    #[test]
    fn intrabc_half_sample_vector_averages_two_samples() {
        let mut px = vec![0u16; 32];
        for row in px.chunks_mut(8) {
            row[..4].copy_from_slice(&[10, 20, 31, 40]);
        }
        // Row 1, x = 0.5 (filter phase 8): (10 + 20 + 1) >> 1 and (20 + 31 + 1) >> 1.
        predict_intrabc(&mut px, 8, (4, 1), (2, 1), (544, 1056), (7, 3), (3, 11), 8);
        assert_eq!(&px[12..14], &[15, 26]);
    }
}
