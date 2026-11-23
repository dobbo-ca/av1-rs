//! Deblocking loop filter, spec 7.14. Filters `fs.curr_frame` in place: for each plane all
//! vertical edges of the frame, then all horizontal edges.
//!
//! Only intra frames are decoded, so `RefFrames[ row ][ col ][ 0 ]` is always `INTRA_FRAME` and the
//! `loop_filter_mode_deltas` / `modeType` branch of 7.14.5 never applies.

use crate::consts::{INTRA_FRAME, MAX_LOOP_FILTER, MI_SIZE, SEG_LVL_ALT_LF_Y_V};
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::frame::FrameState;
use crate::header::FrameHeader;
use crate::seq::SequenceHeader;
use crate::tables::{BLOCK_HEIGHT, BLOCK_WIDTH, SUBSAMPLED_SIZE, TX_HEIGHT, TX_WIDTH};
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
    if fh.loop_filter_level[0] == 0 && fh.loop_filter_level[1] == 0 {
        return Ok(());
    }
    for plane in 0..fs.num_planes {
        if plane == 0 || fh.loop_filter_level[1 + plane] != 0 {
            for pass in 0..2 {
                let (row_step, col_step) = if plane == 0 {
                    (1, 1)
                } else {
                    (1 << fs.subsampling_y, 1 << fs.subsampling_x)
                };
                for row in (0..fs.mi_rows).step_by(row_step) {
                    if row.is_multiple_of(16) {
                        check_cancel(cancel)?;
                    }
                    for col in (0..fs.mi_cols).step_by(col_step) {
                        loop_filter_edge(fs, fh, plane, pass, row, col);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Spec 7.14.2 (edge loop filter process).
fn loop_filter_edge(
    fs: &mut FrameState,
    fh: &FrameHeader,
    plane: usize,
    pass: usize,
    row: usize,
    col: usize,
) {
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
        return;
    }
    let x_p = x >> sub_x;
    let y_p = y >> sub_y;
    let prev_row = row - (dy << sub_y);
    let prev_col = col - (dx << sub_x);
    let mi_idx = row * fs.mi_cols + col;
    let prev_mi_idx = prev_row * fs.mi_cols + prev_col;
    let tx_idx = (row >> sub_y) * fs.mi_cols + (col >> sub_x);
    let prev_tx_idx = (prev_row >> sub_y) * fs.mi_cols + (prev_col >> sub_x);
    let (Some(&mi_size), Some(&tx_sz), Some(&prev_tx_sz), Some(&skip), Some(&ref0)) = (
        fs.mi_sizes.get(mi_idx),
        fs.loopfilter_tx_sizes[plane].get(tx_idx),
        fs.loopfilter_tx_sizes[plane].get(prev_tx_idx),
        fs.skips.get(mi_idx),
        fs.ref_frames0.get(mi_idx),
    ) else {
        return;
    };
    let (tx_sz, prev_tx_sz) = (usize::from(tx_sz), usize::from(prev_tx_sz));
    let plane_size = SUBSAMPLED_SIZE[usize::from(mi_size)][sub_x][sub_y];
    if plane_size >= BLOCK_WIDTH.len() {
        return;
    }
    let is_intra = ref0 <= INTRA_FRAME as i8;
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
        return;
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
        return;
    }
    // 7.14.4 limit, blimit, thresh.
    let sharp = i32::from(fh.loop_filter_sharpness);
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
    let stride = fs.curr_frame[plane].stride;
    let step = if pass == 0 { 1 } else { stride };
    let bd = u32::from(fs.bit_depth);
    for i in 0..MI_SIZE {
        let (sx, sy) = (x_p + dy * i, y_p + dx * i);
        let pos = sy * stride + sx;
        sample_filter(
            &mut fs.curr_frame[plane].px,
            pos,
            step,
            plane,
            bd,
            [limit, blimit, thresh],
            filter_size,
        );
    }
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
fn sample_filter(
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
mod tests {
    use super::*;
    use crate::ledger::Ledger;
    use std::sync::Arc;

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
}
