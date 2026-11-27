//! Intra block copy (spec 5.11.7 `use_intrabc` branch of `intra_frame_mode_info( )`): the motion
//! vector prediction of `find_mv_stack( 0 )` (7.10.2), `assign_mv( 0 )` (5.11.26) with `read_mv`
//! and `read_mv_component` (5.11.31, 5.11.32), and `compute_prediction( )` (5.11.33) with the inter
//! prediction process (7.11.3) for a block predicted from the current frame.
//!
//! What an intra frame lets this module leave out of the spec text:
//! - `RefFrame[ 0 ]` is `INTRA_FRAME` and `RefFrame[ 1 ]` is `NONE`, so `GlobalMvs[ 0 ]` is 0 (no
//!   global motion), the candidate scans compare against `INTRA_FRAME`, and no candidate has a
//!   second motion vector.
//! - `use_ref_frame_mvs` is 0 (no temporal scan), `force_integer_mv` is 1 and
//!   `allow_high_precision_mv` is 0 (the lower precision process rounds to whole samples).
//! - The contexts of `find_mv_stack` (`NewMvContext`, `RefMvContext`, `ZeroMvContext`,
//!   `DrlCtxStack`) feed syntax elements an intrabc block does not read, so they are not computed.
//! - The extra search process finds no candidate (it needs `RefFrames > INTRA_FRAME`) and only
//!   leaves zero vectors in the first two stack slots, which an empty `MvStack` already holds.
//! - `YModes` of every intrabc block is `DC_PRED`, so `NewMvCount` stays 0.
//! - `Mvs` and the neighbours' `IsInters` are only set for intrabc blocks, so
//!   `add_ref_mv_candidate` accepts exactly the intrabc blocks that were decoded already: an
//!   undecoded position has `IsInters` 0, which is the spec's "has been written" test.

use crate::consts::*;
use crate::error::Error;
use crate::predict::predict_intrabc;
use crate::tables::{
    BLOCK_HEIGHT, BLOCK_WIDTH, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, SUBSAMPLED_SIZE,
};
use crate::tile::TileDecoder;

/// `RefStackMv[ idx ][ 0 ]` and `WeightStack[ idx ]` of `find_mv_stack`, with `NumMvFound`.
#[derive(Default)]
struct MvStack {
    mv: [[i32; 2]; MAX_REF_MV_STACK_SIZE],
    weight: [u32; MAX_REF_MV_STACK_SIZE],
    num: usize,
}

/// `Clip3( x, y, z )` (spec 4.7). Unlike `Ord::clamp` it does not panic when `x > y`.
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x { x } else { z.min(y) }
}

/// `Round2Signed( x, n )` (spec 4.7).
fn round2_signed(x: i64, n: usize) -> i64 {
    let r = (x.abs() + ((1i64 << n) >> 1)) >> n;
    if x >= 0 { r } else { -r }
}

/// Spec 7.10.2.10 lower precision process, for `force_integer_mv` equal to 1 (an intra frame).
fn lower_mv_precision(cand_mv: &mut [i32; 2]) {
    for c in cand_mv.iter_mut() {
        let a_int = ((c.unsigned_abs() + 3) >> 3) as i32;
        *c = if *c > 0 { a_int << 3 } else { -(a_int << 3) };
    }
}

/// Spec 7.10.2.11 sorting process on `RefStackMv` / `WeightStack` entries `start..end`.
fn sort_stack(st: &mut MvStack, start: usize, mut end: usize) {
    while end > start {
        let mut new_end = start;
        for idx in (start + 1)..end {
            if st.weight[idx - 1] < st.weight[idx] {
                st.weight.swap(idx - 1, idx);
                st.mv.swap(idx - 1, idx);
                new_end = idx;
            }
        }
        end = new_end;
    }
}

impl TileDecoder<'_> {
    /// The `use_intrabc` part of spec 5.11.7: `find_mv_stack( 0 )` then `assign_mv( 0 )`. Sets
    /// `Mv[ 0 ]` (`self.mv`).
    pub(crate) fn intrabc_mode_info(&mut self) {
        let st = self.find_mv_stack();
        self.assign_mv(&st);
    }

    /// Spec 7.10.2 `find_mv_stack( 0 )` for `RefFrame[ 0 ]` equal to `INTRA_FRAME`.
    fn find_mv_stack(&self) -> MvStack {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size];
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size];
        let mut st = MvStack::default();
        self.scan_row(-1, &mut st);
        self.scan_col(-1, &mut st);
        if bw4.max(bh4) <= 16 {
            self.scan_point(-1, bw4 as isize, &mut st);
        }
        let num_nearest = st.num;
        for w in &mut st.weight[..num_nearest] {
            *w += REF_CAT_LEVEL as u32;
        }
        self.scan_point(-1, -1, &mut st);
        self.scan_row(-3, &mut st);
        self.scan_col(-3, &mut st);
        if bh4 > 1 {
            self.scan_row(-5, &mut st);
        }
        if bw4 > 1 {
            self.scan_col(-5, &mut st);
        }
        sort_stack(&mut st, 0, num_nearest);
        let num = st.num;
        sort_stack(&mut st, num_nearest, num);
        // Context and clamping process (7.10.2.14), the clamping of the stack entries.
        let bh = BLOCK_HEIGHT[self.mi_size] as i32;
        let bw = BLOCK_WIDTH[self.mi_size] as i32;
        for idx in 0..st.num {
            st.mv[idx][0] = self.clamp_mv_row(st.mv[idx][0], MV_BORDER as i32 + bh * 8);
            st.mv[idx][1] = self.clamp_mv_col(st.mv[idx][1], MV_BORDER as i32 + bw * 8);
        }
        st
    }

    /// `clamp_mv_row( mvec, border )` (spec 5.11.53).
    fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size] as i32;
        let (mi_row, mi_rows) = (self.mi_row as i32, self.fs.mi_rows as i32);
        let mb_to_top_edge = -((mi_row * MI_SIZE as i32) * 8);
        let mb_to_bottom_edge = ((mi_rows - bh4 - mi_row) * MI_SIZE as i32) * 8;
        clip3(mb_to_top_edge - border, mb_to_bottom_edge + border, mvec)
    }

    /// `clamp_mv_col( mvec, border )` (spec 5.11.54).
    fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size] as i32;
        let (mi_col, mi_cols) = (self.mi_col as i32, self.fs.mi_cols as i32);
        let mb_to_left_edge = -((mi_col * MI_SIZE as i32) * 8);
        let mb_to_right_edge = ((mi_cols - bw4 - mi_col) * MI_SIZE as i32) * 8;
        clip3(mb_to_left_edge - border, mb_to_right_edge + border, mvec)
    }

    /// Spec 7.10.2.2 scan row process (`isCompound` is 0).
    fn scan_row(&self, delta_row: isize, st: &mut MvStack) {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size];
        let end4 = bw4.min(self.fs.mi_cols.saturating_sub(self.mi_col)).min(16);
        let mut delta_row = delta_row;
        let mut delta_col = 0isize;
        let use_step16 = bw4 >= 16;
        if delta_row.abs() > 1 {
            delta_row += (self.mi_row & 1) as isize;
            delta_col = 1 - (self.mi_col & 1) as isize;
        }
        let mut i = 0;
        while i < end4 {
            let mv_row = self.mi_row as isize + delta_row;
            let mv_col = self.mi_col as isize + delta_col + i as isize;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let idx = self.mi_idx(mv_row as usize, mv_col as usize);
            let mut len = bw4.min(NUM_4X4_BLOCKS_WIDE[usize::from(self.fs.mi_sizes[idx])]);
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(idx, (len * 2) as u32, st);
            i += len;
        }
    }

    /// Spec 7.10.2.3 scan col process (`isCompound` is 0).
    fn scan_col(&self, delta_col: isize, st: &mut MvStack) {
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size];
        let end4 = bh4.min(self.fs.mi_rows.saturating_sub(self.mi_row)).min(16);
        let mut delta_col = delta_col;
        let mut delta_row = 0isize;
        let use_step16 = bh4 >= 16;
        if delta_col.abs() > 1 {
            delta_row = 1 - (self.mi_row & 1) as isize;
            delta_col += (self.mi_col & 1) as isize;
        }
        let mut i = 0;
        while i < end4 {
            let mv_row = self.mi_row as isize + delta_row + i as isize;
            let mv_col = self.mi_col as isize + delta_col;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let idx = self.mi_idx(mv_row as usize, mv_col as usize);
            let mut len = bh4.min(NUM_4X4_BLOCKS_HIGH[usize::from(self.fs.mi_sizes[idx])]);
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(idx, (len * 2) as u32, st);
            i += len;
        }
    }

    /// Spec 7.10.2.4 scan point process (`isCompound` is 0). A position that was not decoded yet
    /// has `IsInters` 0 and is ignored by `add_ref_mv_candidate`.
    fn scan_point(&self, delta_row: isize, delta_col: isize, st: &mut MvStack) {
        let mv_row = self.mi_row as isize + delta_row;
        let mv_col = self.mi_col as isize + delta_col;
        if self.is_inside(mv_row, mv_col) {
            let idx = self.mi_idx(mv_row as usize, mv_col as usize);
            self.add_ref_mv_candidate(idx, 4, st);
        }
    }

    /// Spec 7.10.2.7 add reference motion vector process and 7.10.2.8 search stack process for
    /// the candidate at mode info index `idx`.
    fn add_ref_mv_candidate(&self, idx: usize, weight: u32, st: &mut MvStack) {
        if !self.fs.is_inters[idx] || self.fs.ref_frames0[idx] != INTRA_FRAME as i8 {
            return;
        }
        let Some(&cand) = self.fs.mvs.get(idx) else {
            return;
        };
        let mut cand_mv = cand;
        lower_mv_precision(&mut cand_mv);
        if let Some(i) = st.mv[..st.num].iter().position(|&m| m == cand_mv) {
            st.weight[i] += weight;
        } else if st.num < MAX_REF_MV_STACK_SIZE {
            st.mv[st.num] = cand_mv;
            st.weight[st.num] = weight;
            st.num += 1;
        }
    }

    /// Spec 5.11.26 `assign_mv( 0 )` for `use_intrabc` (`compMode` is `NEWMV`): the prediction
    /// `PredMv[ 0 ]`, then `read_mv( 0 )`.
    fn assign_mv(&mut self, st: &MvStack) {
        let mut pred_mv = st.mv[0];
        if pred_mv == [0, 0] {
            pred_mv = st.mv[1];
        }
        if pred_mv == [0, 0] {
            let sb_size = if self.seq.use_128x128_superblock {
                BLOCK_128X128
            } else {
                BLOCK_64X64
            };
            let sb_size4 = NUM_4X4_BLOCKS_HIGH[sb_size];
            if (self.mi_row as isize) - (sb_size4 as isize) < self.mi_row_start as isize {
                pred_mv = [0, -((sb_size4 * MI_SIZE + INTRABC_DELAY_PIXELS) as i32) * 8];
            } else {
                pred_mv = [-((sb_size4 * MI_SIZE * 8) as i32), 0];
            }
        }
        self.read_mv(pred_mv);
    }

    /// Spec 5.11.31 `read_mv( 0 )` with `MvCtx` equal to `MV_INTRABC_CONTEXT`: sets `Mv[ 0 ]`.
    fn read_mv(&mut self, pred_mv: [i32; 2]) {
        let mut diff_mv = [0i32; 2];
        let mv_joint = self
            .sd
            .read_cdf(&mut self.cdf.mv_joint_cdf[MV_INTRABC_CONTEXT]);
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[0] = self.read_mv_component(0);
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[1] = self.read_mv_component(1);
        }
        self.mv = [pred_mv[0] + diff_mv[0], pred_mv[1] + diff_mv[1]];
    }

    /// Spec 5.11.32 `read_mv_component( comp )`. `force_integer_mv` is 1 (`mv_class0_fr` and
    /// `mv_fr` are 3, not read) and `allow_high_precision_mv` is 0 (`mv_class0_hp` and `mv_hp`
    /// are 1, not read) in an intra frame.
    fn read_mv_component(&mut self, comp: usize) -> i32 {
        const MV_CTX: usize = MV_INTRABC_CONTEXT;
        let mv_sign = self.sd.read_cdf(&mut self.cdf.mv_sign_cdf[MV_CTX][comp]) != 0;
        let mv_class = self.sd.read_cdf(&mut self.cdf.mv_class_cdf[MV_CTX][comp]);
        let mag = if mv_class == MV_CLASS_0 {
            let mv_class0_bit =
                self.sd
                    .read_cdf(&mut self.cdf.mv_class0_bit_cdf[MV_CTX][comp]) as i32;
            ((mv_class0_bit << 3) | (3 << 1) | 1) + 1
        } else {
            let mut d = 0i32;
            for i in 0..mv_class {
                let mv_bit = self.sd.read_cdf(&mut self.cdf.mv_bit_cdf[MV_CTX][comp][i]) as i32;
                d |= mv_bit << i;
            }
            let mag = (CLASS0_SIZE as i32) << (mv_class + 2);
            mag + (((d << 3) | (3 << 1) | 1) + 1)
        };
        if mv_sign { -mag } else { mag }
    }

    /// Spec 5.11.33 `compute_prediction( )` for an intra block copy block (`is_inter` 1,
    /// `RefFrame[ 1 ]` is `NONE` so there is no inter-intra).
    ///
    /// Every block of an intra frame has `RefFrames[ ][ ][ 0 ]` equal to `INTRA_FRAME`, so
    /// `someUseIntra` is always 1: each plane is predicted by one `predict_inter` call of the
    /// whole plane block (`predW`, `predH` are `num4x4W * 4`, `num4x4H * 4`) with the motion
    /// vector of this block (`candRow`, `candCol` are `MiRow`, `MiCol`). For the chroma of a
    /// block narrower than 8 luma samples that is the 4x4 chroma block with this block's vector.
    ///
    /// # Errors
    /// `Invalid` if the chroma block size is `BLOCK_INVALID` (a 4:2:2 stream the spec forbids).
    pub(crate) fn compute_prediction(&mut self) -> Result<(), Error> {
        let planes = if self.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = if plane > 0 {
                (self.fs.subsampling_x, self.fs.subsampling_y)
            } else {
                (0, 0)
            };
            let plane_sz = SUBSAMPLED_SIZE[self.mi_size][sub_x][sub_y];
            if plane_sz >= BLOCK_SIZES {
                return Err(Error::Invalid("invalid chroma block size"));
            }
            let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz];
            let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz];
            let base_x = (self.mi_col >> sub_x) * MI_SIZE;
            let base_y = (self.mi_row >> sub_y) * MI_SIZE;
            self.predict_inter(plane, base_x, base_y, num4x4_w * 4, num4x4_h * 4);
        }
        Ok(())
    }

    /// Spec 7.11.3.1 inter prediction process for `refIdx` -1 (the current frame), a single
    /// reference, no warp (`force_integer_mv` is 1) and `COMPOUND_AVERAGE`, `SIMPLE` motion.
    fn predict_inter(&mut self, plane: usize, x: usize, y: usize, w: usize, h: usize) {
        let bit_depth = u32::from(self.seq.bit_depth);
        // 7.11.3.2 rounding variables derivation process, isCompound 0.
        let (round0, round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
        let (sub_x, sub_y) = if plane > 0 {
            (self.fs.subsampling_x, self.fs.subsampling_y)
        } else {
            (0, 0)
        };
        // 7.11.3.3 motion vector scaling process. RefUpscaledWidth[ -1 ] is UpscaledWidth, which
        // equals FrameWidth when allow_intrabc is 1, and RefFrameHeight[ -1 ] is FrameHeight:
        // xScale and yScale are 1 << REF_SCALE_SHIFT, stepX and stepY are 1 << SCALE_SUBPEL_BITS.
        let scale = 1i64 << REF_SCALE_SHIFT;
        let half_sample = 1i64 << (SUBPEL_BITS - 1);
        let off = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
        let mv = self.mv;
        let orig_x = ((x as i64) << SUBPEL_BITS) + ((2 * i64::from(mv[1])) >> sub_x) + half_sample;
        let orig_y = ((y as i64) << SUBPEL_BITS) + ((2 * i64::from(mv[0])) >> sub_y) + half_sample;
        let base_x = orig_x * scale - (half_sample << REF_SCALE_SHIFT);
        let base_y = orig_y * scale - (half_sample << REF_SCALE_SHIFT);
        let shift = REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS;
        let start_x = round2_signed(base_x, shift) + off;
        let start_y = round2_signed(base_y, shift) + off;
        // 7.11.3.4: RefUpscaledWidth[ -1 ] and RefFrameHeight[ -1 ] are MiCols * MI_SIZE and
        // MiRows * MI_SIZE, so the prediction is not cropped to the frame size.
        let last_x = ((self.fs.mi_cols * MI_SIZE + sub_x) >> sub_x) - 1;
        let last_y = ((self.fs.mi_rows * MI_SIZE + sub_y) >> sub_y) - 1;
        let fp = &mut self.fs.curr_frame[plane];
        predict_intrabc(
            &mut fp.px,
            fp.stride,
            (x, y),
            (w, h),
            (start_x, start_y),
            (last_x, last_y),
            (round0, round1),
            bit_depth,
        );
    }
}
