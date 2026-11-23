//! Tile and block syntax of an intra frame (`KEY_FRAME` / `INTRA_ONLY_FRAME`), spec 5.11:
//! `decode_tile`, `decode_partition`, `decode_block`, `intra_frame_mode_info`, the segment id,
//! `read_cdef`, the delta q / delta lf syntax, intra modes, CFL alphas, palette mode info and
//! tokens, filter intra, the transform size, `read_lr` / `read_lr_unit`, with the CDF selection
//! process of spec 9.3 for each syntax element.
//!
//! The residual (`residual( )`, transform blocks, coefficients, transform type) is not here:
//! `decode_block` calls `self.residual( )`, which `src/coef.rs` implements as an
//! `impl TileDecoder<'_>` method. Prediction happens per transform block inside it, so
//! `compute_prediction( )` has nothing to do in an intra frame and is not called.
//!
//! # Entry point
//!
//! ```ignore
//! let td = TileDecoder::new(tile_bytes, tile_row, tile_col, &frame_cdfs, &mut frame_state,
//!                           &frame_header, &seq)?;
//! let final_cdfs = td.decode_tile(&cancel)?;
//! ```
//!
//! `tile_bytes` is the tile's data (`tileSize` bytes of `tile_group_obu`), `frame_cdfs` the
//! frame's CDFs at the start of the tile (the tile works on a copy, spec 8.2.2). `decode_tile`
//! runs the symbol decoder's exit process (spec 8.2.4) and returns the tile's final CDFs: the
//! caller keeps the ones of `context_update_tile_id` for the frame end update CDF process.
//! `cancel` is checked once per superblock row. The caller creates a new `TileDecoder` per tile
//! and sets `CurrentQIndex` (done in `new`) per tile.
//!
//! The spec's block variables are fields of [`TileDecoder`] in snake case (`MiRow` is `mi_row`).

#![allow(dead_code)]

use crate::cdf::CdfContext;
use crate::consts::*;
use crate::error::Error;
use crate::frame::{Charged, FrameState};
use crate::header::FrameHeader;
use crate::seq::SequenceHeader;
use crate::symbol::SymbolDecoder;
use crate::tables::{
    BLOCK_HEIGHT, BLOCK_WIDTH, MAX_TX_DEPTH, MAX_TX_SIZE_RECT, MI_HEIGHT_LOG2, MI_WIDTH_LOG2,
    NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PALETTE_COLOR_CONTEXT,
    PALETTE_COLOR_HASH_MULTIPLIERS, PARTITION_SUBSIZE, SPLIT_TX_SIZE, SUBSAMPLED_SIZE, TX_HEIGHT,
    TX_WIDTH,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// `Wiener_Taps_Mid` (spec 5.11.2).
const WIENER_TAPS_MID: [i32; 3] = [3, -7, 15];
/// `Sgrproj_Xqd_Mid` (spec 5.11.2).
const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];
/// `Wiener_Taps_Min`, `Wiener_Taps_Max`, `Wiener_Taps_K` (spec 5.11.58).
const WIENER_TAPS_MIN: [i32; 3] = [-5, -23, -17];
const WIENER_TAPS_MAX: [i32; 3] = [10, 8, 46];
const WIENER_TAPS_K: [u32; 3] = [1, 2, 3];
/// `Sgrproj_Xqd_Min`, `Sgrproj_Xqd_Max` (spec 5.11.58).
const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];
const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];

/// `Sgr_Params` (spec 7.17.3): `{ r0, e0, r1, e1 }` per `lr_sgr_set`. Only the radii are read
/// here; the loop restoration process has the same table.
const SGR_PARAMS: [[u8; 4]; 16] = [
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

/// `Intra_Mode_Context` (spec 9.3, `intra_frame_y_mode`).
const INTRA_MODE_CONTEXT: [usize; INTRA_MODES] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `signU` / `signV` values (spec 5.11.45).
const CFL_SIGN_ZERO: usize = 0;
const CFL_SIGN_NEG: usize = 1;

/// Largest superblock in units of 4 samples; the above and left context arrays are rounded up
/// to a multiple of it so a block that overhangs the frame edge still writes in bounds.
const MAX_SB4: usize = 32;

/// Side of the `BlockDecoded` array: `sbSize4 + 1` entries plus the -1 row and column.
const BD: usize = 35;

/// The state of the decode of one tile. See the module documentation for the entry point.
pub(crate) struct TileDecoder<'a> {
    /// The symbol decoder over the tile's bytes.
    pub sd: SymbolDecoder<'a>,
    /// The tile's working copy of the CDFs (`Tile*Cdf`).
    pub cdf: Box<CdfContext>,
    pub seq: &'a SequenceHeader,
    pub fh: &'a FrameHeader,
    /// The frame-level arrays and planes.
    pub fs: &'a mut FrameState,

    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,

    pub mi_row: usize,
    pub mi_col: usize,
    pub mi_size: usize,
    pub has_chroma: bool,
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub skip: bool,
    pub skip_mode: bool,
    pub is_inter: bool,
    pub use_intrabc: bool,
    pub segment_id: usize,
    pub lossless: bool,
    pub y_mode: usize,
    pub uv_mode: usize,
    pub angle_delta_y: i32,
    pub angle_delta_uv: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub cfl_alpha_u: i32,
    pub cfl_alpha_v: i32,
    pub palette_size_y: usize,
    pub palette_size_uv: usize,
    pub palette_colors_y: [u16; 8],
    pub palette_colors_u: [u16; 8],
    pub palette_colors_v: [u16; 8],
    /// `ColorMapY[ i ][ j ]`.
    pub color_map_y: Box<[[u8; 64]; 64]>,
    /// `ColorMapUV[ i ][ j ]`.
    pub color_map_uv: Box<[[u8; 64]; 64]>,
    /// `TxSize`.
    pub tx_size: usize,
    /// `CurrentQIndex`.
    pub current_q_index: i32,
    /// `DeltaLF[ i ]`.
    pub delta_lf: [i32; FRAME_LF_COUNT],
    pub read_deltas: bool,

    /// `RefSgrXqd[ plane ][ pass ]`.
    pub ref_sgr_xqd: [[i32; 2]; 3],
    /// `RefLrWiener[ plane ][ pass ][ i ]`.
    pub ref_lr_wiener: [[[i32; 3]; 2]; 3],

    /// `AboveLevelContext[ plane ][ x4 ]`, indexed by the absolute column in units of 4 samples
    /// of that plane. The length is `MiCols` rounded up to a multiple of 32 (0 for a plane the
    /// stream does not have).
    pub above_level_context: [Vec<u8>; 3],
    /// `AboveDcContext[ plane ][ x4 ]`.
    pub above_dc_context: [Vec<u8>; 3],
    /// `AboveSegPredContext[ x4 ]`.
    pub above_seg_pred_context: Vec<u8>,
    /// `LeftLevelContext[ plane ][ y4 ]`, indexed by the absolute row in units of 4 samples of
    /// that plane (length `MiRows` rounded up to a multiple of 32).
    pub left_level_context: [Vec<u8>; 3],
    /// `LeftDcContext[ plane ][ y4 ]`.
    pub left_dc_context: [Vec<u8>; 3],
    /// `LeftSegPredContext[ y4 ]`.
    pub left_seg_pred_context: Vec<u8>,

    /// `BlockDecoded[ plane ][ y ][ x ]` stored at `[ y + 1 ][ x + 1 ]`; use
    /// [`TileDecoder::block_decoded_at`] and [`TileDecoder::set_block_decoded`].
    pub block_decoded: [[[bool; BD]; BD]; 3],

    _charged: Charged,
}

/// `neg_deinterleave( diff, ref, max )` (spec 5.11.9).
fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        max - (diff + 1)
    }
}

/// `inverse_recenter( r, v )` (spec 5.9.29).
fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `CeilLog2( x )` (spec 4.7).
fn ceil_log2(x: i32) -> u32 {
    if x < 2 {
        return 0;
    }
    let mut i = 1;
    let mut p = 2;
    while p < x {
        i += 1;
        p <<= 1;
    }
    i
}

/// `is_directional_mode( mode )` (spec 5.11.44).
fn is_directional_mode(mode: usize) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

/// `get_palette_color_context( colorMap, r, c, n )` (spec 5.11.50): returns `ColorOrder` and
/// `ColorContextHash`.
fn get_palette_color_context(
    color_map: &[[u8; 64]; 64],
    r: usize,
    c: usize,
    n: usize,
) -> ([u8; PALETTE_COLORS], usize) {
    let mut scores = [0usize; PALETTE_COLORS];
    let mut color_order = [0u8; PALETTE_COLORS];
    for (i, o) in color_order.iter_mut().enumerate() {
        *o = i as u8;
    }
    // Colour indices are below PALETTE_COLORS (they come from NS(n) or from ColorOrder).
    if c > 0 {
        scores[usize::from(color_map[r][c - 1]) & 7] += 2;
    }
    if r > 0 && c > 0 {
        scores[usize::from(color_map[r - 1][c - 1]) & 7] += 1;
    }
    if r > 0 {
        scores[usize::from(color_map[r - 1][c]) & 7] += 2;
    }
    for i in 0..PALETTE_NUM_NEIGHBORS {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in (i + 1)..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_score = scores[max_idx];
            let max_color_order = color_order[max_idx];
            for k in ((i + 1)..=max_idx).rev() {
                scores[k] = scores[k - 1];
                color_order[k] = color_order[k - 1];
            }
            scores[i] = max_score;
            color_order[i] = max_color_order;
        }
    }
    let mut hash = 0;
    for i in 0..PALETTE_NUM_NEIGHBORS {
        hash += scores[i] * PALETTE_COLOR_HASH_MULTIPLIERS[i];
    }
    (color_order, hash)
}

/// The part of `palette_tokens( )` for one colour map (spec 5.11.49): reads the indices of a
/// block of `block_w` x `block_h` samples of which `onscreen_w` x `onscreen_h` are inside the
/// frame, and replicates the last column and row into the rest.
fn read_color_map(
    sd: &mut SymbolDecoder<'_>,
    cdf: &mut CdfContext,
    map: &mut [[u8; 64]; 64],
    uv: bool,
    n: usize,
    block_w: usize,
    block_h: usize,
    onscreen_w: usize,
    onscreen_h: usize,
) {
    let block_w = block_w.min(64);
    let block_h = block_h.min(64);
    if onscreen_w == 0 || onscreen_h == 0 {
        return;
    }
    for i in 1..(onscreen_h + onscreen_w - 1) {
        let hi = i.min(onscreen_w - 1);
        let lo = (i + 1).saturating_sub(onscreen_h);
        for j in (lo..=hi).rev() {
            let (color_order, hash) = get_palette_color_context(map, i - j, j, n);
            let ctx =
                usize::try_from(PALETTE_COLOR_CONTEXT.get(hash).copied().unwrap_or(0)).unwrap_or(0);
            let row = if uv {
                cdf.palette_uv_color_cdf(n, ctx)
            } else {
                cdf.palette_y_color_cdf(n, ctx)
            };
            let idx = sd.read_symbol(row);
            map[i - j][j] = color_order[idx & 7];
        }
    }
    for i in 0..onscreen_h {
        for j in onscreen_w..block_w {
            map[i][j] = map[i][onscreen_w - 1];
        }
    }
    for i in onscreen_h..block_h {
        for j in 0..block_w {
            map[i][j] = map[onscreen_h - 1][j];
        }
    }
}

/// Sets `val` in a `rows` x `cols` rectangle at (`r`, `c`) of a `mi_cols` wide array; positions
/// outside the array are left alone.
fn fill_rows<T: Copy>(
    v: &mut [T],
    mi_cols: usize,
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    val: T,
) {
    for y in 0..rows {
        let start = (r + y) * mi_cols + c;
        if let Some(s) = v.get_mut(start..start + cols) {
            s.fill(val);
        }
    }
}

/// Clears `v[ start..end ]`, clamped to the length of `v`.
fn clear_range(v: &mut [u8], start: usize, end: usize) {
    let end = end.min(v.len());
    let start = start.min(end);
    v[start..end].fill(0);
}

impl<'a> TileDecoder<'a> {
    /// Sets up the decode of tile (`tile_row`, `tile_col`) whose bytes are `data`: the tile
    /// bounds (`MiRowStart` ...), `CurrentQIndex = base_q_idx` and `init_symbol( )` (spec 8.2.2,
    /// including the tile's copy of `frame_cdf`).
    ///
    /// # Errors
    /// `Invalid` for empty tile data or a tile outside the frame, `AllocLimit` / `OutOfMemory`
    /// from the context arrays.
    pub(crate) fn new(
        data: &'a [u8],
        tile_row: usize,
        tile_col: usize,
        frame_cdf: &CdfContext,
        fs: &'a mut FrameState,
        fh: &'a FrameHeader,
        seq: &'a SequenceHeader,
    ) -> Result<TileDecoder<'a>, Error> {
        fn bound(v: &[usize], i: usize) -> Result<usize, Error> {
            v.get(i)
                .copied()
                .ok_or(Error::Invalid("tile index out of range"))
        }
        let mi_row_start = bound(&fs.mi_row_starts, tile_row)?;
        let mi_row_end = bound(&fs.mi_row_starts, tile_row + 1)?;
        let mi_col_start = bound(&fs.mi_col_starts, tile_col)?;
        let mi_col_end = bound(&fs.mi_col_starts, tile_col + 1)?;
        if mi_row_start > mi_row_end
            || mi_col_start > mi_col_end
            || mi_row_end > fs.mi_rows
            || mi_col_end > fs.mi_cols
        {
            return Err(Error::Invalid("tile outside the frame"));
        }
        let sd = SymbolDecoder::new(data, fh.disable_cdf_update)?;
        let mut charged = Charged::new(&fs.ledger);
        let cols = fs.mi_cols.next_multiple_of(MAX_SB4);
        let rows = fs.mi_rows.next_multiple_of(MAX_SB4);
        let np = fs.num_planes;
        let pn = |n: usize, plane: usize| if plane < np { n } else { 0 };
        let above_level_context = [
            charged.vec(pn(cols, 0))?,
            charged.vec(pn(cols, 1))?,
            charged.vec(pn(cols, 2))?,
        ];
        let above_dc_context = [
            charged.vec(pn(cols, 0))?,
            charged.vec(pn(cols, 1))?,
            charged.vec(pn(cols, 2))?,
        ];
        let above_seg_pred_context = charged.vec(cols)?;
        let left_level_context = [
            charged.vec(pn(rows, 0))?,
            charged.vec(pn(rows, 1))?,
            charged.vec(pn(rows, 2))?,
        ];
        let left_dc_context = [
            charged.vec(pn(rows, 0))?,
            charged.vec(pn(rows, 1))?,
            charged.vec(pn(rows, 2))?,
        ];
        let left_seg_pred_context = charged.vec(rows)?;
        Ok(TileDecoder {
            sd,
            cdf: frame_cdf.init_symbol_cdfs(),
            seq,
            fh,
            fs,
            mi_row_start,
            mi_row_end,
            mi_col_start,
            mi_col_end,
            mi_row: 0,
            mi_col: 0,
            mi_size: 0,
            has_chroma: false,
            avail_u: false,
            avail_l: false,
            avail_u_chroma: false,
            avail_l_chroma: false,
            skip: false,
            skip_mode: false,
            is_inter: false,
            use_intrabc: false,
            segment_id: 0,
            lossless: false,
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            palette_size_y: 0,
            palette_size_uv: 0,
            palette_colors_y: [0; 8],
            palette_colors_u: [0; 8],
            palette_colors_v: [0; 8],
            color_map_y: Box::new([[0; 64]; 64]),
            color_map_uv: Box::new([[0; 64]; 64]),
            tx_size: TX_4X4,
            current_q_index: i32::from(fh.base_q_idx),
            delta_lf: [0; FRAME_LF_COUNT],
            read_deltas: false,
            ref_sgr_xqd: [[0; 2]; 3],
            ref_lr_wiener: [[[0; 3]; 2]; 3],
            above_level_context,
            above_dc_context,
            above_seg_pred_context,
            left_level_context,
            left_dc_context,
            left_seg_pred_context,
            block_decoded: [[[false; BD]; BD]; 3],
            _charged: charged,
        })
    }

    /// Spec 5.11.2 `decode_tile( )`, followed by the symbol decoder's exit process (spec 8.2.4).
    /// Returns the tile's final CDFs.
    ///
    /// # Errors
    /// `Cancelled` when `cancel` is set (checked once per superblock row), `Unsupported` for
    /// intra block copy, `Invalid` for a syntax value the spec forbids or for trailing bits that
    /// break the exit process's conformance requirement.
    pub(crate) fn decode_tile(mut self, cancel: &AtomicBool) -> Result<Box<CdfContext>, Error> {
        self.clear_above_context();
        self.delta_lf = [0; FRAME_LF_COUNT];
        for plane in 0..self.fs.num_planes {
            for pass in 0..2 {
                self.ref_sgr_xqd[plane][pass] = SGRPROJ_XQD_MID[pass];
                self.ref_lr_wiener[plane][pass] = WIENER_TAPS_MID;
            }
        }
        let sb_size = if self.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        let sb_size4 = NUM_4X4_BLOCKS_WIDE[sb_size];
        for r in (self.mi_row_start..self.mi_row_end).step_by(sb_size4) {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            self.clear_left_context();
            for c in (self.mi_col_start..self.mi_col_end).step_by(sb_size4) {
                self.read_deltas = self.fh.delta_q_present;
                self.clear_cdef(r, c);
                self.clear_block_decoded_flags(r, c, sb_size4);
                self.read_lr(r, c, sb_size)?;
                self.decode_partition(r, c, sb_size)?;
            }
        }
        let TileDecoder { sd, cdf, .. } = self;
        if !sd.exit() {
            return Err(Error::Invalid("tile padding or trailing bits"));
        }
        Ok(cdf)
    }

    /// `is_inside( candidateR, candidateC )` (spec 5.11.51).
    pub(crate) fn is_inside(&self, candidate_r: isize, candidate_c: isize) -> bool {
        candidate_c >= self.mi_col_start as isize
            && candidate_c < self.mi_col_end as isize
            && candidate_r >= self.mi_row_start as isize
            && candidate_r < self.mi_row_end as isize
    }

    /// Index into the per-mode-info arrays of `FrameState`.
    pub(crate) fn mi_idx(&self, row: usize, col: usize) -> usize {
        row * self.fs.mi_cols + col
    }

    /// `BlockDecoded[ plane ][ y ][ x ]`; false outside the stored range (`y`, `x` from -1).
    pub(crate) fn block_decoded_at(&self, plane: usize, y: i32, x: i32) -> bool {
        let (Ok(y), Ok(x)) = (usize::try_from(y + 1), usize::try_from(x + 1)) else {
            return false;
        };
        self.block_decoded
            .get(plane)
            .and_then(|p| p.get(y))
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Sets `BlockDecoded[ plane ][ y ][ x ]` to `value`; ignored outside the stored range.
    pub(crate) fn set_block_decoded(&mut self, plane: usize, y: i32, x: i32, value: bool) {
        let (Ok(y), Ok(x)) = (usize::try_from(y + 1), usize::try_from(x + 1)) else {
            return;
        };
        if let Some(b) = self
            .block_decoded
            .get_mut(plane)
            .and_then(|p| p.get_mut(y))
            .and_then(|row| row.get_mut(x))
        {
            *b = value;
        }
    }

    /// `clear_above_context( )` (spec 5.11.2): the above context arrays
    /// are set to 0.
    pub(crate) fn clear_above_context(&mut self) {
        for plane in 0..3 {
            self.above_level_context[plane].fill(0);
            self.above_dc_context[plane].fill(0);
        }
        self.above_seg_pred_context.fill(0);
    }

    /// `clear_left_context( )`: the left context arrays are set to 0.
    pub(crate) fn clear_left_context(&mut self) {
        for plane in 0..3 {
            self.left_level_context[plane].fill(0);
            self.left_dc_context[plane].fill(0);
        }
        self.left_seg_pred_context.fill(0);
    }

    /// `clear_cdef( r, c )` (spec 5.11.55).
    fn clear_cdef(&mut self, r: usize, c: usize) {
        let mi_cols = self.fs.mi_cols;
        let n = if self.seq.use_128x128_superblock {
            2
        } else {
            1
        };
        let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64];
        for i in 0..n {
            for j in 0..n {
                let (rr, cc) = (r + i * cdef_size4, c + j * cdef_size4);
                if rr < self.fs.mi_rows && cc < mi_cols {
                    self.fs.cdef_idx[rr * mi_cols + cc] = -1;
                }
            }
        }
    }

    /// `clear_block_decoded_flags( r, c, sbSize4 )` (spec 5.11.3).
    fn clear_block_decoded_flags(&mut self, r: usize, c: usize, sb_size4: usize) {
        for plane in 0..self.fs.num_planes {
            let (sub_x, sub_y) = if plane > 0 {
                (self.fs.subsampling_x, self.fs.subsampling_y)
            } else {
                (0, 0)
            };
            let sb_width4 = ((self.mi_col_end - c) >> sub_x) as i32;
            let sb_height4 = ((self.mi_row_end - r) >> sub_y) as i32;
            let ymax = (sb_size4 >> sub_y) as i32;
            let xmax = (sb_size4 >> sub_x) as i32;
            for y in -1..=ymax {
                for x in -1..=xmax {
                    let v = (y < 0 && x < sb_width4) || (x < 0 && y < sb_height4);
                    self.set_block_decoded(plane, y, x, v);
                }
            }
            self.set_block_decoded(plane, ymax, -1, false);
        }
    }

    /// Spec 5.11.4 `decode_partition( r, c, bSize )`.
    fn decode_partition(&mut self, r: usize, c: usize, b_size: usize) -> Result<(), Error> {
        if r >= self.fs.mi_rows || c >= self.fs.mi_cols {
            return Ok(());
        }
        let avail_u = self.is_inside(r as isize - 1, c as isize);
        let avail_l = self.is_inside(r as isize, c as isize - 1);
        let num4x4 = NUM_4X4_BLOCKS_WIDE[b_size];
        let half_block4x4 = num4x4 >> 1;
        let quarter_block4x4 = half_block4x4 >> 1;
        let has_rows = (r + half_block4x4) < self.fs.mi_rows;
        let has_cols = (c + half_block4x4) < self.fs.mi_cols;
        let partition = if b_size < BLOCK_8X8 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            self.read_partition(r, c, b_size, avail_u, avail_l)
        } else if has_cols {
            if self.read_split_or(r, c, b_size, avail_u, avail_l, true) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            if self.read_split_or(r, c, b_size, avail_u, avail_l, false) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        let sub_size = PARTITION_SUBSIZE[partition][b_size];
        let split_size = PARTITION_SUBSIZE[PARTITION_SPLIT][b_size];
        if sub_size >= BLOCK_SIZES || split_size >= BLOCK_SIZES {
            return Err(Error::Invalid("partition not allowed for this block size"));
        }
        let (h, q) = (half_block4x4, quarter_block4x4);
        match partition {
            PARTITION_NONE => self.decode_block(r, c, sub_size)?,
            PARTITION_HORZ => {
                self.decode_block(r, c, sub_size)?;
                if has_rows {
                    self.decode_block(r + h, c, sub_size)?;
                }
            }
            PARTITION_VERT => {
                self.decode_block(r, c, sub_size)?;
                if has_cols {
                    self.decode_block(r, c + h, sub_size)?;
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(r, c, sub_size)?;
                self.decode_partition(r, c + h, sub_size)?;
                self.decode_partition(r + h, c, sub_size)?;
                self.decode_partition(r + h, c + h, sub_size)?;
            }
            PARTITION_HORZ_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c, sub_size)?;
            }
            PARTITION_HORZ_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_VERT_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r, c + h, sub_size)?;
            }
            PARTITION_VERT_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_HORZ_4 => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + q, c, sub_size)?;
                self.decode_block(r + q * 2, c, sub_size)?;
                if r + q * 3 < self.fs.mi_rows {
                    self.decode_block(r + q * 3, c, sub_size)?;
                }
            }
            _ => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + q, sub_size)?;
                self.decode_block(r, c + q * 2, sub_size)?;
                if c + q * 3 < self.fs.mi_cols {
                    self.decode_block(r, c + q * 3, sub_size)?;
                }
            }
        }
        Ok(())
    }

    /// The `ctx` of the `partition`, `split_or_horz` and `split_or_vert` CDF selection (spec 9.3).
    fn partition_ctx(
        &self,
        r: usize,
        c: usize,
        b_size: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> usize {
        let bsl = MI_WIDTH_LOG2[b_size];
        let above =
            avail_u && MI_WIDTH_LOG2[usize::from(self.fs.mi_sizes[self.mi_idx(r - 1, c)])] < bsl;
        let left =
            avail_l && MI_HEIGHT_LOG2[usize::from(self.fs.mi_sizes[self.mi_idx(r, c - 1)])] < bsl;
        usize::from(left) * 2 + usize::from(above)
    }

    /// The `partition` syntax element (spec 9.3 CDF selection).
    fn read_partition(
        &mut self,
        r: usize,
        c: usize,
        b_size: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> usize {
        let ctx = self.partition_ctx(r, c, b_size, avail_u, avail_l);
        let cdf = &mut *self.cdf;
        let row: &mut [u16] = match MI_WIDTH_LOG2[b_size] {
            1 => &mut cdf.partition_w8_cdf[ctx][..],
            2 => &mut cdf.partition_w16_cdf[ctx][..],
            3 => &mut cdf.partition_w32_cdf[ctx][..],
            4 => &mut cdf.partition_w64_cdf[ctx][..],
            _ => &mut cdf.partition_w128_cdf[ctx][..],
        };
        self.sd.read_symbol(row)
    }

    /// The `split_or_horz` (`horz` true) or `split_or_vert` syntax element (spec 9.3): returns
    /// whether the partition is `PARTITION_SPLIT`.
    fn read_split_or(
        &mut self,
        r: usize,
        c: usize,
        b_size: usize,
        avail_u: bool,
        avail_l: bool,
        horz: bool,
    ) -> bool {
        let ctx = self.partition_ctx(r, c, b_size, avail_u, avail_l);
        let cdf = &self.cdf;
        let row: &[u16] = match MI_WIDTH_LOG2[b_size] {
            1 => &cdf.partition_w8_cdf[ctx][..],
            2 => &cdf.partition_w16_cdf[ctx][..],
            3 => &cdf.partition_w32_cdf[ctx][..],
            4 => &cdf.partition_w64_cdf[ctx][..],
            _ => &cdf.partition_w128_cdf[ctx][..],
        };
        let p = |x: usize| -> i32 {
            let a = i32::from(row.get(x).copied().unwrap_or(0));
            let b = i32::from(row.get(x - 1).copied().unwrap_or(0));
            a - b
        };
        let mut psum = if horz {
            p(PARTITION_VERT)
                + p(PARTITION_SPLIT)
                + p(PARTITION_HORZ_A)
                + p(PARTITION_VERT_A)
                + p(PARTITION_VERT_B)
        } else {
            p(PARTITION_HORZ)
                + p(PARTITION_SPLIT)
                + p(PARTITION_HORZ_A)
                + p(PARTITION_HORZ_B)
                + p(PARTITION_VERT_A)
        };
        if b_size != BLOCK_128X128 {
            psum += if horz {
                p(PARTITION_VERT_4)
            } else {
                p(PARTITION_HORZ_4)
            };
        }
        let mut tmp = [((1 << 15) - psum) as u16, 1 << 15, 0];
        self.sd.read_symbol(&mut tmp) != 0
    }

    /// Spec 5.11.5 `decode_block( r, c, subSize )` for an intra frame.
    fn decode_block(&mut self, r: usize, c: usize, sub_size: usize) -> Result<(), Error> {
        self.mi_row = r;
        self.mi_col = c;
        self.mi_size = sub_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size];
        let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size];
        let (ssx, ssy) = (self.fs.subsampling_x, self.fs.subsampling_y);
        let no_chroma =
            (bh4 == 1 && ssy != 0 && (r & 1) == 0) || (bw4 == 1 && ssx != 0 && (c & 1) == 0);
        self.has_chroma = !no_chroma && self.fs.num_planes > 1;
        self.avail_u = self.is_inside(r as isize - 1, c as isize);
        self.avail_l = self.is_inside(r as isize, c as isize - 1);
        self.avail_u_chroma = self.avail_u;
        self.avail_l_chroma = self.avail_l;
        if self.has_chroma {
            if ssy != 0 && bh4 == 1 {
                self.avail_u_chroma = self.is_inside(r as isize - 2, c as isize);
            }
            if ssx != 0 && bw4 == 1 {
                self.avail_l_chroma = self.is_inside(r as isize, c as isize - 2);
            }
        } else {
            self.avail_u_chroma = false;
            self.avail_l_chroma = false;
        }
        self.intra_frame_mode_info()?;
        self.palette_tokens();
        self.read_block_tx_size();
        if self.skip {
            self.reset_block_context(bw4, bh4);
        }
        let mi_cols = self.fs.mi_cols;
        let rows = bh4.min(self.fs.mi_rows.saturating_sub(r));
        let cols = bw4.min(mi_cols.saturating_sub(c));
        fill_rows(
            &mut self.fs.y_modes,
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.y_mode as u8,
        );
        if self.has_chroma {
            fill_rows(
                &mut self.fs.uv_modes,
                mi_cols,
                r,
                c,
                rows,
                cols,
                self.uv_mode as u8,
            );
        }
        fill_rows(
            &mut self.fs.ref_frames0,
            mi_cols,
            r,
            c,
            rows,
            cols,
            INTRA_FRAME as i8,
        );
        // compute_prediction( ) is a no-op for an intra block: prediction runs per transform
        // block inside residual( ).
        self.residual()?;
        let fs = &mut *self.fs;
        fill_rows(&mut fs.is_inters, mi_cols, r, c, rows, cols, self.is_inter);
        fill_rows(&mut fs.skips, mi_cols, r, c, rows, cols, self.skip);
        fill_rows(
            &mut fs.tx_sizes,
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.tx_size as u8,
        );
        fill_rows(
            &mut fs.mi_sizes,
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.mi_size as u8,
        );
        fill_rows(
            &mut fs.segment_ids,
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.segment_id as u8,
        );
        fill_rows(
            &mut fs.palette_sizes[0],
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.palette_size_y as u8,
        );
        fill_rows(
            &mut fs.palette_sizes[1],
            mi_cols,
            r,
            c,
            rows,
            cols,
            self.palette_size_uv as u8,
        );
        if self.palette_size_y > 0 {
            fill_rows(
                &mut fs.palette_colors[0],
                mi_cols,
                r,
                c,
                rows,
                cols,
                self.palette_colors_y,
            );
        }
        if self.palette_size_uv > 0 {
            fill_rows(
                &mut fs.palette_colors[1],
                mi_cols,
                r,
                c,
                rows,
                cols,
                self.palette_colors_u,
            );
        }
        let delta_lf = self.delta_lf.map(|v| v as i8);
        fill_rows(&mut fs.delta_lfs, mi_cols, r, c, rows, cols, delta_lf);
        Ok(())
    }

    /// `reset_block_context( bw4, bh4 )` (spec 5.11.5).
    fn reset_block_context(&mut self, bw4: usize, bh4: usize) {
        let planes = if self.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = if plane > 0 {
                (self.fs.subsampling_x, self.fs.subsampling_y)
            } else {
                (0, 0)
            };
            let (c0, c1) = (self.mi_col >> sub_x, (self.mi_col + bw4) >> sub_x);
            let (r0, r1) = (self.mi_row >> sub_y, (self.mi_row + bh4) >> sub_y);
            clear_range(&mut self.above_level_context[plane], c0, c1);
            clear_range(&mut self.above_dc_context[plane], c0, c1);
            clear_range(&mut self.left_level_context[plane], r0, r1);
            clear_range(&mut self.left_dc_context[plane], r0, r1);
        }
    }

    /// `seg_feature_active( feature )` (spec 5.11.14).
    fn seg_feature_active(&self, feature: usize) -> bool {
        self.fh.segmentation_enabled && self.fh.feature_enabled[self.segment_id][feature]
    }

    /// Spec 5.11.7 `intra_frame_mode_info( )`.
    fn intra_frame_mode_info(&mut self) -> Result<(), Error> {
        self.skip = false;
        if self.fh.seg_id_pre_skip {
            self.intra_segment_id()?;
        }
        self.skip_mode = false;
        self.read_skip();
        if !self.fh.seg_id_pre_skip {
            self.intra_segment_id()?;
        }
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.use_intrabc =
            self.fh.allow_intrabc && self.sd.read_symbol(&mut self.cdf.intrabc_cdf) != 0;
        if self.use_intrabc {
            return Err(Error::Unsupported("intrabc"));
        }
        self.is_inter = false;
        let above_mode = if self.avail_u {
            usize::from(self.fs.y_modes[self.mi_idx(self.mi_row - 1, self.mi_col)])
        } else {
            DC_PRED
        };
        let left_mode = if self.avail_l {
            usize::from(self.fs.y_modes[self.mi_idx(self.mi_row, self.mi_col - 1)])
        } else {
            DC_PRED
        };
        let above_ctx = INTRA_MODE_CONTEXT[above_mode.min(INTRA_MODES - 1)];
        let left_ctx = INTRA_MODE_CONTEXT[left_mode.min(INTRA_MODES - 1)];
        self.y_mode = self
            .sd
            .read_symbol(&mut self.cdf.intra_frame_y_mode_cdf[above_ctx][left_ctx]);
        self.intra_angle_info_y();
        if self.has_chroma {
            let (ssx, ssy) = (self.fs.subsampling_x, self.fs.subsampling_y);
            let cfl_allowed = if self.lossless {
                SUBSAMPLED_SIZE[self.mi_size][ssx][ssy] == BLOCK_4X4
            } else {
                BLOCK_WIDTH[self.mi_size].max(BLOCK_HEIGHT[self.mi_size]) <= 32
            };
            self.uv_mode = if cfl_allowed {
                self.sd
                    .read_symbol(&mut self.cdf.uv_mode_cfl_allowed_cdf[self.y_mode])
            } else {
                self.sd
                    .read_symbol(&mut self.cdf.uv_mode_cfl_not_allowed_cdf[self.y_mode])
            };
            if self.uv_mode == UV_CFL_PRED {
                self.read_cfl_alphas();
            }
            self.intra_angle_info_uv();
        }
        self.palette_size_y = 0;
        self.palette_size_uv = 0;
        if self.mi_size >= BLOCK_8X8
            && BLOCK_WIDTH[self.mi_size] <= 64
            && BLOCK_HEIGHT[self.mi_size] <= 64
            && self.fh.allow_screen_content_tools
        {
            self.palette_mode_info();
        }
        self.filter_intra_mode_info();
        Ok(())
    }

    /// Spec 5.11.8 `intra_segment_id( )`.
    fn intra_segment_id(&mut self) -> Result<(), Error> {
        if self.fh.segmentation_enabled {
            self.read_segment_id()?;
        } else {
            self.segment_id = 0;
        }
        self.lossless = self.fh.lossless_array[self.segment_id];
        Ok(())
    }

    /// Spec 5.11.9 `read_segment_id( )`.
    fn read_segment_id(&mut self) -> Result<(), Error> {
        let (r, c) = (self.mi_row, self.mi_col);
        let seg = |row: usize, col: usize| i32::from(self.fs.segment_ids[self.mi_idx(row, col)]);
        let prev_ul = if self.avail_u && self.avail_l {
            seg(r - 1, c - 1)
        } else {
            -1
        };
        let prev_u = if self.avail_u { seg(r - 1, c) } else { -1 };
        let prev_l = if self.avail_l { seg(r, c - 1) } else { -1 };
        let pred = if prev_u == -1 {
            if prev_l == -1 { 0 } else { prev_l }
        } else if prev_l == -1 || prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if self.skip {
            self.segment_id = pred as usize;
            return Ok(());
        }
        let ctx = if prev_ul >= 0 && prev_ul == prev_u && prev_ul == prev_l {
            2
        } else if prev_ul >= 0 && (prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l) {
            1
        } else {
            0
        };
        let coded = self.sd.read_symbol(&mut self.cdf.segment_id_cdf[ctx]) as i32;
        let last = i32::from(self.fh.last_active_seg_id);
        let v = neg_deinterleave(coded, pred, last + 1);
        if !(0..=last).contains(&v) {
            return Err(Error::Invalid("segment_id above LastActiveSegId"));
        }
        self.segment_id = v as usize;
        Ok(())
    }

    /// Spec 5.11.11 `read_skip( )`.
    fn read_skip(&mut self) {
        if self.fh.seg_id_pre_skip && self.seg_feature_active(SEG_LVL_SKIP) {
            self.skip = true;
        } else {
            let mut ctx = 0;
            if self.avail_u {
                ctx += usize::from(self.fs.skips[self.mi_idx(self.mi_row - 1, self.mi_col)]);
            }
            if self.avail_l {
                ctx += usize::from(self.fs.skips[self.mi_idx(self.mi_row, self.mi_col - 1)]);
            }
            self.skip = self.sd.read_symbol(&mut self.cdf.skip_cdf[ctx]) != 0;
        }
    }

    /// Spec 5.11.56 `read_cdef( )`.
    fn read_cdef(&mut self) {
        if self.skip || self.fh.coded_lossless || !self.seq.enable_cdef || self.fh.allow_intrabc {
            return;
        }
        let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64];
        let cdef_mask4 = !(cdef_size4 - 1);
        let r = self.mi_row & cdef_mask4;
        let c = self.mi_col & cdef_mask4;
        let mi_cols = self.fs.mi_cols;
        if self.fs.cdef_idx[r * mi_cols + c] == -1 {
            let v = self.sd.read_literal(u32::from(self.fh.cdef_bits)) as i8;
            let w4 = NUM_4X4_BLOCKS_WIDE[self.mi_size];
            let h4 = NUM_4X4_BLOCKS_HIGH[self.mi_size];
            for i in (r..r + h4).step_by(cdef_size4) {
                for j in (c..c + w4).step_by(cdef_size4) {
                    if i < self.fs.mi_rows && j < mi_cols {
                        self.fs.cdef_idx[i * mi_cols + j] = v;
                    }
                }
            }
        }
    }

    /// Spec 5.11.12 `read_delta_qindex( )`.
    fn read_delta_qindex(&mut self) {
        let sb_size = if self.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        if self.mi_size == sb_size && self.skip {
            return;
        }
        if self.read_deltas {
            let mut delta_q_abs = self.sd.read_symbol(&mut self.cdf.delta_q_cdf) as i32;
            if delta_q_abs == DELTA_Q_SMALL as i32 {
                let delta_q_rem_bits = self.sd.read_literal(3) + 1;
                let delta_q_abs_bits = self.sd.read_literal(delta_q_rem_bits) as i32;
                delta_q_abs = delta_q_abs_bits + (1 << delta_q_rem_bits) + 1;
            }
            if delta_q_abs != 0 {
                let sign = self.sd.read_literal(1) != 0;
                let reduced = if sign { -delta_q_abs } else { delta_q_abs };
                self.current_q_index =
                    (self.current_q_index + (reduced << self.fh.delta_q_res)).clamp(1, 255);
            }
        }
    }

    /// Spec 5.11.13 `read_delta_lf( )`.
    fn read_delta_lf(&mut self) {
        let sb_size = if self.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        if self.mi_size == sb_size && self.skip {
            return;
        }
        if self.read_deltas && self.fh.delta_lf_present {
            let mut frame_lf_count = 1;
            if self.fh.delta_lf_multi {
                frame_lf_count = if self.fs.num_planes > 1 {
                    FRAME_LF_COUNT
                } else {
                    FRAME_LF_COUNT - 2
                };
            }
            for i in 0..frame_lf_count {
                let cdf: &mut [u16] = if self.fh.delta_lf_multi {
                    &mut self.cdf.delta_lf_multi_cdf[i][..]
                } else {
                    &mut self.cdf.delta_lf_cdf[..]
                };
                let delta_lf_abs = self.sd.read_symbol(cdf) as i32;
                let delta_lf_abs = if delta_lf_abs == DELTA_LF_SMALL as i32 {
                    let n = self.sd.read_literal(3) + 1;
                    let bits = self.sd.read_literal(n) as i32;
                    bits + (1 << n) + 1
                } else {
                    delta_lf_abs
                };
                if delta_lf_abs != 0 {
                    let sign = self.sd.read_literal(1) != 0;
                    let reduced = if sign { -delta_lf_abs } else { delta_lf_abs };
                    let max = MAX_LOOP_FILTER as i32;
                    self.delta_lf[i] =
                        (self.delta_lf[i] + (reduced << self.fh.delta_lf_res)).clamp(-max, max);
                }
            }
        }
    }

    /// Spec 5.11.42 `intra_angle_info_y( )`.
    fn intra_angle_info_y(&mut self) {
        self.angle_delta_y = 0;
        if self.mi_size >= BLOCK_8X8 && is_directional_mode(self.y_mode) {
            let v = self
                .sd
                .read_symbol(&mut self.cdf.angle_delta_cdf[self.y_mode - V_PRED]);
            self.angle_delta_y = v as i32 - MAX_ANGLE_DELTA as i32;
        }
    }

    /// Spec 5.11.43 `intra_angle_info_uv( )`.
    fn intra_angle_info_uv(&mut self) {
        self.angle_delta_uv = 0;
        if self.mi_size >= BLOCK_8X8 && is_directional_mode(self.uv_mode) {
            let v = self
                .sd
                .read_symbol(&mut self.cdf.angle_delta_cdf[self.uv_mode - V_PRED]);
            self.angle_delta_uv = v as i32 - MAX_ANGLE_DELTA as i32;
        }
    }

    /// Spec 5.11.45 `read_cfl_alphas( )`.
    fn read_cfl_alphas(&mut self) {
        let cfl_alpha_signs = self.sd.read_symbol(&mut self.cdf.cfl_sign_cdf);
        let sign_u = (cfl_alpha_signs + 1) / 3;
        let sign_v = (cfl_alpha_signs + 1) % 3;
        if sign_u != CFL_SIGN_ZERO {
            let ctx = (sign_u - 1) * 3 + sign_v;
            let v = self.sd.read_symbol(&mut self.cdf.cfl_alpha_cdf[ctx]) as i32;
            self.cfl_alpha_u = if sign_u == CFL_SIGN_NEG {
                -(1 + v)
            } else {
                1 + v
            };
        } else {
            self.cfl_alpha_u = 0;
        }
        if sign_v != CFL_SIGN_ZERO {
            let ctx = (sign_v - 1) * 3 + sign_u;
            let v = self.sd.read_symbol(&mut self.cdf.cfl_alpha_cdf[ctx]) as i32;
            self.cfl_alpha_v = if sign_v == CFL_SIGN_NEG {
                -(1 + v)
            } else {
                1 + v
            };
        } else {
            self.cfl_alpha_v = 0;
        }
    }

    /// Spec 5.11.24 `filter_intra_mode_info( )`.
    fn filter_intra_mode_info(&mut self) {
        self.use_filter_intra = false;
        if self.seq.enable_filter_intra
            && self.y_mode == DC_PRED
            && self.palette_size_y == 0
            && BLOCK_WIDTH[self.mi_size].max(BLOCK_HEIGHT[self.mi_size]) <= 32
        {
            self.use_filter_intra = self
                .sd
                .read_symbol(&mut self.cdf.filter_intra_cdf[self.mi_size])
                != 0;
            if self.use_filter_intra {
                self.filter_intra_mode = self.sd.read_symbol(&mut self.cdf.filter_intra_mode_cdf);
            }
        }
    }

    /// `get_palette_cache( plane )` (spec 5.11.46): fills `cache` (`PaletteCache`) and returns
    /// its length.
    fn get_palette_cache(&self, plane: usize, cache: &mut [u16; 16]) -> usize {
        let (r, c) = (self.mi_row, self.mi_col);
        let size_at = |row: usize, col: usize| -> usize {
            self.fs.palette_sizes[plane]
                .get(self.mi_idx(row, col))
                .map_or(0, |&s| usize::from(s).min(8))
        };
        let colors_at = |row: usize, col: usize| -> [u16; 8] {
            self.fs.palette_colors[plane]
                .get(self.mi_idx(row, col))
                .copied()
                .unwrap_or([0; 8])
        };
        let above_n = if (r * MI_SIZE) & 63 != 0 {
            size_at(r - 1, c)
        } else {
            0
        };
        let left_n = if self.avail_l { size_at(r, c - 1) } else { 0 };
        let above = if above_n > 0 {
            colors_at(r - 1, c)
        } else {
            [0; 8]
        };
        let left = if left_n > 0 {
            colors_at(r, c - 1)
        } else {
            [0; 8]
        };
        let mut above_idx = 0;
        let mut left_idx = 0;
        let mut n = 0;
        let mut push = |val: u16, n: &mut usize| {
            if *n == 0 || val != cache[*n - 1] {
                cache[*n] = val;
                *n += 1;
            }
        };
        while above_idx < above_n && left_idx < left_n {
            let above_c = above[above_idx];
            let left_c = left[left_idx];
            if left_c < above_c {
                push(left_c, &mut n);
                left_idx += 1;
            } else {
                push(above_c, &mut n);
                above_idx += 1;
                if left_c == above_c {
                    left_idx += 1;
                }
            }
        }
        while above_idx < above_n {
            push(above[above_idx], &mut n);
            above_idx += 1;
        }
        while left_idx < left_n {
            push(left[left_idx], &mut n);
            left_idx += 1;
        }
        n
    }

    /// Spec 5.11.46 `palette_mode_info( )`.
    fn palette_mode_info(&mut self) {
        let bit_depth = u32::from(self.seq.bit_depth);
        let max_val = (1i32 << bit_depth) - 1;
        let bsize_ctx =
            (MI_WIDTH_LOG2[self.mi_size] + MI_HEIGHT_LOG2[self.mi_size]).saturating_sub(2);
        if self.y_mode == DC_PRED {
            let mut ctx = 0;
            if self.avail_u
                && self.fs.palette_sizes[0][self.mi_idx(self.mi_row - 1, self.mi_col)] > 0
            {
                ctx += 1;
            }
            if self.avail_l
                && self.fs.palette_sizes[0][self.mi_idx(self.mi_row, self.mi_col - 1)] > 0
            {
                ctx += 1;
            }
            let has_palette_y = self
                .sd
                .read_symbol(&mut self.cdf.palette_y_mode_cdf[bsize_ctx][ctx])
                != 0;
            if has_palette_y {
                let n = self
                    .sd
                    .read_symbol(&mut self.cdf.palette_y_size_cdf[bsize_ctx])
                    + 2;
                self.palette_size_y = n;
                let mut cache = [0u16; 16];
                let cache_n = self.get_palette_cache(0, &mut cache);
                let mut colors = [0u16; 8];
                let mut idx = 0;
                for &cached in &cache[..cache_n] {
                    if idx >= n {
                        break;
                    }
                    if self.sd.read_literal(1) != 0 {
                        colors[idx] = cached;
                        idx += 1;
                    }
                }
                if idx < n {
                    colors[idx] = self.sd.read_literal(bit_depth) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < n {
                    let min_bits = bit_depth - 3;
                    palette_bits = min_bits + self.sd.read_literal(2);
                }
                while idx < n {
                    let palette_delta_y = self.sd.read_literal(palette_bits) as i32 + 1;
                    let v = (i32::from(colors[idx - 1]) + palette_delta_y).clamp(0, max_val);
                    colors[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v - 1;
                    palette_bits = palette_bits.min(ceil_log2(range));
                    idx += 1;
                }
                colors[..n].sort_unstable();
                self.palette_colors_y = colors;
            }
        }
        if self.has_chroma && self.uv_mode == DC_PRED {
            let ctx = usize::from(self.palette_size_y > 0);
            let has_palette_uv = self.sd.read_symbol(&mut self.cdf.palette_uv_mode_cdf[ctx]) != 0;
            if has_palette_uv {
                let n = self
                    .sd
                    .read_symbol(&mut self.cdf.palette_uv_size_cdf[bsize_ctx])
                    + 2;
                self.palette_size_uv = n;
                let mut cache = [0u16; 16];
                let cache_n = self.get_palette_cache(1, &mut cache);
                let mut colors = [0u16; 8];
                let mut idx = 0;
                for &cached in &cache[..cache_n] {
                    if idx >= n {
                        break;
                    }
                    if self.sd.read_literal(1) != 0 {
                        colors[idx] = cached;
                        idx += 1;
                    }
                }
                if idx < n {
                    colors[idx] = self.sd.read_literal(bit_depth) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < n {
                    let min_bits = bit_depth - 3;
                    palette_bits = min_bits + self.sd.read_literal(2);
                }
                while idx < n {
                    let palette_delta_u = self.sd.read_literal(palette_bits) as i32;
                    let v = (i32::from(colors[idx - 1]) + palette_delta_u).clamp(0, max_val);
                    colors[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v;
                    palette_bits = palette_bits.min(ceil_log2(range));
                    idx += 1;
                }
                colors[..n].sort_unstable();
                self.palette_colors_u = colors;

                let mut v_colors = [0u16; 8];
                if self.sd.read_literal(1) != 0 {
                    let min_bits = bit_depth - 4;
                    let max_val_v = 1i32 << bit_depth;
                    let palette_bits = min_bits + self.sd.read_literal(2);
                    v_colors[0] = self.sd.read_literal(bit_depth) as u16;
                    for idx in 1..n {
                        let mut palette_delta_v = self.sd.read_literal(palette_bits) as i32;
                        if palette_delta_v != 0 && self.sd.read_literal(1) != 0 {
                            palette_delta_v = -palette_delta_v;
                        }
                        let mut val = i32::from(v_colors[idx - 1]) + palette_delta_v;
                        if val < 0 {
                            val += max_val_v;
                        }
                        if val >= max_val_v {
                            val -= max_val_v;
                        }
                        v_colors[idx] = val.clamp(0, max_val) as u16;
                    }
                } else {
                    for slot in v_colors.iter_mut().take(n) {
                        *slot = self.sd.read_literal(bit_depth) as u16;
                    }
                }
                self.palette_colors_v = v_colors;
            }
        }
    }

    /// Spec 5.11.49 `palette_tokens( )`.
    fn palette_tokens(&mut self) {
        let mut block_height = BLOCK_HEIGHT[self.mi_size];
        let mut block_width = BLOCK_WIDTH[self.mi_size];
        let mut onscreen_height =
            block_height.min(self.fs.mi_rows.saturating_sub(self.mi_row) * MI_SIZE);
        let mut onscreen_width =
            block_width.min(self.fs.mi_cols.saturating_sub(self.mi_col) * MI_SIZE);
        if self.palette_size_y > 0 {
            let n = self.palette_size_y;
            self.color_map_y[0][0] = self.sd.read_ns(n as u32) as u8;
            read_color_map(
                &mut self.sd,
                &mut self.cdf,
                &mut self.color_map_y,
                false,
                n,
                block_width,
                block_height,
                onscreen_width,
                onscreen_height,
            );
        }
        if self.palette_size_uv > 0 {
            let n = self.palette_size_uv;
            self.color_map_uv[0][0] = self.sd.read_ns(n as u32) as u8;
            block_height >>= self.fs.subsampling_y;
            block_width >>= self.fs.subsampling_x;
            onscreen_height >>= self.fs.subsampling_y;
            onscreen_width >>= self.fs.subsampling_x;
            if block_width < 4 {
                block_width += 2;
                onscreen_width += 2;
            }
            if block_height < 4 {
                block_height += 2;
                onscreen_height += 2;
            }
            read_color_map(
                &mut self.sd,
                &mut self.cdf,
                &mut self.color_map_uv,
                true,
                n,
                block_width,
                block_height,
                onscreen_width,
                onscreen_height,
            );
        }
    }

    /// Spec 5.11.16 `read_block_tx_size( )` for an intra block (`is_inter` is 0, so the
    /// `read_var_tx_size` branch is never taken). `InterTxSizes` is not kept: in an intra frame
    /// it equals `TxSizes`, which the CDF selection reads instead.
    fn read_block_tx_size(&mut self) {
        self.read_tx_size(!self.skip || !self.is_inter);
    }

    /// Spec 5.11.15 `read_tx_size( allowSelect )`.
    fn read_tx_size(&mut self, allow_select: bool) {
        if self.lossless {
            self.tx_size = TX_4X4;
            return;
        }
        let max_rect_tx_size = MAX_TX_SIZE_RECT[self.mi_size];
        let max_tx_depth = MAX_TX_DEPTH[self.mi_size];
        self.tx_size = max_rect_tx_size;
        if self.mi_size > BLOCK_4X4 && allow_select && self.fh.tx_mode == TX_MODE_SELECT {
            let ctx = self.tx_depth_ctx(max_rect_tx_size);
            let cdf = &mut *self.cdf;
            let row: &mut [u16] = match max_tx_depth {
                4 => &mut cdf.tx_64x64_cdf[ctx][..],
                3 => &mut cdf.tx_32x32_cdf[ctx][..],
                2 => &mut cdf.tx_16x16_cdf[ctx][..],
                _ => &mut cdf.tx_8x8_cdf[ctx][..],
            };
            let tx_depth = self.sd.read_symbol(row);
            for _ in 0..tx_depth {
                self.tx_size = SPLIT_TX_SIZE[self.tx_size];
            }
        }
    }

    /// The `ctx` of the `tx_depth` CDF selection (spec 9.3). `get_above_tx_width` and
    /// `get_left_tx_height` are evaluated at the block's own row and column, where a skipped
    /// inter neighbour has already been handled by the `IsInters` test.
    fn tx_depth_ctx(&self, max_rect_tx_size: usize) -> usize {
        let max_tx_width = TX_WIDTH[max_rect_tx_size];
        let max_tx_height = TX_HEIGHT[max_rect_tx_size];
        let above_w = if self.avail_u {
            let i = self.mi_idx(self.mi_row - 1, self.mi_col);
            if self.fs.is_inters[i] {
                BLOCK_WIDTH[usize::from(self.fs.mi_sizes[i])]
            } else {
                TX_WIDTH[usize::from(self.fs.tx_sizes[i])]
            }
        } else {
            0
        };
        let left_h = if self.avail_l {
            let i = self.mi_idx(self.mi_row, self.mi_col - 1);
            if self.fs.is_inters[i] {
                BLOCK_HEIGHT[usize::from(self.fs.mi_sizes[i])]
            } else {
                TX_HEIGHT[usize::from(self.fs.tx_sizes[i])]
            }
        } else {
            0
        };
        usize::from(above_w >= max_tx_width) + usize::from(left_h >= max_tx_height)
    }

    /// Spec 5.11.57 `read_lr( r, c, bSize )`.
    fn read_lr(&mut self, r: usize, c: usize, b_size: usize) -> Result<(), Error> {
        if self.fh.allow_intrabc {
            return Ok(());
        }
        let w = NUM_4X4_BLOCKS_WIDE[b_size];
        let h = NUM_4X4_BLOCKS_HIGH[b_size];
        for plane in 0..self.fs.num_planes {
            if self.fh.frame_restoration_type[plane] == RESTORE_NONE {
                continue;
            }
            let (sub_x, sub_y) = if plane == 0 {
                (0, 0)
            } else {
                (self.fs.subsampling_x, self.fs.subsampling_y)
            };
            let unit_size = usize::from(self.fh.loop_restoration_size[plane]).max(1);
            let unit_rows = self.fs.lr_unit_rows[plane];
            let unit_cols = self.fs.lr_unit_cols[plane];
            let unit_row_start = (r * (MI_SIZE >> sub_y)).div_ceil(unit_size);
            let unit_row_end = unit_rows.min(((r + h) * (MI_SIZE >> sub_y)).div_ceil(unit_size));
            let (numerator, denominator) = if self.fh.use_superres {
                (
                    (MI_SIZE >> sub_x) * self.fh.superres_denom as usize,
                    unit_size * SUPERRES_NUM,
                )
            } else {
                (MI_SIZE >> sub_x, unit_size)
            };
            let unit_col_start = (c * numerator).div_ceil(denominator);
            let unit_col_end = unit_cols.min(((c + w) * numerator).div_ceil(denominator));
            for unit_row in unit_row_start..unit_row_end {
                for unit_col in unit_col_start..unit_col_end {
                    self.read_lr_unit(plane, unit_row, unit_col)?;
                }
            }
        }
        Ok(())
    }

    /// Spec 5.11.58 `read_lr_unit( plane, unitRow, unitCol )`.
    fn read_lr_unit(
        &mut self,
        plane: usize,
        unit_row: usize,
        unit_col: usize,
    ) -> Result<(), Error> {
        let idx = unit_row * self.fs.lr_unit_cols[plane] + unit_col;
        let restoration_type = match self.fh.frame_restoration_type[plane] {
            RESTORE_WIENER => {
                if self.sd.read_symbol(&mut self.cdf.use_wiener_cdf) != 0 {
                    RESTORE_WIENER
                } else {
                    RESTORE_NONE
                }
            }
            RESTORE_SGRPROJ => {
                if self.sd.read_symbol(&mut self.cdf.use_sgrproj_cdf) != 0 {
                    RESTORE_SGRPROJ
                } else {
                    RESTORE_NONE
                }
            }
            _ => self.sd.read_symbol(&mut self.cdf.restoration_type_cdf) as u8,
        };
        let fs = &mut *self.fs;
        *fs.lr_type[plane]
            .get_mut(idx)
            .ok_or(Error::Invalid("loop restoration unit out of range"))? = restoration_type;
        if restoration_type == RESTORE_WIENER {
            for pass in 0..2 {
                let first_coeff = if plane > 0 {
                    fs.lr_wiener[plane][idx][pass][0] = 0;
                    1
                } else {
                    0
                };
                for j in first_coeff..3 {
                    let v = decode_signed_subexp_with_ref_bool(
                        &mut self.sd,
                        WIENER_TAPS_MIN[j],
                        WIENER_TAPS_MAX[j] + 1,
                        WIENER_TAPS_K[j],
                        self.ref_lr_wiener[plane][pass][j],
                    );
                    fs.lr_wiener[plane][idx][pass][j] = v as i8;
                    self.ref_lr_wiener[plane][pass][j] = v;
                }
            }
        } else if restoration_type == RESTORE_SGRPROJ {
            let lr_sgr_set = self.sd.read_literal(SGRPROJ_PARAMS_BITS as u32) as usize;
            fs.lr_sgr_set[plane][idx] = lr_sgr_set as u8;
            for i in 0..2 {
                let radius = SGR_PARAMS[lr_sgr_set][i * 2];
                let min = SGRPROJ_XQD_MIN[i];
                let max = SGRPROJ_XQD_MAX[i];
                let v = if radius != 0 {
                    decode_signed_subexp_with_ref_bool(
                        &mut self.sd,
                        min,
                        max + 1,
                        SGRPROJ_PRJ_SUBEXP_K as u32,
                        self.ref_sgr_xqd[plane][i],
                    )
                } else if i == 1 {
                    ((1 << SGRPROJ_PRJ_BITS) - self.ref_sgr_xqd[plane][0]).clamp(min, max)
                } else {
                    0
                };
                fs.lr_sgr_xqd[plane][idx][i] = v as i8;
                self.ref_sgr_xqd[plane][i] = v;
            }
        }
        Ok(())
    }
}

/// `decode_signed_subexp_with_ref_bool( low, high, k, r )` (spec 5.11.58).
fn decode_signed_subexp_with_ref_bool(
    sd: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    k: u32,
    r: i32,
) -> i32 {
    decode_unsigned_subexp_with_ref_bool(sd, high - low, k, r - low) + low
}

/// `decode_unsigned_subexp_with_ref_bool( mx, k, r )` (spec 5.11.58).
fn decode_unsigned_subexp_with_ref_bool(
    sd: &mut SymbolDecoder<'_>,
    mx: i32,
    k: u32,
    r: i32,
) -> i32 {
    let v = decode_subexp_bool(sd, mx, k);
    if (r << 1) <= mx {
        inverse_recenter(r, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - r, v)
    }
}

/// `decode_subexp_bool( numSyms, k )` (spec 5.11.58).
fn decode_subexp_bool(sd: &mut SymbolDecoder<'_>, num_syms: i32, k: u32) -> i32 {
    let mut i = 0u32;
    let mut mk = 0i32;
    loop {
        let b2 = if i > 0 { k + i - 1 } else { k };
        let a = 1i32 << b2;
        if num_syms <= mk + 3 * a {
            return sd.read_ns((num_syms - mk).max(1) as u32) as i32 + mk;
        }
        if sd.read_literal(1) != 0 {
            i += 1;
            mk += a;
        } else {
            return sd.read_literal(b2) as i32 + mk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_deinterleave_matches_the_spec_function() {
        // ref = 0: the value is unchanged.
        assert_eq!(neg_deinterleave(5, 0, 8), 5);
        // ref >= max - 1: mirrored.
        assert_eq!(neg_deinterleave(2, 7, 8), 5);
        // 2 * ref < max: odd diffs go above the reference, even ones below, then plain.
        assert_eq!(neg_deinterleave(1, 2, 8), 3);
        assert_eq!(neg_deinterleave(2, 2, 8), 1);
        assert_eq!(neg_deinterleave(4, 2, 8), 0);
        assert_eq!(neg_deinterleave(5, 2, 8), 5);
        // 2 * ref >= max.
        assert_eq!(neg_deinterleave(1, 5, 8), 6);
        assert_eq!(neg_deinterleave(2, 5, 8), 4);
        assert_eq!(neg_deinterleave(7, 5, 8), 0);
    }

    #[test]
    fn inverse_recenter_and_ceil_log2() {
        assert_eq!(inverse_recenter(5, 3), 3);
        assert_eq!(inverse_recenter(5, 4), 7);
        assert_eq!(inverse_recenter(5, 11), 11);
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
    }

    #[test]
    fn palette_color_context_sorts_by_score() {
        // Left neighbour 1 (+2), above-left 0 (+1), above 2 (+2): scores [1, 2, 2].
        let mut map = [[0u8; 64]; 64];
        map[1][0] = 1;
        map[0][0] = 0;
        map[0][1] = 2;
        let (order, hash) = get_palette_color_context(&map, 1, 1, 3);
        assert_eq!(order, [1, 2, 0, 3, 4, 5, 6, 7]);
        // Sorted scores [2, 2, 1]: 2 * 1 + 2 * 2 + 1 * 2.
        assert_eq!(hash, 8);
        assert_eq!(PALETTE_COLOR_CONTEXT[hash], 1);
    }

    #[test]
    fn clear_range_clamps() {
        let mut v = vec![1u8; 4];
        clear_range(&mut v, 2, 9);
        assert_eq!(v, [1, 1, 0, 0]);
        clear_range(&mut v, 7, 9);
        assert_eq!(v, [1, 1, 0, 0]);
    }
}
