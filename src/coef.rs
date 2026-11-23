//! Residual decoding and reconstruction wiring of an intra block: `residual( )` (spec 5.11.34),
//! `transform_block( )` (5.11.35) with the intra, palette and CFL prediction calls, `get_tx_size`
//! (5.11.37), `coeffs( )` (5.11.39) with its CDF selection (9.3), `compute_tx_type` (5.11.40),
//! `transform_type( )` (5.11.47) and `get_tx_set( )` (5.11.48), and the call of the
//! reconstruction process (7.12.3) with the quantizers of 7.12.2.
//!
//! The only inter blocks of the frames this decoder handles are intra block copy blocks
//! (`is_inter` 1, spec 5.11.7), so the inter branches of the spec are the ones an intrabc
//! block takes: `transform_tree` (5.11.36), the inter transform sets and `inter_tx_type`. Their
//! prediction is done by `compute_prediction( )` (`src/intrabc.rs`) before `residual( )`.
//! `TxTypes` is only kept as a frame array (`FrameState::tx_types`) for inter blocks, whose
//! chroma reads the luma type back; an intra block's luma type is only read back by the same
//! transform block, and its chroma derives it from `UVMode`.
//!
//! Entry point: `TileDecoder::residual( )`, called by `decode_block` after the mode info.
//!
//! The padded levels array behind the `coeff_base` and `coeff_br` contexts is ported from
//! dav1d's `src/recon_tmpl.c` (`decode_coefs`; Copyright 2018-2019, VideoLAN and dav1d authors,
//! Copyright 2018-2019, Two Orioles, LLC; BSD-2-Clause, see NOTICE and docs/dependencies.md).

use crate::consts::*;
use crate::error::Error;
use crate::predict::{IntraEdges, IntraParams, predict_cfl, predict_intra, predict_palette};
use crate::quant::{ac_q, dc_q};
use crate::recon::{ReconParams, reconstruct};
use crate::scan::get_scan;
use crate::tables::{
    ADJUSTED_TX_SIZE, BLOCK_HEIGHT, BLOCK_WIDTH, MAX_TX_SIZE_RECT, MODE_TO_TXFM,
    NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, SIG_REF_DIFF_OFFSET, SUBSAMPLED_SIZE, TX_HEIGHT,
    TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
};
use crate::tile::{TileDecoder, fill_rows};

/// `Tx_Type_In_Set_Intra` (spec 5.11.40).
const TX_TYPE_IN_SET_INTRA: [[u8; TX_TYPES]; TX_SET_TYPES_INTRA] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
];

/// `Tx_Type_In_Set_Inter` (spec 5.11.40).
const TX_TYPE_IN_SET_INTER: [[u8; TX_TYPES]; TX_SET_TYPES_INTER] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
];

/// `Tx_Type_Inter_Inv_Set1` (spec 5.11.47).
const TX_TYPE_INTER_INV_SET1: [usize; 16] = [
    IDTX,
    V_DCT,
    H_DCT,
    V_ADST,
    H_ADST,
    V_FLIPADST,
    H_FLIPADST,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set2` (spec 5.11.47).
const TX_TYPE_INTER_INV_SET2: [usize; 12] = [
    IDTX,
    V_DCT,
    H_DCT,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set3` (spec 5.11.47).
const TX_TYPE_INTER_INV_SET3: [usize; 2] = [IDTX, DCT_DCT];

/// `Tx_Type_Intra_Inv_Set1` (spec 5.11.47).
const TX_TYPE_INTRA_INV_SET1: [usize; 7] =
    [IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

/// `Tx_Type_Intra_Inv_Set2` (spec 5.11.47).
const TX_TYPE_INTRA_INV_SET2: [usize; 5] = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

/// `Filter_Intra_Mode_To_Intra_Dir` (spec 9.3, `intra_tx_type`).
const FILTER_INTRA_MODE_TO_INTRA_DIR: [usize; 5] = [DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED];

/// `Coeff_Base_Pos_Ctx_Offset` (spec 9.3, `coeff_base`).
const COEFF_BASE_POS_CTX_OFFSET: [usize; 3] = [
    SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_2D + 5,
    SIG_COEF_CONTEXTS_2D + 10,
];

/// `Mag_Ref_Offset_With_Tx_Class` (spec 9.3, `coeff_br`): `[ row, col ]` offsets.
const MAG_REF_OFFSET_WITH_TX_CLASS: [[[usize; 2]; 3]; 3] = [
    [[0, 1], [1, 0], [1, 1]],
    [[0, 1], [1, 0], [0, 2]],
    [[0, 1], [1, 0], [2, 0]],
];

/// `Coeff_Base_Ctx_Offset[ TX_SIZES_ALL ][ 5 ][ 5 ]` (spec 9.3, `coeff_base`).
const COEFF_BASE_CTX_OFFSET: [[[u8; 5]; 5]; TX_SIZES_ALL] = [
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
];

/// Largest `Quant[ ]` array (spec 5.11.39: `Min( 1024, w * h )`). `TileDecoder::quant` is
/// laid out `Quant[ i * tw + j ]`, `tw = Min( 32, w )`.
pub(crate) const QUANT_LEN: usize = 1024;

/// Zero columns right of, and zero rows below, the levels of a transform block in
/// `TileDecoder::levels` (dav1d's approach): the largest offset of `Sig_Ref_Diff_Offset`
/// (spec 9.3), so the context reads of `coeff_base_ctx` and `coeff_br_ctx` stay inside the
/// padded block and read zero where the spec skips a position outside the block.
const TX_PAD: usize = 4;

/// The padded levels of the largest coded area (32x32).
pub(crate) const LEVELS_LEN: usize = (32 + TX_PAD) * (32 + TX_PAD);

/// `SegQMLevel` value that disables the quantizer matrix (`NUM_QM_LEVELS - 1`).
const QM_LEVEL_FLAT: usize = 15;

/// Values that live across the transform blocks of one `residual( )` call.
struct ResidualScratch {
    /// `MaxLumaW` and `MaxLumaH` (spec 5.11.35).
    max_luma_w: usize,
    max_luma_h: usize,
}

/// `get_tx_class( txType )` (spec 9.3, `coeff_base`).
fn get_tx_class(tx_type: usize) -> usize {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    }
}

/// `is_tx_type_in_set( txSet, txType )` for an intra block (spec 5.11.40).
fn is_tx_type_in_set(tx_set: usize, tx_type: usize) -> bool {
    TX_TYPE_IN_SET_INTRA[tx_set][tx_type] != 0
}

/// `is_tx_type_in_set( txSet, txType )` for an inter block (spec 5.11.40).
fn is_tx_type_in_set_inter(tx_set: usize, tx_type: usize) -> bool {
    TX_TYPE_IN_SET_INTER[tx_set][tx_type] != 0
}

/// `find_tx_size( w, h )` (spec 5.11.36): the transform size of `w` x `h` samples, if any.
fn find_tx_size(w: usize, h: usize) -> Option<usize> {
    (0..TX_SIZES_ALL).find(|&tx_sz| TX_WIDTH[tx_sz] == w && TX_HEIGHT[tx_sz] == h)
}

/// The `ctx` of `coeff_base_eob`: `get_coeff_base_ctx( ..., isEob = 1 ) - SIG_COEF_CONTEXTS +
/// SIG_COEF_CONTEXTS_EOB` (spec 9.3), for the scan index `c`.
fn coeff_base_eob_ctx(tx_sz: usize, c: usize) -> usize {
    let adj = ADJUSTED_TX_SIZE[tx_sz];
    let area = TX_HEIGHT[adj] << TX_WIDTH_LOG2[adj];
    if c == 0 {
        0
    } else if c <= area / 8 {
        1
    } else if c <= area / 4 {
        2
    } else {
        3
    }
}

/// `get_coeff_base_ctx( txSz, plane, x4, y4, pos, c, 0 )` (spec 9.3, `coeff_base`) for a
/// transform class `tx_class` at (`row`, `col`). `levels` holds the levels decoded so far,
/// padded (row stride `stride`, see [`TX_PAD`]).
fn coeff_base_ctx(
    levels: &[u8],
    stride: usize,
    tx_sz: usize,
    tx_class: usize,
    row: usize,
    col: usize,
) -> usize {
    let i = row * stride + col;
    let mag: usize = SIG_REF_DIFF_OFFSET[tx_class]
        .iter()
        .map(|off| usize::from(levels[i + off[0] * stride + off[1]].min(3)))
        .sum();
    let ctx = ((mag + 1) >> 1).min(4);
    if tx_class == TX_CLASS_2D {
        if row == 0 && col == 0 {
            return 0;
        }
        return ctx + usize::from(COEFF_BASE_CTX_OFFSET[tx_sz][row.min(4)][col.min(4)]);
    }
    let idx = if tx_class == TX_CLASS_VERT { row } else { col };
    ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
}

/// The `ctx` of `coeff_br` (spec 9.3) at (`row`, `col`), from the padded `levels` as for
/// [`coeff_base_ctx`].
fn coeff_br_ctx(levels: &[u8], stride: usize, tx_class: usize, row: usize, col: usize) -> usize {
    let i = row * stride + col;
    let mag: usize = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class]
        .iter()
        .map(|off| {
            usize::from(
                levels[i + off[0] * stride + off[1]]
                    .min((COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as u8),
            )
        })
        .sum();
    let mag = ((mag + 1) >> 1).min(6);
    if row == 0 && col == 0 {
        mag
    } else if tx_class == TX_CLASS_2D {
        if row < 2 && col < 2 {
            mag + 7
        } else {
            mag + 14
        }
    } else if tx_class == TX_CLASS_HORIZ {
        if col == 0 { mag + 7 } else { mag + 14 }
    } else if row == 0 {
        mag + 7
    } else {
        mag + 14
    }
}

/// Sets `v[ start..start + n ]` to `val`, clamped to the length of `v`.
fn set_ctx(v: &mut [u8], start: usize, n: usize, val: u8) {
    let end = (start + n).min(v.len());
    if start < end {
        v[start..end].fill(val);
    }
}

impl TileDecoder<'_> {
    /// Spec 5.11.34 `residual( )` for an intra block, including the 64x64 chunking.
    ///
    /// # Errors
    /// `Invalid` if the chroma block size is `BLOCK_INVALID` (a 4:2:2 stream the spec forbids) or
    /// a `golomb_length_bit` rule is broken.
    pub(crate) fn residual(&mut self) -> Result<(), Error> {
        let width_chunks = (BLOCK_WIDTH[self.mi_size] >> 6).max(1);
        let height_chunks = (BLOCK_HEIGHT[self.mi_size] >> 6).max(1);
        let mi_size_chunk = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64
        } else {
            self.mi_size
        };
        let planes = if self.has_chroma { 3 } else { 1 };
        if self.has_chroma {
            let (ssx, ssy) = (self.fs.subsampling_x, self.fs.subsampling_y);
            if SUBSAMPLED_SIZE[self.mi_size][ssx][ssy] >= BLOCK_SIZES
                || SUBSAMPLED_SIZE[mi_size_chunk][ssx][ssy] >= BLOCK_SIZES
            {
                return Err(Error::Invalid("invalid chroma block size"));
            }
        }
        let mut s = ResidualScratch {
            max_luma_w: 0,
            max_luma_h: 0,
        };
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                for plane in 0..planes {
                    let tx_sz = if self.lossless {
                        TX_4X4
                    } else {
                        self.get_tx_size(plane, self.tx_size)
                    };
                    let step_x = TX_WIDTH[tx_sz] >> 2;
                    let step_y = TX_HEIGHT[tx_sz] >> 2;
                    let plane_sz = self.get_plane_residual_size(mi_size_chunk, plane);
                    let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz];
                    let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz];
                    let (sub_x, sub_y) = self.plane_subsampling(plane);
                    if self.is_inter && !self.lossless && plane == 0 {
                        let base_x = (self.mi_col + (chunk_x << 4)) * MI_SIZE;
                        let base_y = (self.mi_row + (chunk_y << 4)) * MI_SIZE;
                        self.transform_tree(base_x, base_y, num4x4_w * 4, num4x4_h * 4, &mut s)?;
                        continue;
                    }
                    let base_x_block = (self.mi_col >> sub_x) * MI_SIZE;
                    let base_y_block = (self.mi_row >> sub_y) * MI_SIZE;
                    for y in (0..num4x4_h).step_by(step_y) {
                        for x in (0..num4x4_w).step_by(step_x) {
                            self.transform_block(
                                plane,
                                base_x_block,
                                base_y_block,
                                tx_sz,
                                x + ((chunk_x << 4) >> sub_x),
                                y + ((chunk_y << 4) >> sub_y),
                                &mut s,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Spec 5.11.36 `transform_tree( startX, startY, w, h )`, for the luma of an inter block.
    ///
    /// # Errors
    /// `Invalid` if a size of the tree is not a transform size (not reachable from the block
    /// sizes the syntax allows).
    fn transform_tree(
        &mut self,
        start_x: usize,
        start_y: usize,
        w: usize,
        h: usize,
        s: &mut ResidualScratch,
    ) -> Result<(), Error> {
        let max_x = self.fs.mi_cols * MI_SIZE;
        let max_y = self.fs.mi_rows * MI_SIZE;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let row = start_y >> MI_SIZE_LOG2;
        let col = start_x >> MI_SIZE_LOG2;
        let luma_tx_sz = usize::from(self.fs.inter_tx_sizes[self.mi_idx(row, col)]);
        let luma_w = TX_WIDTH[luma_tx_sz];
        let luma_h = TX_HEIGHT[luma_tx_sz];
        if w <= luma_w && h <= luma_h {
            let tx_sz = find_tx_size(w, h).ok_or(Error::Invalid("transform tree size"))?;
            self.transform_block(0, start_x, start_y, tx_sz, 0, 0, s)
        } else if w > h {
            self.transform_tree(start_x, start_y, w / 2, h, s)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h, s)
        } else if w < h {
            self.transform_tree(start_x, start_y, w, h / 2, s)?;
            self.transform_tree(start_x, start_y + h / 2, w, h / 2, s)
        } else {
            self.transform_tree(start_x, start_y, w / 2, h / 2, s)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h / 2, s)?;
            self.transform_tree(start_x, start_y + h / 2, w / 2, h / 2, s)?;
            self.transform_tree(start_x + w / 2, start_y + h / 2, w / 2, h / 2, s)
        }
    }

    /// `(subX, subY)` of `plane`.
    fn plane_subsampling(&self, plane: usize) -> (usize, usize) {
        if plane > 0 {
            (self.fs.subsampling_x, self.fs.subsampling_y)
        } else {
            (0, 0)
        }
    }

    /// Spec 5.11.38 `get_plane_residual_size( subsize, plane )`. The caller has checked that the
    /// result is not `BLOCK_INVALID`.
    fn get_plane_residual_size(&self, subsize: usize, plane: usize) -> usize {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        SUBSAMPLED_SIZE[subsize][sub_x][sub_y]
    }

    /// Spec 5.11.37 `get_tx_size( plane, txSz )`.
    fn get_tx_size(&self, plane: usize, tx_sz: usize) -> usize {
        if plane == 0 {
            return tx_sz;
        }
        let uv_tx = MAX_TX_SIZE_RECT[self.get_plane_residual_size(self.mi_size, plane)];
        if TX_WIDTH[uv_tx] == 64 || TX_HEIGHT[uv_tx] == 64 {
            if TX_WIDTH[uv_tx] == 16 {
                return TX_16X32;
            }
            if TX_HEIGHT[uv_tx] == 16 {
                return TX_32X16;
            }
            return TX_32X32;
        }
        uv_tx
    }

    /// Spec 5.11.35 `transform_block( plane, baseX, baseY, txSz, x, y )`. An inter block was
    /// predicted by `compute_prediction( )`, so only its residual is added here.
    fn transform_block(
        &mut self,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx_sz: usize,
        x: usize,
        y: usize,
        s: &mut ResidualScratch,
    ) -> Result<(), Error> {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let row = (start_y << sub_y) >> MI_SIZE_LOG2;
        let col = (start_x << sub_x) >> MI_SIZE_LOG2;
        let sb_mask = if self.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let sub_block_mi_row = ((row & sb_mask) >> sub_y) as i32;
        let sub_block_mi_col = ((col & sb_mask) >> sub_x) as i32;
        let step_x = TX_WIDTH[tx_sz] >> MI_SIZE_LOG2;
        let step_y = TX_HEIGHT[tx_sz] >> MI_SIZE_LOG2;
        let max_x = (self.fs.mi_cols * MI_SIZE) >> sub_x;
        let max_y = (self.fs.mi_rows * MI_SIZE) >> sub_y;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let tw = TX_WIDTH[tx_sz];
        let th = TX_HEIGHT[tx_sz];
        let bit_depth = u32::from(self.seq.bit_depth);

        let palette_size = if plane == 0 {
            self.palette_size_y
        } else {
            self.palette_size_uv
        };
        if self.is_inter {
            // Predicted by compute_prediction( ) (spec 5.11.35: only `!is_inter` predicts here).
        } else if palette_size > 0 {
            // predict_palette( plane, startX, startY, x, y, txSz ), spec 7.11.4.
            if y * 4 + th > 64 || x * 4 + tw > 64 {
                return Err(Error::Invalid("palette map out of range"));
            }
            let (palette, map) = match plane {
                0 => (&self.palette_colors_y, &*self.color_map_y),
                1 => (&self.palette_colors_u, &*self.color_map_uv),
                _ => (&self.palette_colors_v, &*self.color_map_uv),
            };
            let fp = &mut self.fs.curr_frame[plane];
            predict_palette(
                &mut fp.px,
                fp.stride,
                start_x,
                start_y,
                tw,
                th,
                palette,
                &map.as_flattened()[y * 4 * 64 + x * 4..],
                64,
            );
        } else {
            let is_cfl = plane > 0 && self.uv_mode == UV_CFL_PRED;
            let (mode, angle_delta) = if plane == 0 {
                (self.y_mode, self.angle_delta_y)
            } else if is_cfl {
                (DC_PRED, self.angle_delta_uv)
            } else {
                (self.uv_mode, self.angle_delta_uv)
            };
            let (avail_l, avail_u) = if plane == 0 {
                (self.avail_l, self.avail_u)
            } else {
                (self.avail_l_chroma, self.avail_u_chroma)
            };
            let edges = IntraEdges {
                have_left: avail_l || x > 0,
                have_above: avail_u || y > 0,
                have_above_right: self.block_decoded_at(
                    plane,
                    sub_block_mi_row - 1,
                    sub_block_mi_col + step_x as i32,
                ),
                have_below_left: self.block_decoded_at(
                    plane,
                    sub_block_mi_row + step_y as i32,
                    sub_block_mi_col - 1,
                ),
            };
            let params = IntraParams {
                mode,
                angle_delta,
                log2w: TX_WIDTH_LOG2[tx_sz],
                log2h: TX_HEIGHT_LOG2[tx_sz],
                use_filter_intra: plane == 0 && self.use_filter_intra,
                filter_intra_mode: self.filter_intra_mode,
                enable_intra_edge_filter: self.seq.enable_intra_edge_filter,
                smooth_neighbor: self.get_filter_type(plane),
                max_x: max_x - 1,
                max_y: max_y - 1,
                bit_depth,
            };
            let fp = &mut self.fs.curr_frame[plane];
            predict_intra(&mut fp.px, fp.stride, start_x, start_y, edges, &params);
            if is_cfl {
                let alpha = if plane == 1 {
                    self.cfl_alpha_u
                } else {
                    self.cfl_alpha_v
                };
                let (luma, chroma) = self.fs.curr_frame.split_at_mut(1);
                let luma = &luma[0];
                let chroma = &mut chroma[plane - 1];
                predict_cfl(
                    &mut chroma.px,
                    chroma.stride,
                    &luma.px,
                    luma.stride,
                    start_x,
                    start_y,
                    tw,
                    th,
                    sub_x != 0,
                    sub_y != 0,
                    s.max_luma_w,
                    s.max_luma_h,
                    alpha,
                    bit_depth,
                );
            }
        }
        if plane == 0 && !self.is_inter {
            s.max_luma_w = start_x + step_x * 4;
            s.max_luma_h = start_y + step_y * 4;
        }

        if !self.skip {
            let (eob, plane_tx_type) = self.coeffs(plane, start_x, start_y, tx_sz)?;
            if eob > 0 {
                self.reconstruct_block(plane, start_x, start_y, tx_sz, plane_tx_type);
            }
        }

        let mi_cols = self.fs.mi_cols;
        let mi_rows = self.fs.mi_rows;
        for i in 0..step_y {
            for j in 0..step_x {
                let (r, c) = ((row >> sub_y) + i, (col >> sub_x) + j);
                if r < mi_rows
                    && c < mi_cols
                    && let Some(v) = self.fs.loopfilter_tx_sizes[plane].get_mut(r * mi_cols + c)
                {
                    *v = tx_sz as u8;
                }
                self.set_block_decoded(
                    plane,
                    sub_block_mi_row + i as i32,
                    sub_block_mi_col + j as i32,
                    true,
                );
            }
        }
        Ok(())
    }

    /// The call of the reconstruction process (spec 7.12.3) with `get_dc_quant( plane )` and
    /// `get_ac_quant( plane )` (spec 7.12.2), on `self.quant`.
    fn reconstruct_block(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
        plane_tx_type: usize,
    ) {
        let fh = self.fh;
        let bit_depth = u32::from(self.seq.bit_depth);
        let qindex = fh.get_qindex(false, self.segment_id, self.current_q_index);
        let (dc_delta, ac_delta) = match plane {
            0 => (fh.delta_q_y_dc, 0),
            1 => (fh.delta_q_u_dc, fh.delta_q_u_ac),
            _ => (fh.delta_q_v_dc, fh.delta_q_v_ac),
        };
        let qm_level = if fh.using_qmatrix {
            usize::from(fh.seg_qm_level[plane][self.segment_id])
        } else {
            QM_LEVEL_FLAT
        };
        let p = ReconParams {
            tx_sz,
            tx_type: plane_tx_type,
            plane,
            dc_q: dc_q(qindex + i32::from(dc_delta), bit_depth),
            ac_q: ac_q(qindex + i32::from(ac_delta), bit_depth),
            qm_level,
            lossless: self.lossless,
            bit_depth,
        };
        let fp = &mut self.fs.curr_frame[plane];
        reconstruct(
            &mut fp.px,
            fp.stride,
            start_x,
            start_y,
            &p,
            &self.quant[..],
            &mut self.itx,
            crate::simd::best(),
        );
    }

    /// `get_filter_type( plane )` (spec 7.11.2.8).
    fn get_filter_type(&self, plane: usize) -> bool {
        let ssx = self.fs.subsampling_x != 0;
        let ssy = self.fs.subsampling_y != 0;
        let mut above_smooth = false;
        let mut left_smooth = false;
        let avail_u = if plane == 0 {
            self.avail_u
        } else {
            self.avail_u_chroma
        };
        let avail_l = if plane == 0 {
            self.avail_l
        } else {
            self.avail_l_chroma
        };
        if avail_u {
            let mut r = self.mi_row as isize - 1;
            let mut c = self.mi_col as isize;
            if plane > 0 {
                if ssx && self.mi_col & 1 == 0 {
                    c += 1;
                }
                if ssy && self.mi_row & 1 != 0 {
                    r -= 1;
                }
            }
            above_smooth = self.is_smooth(r, c, plane);
        }
        if avail_l {
            let mut r = self.mi_row as isize;
            let mut c = self.mi_col as isize - 1;
            if plane > 0 {
                if ssx && self.mi_col & 1 != 0 {
                    c -= 1;
                }
                if ssy && self.mi_row & 1 == 0 {
                    r += 1;
                }
            }
            left_smooth = self.is_smooth(r, c, plane);
        }
        above_smooth || left_smooth
    }

    /// `is_smooth( row, col, plane )` (spec 7.11.2.8). Every block of an intra frame is intra, so
    /// the `RefFrames` test of the spec never returns early. A position outside the frame is not
    /// smooth.
    fn is_smooth(&self, row: isize, col: isize, plane: usize) -> bool {
        let (Ok(row), Ok(col)) = (usize::try_from(row), usize::try_from(col)) else {
            return false;
        };
        if row >= self.fs.mi_rows || col >= self.fs.mi_cols {
            return false;
        }
        let i = self.mi_idx(row, col);
        let mode = if plane == 0 {
            self.fs.y_modes[i]
        } else {
            self.fs.uv_modes[i]
        };
        matches!(
            usize::from(mode),
            SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED
        )
    }

    /// Spec 5.11.48 `get_tx_set( txSz )`.
    fn get_tx_set(&self, tx_sz: usize) -> usize {
        let tx_sz_sqr = TX_SIZE_SQR[tx_sz];
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_sz];
        if tx_sz_sqr_up > TX_32X32 {
            return TX_SET_DCTONLY;
        }
        if self.is_inter {
            return if self.fh.reduced_tx_set || tx_sz_sqr_up == TX_32X32 {
                TX_SET_INTER_3
            } else if tx_sz_sqr == TX_16X16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            };
        }
        if tx_sz_sqr_up >= TX_32X32 {
            TX_SET_DCTONLY
        } else if self.fh.reduced_tx_set || tx_sz_sqr == TX_16X16 {
            TX_SET_INTRA_2
        } else {
            TX_SET_INTRA_1
        }
    }

    /// Spec 5.11.40 `compute_tx_type( plane, txSz, blockX, blockY )`, with `blockX` and `blockY`
    /// the `x4` and `y4` of the transform block. `luma_tx_type` is `TxTypes[ blockY ][ blockX ]`
    /// of the luma transform block just decoded (only read when `plane` is 0).
    fn compute_tx_type(
        &self,
        plane: usize,
        tx_sz: usize,
        block_x: usize,
        block_y: usize,
        luma_tx_type: usize,
    ) -> usize {
        if self.lossless || TX_SIZE_SQR_UP[tx_sz] > TX_32X32 {
            return DCT_DCT;
        }
        let tx_set = self.get_tx_set(tx_sz);
        if plane == 0 {
            return luma_tx_type;
        }
        if self.is_inter {
            let x4 = self.mi_col.max(block_x << self.fs.subsampling_x);
            let y4 = self.mi_row.max(block_y << self.fs.subsampling_y);
            let tx_type = self
                .fs
                .tx_types
                .get(self.mi_idx(y4, x4))
                .map_or(DCT_DCT, |&t| usize::from(t));
            return if is_tx_type_in_set_inter(tx_set, tx_type) {
                tx_type
            } else {
                DCT_DCT
            };
        }
        let tx_type = MODE_TO_TXFM[self.uv_mode];
        if is_tx_type_in_set(tx_set, tx_type) {
            tx_type
        } else {
            DCT_DCT
        }
    }

    /// Spec 5.11.47 `transform_type( x4, y4, txSz )`: returns `TxType`. The caller stores it in
    /// `TxTypes` for an inter block.
    fn transform_type(&mut self, tx_sz: usize) -> usize {
        let set = self.get_tx_set(tx_sz);
        let qindex = self
            .fh
            .get_qindex(true, self.segment_id, self.current_q_index);
        if set == TX_SET_DCTONLY || qindex <= 0 {
            return DCT_DCT;
        }
        let sqr = TX_SIZE_SQR[tx_sz];
        if self.is_inter {
            // set 1 is only chosen for 4x4 and 8x8 squares (get_tx_set).
            return match set {
                TX_SET_INTER_1 => {
                    let sym = self
                        .sd
                        .read_symbol(&mut self.cdf.inter_tx_type_set1_cdf[sqr]);
                    TX_TYPE_INTER_INV_SET1[sym]
                }
                TX_SET_INTER_2 => {
                    let sym = self.sd.read_symbol(&mut self.cdf.inter_tx_type_set2_cdf);
                    TX_TYPE_INTER_INV_SET2[sym]
                }
                _ => {
                    let sym = self
                        .sd
                        .read_symbol(&mut self.cdf.inter_tx_type_set3_cdf[sqr]);
                    TX_TYPE_INTER_INV_SET3[sym]
                }
            };
        }
        let intra_dir = if self.use_filter_intra {
            FILTER_INTRA_MODE_TO_INTRA_DIR[self.filter_intra_mode]
        } else {
            self.y_mode
        };
        // set 1 is only chosen for 4x4 and 8x8 squares, set 2 for at most 16x16 (get_tx_set).
        if set == TX_SET_INTRA_1 {
            let sym = self
                .sd
                .read_symbol(&mut self.cdf.intra_tx_type_set1_cdf[sqr][intra_dir]);
            TX_TYPE_INTRA_INV_SET1[sym]
        } else {
            let sym = self
                .sd
                .read_symbol(&mut self.cdf.intra_tx_type_set2_cdf[sqr][intra_dir]);
            TX_TYPE_INTRA_INV_SET2[sym]
        }
    }

    /// `TxTypes[ y4 + j ][ x4 + i ] = tx_type` for the transform block (spec 5.11.39 and
    /// 5.11.47); positions outside the frame are not stored.
    fn set_tx_types(&mut self, x4: usize, y4: usize, tx_sz: usize, tx_type: usize) {
        let mi_cols = self.fs.mi_cols;
        let rows = (TX_HEIGHT[tx_sz] >> 2).min(self.fs.mi_rows.saturating_sub(y4));
        let cols = (TX_WIDTH[tx_sz] >> 2).min(mi_cols.saturating_sub(x4));
        fill_rows(
            &mut self.fs.tx_types,
            mi_cols,
            y4,
            x4,
            rows,
            cols,
            tx_type as u8,
        );
    }

    /// The `ctx` of `all_zero` (spec 9.3).
    fn all_zero_ctx(&self, plane: usize, x4: usize, y4: usize, tx_sz: usize) -> usize {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let max_x4 = self.fs.mi_cols >> sub_x;
        let max_y4 = self.fs.mi_rows >> sub_y;
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let w4 = w >> 2;
        let h4 = h >> 2;
        let bsize = self.get_plane_residual_size(self.mi_size, plane);
        let bw = BLOCK_WIDTH[bsize];
        let bh = BLOCK_HEIGHT[bsize];
        let above_level = &self.above_level_context[plane];
        let left_level = &self.left_level_context[plane];
        if plane == 0 {
            let mut top = 0u8;
            let mut left = 0u8;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(above_level[x4 + k]);
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(left_level[y4 + k]);
                }
            }
            if bw == w && bh == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let above_dc = &self.above_dc_context[plane];
            let left_dc = &self.left_dc_context[plane];
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    above |= above_level[x4 + i] | above_dc[x4 + i];
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    left |= left_level[y4 + i] | left_dc[y4 + i];
                }
            }
            let mut ctx = usize::from(above != 0) + usize::from(left != 0) + 7;
            if bw * bh > w * h {
                ctx += 3;
            }
            ctx
        }
    }

    /// The `ctx` of `dc_sign` (spec 9.3).
    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize, w4: usize, h4: usize) -> usize {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let max_x4 = self.fs.mi_cols >> sub_x;
        let max_y4 = self.fs.mi_rows >> sub_y;
        let mut dc_sign = 0i32;
        let mut count = |sign: u8| {
            if sign == 1 {
                dc_sign -= 1;
            } else if sign == 2 {
                dc_sign += 1;
            }
        };
        for k in 0..w4 {
            if x4 + k < max_x4 {
                count(self.above_dc_context[plane][x4 + k]);
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                count(self.left_dc_context[plane][y4 + k]);
            }
        }
        match dc_sign {
            ..0 => 1,
            0 => 0,
            _ => 2,
        }
    }

    /// Spec 5.11.39 `coeffs( plane, startX, startY, txSz )`: fills `self.quant` (the spec's
    /// `Quant[ ]`) and the above / left level and dc contexts. Returns `eob` and `PlaneTxType`.
    ///
    /// # Errors
    /// `Invalid` when `golomb_length_bit` is 0 at `length` 20 (spec 6.10.35 conformance).
    fn coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> Result<(usize, usize), Error> {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = TX_WIDTH[tx_sz] >> 2;
        let h4 = TX_HEIGHT[tx_sz] >> 2;
        let tx_sz_ctx = (TX_SIZE_SQR[tx_sz] + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1;
        let ptype = usize::from(plane > 0);

        let mut eob = 0usize;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        let mut plane_tx_type = DCT_DCT;

        let ctx = self.all_zero_ctx(plane, x4, y4, tx_sz);
        let all_zero = self
            .sd
            .read_symbol(&mut self.cdf.txb_skip_cdf[tx_sz_ctx][ctx])
            != 0;
        if all_zero {
            if plane == 0 && self.is_inter {
                self.set_tx_types(x4, y4, tx_sz, DCT_DCT);
            }
        } else {
            let luma_tx_type = if plane == 0 {
                let tx_type = self.transform_type(tx_sz);
                if self.is_inter {
                    self.set_tx_types(x4, y4, tx_sz, tx_type);
                }
                tx_type
            } else {
                DCT_DCT
            };
            plane_tx_type = self.compute_tx_type(plane, tx_sz, x4, y4, luma_tx_type);
            let tx_class = get_tx_class(plane_tx_type);
            let scan = get_scan(tx_sz, plane_tx_type);

            let eob_multisize = TX_WIDTH_LOG2[tx_sz].min(5) + TX_HEIGHT_LOG2[tx_sz].min(5) - 4;
            let ctx = usize::from(tx_class != TX_CLASS_2D);
            let eob_pt = 1 + match eob_multisize {
                0 => self.sd.read_symbol(&mut self.cdf.eob_pt_16_cdf[ptype][ctx]),
                1 => self.sd.read_symbol(&mut self.cdf.eob_pt_32_cdf[ptype][ctx]),
                2 => self.sd.read_symbol(&mut self.cdf.eob_pt_64_cdf[ptype][ctx]),
                3 => self
                    .sd
                    .read_symbol(&mut self.cdf.eob_pt_128_cdf[ptype][ctx]),
                4 => self
                    .sd
                    .read_symbol(&mut self.cdf.eob_pt_256_cdf[ptype][ctx]),
                5 => self.sd.read_symbol(&mut self.cdf.eob_pt_512_cdf[ptype]),
                _ => self.sd.read_symbol(&mut self.cdf.eob_pt_1024_cdf[ptype]),
            };
            eob = if eob_pt < 2 {
                eob_pt
            } else {
                (1 << (eob_pt - 2)) + 1
            };
            if eob_pt >= 3 {
                let eob_extra = self
                    .sd
                    .read_symbol(&mut self.cdf.eob_extra_cdf[tx_sz_ctx][ptype][eob_pt - 3])
                    != 0;
                if eob_extra {
                    eob += 1 << (eob_pt - 3);
                }
                for i in 1..(eob_pt - 2) {
                    if self.sd.read_literal(1) != 0 {
                        eob += 1 << (eob_pt - 2 - 1 - i);
                    }
                }
            }

            // Quant[ ] is only read by the reconstruction of a block with eob > 0, so it is
            // cleared only here.
            let seg_eob = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
                512
            } else {
                (TX_WIDTH[tx_sz] * TX_HEIGHT[tx_sz]).min(QUANT_LEN)
            };
            self.quant[..seg_eob].fill(0);
            // The coded area, as `Quant[ ]` is laid out, with TX_PAD zero columns and rows.
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let bwl = TX_WIDTH_LOG2[adj];
            let stride = (1 << bwl) + TX_PAD;
            self.levels[..(TX_HEIGHT[adj] + TX_PAD) * stride].fill(0);

            for c in (0..eob).rev() {
                let pos = usize::from(*scan.get(c).ok_or(Error::Invalid("eob beyond the scan"))?);
                let row = pos >> bwl;
                let col = pos - (row << bwl);
                let mut level = if c == eob - 1 {
                    let ctx = coeff_base_eob_ctx(tx_sz, c);
                    self.sd
                        .read_symbol(&mut self.cdf.coeff_base_eob_cdf[tx_sz_ctx][ptype][ctx])
                        + 1
                } else {
                    let ctx = coeff_base_ctx(&self.levels[..], stride, tx_sz, tx_class, row, col);
                    self.sd
                        .read_symbol(&mut self.cdf.coeff_base_cdf[tx_sz_ctx][ptype][ctx])
                };
                if level > NUM_BASE_LEVELS {
                    let ctx = coeff_br_ctx(&self.levels[..], stride, tx_class, row, col);
                    let br_cdf: &mut [u16] =
                        &mut self.cdf.coeff_br_cdf[tx_sz_ctx.min(TX_32X32)][ptype][ctx];
                    for _ in 0..COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) {
                        let coeff_br = self.sd.read_symbol(br_cdf);
                        level += coeff_br;
                        if coeff_br < BR_CDF_SIZE - 1 {
                            break;
                        }
                    }
                }
                // At most NUM_BASE_LEVELS + 1 + COEFF_BASE_RANGE = 15.
                self.levels[row * stride + col] = level as u8;
            }

            for c in 0..eob {
                let pos = usize::from(scan[c]);
                let mut q = i32::from(self.levels[pos + (pos >> bwl) * TX_PAD]);
                let mut sign = false;
                if q != 0 {
                    if c == 0 {
                        let ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
                        sign = self.sd.read_symbol(&mut self.cdf.dc_sign_cdf[ptype][ctx]) != 0;
                    } else {
                        sign = self.sd.read_literal(1) != 0;
                    }
                }
                if q > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
                    let mut length = 0;
                    loop {
                        length += 1;
                        if self.sd.read_literal(1) != 0 {
                            break;
                        }
                        if length == 20 {
                            return Err(Error::Invalid("golomb_length_bit"));
                        }
                    }
                    let mut x = 1i32;
                    for _ in 0..length - 1 {
                        x = (x << 1) | self.sd.read_literal(1) as i32;
                    }
                    q = x + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as i32;
                }
                if pos == 0 && q > 0 {
                    dc_category = if sign { 1 } else { 2 };
                }
                q &= 0xFFFFF;
                cul_level += q as u32;
                self.quant[pos] = if sign { -q } else { q };
            }
            cul_level = cul_level.min(63);
        }

        let cul_level = cul_level as u8;
        set_ctx(&mut self.above_level_context[plane], x4, w4, cul_level);
        set_ctx(&mut self.above_dc_context[plane], x4, w4, dc_category);
        set_ctx(&mut self.left_level_context[plane], y4, h4, cul_level);
        set_ctx(&mut self.left_dc_context[plane], y4, h4, dc_category);
        Ok((eob, plane_tx_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The bounds-checked contexts that the padded-levels ones replaced: the oracle.
    /// `get_coeff_base_ctx( txSz, plane, x4, y4, pos, c, 0 )` (spec 9.3, `coeff_base`) for a
    /// transform class `tx_class`. `quant` holds the levels decoded so far.
    fn coeff_base_ctx_oracle(quant: &[i32], tx_sz: usize, tx_class: usize, pos: usize) -> usize {
        let adj = ADJUSTED_TX_SIZE[tx_sz];
        let bwl = TX_WIDTH_LOG2[adj];
        let width = 1usize << bwl;
        let height = TX_HEIGHT[adj];
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0usize;
        for off in &SIG_REF_DIFF_OFFSET[tx_class] {
            let ref_row = row + off[0];
            let ref_col = col + off[1];
            if ref_row < height && ref_col < width {
                mag += quant[(ref_row << bwl) + ref_col].unsigned_abs().min(3) as usize;
            }
        }
        let ctx = ((mag + 1) >> 1).min(4);
        if tx_class == TX_CLASS_2D {
            if row == 0 && col == 0 {
                return 0;
            }
            return ctx + usize::from(COEFF_BASE_CTX_OFFSET[tx_sz][row.min(4)][col.min(4)]);
        }
        let idx = if tx_class == TX_CLASS_VERT { row } else { col };
        ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
    }

    /// The `ctx` of `coeff_br` (spec 9.3).
    fn coeff_br_ctx_oracle(quant: &[i32], tx_sz: usize, tx_class: usize, pos: usize) -> usize {
        let adj = ADJUSTED_TX_SIZE[tx_sz];
        let bwl = TX_WIDTH_LOG2[adj];
        let txw = TX_WIDTH[adj];
        let txh = TX_HEIGHT[adj];
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0usize;
        for off in &MAG_REF_OFFSET_WITH_TX_CLASS[tx_class] {
            let ref_row = row + off[0];
            let ref_col = col + off[1];
            if ref_row < txh && ref_col < (1 << bwl) {
                mag += quant[ref_row * txw + ref_col]
                    .min((COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as i32)
                    as usize;
            }
        }
        let mag = ((mag + 1) >> 1).min(6);
        if pos == 0 {
            mag
        } else if tx_class == TX_CLASS_2D {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        } else if tx_class == TX_CLASS_HORIZ {
            if col == 0 { mag + 7 } else { mag + 14 }
        } else if row == 0 {
            mag + 7
        } else {
            mag + 14
        }
    }

    /// The padded-levels contexts equal these bounds-checked ones on random levels (0..=15),
    /// for every transform size and class and every position.
    #[test]
    fn padded_contexts_match_the_oracle() {
        let mut rng = crate::symbol::XorShift(11);
        for tx_sz in 0..TX_SIZES_ALL {
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let bwl = TX_WIDTH_LOG2[adj];
            let (w, h) = (1usize << bwl, TX_HEIGHT[adj]);
            let stride = w + TX_PAD;
            for tx_class in [TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT] {
                for _ in 0..4 {
                    let mut q = [0i32; QUANT_LEN];
                    let mut levels = [0u8; LEVELS_LEN];
                    for pos in 0..w * h {
                        let v = match rng.next_u32() % 4 {
                            0 => 0,
                            1 => 15,
                            _ => rng.next_u32() % 16,
                        };
                        q[pos] = v as i32;
                        levels[(pos >> bwl) * stride + (pos & (w - 1))] = v as u8;
                    }
                    for pos in 0..w * h {
                        let (row, col) = (pos >> bwl, pos & (w - 1));
                        assert_eq!(
                            coeff_base_ctx(&levels, stride, tx_sz, tx_class, row, col),
                            coeff_base_ctx_oracle(&q, tx_sz, tx_class, pos),
                            "tx_sz {tx_sz} class {tx_class} pos {pos}"
                        );
                        assert_eq!(
                            coeff_br_ctx(&levels, stride, tx_class, row, col),
                            coeff_br_ctx_oracle(&q, tx_sz, tx_class, pos),
                            "tx_sz {tx_sz} class {tx_class} pos {pos}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tx_class_of_the_1d_types() {
        assert_eq!(get_tx_class(DCT_DCT), TX_CLASS_2D);
        assert_eq!(get_tx_class(IDTX), TX_CLASS_2D);
        assert_eq!(get_tx_class(V_DCT), TX_CLASS_VERT);
        assert_eq!(get_tx_class(V_FLIPADST), TX_CLASS_VERT);
        assert_eq!(get_tx_class(H_ADST), TX_CLASS_HORIZ);
    }

    #[test]
    fn tx_type_sets_of_intra_blocks() {
        assert!(is_tx_type_in_set(TX_SET_INTRA_1, IDTX));
        assert!(is_tx_type_in_set(TX_SET_INTRA_1, H_DCT));
        assert!(!is_tx_type_in_set(TX_SET_INTRA_2, H_DCT));
        assert!(is_tx_type_in_set(TX_SET_INTRA_2, ADST_ADST));
        assert!(!is_tx_type_in_set(TX_SET_DCTONLY, ADST_DCT));
    }

    #[test]
    fn coeff_base_ctx_4x4() {
        let mut q = [0i32; QUANT_LEN];
        // DC position is always context 0.
        q[1] = 3;
        assert_eq!(coeff_base_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 0), 0);
        // pos 1 (row 0, col 1): neighbours (0,2) (1,1)... all zero, offset 1.
        assert_eq!(coeff_base_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 1), 1);
        // pos 5 (row 1, col 1): neighbours 6 and 9 hold 3 and 1: mag 4, ctx 2, offset 6.
        q[6] = 3;
        q[9] = 1;
        assert_eq!(coeff_base_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 5), 8);
        // Horizontal class, pos 1 (col 1): offset SIG_COEF_CONTEXTS_2D + 5 plus the (0,2) etc.
        // neighbours: (0,2) = q[2] = 0, (1,1) = q[5] = 0, (0,3) .. all zero.
        assert_eq!(
            coeff_base_ctx_oracle(&q, TX_4X4, TX_CLASS_HORIZ, 1),
            SIG_COEF_CONTEXTS_2D + 5
        );
    }

    #[test]
    fn coeff_br_ctx_4x4() {
        let mut q = [0i32; QUANT_LEN];
        q[1] = 40;
        q[4] = 2;
        // DC: mag = min(40, 15) + 2 = 17, (17 + 1) >> 1 = 9, capped at 6.
        assert_eq!(coeff_br_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 0), 6);
        // pos 1 is in the 2x2 corner: mag from (1,1), (2,0)... all zero except none: 0 + 7.
        assert_eq!(coeff_br_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 1), 7);
        // pos 10 (row 2, col 2) outside the corner: 14.
        assert_eq!(coeff_br_ctx_oracle(&q, TX_4X4, TX_CLASS_2D, 10), 14);
    }

    #[test]
    fn coeff_base_eob_ctx_4x4() {
        assert_eq!(coeff_base_eob_ctx(TX_4X4, 0), 0);
        assert_eq!(coeff_base_eob_ctx(TX_4X4, 2), 1);
        assert_eq!(coeff_base_eob_ctx(TX_4X4, 3), 2);
        assert_eq!(coeff_base_eob_ctx(TX_4X4, 4), 2);
        assert_eq!(coeff_base_eob_ctx(TX_4X4, 5), 3);
        // 64x64 uses the 32x32 coded area: 1024 / 8 = 128.
        assert_eq!(coeff_base_eob_ctx(TX_64X64, 128), 1);
        assert_eq!(coeff_base_eob_ctx(TX_64X64, 129), 2);
    }

    #[test]
    fn set_ctx_clamps() {
        let mut v = vec![7u8; 4];
        set_ctx(&mut v, 2, 8, 0);
        assert_eq!(v, [7, 7, 0, 0]);
        set_ctx(&mut v, 9, 2, 1);
        assert_eq!(v, [7, 7, 0, 0]);
    }
}
