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
//! The padded levels array behind the `coeff_base` and `coeff_br` contexts and the linked list
//! of nonzero positions that the sign pass walks are ported from
//! dav1d's `src/recon_tmpl.c` (`decode_coefs`; Copyright 2018-2019, VideoLAN and dav1d authors,
//! Copyright 2018-2019, Two Orioles, LLC; BSD-2-Clause, see NOTICE and docs/dependencies.md).

use crate::consts::*;
use crate::error::Error;
use crate::predict::{IntraEdges, IntraParams, predict_cfl, predict_intra_simd, predict_palette};
use crate::quant::{ac_q, dc_q};
use crate::recon::{Dequant, ReconParams, transform_dq};
use crate::scan::get_scan;
use crate::symbol::SymbolDecoder;
use crate::tables::{
    ADJUSTED_TX_SIZE, BLOCK_HEIGHT, BLOCK_WIDTH, MAX_TX_SIZE_RECT, MODE_TO_TXFM,
    NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, SIG_REF_DIFF_OFFSET, SUBSAMPLED_SIZE, TX_HEIGHT,
    TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
};
use crate::tile::{TileDecoder, fill_cols, fill_rows};
use crate::transform::{DQ_LEN, DQ_STRIDE};

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

/// `Mag_Ref_Offset_With_Tx_Class` (spec 9.3, `coeff_br`): `[ row, col ]` offsets. The decoder
/// reads the first three `Sig_Ref_Diff_Offset` entries instead, which are the same (tested).
#[cfg(test)]
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
/// (spec 9.3), so the context reads of `LevelCtx::base_ctx` and `LevelCtx::br_ctx` stay inside
/// the padded block and read zero where the spec skips a position outside the block.
const TX_PAD: usize = 4;

/// The padded levels of the largest coded area (32x32, `(32 + TX_PAD)^2` entries), rounded up
/// to a power of two: the coefficient loops index it masked with `LEVELS_LEN - 1` (no bounds
/// check; every index is below `(32 + TX_PAD)^2`, so the mask changes nothing). `Quant[ ]`
/// indices (scan positions, below 1024) are masked the same way.
pub(crate) const LEVELS_LEN: usize = 2048;
const _: () = assert!(LEVELS_LEN.is_power_of_two() && LEVELS_LEN >= (32 + TX_PAD) * (32 + TX_PAD));
const _: () = assert!(QUANT_LEN.is_power_of_two());

/// `levels[ i ]`, `i` masked into the array (see [`LEVELS_LEN`]).
#[inline(always)]
fn lv(i: usize) -> usize {
    i & (LEVELS_LEN - 1)
}

/// `quant[ pos ]`, `pos` masked into the array (see [`LEVELS_LEN`]).
#[inline(always)]
fn qp(pos: usize) -> usize {
    pos & (QUANT_LEN - 1)
}

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

/// The per-block setup of the `coeff_base` and `coeff_br` contexts (spec 9.3): the layout of
/// the padded levels (see [`TX_PAD`]) and the neighbour offsets of the transform class as flat
/// indices, so the coefficient loop does no table lookups for them.
struct LevelCtx {
    /// `bwl`: log2 of the coded width.
    bwl: usize,
    /// The row stride of the padded levels.
    stride: usize,
    /// `Sig_Ref_Diff_Offset[ txClass ]` as offsets into the levels.
    base_offs: [usize; 5],
    /// `Coeff_Base_Ctx_Offset[ txSz ]`.
    base_ctx_offset: [[u8; 5]; 5],
}

impl LevelCtx {
    fn new(tx_sz: usize, tx_class: usize) -> LevelCtx {
        let bwl = TX_WIDTH_LOG2[ADJUSTED_TX_SIZE[tx_sz]];
        let stride = (1 << bwl) + TX_PAD;
        LevelCtx {
            bwl,
            stride,
            base_offs: SIG_REF_DIFF_OFFSET[tx_class].map(|o| o[0] * stride + o[1]),
            base_ctx_offset: COEFF_BASE_CTX_OFFSET[tx_sz],
        }
    }

    /// The sums of the packed levels (see [`pack_level`]) of the five `Sig_Ref_Diff_Offset`
    /// neighbours of (`row`, `col`) and of the first three of them. The three
    /// `Mag_Ref_Offset_With_Tx_Class` neighbours of `coeff_br` are those first three for every
    /// class (spec 9.3), so one set of loads serves [`LevelCtx::base_ctx`] and
    /// [`LevelCtx::br_ctx`].
    #[inline(always)]
    fn neighbours(&self, levels: &[u8; LEVELS_LEN], row: usize, col: usize) -> (usize, usize) {
        let i = row * self.stride + col;
        let at = |k: usize| usize::from(levels[lv(i + self.base_offs[k])]);
        let s3 = at(0) + at(1) + at(2);
        (s3 + at(3) + at(4), s3)
    }

    /// `get_coeff_base_ctx( txSz, plane, x4, y4, pos, c, 0 )` (spec 9.3, `coeff_base`) at
    /// (`row`, `col`) for the transform class `CLASS`, except that the DC position of a 2D
    /// class (context 0, no neighbours) is the caller's. `s5` is the first sum of
    /// [`LevelCtx::neighbours`].
    ///
    /// The spec's `Min( 3, level )` sum S and the plain sum L of the five neighbours give
    /// `s5 = 64 S + L`. When S <= 7, at most two neighbours are 3 or more, so L <= 36 < 64 and
    /// `( s5 + 64 ) >> 7 == ( S + 1 ) >> 1`; when S >= 8, `s5 > 512` and the result is 4.
    #[inline(always)]
    fn base_ctx<const CLASS: usize>(&self, s5: usize, row: usize, col: usize) -> usize {
        let ctx = if s5 > 512 { 4 } else { (s5 + 64) >> 7 };
        if CLASS == TX_CLASS_2D {
            return ctx + usize::from(self.base_ctx_offset[row.min(4)][col.min(4)]);
        }
        let idx = if CLASS == TX_CLASS_VERT { row } else { col };
        ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
    }

    /// The `ctx` of `coeff_br` (spec 9.3) at (`row`, `col`). `s3` is the second sum of
    /// [`LevelCtx::neighbours`]: the levels (at most 15, so the spec's `Min( 15, level )` is
    /// the identity) are the low 6 bits of the packed values, and their sum is at most 45, so
    /// it does not carry into the `Min( 3, level )` bits.
    #[inline(always)]
    fn br_ctx<const CLASS: usize>(&self, s3: usize, row: usize, col: usize) -> usize {
        let mag = (((s3 & 63) + 1) >> 1).min(6);
        if row == 0 && col == 0 {
            mag
        } else if CLASS == TX_CLASS_2D {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        } else if CLASS == TX_CLASS_HORIZ {
            if col == 0 { mag + 7 } else { mag + 14 }
        } else if row == 0 {
            mag + 7
        } else {
            mag + 14
        }
    }
}

/// A level (at most 15) as stored in the padded levels: `level | Min( 3, level ) << 6` (dav1d's
/// way to get the sum of the `Min( 3, level )` of the base context with the same loads as the
/// plain sum of the br context).
#[inline(always)]
fn pack_level(level: usize) -> u8 {
    (level | (level.min(3) << 6)) as u8
}

/// Reads the `coeff_base_eob`, `coeff_base` and `coeff_br` levels of the scan positions
/// `scan` (`eob` of them) from the last to the first, as the first loop of spec 5.11.39 does,
/// for the transform class `CLASS`. Stores each level in `levels` (padded) and, for each
/// nonzero one, `Quant[ pos ] = level << 11 | next` where `next` is the position of the next
/// nonzero level in scan order (0 ends the list); the position returned is the head of that
/// list (dav1d's way to visit only the nonzero positions in the sign pass). No node links to
/// position 0, which is `scan[ 0 ]` for every scan (tested), so 0 is free as the end mark.
/// The eob position is always nonzero, so the list is never empty.
#[inline(always)]
fn read_levels<const CLASS: usize>(
    sd: &mut SymbolDecoder<'_>,
    lc: &LevelCtx,
    scan: &[u16],
    eob_cdf: &mut [u16; 4],
    base_cdfs: &mut [[u16; 5]; 42],
    br_cdfs: &mut [[u16; 5]; 21],
    levels: &mut [u8; LEVELS_LEN],
    quant: &mut [i32; QUANT_LEN],
) -> usize {
    // The brs of a level above NUM_BASE_LEVELS, from the neighbour sum `s3`.
    let mut read_br = |sd: &mut SymbolDecoder<'_>, s3, row, col, mut level: usize| {
        let br_cdf = &mut br_cdfs[lc.br_ctx::<CLASS>(s3, row, col)];
        for _ in 0..COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) {
            let coeff_br = sd.read_cdf(br_cdf);
            level += coeff_br;
            if coeff_br < BR_CDF_SIZE - 1 {
                break;
            }
        }
        level
    };
    let last = scan.len() - 1;
    // The eob position: its level is at least 1.
    let mut next = {
        let pos = usize::from(scan[last]);
        let row = pos >> lc.bwl;
        let col = pos - (row << lc.bwl);
        let mut level = sd.read_cdf(eob_cdf) + 1;
        if level > NUM_BASE_LEVELS {
            level = read_br(&mut *sd, lc.neighbours(levels, row, col).1, row, col, level);
        }
        levels[lv(row * lc.stride + col)] = pack_level(level);
        quant[qp(pos)] = (level as i32) << 11;
        pos
    };
    for &pos in scan.get(1..last).unwrap_or_default().iter().rev() {
        let pos = usize::from(pos);
        let row = pos >> lc.bwl;
        let col = pos - (row << lc.bwl);
        let (s5, s3) = lc.neighbours(levels, row, col);
        let mut level = sd.read_cdf(&mut base_cdfs[lc.base_ctx::<CLASS>(s5, row, col)]);
        if level > NUM_BASE_LEVELS {
            level = read_br(&mut *sd, s3, row, col, level);
        }
        levels[lv(row * lc.stride + col)] = pack_level(level);
        if level != 0 {
            quant[qp(pos)] = ((level as i32) << 11) | next as i32;
            next = pos;
        }
    }
    if last > 0 {
        // DC: the context of a 2D class is 0, the other classes use the neighbours.
        let (s5, s3) = lc.neighbours(levels, 0, 0);
        let ctx = if CLASS == TX_CLASS_2D {
            0
        } else {
            lc.base_ctx::<CLASS>(s5, 0, 0)
        };
        let mut level = sd.read_cdf(&mut base_cdfs[ctx]);
        if level > NUM_BASE_LEVELS {
            level = read_br(&mut *sd, s3, 0, 0, level);
        }
        levels[0] = pack_level(level);
        if level != 0 {
            quant[0] = ((level as i32) << 11) | next as i32;
            next = 0;
        }
    }
    next
}

/// The entries `start..start + n` of an above or left context array that lie inside the plane
/// (index below `max`): the positions the spec's context loops visit.
fn ctx_run(v: &[u8], start: usize, n: usize, max: usize) -> &[u8] {
    v.get(start..start + n.min(max.saturating_sub(start)))
        .unwrap_or_default()
}

/// The `ctx` of `dc_sign` (spec 9.3) from the in-plane runs of `AboveDcContext` and
/// `LeftDcContext`.
fn dc_sign_ctx_of(above: &[u8], left: &[u8]) -> usize {
    let dc_sign: i32 = above
        .iter()
        .chain(left)
        .map(|&s| i32::from(s == 2) - i32::from(s == 1))
        .sum();
    match dc_sign {
        ..0 => 1,
        0 => 0,
        _ => 2,
    }
}

/// Sets `v[ start..start + n ]` to `val`, clamped to the length of `v`.
fn set_ctx(v: &mut [u8], start: usize, n: usize, val: u8) {
    let end = (start + n).min(v.len());
    if start < end {
        fill_cols(&mut v[start..end], val);
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
            let use_filter_intra = plane == 0 && self.use_filter_intra;
            // Only the directional modes read haveAboveRt and haveBelowLft (through the edge
            // entries past w and h) and the filter type, so the others skip computing them.
            let dir = !use_filter_intra && (V_PRED..=D67_PRED).contains(&mode);
            let edges = IntraEdges {
                have_left: avail_l || x > 0,
                have_above: avail_u || y > 0,
                have_above_right: dir
                    && self.block_decoded_at(
                        plane,
                        sub_block_mi_row - 1,
                        sub_block_mi_col + step_x as i32,
                    ),
                have_below_left: dir
                    && self.block_decoded_at(
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
                use_filter_intra,
                filter_intra_mode: self.filter_intra_mode,
                enable_intra_edge_filter: self.seq.enable_intra_edge_filter,
                smooth_neighbor: dir
                    && self.seq.enable_intra_edge_filter
                    && self.get_filter_type(plane),
                max_x: max_x - 1,
                max_y: max_y - 1,
                bit_depth,
            };
            let fp = &mut self.fs.curr_frame[plane];
            predict_intra_simd(
                crate::simd::best(),
                &mut fp.px,
                fp.stride,
                start_x,
                start_y,
                edges,
                &params,
                &mut self.itx.edges,
            );
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
                    &mut self.itx.ac,
                );
            }
        }
        if plane == 0 && !self.is_inter {
            s.max_luma_w = start_x + step_x * 4;
            s.max_luma_h = start_y + step_y * 4;
        }

        if !self.skip {
            let (eob, plane_tx_type, region) = self.coeffs(plane, start_x, start_y, tx_sz)?;
            if eob > 0 {
                self.reconstruct_block(plane, start_x, start_y, tx_sz, plane_tx_type, region);
            }
        }

        let mi_cols = self.fs.mi_cols;
        let (r0, c0) = (row >> sub_y, col >> sub_x);
        let rows = step_y.min(self.fs.mi_rows.saturating_sub(r0));
        let cols = step_x.min(mi_cols.saturating_sub(c0));
        fill_rows(
            &mut self.fs.loopfilter_tx_sizes[plane],
            mi_cols,
            r0,
            c0,
            rows,
            cols,
            tx_sz as u8,
        );
        // BlockDecoded[ plane ][ y ][ x ] is stored at [ y + 1 ][ x + 1 ] (set_block_decoded);
        // the subblock position is not negative.
        let (bd_row, bd_col) = (sub_block_mi_row as usize + 1, sub_block_mi_col as usize + 1);
        for r in self.block_decoded[plane]
            .iter_mut()
            .skip(bd_row)
            .take(step_y)
        {
            let end = (bd_col + step_x).min(r.len());
            fill_cols(&mut r[bd_col.min(end)..end], true);
        }
        Ok(())
    }

    /// The inverse transform and sample update of the reconstruction process (spec 7.12.3) on
    /// the `Dequant` values that `coeffs` wrote into `self.itx` for the coded `region` (rows,
    /// columns) it returned. That region of `self.itx` is all zero again afterwards, as
    /// `coeffs` expects for the next transform block.
    fn reconstruct_block(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
        plane_tx_type: usize,
        region: (usize, usize),
    ) {
        let p = self.recon_params(plane, tx_sz, plane_tx_type);
        let fp = &mut self.fs.curr_frame[plane];
        transform_dq(
            &mut fp.px,
            fp.stride,
            start_x,
            start_y,
            &p,
            region,
            &mut self.itx,
            crate::simd::best(),
        );
    }

    /// The [`ReconParams`] of a transform block: `get_dc_quant( plane )` and
    /// `get_ac_quant( plane )` (spec 7.12.2) and the quantizer matrix level.
    fn recon_params(&self, plane: usize, tx_sz: usize, plane_tx_type: usize) -> ReconParams {
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
        ReconParams {
            tx_sz,
            tx_type: plane_tx_type,
            plane,
            dc_q: dc_q(qindex + i32::from(dc_delta), bit_depth),
            ac_q: ac_q(qindex + i32::from(ac_delta), bit_depth),
            qm_level,
            lossless: self.lossless,
            bit_depth,
        }
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
                    let sym = self.sd.read_cdf(&mut self.cdf.inter_tx_type_set1_cdf[sqr]);
                    TX_TYPE_INTER_INV_SET1[sym]
                }
                TX_SET_INTER_2 => {
                    let sym = self.sd.read_cdf(&mut self.cdf.inter_tx_type_set2_cdf);
                    TX_TYPE_INTER_INV_SET2[sym]
                }
                _ => {
                    let sym = self.sd.read_cdf(&mut self.cdf.inter_tx_type_set3_cdf[sqr]);
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
                .read_cdf(&mut self.cdf.intra_tx_type_set1_cdf[sqr][intra_dir]);
            TX_TYPE_INTRA_INV_SET1[sym]
        } else {
            let sym = self
                .sd
                .read_cdf(&mut self.cdf.intra_tx_type_set2_cdf[sqr][intra_dir]);
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
        let above_level = ctx_run(&self.above_level_context[plane], x4, w4, max_x4);
        let left_level = ctx_run(&self.left_level_context[plane], y4, h4, max_y4);
        if plane == 0 {
            let top = above_level.iter().copied().max().unwrap_or(0);
            let left = left_level.iter().copied().max().unwrap_or(0);
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
            let above_dc = ctx_run(&self.above_dc_context[plane], x4, w4, max_x4);
            let left_dc = ctx_run(&self.left_dc_context[plane], y4, h4, max_y4);
            let or = |a: &[u8], b: &[u8]| a.iter().chain(b).fold(0u8, |acc, &v| acc | v);
            let above = or(above_level, above_dc);
            let left = or(left_level, left_dc);
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
        dc_sign_ctx_of(
            ctx_run(&self.above_dc_context[plane], x4, w4, max_x4),
            ctx_run(&self.left_dc_context[plane], y4, h4, max_y4),
        )
    }

    /// Spec 5.11.39 `coeffs( plane, startX, startY, txSz )`: writes `Dequant` (spec 7.12.3) of
    /// each nonzero coefficient of the spec's `Quant[ ]` into `self.itx` and fills the above /
    /// left level and dc contexts. Returns `eob`, `PlaneTxType` and the coded region of
    /// `Quant[ ]` (rows, columns; see `recon::reconstruct_region`).
    ///
    /// `self.quant` (the levels and the list of nonzero positions here), `self.levels` and the
    /// `Dequant` values of `self.itx` are all zero on entry (they start zeroed and only ever
    /// hold one transform block): the levels and `self.quant` are cleared here as the sign loop
    /// reads them, and the `Dequant` region by `reconstruct_block` (called for every block
    /// with eob > 0), so no block clears the whole arrays.
    ///
    /// # Errors
    /// `Invalid` when `golomb_length_bit` is 0 at `length` 20 (spec 6.10.35 conformance).
    fn coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> Result<(usize, usize, (usize, usize)), Error> {
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
        // The coded region: the last nonzero row and column of Quant[ ], plus 1 (at least 1).
        let (mut rows, mut cols) = (1, 1);

        let ctx = self.all_zero_ctx(plane, x4, y4, tx_sz);
        let all_zero = self.sd.read_cdf(&mut self.cdf.txb_skip_cdf[tx_sz_ctx][ctx]) != 0;
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
                0 => self.sd.read_cdf(&mut self.cdf.eob_pt_16_cdf[ptype][ctx]),
                1 => self.sd.read_cdf(&mut self.cdf.eob_pt_32_cdf[ptype][ctx]),
                2 => self.sd.read_cdf(&mut self.cdf.eob_pt_64_cdf[ptype][ctx]),
                3 => self.sd.read_cdf(&mut self.cdf.eob_pt_128_cdf[ptype][ctx]),
                4 => self.sd.read_cdf(&mut self.cdf.eob_pt_256_cdf[ptype][ctx]),
                5 => self.sd.read_cdf(&mut self.cdf.eob_pt_512_cdf[ptype]),
                _ => self.sd.read_cdf(&mut self.cdf.eob_pt_1024_cdf[ptype]),
            };
            eob = if eob_pt < 2 {
                eob_pt
            } else {
                (1 << (eob_pt - 2)) + 1
            };
            if eob_pt >= 3 {
                let eob_extra = self
                    .sd
                    .read_cdf(&mut self.cdf.eob_extra_cdf[tx_sz_ctx][ptype][eob_pt - 3])
                    != 0;
                if eob_extra {
                    eob += 1 << (eob_pt - 3);
                }
                for i in 1..(eob_pt - 2) {
                    if self.sd.read_bool() {
                        eob += 1 << (eob_pt - 2 - 1 - i);
                    }
                }
            }

            // The first symbol read below is at scan index eob - 1, so refusing a too long eob
            // here reads the same symbols as refusing it there.
            let scan = scan
                .get(..eob)
                .ok_or(Error::Invalid("eob beyond the scan"))?;
            // The per-tx-size setup, hoisted out of the coefficient loops.
            let lc = LevelCtx::new(tx_sz, tx_class);
            let eob_cdf = &mut self.cdf.coeff_base_eob_cdf[tx_sz_ctx][ptype]
                [coeff_base_eob_ctx(tx_sz, eob - 1)];
            let base_cdfs = &mut self.cdf.coeff_base_cdf[tx_sz_ctx][ptype];
            let br_cdfs = &mut self.cdf.coeff_br_cdf[tx_sz_ctx.min(TX_32X32)][ptype];
            let (sd, levels, quant) = (&mut self.sd, &mut *self.levels, &mut *self.quant);
            let mut pos = match tx_class {
                TX_CLASS_2D => read_levels::<TX_CLASS_2D>(
                    sd, &lc, scan, eob_cdf, base_cdfs, br_cdfs, levels, quant,
                ),
                TX_CLASS_HORIZ => read_levels::<TX_CLASS_HORIZ>(
                    sd, &lc, scan, eob_cdf, base_cdfs, br_cdfs, levels, quant,
                ),
                _ => read_levels::<TX_CLASS_VERT>(
                    sd, &lc, scan, eob_cdf, base_cdfs, br_cdfs, levels, quant,
                ),
            };

            // Reads no symbol, so it can run before the loop that needs it.
            let dc_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
            let dequant = Dequant::new(&self.recon_params(plane, tx_sz, plane_tx_type));
            let levels = &mut *self.levels;
            let quant = &mut *self.quant;
            let dq = &mut self.itx.dq;
            // The sign pass visits the nonzero positions in scan order (see `read_levels`).
            loop {
                let packed = quant[qp(pos)];
                let next = (packed & 0x7FF) as usize;
                let mut q = packed >> 11;
                let row = pos >> lc.bwl;
                levels[lv(pos + row * TX_PAD)] = 0;
                let sign = if pos == 0 {
                    self.sd.read_cdf(&mut self.cdf.dc_sign_cdf[ptype][dc_ctx]) != 0
                } else {
                    self.sd.read_bool()
                };
                if q > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
                    let mut length = 0;
                    loop {
                        length += 1;
                        if self.sd.read_bool() {
                            break;
                        }
                        if length == 20 {
                            // Back to all zero for the next tile decoder that reuses them.
                            levels.fill(0);
                            quant.fill(0);
                            dq.fill(0);
                            return Err(Error::Invalid("golomb_length_bit"));
                        }
                    }
                    let mut x = 1i32;
                    for _ in 0..length - 1 {
                        x = (x << 1) | i32::from(self.sd.read_bool());
                    }
                    q = x + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as i32;
                }
                // q is at least 1 here.
                if pos == 0 {
                    dc_category = if sign { 1 } else { 2 };
                }
                q &= 0xFFFFF;
                cul_level += q as u32;
                let col = pos - (row << lc.bwl);
                if q != 0 {
                    rows = rows.max(row + 1);
                    cols = cols.max(col + 1);
                }
                // Dequant[ row ][ col ] at once (spec 7.12.3), so only the nonzero coefficients
                // are dequantized; row and col are below 32, so the mask changes nothing.
                if let Some(d) = &dequant {
                    dq[(col * DQ_STRIDE + row) & (DQ_LEN - 1)] =
                        d.coef(if sign { -q } else { q }, pos);
                }
                quant[qp(pos)] = 0;
                if next == 0 {
                    break;
                }
                pos = next;
            }
            cul_level = cul_level.min(63);
        }

        let cul_level = cul_level as u8;
        set_ctx(&mut self.above_level_context[plane], x4, w4, cul_level);
        set_ctx(&mut self.above_dc_context[plane], x4, w4, dc_category);
        set_ctx(&mut self.left_level_context[plane], y4, h4, cul_level);
        set_ctx(&mut self.left_dc_context[plane], y4, h4, dc_category);
        Ok((eob, plane_tx_type, (rows, cols)))
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

    /// `LevelCtx::base_ctx` for a runtime class, with the DC context of a 2D class (0) that
    /// `read_levels` applies itself.
    fn base_ctx_of(
        tx_sz: usize,
        tx_class: usize,
        levels: &[u8; LEVELS_LEN],
        row: usize,
        col: usize,
    ) -> usize {
        let lc = LevelCtx::new(tx_sz, tx_class);
        let s5 = lc.neighbours(levels, row, col).0;
        match tx_class {
            TX_CLASS_2D if row == 0 && col == 0 => 0,
            TX_CLASS_2D => lc.base_ctx::<TX_CLASS_2D>(s5, row, col),
            TX_CLASS_HORIZ => lc.base_ctx::<TX_CLASS_HORIZ>(s5, row, col),
            _ => lc.base_ctx::<TX_CLASS_VERT>(s5, row, col),
        }
    }

    /// `LevelCtx::br_ctx` for a runtime class.
    fn br_ctx_of(
        tx_sz: usize,
        tx_class: usize,
        levels: &[u8; LEVELS_LEN],
        row: usize,
        col: usize,
    ) -> usize {
        let lc = LevelCtx::new(tx_sz, tx_class);
        let s3 = lc.neighbours(levels, row, col).1;
        match tx_class {
            TX_CLASS_2D => lc.br_ctx::<TX_CLASS_2D>(s3, row, col),
            TX_CLASS_HORIZ => lc.br_ctx::<TX_CLASS_HORIZ>(s3, row, col),
            _ => lc.br_ctx::<TX_CLASS_VERT>(s3, row, col),
        }
    }

    /// The packed sums equal the spec's sums for every set of neighbour levels 0..=15: the five
    /// of the base context and the first three of the br context.
    #[test]
    fn packed_sums_are_exact() {
        for a in 0..16 {
            for b in 0..16 {
                for c in 0..16 {
                    let s3: usize = [a, b, c].iter().map(|&l| usize::from(pack_level(l))).sum();
                    assert_eq!(s3 & 63, a + b + c);
                    for d in 0..16 {
                        for e in 0..16 {
                            let ls = [a, b, c, d, e];
                            let s5: usize = ls.iter().map(|&l| usize::from(pack_level(l))).sum();
                            let mag: usize = ls.iter().map(|&l| l.min(3)).sum();
                            let want = ((mag + 1) >> 1).min(4);
                            assert_eq!(if s5 > 512 { 4 } else { (s5 + 64) >> 7 }, want);
                        }
                    }
                }
            }
        }
    }

    /// The br neighbours are the first three base neighbours for every class.
    #[test]
    fn br_offsets_are_the_first_base_offsets() {
        for class in 0..3 {
            assert_eq!(
                MAG_REF_OFFSET_WITH_TX_CLASS[class][..],
                SIG_REF_DIFF_OFFSET[class][..3]
            );
        }
    }

    /// `read_levels` links the nonzero positions with 0 as the end mark and never links to
    /// position 0, which needs every scan to start at 0.
    #[test]
    fn every_scan_starts_at_position_0() {
        for tx_sz in 0..TX_SIZES_ALL {
            for tx_type in 0..TX_TYPES {
                assert_eq!(
                    get_scan(tx_sz, tx_type)[0],
                    0,
                    "tx_sz {tx_sz} type {tx_type}"
                );
            }
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
                        levels[(pos >> bwl) * stride + (pos & (w - 1))] = pack_level(v as usize);
                    }
                    for pos in 0..w * h {
                        let (row, col) = (pos >> bwl, pos & (w - 1));
                        assert_eq!(
                            base_ctx_of(tx_sz, tx_class, &levels, row, col),
                            coeff_base_ctx_oracle(&q, tx_sz, tx_class, pos),
                            "tx_sz {tx_sz} class {tx_class} pos {pos}"
                        );
                        assert_eq!(
                            br_ctx_of(tx_sz, tx_class, &levels, row, col),
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

    /// The in-plane runs and the `dc_sign` context built on them equal the spec's loops (visit
    /// `k < n` with `start + k < max`) on random context arrays, starts and plane ends,
    /// including runs that start at or past the plane end.
    #[test]
    fn context_runs_match_the_spec_loops() {
        let mut rng = crate::symbol::XorShift(5);
        for _ in 0..2000 {
            let len = 32 * (1 + rng.next_u32() as usize % 3);
            let v: Vec<u8> = (0..len).map(|_| (rng.next_u32() % 4) as u8).collect();
            let u: Vec<u8> = (0..len).map(|_| (rng.next_u32() % 4) as u8).collect();
            let max = rng.next_u32() as usize % (len + 1);
            let start = rng.next_u32() as usize % (len + 1);
            let n = [1, 2, 4, 8, 16][rng.next_u32() as usize % 5].min(len - start.min(len));
            let spec: Vec<u8> = (0..n)
                .filter(|&k| start + k < max)
                .map(|k| v[start + k])
                .collect();
            assert_eq!(ctx_run(&v, start, n, max), &spec[..]);
            let mut dc_sign = 0i32;
            for k in 0..n {
                if start + k < max {
                    match v[start + k] {
                        1 => dc_sign -= 1,
                        2 => dc_sign += 1,
                        _ => {}
                    }
                }
            }
            for k in 0..n {
                if start + k < max {
                    match u[start + k] {
                        1 => dc_sign -= 1,
                        2 => dc_sign += 1,
                        _ => {}
                    }
                }
            }
            // Spec: 1 when negative, 2 when positive, else 0.
            let want = [1, 0, 2][(dc_sign.signum() + 1) as usize];
            assert_eq!(
                dc_sign_ctx_of(ctx_run(&v, start, n, max), ctx_run(&u, start, n, max)),
                want
            );
        }
    }
}
