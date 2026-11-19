//! Conversion and lookup tables of spec 10 ("Conversion tables"), plus `Max_Tx_Depth` and
//! `Subsampled_Size` from spec 5.11.
//!
//! Every table keeps its spec name in upper snake case and the spec's dimensions. Values that
//! are sizes or indices are `usize`; the others use the smallest type that holds them.

#![allow(dead_code)]

use crate::consts::*;

/// Base 2 logarithm of the block width in units of 4 samples.
pub(crate) static MI_WIDTH_LOG2: [usize; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];

/// Base 2 logarithm of the block height in units of 4 samples.
pub(crate) static MI_HEIGHT_LOG2: [usize; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];

/// Block width in units of 4 samples.
pub(crate) static NUM_4X4_BLOCKS_WIDE: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];

/// Block height in units of 4 samples.
pub(crate) static NUM_4X4_BLOCKS_HIGH: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// Block width in samples (4 * `NUM_4X4_BLOCKS_WIDE`).
pub(crate) static BLOCK_WIDTH: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];

/// Block height in samples (4 * `NUM_4X4_BLOCKS_HIGH`).
pub(crate) static BLOCK_HEIGHT: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// Maps a block size to a context for intra syntax elements.
pub(crate) static SIZE_GROUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

/// Largest transform size (square or rectangular) usable for a luma block size.
pub(crate) static MAX_TX_SIZE_RECT: [usize; 22] = [
    TX_4X4, TX_4X8, TX_8X4, TX_8X8, TX_8X16, TX_16X8, TX_16X16, TX_16X32, TX_32X16, TX_32X32,
    TX_32X64, TX_64X32, TX_64X64, TX_64X64, TX_64X64, TX_64X64, TX_4X16, TX_16X4, TX_8X32, TX_32X8,
    TX_16X64, TX_64X16,
];

/// Spec 5.11.15: number of splits needed to reach a 4x4 transform from the block size.
pub(crate) static MAX_TX_DEPTH: [usize; 22] = [
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
];

/// Sub-block size of partition type `p` for a square block size (never read for rectangular ones).
pub(crate) static PARTITION_SUBSIZE: [[usize; 22]; 10] = [
    [
        BLOCK_4X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_128X128,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_128X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X128,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_128X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_128X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X128,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X128,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X64,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
    ],
];

/// Spec 5.11 `get_plane_residual_size`: `Subsampled_Size[ subsize ][ subx ][ suby ]` (plane residual size).
pub(crate) static SUBSAMPLED_SIZE: [[[usize; 2]; 2]; 22] = [
    [[BLOCK_4X4, BLOCK_4X4], [BLOCK_4X4, BLOCK_4X4]],
    [[BLOCK_4X8, BLOCK_4X4], [BLOCK_INVALID, BLOCK_4X4]],
    [[BLOCK_8X4, BLOCK_INVALID], [BLOCK_4X4, BLOCK_4X4]],
    [[BLOCK_8X8, BLOCK_8X4], [BLOCK_4X8, BLOCK_4X4]],
    [[BLOCK_8X16, BLOCK_8X8], [BLOCK_INVALID, BLOCK_4X8]],
    [[BLOCK_16X8, BLOCK_INVALID], [BLOCK_8X8, BLOCK_8X4]],
    [[BLOCK_16X16, BLOCK_16X8], [BLOCK_8X16, BLOCK_8X8]],
    [[BLOCK_16X32, BLOCK_16X16], [BLOCK_INVALID, BLOCK_8X16]],
    [[BLOCK_32X16, BLOCK_INVALID], [BLOCK_16X16, BLOCK_16X8]],
    [[BLOCK_32X32, BLOCK_32X16], [BLOCK_16X32, BLOCK_16X16]],
    [[BLOCK_32X64, BLOCK_32X32], [BLOCK_INVALID, BLOCK_16X32]],
    [[BLOCK_64X32, BLOCK_INVALID], [BLOCK_32X32, BLOCK_32X16]],
    [[BLOCK_64X64, BLOCK_64X32], [BLOCK_32X64, BLOCK_32X32]],
    [[BLOCK_64X128, BLOCK_64X64], [BLOCK_INVALID, BLOCK_32X64]],
    [[BLOCK_128X64, BLOCK_INVALID], [BLOCK_64X64, BLOCK_64X32]],
    [[BLOCK_128X128, BLOCK_128X64], [BLOCK_64X128, BLOCK_64X64]],
    [[BLOCK_4X16, BLOCK_4X8], [BLOCK_INVALID, BLOCK_4X8]],
    [[BLOCK_16X4, BLOCK_INVALID], [BLOCK_8X4, BLOCK_8X4]],
    [[BLOCK_8X32, BLOCK_8X16], [BLOCK_INVALID, BLOCK_4X16]],
    [[BLOCK_32X8, BLOCK_INVALID], [BLOCK_16X8, BLOCK_16X4]],
    [[BLOCK_16X64, BLOCK_16X32], [BLOCK_INVALID, BLOCK_8X32]],
    [[BLOCK_64X16, BLOCK_INVALID], [BLOCK_32X16, BLOCK_32X8]],
];

/// Transform size after one split (spec 5.11.17).
pub(crate) static SPLIT_TX_SIZE: [usize; 19] = [
    TX_4X4, TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16,
    TX_32X32, TX_32X32, TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_16X32, TX_32X16,
];

/// Default transform type for an intra prediction mode (UV_CFL_PRED included).
pub(crate) static MODE_TO_TXFM: [usize; 14] = [
    DCT_DCT,   // DC_PRED
    ADST_DCT,  // V_PRED
    DCT_ADST,  // H_PRED
    DCT_DCT,   // D45_PRED
    ADST_ADST, // D135_PRED
    ADST_DCT,  // D113_PRED
    DCT_ADST,  // D157_PRED
    DCT_ADST,  // D203_PRED
    ADST_DCT,  // D67_PRED
    ADST_ADST, // SMOOTH_PRED
    ADST_DCT,  // SMOOTH_V_PRED
    DCT_ADST,  // SMOOTH_H_PRED
    ADST_ADST, // PAETH_PRED
    DCT_DCT,   // UV_CFL_PRED
];

/// Palette colour context from the neighbour hash (spec 7.11.4 / 8.3.2); -1 is never read.
pub(crate) static PALETTE_COLOR_CONTEXT: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];

/// Weights of the three palette colour neighbours in the context hash.
pub(crate) static PALETTE_COLOR_HASH_MULTIPLIERS: [usize; 3] = [1, 2, 2];

pub(crate) static SM_WEIGHTS_TX_4X4: [u8; 4] = [255, 149, 85, 64];
pub(crate) static SM_WEIGHTS_TX_8X8: [u8; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
pub(crate) static SM_WEIGHTS_TX_16X16: [u8; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
pub(crate) static SM_WEIGHTS_TX_32X32: [u8; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
pub(crate) static SM_WEIGHTS_TX_64X64: [u8; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133,
    127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38,
    35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// Prediction angle in degrees for each intra mode (spec 7.11.2).
pub(crate) static MODE_TO_ANGLE: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// Directional prediction step per angle (spec 7.11.2.4).
pub(crate) static DR_INTRA_DERIVATIVE: [u16; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

/// Recursive (filter intra) taps, indexed [filter_intra_mode][position in 4x2 cell][tap].
pub(crate) static INTRA_FILTER_TAPS: [[[i8; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// Square transform size with side `Min( w, h )`.
pub(crate) static TX_SIZE_SQR: [usize; 19] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16,
    TX_16X16, TX_32X32, TX_32X32, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16,
];

/// Square transform size with side `Max( w, h )`.
pub(crate) static TX_SIZE_SQR_UP: [usize; 19] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_8X8, TX_8X8, TX_16X16, TX_16X16, TX_32X32,
    TX_32X32, TX_64X64, TX_64X64, TX_16X16, TX_16X16, TX_32X32, TX_32X32, TX_64X64, TX_64X64,
];

pub(crate) static TX_WIDTH: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];

pub(crate) static TX_HEIGHT: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

pub(crate) static TX_WIDTH_LOG2: [usize; 19] =
    [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];

pub(crate) static TX_HEIGHT_LOG2: [usize; 19] =
    [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];

/// Wedge index bit count per block size (0 when wedges are not allowed).
pub(crate) static WEDGE_BITS: [u8; 22] = [
    0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0,
];

/// Neighbour offsets `[row, col]` for the coefficient contexts, per tx class (spec 8.3.2).
pub(crate) static SIG_REF_DIFF_OFFSET: [[[usize; 2]; 5]; 3] = [
    [[0, 1], [1, 0], [1, 1], [0, 2], [2, 0]],
    [[0, 1], [1, 0], [0, 2], [0, 3], [0, 4]],
    [[0, 1], [1, 0], [2, 0], [3, 0], [4, 0]],
];

/// Transform size actually coded (64-wide/high transforms are limited to 32).
pub(crate) static ADJUSTED_TX_SIZE: [usize; 19] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_32X32, TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_16X32,
    TX_32X16, TX_32X32, TX_32X32, TX_4X16, TX_16X4, TX_8X32, TX_32X8, TX_16X32, TX_32X16,
];

/// Film grain random samples (spec 7.18.3.2).
pub(crate) static GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388,
    -732, -104, -188, 800, 112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500,
    -228, 12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192, 1000, -332, 652, -136,
    -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124, -24, 396, 540, -12, -104, 640,
    464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40, -156, 252,
    -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60,
    -264, -96, -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792,
    304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548,
    488, -344, -136, 488, -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168, -108,
    464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116, 436, 504, -232, 328,
    844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4,
    196, 44, -36, -648, 932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448,
    208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540, 28, 136, -88, -436, 256, 296,
    -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768,
    388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648, -136, 372,
    -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112,
    -220, -100, 996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412,
    -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240,
    -208, -40, 100, -592, 344, -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752, -100, 560, -1020, 180,
    -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164,
    -376, -684, 592, -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864,
    424, -728, 988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776,
    -824, -884, 524, -548, -564, -68, -164, -96, 692, 364, -692, -1012, -68, 260, -480, 876, -1116,
    452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112, -440,
    -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276,
    -508, 940, 24, -400, -120, 756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892, -888, 60, 164, 68, -760, 216,
    -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776,
    1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272,
    460, -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560,
    1040, -472, 116, -848, -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972,
    -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828, 524, 212, -212, 52, 12, 200, 268,
    -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436, 316, 372,
    -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4,
    -1168, 368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272,
    -256, -420, 320, 208, -144, -156, 156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360,
    32, -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840, 784, -460, 656, -544,
    -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308, 308, -112, -636, -760,
    280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636, -76,
    392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24,
    -1044, -32, 40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424,
    -1008, -324, 588, -1496, 576, 460, -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52,
    -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164, -304, -540,
    120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148, 716, -604, 404,
    -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508,
    132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384,
    -188, -612, -576, 736, 508, 892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508,
    -144, -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472, -204, 908, -188, -1000,
    -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760, -176, 376, -524, -452,
    -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832,
    148, 248, 652, 616, 1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64,
    156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524, -268, -720, 540,
    -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468,
    -28, -120, -32, -44, 1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156,
    36, 492, -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808, -416, 592, 316, -480,
    56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324, -416, -600, 768, 268, -248, -88, -132,
    -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480, -216, -92, 1688,
    -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332, 204,
    760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728,
    -520, 276, -188, 80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272,
    -404, 180, 816, -796, 48, 152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316, 476,
    12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576, 260, -352, 120, -16, -136, -260,
    -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84,
    192, 852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848,
    -344, 224, 208, -200, 484, 128, -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112,
    -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84, 200, -152, 0, -144, 584,
    280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300,
    24, -676, -596, 132, 1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488,
    424, 24, 756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424,
    888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628, -348, -228, -380, 1012, -452,
    -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344, 428, 548,
    -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284,
    -12, 824, 164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648,
    -620, -684, -24, -376, -384, -108, -920, -1032, 768, 180, -264, -508, -1268, -260, -60, 300,
    -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4, -216, -32,
    836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228,
    132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376,
    -108, -92, 788, 368, -56, 544, 400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200, -56, 276, -88, 484, -324,
    948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116,
    260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676,
    -396, -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212,
    252, -660, -708, 484, -444, -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196, 76, 388, 232, 412, -1168,
    -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168, -608, 448, 160, -532, -272,
    28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524, -572, -556,
    -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488,
    -192, -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12,
    -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716,
    592, -80, 436, 360, 4, -248, 160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300, -212, -1144, 32, -724,
    800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552,
    -52, -100, -656, 208, -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272,
    -876, -664, -648, -332, -136, 16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92, 356, 404, -408, 252, 196, -176,
    -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508,
    -476, 324, -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348,
    -288, -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12,
    168, 1020, 512, -552, 924, -148, 716, 188, 164, -340, -520, -184, 880, -152, -680, -208, -1156,
    -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280, -224, 96,
    568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428, -484,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dimensions_agree() {
        for b in 0..22 {
            assert_eq!(BLOCK_WIDTH[b], 4 * NUM_4X4_BLOCKS_WIDE[b]);
            assert_eq!(BLOCK_HEIGHT[b], 4 * NUM_4X4_BLOCKS_HIGH[b]);
            assert_eq!(NUM_4X4_BLOCKS_WIDE[b], 1 << MI_WIDTH_LOG2[b]);
            assert_eq!(NUM_4X4_BLOCKS_HIGH[b], 1 << MI_HEIGHT_LOG2[b]);
        }
    }

    #[test]
    fn tx_dimensions_agree() {
        for t in 0..19 {
            assert_eq!(TX_WIDTH[t], 1 << TX_WIDTH_LOG2[t]);
            assert_eq!(TX_HEIGHT[t], 1 << TX_HEIGHT_LOG2[t]);
            assert_eq!(TX_WIDTH[TX_SIZE_SQR[t]], TX_WIDTH[t].min(TX_HEIGHT[t]));
            assert_eq!(TX_WIDTH[TX_SIZE_SQR_UP[t]], TX_WIDTH[t].max(TX_HEIGHT[t]));
            assert!(TX_WIDTH[ADJUSTED_TX_SIZE[t]] <= 32 && TX_HEIGHT[ADJUSTED_TX_SIZE[t]] <= 32);
        }
    }

    #[test]
    fn partition_and_subsampling() {
        // PARTITION_SPLIT of a 16x16 block gives 8x8; PARTITION_HORZ gives 16x8.
        assert_eq!(PARTITION_SUBSIZE[3][BLOCK_16X16], BLOCK_8X8);
        assert_eq!(PARTITION_SUBSIZE[1][BLOCK_16X16], BLOCK_16X8);
        // 4:2:0 chroma of a 16x16 luma block is 8x8; 4:4:4 keeps it.
        assert_eq!(SUBSAMPLED_SIZE[BLOCK_16X16][1][1], BLOCK_8X8);
        assert_eq!(SUBSAMPLED_SIZE[BLOCK_16X16][0][0], BLOCK_16X16);
        // 4:2:2 of 4x8 is invalid.
        assert_eq!(SUBSAMPLED_SIZE[BLOCK_4X8][1][0], BLOCK_INVALID);
        for r in &PARTITION_SUBSIZE {
            assert_eq!(r.len(), 22);
        }
    }

    #[test]
    fn split_and_mode_tables() {
        assert_eq!(SPLIT_TX_SIZE[TX_64X64], TX_32X32);
        assert_eq!(SPLIT_TX_SIZE[TX_16X32], TX_8X16);
        assert_eq!(MODE_TO_TXFM[0], DCT_DCT);
        assert_eq!(MODE_TO_TXFM[1], ADST_DCT);
        assert_eq!(MODE_TO_TXFM[4], ADST_ADST);
        assert_eq!(MODE_TO_TXFM[13], DCT_DCT);
        assert_eq!(MODE_TO_ANGLE[8], 67);
        assert_eq!(DR_INTRA_DERIVATIVE[3], 1023);
        assert_eq!(DR_INTRA_DERIVATIVE[87], 3);
    }

    #[test]
    fn smooth_weights_decrease() {
        for w in [
            &SM_WEIGHTS_TX_16X16[..],
            &SM_WEIGHTS_TX_32X32,
            &SM_WEIGHTS_TX_64X64,
        ] {
            assert_eq!(w[0], 255);
            assert!(w.windows(2).all(|p| p[0] >= p[1]));
        }
    }

    #[test]
    fn gaussian_sequence_ends() {
        assert_eq!(GAUSSIAN_SEQUENCE[0], 56);
        assert_eq!(GAUSSIAN_SEQUENCE[2047], -484);
    }
}
