//! Spec constants: every constant of section 3 ("Symbols and abbreviated terms") with its name
//! and value, then the enumerations of sections 6 and 7 with the spec's numbering.
//!
//! Section 3 constants and everything used as a table index are `usize`. Values that are stored
//! in header structs as small integers (OBU types, frame types, color codes, restoration and tx
//! modes) are `u8`. `NONE` (the "no reference frame" value, -1 in the spec) is `i8`.

#![allow(dead_code)]

// ---------------------------------------------------------------------------------------------
// Section 3: symbols and abbreviated terms.
// ---------------------------------------------------------------------------------------------

/// Number of reference frames that can be used for inter prediction
pub const REFS_PER_FRAME: usize = 7;
/// Number of reference frame types (including intra type)
pub const TOTAL_REFS_PER_FRAME: usize = 8;
/// Number of contexts when decoding `y_mode`
pub const BLOCK_SIZE_GROUPS: usize = 4;
/// Number of different block sizes used
pub const BLOCK_SIZES: usize = 22;
/// Sentinel value to mark partition choices that are not allowed
pub const BLOCK_INVALID: usize = 22;
/// Maximum size of a superblock in luma samples
pub const MAX_SB_SIZE: usize = 128;
/// Smallest size of a mode info block in luma samples
pub const MI_SIZE: usize = 4;
/// Base 2 logarithm of smallest size of a mode info block
pub const MI_SIZE_LOG2: usize = 2;
/// Maximum width of a tile in units of luma samples
pub const MAX_TILE_WIDTH: usize = 4096;
/// Maximum area of a tile in units of luma samples
pub const MAX_TILE_AREA: usize = 4096 * 2304;
/// Maximum number of tile rows
pub const MAX_TILE_ROWS: usize = 64;
/// Maximum number of tile columns
pub const MAX_TILE_COLS: usize = 64;
/// Number of horizontal luma samples before intra block copy can be used
pub const INTRABC_DELAY_PIXELS: usize = 256;
/// Number of 64 by 64 blocks before intra block copy can be used
pub const INTRABC_DELAY_SB64: usize = 4;
/// Number of frames that can be stored for future reference
pub const NUM_REF_FRAMES: usize = 8;
/// Number of contexts for `is_inter`
pub const IS_INTER_CONTEXTS: usize = 4;
/// Number of contexts for `single_ref`, `comp_ref`, `comp_bwdref`, `uni_comp_ref`, `uni_comp_ref_p1` and `uni_comp_ref_p2`
pub const REF_CONTEXTS: usize = 3;
/// Number of segments allowed in segmentation map
pub const MAX_SEGMENTS: usize = 8;
/// Number of contexts for `segment_id`
pub const SEGMENT_ID_CONTEXTS: usize = 3;
/// Index for quantizer segment feature
pub const SEG_LVL_ALT_Q: usize = 0;
/// Index for vertical luma loop filter segment feature
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;
/// Index for reference frame segment feature
pub const SEG_LVL_REF_FRAME: usize = 5;
/// Index for skip segment feature
pub const SEG_LVL_SKIP: usize = 6;
/// Index for global mv feature
pub const SEG_LVL_GLOBALMV: usize = 7;
/// Number of segment features
pub const SEG_LVL_MAX: usize = 8;
/// Number of different plane types (luma or chroma)
pub const PLANE_TYPES: usize = 2;
/// Number of contexts for transform size
pub const TX_SIZE_CONTEXTS: usize = 3;
/// Number of values for `interp_filter`
pub const INTERP_FILTERS: usize = 3;
/// Number of contexts for `interp_filter`
pub const INTERP_FILTER_CONTEXTS: usize = 16;
/// Number of contexts for decoding skip_mode
pub const SKIP_MODE_CONTEXTS: usize = 3;
/// Number of contexts for decoding skip
pub const SKIP_CONTEXTS: usize = 3;
/// Number of contexts when decoding `partition`
pub const PARTITION_CONTEXTS: usize = 4;
/// Number of square transform sizes
pub const TX_SIZES: usize = 5;
/// Number of transform sizes (including non-square sizes)
pub const TX_SIZES_ALL: usize = 19;
/// Number of values for `tx_mode`
pub const TX_MODES: usize = 3;
/// Inverse transform rows with `DCT` and columns with `DCT`
pub const DCT_DCT: usize = 0;
/// Inverse transform rows with `DCT` and columns with `ADST`
pub const ADST_DCT: usize = 1;
/// Inverse transform rows with `ADST` and columns with `DCT`
pub const DCT_ADST: usize = 2;
/// Inverse transform rows with `ADST` and columns with `ADST`
pub const ADST_ADST: usize = 3;
/// Inverse transform rows with `DCT` and columns with `FLIPADST`
pub const FLIPADST_DCT: usize = 4;
/// Inverse transform rows with `FLIPADST` and columns with `DCT`
pub const DCT_FLIPADST: usize = 5;
/// Inverse transform rows with `FLIPADST` and columns with `FLIPADST`
pub const FLIPADST_FLIPADST: usize = 6;
/// Inverse transform rows with `FLIPADST` and columns with `ADST`
pub const ADST_FLIPADST: usize = 7;
/// Inverse transform rows with `ADST` and columns with `FLIPADST`
pub const FLIPADST_ADST: usize = 8;
/// Inverse transform rows with identity and columns with identity
pub const IDTX: usize = 9;
/// Inverse transform rows with identity and columns with `DCT`
pub const V_DCT: usize = 10;
/// Inverse transform rows with `DCT` and columns with identity
pub const H_DCT: usize = 11;
/// Inverse transform rows with identity and columns with `ADST`
pub const V_ADST: usize = 12;
/// Inverse transform rows with `ADST` and columns with identity
pub const H_ADST: usize = 13;
/// Inverse transform rows with identity and columns with `FLIPADST`
pub const V_FLIPADST: usize = 14;
/// Inverse transform rows with `FLIPADST` and columns with identity
pub const H_FLIPADST: usize = 15;
/// Number of inverse transform types
pub const TX_TYPES: usize = 16;
/// Number of values for `YMode`
pub const MB_MODE_COUNT: usize = 17;
/// Number of values for `y_mode`
pub const INTRA_MODES: usize = 13;
/// Number of values for `uv_mode` when chroma from luma is not allowed
pub const UV_INTRA_MODES_CFL_NOT_ALLOWED: usize = 13;
/// Number of values for `uv_mode` when chroma from luma is allowed
pub const UV_INTRA_MODES_CFL_ALLOWED: usize = 14;
/// Number of values for `compound_mode`
pub const COMPOUND_MODES: usize = 8;
/// Number of contexts for `compound_mode`
pub const COMPOUND_MODE_CONTEXTS: usize = 8;
/// Number of new mv values used when constructing context for `compound_mode`
pub const COMP_NEWMV_CTXS: usize = 5;
/// Number of contexts for `new_mv`
pub const NEW_MV_CONTEXTS: usize = 6;
/// Number of contexts for `zero_mv`
pub const ZERO_MV_CONTEXTS: usize = 2;
/// Number of contexts for `ref_mv`
pub const REF_MV_CONTEXTS: usize = 6;
/// Number of contexts for `drl_mode`
pub const DRL_MODE_CONTEXTS: usize = 3;
/// Number of contexts for decoding motion vectors including one for intra block copy
pub const MV_CONTEXTS: usize = 2;
/// Motion vector context used for intra block copy
pub const MV_INTRABC_CONTEXT: usize = 1;
/// Number of values for `mv_joint`
pub const MV_JOINTS: usize = 4;
/// Number of values for `mv_class`
pub const MV_CLASSES: usize = 11;
/// Number of values for `mv_class0_bit`
pub const CLASS0_SIZE: usize = 2;
/// Maximum number of bits for decoding motion vectors
pub const MV_OFFSET_BITS: usize = 10;
/// Maximum value used for loop filtering
pub const MAX_LOOP_FILTER: usize = 63;
/// Number of bits of precision when scaling reference frames
pub const REF_SCALE_SHIFT: usize = 14;
/// Number of bits of precision when choosing an inter prediction filter kernel
pub const SUBPEL_BITS: usize = 4;
/// `( 1 << SUBPEL_BITS ) - 1`
pub const SUBPEL_MASK: usize = 15;
/// Number of bits of precision when computing inter prediction locations
pub const SCALE_SUBPEL_BITS: usize = 10;
/// Value used when clipping motion vectors
pub const MV_BORDER: usize = 128;
/// Number of values for color contexts
pub const PALETTE_COLOR_CONTEXTS: usize = 5;
/// Number of mappings between color context hash and color context
pub const PALETTE_MAX_COLOR_CONTEXT_HASH: usize = 8;
/// Number of values for palette block size
pub const PALETTE_BLOCK_SIZE_CONTEXTS: usize = 7;
/// Number of values for palette Y plane mode contexts
pub const PALETTE_Y_MODE_CONTEXTS: usize = 3;
/// Number of values for palette U and V plane mode contexts
pub const PALETTE_UV_MODE_CONTEXTS: usize = 2;
/// Number of values for palette_size
pub const PALETTE_SIZES: usize = 7;
/// Number of values for palette_color
pub const PALETTE_COLORS: usize = 8;
/// Number of neighbors considered within palette computation
pub const PALETTE_NUM_NEIGHBORS: usize = 3;
/// Value indicating alternative encoding of quantizer index delta values
pub const DELTA_Q_SMALL: usize = 3;
/// Value indicating alternative encoding of loop filter delta values
pub const DELTA_LF_SMALL: usize = 3;
/// Number of values in the quantizer matrix
pub const QM_TOTAL_SIZE: usize = 3344;
/// Maximum magnitude of AngleDeltaY and AngleDeltaUV
pub const MAX_ANGLE_DELTA: usize = 3;
/// Number of directional intra modes
pub const DIRECTIONAL_MODES: usize = 8;
/// Number of degrees of step per unit increase in AngleDeltaY or AngleDeltaUV.
pub const ANGLE_STEP: usize = 3;
/// Number of intra transform set types
pub const TX_SET_TYPES_INTRA: usize = 3;
/// Number of inter transform set types
pub const TX_SET_TYPES_INTER: usize = 4;
/// Internal precision of warped motion models
pub const WARPEDMODEL_PREC_BITS: usize = 16;
/// Warp model is just an identity transform
pub const IDENTITY: usize = 0;
/// Warp model is a pure translation
pub const TRANSLATION: usize = 1;
/// Warp model is a rotation + symmetric zoom + translation
pub const ROTZOOM: usize = 2;
/// Warp model is a general affine transform
pub const AFFINE: usize = 3;
/// Number of bits encoded for translational components of global motion models, if part of a ROTZOOM or AFFINE model
pub const GM_ABS_TRANS_BITS: usize = 12;
/// Number of bits encoded for translational components of global motion models, if part of a TRANSLATION model
pub const GM_ABS_TRANS_ONLY_BITS: usize = 9;
/// Number of bits encoded for non-translational components of global motion models
pub const GM_ABS_ALPHA_BITS: usize = 12;
/// Number of fractional bits of entries in divisor lookup table
pub const DIV_LUT_PREC_BITS: usize = 14;
/// Number of fractional bits for lookup in divisor lookup table
pub const DIV_LUT_BITS: usize = 8;
/// Number of entries in divisor lookup table
pub const DIV_LUT_NUM: usize = 257;
/// Number of values for motion modes
pub const MOTION_MODES: usize = 3;
/// Use translation or global motion compensation
pub const SIMPLE: usize = 0;
/// Use overlapped block motion compensation
pub const OBMC: usize = 1;
/// Use local warp motion compensation
pub const LOCALWARP: usize = 2;
/// Largest number of samples used when computing a local warp
pub const LEAST_SQUARES_SAMPLES_MAX: usize = 8;
/// Largest motion vector difference to include in local warp computation
pub const LS_MV_MAX: usize = 256;
/// Clamping value used for translation components of warp
pub const WARPEDMODEL_TRANS_CLAMP: usize = 1 << 23;
/// Clamping value used for matrix components of warp
pub const WARPEDMODEL_NONDIAGAFFINE_CLAMP: usize = 1 << 13;
/// Number of phases used in warped filtering
pub const WARPEDPIXEL_PREC_SHIFTS: usize = 1 << 6;
/// Number of extra bits of precision in warped filtering
pub const WARPEDDIFF_PREC_BITS: usize = 10;
/// Number of fractional bits for sending non-translational warp model coefficients
pub const GM_ALPHA_PREC_BITS: usize = 15;
/// Number of fractional bits for sending translational warp model coefficients
pub const GM_TRANS_PREC_BITS: usize = 6;
/// Number of fractional bits used for pure translational warps
pub const GM_TRANS_ONLY_PREC_BITS: usize = 3;
/// Number of inter intra modes
pub const INTERINTRA_MODES: usize = 4;
/// Size of MasterMask array
pub const MASK_MASTER_SIZE: usize = 64;
/// Number of contexts for `segment_id_predicted`
pub const SEGMENT_ID_PREDICTED_CONTEXTS: usize = 3;
/// Number of syntax elements for forward reference frames
pub const FWD_REFS: usize = 4;
/// Number of syntax elements for backward reference frames
pub const BWD_REFS: usize = 3;
/// Number of syntax elements for single reference frames
pub const SINGLE_REFS: usize = 7;
/// Number of syntax elements for unidirectional compound reference frames
pub const UNIDIR_COMP_REFS: usize = 4;
/// Number of values for `compound_type`
pub const COMPOUND_TYPES: usize = 2;
/// Number of values for `cfl_alpha_signs`
pub const CFL_JOINT_SIGNS: usize = 8;
/// Number of values for `cfl_alpha_u` and `cfl_alpha_v`
pub const CFL_ALPHABET_SIZE: usize = 16;
/// Number of contexts for `comp_mode`
pub const COMP_INTER_CONTEXTS: usize = 5;
/// Number of contexts for `comp_ref_type`
pub const COMP_REF_TYPE_CONTEXTS: usize = 5;
/// Number of contexts for `cfl_alpha_u` and `cfl_alpha_v`
pub const CFL_ALPHA_CONTEXTS: usize = 6;
/// Number of each of left and above contexts for `intra_frame_y_mode`
pub const INTRA_MODE_CONTEXTS: usize = 5;
/// Number of contexts for `comp_group_idx`
pub const COMP_GROUP_IDX_CONTEXTS: usize = 6;
/// Number of contexts for `compound_idx`
pub const COMPOUND_IDX_CONTEXTS: usize = 6;
/// Number of filter kernels for the intra edge filter
pub const INTRA_EDGE_KERNELS: usize = 3;
/// Number of kernel taps for the intra edge filter
pub const INTRA_EDGE_TAPS: usize = 5;
/// Number of loop filter strength values
pub const FRAME_LF_COUNT: usize = 4;
/// Maximum depth for variable transform trees
pub const MAX_VARTX_DEPTH: usize = 2;
/// Number of contexts for txfm_split
pub const TXFM_PARTITION_CONTEXTS: usize = 21;
/// Bonus weight for close motion vectors
pub const REF_CAT_LEVEL: usize = 640;
/// Maximum number of motion vectors in the stack
pub const MAX_REF_MV_STACK_SIZE: usize = 8;
/// Stack size for motion field motion vectors
pub const MFMV_STACK_SIZE: usize = 3;
/// Maximum times the transform can be split
pub const MAX_TX_DEPTH: usize = 2;
/// Number of directions for the wedge mask process
pub const WEDGE_TYPES: usize = 16;
/// Number of bits used in Wiener filter coefficients
pub const FILTER_BITS: usize = 7;
/// Number of Wiener filter coefficients to read
pub const WIENER_COEFFS: usize = 3;
/// Number of bits needed to specify self guided filter set
pub const SGRPROJ_PARAMS_BITS: usize = 4;
/// Controls how self guided deltas are read
pub const SGRPROJ_PRJ_SUBEXP_K: usize = 4;
/// Precision bits during self guided restoration
pub const SGRPROJ_PRJ_BITS: usize = 7;
/// Restoration precision bits generated higher than source before projection
pub const SGRPROJ_RST_BITS: usize = 4;
/// Precision of mtable division table
pub const SGRPROJ_MTABLE_BITS: usize = 20;
/// Precision of division by n table
pub const SGRPROJ_RECIP_BITS: usize = 12;
/// Internal precision bits for core selfguided_restoration
pub const SGRPROJ_SGR_BITS: usize = 8;
/// Number of bits to reduce CDF precision during arithmetic coding
pub const EC_PROB_SHIFT: usize = 6;
/// Minimum probability assigned to each symbol during arithmetic coding
pub const EC_MIN_PROB: usize = 4;
/// Value that indicates the allow_screen_content_tools syntax element is coded
pub const SELECT_SCREEN_CONTENT_TOOLS: usize = 2;
/// Value that indicates the force_integer_mv syntax element is coded
pub const SELECT_INTEGER_MV: usize = 2;
/// Maximum size of a loop restoration tile
pub const RESTORATION_TILESIZE_MAX: usize = 256;
/// Maximum distance when computing weighted prediction
pub const MAX_FRAME_DISTANCE: usize = 31;
/// Maximum horizontal offset of a projected motion vector
pub const MAX_OFFSET_WIDTH: usize = 8;
/// Maximum vertical offset of a projected motion vector
pub const MAX_OFFSET_HEIGHT: usize = 0;
/// Rounding bitwidth for the parameters to the shear process
pub const WARP_PARAM_REDUCE_BITS: usize = 6;
/// Number of quantizer base levels
pub const NUM_BASE_LEVELS: usize = 2;
/// The quantizer range above NUM_BASE_LEVELS above which the Exp-Golomb coding process is activated
pub const COEFF_BASE_RANGE: usize = 12;
/// Number of values for `coeff_br`
pub const BR_CDF_SIZE: usize = 4;
/// Number of contexts for `coeff_base_eob`
pub const SIG_COEF_CONTEXTS_EOB: usize = 4;
/// Context offset for `coeff_base` for horizontal-only or vertical-only transforms.
pub const SIG_COEF_CONTEXTS_2D: usize = 26;
/// Number of contexts for `coeff_base`
pub const SIG_COEF_CONTEXTS: usize = 42;
/// Maximum number of context samples to be used in determining the context index for `coeff_base` and `coeff_base_eob`.
pub const SIG_REF_DIFF_OFFSET_NUM: usize = 5;
/// Numerator for upscaling ratio
pub const SUPERRES_NUM: usize = 8;
/// Smallest denominator for upscaling ratio
pub const SUPERRES_DENOM_MIN: usize = 9;
/// Number of bits sent to specify denominator of upscaling ratio
pub const SUPERRES_DENOM_BITS: usize = 3;
/// Number of bits of fractional precision for upscaling filter selection
pub const SUPERRES_FILTER_BITS: usize = 6;
/// Number of phases of upscaling filters
pub const SUPERRES_FILTER_SHIFTS: usize = 1 << SUPERRES_FILTER_BITS;
/// Number of taps of upscaling filters
pub const SUPERRES_FILTER_TAPS: usize = 8;
/// Sample offset for upscaling filters
pub const SUPERRES_FILTER_OFFSET: usize = 3;
/// Number of fractional bits for computing position in upscaling
pub const SUPERRES_SCALE_BITS: usize = 14;
/// Mask for computing position in upscaling
pub const SUPERRES_SCALE_MASK: usize = (1 << 14) - 1;
/// Difference in precision between SUPERRES_SCALE_BITS and SUPERRES_FILTER_BITS
pub const SUPERRES_EXTRA_BITS: usize = 8;
/// Number of contexts for `all_zero`
pub const TXB_SKIP_CONTEXTS: usize = 13;
/// Number of contexts for `eob_extra`
pub const EOB_COEF_CONTEXTS: usize = 9;
/// Number of contexts for `dc_sign`
pub const DC_SIGN_CONTEXTS: usize = 3;
/// Number of contexts for `coeff_br`
pub const LEVEL_CONTEXTS: usize = 21;
/// Transform class for transform types performing non-identity transforms in both directions
pub const TX_CLASS_2D: usize = 0;
/// Transform class for transforms performing only a horizontal non-identity transform
pub const TX_CLASS_HORIZ: usize = 1;
/// Transform class for transforms performing only a vertical non-identity transform
pub const TX_CLASS_VERT: usize = 2;
/// Largest reference MV component that can be saved
pub const REFMVS_LIMIT: usize = (1 << 12) - 1;
/// Scaling shift for intra filtering process
pub const INTRA_FILTER_SCALE_BITS: usize = 4;
/// Number of types of intra filtering
pub const INTRA_FILTER_MODES: usize = 5;
/// Number of selectable context types for the coeff( ) syntax structure
pub const COEFF_CDF_Q_CTXS: usize = 4;
/// Value of `primary_ref_frame` indicating that there is no primary reference frame
pub const PRIMARY_REF_NONE: usize = 7;
/// Number of frames in buffer pool
pub const BUFFER_POOL_MAX_SIZE: usize = 10;

// ---------------------------------------------------------------------------------------------
// Enumerations (spec numbering).
// ---------------------------------------------------------------------------------------------

// OBU types (6.2.2).
pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_METADATA: u8 = 5;
pub const OBU_FRAME: u8 = 6;
pub const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
pub const OBU_TILE_LIST: u8 = 8;
pub const OBU_PADDING: u8 = 15;

// Metadata types (6.7.1).
pub const METADATA_TYPE_HDR_CLL: u64 = 1;
pub const METADATA_TYPE_HDR_MDCV: u64 = 2;
pub const METADATA_TYPE_SCALABILITY: u64 = 3;
pub const METADATA_TYPE_ITUT_T35: u64 = 4;
pub const METADATA_TYPE_TIMECODE: u64 = 5;
/// `scalability_mode_idc` value that carries a scalability structure.
pub const SCALABILITY_SS: u8 = 14;

// Frame types (6.8.2).
pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY_FRAME: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;

// Interpolation filters (6.8.9).
pub const EIGHTTAP: u8 = 0;
pub const EIGHTTAP_SMOOTH: u8 = 1;
pub const EIGHTTAP_SHARP: u8 = 2;
pub const BILINEAR: u8 = 3;
pub const SWITCHABLE: u8 = 4;

// Color primaries (6.4.2).
pub const CP_BT_709: u8 = 1;
pub const CP_UNSPECIFIED: u8 = 2;
pub const CP_BT_470_M: u8 = 4;
pub const CP_BT_470_B_G: u8 = 5;
pub const CP_BT_601: u8 = 6;
pub const CP_SMPTE_240: u8 = 7;
pub const CP_GENERIC_FILM: u8 = 8;
pub const CP_BT_2020: u8 = 9;
pub const CP_XYZ: u8 = 10;
pub const CP_SMPTE_431: u8 = 11;
pub const CP_SMPTE_432: u8 = 12;
pub const CP_EBU_3213: u8 = 22;

// Transfer characteristics (6.4.2).
pub const TC_RESERVED_0: u8 = 0;
pub const TC_BT_709: u8 = 1;
pub const TC_UNSPECIFIED: u8 = 2;
pub const TC_RESERVED_3: u8 = 3;
pub const TC_BT_470_M: u8 = 4;
pub const TC_BT_470_B_G: u8 = 5;
pub const TC_BT_601: u8 = 6;
pub const TC_SMPTE_240: u8 = 7;
pub const TC_LINEAR: u8 = 8;
pub const TC_LOG_100: u8 = 9;
pub const TC_LOG_100_SQRT10: u8 = 10;
pub const TC_IEC_61966: u8 = 11;
pub const TC_BT_1361: u8 = 12;
pub const TC_SRGB: u8 = 13;
pub const TC_BT_2020_10_BIT: u8 = 14;
pub const TC_BT_2020_12_BIT: u8 = 15;
pub const TC_SMPTE_2084: u8 = 16;
pub const TC_SMPTE_428: u8 = 17;
pub const TC_HLG: u8 = 18;

// Matrix coefficients (6.4.2).
pub const MC_IDENTITY: u8 = 0;
pub const MC_BT_709: u8 = 1;
pub const MC_UNSPECIFIED: u8 = 2;
pub const MC_RESERVED_3: u8 = 3;
pub const MC_FCC: u8 = 4;
pub const MC_BT_470_B_G: u8 = 5;
pub const MC_BT_601: u8 = 6;
pub const MC_SMPTE_240: u8 = 7;
pub const MC_SMPTE_YCGCO: u8 = 8;
pub const MC_BT_2020_NCL: u8 = 9;
pub const MC_BT_2020_CL: u8 = 10;
pub const MC_SMPTE_2085: u8 = 11;
pub const MC_CHROMAT_NCL: u8 = 12;
pub const MC_CHROMAT_CL: u8 = 13;
pub const MC_ICTCP: u8 = 14;

// Chroma sample positions (6.4.2).
pub const CSP_UNKNOWN: u8 = 0;
pub const CSP_VERTICAL: u8 = 1;
pub const CSP_COLOCATED: u8 = 2;
pub const CSP_RESERVED: u8 = 3;

// Reference frames (6.10.24). `NONE` is -1 in the spec.
pub const NONE: i8 = -1;
pub const INTRA_FRAME: usize = 0;
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

// Segmentation features (the rest, SEG_LVL_ALT_Q, SEG_LVL_ALT_LF_Y_V, SEG_LVL_REF_FRAME,
// SEG_LVL_SKIP, SEG_LVL_GLOBALMV and SEG_LVL_MAX, are in section 3).
pub const SEG_LVL_ALT_LF_Y_H: usize = 2;
pub const SEG_LVL_ALT_LF_U: usize = 3;
pub const SEG_LVL_ALT_LF_V: usize = 4;

// Block sizes (6.10.4). `BLOCK_INVALID` (22) is in section 3.
pub const BLOCK_4X4: usize = 0;
pub const BLOCK_4X8: usize = 1;
pub const BLOCK_8X4: usize = 2;
pub const BLOCK_8X8: usize = 3;
pub const BLOCK_8X16: usize = 4;
pub const BLOCK_16X8: usize = 5;
pub const BLOCK_16X16: usize = 6;
pub const BLOCK_16X32: usize = 7;
pub const BLOCK_32X16: usize = 8;
pub const BLOCK_32X32: usize = 9;
pub const BLOCK_32X64: usize = 10;
pub const BLOCK_64X32: usize = 11;
pub const BLOCK_64X64: usize = 12;
pub const BLOCK_64X128: usize = 13;
pub const BLOCK_128X64: usize = 14;
pub const BLOCK_128X128: usize = 15;
pub const BLOCK_4X16: usize = 16;
pub const BLOCK_16X4: usize = 17;
pub const BLOCK_8X32: usize = 18;
pub const BLOCK_32X8: usize = 19;
pub const BLOCK_16X64: usize = 20;
pub const BLOCK_64X16: usize = 21;

// Partition types (6.10.4).
pub const PARTITION_NONE: usize = 0;
pub const PARTITION_HORZ: usize = 1;
pub const PARTITION_VERT: usize = 2;
pub const PARTITION_SPLIT: usize = 3;
pub const PARTITION_HORZ_A: usize = 4;
pub const PARTITION_HORZ_B: usize = 5;
pub const PARTITION_VERT_A: usize = 6;
pub const PARTITION_VERT_B: usize = 7;
pub const PARTITION_HORZ_4: usize = 8;
pub const PARTITION_VERT_4: usize = 9;

// Intra prediction modes (6.10.22); `UV_CFL_PRED` only occurs as `uv_mode`.
pub const DC_PRED: usize = 0;
pub const V_PRED: usize = 1;
pub const H_PRED: usize = 2;
pub const D45_PRED: usize = 3;
pub const D135_PRED: usize = 4;
pub const D113_PRED: usize = 5;
pub const D157_PRED: usize = 6;
pub const D203_PRED: usize = 7;
pub const D67_PRED: usize = 8;
pub const SMOOTH_PRED: usize = 9;
pub const SMOOTH_V_PRED: usize = 10;
pub const SMOOTH_H_PRED: usize = 11;
pub const PAETH_PRED: usize = 12;
pub const UV_CFL_PRED: usize = 13;

// Filter intra modes (6.10.23).
pub const FILTER_DC_PRED: usize = 0;
pub const FILTER_V_PRED: usize = 1;
pub const FILTER_H_PRED: usize = 2;
pub const FILTER_D157_PRED: usize = 3;
pub const FILTER_PAETH_PRED: usize = 4;

// Transform sizes (6.10.17), in spec order. The transform types `DCT_DCT` .. `H_FLIPADST` are in
// section 3.
pub const TX_4X4: usize = 0;
pub const TX_8X8: usize = 1;
pub const TX_16X16: usize = 2;
pub const TX_32X32: usize = 3;
pub const TX_64X64: usize = 4;
pub const TX_4X8: usize = 5;
pub const TX_8X4: usize = 6;
pub const TX_8X16: usize = 7;
pub const TX_16X8: usize = 8;
pub const TX_16X32: usize = 9;
pub const TX_32X16: usize = 10;
pub const TX_32X64: usize = 11;
pub const TX_64X32: usize = 12;
pub const TX_4X16: usize = 13;
pub const TX_16X4: usize = 14;
pub const TX_8X32: usize = 15;
pub const TX_32X8: usize = 16;
pub const TX_16X64: usize = 17;
pub const TX_64X16: usize = 18;

// Transform modes (6.8.21).
pub const ONLY_4X4: u8 = 0;
pub const TX_MODE_LARGEST: u8 = 1;
pub const TX_MODE_SELECT: u8 = 2;

// Transform sets (6.10.15, `get_tx_set`).
pub const TX_SET_DCTONLY: usize = 0;
pub const TX_SET_INTRA_1: usize = 1;
pub const TX_SET_INTRA_2: usize = 2;
pub const TX_SET_INTER_1: usize = 1;
pub const TX_SET_INTER_2: usize = 2;
pub const TX_SET_INTER_3: usize = 3;

// Loop restoration types (the FrameRestorationType semantics). The syntax element `lr_type` maps to
// these through the spec's `Remap_Lr_Type` (see `crate::header::REMAP_LR_TYPE`).
pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_WIENER: u8 = 1;
pub const RESTORE_SGRPROJ: u8 = 2;
pub const RESTORE_SWITCHABLE: u8 = 3;
