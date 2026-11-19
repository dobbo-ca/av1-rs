//! The frame header OBU (spec 5.9) for intra frames, the reference slot state it reads and
//! updates (7.20, 7.21), and the tile group header (5.11.1).
//!
//! Only `KEY_FRAME` and `INTRA_ONLY_FRAME` are parsed. An `INTER_FRAME` or `SWITCH_FRAME` is
//! `Error::InterFrame` as soon as `frame_type` is known. Because an intra frame always has
//! `primary_ref_frame == PRIMARY_REF_NONE`, `load_previous()` never runs: the loop filter deltas
//! and segmentation features start from the `setup_past_independence()` defaults.
#![allow(dead_code)]

use crate::bits::BitReader;
use crate::consts::*;
use crate::error::Error;
use crate::seq::SequenceHeader;

/// The film grain syntax elements of spec 5.9.30, stored as read (the `_plus_128` and `_minus_8`
/// forms are kept as coded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub film_grain_params_ref_idx: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; 16],
    pub point_y_scaling: [u8; 16],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 16],
    pub point_cb_scaling: [u8; 16],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 16],
    pub point_cr_scaling: [u8; 16],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// `Remap_Lr_Type`: the `lr_type` syntax element to `FrameRestorationType`.
pub(crate) const REMAP_LR_TYPE: [u8; 4] = [
    RESTORE_NONE,
    RESTORE_SWITCHABLE,
    RESTORE_WIENER,
    RESTORE_SGRPROJ,
];

/// `Segmentation_Feature_Bits`.
pub(crate) const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
/// `Segmentation_Feature_Signed`.
pub(crate) const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
/// `Segmentation_Feature_Max`.
pub(crate) const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [
    255,
    MAX_LOOP_FILTER as i32,
    MAX_LOOP_FILTER as i32,
    MAX_LOOP_FILTER as i32,
    MAX_LOOP_FILTER as i32,
    7,
    0,
    0,
];

/// The default `loop_filter_ref_deltas` of `setup_past_independence()`, indexed by reference
/// frame (`INTRA_FRAME` first).
const DEFAULT_LOOP_FILTER_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];

/// Everything `tile_info()` computes (spec 5.9.15).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TileInfo {
    pub tile_cols: usize,
    pub tile_rows: usize,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    /// `TileCols + 1` entries; the last is `MiCols`.
    pub mi_col_starts: Vec<u32>,
    /// `TileRows + 1` entries; the last is `MiRows`.
    pub mi_row_starts: Vec<u32>,
    pub context_update_tile_id: usize,
    pub tile_size_bytes: u32,
}

/// A parsed `frame_header_obu()` of an intra frame, or of a `show_existing_frame`. Names are the
/// spec's in snake case. For a `show_existing_frame` only the first group of fields (up to
/// `film_grain`, as loaded) is meaningful; the rest is zero.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: usize,
    pub frame_type: u8,
    pub frame_is_intra: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: usize,
    pub refresh_frame_flags: u8,
    /// `FrameWidth`: the coded width, after superres downscaling.
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    pub superres_denom: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub allow_intrabc: bool,
    pub disable_frame_end_update_cdf: bool,
    pub film_grain: FilmGrainParams,

    pub tile_info: TileInfo,

    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,

    pub segmentation_enabled: bool,
    pub segmentation_update_map: bool,
    pub segmentation_temporal_update: bool,
    pub segmentation_update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,

    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,

    pub coded_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    /// `SegQMLevel[plane][segment_id]`, only set when `using_qmatrix`.
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub all_lossless: bool,

    pub loop_filter_level: [u8; 4],
    pub loop_filter_sharpness: u8,
    pub loop_filter_delta_enabled: bool,
    pub loop_filter_delta_update: bool,
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub loop_filter_mode_deltas: [i8; 2],

    /// `CdefDamping`.
    pub cdef_damping: u8,
    pub cdef_bits: u8,
    pub cdef_y_pri_strength: [u8; 8],
    pub cdef_y_sec_strength: [u8; 8],
    pub cdef_uv_pri_strength: [u8; 8],
    pub cdef_uv_sec_strength: [u8; 8],

    pub frame_restoration_type: [u8; 3],
    pub loop_restoration_size: [u16; 3],
    pub uses_lr: bool,

    /// `TxMode`.
    pub tx_mode: u8,
    pub reduced_tx_set: bool,
}

/// What the decoder remembers about one reference slot: the `Ref*` and `Saved*` arrays of spec
/// 7.20 that an intra-only decoder can use. (The pixels are kept by the decoder.)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RefInfo {
    pub ref_valid: bool,
    pub ref_frame_id: u32,
    pub ref_upscaled_width: u32,
    pub ref_frame_width: u32,
    pub ref_frame_height: u32,
    pub ref_render_width: u32,
    pub ref_render_height: u32,
    pub ref_mi_cols: u32,
    pub ref_mi_rows: u32,
    pub ref_frame_type: u8,
    pub ref_showable_frame: bool,
    pub ref_subsampling_x: u8,
    pub ref_subsampling_y: u8,
    pub ref_bit_depth: u8,
    pub ref_order_hint: u32,
    pub saved_film_grain: FilmGrainParams,
    pub saved_loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub saved_loop_filter_mode_deltas: [i8; 2],
    pub saved_feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub saved_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// The eight reference slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RefState {
    pub slots: [RefInfo; NUM_REF_FRAMES],
}

impl RefState {
    /// The header half of the reference frame update process (spec 7.20) for a decoded frame:
    /// every slot whose bit is set in `refresh_frame_flags` takes this frame's state.
    pub(crate) fn update(&mut self, h: &FrameHeader, seq: &SequenceHeader) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if (h.refresh_frame_flags >> i) & 1 == 0 {
                continue;
            }
            *slot = RefInfo {
                ref_valid: true,
                ref_frame_id: h.current_frame_id,
                ref_upscaled_width: h.upscaled_width,
                ref_frame_width: h.frame_width,
                ref_frame_height: h.frame_height,
                ref_render_width: h.render_width,
                ref_render_height: h.render_height,
                ref_mi_cols: h.mi_cols,
                ref_mi_rows: h.mi_rows,
                ref_frame_type: h.frame_type,
                ref_showable_frame: h.showable_frame,
                ref_subsampling_x: seq.subsampling_x,
                ref_subsampling_y: seq.subsampling_y,
                ref_bit_depth: seq.bit_depth,
                ref_order_hint: h.order_hint,
                saved_film_grain: if seq.film_grain_params_present {
                    h.film_grain
                } else {
                    FilmGrainParams::default()
                },
                saved_loop_filter_ref_deltas: h.loop_filter_ref_deltas,
                saved_loop_filter_mode_deltas: h.loop_filter_mode_deltas,
                saved_feature_enabled: h.feature_enabled,
                saved_feature_data: h.feature_data,
            };
        }
    }

    /// The effect of a shown key frame through `show_existing_frame` (the reference frame
    /// loading process of 7.21 followed by the update process with `refresh_frame_flags` equal to
    /// all frames): every slot becomes a copy of slot `idx`. `idx` is below `NUM_REF_FRAMES`.
    pub(crate) fn refresh_all_from(&mut self, idx: usize) {
        let src = self.slots[idx].clone();
        for slot in self.slots.iter_mut() {
            *slot = src.clone();
        }
    }
}

/// `tile_log2( blkSize, target )` of spec 5.9.16.
fn tile_log2(blk_size: usize, target: usize) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Reads one `read_delta_q()` (spec 5.9.13).
fn read_delta_q(r: &mut BitReader<'_>) -> Result<i8, Error> {
    if r.flag()? { Ok(r.su(7)? as i8) } else { Ok(0) }
}

impl FrameHeader {
    /// Parses `uncompressed_header()` (spec 5.9.2) from `r`, which stands at the first bit of the
    /// frame header. `seq` is the active sequence header, `refs` the reference slot state, which
    /// this changes where the spec does (`RefValid`, `RefOrderHint` on a shown key frame).
    /// `temporal_id` and `spatial_id` come from the OBU extension header.
    ///
    /// The reader is left right after the last header bit; the caller does `byte_alignment()`
    /// for an `OBU_FRAME`.
    ///
    /// # Errors
    /// `InterFrame` for `INTER_FRAME` and `SWITCH_FRAME`, `Truncated` if the data ends early,
    /// `Invalid` for values the spec forbids: a frame larger than the sequence maximum, an intra
    /// only frame refreshing all slots, too many tiles, a `show_existing_frame` of an empty or
    /// not showable slot.
    pub(crate) fn parse(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        refs: &mut RefState,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<FrameHeader, Error> {
        let mut h = FrameHeader {
            temporal_id,
            spatial_id,
            primary_ref_frame: PRIMARY_REF_NONE,
            // The `setup_past_independence()` defaults.
            loop_filter_delta_enabled: true,
            loop_filter_ref_deltas: DEFAULT_LOOP_FILTER_REF_DELTAS,
            ..FrameHeader::default()
        };
        let id_len = if seq.frame_id_numbers_present_flag {
            u32::from(seq.additional_frame_id_length_minus_1)
                + u32::from(seq.delta_frame_id_length_minus_2)
                + 3
        } else {
            0
        };
        let all_frames = ((1u32 << NUM_REF_FRAMES) - 1) as u8;
        if seq.reduced_still_picture_header {
            h.frame_type = KEY_FRAME;
            h.frame_is_intra = true;
            h.show_frame = true;
            h.error_resilient_mode = true;
        } else {
            h.show_existing_frame = r.flag()?;
            if h.show_existing_frame {
                h.frame_to_show_map_idx = r.f(3)? as usize;
                if seq.decoder_model_info_present_flag && !seq.equal_picture_interval {
                    r.f(u32::from(seq.frame_presentation_time_length_minus_1) + 1)?;
                }
                if seq.frame_id_numbers_present_flag {
                    r.f(id_len)?; // display_frame_id
                }
                let slot = &refs.slots[h.frame_to_show_map_idx];
                if !slot.ref_valid {
                    return Err(Error::Invalid("show_existing_frame of an empty slot"));
                }
                if !slot.ref_showable_frame {
                    return Err(Error::Invalid(
                        "show_existing_frame of a frame that is not showable",
                    ));
                }
                h.frame_type = slot.ref_frame_type;
                if h.frame_type == KEY_FRAME {
                    h.refresh_frame_flags = all_frames;
                }
                if seq.film_grain_params_present {
                    h.film_grain = slot.saved_film_grain;
                }
                return Ok(h);
            }
            h.frame_type = r.f(2)? as u8;
            if h.frame_type == INTER_FRAME || h.frame_type == SWITCH_FRAME {
                return Err(Error::InterFrame);
            }
            h.frame_is_intra = true;
            h.show_frame = r.flag()?;
            if h.show_frame && seq.decoder_model_info_present_flag && !seq.equal_picture_interval {
                r.f(u32::from(seq.frame_presentation_time_length_minus_1) + 1)?;
            }
            h.showable_frame = if h.show_frame {
                h.frame_type != KEY_FRAME
            } else {
                r.flag()?
            };
            h.error_resilient_mode = if h.frame_type == KEY_FRAME && h.show_frame {
                true
            } else {
                r.flag()?
            };
        }
        if h.frame_type == KEY_FRAME && h.show_frame {
            for slot in refs.slots.iter_mut() {
                slot.ref_valid = false;
                slot.ref_order_hint = 0;
            }
        }
        h.disable_cdf_update = r.flag()?;
        h.allow_screen_content_tools = if u32::from(seq.seq_force_screen_content_tools)
            == SELECT_SCREEN_CONTENT_TOOLS as u32
        {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        if h.allow_screen_content_tools && seq.seq_force_integer_mv == SELECT_INTEGER_MV as u8 {
            r.flag()?; // force_integer_mv; an intra frame has it equal to 1 whatever is coded
        }
        h.force_integer_mv = true;
        if seq.frame_id_numbers_present_flag {
            h.current_frame_id = r.f(id_len)?;
            mark_ref_frames(refs, h.current_frame_id, id_len, seq);
        }
        h.frame_size_override_flag = if seq.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };
        h.order_hint = r.f(u32::from(seq.order_hint_bits))?;
        if seq.decoder_model_info_present_flag {
            let buffer_removal_time_present_flag = r.flag()?;
            if buffer_removal_time_present_flag {
                for op_num in 0..=usize::from(seq.operating_points_cnt_minus_1) {
                    if !seq.decoder_model_present_for_this_op[op_num] {
                        continue;
                    }
                    let op_pt_idc = u32::from(seq.operating_point_idc[op_num]);
                    let in_temporal_layer = (op_pt_idc >> temporal_id) & 1 != 0;
                    let in_spatial_layer = (op_pt_idc >> (u32::from(spatial_id) + 8)) & 1 != 0;
                    if op_pt_idc == 0 || (in_temporal_layer && in_spatial_layer) {
                        r.f(u32::from(seq.buffer_removal_time_length_minus_1) + 1)?;
                    }
                }
            }
        }
        h.refresh_frame_flags = if h.frame_type == KEY_FRAME && h.show_frame {
            all_frames
        } else {
            r.f(8)? as u8
        };
        if h.frame_type == INTRA_ONLY_FRAME && h.refresh_frame_flags == all_frames {
            return Err(Error::Invalid("intra only frame refreshes every slot"));
        }
        if h.refresh_frame_flags != all_frames && h.error_resilient_mode && seq.enable_order_hint {
            for slot in refs.slots.iter_mut() {
                let ref_order_hint = r.f(u32::from(seq.order_hint_bits))?;
                if ref_order_hint != slot.ref_order_hint {
                    slot.ref_valid = false;
                }
            }
        }
        h.frame_size(r, seq)?;
        h.render_size(r)?;
        if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
            h.allow_intrabc = r.flag()?;
        }
        h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || h.disable_cdf_update
        {
            true
        } else {
            r.flag()?
        };
        h.tile_info = tile_info(r, seq, h.mi_cols as usize, h.mi_rows as usize)?;
        h.quantization_params(r, seq)?;
        h.segmentation_params(r)?;
        h.delta_q_params(r)?;
        h.delta_lf_params(r)?;
        h.lossless();
        h.loop_filter_params(r, seq)?;
        h.cdef_params(r, seq)?;
        h.lr_params(r, seq)?;
        h.tx_mode = if h.coded_lossless {
            ONLY_4X4
        } else if r.flag()? {
            TX_MODE_SELECT
        } else {
            TX_MODE_LARGEST
        };
        // frame_reference_mode, skip_mode_params, global_motion_params and allow_warped_motion
        // read nothing in an intra frame.
        h.reduced_tx_set = r.flag()?;
        h.film_grain_params(r, seq, refs)?;
        Ok(h)
    }

    /// `frame_size()` with `superres_params()` and `compute_image_size()` (spec 5.9.5, 5.9.8,
    /// 5.9.9).
    fn frame_size(&mut self, r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<(), Error> {
        if self.frame_size_override_flag {
            let frame_width_minus_1 = r.f(u32::from(seq.frame_width_bits_minus_1) + 1)?;
            let frame_height_minus_1 = r.f(u32::from(seq.frame_height_bits_minus_1) + 1)?;
            if frame_width_minus_1 > seq.max_frame_width_minus_1
                || frame_height_minus_1 > seq.max_frame_height_minus_1
            {
                return Err(Error::Invalid("frame size above the sequence maximum"));
            }
            self.frame_width = frame_width_minus_1 + 1;
            self.frame_height = frame_height_minus_1 + 1;
        } else {
            self.frame_width = seq.max_frame_width_minus_1 + 1;
            self.frame_height = seq.max_frame_height_minus_1 + 1;
        }
        // superres_params()
        self.use_superres = seq.enable_superres && r.flag()?;
        self.superres_denom = if self.use_superres {
            r.f(SUPERRES_DENOM_BITS as u32)? + SUPERRES_DENOM_MIN as u32
        } else {
            SUPERRES_NUM as u32
        };
        self.upscaled_width = self.frame_width;
        self.frame_width = (self.upscaled_width * SUPERRES_NUM as u32 + self.superres_denom / 2)
            / self.superres_denom;
        // compute_image_size()
        self.mi_cols = 2 * ((self.frame_width + 7) >> 3);
        self.mi_rows = 2 * ((self.frame_height + 7) >> 3);
        Ok(())
    }

    /// `render_size()` (spec 5.9.6).
    fn render_size(&mut self, r: &mut BitReader<'_>) -> Result<(), Error> {
        if r.flag()? {
            self.render_width = r.f(16)? + 1;
            self.render_height = r.f(16)? + 1;
        } else {
            self.render_width = self.upscaled_width;
            self.render_height = self.frame_height;
        }
        Ok(())
    }

    /// `quantization_params()` (spec 5.9.12).
    fn quantization_params(
        &mut self,
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> Result<(), Error> {
        self.base_q_idx = r.f(8)? as u8;
        self.delta_q_y_dc = read_delta_q(r)?;
        if seq.num_planes > 1 {
            let diff_uv_delta = seq.separate_uv_delta_q && r.flag()?;
            self.delta_q_u_dc = read_delta_q(r)?;
            self.delta_q_u_ac = read_delta_q(r)?;
            if diff_uv_delta {
                self.delta_q_v_dc = read_delta_q(r)?;
                self.delta_q_v_ac = read_delta_q(r)?;
            } else {
                self.delta_q_v_dc = self.delta_q_u_dc;
                self.delta_q_v_ac = self.delta_q_u_ac;
            }
        }
        self.using_qmatrix = r.flag()?;
        if self.using_qmatrix {
            self.qm_y = r.f(4)? as u8;
            self.qm_u = r.f(4)? as u8;
            self.qm_v = if seq.separate_uv_delta_q {
                r.f(4)? as u8
            } else {
                self.qm_u
            };
        }
        Ok(())
    }

    /// `segmentation_params()` (spec 5.9.14) for a frame with `primary_ref_frame` equal to
    /// `PRIMARY_REF_NONE`: when enabled, the map and the data are always updated.
    fn segmentation_params(&mut self, r: &mut BitReader<'_>) -> Result<(), Error> {
        self.segmentation_enabled = r.flag()?;
        if self.segmentation_enabled {
            self.segmentation_update_map = true;
            self.segmentation_temporal_update = false;
            self.segmentation_update_data = true;
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let feature_enabled = r.flag()?;
                    self.feature_enabled[i][j] = feature_enabled;
                    let mut clipped_value = 0;
                    if feature_enabled {
                        let bits_to_read = SEGMENTATION_FEATURE_BITS[j];
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        if SEGMENTATION_FEATURE_SIGNED[j] {
                            let feature_value = r.su(1 + bits_to_read)?;
                            clipped_value = feature_value.clamp(-limit, limit);
                        } else {
                            let feature_value = r.f(bits_to_read)? as i32;
                            clipped_value = feature_value.clamp(0, limit);
                        }
                    }
                    self.feature_data[i][j] = clipped_value as i16;
                }
            }
        }
        self.seg_id_pre_skip = false;
        self.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if self.feature_enabled[i][j] {
                    self.last_active_seg_id = i as u8;
                    if j >= SEG_LVL_REF_FRAME {
                        self.seg_id_pre_skip = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// `delta_q_params()` (spec 5.9.17).
    fn delta_q_params(&mut self, r: &mut BitReader<'_>) -> Result<(), Error> {
        if self.base_q_idx > 0 {
            self.delta_q_present = r.flag()?;
        }
        if self.delta_q_present {
            self.delta_q_res = r.f(2)? as u8;
        }
        Ok(())
    }

    /// `delta_lf_params()` (spec 5.9.18).
    fn delta_lf_params(&mut self, r: &mut BitReader<'_>) -> Result<(), Error> {
        if self.delta_q_present {
            if !self.allow_intrabc {
                self.delta_lf_present = r.flag()?;
            }
            if self.delta_lf_present {
                self.delta_lf_res = r.f(2)? as u8;
                self.delta_lf_multi = r.flag()?;
            }
        }
        Ok(())
    }

    /// `seg_feature_active_idx( idx, feature )` of spec 5.11.14.
    pub(crate) fn seg_feature_active_idx(&self, idx: usize, feature: usize) -> bool {
        self.segmentation_enabled && self.feature_enabled[idx][feature]
    }

    /// `get_qindex( ignoreDeltaQ, segmentId )` of spec 7.12.2. `current_q_index` is the
    /// `CurrentQIndex` of the tile decode (only used when `ignore_delta_q` is false).
    /// `segment_id` is below `MAX_SEGMENTS`.
    pub(crate) fn get_qindex(
        &self,
        ignore_delta_q: bool,
        segment_id: usize,
        current_q_index: i32,
    ) -> i32 {
        if self.seg_feature_active_idx(segment_id, SEG_LVL_ALT_Q) {
            let data = i32::from(self.feature_data[segment_id][SEG_LVL_ALT_Q]);
            let mut qindex = i32::from(self.base_q_idx) + data;
            if !ignore_delta_q && self.delta_q_present {
                qindex = current_q_index + data;
            }
            qindex.clamp(0, 255)
        } else if !ignore_delta_q && self.delta_q_present {
            current_q_index
        } else {
            i32::from(self.base_q_idx)
        }
    }

    /// The `CodedLossless`, `LosslessArray`, `SegQMLevel` and `AllLossless` part of
    /// `uncompressed_header()`.
    fn lossless(&mut self) {
        self.coded_lossless = true;
        for segment_id in 0..MAX_SEGMENTS {
            let qindex = self.get_qindex(true, segment_id, 0);
            let lossless = qindex == 0
                && self.delta_q_y_dc == 0
                && self.delta_q_u_ac == 0
                && self.delta_q_u_dc == 0
                && self.delta_q_v_ac == 0
                && self.delta_q_v_dc == 0;
            self.lossless_array[segment_id] = lossless;
            if !lossless {
                self.coded_lossless = false;
            }
            if self.using_qmatrix {
                let levels = if lossless {
                    [15, 15, 15]
                } else {
                    [self.qm_y, self.qm_u, self.qm_v]
                };
                for plane in 0..3 {
                    self.seg_qm_level[plane][segment_id] = levels[plane];
                }
            }
        }
        self.all_lossless = self.coded_lossless && self.frame_width == self.upscaled_width;
    }

    /// `loop_filter_params()` (spec 5.9.11).
    fn loop_filter_params(
        &mut self,
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> Result<(), Error> {
        if self.coded_lossless || self.allow_intrabc {
            self.loop_filter_level[0] = 0;
            self.loop_filter_level[1] = 0;
            self.loop_filter_ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
            self.loop_filter_mode_deltas = [0; 2];
            return Ok(());
        }
        self.loop_filter_level[0] = r.f(6)? as u8;
        self.loop_filter_level[1] = r.f(6)? as u8;
        if seq.num_planes > 1 && (self.loop_filter_level[0] != 0 || self.loop_filter_level[1] != 0)
        {
            self.loop_filter_level[2] = r.f(6)? as u8;
            self.loop_filter_level[3] = r.f(6)? as u8;
        }
        self.loop_filter_sharpness = r.f(3)? as u8;
        self.loop_filter_delta_enabled = r.flag()?;
        if self.loop_filter_delta_enabled {
            self.loop_filter_delta_update = r.flag()?;
            if self.loop_filter_delta_update {
                for i in 0..TOTAL_REFS_PER_FRAME {
                    if r.flag()? {
                        self.loop_filter_ref_deltas[i] = r.su(1 + 6)? as i8;
                    }
                }
                for i in 0..2 {
                    if r.flag()? {
                        self.loop_filter_mode_deltas[i] = r.su(1 + 6)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    /// `cdef_params()` (spec 5.9.19).
    fn cdef_params(&mut self, r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<(), Error> {
        if self.coded_lossless || self.allow_intrabc || !seq.enable_cdef {
            self.cdef_bits = 0;
            self.cdef_y_pri_strength[0] = 0;
            self.cdef_y_sec_strength[0] = 0;
            self.cdef_uv_pri_strength[0] = 0;
            self.cdef_uv_sec_strength[0] = 0;
            self.cdef_damping = 3;
            return Ok(());
        }
        self.cdef_damping = r.f(2)? as u8 + 3;
        self.cdef_bits = r.f(2)? as u8;
        for i in 0..(1usize << self.cdef_bits) {
            self.cdef_y_pri_strength[i] = r.f(4)? as u8;
            self.cdef_y_sec_strength[i] = r.f(2)? as u8;
            if self.cdef_y_sec_strength[i] == 3 {
                self.cdef_y_sec_strength[i] += 1;
            }
            if seq.num_planes > 1 {
                self.cdef_uv_pri_strength[i] = r.f(4)? as u8;
                self.cdef_uv_sec_strength[i] = r.f(2)? as u8;
                if self.cdef_uv_sec_strength[i] == 3 {
                    self.cdef_uv_sec_strength[i] += 1;
                }
            }
        }
        Ok(())
    }

    /// `lr_params()` (spec 5.9.20).
    fn lr_params(&mut self, r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<(), Error> {
        self.frame_restoration_type = [RESTORE_NONE; 3];
        self.uses_lr = false;
        if self.all_lossless || self.allow_intrabc || !seq.enable_restoration {
            return Ok(());
        }
        let mut uses_chroma_lr = false;
        for i in 0..usize::from(seq.num_planes) {
            let lr_type = r.f(2)? as usize;
            self.frame_restoration_type[i] = REMAP_LR_TYPE[lr_type];
            if self.frame_restoration_type[i] != RESTORE_NONE {
                self.uses_lr = true;
                if i > 0 {
                    uses_chroma_lr = true;
                }
            }
        }
        if self.uses_lr {
            let mut lr_unit_shift = r.f(1)?;
            if seq.use_128x128_superblock {
                lr_unit_shift += 1;
            } else if lr_unit_shift != 0 {
                lr_unit_shift += r.f(1)?; // lr_unit_extra_shift
            }
            self.loop_restoration_size[0] =
                (RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift)) as u16;
            let lr_uv_shift = if seq.subsampling_x != 0 && seq.subsampling_y != 0 && uses_chroma_lr
            {
                r.f(1)?
            } else {
                0
            };
            self.loop_restoration_size[1] = self.loop_restoration_size[0] >> lr_uv_shift;
            self.loop_restoration_size[2] = self.loop_restoration_size[0] >> lr_uv_shift;
        }
        Ok(())
    }

    /// `film_grain_params()` (spec 5.9.30). `refs` supplies `load_grain_params()`.
    fn film_grain_params(
        &mut self,
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        refs: &RefState,
    ) -> Result<(), Error> {
        self.film_grain = FilmGrainParams::default();
        if !seq.film_grain_params_present || (!self.show_frame && !self.showable_frame) {
            return Ok(());
        }
        let mut g = FilmGrainParams {
            apply_grain: r.flag()?,
            ..FilmGrainParams::default()
        };
        if !g.apply_grain {
            return Ok(());
        }
        g.grain_seed = r.f(16)? as u16;
        g.update_grain = if self.frame_type == INTER_FRAME {
            r.flag()?
        } else {
            true
        };
        if !g.update_grain {
            g.film_grain_params_ref_idx = r.f(3)? as u8;
            let temp_grain_seed = g.grain_seed;
            g = refs.slots[usize::from(g.film_grain_params_ref_idx)].saved_film_grain;
            g.grain_seed = temp_grain_seed;
            self.film_grain = g;
            return Ok(());
        }
        g.num_y_points = r.f(4)? as u8;
        if g.num_y_points > 14 {
            return Err(Error::Invalid("num_y_points above 14"));
        }
        for i in 0..usize::from(g.num_y_points) {
            g.point_y_value[i] = r.f(8)? as u8;
            g.point_y_scaling[i] = r.f(8)? as u8;
        }
        g.chroma_scaling_from_luma = !seq.mono_chrome && r.flag()?;
        if seq.mono_chrome
            || g.chroma_scaling_from_luma
            || (seq.subsampling_x == 1 && seq.subsampling_y == 1 && g.num_y_points == 0)
        {
            g.num_cb_points = 0;
            g.num_cr_points = 0;
        } else {
            g.num_cb_points = r.f(4)? as u8;
            if g.num_cb_points > 10 {
                return Err(Error::Invalid("num_cb_points above 10"));
            }
            for i in 0..usize::from(g.num_cb_points) {
                g.point_cb_value[i] = r.f(8)? as u8;
                g.point_cb_scaling[i] = r.f(8)? as u8;
            }
            g.num_cr_points = r.f(4)? as u8;
            if g.num_cr_points > 10 {
                return Err(Error::Invalid("num_cr_points above 10"));
            }
            for i in 0..usize::from(g.num_cr_points) {
                g.point_cr_value[i] = r.f(8)? as u8;
                g.point_cr_scaling[i] = r.f(8)? as u8;
            }
        }
        g.grain_scaling_minus_8 = r.f(2)? as u8;
        g.ar_coeff_lag = r.f(2)? as u8;
        let num_pos_luma = 2 * usize::from(g.ar_coeff_lag) * (usize::from(g.ar_coeff_lag) + 1);
        let mut num_pos_chroma = num_pos_luma;
        if g.num_y_points != 0 {
            num_pos_chroma += 1;
            for i in 0..num_pos_luma {
                g.ar_coeffs_y_plus_128[i] = r.f(8)? as u8;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cb_plus_128[i] = r.f(8)? as u8;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cr_plus_128[i] = r.f(8)? as u8;
            }
        }
        g.ar_coeff_shift_minus_6 = r.f(2)? as u8;
        g.grain_scale_shift = r.f(2)? as u8;
        if g.num_cb_points != 0 {
            g.cb_mult = r.f(8)? as u8;
            g.cb_luma_mult = r.f(8)? as u8;
            g.cb_offset = r.f(9)? as u16;
        }
        if g.num_cr_points != 0 {
            g.cr_mult = r.f(8)? as u8;
            g.cr_luma_mult = r.f(8)? as u8;
            g.cr_offset = r.f(9)? as u16;
        }
        g.overlap_flag = r.flag()?;
        g.clip_to_restricted_range = r.flag()?;
        self.film_grain = g;
        Ok(())
    }
}

/// `mark_ref_frames( idLen )` (spec 5.9.4): slots whose frame id is too far from
/// `current_frame_id` are no longer valid.
fn mark_ref_frames(refs: &mut RefState, current_frame_id: u32, id_len: u32, seq: &SequenceHeader) {
    let diff_len = u32::from(seq.delta_frame_id_length_minus_2) + 2;
    for slot in refs.slots.iter_mut() {
        let id = u64::from(slot.ref_frame_id);
        let cur = u64::from(current_frame_id);
        if cur > (1u64 << diff_len) {
            if !(cur - (1u64 << diff_len)..=cur).contains(&id) {
                slot.ref_valid = false;
            }
        } else if id > cur && id < (1u64 << id_len) + cur - (1u64 << diff_len) {
            slot.ref_valid = false;
        }
    }
}

/// `tile_info()` (spec 5.9.15).
///
/// # Errors
/// `Truncated`, or `Invalid` for more than `MAX_TILE_COLS` or `MAX_TILE_ROWS` tiles, tile sizes
/// that do not add up, or a `context_update_tile_id` outside the tile list.
fn tile_info(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    mi_cols: usize,
    mi_rows: usize,
) -> Result<TileInfo, Error> {
    let sb128 = seq.use_128x128_superblock;
    let sb_cols = if sb128 {
        (mi_cols + 31) >> 5
    } else {
        (mi_cols + 15) >> 4
    };
    let sb_rows = if sb128 {
        (mi_rows + 31) >> 5
    } else {
        (mi_rows + 15) >> 4
    };
    let sb_shift = if sb128 { 5 } else { 4 };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let mut t = TileInfo::default();
    let uniform_tile_spacing_flag = r.flag()?;
    if uniform_tile_spacing_flag {
        t.tile_cols_log2 = min_log2_tile_cols;
        while t.tile_cols_log2 < max_log2_tile_cols {
            if r.flag()? {
                t.tile_cols_log2 += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1 << t.tile_cols_log2) - 1) >> t.tile_cols_log2;
        let mut start_sb = 0;
        while start_sb < sb_cols {
            t.mi_col_starts.push((start_sb << sb_shift) as u32);
            start_sb += tile_width_sb;
        }
        t.tile_cols = t.mi_col_starts.len();
        t.mi_col_starts.push(mi_cols as u32);

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.tile_cols_log2);
        t.tile_rows_log2 = min_log2_tile_rows;
        while t.tile_rows_log2 < max_log2_tile_rows {
            if r.flag()? {
                t.tile_rows_log2 += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1 << t.tile_rows_log2) - 1) >> t.tile_rows_log2;
        let mut start_sb = 0;
        while start_sb < sb_rows {
            t.mi_row_starts.push((start_sb << sb_shift) as u32);
            start_sb += tile_height_sb;
        }
        t.tile_rows = t.mi_row_starts.len();
        t.mi_row_starts.push(mi_rows as u32);
    } else {
        let mut widest_tile_sb = 0;
        let mut start_sb = 0;
        while start_sb < sb_cols {
            if t.mi_col_starts.len() >= MAX_TILE_COLS {
                return Err(Error::Invalid("more than 64 tile columns"));
            }
            t.mi_col_starts.push((start_sb << sb_shift) as u32);
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            let size_sb = r.ns(max_width as u32)? as usize + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start_sb += size_sb;
        }
        t.tile_cols = t.mi_col_starts.len();
        t.mi_col_starts.push(mi_cols as u32);
        t.tile_cols_log2 = tile_log2(1, t.tile_cols);

        if min_log2_tiles > 0 {
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
        } else {
            max_tile_area_sb = sb_rows * sb_cols;
        }
        let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);

        let mut start_sb = 0;
        while start_sb < sb_rows {
            if t.mi_row_starts.len() >= MAX_TILE_ROWS {
                return Err(Error::Invalid("more than 64 tile rows"));
            }
            t.mi_row_starts.push((start_sb << sb_shift) as u32);
            let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
            let size_sb = r.ns(max_height as u32)? as usize + 1;
            start_sb += size_sb;
        }
        t.tile_rows = t.mi_row_starts.len();
        t.mi_row_starts.push(mi_rows as u32);
        t.tile_rows_log2 = tile_log2(1, t.tile_rows);
    }
    if t.tile_cols > MAX_TILE_COLS || t.tile_rows > MAX_TILE_ROWS {
        return Err(Error::Invalid("too many tiles"));
    }
    if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
        t.context_update_tile_id = r.f(t.tile_rows_log2 + t.tile_cols_log2)? as usize;
        if t.context_update_tile_id >= t.tile_cols * t.tile_rows {
            return Err(Error::Invalid("context_update_tile_id out of range"));
        }
        t.tile_size_bytes = r.f(2)? + 1;
    }
    Ok(t)
}

/// The byte range of one tile's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileRange {
    /// `TileNum`: the tile's index in raster order.
    pub tile_num: usize,
    /// Start of the tile data, as an offset into the data the caller passed in.
    pub start: usize,
    /// End (exclusive) of the tile data.
    pub end: usize,
}

/// The tiles of one tile group OBU (spec 5.11.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TileGroup {
    pub tg_start: usize,
    pub tg_end: usize,
    pub tiles: Vec<TileRange>,
}

/// Parses the header of a `tile_group_obu( sz )` and cuts out the byte range of each tile.
/// `data` is the tile group payload (for an `OBU_FRAME`, the bytes after the frame header and its
/// byte alignment); ranges are offsets into it plus `base`. `next_tile` is the `TileNum` the
/// group must start at.
///
/// # Errors
/// `Truncated` if the header or a tile size is cut off, `Invalid` if `tg_start` is not
/// `next_tile`, `tg_end` is before `tg_start` or past the last tile, a tile is larger than
/// what is left of the OBU, or the last tile is empty.
pub(crate) fn parse_tile_group(
    data: &[u8],
    base: usize,
    ti: &TileInfo,
    next_tile: usize,
) -> Result<TileGroup, Error> {
    let num_tiles = ti.tile_cols * ti.tile_rows;
    let mut r = BitReader::new(data);
    if num_tiles == 0 {
        return Err(Error::Invalid("no tiles"));
    }
    let tile_start_and_end_present_flag = num_tiles > 1 && r.flag()?;
    let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present_flag {
        (0, num_tiles - 1)
    } else {
        let tile_bits = ti.tile_cols_log2 + ti.tile_rows_log2;
        (r.f(tile_bits)? as usize, r.f(tile_bits)? as usize)
    };
    r.byte_alignment()?;
    if tg_start != next_tile {
        return Err(Error::Invalid("tg_start is not the next tile"));
    }
    if !(tg_start..num_tiles).contains(&tg_end) {
        return Err(Error::Invalid("tg_end out of range"));
    }
    let mut pos = r.get_position() / 8;
    let mut tiles = Vec::with_capacity(tg_end - tg_start + 1);
    for tile_num in tg_start..=tg_end {
        let left = data.len() - pos;
        let tile_size = if tile_num == tg_end {
            left
        } else {
            let mut tr = BitReader::new(&data[pos..]);
            let tile_size = tr.le(ti.tile_size_bytes)? as usize + 1;
            pos += ti.tile_size_bytes as usize;
            tile_size
        };
        let end = pos
            .checked_add(tile_size)
            .filter(|&e| e <= data.len())
            .ok_or(Error::Invalid("tile size passes the end of the OBU"))?;
        if tile_size == 0 {
            return Err(Error::Invalid("empty tile"));
        }
        tiles.push(TileRange {
            tile_num,
            start: base + pos,
            end: base + end,
        });
        pos = end;
    }
    Ok(TileGroup {
        tg_start,
        tg_end,
        tiles,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::seq::tests::{BitWriter, still_sequence_header};

    fn seq(sb128: bool) -> SequenceHeader {
        let mut s = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        s.use_128x128_superblock = sb128;
        s
    }

    #[test]
    fn single_tile_reads_no_bits() {
        // 64x32 frame: MiCols 16, MiRows 8, one 64x64 superblock; uniform spacing flag only.
        let mut w = BitWriter::default();
        w.put(1, 1);
        let mut r = BitReader::new(&w.bytes);
        let t = tile_info(&mut r, &seq(false), 16, 8).unwrap();
        assert_eq!((t.tile_cols, t.tile_rows), (1, 1));
        assert_eq!(t.mi_col_starts, [0, 16]);
        assert_eq!(t.mi_row_starts, [0, 8]);
        assert_eq!(r.get_position(), 1);
    }

    #[test]
    fn uniform_four_tile_columns() {
        // 1024x256: MiCols 256 (16 superblocks), MiRows 64 (4). Increment twice, stop, no row
        // increment, context_update_tile_id = 3 (2 bits), tile_size_bytes_minus_1 = 1.
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(3, 0b110);
        w.put(1, 0);
        w.put(2, 3);
        w.put(2, 1);
        let mut r = BitReader::new(&w.bytes);
        let t = tile_info(&mut r, &seq(false), 256, 64).unwrap();
        assert_eq!((t.tile_cols, t.tile_rows), (4, 1));
        assert_eq!((t.tile_cols_log2, t.tile_rows_log2), (2, 0));
        assert_eq!(t.mi_col_starts, [0, 64, 128, 192, 256]);
        assert_eq!(t.mi_row_starts, [0, 64]);
        assert_eq!((t.context_update_tile_id, t.tile_size_bytes), (3, 2));
    }

    #[test]
    fn explicit_tile_sizes() {
        // 256x128: 4 x 2 superblocks. Column widths 1 then 3 (ns(4) = 00, ns(3) = 11), one
        // row of height 2 (ns(2) = 1), context_update_tile_id 1 (1 bit), tile_size_bytes 4.
        let mut w = BitWriter::default();
        w.put(1, 0);
        w.put(2, 0b00);
        w.put(2, 0b11);
        w.put(1, 1);
        w.put(1, 1);
        w.put(2, 3);
        let mut r = BitReader::new(&w.bytes);
        let t = tile_info(&mut r, &seq(false), 64, 32).unwrap();
        assert_eq!((t.tile_cols, t.tile_rows), (2, 1));
        assert_eq!(t.mi_col_starts, [0, 16, 64]);
        assert_eq!(t.mi_row_starts, [0, 32]);
        assert_eq!((t.context_update_tile_id, t.tile_size_bytes), (1, 4));
    }

    #[test]
    fn large_frame_needs_more_tile_columns() {
        // 8192 wide with 64x64 superblocks: 128 superblocks and a tile is at most 64 wide, so
        // TileColsLog2 starts at 1. One increment bit (0) stops there; one row of superblocks
        // reads no row bits; then context_update_tile_id (1 bit) and tile_size_bytes_minus_1.
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(1, 0); // no increment of TileColsLog2
        w.put(1, 0); // context_update_tile_id
        w.put(2, 0); // tile_size_bytes_minus_1
        let mut r = BitReader::new(&w.bytes);
        let t = tile_info(&mut r, &seq(false), 2048, 16).unwrap();
        assert_eq!(t.tile_cols, 2);
        assert_eq!(t.mi_col_starts, [0, 1024, 2048]);
    }

    #[test]
    fn tile_group_single_tile_takes_all() {
        let ti = TileInfo {
            tile_cols: 1,
            tile_rows: 1,
            mi_col_starts: vec![0, 16],
            mi_row_starts: vec![0, 8],
            ..TileInfo::default()
        };
        let tg = parse_tile_group(&[1, 2, 3], 10, &ti, 0).unwrap();
        assert_eq!(
            tg.tiles,
            [TileRange {
                tile_num: 0,
                start: 10,
                end: 13
            }]
        );
        assert_eq!((tg.tg_start, tg.tg_end), (0, 0));
    }

    #[test]
    fn tile_group_two_tiles_with_sizes() {
        let ti = TileInfo {
            tile_cols: 2,
            tile_rows: 1,
            tile_cols_log2: 1,
            tile_size_bytes: 2,
            mi_col_starts: vec![0, 8, 16],
            mi_row_starts: vec![0, 8],
            ..TileInfo::default()
        };
        // flag 0 (no start/end), padded to a byte; first tile has 3 bytes; the last the rest.
        let data = [0x00, 0x02, 0x00, 7, 8, 9, 5, 6];
        let tg = parse_tile_group(&data, 0, &ti, 0).unwrap();
        assert_eq!(
            tg.tiles,
            [
                TileRange {
                    tile_num: 0,
                    start: 3,
                    end: 6
                },
                TileRange {
                    tile_num: 1,
                    start: 6,
                    end: 8
                }
            ]
        );
        // A first tile that claims more than what is there.
        let bad = [0x00, 0x09, 0x00, 7, 8];
        assert!(matches!(
            parse_tile_group(&bad, 0, &ti, 0),
            Err(Error::Invalid(_))
        ));
        // A group that starts at the wrong tile: flag 1, tg_start 1, tg_end 1.
        let late = [0b1110_0000, 5];
        assert!(matches!(
            parse_tile_group(&late, 0, &ti, 0),
            Err(Error::Invalid(_))
        ));
        let ok = parse_tile_group(&late, 0, &ti, 1).unwrap();
        assert_eq!(
            ok.tiles,
            [TileRange {
                tile_num: 1,
                start: 1,
                end: 2
            }]
        );
    }

    /// The frame header bits of the still picture of `still_sequence_header()` (64x32, key frame,
    /// reduced still picture header), without trailing bits.
    pub(crate) fn still_frame_header(w: &mut BitWriter) {
        w.put(1, 0); // disable_cdf_update
        w.put(1, 0); // allow_screen_content_tools (seq_force is SELECT: coded)
        w.put(1, 0); // render_and_frame_size_different (frame_size has no bits)
        // frame_size: no override, enable_superres = 0.
        w.put(1, 1); // uniform_tile_spacing_flag
        w.put(8, 100); // base_q_idx
        w.put(1, 0); // DeltaQYDc coded
        w.put(1, 0); // DeltaQUDc coded
        w.put(1, 0); // DeltaQUAc coded
        w.put(1, 0); // using_qmatrix
        w.put(1, 0); // segmentation_enabled
        w.put(1, 0); // delta_q_present
        w.put(6, 10); // loop_filter_level[0]
        w.put(6, 11); // loop_filter_level[1]
        w.put(6, 12); // loop_filter_level[2]
        w.put(6, 13); // loop_filter_level[3]
        w.put(3, 2); // loop_filter_sharpness
        w.put(1, 0); // loop_filter_delta_enabled
        w.put(2, 1); // cdef_damping_minus_3
        w.put(2, 1); // cdef_bits
        for i in 0..2 {
            w.put(4, i + 1); // cdef_y_pri_strength
            w.put(2, 3); // cdef_y_sec_strength (3 becomes 4)
            w.put(4, 5); // cdef_uv_pri_strength
            w.put(2, 0); // cdef_uv_sec_strength
        }
        w.put(2, 0); // lr_type[0] = NONE
        w.put(2, 1); // lr_type[1] = SWITCHABLE
        w.put(2, 0); // lr_type[2] = NONE
        w.put(1, 1); // lr_unit_shift
        w.put(1, 0); // lr_unit_extra_shift
        w.put(1, 1); // lr_uv_shift
        w.put(1, 1); // tx_mode_select
        w.put(1, 0); // reduced_tx_set
    }

    #[test]
    fn key_frame_header_of_a_still_picture() {
        let s = seq(false);
        let mut w = BitWriter::default();
        still_frame_header(&mut w);
        w.trailing();
        let mut refs = RefState::default();
        let mut r = BitReader::new(&w.bytes);
        let h = FrameHeader::parse(&mut r, &s, &mut refs, 0, 0).unwrap();
        assert_eq!(h.frame_type, KEY_FRAME);
        assert!(h.show_frame && h.frame_is_intra && !h.show_existing_frame);
        assert_eq!((h.frame_width, h.frame_height), (64, 32));
        assert_eq!((h.upscaled_width, h.mi_cols, h.mi_rows), (64, 16, 8));
        assert_eq!((h.render_width, h.render_height), (64, 32));
        assert_eq!(h.refresh_frame_flags, 0xff);
        assert_eq!(h.base_q_idx, 100);
        assert!(!h.coded_lossless);
        assert_eq!(h.loop_filter_level, [10, 11, 12, 13]);
        assert_eq!(h.loop_filter_sharpness, 2);
        assert_eq!(h.loop_filter_ref_deltas, [1, 0, 0, 0, -1, 0, -1, -1]);
        assert_eq!((h.cdef_damping, h.cdef_bits), (4, 1));
        assert_eq!(h.cdef_y_pri_strength[..2], [1, 2]);
        assert_eq!(h.cdef_y_sec_strength[..2], [4, 4]);
        assert_eq!(
            h.frame_restoration_type,
            [RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_NONE]
        );
        assert_eq!(h.loop_restoration_size, [128, 64, 64]);
        assert!(h.uses_lr);
        assert_eq!(h.tx_mode, TX_MODE_SELECT);
        assert!(!h.film_grain.apply_grain);
        // The reference update then fills every slot.
        refs.update(&h, &s);
        assert!(
            refs.slots
                .iter()
                .all(|s| s.ref_valid && s.ref_frame_type == KEY_FRAME)
        );
        assert_eq!(refs.slots[3].ref_upscaled_width, 64);
    }

    #[test]
    fn inter_frame_stops_the_parse() {
        let mut s = seq(false);
        s.reduced_still_picture_header = false;
        let mut w = BitWriter::default();
        w.put(1, 0); // show_existing_frame
        w.put(2, 1); // frame_type = INTER_FRAME
        let mut r = BitReader::new(&w.bytes);
        let e = FrameHeader::parse(&mut r, &s, &mut RefState::default(), 0, 0).unwrap_err();
        assert_eq!(e, Error::InterFrame);
    }

    #[test]
    fn show_existing_frame_needs_a_showable_slot() {
        let mut s = seq(false);
        s.reduced_still_picture_header = false;
        let mut w = BitWriter::default();
        w.put(1, 1); // show_existing_frame
        w.put(3, 2); // frame_to_show_map_idx
        let mut refs = RefState::default();
        let mut r = BitReader::new(&w.bytes);
        assert!(FrameHeader::parse(&mut r, &s, &mut refs, 0, 0).is_err());
        refs.slots[2].ref_valid = true;
        refs.slots[2].ref_showable_frame = true;
        refs.slots[2].ref_frame_type = KEY_FRAME;
        let mut r = BitReader::new(&w.bytes);
        let h = FrameHeader::parse(&mut r, &s, &mut refs, 0, 0).unwrap();
        assert!(h.show_existing_frame);
        assert_eq!(h.frame_to_show_map_idx, 2);
        assert_eq!(h.refresh_frame_flags, 0xff);
    }

    #[test]
    fn qindex_follows_the_segment_feature() {
        let mut h = FrameHeader {
            base_q_idx: 50,
            ..FrameHeader::default()
        };
        assert_eq!(h.get_qindex(true, 3, 99), 50);
        h.delta_q_present = true;
        assert_eq!(h.get_qindex(false, 3, 99), 99);
        h.segmentation_enabled = true;
        h.feature_enabled[3][SEG_LVL_ALT_Q] = true;
        h.feature_data[3][SEG_LVL_ALT_Q] = -255;
        assert_eq!(h.get_qindex(true, 3, 99), 0);
        assert_eq!(h.get_qindex(false, 3, 99), 0);
        h.feature_data[3][SEG_LVL_ALT_Q] = 10;
        assert_eq!(h.get_qindex(false, 3, 99), 109);
    }
}
