//! The sequence header OBU (spec 5.5, semantics 6.4): every field, kept as parsed.
#![allow(dead_code)]

use crate::bits::BitReader;
use crate::consts::*;
use crate::error::Error;

/// Most operating points a sequence header can carry (`operating_points_cnt_minus_1` is 5 bits).
pub(crate) const MAX_OPERATING_POINTS: usize = 32;

/// A parsed `sequence_header_obu()`. Field names are the spec's, in snake case; one-bit flags are
/// `bool`, `subsampling_x` and `subsampling_y` are 0 or 1 because the spec shifts by them.
/// Equality is full structural equality, which the decoder uses to detect a changed sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present_flag: bool,
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
    pub decoder_model_info_present_flag: bool,
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
    pub initial_display_delay_present_flag: bool,
    pub operating_points_cnt_minus_1: u8,
    pub operating_point_idc: [u16; MAX_OPERATING_POINTS],
    pub seq_level_idx: [u8; MAX_OPERATING_POINTS],
    pub seq_tier: [u8; MAX_OPERATING_POINTS],
    pub decoder_model_present_for_this_op: [bool; MAX_OPERATING_POINTS],
    pub decoder_buffer_delay: [u32; MAX_OPERATING_POINTS],
    pub encoder_buffer_delay: [u32; MAX_OPERATING_POINTS],
    pub low_delay_mode_flag: [bool; MAX_OPERATING_POINTS],
    pub initial_display_delay_present_for_this_op: [bool; MAX_OPERATING_POINTS],
    pub initial_display_delay_minus_1: [u8; MAX_OPERATING_POINTS],
    /// The operating point `choose_operating_point()` picked.
    pub operating_point: usize,
    /// `OperatingPointIdc`: `operating_point_idc[operating_point]`.
    pub cur_operating_point_idc: u16,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u32,
    pub max_frame_height_minus_1: u32,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_choose_screen_content_tools: bool,
    /// 0, 1 or `SELECT_SCREEN_CONTENT_TOOLS`.
    pub seq_force_screen_content_tools: u8,
    /// 0, 1 or `SELECT_INTEGER_MV`.
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits`.
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    /// `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    pub mono_chrome: bool,
    /// `NumPlanes`: 1 for monochrome, else 3.
    pub num_planes: u8,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Parses the payload of an `OBU_SEQUENCE_HEADER` (spec 5.5.1, the bytes after the OBU
    /// header and size; the trailing bits are not checked). `operating_point` is what
    /// `choose_operating_point()` returns, clamped to `operating_points_cnt_minus_1`.
    ///
    /// # Errors
    /// `Truncated` if the data ends early; `Invalid` for `seq_profile` above 2 or
    /// `reduced_still_picture_header` without `still_picture`.
    pub(crate) fn parse(data: &[u8], operating_point: usize) -> Result<Self, Error> {
        let mut r = BitReader::new(data);
        let seq_profile = r.f(3)? as u8;
        if seq_profile > 2 {
            return Err(Error::Invalid("seq_profile above 2"));
        }
        let still_picture = r.flag()?;
        let reduced_still_picture_header = r.flag()?;
        if reduced_still_picture_header && !still_picture {
            return Err(Error::Invalid(
                "reduced_still_picture_header without still_picture",
            ));
        }
        let mut s = SequenceHeader {
            seq_profile,
            still_picture,
            reduced_still_picture_header,
            timing_info_present_flag: false,
            num_units_in_display_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            num_ticks_per_picture_minus_1: 0,
            decoder_model_info_present_flag: false,
            buffer_delay_length_minus_1: 0,
            num_units_in_decoding_tick: 0,
            buffer_removal_time_length_minus_1: 0,
            frame_presentation_time_length_minus_1: 0,
            initial_display_delay_present_flag: false,
            operating_points_cnt_minus_1: 0,
            operating_point_idc: [0; MAX_OPERATING_POINTS],
            seq_level_idx: [0; MAX_OPERATING_POINTS],
            seq_tier: [0; MAX_OPERATING_POINTS],
            decoder_model_present_for_this_op: [false; MAX_OPERATING_POINTS],
            decoder_buffer_delay: [0; MAX_OPERATING_POINTS],
            encoder_buffer_delay: [0; MAX_OPERATING_POINTS],
            low_delay_mode_flag: [false; MAX_OPERATING_POINTS],
            initial_display_delay_present_for_this_op: [false; MAX_OPERATING_POINTS],
            initial_display_delay_minus_1: [(BUFFER_POOL_MAX_SIZE - 1) as u8; MAX_OPERATING_POINTS],
            operating_point: 0,
            cur_operating_point_idc: 0,
            frame_width_bits_minus_1: 0,
            frame_height_bits_minus_1: 0,
            max_frame_width_minus_1: 0,
            max_frame_height_minus_1: 0,
            frame_id_numbers_present_flag: false,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            use_128x128_superblock: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: false,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: false,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            seq_choose_screen_content_tools: false,
            seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS as u8,
            seq_force_integer_mv: SELECT_INTEGER_MV as u8,
            order_hint_bits: 0,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            bit_depth: 8,
            mono_chrome: false,
            num_planes: 3,
            color_description_present_flag: false,
            color_primaries: CP_UNSPECIFIED,
            transfer_characteristics: TC_UNSPECIFIED,
            matrix_coefficients: MC_UNSPECIFIED,
            color_range: false,
            subsampling_x: 1,
            subsampling_y: 1,
            chroma_sample_position: CSP_UNKNOWN,
            separate_uv_delta_q: false,
            film_grain_params_present: false,
        };
        if reduced_still_picture_header {
            s.seq_level_idx[0] = r.f(5)? as u8;
        } else {
            s.timing_info_present_flag = r.flag()?;
            if s.timing_info_present_flag {
                s.num_units_in_display_tick = r.f(32)?;
                s.time_scale = r.f(32)?;
                s.equal_picture_interval = r.flag()?;
                if s.equal_picture_interval {
                    s.num_ticks_per_picture_minus_1 = r.uvlc()?;
                }
                s.decoder_model_info_present_flag = r.flag()?;
                if s.decoder_model_info_present_flag {
                    s.buffer_delay_length_minus_1 = r.f(5)? as u8;
                    s.num_units_in_decoding_tick = r.f(32)?;
                    s.buffer_removal_time_length_minus_1 = r.f(5)? as u8;
                    s.frame_presentation_time_length_minus_1 = r.f(5)? as u8;
                }
            }
            s.initial_display_delay_present_flag = r.flag()?;
            s.operating_points_cnt_minus_1 = r.f(5)? as u8;
            for i in 0..=usize::from(s.operating_points_cnt_minus_1) {
                s.operating_point_idc[i] = r.f(12)? as u16;
                s.seq_level_idx[i] = r.f(5)? as u8;
                if s.seq_level_idx[i] > 7 {
                    s.seq_tier[i] = r.f(1)? as u8;
                }
                if s.decoder_model_info_present_flag {
                    s.decoder_model_present_for_this_op[i] = r.flag()?;
                    if s.decoder_model_present_for_this_op[i] {
                        let n = u32::from(s.buffer_delay_length_minus_1) + 1;
                        s.decoder_buffer_delay[i] = r.f(n)?;
                        s.encoder_buffer_delay[i] = r.f(n)?;
                        s.low_delay_mode_flag[i] = r.flag()?;
                    }
                }
                if s.initial_display_delay_present_flag {
                    s.initial_display_delay_present_for_this_op[i] = r.flag()?;
                    if s.initial_display_delay_present_for_this_op[i] {
                        s.initial_display_delay_minus_1[i] = r.f(4)? as u8;
                    }
                }
            }
        }
        s.operating_point = operating_point.min(usize::from(s.operating_points_cnt_minus_1));
        s.cur_operating_point_idc = s.operating_point_idc[s.operating_point];
        s.frame_width_bits_minus_1 = r.f(4)? as u8;
        s.frame_height_bits_minus_1 = r.f(4)? as u8;
        s.max_frame_width_minus_1 = r.f(u32::from(s.frame_width_bits_minus_1) + 1)?;
        s.max_frame_height_minus_1 = r.f(u32::from(s.frame_height_bits_minus_1) + 1)?;
        if !reduced_still_picture_header {
            s.frame_id_numbers_present_flag = r.flag()?;
        }
        if s.frame_id_numbers_present_flag {
            s.delta_frame_id_length_minus_2 = r.f(4)? as u8;
            s.additional_frame_id_length_minus_1 = r.f(3)? as u8;
        }
        s.use_128x128_superblock = r.flag()?;
        s.enable_filter_intra = r.flag()?;
        s.enable_intra_edge_filter = r.flag()?;
        if !reduced_still_picture_header {
            s.enable_interintra_compound = r.flag()?;
            s.enable_masked_compound = r.flag()?;
            s.enable_warped_motion = r.flag()?;
            s.enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                s.enable_jnt_comp = r.flag()?;
                s.enable_ref_frame_mvs = r.flag()?;
            }
            s.seq_choose_screen_content_tools = r.flag()?;
            if !s.seq_choose_screen_content_tools {
                s.seq_force_screen_content_tools = r.f(1)? as u8;
            }
            if s.seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.flag()?;
                if !seq_choose_integer_mv {
                    s.seq_force_integer_mv = r.f(1)? as u8;
                }
            }
            if s.enable_order_hint {
                s.order_hint_bits = r.f(3)? as u8 + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.color_config(&mut r)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }

    /// `color_config()` (spec 5.5.2).
    fn color_config(&mut self, r: &mut BitReader<'_>) -> Result<(), Error> {
        let high_bitdepth = r.flag()?;
        if self.seq_profile == 2 && high_bitdepth {
            let twelve_bit = r.flag()?;
            self.bit_depth = if twelve_bit { 12 } else { 10 };
        } else {
            self.bit_depth = if high_bitdepth { 10 } else { 8 };
        }
        self.mono_chrome = if self.seq_profile == 1 {
            false
        } else {
            r.flag()?
        };
        self.num_planes = if self.mono_chrome { 1 } else { 3 };
        self.color_description_present_flag = r.flag()?;
        if self.color_description_present_flag {
            self.color_primaries = r.f(8)? as u8;
            self.transfer_characteristics = r.f(8)? as u8;
            self.matrix_coefficients = r.f(8)? as u8;
        }
        if self.mono_chrome {
            self.color_range = r.flag()?;
            self.subsampling_x = 1;
            self.subsampling_y = 1;
            self.chroma_sample_position = CSP_UNKNOWN;
            self.separate_uv_delta_q = false;
            return Ok(());
        }
        if self.color_primaries == CP_BT_709
            && self.transfer_characteristics == TC_SRGB
            && self.matrix_coefficients == MC_IDENTITY
        {
            self.color_range = true;
            self.subsampling_x = 0;
            self.subsampling_y = 0;
        } else {
            self.color_range = r.flag()?;
            match self.seq_profile {
                0 => {
                    self.subsampling_x = 1;
                    self.subsampling_y = 1;
                }
                1 => {
                    self.subsampling_x = 0;
                    self.subsampling_y = 0;
                }
                _ => {
                    if self.bit_depth == 12 {
                        self.subsampling_x = r.f(1)? as u8;
                        self.subsampling_y = if self.subsampling_x != 0 {
                            r.f(1)? as u8
                        } else {
                            0
                        };
                    } else {
                        self.subsampling_x = 1;
                        self.subsampling_y = 0;
                    }
                }
            }
            if self.subsampling_x != 0 && self.subsampling_y != 0 {
                self.chroma_sample_position = r.f(2)? as u8;
            }
        }
        self.separate_uv_delta_q = r.flag()?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Writes bits high first, for building test streams.
    #[derive(Default)]
    pub(crate) struct BitWriter {
        pub bytes: Vec<u8>,
        n: usize,
    }

    impl BitWriter {
        pub(crate) fn put(&mut self, bits: u32, value: u64) {
            for i in (0..bits).rev() {
                if self.n.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = ((value >> i) & 1) as u8;
                let last = self.bytes.len() - 1;
                self.bytes[last] |= bit << (7 - self.n % 8);
                self.n += 1;
            }
        }

        /// Zero bits to the byte boundary.
        pub(crate) fn align(&mut self) {
            while !self.n.is_multiple_of(8) {
                self.put(1, 0);
            }
        }

        /// A one bit, then zero bits to the byte boundary.
        pub(crate) fn trailing(&mut self) {
            self.put(1, 1);
            while !self.n.is_multiple_of(8) {
                self.put(1, 0);
            }
        }
    }

    /// A reduced still picture header sequence header: 64x32, 8-bit 4:2:0, 64x64 superblocks.
    pub(crate) fn still_sequence_header() -> Vec<u8> {
        let mut w = BitWriter::default();
        w.put(3, 0); // seq_profile
        w.put(1, 1); // still_picture
        w.put(1, 1); // reduced_still_picture_header
        w.put(5, 4); // seq_level_idx[0]
        w.put(4, 7); // frame_width_bits_minus_1
        w.put(4, 6); // frame_height_bits_minus_1
        w.put(8, 63); // max_frame_width_minus_1
        w.put(7, 31); // max_frame_height_minus_1
        w.put(1, 0); // use_128x128_superblock
        w.put(1, 1); // enable_filter_intra
        w.put(1, 1); // enable_intra_edge_filter
        w.put(1, 0); // enable_superres
        w.put(1, 1); // enable_cdef
        w.put(1, 1); // enable_restoration
        w.put(1, 0); // high_bitdepth
        w.put(1, 0); // mono_chrome
        w.put(1, 0); // color_description_present_flag
        w.put(1, 0); // color_range
        w.put(2, 0); // chroma_sample_position
        w.put(1, 0); // separate_uv_delta_q
        w.put(1, 0); // film_grain_params_present
        w.trailing();
        w.bytes
    }

    #[test]
    fn parses_reduced_still_picture_header() {
        let s = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        assert!(s.still_picture && s.reduced_still_picture_header);
        assert_eq!(s.seq_level_idx[0], 4);
        assert_eq!(
            (s.max_frame_width_minus_1, s.max_frame_height_minus_1),
            (63, 31)
        );
        assert!(!s.use_128x128_superblock);
        assert!(s.enable_filter_intra && s.enable_intra_edge_filter);
        assert!(!s.enable_superres && s.enable_cdef && s.enable_restoration);
        assert_eq!(s.bit_depth, 8);
        assert!(!s.mono_chrome);
        assert_eq!((s.num_planes, s.subsampling_x, s.subsampling_y), (3, 1, 1));
        assert_eq!(s.color_primaries, CP_UNSPECIFIED);
        assert_eq!(
            s.seq_force_screen_content_tools,
            SELECT_SCREEN_CONTENT_TOOLS as u8
        );
        assert_eq!(s.operating_points_cnt_minus_1, 0);
        assert_eq!(s.cur_operating_point_idc, 0);
        assert_eq!(s.order_hint_bits, 0);
        assert!(!s.film_grain_params_present);
    }

    #[test]
    fn operating_point_is_clamped() {
        let s = SequenceHeader::parse(&still_sequence_header(), 9).unwrap();
        assert_eq!(s.operating_point, 0);
    }

    #[test]
    fn rejects_bad_profile_and_truncation() {
        let mut data = still_sequence_header();
        assert_eq!(SequenceHeader::parse(&data[..5], 0), Err(Error::Truncated));
        data[0] |= 0b1110_0000; // seq_profile 7
        assert!(matches!(
            SequenceHeader::parse(&data, 0),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn full_header_with_operating_points_and_444() {
        let mut w = BitWriter::default();
        w.put(3, 1); // seq_profile 1 (4:4:4)
        w.put(1, 0); // still_picture
        w.put(1, 0); // reduced_still_picture_header
        w.put(1, 0); // timing_info_present_flag
        w.put(1, 0); // initial_display_delay_present_flag
        w.put(5, 1); // operating_points_cnt_minus_1
        w.put(12, 0x103); // operating_point_idc[0]: temporal 0,1, spatial 0
        w.put(5, 9); // seq_level_idx[0]
        w.put(1, 1); // seq_tier[0]
        w.put(12, 0x101); // operating_point_idc[1]
        w.put(5, 2); // seq_level_idx[1]
        w.put(4, 11); // frame_width_bits_minus_1
        w.put(4, 11); // frame_height_bits_minus_1
        w.put(12, 1919); // max_frame_width_minus_1
        w.put(12, 1079); // max_frame_height_minus_1
        w.put(1, 0); // frame_id_numbers_present_flag
        w.put(1, 1); // use_128x128_superblock
        w.put(1, 0); // enable_filter_intra
        w.put(1, 0); // enable_intra_edge_filter
        w.put(4, 0b0000); // interintra, masked, warped, dual_filter
        w.put(1, 1); // enable_order_hint
        w.put(1, 0); // enable_jnt_comp
        w.put(1, 0); // enable_ref_frame_mvs
        w.put(1, 0); // seq_choose_screen_content_tools
        w.put(1, 1); // seq_force_screen_content_tools
        w.put(1, 0); // seq_choose_integer_mv
        w.put(1, 1); // seq_force_integer_mv
        w.put(3, 6); // order_hint_bits_minus_1
        w.put(3, 0); // superres, cdef, restoration
        w.put(1, 1); // high_bitdepth
        w.put(1, 0); // color_description_present_flag
        w.put(1, 1); // color_range
        w.put(1, 1); // separate_uv_delta_q
        w.put(1, 1); // film_grain_params_present
        w.trailing();
        let s = SequenceHeader::parse(&w.bytes, 1).unwrap();
        assert_eq!(s.seq_profile, 1);
        assert_eq!(s.operating_points_cnt_minus_1, 1);
        assert_eq!(s.operating_point_idc[..2], [0x103, 0x101]);
        assert_eq!(
            (s.seq_level_idx[0], s.seq_tier[0], s.seq_tier[1]),
            (9, 1, 0)
        );
        assert_eq!(s.cur_operating_point_idc, 0x101);
        assert_eq!(
            (s.max_frame_width_minus_1, s.max_frame_height_minus_1),
            (1919, 1079)
        );
        assert!(s.use_128x128_superblock);
        assert_eq!(s.seq_force_screen_content_tools, 1);
        assert_eq!(s.seq_force_integer_mv, 1);
        assert_eq!(s.order_hint_bits, 7);
        assert_eq!(s.bit_depth, 10);
        assert_eq!((s.subsampling_x, s.subsampling_y), (0, 0));
        assert!(s.color_range && s.separate_uv_delta_q && s.film_grain_params_present);
    }
}
