//! The CDF arrays of the symbol decoder, spec 6.8.2 (`init_non_coeff_cdfs`, `init_coeff_cdfs`,
//! `load_cdfs`), 7.20 (`save_cdfs`), 8.2 (tile copies, saved copies) and the 7.4 frame end update CDF
//! process.
//!
//! Every spec CDF array that decoding a KEY_FRAME or INTRA_ONLY_FRAME (including intra block
//! copy) needs is a field of [`CdfContext`], named after the spec array with CamelCase turned
//! into snake_case (`TxbSkipCdf` -> `txb_skip_cdf`). Rows keep the spec's representation: the
//! cumulative values ending in 32768, then the adaptation counter. Inter-only arrays
//! (`IsInterCdf`, `CompModeCdf`, `YModeCdf`, ...) are not kept.
//!
//! Notes on the layout:
//! - The `Mv*Cdf` arrays have the spec's leading `MV_CONTEXTS` index (`MV_INTRABC_CONTEXT` is 1),
//!   then the component index (0 = vertical, 1 = horizontal) where the spec has one.
//! - `PaletteSize2YColorCdf` is `palette_size_2_y_color_cdf` and so on; the
//!   [`CdfContext::palette_y_color_cdf`] and [`CdfContext::palette_uv_color_cdf`] helpers pick
//!   one by palette size.
//! - `IntraFrameYModeCdf` is not in the spec's `init_non_coeff_cdfs` list: each tile starts
//!   from the default (see [`CdfContext::init_symbol_cdfs`]). It is still a field here, and is
//!   reset to the default by `init_non_coeff_cdfs` too.

#![allow(dead_code)]

use crate::cdf_tables::*;

/// Visits every CDF row (cumulative values then the counter) of a nested array of rows.
trait CdfRows {
    fn for_each_row(&mut self, f: &mut dyn FnMut(&mut [u16]));
}

impl<const N: usize> CdfRows for [u16; N] {
    fn for_each_row(&mut self, f: &mut dyn FnMut(&mut [u16])) {
        f(self);
    }
}

impl<T: CdfRows, const N: usize> CdfRows for [T; N] {
    fn for_each_row(&mut self, f: &mut dyn FnMut(&mut [u16])) {
        for item in self.iter_mut() {
            item.for_each_row(f);
        }
    }
}

/// Declares the struct and the methods that walk every field. `non_coeff` fields are reset by
/// `init_non_coeff_cdfs`; `coeff` fields by `init_coeff_cdfs`, which picks the default table
/// by the quantizer index class.
macro_rules! cdf_context {
    (
        non_coeff { $( $(#[$m:meta])* $f:ident : $t:ty = $d:expr; )* }
        coeff { $( $(#[$cm:meta])* $cf:ident : $ct:ty = $cd:ident; )* }
    ) => {
        /// All CDF arrays used by intra frame decoding. Large (about 13 KB): keep it in a `Box`.
        #[derive(Clone)]
        pub(crate) struct CdfContext {
            $( $(#[$m])* pub(crate) $f: $t, )*
            $( $(#[$cm])* pub(crate) $cf: $ct, )*
        }

        impl CdfContext {
            /// A context holding the default CDFs (coefficient CDFs of qindex class 0).
            pub(crate) fn new_boxed() -> Box<CdfContext> {
                Box::new(CdfContext {
                    $( $f: $d, )*
                    $( $cf: $cd[0], )*
                })
            }

            /// Spec 6.8.2 `init_non_coeff_cdfs`: every CDF not used by the `coeff( )` syntax
            /// structure is set to its default.
            pub(crate) fn init_non_coeff_cdfs(&mut self) {
                $( self.$f = $d; )*
            }

            /// Spec 6.8.2 `init_coeff_cdfs`: the coefficient CDFs are set to the defaults of the
            /// class chosen by `base_q_idx` (<= 20, <= 60, <= 120, above).
            pub(crate) fn init_coeff_cdfs(&mut self, base_q_idx: u8) {
                let idx = match base_q_idx {
                    0..=20 => 0,
                    21..=60 => 1,
                    61..=120 => 2,
                    _ => 3,
                };
                $( self.$cf = $cd[idx]; )*
            }

            /// Calls `f` on every CDF row of every array.
            fn for_each_row(&mut self, f: &mut dyn FnMut(&mut [u16])) {
                $( self.$f.for_each_row(f); )*
                $( self.$cf.for_each_row(f); )*
            }
        }
    };
}

cdf_context! {
    non_coeff {
        intra_frame_y_mode_cdf: [[[u16; 14]; 5]; 5] = DEFAULT_INTRA_FRAME_Y_MODE_CDF;
        uv_mode_cfl_not_allowed_cdf: [[u16; 14]; 13] = DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF;
        uv_mode_cfl_allowed_cdf: [[u16; 15]; 13] = DEFAULT_UV_MODE_CFL_ALLOWED_CDF;
        angle_delta_cdf: [[u16; 8]; 8] = DEFAULT_ANGLE_DELTA_CDF;
        intrabc_cdf: [u16; 3] = DEFAULT_INTRABC_CDF;
        partition_w8_cdf: [[u16; 5]; 4] = DEFAULT_PARTITION_W8_CDF;
        partition_w16_cdf: [[u16; 11]; 4] = DEFAULT_PARTITION_W16_CDF;
        partition_w32_cdf: [[u16; 11]; 4] = DEFAULT_PARTITION_W32_CDF;
        partition_w64_cdf: [[u16; 11]; 4] = DEFAULT_PARTITION_W64_CDF;
        partition_w128_cdf: [[u16; 9]; 4] = DEFAULT_PARTITION_W128_CDF;
        segment_id_cdf: [[u16; 9]; 3] = DEFAULT_SEGMENT_ID_CDF;
        tx_8x8_cdf: [[u16; 3]; 3] = DEFAULT_TX_8X8_CDF;
        tx_16x16_cdf: [[u16; 4]; 3] = DEFAULT_TX_16X16_CDF;
        tx_32x32_cdf: [[u16; 4]; 3] = DEFAULT_TX_32X32_CDF;
        tx_64x64_cdf: [[u16; 4]; 3] = DEFAULT_TX_64X64_CDF;
        txfm_split_cdf: [[u16; 3]; 21] = DEFAULT_TXFM_SPLIT_CDF;
        filter_intra_mode_cdf: [u16; 6] = DEFAULT_FILTER_INTRA_MODE_CDF;
        filter_intra_cdf: [[u16; 3]; 22] = DEFAULT_FILTER_INTRA_CDF;
        skip_cdf: [[u16; 3]; 3] = DEFAULT_SKIP_CDF;
        mv_joint_cdf: [[u16; 5]; 2] = [DEFAULT_MV_JOINT_CDF; 2];
        mv_class_cdf: [[[u16; 12]; 2]; 2] = [DEFAULT_MV_CLASS_CDF; 2];
        mv_class0_bit_cdf: [[[u16; 3]; 2]; 2] = [[DEFAULT_MV_CLASS0_BIT_CDF; 2]; 2];
        mv_fr_cdf: [[[u16; 5]; 2]; 2] = [DEFAULT_MV_FR_CDF; 2];
        mv_class0_fr_cdf: [[[[u16; 5]; 2]; 2]; 2] = [DEFAULT_MV_CLASS0_FR_CDF; 2];
        mv_class0_hp_cdf: [[[u16; 3]; 2]; 2] = [[DEFAULT_MV_CLASS0_HP_CDF; 2]; 2];
        mv_sign_cdf: [[[u16; 3]; 2]; 2] = [[DEFAULT_MV_SIGN_CDF; 2]; 2];
        mv_bit_cdf: [[[[u16; 3]; 10]; 2]; 2] = [[DEFAULT_MV_BIT_CDF; 2]; 2];
        mv_hp_cdf: [[[u16; 3]; 2]; 2] = [[DEFAULT_MV_HP_CDF; 2]; 2];
        palette_y_mode_cdf: [[[u16; 3]; 3]; 7] = DEFAULT_PALETTE_Y_MODE_CDF;
        palette_uv_mode_cdf: [[u16; 3]; 2] = DEFAULT_PALETTE_UV_MODE_CDF;
        palette_y_size_cdf: [[u16; 8]; 7] = DEFAULT_PALETTE_Y_SIZE_CDF;
        palette_uv_size_cdf: [[u16; 8]; 7] = DEFAULT_PALETTE_UV_SIZE_CDF;
        palette_size_2_y_color_cdf: [[u16; 3]; 5] = DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF;
        palette_size_3_y_color_cdf: [[u16; 4]; 5] = DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF;
        palette_size_4_y_color_cdf: [[u16; 5]; 5] = DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF;
        palette_size_5_y_color_cdf: [[u16; 6]; 5] = DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF;
        palette_size_6_y_color_cdf: [[u16; 7]; 5] = DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF;
        palette_size_7_y_color_cdf: [[u16; 8]; 5] = DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF;
        palette_size_8_y_color_cdf: [[u16; 9]; 5] = DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF;
        palette_size_2_uv_color_cdf: [[u16; 3]; 5] = DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF;
        palette_size_3_uv_color_cdf: [[u16; 4]; 5] = DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF;
        palette_size_4_uv_color_cdf: [[u16; 5]; 5] = DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF;
        palette_size_5_uv_color_cdf: [[u16; 6]; 5] = DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF;
        palette_size_6_uv_color_cdf: [[u16; 7]; 5] = DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF;
        palette_size_7_uv_color_cdf: [[u16; 8]; 5] = DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF;
        palette_size_8_uv_color_cdf: [[u16; 9]; 5] = DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF;
        delta_q_cdf: [u16; 5] = DEFAULT_DELTA_Q_CDF;
        delta_lf_cdf: [u16; 5] = DEFAULT_DELTA_LF_CDF;
        delta_lf_multi_cdf: [[u16; 5]; 4] = [DEFAULT_DELTA_LF_CDF; 4];
        intra_tx_type_set1_cdf: [[[u16; 8]; 13]; 2] = DEFAULT_INTRA_TX_TYPE_SET1_CDF;
        intra_tx_type_set2_cdf: [[[u16; 6]; 13]; 3] = DEFAULT_INTRA_TX_TYPE_SET2_CDF;
        inter_tx_type_set1_cdf: [[u16; 17]; 2] = DEFAULT_INTER_TX_TYPE_SET1_CDF;
        inter_tx_type_set2_cdf: [u16; 13] = DEFAULT_INTER_TX_TYPE_SET2_CDF;
        inter_tx_type_set3_cdf: [[u16; 3]; 4] = DEFAULT_INTER_TX_TYPE_SET3_CDF;
        cfl_sign_cdf: [u16; 9] = DEFAULT_CFL_SIGN_CDF;
        cfl_alpha_cdf: [[u16; 17]; 6] = DEFAULT_CFL_ALPHA_CDF;
        use_wiener_cdf: [u16; 3] = DEFAULT_USE_WIENER_CDF;
        use_sgrproj_cdf: [u16; 3] = DEFAULT_USE_SGRPROJ_CDF;
        restoration_type_cdf: [u16; 4] = DEFAULT_RESTORATION_TYPE_CDF;
    }
    coeff {
        txb_skip_cdf: [[[u16; 3]; 13]; 5] = DEFAULT_TXB_SKIP_CDF;
        eob_pt_16_cdf: [[[u16; 6]; 2]; 2] = DEFAULT_EOB_PT_16_CDF;
        eob_pt_32_cdf: [[[u16; 7]; 2]; 2] = DEFAULT_EOB_PT_32_CDF;
        eob_pt_64_cdf: [[[u16; 8]; 2]; 2] = DEFAULT_EOB_PT_64_CDF;
        eob_pt_128_cdf: [[[u16; 9]; 2]; 2] = DEFAULT_EOB_PT_128_CDF;
        eob_pt_256_cdf: [[[u16; 10]; 2]; 2] = DEFAULT_EOB_PT_256_CDF;
        eob_pt_512_cdf: [[u16; 11]; 2] = DEFAULT_EOB_PT_512_CDF;
        eob_pt_1024_cdf: [[u16; 12]; 2] = DEFAULT_EOB_PT_1024_CDF;
        eob_extra_cdf: [[[[u16; 3]; 9]; 2]; 5] = DEFAULT_EOB_EXTRA_CDF;
        dc_sign_cdf: [[[u16; 3]; 3]; 2] = DEFAULT_DC_SIGN_CDF;
        coeff_base_eob_cdf: [[[[u16; 4]; 4]; 2]; 5] = DEFAULT_COEFF_BASE_EOB_CDF;
        coeff_base_cdf: [[[[u16; 5]; 42]; 2]; 5] = DEFAULT_COEFF_BASE_CDF;
        coeff_br_cdf: [[[[u16; 5]; 21]; 2]; 5] = DEFAULT_COEFF_BR_CDF;
    }
}

/// Only the name: the arrays are too large to print.
impl std::fmt::Debug for CdfContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CdfContext")
    }
}

impl CdfContext {
    /// Spec 6.8.2 `load_cdfs( ctx )`: copies the CDFs saved for a reference frame (`saved`) and
    /// then sets the last entry of every array, the symbol counter, to 0.
    pub(crate) fn load_cdfs(&mut self, saved: &CdfContext) {
        self.clone_from(saved);
        self.reset_counters();
    }

    /// Spec 7.20 `save_cdfs( ctx )`: a copy of all CDF arrays, to store in a reference slot.
    ///
    /// The same copy implements the exit process "Saved" arrays (8.2.4, taken from a tile's
    /// context) and the frame end update CDF process (frame arrays = saved arrays, which is
    /// plain assignment of the `Box`).
    pub(crate) fn save_cdfs(&self) -> Box<CdfContext> {
        Box::new(self.clone())
    }

    /// Spec 8.2.2, the CDF copies made by `init_symbol`: the tile's working context is a copy of
    /// the frame's CDFs, except that `IntraFrameYModeCdf` is reset to its default.
    pub(crate) fn init_symbol_cdfs(&self) -> Box<CdfContext> {
        let mut tile = self.save_cdfs();
        tile.intra_frame_y_mode_cdf = DEFAULT_INTRA_FRAME_Y_MODE_CDF;
        tile
    }

    /// [`CdfContext::init_symbol_cdfs`] into an existing context (no allocation).
    pub(crate) fn init_symbol_cdfs_into(&self, tile: &mut CdfContext) {
        tile.clone_from(self);
        tile.intra_frame_y_mode_cdf = DEFAULT_INTRA_FRAME_Y_MODE_CDF;
    }

    /// The counter-clearing step of spec 6.8.2 `load_cdfs`: sets the last entry of every CDF
    /// array to 0.
    pub(crate) fn reset_counters(&mut self) {
        self.for_each_row(&mut |row| {
            if let Some(counter) = row.last_mut() {
                *counter = 0;
            }
        });
    }

    /// The `PaletteSize{n}YColorCdf` array for `palette_size` (2..=8, from the validated syntax
    /// element; other values select size 8), row `ctx` (0..5, from the color context).
    pub(crate) fn palette_y_color_cdf(&mut self, palette_size: usize, ctx: usize) -> &mut [u16] {
        match palette_size {
            2 => &mut self.palette_size_2_y_color_cdf[ctx][..],
            3 => &mut self.palette_size_3_y_color_cdf[ctx][..],
            4 => &mut self.palette_size_4_y_color_cdf[ctx][..],
            5 => &mut self.palette_size_5_y_color_cdf[ctx][..],
            6 => &mut self.palette_size_6_y_color_cdf[ctx][..],
            7 => &mut self.palette_size_7_y_color_cdf[ctx][..],
            _ => &mut self.palette_size_8_y_color_cdf[ctx][..],
        }
    }

    /// As [`CdfContext::palette_y_color_cdf`], for the `PaletteSize{n}UVColorCdf` arrays.
    pub(crate) fn palette_uv_color_cdf(&mut self, palette_size: usize, ctx: usize) -> &mut [u16] {
        match palette_size {
            2 => &mut self.palette_size_2_uv_color_cdf[ctx][..],
            3 => &mut self.palette_size_3_uv_color_cdf[ctx][..],
            4 => &mut self.palette_size_4_uv_color_cdf[ctx][..],
            5 => &mut self.palette_size_5_uv_color_cdf[ctx][..],
            6 => &mut self.palette_size_6_uv_color_cdf[ctx][..],
            7 => &mut self.palette_size_7_uv_color_cdf[ctx][..],
            _ => &mut self.palette_size_8_uv_color_cdf[ctx][..],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rows_are_valid_cdfs() {
        let mut cdf = CdfContext::new_boxed();
        let mut rows = 0;
        cdf.for_each_row(&mut |row| {
            rows += 1;
            let n = row.len() - 1;
            assert_eq!(row[n], 0, "counter starts at 0");
            assert_eq!(row[n - 1], 32768, "last cumulative value is 1 << 15");
            assert!(row[..n].windows(2).all(|w| w[0] <= w[1]), "row {row:?}");
        });
        assert!(rows > 1000);
    }

    #[test]
    fn every_coefficient_class_is_valid() {
        for q in [0, 21, 61, 121] {
            let mut cdf = CdfContext::new_boxed();
            cdf.init_coeff_cdfs(q);
            cdf.for_each_row(&mut |row| {
                let n = row.len() - 1;
                assert_eq!(row[n - 1], 32768);
                assert_eq!(row[n], 0);
            });
        }
    }

    #[test]
    fn init_coeff_cdfs_picks_class_at_the_thresholds() {
        let classes = [
            (0, 0),
            (20, 0),
            (21, 1),
            (60, 1),
            (61, 2),
            (120, 2),
            (121, 3),
            (255, 3),
        ];
        let mut cdf = CdfContext::new_boxed();
        for (base_q_idx, idx) in classes {
            cdf.init_coeff_cdfs(base_q_idx);
            assert!(
                cdf.txb_skip_cdf == DEFAULT_TXB_SKIP_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(
                cdf.eob_pt_16_cdf == DEFAULT_EOB_PT_16_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(
                cdf.eob_pt_512_cdf == DEFAULT_EOB_PT_512_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(
                cdf.eob_extra_cdf == DEFAULT_EOB_EXTRA_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(
                cdf.dc_sign_cdf == DEFAULT_DC_SIGN_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(cdf.coeff_base_eob_cdf == DEFAULT_COEFF_BASE_EOB_CDF[idx]);
            assert!(
                cdf.coeff_base_cdf == DEFAULT_COEFF_BASE_CDF[idx],
                "q {base_q_idx}"
            );
            assert!(
                cdf.coeff_br_cdf == DEFAULT_COEFF_BR_CDF[idx],
                "q {base_q_idx}"
            );
        }
        // The classes really differ, so the checks above tell them apart.
        assert!(DEFAULT_COEFF_BASE_CDF[0] != DEFAULT_COEFF_BASE_CDF[1]);
        assert!(DEFAULT_COEFF_BASE_CDF[2] != DEFAULT_COEFF_BASE_CDF[3]);
    }

    #[test]
    fn init_non_coeff_cdfs_restores_defaults_and_leaves_coefficients() {
        let mut cdf = CdfContext::new_boxed();
        cdf.init_coeff_cdfs(200);
        cdf.skip_cdf[1][0] = 1;
        cdf.mv_bit_cdf[1][1][9][0] = 1;
        cdf.init_non_coeff_cdfs();
        assert!(cdf.skip_cdf == DEFAULT_SKIP_CDF);
        assert_eq!(cdf.mv_bit_cdf[1][1], DEFAULT_MV_BIT_CDF);
        assert!(cdf.txb_skip_cdf == DEFAULT_TXB_SKIP_CDF[3]);
    }

    #[test]
    fn load_cdfs_copies_and_clears_counters() {
        let mut saved = CdfContext::new_boxed();
        saved.skip_cdf[2] = [1000, 32768, 17];
        saved.coeff_base_cdf[4][1][41][3] = 9;
        saved.intrabc_cdf[2] = 3;
        let mut cdf = CdfContext::new_boxed();
        cdf.load_cdfs(&saved);
        assert_eq!(cdf.skip_cdf[2], [1000, 32768, 0]);
        assert_eq!(cdf.coeff_base_cdf[4][1][41][3], 9);
        assert_eq!(cdf.intrabc_cdf, DEFAULT_INTRABC_CDF);
        // The source is untouched.
        assert_eq!(saved.skip_cdf[2][2], 17);
    }

    #[test]
    fn save_cdfs_keeps_counters() {
        let mut cdf = CdfContext::new_boxed();
        cdf.skip_cdf[0][2] = 5;
        assert_eq!(cdf.save_cdfs().skip_cdf[0][2], 5);
    }

    #[test]
    fn tile_copy_resets_intra_frame_y_mode_only() {
        let mut cdf = CdfContext::new_boxed();
        cdf.intra_frame_y_mode_cdf[4][4][0] = 1;
        cdf.skip_cdf[0][0] = 1;
        let tile = cdf.init_symbol_cdfs();
        assert!(tile.intra_frame_y_mode_cdf == DEFAULT_INTRA_FRAME_Y_MODE_CDF);
        assert_eq!(tile.skip_cdf[0][0], 1);
    }

    #[test]
    fn palette_helpers_pick_the_row_of_the_right_width() {
        let mut cdf = CdfContext::new_boxed();
        for size in 2..=8 {
            assert_eq!(cdf.palette_y_color_cdf(size, 4).len(), size + 1);
            assert_eq!(cdf.palette_uv_color_cdf(size, 0).len(), size + 1);
        }
    }
}
