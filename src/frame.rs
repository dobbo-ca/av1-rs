//! The frame-level state of a decode: the sample planes (`CurrFrame`, `LrFrame`) and the frame
//! dimensions the tile decode needs. Buffers are charged to the ledger before they are
//! allocated and released when they drop.
#![allow(dead_code)]

use crate::consts::RESTORE_NONE;
use crate::error::Error;
use crate::header::FrameHeader;
use crate::ledger::{Ledger, zeroed};
use crate::seq::SequenceHeader;
use std::sync::Arc;

/// One plane of samples, one `u16` per sample for every bit depth (rows are `stride` samples
/// apart; `stride >= width`). The buffer is booked in the ledger for as long as the plane lives.
#[derive(Debug)]
pub(crate) struct Plane {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub px: Vec<u16>,
    ledger: Arc<Ledger>,
    charged: usize,
}

impl Plane {
    /// A zeroed plane of `width` x `height` samples with `stride == width`.
    ///
    /// # Errors
    /// `AllocLimit` if the ledger is exhausted, `OutOfMemory` if the allocation fails.
    pub(crate) fn new(width: usize, height: usize, ledger: &Arc<Ledger>) -> Result<Plane, Error> {
        let n = width.checked_mul(height).ok_or(Error::AllocLimit)?;
        let mut px = Vec::new();
        zeroed(&mut px, n, ledger)?;
        Ok(Plane {
            width,
            height,
            stride: width,
            px,
            ledger: Arc::clone(ledger),
            charged: n * size_of::<u16>(),
        })
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        self.ledger.release(self.charged);
    }
}

/// Bytes of the per-mode-info and loop restoration arrays of a [`FrameState`], booked in the
/// ledger and released when the state drops (a separate field, so the state's other fields can
/// still be moved out of it).
#[derive(Debug)]
pub(crate) struct Charged {
    ledger: Arc<Ledger>,
    bytes: usize,
}

impl Charged {
    /// An empty account on `ledger`.
    pub(crate) fn new(ledger: &Arc<Ledger>) -> Charged {
        Charged {
            ledger: Arc::clone(ledger),
            bytes: 0,
        }
    }

    /// A zeroed `Vec` of `n` elements, charged to the ledger.
    pub(crate) fn vec<T: Copy + Default>(&mut self, n: usize) -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        zeroed(&mut v, n, &self.ledger)?;
        self.bytes += n * size_of::<T>();
        Ok(v)
    }
}

impl Drop for Charged {
    fn drop(&mut self) {
        self.ledger.release(self.bytes);
    }
}

/// `count_units_in_frame( unitSize, frameSize )` of spec 5.11.57.
fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// The spec's frame-level decode state for the frame being decoded: the dimensions, the sample
/// planes and the per-mode-info arrays of spec 5.11 (named as in docs/design.md).
///
/// The per-mode-info arrays hold `mi_rows * mi_cols` entries, index `row * mi_cols + col`.
#[derive(Debug)]
pub(crate) struct FrameState {
    /// `MiSizes`.
    pub mi_sizes: Vec<u8>,
    /// `Skips`.
    pub skips: Vec<bool>,
    /// `IsInters`.
    pub is_inters: Vec<bool>,
    /// `SegmentIds`.
    pub segment_ids: Vec<u8>,
    /// `PrevSegmentIds`: not allocated (empty). An intra frame has `primary_ref_frame` equal to
    /// `PRIMARY_REF_NONE`, so the spec's `PrevSegmentIds` are all 0 and nothing reads them.
    pub prev_segment_ids: Vec<u8>,
    /// `YModes`.
    pub y_modes: Vec<u8>,
    /// `UVModes`.
    pub uv_modes: Vec<u8>,
    /// `RefFrames[ row ][ col ][ 0 ]` (`INTRA_FRAME` = 0 in an intra frame).
    pub ref_frames0: Vec<i8>,
    /// `TxSizes`. In an intra frame `InterTxSizes` holds the same values, so the tx size
    /// context (9.3 `get_above_tx_width`) reads this array too.
    pub tx_sizes: Vec<u8>,
    /// `InterTxSizes`. Only the intra block copy blocks of a frame with `allow_intrabc` make it
    /// differ from `tx_sizes`, so it is allocated (and written) only for such a frame; empty
    /// otherwise.
    pub inter_tx_sizes: Vec<u8>,
    /// `Mvs[ row ][ col ][ 0 ]` (`[ row, col ]` component order as in the spec), written for the
    /// intra block copy blocks only (`RefFrames[ ][ ][ 1 ]` is `NONE` in an intra frame).
    /// Empty unless `allow_intrabc`.
    pub mvs: Vec<[i32; 2]>,
    /// `TxTypes` of the inter (intra block copy) blocks, which chroma and the next transform
    /// blocks read back. Empty unless `allow_intrabc`.
    pub tx_types: Vec<u8>,
    /// `PaletteSizes[ plane ]` for plane 0 (Y) and 1 (UV).
    pub palette_sizes: [Vec<u8>; 2],
    /// `PaletteColors[ plane ]` (Y colours; U colours). Empty unless `allow_screen_content_tools`.
    pub palette_colors: [Vec<[u16; 8]>; 2],
    /// `DeltaLFs`.
    pub delta_lfs: Vec<[i8; 4]>,
    /// `LoopfilterTxSizes[ plane ][ row >> ssy ][ col >> ssx ]`, stored at index
    /// `(row >> ssy) * mi_cols + (col >> ssx)`; empty for planes the stream does not have.
    pub loopfilter_tx_sizes: [Vec<u8>; 3],
    /// `cdef_idx[ r ][ c ]`, stored at the mi index of the top-left of a 64x64 block; -1 means
    /// not coded.
    pub cdef_idx: Vec<i8>,
    /// `LrType[ plane ][ unitRow ][ unitCol ]`, index `unit_row * lr_unit_cols[ plane ] + unit_col`;
    /// empty for a plane with `FrameRestorationType` equal to `RESTORE_NONE`.
    pub lr_type: [Vec<u8>; 3],
    /// `LrWiener[ plane ][ unitRow ][ unitCol ][ pass ][ i ]`.
    pub lr_wiener: [Vec<[[i8; 3]; 2]>; 3],
    /// `LrSgrSet`.
    pub lr_sgr_set: [Vec<u8>; 3],
    /// `LrSgrXqd[ plane ][ unitRow ][ unitCol ][ i ]`.
    pub lr_sgr_xqd: [Vec<[i8; 2]>; 3],
    /// `unitRows` of each plane (spec 5.11.57); 0 for a plane without loop restoration.
    pub lr_unit_rows: [usize; 3],
    /// `unitCols` of each plane; 0 for a plane without loop restoration.
    pub lr_unit_cols: [usize; 3],
    /// The ledger the arrays above were charged to (tile decoders charge their own buffers to it).
    pub ledger: Arc<Ledger>,
    _charged: Charged,

    /// `FrameWidth`: the coded width (before superres upscaling).
    pub frame_width: usize,
    pub frame_height: usize,
    /// `UpscaledWidth`.
    pub upscaled_width: usize,
    pub mi_rows: usize,
    pub mi_cols: usize,
    /// `MiRowStarts` and `MiColStarts`, one entry more than there are tile rows or columns.
    pub mi_row_starts: Vec<usize>,
    pub mi_col_starts: Vec<usize>,
    pub num_planes: usize,
    pub subsampling_x: usize,
    pub subsampling_y: usize,
    pub bit_depth: u8,
    /// `CurrFrame`: the reconstructed samples before the in-loop filters. Plane 0 is the luma
    /// width and height rounded up to a multiple of 128, planes 1 and 2 that size shifted down by
    /// the subsampling, so every block write of a superblock is in bounds.
    pub curr_frame: Vec<Plane>,
    /// `LrFrame`: the final frame, `UpscaledWidth` x `FrameHeight` luma samples and the matching
    /// chroma size (`(size + subsampling) >> subsampling`), exact (no padding). The frame decode
    /// leaves it here after the loop filter, CDEF, superres and loop restoration; the decoder
    /// then stores and outputs it.
    pub lr_frame: Vec<Plane>,
}

impl FrameState {
    /// Allocates the `CurrFrame` planes for a frame with header `h` (the dimensions were already
    /// checked against `Limits` by the caller).
    ///
    /// # Errors
    /// `AllocLimit` or `OutOfMemory` from the plane allocations.
    pub(crate) fn new(
        h: &FrameHeader,
        seq: &SequenceHeader,
        ledger: &Arc<Ledger>,
    ) -> Result<FrameState, Error> {
        let num_planes = usize::from(seq.num_planes);
        let subsampling_x = usize::from(seq.subsampling_x);
        let subsampling_y = usize::from(seq.subsampling_y);
        let luma_w = (h.frame_width as usize + 127) & !127;
        let luma_h = (h.frame_height as usize + 127) & !127;
        let mut curr_frame = Vec::with_capacity(num_planes);
        for plane in 0..num_planes {
            let (pw, ph) = if plane == 0 {
                (luma_w, luma_h)
            } else {
                (luma_w >> subsampling_x, luma_h >> subsampling_y)
            };
            curr_frame.push(Plane::new(pw, ph, ledger)?);
        }
        let mi_rows = h.mi_rows as usize;
        let mi_cols = h.mi_cols as usize;
        let n = mi_rows.checked_mul(mi_cols).ok_or(Error::AllocLimit)?;
        let mut charged = Charged::new(ledger);
        let planes_n = |plane: usize| if plane < num_planes { n } else { 0 };
        let palette_n = if h.allow_screen_content_tools { n } else { 0 };
        let mi_sizes = charged.vec(n)?;
        let skips = charged.vec(n)?;
        let is_inters = charged.vec(n)?;
        let segment_ids = charged.vec(n)?;
        let y_modes = charged.vec(n)?;
        let uv_modes = charged.vec(n)?;
        let ref_frames0 = charged.vec(n)?;
        let tx_sizes = charged.vec(n)?;
        let ibc_n = if h.allow_intrabc { n } else { 0 };
        let inter_tx_sizes = charged.vec(ibc_n)?;
        let mvs = charged.vec(ibc_n)?;
        let tx_types = charged.vec(ibc_n)?;
        let palette_sizes = [charged.vec(n)?, charged.vec(planes_n(1))?];
        let palette_colors = [
            charged.vec(palette_n)?,
            charged.vec(palette_n.min(planes_n(1)))?,
        ];
        let delta_lfs = charged.vec(n)?;
        let loopfilter_tx_sizes = [
            charged.vec(n)?,
            charged.vec(planes_n(1))?,
            charged.vec(planes_n(2))?,
        ];
        let mut cdef_idx: Vec<i8> = charged.vec(n)?;
        cdef_idx.fill(-1);
        let mut lr_unit_rows = [0; 3];
        let mut lr_unit_cols = [0; 3];
        for plane in 0..num_planes {
            if h.frame_restoration_type[plane] != RESTORE_NONE {
                let (sx, sy) = if plane == 0 {
                    (0, 0)
                } else {
                    (subsampling_x, subsampling_y)
                };
                let unit_size = usize::from(h.loop_restoration_size[plane]).max(1);
                let round2 = |x: usize, s: usize| (x + ((1 << s) >> 1)) >> s;
                lr_unit_rows[plane] =
                    count_units_in_frame(unit_size, round2(h.frame_height as usize, sy));
                lr_unit_cols[plane] =
                    count_units_in_frame(unit_size, round2(h.upscaled_width as usize, sx));
            }
        }
        let units = |plane: usize| lr_unit_rows[plane] * lr_unit_cols[plane];
        let lr_type = [
            charged.vec(units(0))?,
            charged.vec(units(1))?,
            charged.vec(units(2))?,
        ];
        let lr_wiener = [
            charged.vec(units(0))?,
            charged.vec(units(1))?,
            charged.vec(units(2))?,
        ];
        let lr_sgr_set = [
            charged.vec(units(0))?,
            charged.vec(units(1))?,
            charged.vec(units(2))?,
        ];
        let lr_sgr_xqd = [
            charged.vec(units(0))?,
            charged.vec(units(1))?,
            charged.vec(units(2))?,
        ];
        Ok(FrameState {
            mi_sizes,
            skips,
            is_inters,
            segment_ids,
            prev_segment_ids: Vec::new(),
            y_modes,
            uv_modes,
            ref_frames0,
            tx_sizes,
            inter_tx_sizes,
            mvs,
            tx_types,
            palette_sizes,
            palette_colors,
            delta_lfs,
            loopfilter_tx_sizes,
            cdef_idx,
            lr_type,
            lr_wiener,
            lr_sgr_set,
            lr_sgr_xqd,
            lr_unit_rows,
            lr_unit_cols,
            ledger: Arc::clone(ledger),
            _charged: charged,
            frame_width: h.frame_width as usize,
            frame_height: h.frame_height as usize,
            upscaled_width: h.upscaled_width as usize,
            mi_rows: h.mi_rows as usize,
            mi_cols: h.mi_cols as usize,
            mi_row_starts: h
                .tile_info
                .mi_row_starts
                .iter()
                .map(|&v| v as usize)
                .collect(),
            mi_col_starts: h
                .tile_info
                .mi_col_starts
                .iter()
                .map(|&v| v as usize)
                .collect(),
            num_planes,
            subsampling_x,
            subsampling_y,
            bit_depth: seq.bit_depth,
            curr_frame,
            lr_frame: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::SMALL;

    #[test]
    fn plane_is_charged_and_released() {
        let ledger = Arc::new(Ledger::default());
        ledger.reset(SMALL + 100, 0).unwrap();
        let p = Plane::new(5, 4, &ledger).unwrap();
        assert_eq!(p.px.len(), 20);
        assert_eq!(ledger.left(), 100 - 40);
        assert_eq!(Plane::new(8, 8, &ledger).unwrap_err(), Error::AllocLimit);
        drop(p);
        assert_eq!(ledger.left(), 100);
    }

    #[test]
    fn frame_state_rounds_planes_up_to_128() {
        let seq = SequenceHeader::parse(&crate::seq::tests::still_sequence_header(), 0).unwrap();
        let h = FrameHeader {
            frame_width: 64,
            frame_height: 32,
            upscaled_width: 64,
            mi_cols: 16,
            mi_rows: 8,
            ..FrameHeader::default()
        };
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 20, 0).unwrap();
        let s = FrameState::new(&h, &seq, &ledger).unwrap();
        assert_eq!(s.curr_frame.len(), 3);
        assert_eq!((s.curr_frame[0].width, s.curr_frame[0].height), (128, 128));
        assert_eq!((s.curr_frame[1].width, s.curr_frame[2].height), (64, 64));
    }

    #[test]
    fn mode_info_arrays_and_lr_units_are_sized_and_released() {
        let seq = SequenceHeader::parse(&crate::seq::tests::still_sequence_header(), 0).unwrap();
        let h = FrameHeader {
            frame_width: 64,
            frame_height: 32,
            upscaled_width: 64,
            mi_cols: 16,
            mi_rows: 8,
            frame_restoration_type: [crate::consts::RESTORE_WIENER, RESTORE_NONE, RESTORE_NONE],
            loop_restoration_size: [64, 0, 0],
            ..FrameHeader::default()
        };
        let ledger = Arc::new(Ledger::default());
        ledger.reset(1 << 20, 0).unwrap();
        let before = ledger.left();
        let s = FrameState::new(&h, &seq, &ledger).unwrap();
        assert_eq!(s.mi_sizes.len(), 128);
        assert_eq!(s.cdef_idx, vec![-1; 128]);
        // (64 + 32) / 64 = 1 unit wide, max((32 + 32) / 64, 1) = 1 unit high.
        assert_eq!((s.lr_unit_rows[0], s.lr_unit_cols[0]), (1, 1));
        assert_eq!(s.lr_type[0].len(), 1);
        assert!(s.lr_type[1].is_empty());
        drop(s);
        assert_eq!(ledger.left(), before);
    }

    #[test]
    fn count_units_rounds_to_nearest_and_is_at_least_one() {
        assert_eq!(count_units_in_frame(64, 100), 2);
        assert_eq!(count_units_in_frame(64, 90), 1);
        assert_eq!(count_units_in_frame(64, 10), 1);
    }
}
