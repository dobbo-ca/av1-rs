//! The frame-level state of a decode: the sample planes (`CurrFrame`, `LrFrame`) and the frame
//! dimensions the tile decode needs. Buffers are charged to the ledger before they are
//! allocated and released when they drop.
#![allow(dead_code)]

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

/// The spec's frame-level decode state for the frame being decoded.
///
/// The tile decode (the tile agent) adds the per-mode-info arrays here (`mi_sizes`, `skips`,
/// `segment_ids`, `y_modes`, ... as listed in docs/design.md), each `mi_rows * mi_cols` entries
/// charged to the ledger; this struct only has the dimensions and the planes for now.
#[derive(Debug)]
pub(crate) struct FrameState {
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
        Ok(FrameState {
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
}
