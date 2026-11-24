//! The frame-level state of a decode: the sample planes (`CurrFrame`, `LrFrame`) and the frame
//! dimensions the tile decode needs. Buffers are charged to the ledger before they are
//! allocated and released when they drop.
#![allow(dead_code)]

use crate::cdf::CdfContext;
use crate::consts::{MI_SIZE, RESTORE_NONE};
use crate::error::Error;
use crate::header::FrameHeader;
use crate::ledger::{Ledger, zeroed};
use crate::seq::SequenceHeader;
use crate::tile::TileScratch;
use std::ops::Range;
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
    /// A zeroed plane of `width` x `height` samples with `stride == width`, on a buffer from the
    /// ledger's pool when one is large enough.
    ///
    /// # Errors
    /// `AllocLimit` if the ledger is exhausted, `OutOfMemory` if the allocation fails.
    pub(crate) fn new(width: usize, height: usize, ledger: &Arc<Ledger>) -> Result<Plane, Error> {
        let n = width.checked_mul(height).ok_or(Error::AllocLimit)?;
        let (mut px, charged) = ledger.take::<u16>(n)?;
        px.resize(n, 0);
        Ok(Plane {
            width,
            height,
            stride: width,
            px,
            ledger: Arc::clone(ledger),
            charged,
        })
    }
}

impl Drop for Plane {
    /// The buffer goes back to the ledger's pool, still charged.
    fn drop(&mut self) {
        self.ledger.give(std::mem::take(&mut self.px), self.charged);
    }
}

/// The part of a plane a filter stage writes: whole rows (a row band) or a column strip. `px`
/// holds rows `stride` samples apart and its first sample is sample (`x0`, `y0`) of the plane.
/// A view of the whole plane ([`View::of`]) has origin (0, 0); a band or strip is a copy the
/// threaded wrapup gives one worker (docs/threading.md).
#[derive(Debug)]
pub(crate) struct View<'a> {
    pub px: &'a mut [u16],
    pub stride: usize,
    pub x0: usize,
    pub y0: usize,
}

impl<'a> View<'a> {
    /// All of `p`.
    pub(crate) fn of(p: &'a mut Plane) -> View<'a> {
        View {
            stride: p.stride,
            px: p.px.as_mut_slice(),
            x0: 0,
            y0: 0,
        }
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

/// Copies the `rows` x `cols` top-left rectangle of `src` (rows `src_stride` apart) into `dst`
/// (rows `dst_stride` apart) at (`row0`, `col0`). Does nothing for an array that is not
/// allocated (empty `src`), and skips a row that does not fit.
fn copy_rect<T: Copy>(
    dst: &mut [T],
    dst_stride: usize,
    (row0, col0): (usize, usize),
    src: &[T],
    src_stride: usize,
    (rows, cols): (usize, usize),
) {
    if src.is_empty() {
        return;
    }
    for r in 0..rows {
        let s = src.get(r * src_stride..r * src_stride + cols);
        let d_start = (row0 + r) * dst_stride + col0;
        let d = dst.get_mut(d_start..d_start + cols);
        if let (Some(s), Some(d)) = (s, d) {
            d.copy_from_slice(s);
        }
    }
}

/// `count_units_in_frame( unitSize, frameSize )` of spec 5.11.57.
fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// `lr_type` value of a unit a tile has not written (a tile state made by
/// [`FrameState::new_tile`] starts with every unit unset).
const LR_UNSET: u8 = 255;

/// The spec's frame-level decode state for the frame being decoded: the dimensions, the sample
/// planes and the per-mode-info arrays of spec 5.11 (named as in docs/design.md).
///
/// The per-mode-info arrays hold `mi_rows * mi_cols` entries, index `row * mi_cols + col`.
///
/// A state made by [`FrameState::new_tile`] is the same structure for one tile, in the tile's
/// own coordinates: `mi_rows`, `mi_cols`, the arrays and the planes are the tile's, its origin
/// in the frame is `mi_row0` / `mi_col0`, and the tile decode runs on it unchanged (tile row 0,
/// tile column 0 of its own `mi_row_starts` / `mi_col_starts`). Tiles start on superblock
/// boundaries and read nothing outside themselves, so this is the same decode as on the frame
/// state; [`FrameState::merge_tile`] copies the result into the frame state.
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
    /// Position of this state's top-left mode info unit in the frame: 0 for the frame state, the
    /// tile origin for a [`FrameState::new_tile`] state.
    pub mi_row0: usize,
    pub mi_col0: usize,
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
    /// The buffers of a tile decode, kept between the tiles and frames this state decodes
    /// (see [`TileScratch`]).
    pub tile_scratch: Option<TileScratch>,
    /// The frame CDFs of the last frame decoded on this state, reused by the next.
    pub frame_cdf: Option<Box<CdfContext>>,
    /// The sizes every buffer above was allocated for.
    shape: Shape,
}

/// What decides the size of every array and plane of a [`FrameState`]: a state of an earlier
/// frame with the same shape is reused as it is ([`FrameState::reuse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    mi: (usize, usize),
    luma: (usize, usize),
    num_planes: usize,
    subsampling: (usize, usize),
    palette: bool,
    intrabc: bool,
    lr_unit_rows: [usize; 3],
    lr_unit_cols: [usize; 3],
}

/// `unitRows` and `unitCols` of each plane (spec 5.11.57); 0 for a plane without loop
/// restoration.
fn lr_units(
    h: &FrameHeader,
    num_planes: usize,
    (subsampling_x, subsampling_y): (usize, usize),
) -> ([usize; 3], [usize; 3]) {
    let mut lr_unit_rows = [0; 3];
    let mut lr_unit_cols = [0; 3];
    for plane in 0..num_planes.min(3) {
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
    (lr_unit_rows, lr_unit_cols)
}

impl Shape {
    fn of(
        h: &FrameHeader,
        seq: &SequenceHeader,
        mi: (usize, usize),
        luma: (usize, usize),
    ) -> Shape {
        let num_planes = usize::from(seq.num_planes);
        let subsampling = (
            usize::from(seq.subsampling_x),
            usize::from(seq.subsampling_y),
        );
        let (lr_unit_rows, lr_unit_cols) = lr_units(h, num_planes, subsampling);
        Shape {
            mi,
            luma,
            num_planes,
            subsampling,
            palette: h.allow_screen_content_tools,
            intrabc: h.allow_intrabc,
            lr_unit_rows,
            lr_unit_cols,
        }
    }
}

impl FrameState {
    /// [`FrameState::new`], on the arrays of `spare` (the state of an earlier frame of the same
    /// decoder) when the frame has the same [`Shape`]: they are cleared as `new` would allocate
    /// them, and charged once as before. Otherwise `spare` is dropped first, so its memory is
    /// free for the new state.
    ///
    /// # Errors
    /// `AllocLimit` or `OutOfMemory` from the buffers.
    pub(crate) fn reuse(
        spare: Option<FrameState>,
        h: &FrameHeader,
        seq: &SequenceHeader,
        ledger: &Arc<Ledger>,
    ) -> Result<FrameState, Error> {
        let (mi, luma) = Self::frame_dims(h);
        if let Some(mut s) = spare
            && s.shape == Shape::of(h, seq, mi, luma)
            && Arc::ptr_eq(&s.ledger, ledger)
        {
            s.reset(h, seq)?;
            return Ok(s);
        }
        Self::new(h, seq, ledger)
    }

    /// `(mi_rows, mi_cols)` and the luma plane size (rounded up to 128) of a frame state.
    fn frame_dims(h: &FrameHeader) -> ((usize, usize), (usize, usize)) {
        let luma_w = (h.frame_width as usize + 127) & !127;
        let luma_h = (h.frame_height as usize + 127) & !127;
        ((h.mi_rows as usize, h.mi_cols as usize), (luma_w, luma_h))
    }

    /// Clears a state of the same shape for the frame of header `h`: every array as
    /// [`FrameState::build`] makes it, new zeroed `CurrFrame` planes (from the pool), no
    /// `LrFrame`.
    fn reset(&mut self, h: &FrameHeader, seq: &SequenceHeader) -> Result<(), Error> {
        self.mi_sizes.fill(0);
        self.skips.fill(false);
        self.is_inters.fill(false);
        self.segment_ids.fill(0);
        self.prev_segment_ids.clear();
        self.y_modes.fill(0);
        self.uv_modes.fill(0);
        self.ref_frames0.fill(0);
        self.tx_sizes.fill(0);
        self.inter_tx_sizes.fill(0);
        self.mvs.fill([0; 2]);
        self.tx_types.fill(0);
        for i in 0..2 {
            self.palette_sizes[i].fill(0);
            self.palette_colors[i].fill([0; 8]);
        }
        self.delta_lfs.fill([0; 4]);
        for i in 0..3 {
            self.loopfilter_tx_sizes[i].fill(0);
            self.lr_type[i].fill(0);
            self.lr_wiener[i].fill([[0; 3]; 2]);
            self.lr_sgr_set[i].fill(0);
            self.lr_sgr_xqd[i].fill([0; 2]);
        }
        self.cdef_idx.fill(-1);
        self.frame_width = h.frame_width as usize;
        self.frame_height = h.frame_height as usize;
        self.upscaled_width = h.upscaled_width as usize;
        self.mi_row0 = 0;
        self.mi_col0 = 0;
        self.mi_row_starts.clear();
        self.mi_row_starts
            .extend(h.tile_info.mi_row_starts.iter().map(|&v| v as usize));
        self.mi_col_starts.clear();
        self.mi_col_starts
            .extend(h.tile_info.mi_col_starts.iter().map(|&v| v as usize));
        self.bit_depth = seq.bit_depth;
        self.lr_frame.clear();
        self.curr_frame.clear();
        let (luma_w, luma_h) = self.shape.luma;
        for plane in 0..self.num_planes {
            let (pw, ph) = if plane == 0 {
                (luma_w, luma_h)
            } else {
                (luma_w >> self.subsampling_x, luma_h >> self.subsampling_y)
            };
            let p = Plane::new(pw, ph, &self.ledger)?;
            self.curr_frame.push(p);
        }
        Ok(())
    }

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
        let (mi, luma) = Self::frame_dims(h);
        Self::build(h, seq, ledger, mi, luma)
    }

    /// The state for tile (`tile_row`, `tile_col`) of the frame with header `h`: arrays and
    /// planes of the tile's size only (planes rounded up to 128, so every block write of a
    /// superblock is in bounds), the loop restoration arrays of the whole frame (a few bytes per
    /// unit) with every unit unset. Dimensions were checked against `Limits` by the caller.
    ///
    /// # Errors
    /// `Invalid` for a tile outside the frame, `AllocLimit` or `OutOfMemory` from the buffers.
    pub(crate) fn new_tile(
        h: &FrameHeader,
        seq: &SequenceHeader,
        ledger: &Arc<Ledger>,
        tile_row: usize,
        tile_col: usize,
    ) -> Result<FrameState, Error> {
        let at = |v: &[u32], i: usize| {
            v.get(i)
                .map(|&x| x as usize)
                .ok_or(Error::Invalid("tile index out of range"))
        };
        let starts = &h.tile_info;
        let (row0, row1) = (
            at(&starts.mi_row_starts[..], tile_row)?,
            at(&starts.mi_row_starts[..], tile_row + 1)?,
        );
        let (col0, col1) = (
            at(&starts.mi_col_starts[..], tile_col)?,
            at(&starts.mi_col_starts[..], tile_col + 1)?,
        );
        if row0 > row1 || col0 > col1 || row1 > h.mi_rows as usize || col1 > h.mi_cols as usize {
            return Err(Error::Invalid("tile outside the frame"));
        }
        let (mi_rows, mi_cols) = (row1 - row0, col1 - col0);
        let luma_w = (mi_cols * MI_SIZE + 127) & !127;
        let luma_h = (mi_rows * MI_SIZE + 127) & !127;
        let mut s = Self::build(h, seq, ledger, (mi_rows, mi_cols), (luma_w, luma_h))?;
        s.mi_row0 = row0;
        s.mi_col0 = col0;
        s.mi_row_starts = vec![0, mi_rows];
        s.mi_col_starts = vec![0, mi_cols];
        for t in &mut s.lr_type {
            t.fill(LR_UNSET);
        }
        Ok(s)
    }

    /// Copies what the decode of the tile state `t` wrote into this frame state: the tile's
    /// rectangle of every mode-info array, the loop restoration units it set, and the samples of
    /// the superblocks it covers (`sb_size` is 64 or 128). Tiles are disjoint, so the order of
    /// the merges does not matter.
    pub(crate) fn merge_tile(&mut self, t: &FrameState, sb_size: usize) {
        let (r0, c0, rows, cols) = (t.mi_row0, t.mi_col0, t.mi_rows, t.mi_cols);
        let (dst_stride, src_stride) = (self.mi_cols, t.mi_cols);
        let (at, dims) = ((r0, c0), (rows, cols));
        let (ds, ss) = (dst_stride, src_stride);
        copy_rect(&mut self.mi_sizes, ds, at, &t.mi_sizes, ss, dims);
        copy_rect(&mut self.skips, ds, at, &t.skips, ss, dims);
        copy_rect(&mut self.is_inters, ds, at, &t.is_inters, ss, dims);
        copy_rect(&mut self.segment_ids, ds, at, &t.segment_ids, ss, dims);
        copy_rect(&mut self.y_modes, ds, at, &t.y_modes, ss, dims);
        copy_rect(&mut self.uv_modes, ds, at, &t.uv_modes, ss, dims);
        copy_rect(&mut self.ref_frames0, ds, at, &t.ref_frames0, ss, dims);
        copy_rect(&mut self.tx_sizes, ds, at, &t.tx_sizes, ss, dims);
        copy_rect(
            &mut self.inter_tx_sizes,
            ds,
            at,
            &t.inter_tx_sizes,
            ss,
            dims,
        );
        copy_rect(&mut self.mvs, ds, at, &t.mvs, ss, dims);
        copy_rect(&mut self.tx_types, ds, at, &t.tx_types, ss, dims);
        copy_rect(&mut self.delta_lfs, ds, at, &t.delta_lfs, ss, dims);
        copy_rect(&mut self.cdef_idx, ds, at, &t.cdef_idx, ss, dims);
        for i in 0..2 {
            copy_rect(
                &mut self.palette_sizes[i],
                ds,
                at,
                &t.palette_sizes[i],
                ss,
                dims,
            );
            copy_rect(
                &mut self.palette_colors[i],
                ds,
                at,
                &t.palette_colors[i],
                ss,
                dims,
            );
        }
        for plane in 0..self.num_planes {
            // `LoopfilterTxSizes` of a chroma plane is stored at the subsampled position.
            let (sx, sy) = if plane == 0 {
                (0, 0)
            } else {
                (self.subsampling_x, self.subsampling_y)
            };
            copy_rect(
                &mut self.loopfilter_tx_sizes[plane],
                dst_stride,
                (r0 >> sy, c0 >> sx),
                &t.loopfilter_tx_sizes[plane],
                src_stride,
                ((rows + sy) >> sy, (cols + sx) >> sx),
            );
            // The samples of the tile's superblocks (blocks never cross a superblock, so this
            // is all the tile can have written), clipped to the frame plane.
            let (Some(dst), Some(src)) = (self.curr_frame.get_mut(plane), t.curr_frame.get(plane))
            else {
                continue;
            };
            let (ox, oy) = ((c0 * MI_SIZE) >> sx, (r0 * MI_SIZE) >> sy);
            let w = ((cols * MI_SIZE).next_multiple_of(sb_size) >> sx)
                .min(src.width)
                .min(dst.width.saturating_sub(ox));
            let hgt = ((rows * MI_SIZE).next_multiple_of(sb_size) >> sy)
                .min(src.height)
                .min(dst.height.saturating_sub(oy));
            for y in 0..hgt {
                let s = src.px.get(y * src.stride..y * src.stride + w);
                let d = dst
                    .px
                    .get_mut((oy + y) * dst.stride + ox..(oy + y) * dst.stride + ox + w);
                if let (Some(s), Some(d)) = (s, d) {
                    d.copy_from_slice(s);
                }
            }
        }
        for plane in 0..3 {
            for (idx, &ty) in t.lr_type[plane].iter().enumerate() {
                if ty == LR_UNSET {
                    continue;
                }
                if let (Some(a), Some(b), Some(c), Some(d)) = (
                    self.lr_type[plane].get_mut(idx),
                    self.lr_wiener[plane].get_mut(idx),
                    self.lr_sgr_set[plane].get_mut(idx),
                    self.lr_sgr_xqd[plane].get_mut(idx),
                ) {
                    *a = ty;
                    *b = t.lr_wiener[plane].get(idx).copied().unwrap_or_default();
                    *c = t.lr_sgr_set[plane].get(idx).copied().unwrap_or_default();
                    *d = t.lr_sgr_xqd[plane].get(idx).copied().unwrap_or_default();
                }
            }
        }
    }

    /// Copies the mode info rows `rows` of every array of `src`, a state of the same shape, and
    /// all of its loop restoration units into this state (not the samples). The pipelined frame
    /// decode hands the rows a tile decode has finished to the filters with it
    /// (docs/threading.md). An array whose size differs is skipped.
    pub(crate) fn copy_rows(&mut self, src: &FrameState, rows: Range<usize>) {
        fn copy<T: Copy>(dst: &mut [T], src: &[T], stride: usize, rows: Range<usize>) {
            let at = rows.start * stride..rows.end * stride;
            if let (Some(d), Some(s)) = (dst.get_mut(at.clone()), src.get(at)) {
                d.copy_from_slice(s);
            }
        }
        let (n, r) = (self.mi_cols, rows);
        if src.mi_cols != n {
            return;
        }
        copy(&mut self.mi_sizes, &src.mi_sizes, n, r.clone());
        copy(&mut self.skips, &src.skips, n, r.clone());
        copy(&mut self.is_inters, &src.is_inters, n, r.clone());
        copy(&mut self.segment_ids, &src.segment_ids, n, r.clone());
        copy(&mut self.y_modes, &src.y_modes, n, r.clone());
        copy(&mut self.uv_modes, &src.uv_modes, n, r.clone());
        copy(&mut self.ref_frames0, &src.ref_frames0, n, r.clone());
        copy(&mut self.tx_sizes, &src.tx_sizes, n, r.clone());
        copy(&mut self.inter_tx_sizes, &src.inter_tx_sizes, n, r.clone());
        copy(&mut self.mvs, &src.mvs, n, r.clone());
        copy(&mut self.tx_types, &src.tx_types, n, r.clone());
        copy(&mut self.delta_lfs, &src.delta_lfs, n, r.clone());
        copy(&mut self.cdef_idx, &src.cdef_idx, n, r.clone());
        for i in 0..2 {
            copy(
                &mut self.palette_sizes[i],
                &src.palette_sizes[i],
                n,
                r.clone(),
            );
            copy(
                &mut self.palette_colors[i],
                &src.palette_colors[i],
                n,
                r.clone(),
            );
        }
        for plane in 0..3 {
            // `LoopfilterTxSizes` of a chroma plane is stored at the subsampled row.
            let sy = if plane == 0 { 0 } else { self.subsampling_y };
            let tx_rows = (r.start >> sy)..((r.end + sy) >> sy);
            let (d, s) = (
                &mut self.loopfilter_tx_sizes[plane],
                &src.loopfilter_tx_sizes[plane],
            );
            copy(d, s, n, tx_rows);
            let len = self.lr_type[plane].len();
            copy(&mut self.lr_type[plane], &src.lr_type[plane], 1, 0..len);
            copy(&mut self.lr_wiener[plane], &src.lr_wiener[plane], 1, 0..len);
            copy(
                &mut self.lr_sgr_set[plane],
                &src.lr_sgr_set[plane],
                1,
                0..len,
            );
            copy(
                &mut self.lr_sgr_xqd[plane],
                &src.lr_sgr_xqd[plane],
                1,
                0..len,
            );
        }
    }

    /// Allocates a state of `mi_rows` x `mi_cols` mode info units and luma planes of
    /// `luma_w` x `luma_h` samples (chroma shifted down) for the frame of header `h`.
    fn build(
        h: &FrameHeader,
        seq: &SequenceHeader,
        ledger: &Arc<Ledger>,
        (mi_rows, mi_cols): (usize, usize),
        (luma_w, luma_h): (usize, usize),
    ) -> Result<FrameState, Error> {
        let num_planes = usize::from(seq.num_planes);
        let subsampling_x = usize::from(seq.subsampling_x);
        let subsampling_y = usize::from(seq.subsampling_y);
        let mut curr_frame = Vec::with_capacity(num_planes);
        for plane in 0..num_planes {
            let (pw, ph) = if plane == 0 {
                (luma_w, luma_h)
            } else {
                (luma_w >> subsampling_x, luma_h >> subsampling_y)
            };
            curr_frame.push(Plane::new(pw, ph, ledger)?);
        }
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
        let shape = Shape::of(h, seq, (mi_rows, mi_cols), (luma_w, luma_h));
        let (lr_unit_rows, lr_unit_cols) = (shape.lr_unit_rows, shape.lr_unit_cols);
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
            mi_rows,
            mi_cols,
            mi_row0: 0,
            mi_col0: 0,
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
            tile_scratch: None,
            frame_cdf: None,
            shape,
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
    fn reused_state_is_cleared_like_a_new_one() {
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
        let mut s = FrameState::new(&h, &seq, &ledger).unwrap();
        let fresh = FrameState::new(&h, &seq, &ledger).unwrap();
        s.mi_sizes.fill(3);
        s.skips.fill(true);
        s.y_modes.fill(5);
        s.tx_sizes.fill(2);
        s.delta_lfs.fill([1; 4]);
        s.loopfilter_tx_sizes[1].fill(4);
        s.cdef_idx.fill(2);
        s.lr_type[0].fill(1);
        s.lr_wiener[0].fill([[3; 3]; 2]);
        s.lr_sgr_xqd[0].fill([4; 2]);
        s.curr_frame[0].px.fill(9);
        s.mi_row0 = 7;
        let before = ledger.left();
        let r = FrameState::reuse(Some(s), &h, &seq, &ledger).unwrap();
        // Same arrays: nothing more is charged.
        assert_eq!(ledger.left(), before);
        assert_eq!(r.mi_sizes, fresh.mi_sizes);
        assert_eq!(r.skips, fresh.skips);
        assert_eq!(r.y_modes, fresh.y_modes);
        assert_eq!(r.tx_sizes, fresh.tx_sizes);
        assert_eq!(r.delta_lfs, fresh.delta_lfs);
        assert_eq!(r.loopfilter_tx_sizes, fresh.loopfilter_tx_sizes);
        assert_eq!(r.cdef_idx, fresh.cdef_idx);
        assert_eq!(r.lr_type, fresh.lr_type);
        assert_eq!(r.lr_wiener, fresh.lr_wiener);
        assert_eq!(r.lr_sgr_xqd, fresh.lr_sgr_xqd);
        assert_eq!(r.mi_row0, 0);
        assert_eq!(r.mi_row_starts, fresh.mi_row_starts);
        assert_eq!(r.curr_frame.len(), fresh.curr_frame.len());
        for (p, q) in r.curr_frame.iter().zip(&fresh.curr_frame) {
            assert_eq!((p.width, p.height, &p.px), (q.width, q.height, &q.px));
        }
        // Another shape gets new arrays.
        let wide = FrameHeader {
            frame_width: 128,
            upscaled_width: 128,
            mi_cols: 32,
            ..h.clone()
        };
        let w = FrameState::reuse(Some(r), &wide, &seq, &ledger).unwrap();
        assert_eq!(w.mi_sizes.len(), 8 * 32);
    }

    #[test]
    fn count_units_rounds_to_nearest_and_is_at_least_one() {
        assert_eq!(count_units_in_frame(64, 100), 2);
        assert_eq!(count_units_in_frame(64, 90), 1);
        assert_eq!(count_units_in_frame(64, 10), 1);
    }
}
