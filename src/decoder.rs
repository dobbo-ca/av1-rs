//! The decoder state machine: OBU dispatch, the sequence and frame header handling, tile group
//! collection, the reference slots (spec 7.20, 7.21), frame output (7.18) and the limit checks.
//!
//! The pixel work of a frame is [`Decoder::decode_frame_tiles`]: the tiles, then the in-loop
//! filters. [`probe`](crate::probe) runs the same path without it.
#![allow(dead_code)]

use crate::bits::BitReader;
use crate::cdef::cdef_alloc;
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::error::Error;
use crate::frame::{FrameState, Plane as WorkPlane};
use crate::grain::{GrainInput, apply_film_grain};
use crate::header::{FrameHeader, RefInfo, RefState, TileRange, parse_tile_group};
use crate::ledger::Ledger;
use crate::lf::loop_filter_frame;
use crate::limits::Limits;
use crate::lr::lr_frame;
use crate::obu::{Obu, ObuIter, parse_metadata};
use crate::parallel::run_parallel;
use crate::seq::SequenceHeader;
use crate::superres::upscale;
use crate::tile::TileDecoder;
use crate::{ColorInfo, Format, Frame, FrameInfo, Metadata, Options, Plane, PlaneData};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A frame whose header was read and whose tile groups are still coming.
#[derive(Debug)]
struct Pending {
    header: FrameHeader,
    /// The tiles received so far, in order.
    tiles: Vec<TileRange>,
    /// `TileNum` the next tile group must start at.
    next_tile: usize,
}

/// What an output frame needs to know about itself, from its frame header or, for a
/// `show_existing_frame`, from its reference slot.
struct Shown {
    frame_type: u8,
    /// `UpscaledWidth`.
    width: u32,
    /// `FrameWidth`.
    frame_width: u32,
    height: u32,
    render_width: u32,
    render_height: u32,
    bit_depth: u8,
    subsampling_x: u8,
    subsampling_y: u8,
}

impl Shown {
    fn of_header(h: &FrameHeader, seq: &SequenceHeader) -> Shown {
        Shown {
            frame_type: h.frame_type,
            width: h.upscaled_width,
            frame_width: h.frame_width,
            height: h.frame_height,
            render_width: h.render_width,
            render_height: h.render_height,
            bit_depth: seq.bit_depth,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
        }
    }

    fn of_slot(slot: &RefInfo) -> Shown {
        Shown {
            frame_type: slot.ref_frame_type,
            width: slot.ref_upscaled_width,
            frame_width: slot.ref_frame_width,
            height: slot.ref_frame_height,
            render_width: slot.ref_render_width,
            render_height: slot.ref_render_height,
            bit_depth: slot.ref_bit_depth,
            subsampling_x: slot.ref_subsampling_x,
            subsampling_y: slot.ref_subsampling_y,
        }
    }
}

/// True when two sequence headers describe the same coded video sequence: equal in everything
/// except `operating_parameters_info()`, which the spec lets change inside a sequence.
fn same_sequence(a: &SequenceHeader, b: &SequenceHeader) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    for s in [&mut a, &mut b] {
        s.decoder_buffer_delay = [0; crate::seq::MAX_OPERATING_POINTS];
        s.encoder_buffer_delay = [0; crate::seq::MAX_OPERATING_POINTS];
        s.low_delay_mode_flag = [false; crate::seq::MAX_OPERATING_POINTS];
    }
    a == b
}

/// A streaming AV1 intra decoder. Feed it one temporal unit at a time; frames decoded before an
/// error are still available through [`Decoder::take_frames`].
///
/// A [`Decoder`] keeps the reference slots between calls, so the temporal units of one stream go
/// through one decoder in order.
#[derive(Debug)]
pub struct Decoder {
    options: Options,
    limits: Limits,
    ledger: Arc<Ledger>,
    /// Parse headers only (the `probe` mode): no pixels, no output frames.
    probe_only: bool,
    seq: Option<SequenceHeader>,
    refs: RefState,
    /// `FrameStore`: the final planes of each reference slot. Slots that took the same frame
    /// share one allocation.
    store: [Option<Arc<Vec<WorkPlane>>>; NUM_REF_FRAMES],
    pending: Option<Pending>,
    /// The frame being decoded between `decode_frame_tiles` and the reference update.
    state: Option<FrameState>,
    metadata: Vec<Metadata>,
    metadata_bytes: usize,
    /// Output frames of the current temporal unit, before the layer selection.
    ready: Vec<Frame>,
    infos: Vec<FrameInfo>,
}

impl Decoder {
    /// A decoder with its own working memory budget (`limits.max_alloc`).
    pub fn new(options: Options, limits: Limits) -> Decoder {
        Self::build(options, limits, false)
    }

    /// A decoder that charges `ledger`, shared with the other decoders of one call (the tiles of
    /// an AVIF grid that decode at once); the caller has reset it to its `max_alloc`.
    pub(crate) fn with_ledger(options: Options, limits: Limits, ledger: Arc<Ledger>) -> Decoder {
        let mut d = Self::build(options, limits, false);
        d.ledger = ledger;
        d
    }

    /// A decoder that only parses headers.
    pub(crate) fn probing(limits: Limits) -> Decoder {
        Self::build(Options::default(), limits, true)
    }

    fn build(options: Options, limits: Limits, probe_only: bool) -> Decoder {
        let ledger = Arc::new(Ledger::default());
        // A `max_alloc` below the small-allocation allowance only means that every later charge
        // fails with `AllocLimit`.
        let _ = ledger.reset(limits.max_alloc, 0);
        Decoder {
            options,
            limits,
            ledger,
            probe_only,
            seq: None,
            refs: RefState::default(),
            store: Default::default(),
            pending: None,
            state: None,
            metadata: Vec::new(),
            metadata_bytes: 0,
            ready: Vec::new(),
            infos: Vec::new(),
        }
    }

    /// Decodes one temporal unit and returns the frames it outputs. For
    /// [`Format::Section5`] `data` is the OBUs between temporal delimiters (a delimiter at the
    /// start is fine); for [`Format::AnnexB`] it is one `temporal_unit( sz )` payload without its
    /// leading size. [`split_temporal_units`](crate::split_temporal_units) cuts a stream the same
    /// way.
    ///
    /// Without layers (`OperatingPointIdc` 0) every shown frame is returned. With layers only
    /// the frame of the highest spatial layer is, unless `Options::all_layers` is set.
    ///
    /// # Errors
    /// `Truncated` or `Invalid` for a broken stream, `Unsupported` for a feature this decoder
    /// does not handle (large scale tile lists), `InterFrame` at the
    /// first inter frame, `WidthLimit`, `HeightLimit`, `PixelLimit`, `AllocLimit`,
    /// `OutOfMemory` and `MetadataLimit` from the limits, and `Cancelled` when `cancel` is set
    /// (checked before every OBU and by the tile decode). Frames of this temporal unit that were
    /// finished before the error are returned by [`Decoder::take_frames`].
    pub fn decode_temporal_unit(
        &mut self,
        data: &[u8],
        format: Format,
        cancel: &AtomicBool,
    ) -> Result<Vec<Frame>, Error> {
        self.ready.clear();
        self.metadata.clear();
        self.metadata_bytes = 0;
        for obu in ObuIter::new(data, format) {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            self.handle_obu(data, obu?, cancel)?;
        }
        Ok(self.select_output())
    }

    /// The frames of the current temporal unit that were output before an error stopped it.
    /// After a successful [`Decoder::decode_temporal_unit`] there are none left.
    pub fn take_frames(&mut self) -> Vec<Frame> {
        self.select_output()
    }

    /// The frame facts collected in probe mode.
    pub(crate) fn take_infos(&mut self) -> Vec<FrameInfo> {
        std::mem::take(&mut self.infos)
    }

    /// The layer selection of spec 7.18.1: frames of the highest spatial layer only, when the
    /// stream has layers.
    fn select_output(&mut self) -> Vec<Frame> {
        let frames = std::mem::take(&mut self.ready);
        let layered = self
            .seq
            .as_ref()
            .is_some_and(|s| s.cur_operating_point_idc != 0);
        if self.options.all_layers || !layered {
            return frames;
        }
        match frames.iter().map(|f| f.spatial_id).max() {
            Some(top) => frames
                .into_iter()
                .rev()
                .find(|f| f.spatial_id == top)
                .into_iter()
                .collect(),
            None => frames,
        }
    }

    /// `open_bitstream_unit()` dispatch (spec 5.3.1). `data` is the whole temporal unit, which
    /// the tile ranges point into.
    fn handle_obu(&mut self, data: &[u8], obu: Obu<'_>, cancel: &AtomicBool) -> Result<(), Error> {
        let idc = self.seq.as_ref().map_or(0, |s| s.cur_operating_point_idc);
        if obu.header.dropped_by(idc) {
            return Ok(());
        }
        match obu.header.obu_type {
            OBU_SEQUENCE_HEADER => self.sequence_header(obu.payload),
            OBU_TEMPORAL_DELIMITER => {
                // `SeenFrameHeader = 0`: an unfinished frame is abandoned.
                self.pending = None;
                self.metadata.clear();
                self.metadata_bytes = 0;
                Ok(())
            }
            OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER => {
                self.frame_header(data, &obu, false, cancel)
            }
            OBU_FRAME => self.frame_header(data, &obu, true, cancel),
            OBU_TILE_GROUP => self.tile_group(data, obu.payload, obu.payload_start, cancel),
            OBU_METADATA => self.metadata_obu(obu.payload),
            OBU_TILE_LIST => Err(Error::Unsupported("large scale tile")),
            // Padding and reserved OBU types are ignored.
            _ => Ok(()),
        }
    }

    /// `sequence_header_obu()`. A sequence that differs from the active one starts a new coded
    /// video sequence: the reference slots are emptied.
    fn sequence_header(&mut self, payload: &[u8]) -> Result<(), Error> {
        let new = SequenceHeader::parse(payload, self.options.operating_point)?;
        if self
            .seq
            .as_ref()
            .is_some_and(|old| !same_sequence(old, &new))
        {
            self.refs = RefState::default();
            self.store = Default::default();
            self.pending = None;
            self.state = None;
        }
        self.seq = Some(new);
        Ok(())
    }

    /// `frame_header_obu()` and, for an `OBU_FRAME`, the tile group that follows the header.
    fn frame_header(
        &mut self,
        data: &[u8],
        obu: &Obu<'_>,
        is_frame: bool,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        if self.pending.is_some() {
            if is_frame {
                return Err(Error::Invalid("frame OBU while a frame is unfinished"));
            }
            // `frame_header_copy()`: a repeat of the header of the frame in progress.
            return Ok(());
        }
        let seq = self
            .seq
            .as_ref()
            .ok_or(Error::Invalid("frame header before the sequence header"))?;
        let mut r = BitReader::new(obu.payload);
        let h = FrameHeader::parse(
            &mut r,
            seq,
            &mut self.refs,
            obu.header.temporal_id,
            obu.header.spatial_id,
        )?;
        if h.show_existing_frame {
            if is_frame {
                return Err(Error::Invalid("show_existing_frame in a frame OBU"));
            }
            return self.show_existing(h);
        }
        self.check_limits(&h)?;
        if self.probe_only {
            let mono = seq.mono_chrome;
            let s = Shown::of_header(&h, seq);
            self.infos.push(frame_info(&s, &h, mono));
        }
        self.pending = Some(Pending {
            header: h,
            tiles: Vec::new(),
            next_tile: 0,
        });
        if is_frame {
            r.byte_alignment()?;
            let header_bytes = r.get_position() / 8;
            let rest = obu.payload.get(header_bytes..).ok_or(Error::Truncated)?;
            self.tile_group(data, rest, obu.payload_start + header_bytes, cancel)?;
        }
        Ok(())
    }

    /// The size limits, checked before any buffer of the frame is allocated.
    fn check_limits(&self, h: &FrameHeader) -> Result<(), Error> {
        if h.upscaled_width > self.limits.max_width {
            return Err(Error::WidthLimit);
        }
        if h.frame_height > self.limits.max_height {
            return Err(Error::HeightLimit);
        }
        if u64::from(h.upscaled_width) * u64::from(h.frame_height) > self.limits.max_pixels {
            return Err(Error::PixelLimit);
        }
        Ok(())
    }

    /// A `show_existing_frame` header: outputs the frame in the slot and, for a key frame, runs
    /// the reference frame loading and update processes (7.21, 7.20), which copy it to every slot.
    fn show_existing(&mut self, h: FrameHeader) -> Result<(), Error> {
        let idx = h.frame_to_show_map_idx;
        let s = Shown::of_slot(&self.refs.slots[idx]);
        if self.probe_only {
            let mono = self.seq.as_ref().is_some_and(|q| q.mono_chrome);
            self.infos.push(frame_info(&s, &h, mono));
        } else {
            let planes = self.store[idx].clone().ok_or(Error::Invalid(
                "show_existing_frame of a slot without a decoded frame",
            ))?;
            let frame = self.make_frame(&s, &h, &planes)?;
            self.ready.push(frame);
        }
        if h.frame_type == KEY_FRAME {
            self.refs.refresh_all_from(idx);
            let shared = self.store[idx].clone();
            for slot in self.store.iter_mut() {
                *slot = shared.clone();
            }
        }
        Ok(())
    }

    /// `tile_group_obu()`: records the tiles; when the last tile of the frame is in, decodes it.
    fn tile_group(
        &mut self,
        data: &[u8],
        payload: &[u8],
        payload_start: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let p = self
            .pending
            .as_mut()
            .ok_or(Error::Invalid("tile group without a frame header"))?;
        let tg = parse_tile_group(payload, payload_start, &p.header.tile_info, p.next_tile)?;
        p.next_tile = tg.tg_end + 1;
        p.tiles.extend_from_slice(&tg.tiles);
        let num_tiles = p.header.tile_info.tile_cols * p.header.tile_info.tile_rows;
        if p.next_tile == num_tiles {
            self.finish_frame(data, cancel)?;
        }
        Ok(())
    }

    /// All tiles of the frame are here: decode them, update the reference slots (7.20) and
    /// output the frame if it is shown (7.18).
    fn finish_frame(&mut self, data: &[u8], cancel: &AtomicBool) -> Result<(), Error> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let seq = self
            .seq
            .clone()
            .ok_or(Error::Invalid("frame without a sequence header"))?;
        let h = p.header;
        let mut stored = None;
        if !self.probe_only {
            self.state = Some(FrameState::new(&h, &seq, &self.ledger)?);
            self.decode_frame_tiles(&h, &seq, data, &p.tiles, cancel)?;
            let state = self
                .state
                .take()
                .ok_or(Error::Invalid("frame decode left no state"))?;
            if state.lr_frame.len() != usize::from(seq.num_planes) {
                return Err(Error::Invalid("frame decode produced no output planes"));
            }
            stored = Some(Arc::new(state.lr_frame));
        }
        self.refs.update(&h, &seq);
        for (i, slot) in self.store.iter_mut().enumerate() {
            if (h.refresh_frame_flags >> i) & 1 != 0 {
                *slot = stored.clone();
            }
        }
        if h.show_frame
            && let Some(planes) = &stored
        {
            let s = Shown::of_header(&h, &seq);
            let frame = self.make_frame(&s, &h, planes)?;
            self.ready.push(frame);
        }
        Ok(())
    }

    /// Decodes the tiles of the current frame (spec 5.11.1 `decode_tile()` for every tile in
    /// `tiles`, then the frame wrapup of 7.4: loop filter, CDEF, superres, loop restoration) and
    /// leaves the final frame in `self.state`'s `lr_frame`: one plane per colour plane, exactly
    /// `UpscaledWidth` x `FrameHeight` (chroma subsampled, rounded up) in size.
    ///
    /// `data` is the temporal unit; `tiles` are byte ranges into it, in tile order. `self.state`
    /// holds the allocated `CurrFrame`. Tiles decode on up to `Limits::max_threads` threads (one
    /// scope per frame, see docs/threading.md), serially for `max_threads` 1, a single tile or
    /// an `allow_intrabc` frame; `cancel` is checked before every tile and by every tile once per
    /// superblock row.
    ///
    /// The frame CDFs are the defaults for `base_q_idx` (`primary_ref_frame` is `NONE` in an
    /// intra frame). The tile CDFs the exit process returns are dropped: the frame end update
    /// (7.4) only feeds a later frame that loads them, and no intra frame does.
    ///
    /// # Errors
    /// `Truncated` and `Invalid` from the symbol decoder, `Cancelled`, `AllocLimit`,
    /// `OutOfMemory`.
    fn decode_frame_tiles(
        &mut self,
        h: &FrameHeader,
        seq: &SequenceHeader,
        data: &[u8],
        tiles: &[TileRange],
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let fs = self
            .state
            .take()
            .ok_or(Error::Invalid("frame decode without a frame state"))?;
        let mut fs = decode_tiles(
            fs,
            &self.ledger,
            self.limits.max_threads,
            h,
            seq,
            data,
            tiles,
            cancel,
        )?;
        fs.lr_frame = wrapup(&mut fs, seq, h)?;
        self.state = Some(fs);
        Ok(())
    }

    /// `metadata_obu()`: keeps the raw payload of every metadata OBU for the next output frame,
    /// within `Limits::max_metadata`, unless `Options::keep_metadata` is false.
    fn metadata_obu(&mut self, payload: &[u8]) -> Result<(), Error> {
        if !self.options.keep_metadata {
            return Ok(());
        }
        let m = parse_metadata(payload)?;
        let cost = m.payload.len() + size_of::<Metadata>();
        let total = self.metadata_bytes.saturating_add(cost);
        if total > self.limits.max_metadata {
            return Err(Error::MetadataLimit);
        }
        self.metadata_bytes = total;
        self.metadata.push(m);
        Ok(())
    }

    /// Builds the output frame from stored planes (spec 7.18): film grain on a copy when
    /// enabled, then the samples as `u8` or `u16`. The buffers it builds are charged to the
    /// ledger while it works; the returned frame is the caller's.
    fn make_frame(
        &mut self,
        s: &Shown,
        h: &FrameHeader,
        planes: &[WorkPlane],
    ) -> Result<Frame, Error> {
        let seq = self
            .seq
            .as_ref()
            .ok_or(Error::Invalid("output without a sequence header"))?;
        let mono = seq.mono_chrome;
        if planes.len() != if mono { 1 } else { 3 } {
            return Err(Error::Invalid("stored frame has the wrong plane count"));
        }
        let color = ColorInfo {
            color_primaries: seq.color_primaries,
            transfer_characteristics: seq.transfer_characteristics,
            matrix_coefficients: seq.matrix_coefficients,
            color_range: seq.color_range,
            chroma_sample_position: seq.chroma_sample_position,
        };
        let apply =
            self.options.film_grain && seq.film_grain_params_present && h.film_grain.apply_grain;
        let mut out = Vec::with_capacity(planes.len());
        if apply {
            let empty: &[u16] = &[];
            let input = GrainInput {
                bit_depth: u32::from(s.bit_depth),
                mono_chrome: mono,
                subsampling_x: s.subsampling_x != 0,
                subsampling_y: s.subsampling_y != 0,
                matrix_coefficients: seq.matrix_coefficients,
                width: s.width as usize,
                height: s.height as usize,
                planes: [
                    planes[0].px.as_slice(),
                    planes.get(1).map_or(empty, |p| p.px.as_slice()),
                    planes.get(2).map_or(empty, |p| p.px.as_slice()),
                ],
                strides: [
                    planes[0].stride,
                    planes.get(1).map_or(0, |p| p.stride),
                    planes.get(2).map_or(0, |p| p.stride),
                ],
            };
            // Same layout as the input: a plane as wide as the input stride.
            let mut grained = Vec::with_capacity(planes.len());
            for p in planes {
                grained.push(WorkPlane::new(p.stride, p.height, &self.ledger)?);
            }
            {
                let mut it = grained.iter_mut().map(|p| p.px.as_mut_slice());
                let o0 = it.next().unwrap_or_default();
                let o1 = it.next().unwrap_or_default();
                let o2 = it.next().unwrap_or_default();
                apply_film_grain(&h.film_grain, &input, [o0, o1, o2]);
            }
            for (g, p) in grained.iter().zip(planes) {
                out.push(self.to_public(g, p.width, p.height, s.bit_depth)?);
            }
        } else {
            for p in planes {
                out.push(self.to_public(p, p.width, p.height, s.bit_depth)?);
            }
        }
        let metadata = std::mem::take(&mut self.metadata);
        self.metadata_bytes = 0;
        Ok(Frame {
            width: s.width,
            height: s.height,
            render_width: s.render_width,
            render_height: s.render_height,
            bit_depth: s.bit_depth,
            mono_chrome: mono,
            subsampling_x: s.subsampling_x != 0,
            subsampling_y: s.subsampling_y != 0,
            color,
            planes: out,
            metadata,
            film_grain_applied: apply,
            frame_type: s.frame_type,
            temporal_id: h.temporal_id,
            spatial_id: h.spatial_id,
        })
    }

    /// Copies the visible `width` x `height` samples of `p` into an output plane, `u8` for 8-bit
    /// frames and `u16` otherwise. `width` and `height` are within the plane.
    fn to_public(
        &self,
        p: &WorkPlane,
        width: usize,
        height: usize,
        bit_depth: u8,
    ) -> Result<Plane, Error> {
        let n = width.checked_mul(height).ok_or(Error::AllocLimit)?;
        let bytes = if bit_depth == 8 {
            n
        } else {
            n.saturating_mul(2)
        };
        let _held = self.ledger.charge(bytes)?;
        let data = if bit_depth == 8 {
            let mut v: Vec<u8> = Vec::new();
            v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
            for row in 0..height {
                v.extend(row_of(p, row, width)?.iter().map(|&x| x as u8));
            }
            PlaneData::U8(v)
        } else {
            let mut v: Vec<u16> = Vec::new();
            v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
            for row in 0..height {
                v.extend_from_slice(row_of(p, row, width)?);
            }
            PlaneData::U16(v)
        };
        Ok(Plane {
            width: width as u32,
            height: height as u32,
            data,
        })
    }
}

/// The tile phase of a frame: `decode_tile()` of every tile of `tiles` into `fs` (spec 5.11.1).
///
/// With `max_threads` above 1 and more than one tile, the tiles decode on up to
/// `min( max_threads, tiles )` threads of one scope (the calling thread is one of them). A worker
/// decodes a tile into its own tile-sized [`FrameState::new_tile`] state, which the same tile
/// decode fills as it does the frame state, and merges it into `fs` (tiles are disjoint, so the
/// order of the merges does not matter). Otherwise the tiles decode in order straight into `fs`.
/// A frame with `allow_intrabc` is always the second case: intra block copy reads samples and
/// mode info of other tiles. The result is the same for every `max_threads` (docs/threading.md).
fn decode_tiles(
    mut fs: FrameState,
    ledger: &Arc<Ledger>,
    max_threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    data: &[u8],
    tiles: &[TileRange],
    cancel: &AtomicBool,
) -> Result<FrameState, Error> {
    let mut frame_cdf = CdfContext::new_boxed();
    frame_cdf.init_non_coeff_cdfs();
    frame_cdf.init_coeff_cdfs(h.base_q_idx);
    let tile_cols = h.tile_info.tile_cols.max(1);
    let decode =
        |fs: &mut FrameState, t: &TileRange, row: usize, col: usize| -> Result<(), Error> {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            let bytes = data
                .get(t.start..t.end)
                .ok_or(Error::Invalid("tile data outside the temporal unit"))?;
            TileDecoder::new(bytes, row, col, &frame_cdf, fs, h, seq)?.decode_tile(cancel)?;
            Ok(())
        };
    let threads = if h.allow_intrabc { 1 } else { max_threads };
    if threads.max(1).min(tiles.len()) <= 1 {
        for t in tiles {
            decode(&mut fs, t, t.tile_num / tile_cols, t.tile_num % tile_cols)?;
        }
        return Ok(fs);
    }
    let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
    let frame = Mutex::new(fs);
    run_parallel(tiles.len(), threads, |i| {
        let t = tiles
            .get(i)
            .ok_or(Error::Invalid("tile index out of range"))?;
        let mut local = FrameState::new_tile(
            h,
            seq,
            ledger,
            t.tile_num / tile_cols,
            t.tile_num % tile_cols,
        )?;
        decode(&mut local, t, 0, 0)?;
        frame
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .merge_tile(&local, sb_size);
        Ok(())
    })?;
    Ok(frame.into_inner().unwrap_or_else(|e| e.into_inner()))
}

/// Output planes of exactly `UpscaledWidth` x `FrameHeight` (chroma `(size + ss) >> ss`), each a
/// copy of the top left of the matching plane of `src` when given.
fn exact_planes(fs: &FrameState, src: Option<&[WorkPlane]>) -> Result<Vec<WorkPlane>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(fs.num_planes)
        .map_err(|_| Error::OutOfMemory)?;
    for i in 0..fs.num_planes {
        let (sx, sy) = if i == 0 {
            (0, 0)
        } else {
            (fs.subsampling_x, fs.subsampling_y)
        };
        let w = (fs.upscaled_width + sx) >> sx;
        let ht = (fs.frame_height + sy) >> sy;
        let mut p = WorkPlane::new(w, ht, &fs.ledger)?;
        if let Some(s) = src.and_then(|s| s.get(i)) {
            for y in 0..ht {
                let from =
                    s.px.get(y * s.stride..y * s.stride + w)
                        .ok_or(Error::Invalid("plane smaller than its size"))?;
                p.px[y * w..(y + 1) * w].copy_from_slice(from);
            }
        }
        out.push(p);
    }
    Ok(out)
}

/// The frame wrapup of spec 7.4 after the last tile: deblocking (7.14), CDEF (7.15), superres
/// (7.16) on the CDEF output and, for loop restoration, on the deblocked frame, then loop
/// restoration (7.17). Returns `LrFrame`.
fn wrapup(
    fs: &mut FrameState,
    seq: &SequenceHeader,
    h: &FrameHeader,
) -> Result<Vec<WorkPlane>, Error> {
    loop_filter_frame(fs, seq, h);
    let cdef = cdef_alloc(fs, seq, h)?;
    let (up_cdef, up_curr) = if h.frame_width != h.upscaled_width {
        let mut up_cdef = exact_planes(fs, None)?;
        upscale(&cdef, fs, seq, h, &mut up_cdef);
        drop(cdef);
        let up_curr = if h.uses_lr {
            let mut p = exact_planes(fs, None)?;
            upscale(&fs.curr_frame, fs, seq, h, &mut p);
            Some(p)
        } else {
            None
        };
        (up_cdef, up_curr)
    } else {
        (exact_planes(fs, Some(cdef.as_slice()))?, None)
    };
    if !h.uses_lr {
        return Ok(up_cdef);
    }
    let mut out = exact_planes(fs, Some(up_cdef.as_slice()))?;
    match &up_curr {
        Some(c) => lr_frame(fs, seq, h, &up_cdef, c, &mut out),
        None => lr_frame(fs, seq, h, &up_cdef, &fs.curr_frame, &mut out),
    }
    Ok(out)
}

/// The first `width` samples of row `row` of a plane.
fn row_of(p: &WorkPlane, row: usize, width: usize) -> Result<&[u16], Error> {
    let start = row * p.stride;
    p.px.get(start..start + width)
        .ok_or(Error::Invalid("plane smaller than its size"))
}

/// The header facts of a frame for [`probe`](crate::probe).
fn frame_info(s: &Shown, h: &FrameHeader, mono_chrome: bool) -> FrameInfo {
    FrameInfo {
        frame_type: s.frame_type,
        show_frame: h.show_frame,
        show_existing_frame: h.show_existing_frame,
        width: s.width,
        height: s.height,
        frame_width: s.frame_width,
        render_width: s.render_width,
        render_height: s.render_height,
        bit_depth: s.bit_depth,
        mono_chrome,
        subsampling_x: s.subsampling_x != 0,
        subsampling_y: s.subsampling_y != 0,
        temporal_id: h.temporal_id,
        spatial_id: h.spatial_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::tests::still_frame_header;
    use crate::seq::tests::{BitWriter, still_sequence_header};
    use crate::{decode, probe};

    /// An OBU with a one byte size field.
    fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = vec![(obu_type << 3) | 2, payload.len() as u8];
        v.extend_from_slice(payload);
        v
    }

    fn frame_obu_payload() -> Vec<u8> {
        let mut w = BitWriter::default();
        still_frame_header(&mut w);
        w.align();
        let mut payload = w.bytes;
        payload.extend_from_slice(&[1, 2, 3, 4]); // the tile data
        payload
    }

    /// Temporal delimiter, sequence header and one key frame (64x32), low-overhead format.
    fn still_obus() -> Vec<Vec<u8>> {
        vec![
            obu(OBU_TEMPORAL_DELIMITER, &[]),
            obu(OBU_SEQUENCE_HEADER, &still_sequence_header()),
            obu(OBU_FRAME, &frame_obu_payload()),
        ]
    }

    fn cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[test]
    fn probe_reads_the_headers() {
        let data: Vec<u8> = still_obus().concat();
        let p = probe(&data, Format::Section5, &Limits::default()).unwrap();
        assert_eq!(p.stopped, None);
        assert_eq!(p.frames.len(), 1);
        let f = &p.frames[0];
        assert_eq!(f.frame_type, KEY_FRAME);
        assert!(f.show_frame && !f.show_existing_frame);
        assert_eq!((f.width, f.height, f.frame_width), (64, 32, 64));
        assert_eq!((f.render_width, f.render_height), (64, 32));
        assert_eq!(f.bit_depth, 8);
        assert!(!f.mono_chrome && f.subsampling_x && f.subsampling_y);
    }

    #[test]
    fn probe_reads_annex_b_the_same() {
        let obus = still_obus();
        let mut frame_unit = Vec::new();
        for o in &obus {
            frame_unit.push(o.len() as u8); // obu_length
            frame_unit.extend_from_slice(o);
        }
        let mut stream = vec![(1 + frame_unit.len()) as u8, frame_unit.len() as u8];
        stream.extend_from_slice(&frame_unit);
        let p = probe(&stream, Format::AnnexB, &Limits::default()).unwrap();
        assert_eq!(p.stopped, None);
        assert_eq!(p.frames.len(), 1);
        assert_eq!(p.frames[0].width, 64);
    }

    #[test]
    fn size_limits_apply_before_allocation() {
        let data: Vec<u8> = still_obus().concat();
        let narrow = Limits {
            max_width: 32,
            ..Limits::default()
        };
        let p = probe(&data, Format::Section5, &narrow).unwrap();
        assert_eq!(p.stopped, Some(Error::WidthLimit));
        assert!(p.frames.is_empty());
        let flat = Limits {
            max_height: 16,
            ..Limits::default()
        };
        assert_eq!(
            probe(&data, Format::Section5, &flat).unwrap().stopped,
            Some(Error::HeightLimit)
        );
        let small = Limits {
            max_pixels: 64 * 31,
            ..Limits::default()
        };
        assert_eq!(
            probe(&data, Format::Section5, &small).unwrap().stopped,
            Some(Error::PixelLimit)
        );
    }

    #[test]
    fn frame_buffers_are_charged_to_max_alloc() {
        let data: Vec<u8> = still_obus().concat();
        // The 128x128 luma plane alone is 32 KiB of u16 samples.
        let tiny = Limits {
            max_alloc: 4096,
            ..Limits::default()
        };
        let e = decode(
            &data,
            Format::Section5,
            &Options::default(),
            &tiny,
            &cancel(),
        )
        .unwrap_err();
        assert_eq!(e, Error::AllocLimit);
    }

    #[test]
    fn ledger_peak_stays_within_max_alloc_and_is_released_on_error() {
        let data: Vec<u8> = still_obus().concat();
        for max_alloc in [1024, 4096, 40_000, 100_000, 1 << 20] {
            let mut d = Decoder::new(
                Options::default(),
                Limits {
                    max_alloc,
                    max_threads: 1,
                    ..Limits::default()
                },
            );
            let before = d.ledger.left();
            let r = d.decode_temporal_unit(&data, Format::Section5, &cancel());
            assert!(d.ledger.peak() <= max_alloc as u64);
            if r.is_err() {
                assert_eq!(d.ledger.left(), before);
            }
        }
    }

    #[test]
    fn cancel_flag_is_honoured() {
        let data: Vec<u8> = still_obus().concat();
        let stop = AtomicBool::new(true);
        let mut d = Decoder::new(Options::default(), Limits::default());
        assert_eq!(
            d.decode_temporal_unit(&data, Format::Section5, &stop),
            Err(Error::Cancelled)
        );
    }

    #[test]
    fn frame_before_sequence_header_is_invalid() {
        let data = [
            obu(OBU_TEMPORAL_DELIMITER, &[]),
            obu(OBU_FRAME, &frame_obu_payload()),
        ]
        .concat();
        let mut d = Decoder::new(Options::default(), Limits::default());
        let e = d
            .decode_temporal_unit(&data, Format::Section5, &cancel())
            .unwrap_err();
        assert_eq!(e.class(), "invalid");
    }

    #[test]
    fn tile_list_obu_is_unsupported() {
        let data = obu(OBU_TILE_LIST, &[0]);
        let mut d = Decoder::new(Options::default(), Limits::default());
        assert_eq!(
            d.decode_temporal_unit(&data, Format::Section5, &cancel()),
            Err(Error::Unsupported("large scale tile"))
        );
    }

    #[test]
    fn metadata_is_bounded_by_the_limit() {
        let m = obu(OBU_METADATA, &[1, 0x03, 0xe8, 0x01, 0x90, 0x80]);
        let mut d = Decoder::new(
            Options::default(),
            Limits {
                max_metadata: 8,
                ..Limits::default()
            },
        );
        assert_eq!(
            d.decode_temporal_unit(&m, Format::Section5, &cancel()),
            Err(Error::MetadataLimit)
        );
        let mut d = Decoder::new(Options::default(), Limits::default());
        assert!(
            d.decode_temporal_unit(&m, Format::Section5, &cancel())
                .unwrap()
                .is_empty()
        );
        assert_eq!(d.metadata.len(), 1);
    }

    #[test]
    fn padding_and_reserved_obus_are_ignored() {
        let data = [obu(OBU_PADDING, &[1, 2, 3]), obu(9, &[7])].concat();
        let mut d = Decoder::new(Options::default(), Limits::default());
        assert!(
            d.decode_temporal_unit(&data, Format::Section5, &cancel())
                .unwrap()
                .is_empty()
        );
    }

    /// A 192x128 4:2:0 frame header with 3 x 2 tiles of 64x64, loop restoration on, and the
    /// bytes of its tiles: pseudo-random data, which the symbol decoder reads as some valid
    /// syntax (or a syntax error, the same for every thread count).
    fn six_tiles(seed: u64) -> (FrameHeader, Vec<u8>, Vec<TileRange>) {
        let h = FrameHeader {
            frame_width: 192,
            frame_height: 128,
            upscaled_width: 192,
            mi_cols: 48,
            mi_rows: 32,
            base_q_idx: 60,
            tx_mode: TX_MODE_SELECT,
            frame_restoration_type: [RESTORE_WIENER, RESTORE_SGRPROJ, RESTORE_SWITCHABLE],
            loop_restoration_size: [64, 32, 32],
            uses_lr: true,
            tile_info: crate::header::TileInfo {
                tile_cols: 3,
                tile_rows: 2,
                mi_col_starts: vec![0, 16, 32, 48],
                mi_row_starts: vec![0, 16, 32],
                ..Default::default()
            },
            ..FrameHeader::default()
        };
        let mut x = seed | 1;
        let (mut data, mut tiles) = (Vec::new(), Vec::new());
        for tile_num in 0..6 {
            let start = data.len();
            for _ in 0..2000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                data.push((x >> 24) as u8);
            }
            tiles.push(TileRange {
                tile_num,
                start,
                end: data.len(),
            });
        }
        (h, data, tiles)
    }

    /// The tile phase of `six_tiles( seed )` with `threads` threads and `max_alloc`; the ledger is
    /// returned to read its peak.
    fn run_tiles(
        seed: u64,
        threads: usize,
        max_alloc: usize,
    ) -> (Result<FrameState, Error>, Arc<Ledger>) {
        let seq = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        let (h, data, tiles) = six_tiles(seed);
        let ledger = Arc::new(Ledger::default());
        ledger.reset(max_alloc, 0).unwrap();
        let r = FrameState::new(&h, &seq, &ledger)
            .and_then(|fs| decode_tiles(fs, &ledger, threads, &h, &seq, &data, &tiles, &cancel()));
        (r, ledger)
    }

    fn assert_same_state(a: &FrameState, b: &FrameState) {
        assert_eq!(a.mi_sizes, b.mi_sizes);
        assert_eq!(a.skips, b.skips);
        assert_eq!(a.is_inters, b.is_inters);
        assert_eq!(a.segment_ids, b.segment_ids);
        assert_eq!(a.y_modes, b.y_modes);
        assert_eq!(a.uv_modes, b.uv_modes);
        assert_eq!(a.ref_frames0, b.ref_frames0);
        assert_eq!(a.tx_sizes, b.tx_sizes);
        assert_eq!(a.palette_sizes, b.palette_sizes);
        assert_eq!(a.delta_lfs, b.delta_lfs);
        assert_eq!(a.loopfilter_tx_sizes, b.loopfilter_tx_sizes);
        assert_eq!(a.cdef_idx, b.cdef_idx);
        assert_eq!(a.lr_type, b.lr_type);
        assert_eq!(a.lr_wiener, b.lr_wiener);
        assert_eq!(a.lr_sgr_set, b.lr_sgr_set);
        assert_eq!(a.lr_sgr_xqd, b.lr_sgr_xqd);
        assert_eq!(a.curr_frame.len(), b.curr_frame.len());
        for (p, q) in a.curr_frame.iter().zip(&b.curr_frame) {
            assert_eq!(p.px, q.px);
        }
    }

    #[test]
    fn tiles_decode_the_same_for_every_thread_count() {
        let mut decoded = 0;
        for seed in 1..=6 {
            let (serial, _) = run_tiles(seed, 1, 64 << 20);
            for threads in [2, 3, 6, 8] {
                let (par, ledger) = run_tiles(seed, threads, 64 << 20);
                match (&serial, &par) {
                    (Ok(a), Ok(b)) => assert_same_state(a, b),
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    _ => panic!("serial and {threads} threads differ in Ok or Err"),
                }
                assert!(ledger.peak() <= 64 << 20);
                drop(par);
                assert_eq!(ledger.left(), (64 << 20) - crate::ledger::SMALL);
            }
            decoded += usize::from(serial.is_ok());
        }
        // Most random tile data is valid syntax; this guards against a test that compares nothing.
        assert!(decoded > 0);
    }

    #[test]
    fn tile_threads_stay_within_max_alloc() {
        for max_alloc in [1024, 100 << 10, 120 << 10, 400 << 10, 1 << 20] {
            for threads in [1, 2, 3] {
                let (r, ledger) = run_tiles(1, threads, max_alloc);
                assert!(ledger.peak() <= max_alloc as u64);
                assert!(r.is_ok() || r.as_ref().err() != Some(&Error::OutOfMemory));
                drop(r);
                assert_eq!(
                    ledger.left(),
                    max_alloc.saturating_sub(crate::ledger::SMALL)
                );
            }
        }
    }
}
