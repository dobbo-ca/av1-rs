//! The decoder state machine: OBU dispatch, the sequence and frame header handling, tile group
//! collection, the reference slots (spec 7.20, 7.21), frame output (7.18) and the limit checks.
//!
//! The pixel work of a frame is [`Decoder::decode_frame_tiles`]: the tiles, then the in-loop
//! filters. [`probe`](crate::probe) runs the same path without it.
#![allow(dead_code)]

use crate::bits::BitReader;
use crate::cdef::{cdef_alloc, cdef_frame, cdef_on, cdef_planes, cdef_rows};
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::error::Error;
use crate::frame::{FrameState, Plane as WorkPlane, View};
use crate::grain::{GrainInput, apply_film_grain};
use crate::header::{FrameHeader, RefInfo, RefState, TileRange, parse_tile_group};
use crate::ledger::Ledger;
use crate::lf::{lf_on, lf_plane_on, loop_filter_frame, loop_filter_rows};
use crate::limits::Limits;
use crate::lr::{lr_frame, lr_plane, lr_stripe};
use crate::obu::{Obu, ObuIter, parse_metadata};
use crate::parallel::{Team, with_team};
use crate::seq::SequenceHeader;
use crate::superres::{upscale, upscale_rows};
use crate::tile::TileDecoder;
use crate::{ColorInfo, Format, Frame, FrameInfo, Metadata, Options, Plane, PlaneData};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{
    Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    TryLockError,
};

/// A frame whose header was read and whose tile groups are still coming.
#[derive(Debug)]
struct Pending {
    header: FrameHeader,
    /// The tiles received so far, in order.
    tiles: Vec<TileRange>,
    /// `TileNum` the next tile group must start at.
    next_tile: usize,
}

/// Most plane lists of recycled frames a decoder keeps.
const SPARE_LISTS: usize = 16;

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
    /// Charge the kept metadata to `ledger` (set with a shared ledger, so `max_metadata` is not
    /// counted per frame and per decoder on top of it).
    charge_metadata: bool,
    /// Output frames of the current temporal unit, before the layer selection.
    ready: Vec<Frame>,
    /// Ledger bytes held by the planes built for `ready` (and by a frame that failed half-built),
    /// given back when the unit's frames are handed out.
    ready_bytes: usize,
    infos: Vec<FrameInfo>,
    /// The frame state of the last frame (mode-info arrays, tile buffers, frame CDFs; its planes
    /// are in the ledger's pool), reused by the next frame of the same shape.
    spare: Option<FrameState>,
    /// The empty tile list of the next frame (kept for its capacity).
    tile_buf: Vec<TileRange>,
    /// Empty plane lists of the frames handed back with [`Decoder::recycle`].
    spare_planes: Vec<Vec<Plane>>,
    /// False only for the `fuzzing` oracle: no frame state is reused.
    pub(crate) reuse: bool,
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
        d.charge_metadata = true;
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
            charge_metadata: false,
            ready: Vec::new(),
            ready_bytes: 0,
            infos: Vec::new(),
            spare: None,
            tile_buf: Vec::new(),
            spare_planes: Vec::new(),
            reuse: true,
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
        // Tile ranges are offsets into one call's `data`: a frame cannot continue in the next call.
        self.pending = None;
        self.ready.clear();
        self.release_ready();
        self.clear_metadata();
        for obu in ObuIter::new(data, format) {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            if let Err(e) = obu.and_then(|o| self.handle_obu(data, o, cancel)) {
                // After an error the decoder keeps no frame state, only pooled buffers.
                self.spare = None;
                return Err(e);
            }
        }
        Ok(self.select_output())
    }

    /// Hands frames returned by this decoder back for reuse: their sample buffers, the plane
    /// lists and the `Vec` itself. The next frames this decoder outputs are built in them, so with
    /// the frame buffers the decoder keeps from one frame to the next (`docs/limits.md`), a
    /// [`Decoder::decode_temporal_unit`] of the same shape as the previous one, with
    /// `Limits::max_threads` 1, allocates nothing once its frames were recycled.
    ///
    /// The buffers count against `Limits::max_alloc` while the decoder keeps them; a buffer that
    /// does not fit is dropped, and kept buffers are dropped when a later allocation needs their
    /// room, so recycling never makes a decode fail. Frames of another decoder may be recycled
    /// too (only their buffers are kept).
    ///
    /// # Errors
    /// None: this cannot fail.
    pub fn recycle(&mut self, mut frames: Vec<Frame>) {
        for mut f in frames.drain(..) {
            for p in f.planes.drain(..) {
                match p.data {
                    PlaneData::U8(v) => self.ledger.adopt(v),
                    PlaneData::U16(v) => self.ledger.adopt(v),
                }
            }
            // The lists are not charged (a few words each), so only a few are kept.
            if f.planes.capacity() > 0
                && self.spare_planes.len() < SPARE_LISTS
                && self.spare_planes.try_reserve(1).is_ok()
            {
                self.spare_planes.push(f.planes);
            }
        }
        if self.ready.is_empty() && self.ready.capacity() < frames.capacity() {
            self.ready = frames;
        }
    }

    /// Gives back the ledger bytes held by the unit's output planes.
    fn release_ready(&mut self) {
        self.ledger.release(std::mem::take(&mut self.ready_bytes));
    }

    /// Books the samples of `frames` (returned by this decoder) on its ledger until the decoder
    /// drops: [`crate::decode`] keeps every frame of the call, so they count against `max_alloc`.
    pub(crate) fn hold_output(&self, frames: &[Frame]) -> Result<(), Error> {
        for f in frames {
            self.ledger.reserve(crate::avif::frame_bytes(f))?;
        }
        Ok(())
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
        self.release_ready();
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
                self.clear_metadata();
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
            return self.show_existing(h, cancel);
        }
        self.check_limits(&h)?;
        if self.probe_only {
            let mono = seq.mono_chrome;
            let s = Shown::of_header(&h, seq);
            self.infos.push(frame_info(&s, &h, mono));
        }
        let mut tiles = std::mem::take(&mut self.tile_buf);
        tiles.clear();
        self.pending = Some(Pending {
            header: h,
            tiles,
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
    fn show_existing(&mut self, h: FrameHeader, cancel: &AtomicBool) -> Result<(), Error> {
        let idx = h.frame_to_show_map_idx;
        let s = Shown::of_slot(&self.refs.slots[idx]);
        if self.probe_only {
            let mono = self.seq.as_ref().is_some_and(|q| q.mono_chrome);
            self.infos.push(frame_info(&s, &h, mono));
        } else {
            let planes = self.store[idx].clone().ok_or(Error::Invalid(
                "show_existing_frame of a slot without a decoded frame",
            ))?;
            let frame = self.make_frame(&s, &h, &planes, cancel)?;
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
        let tg = parse_tile_group(
            payload,
            payload_start,
            &p.header.tile_info,
            p.next_tile,
            &mut p.tiles,
        )?;
        p.next_tile = tg.tg_end + 1;
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
        let mut tiles = p.tiles;
        let mut stored = None;
        if !self.probe_only {
            // The slots this frame refreshes are emptied first, as dav1d does (a frame that fails
            // leaves them empty), so the planes of a frame no slot keeps any more are free for this
            // one, and its `Arc` holds the new frame.
            let mut spare_arc = None;
            for (i, slot) in self.store.iter_mut().enumerate() {
                if (h.refresh_frame_flags >> i) & 1 != 0
                    && let Some(mut a) = slot.take()
                    && let Some(v) = Arc::get_mut(&mut a)
                {
                    self.ledger.put_planes(std::mem::take(v));
                    spare_arc = Some(a);
                }
            }
            let spare = self.spare.take().filter(|_| self.reuse);
            self.state = Some(FrameState::reuse(spare, &h, &seq, &self.ledger)?);
            let decoded = self.decode_frame_tiles(&h, &seq, data, &tiles, cancel);
            tiles.clear();
            self.tile_buf = tiles;
            decoded?;
            let mut state = self
                .state
                .take()
                .ok_or(Error::Invalid("frame decode left no state"))?;
            if state.lr_frame.len() != usize::from(seq.num_planes) {
                return Err(Error::Invalid("frame decode produced no output planes"));
            }
            let planes = std::mem::take(&mut state.lr_frame);
            stored = match spare_arc.as_mut().and_then(Arc::get_mut) {
                Some(v) => {
                    *v = planes;
                    spare_arc
                }
                None => Some(Arc::new(planes)),
            };
            // `CurrFrame` goes to the pool; the arrays wait for the next frame.
            state.curr_frame.clear();
            self.spare = Some(state);
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
            let frame = self.make_frame(&s, &h, planes, cancel)?;
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
    /// holds the allocated `CurrFrame`. The tiles and the filters run on up to
    /// `Limits::max_threads` threads (one scope per frame, see [`decode_frame`] and
    /// docs/threading.md); `cancel` is checked before every tile, by every tile once per
    /// superblock row and by every filter item.
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
        let fs = decode_frame(
            fs,
            &self.ledger,
            self.limits.max_threads,
            h,
            seq,
            data,
            tiles,
            cancel,
        )?;
        self.state = Some(fs);
        Ok(())
    }

    /// `metadata_obu()`: keeps the raw payload of every metadata OBU for the next output frame,
    /// within `Limits::max_metadata`, unless `Options::keep_metadata` is false.
    fn metadata_obu(&mut self, payload: &[u8]) -> Result<(), Error> {
        if !self.options.keep_metadata {
            return Ok(());
        }
        // An OBU larger than the whole limit is refused before it is copied.
        if payload.len() > self.limits.max_metadata {
            return Err(Error::MetadataLimit);
        }
        let m = parse_metadata(payload)?;
        let cost = m.payload.len() + size_of::<Metadata>();
        let total = self.metadata_bytes.saturating_add(cost);
        if total > self.limits.max_metadata {
            return Err(Error::MetadataLimit);
        }
        if self.charge_metadata {
            self.ledger.reserve(cost)?;
        }
        self.metadata_bytes = total;
        self.metadata.push(m);
        Ok(())
    }

    /// Drops the kept metadata, and gives back what a shared ledger was charged for it.
    fn clear_metadata(&mut self) {
        if self.charge_metadata {
            self.ledger.release(self.metadata_bytes);
        }
        self.metadata.clear();
        self.metadata_bytes = 0;
    }

    /// Builds the output frame from stored planes (spec 7.18): film grain on a copy when
    /// enabled, then the samples as `u8` or `u16`. The buffers it builds are charged to the
    /// ledger while it works; the returned frame is the caller's.
    fn make_frame(
        &mut self,
        s: &Shown,
        h: &FrameHeader,
        planes: &[WorkPlane],
        cancel: &AtomicBool,
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
        let mut out = self.spare_planes.pop().unwrap_or_default();
        out.try_reserve_exact(planes.len())
            .map_err(|_| Error::OutOfMemory)?;
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
            let mut grained = self.ledger.planes(planes.len())?;
            for p in planes {
                grained.push(WorkPlane::new(p.stride, p.height, &self.ledger)?);
            }
            {
                let mut it = grained.iter_mut().map(|p| p.px.as_mut_slice());
                let o0 = it.next().unwrap_or_default();
                let o1 = it.next().unwrap_or_default();
                let o2 = it.next().unwrap_or_default();
                apply_film_grain(
                    &h.film_grain,
                    &input,
                    [o0, o1, o2],
                    &self.ledger,
                    cancel,
                    self.limits.max_threads,
                    crate::simd::best(),
                )?;
            }
            for (g, p) in grained.iter().zip(planes) {
                out.push(self.public_plane(g, p.width, p.height, s.bit_depth)?);
            }
            self.ledger.put_planes(grained);
        } else {
            for p in planes {
                out.push(self.public_plane(p, p.width, p.height, s.bit_depth)?);
            }
        }
        // The frame owns the metadata now; a shared ledger keeps it booked for the rest of the call.
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
    fn public_plane(
        &mut self,
        p: &WorkPlane,
        width: usize,
        height: usize,
        bit_depth: u8,
    ) -> Result<Plane, Error> {
        let n = width.checked_mul(height).ok_or(Error::AllocLimit)?;
        // The buffer (recycled or new) stays charged until the unit's frames are handed out
        // (`release_ready`), so frames waiting in `ready` count against `max_alloc`.
        let data = if bit_depth == 8 {
            let (mut v, bytes) = self.ledger.take::<u8>(n)?;
            self.ready_bytes += bytes;
            for row in 0..height {
                v.extend(row_of(p, row, width)?.iter().map(|&x| x as u8));
            }
            PlaneData::U8(v)
        } else {
            let (mut v, bytes) = self.ledger.take::<u16>(n)?;
            self.ready_bytes += bytes;
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

/// The pixel work of a frame: `decode_tile()` of every tile of `tiles` into `fs` (spec 5.11.1),
/// then the wrapup of 7.4 (loop filter, CDEF, superres, loop restoration); returns `fs` with
/// `LrFrame` in `lr_frame`.
///
/// With one thread ([`team_size`]) the tiles decode in order straight into `fs` and
/// [`wrapup`] runs on the whole frame, spawning nothing. Otherwise one team of threads
/// ([`with_team`], one scope per frame) runs the stages of [`Wrap::lead`]: the tiles (each
/// into its own tile-sized [`FrameState::new_tile`] state, merged into `fs`; an `allow_intrabc`
/// frame decodes them in order on the calling thread, since intra block copy reads other
/// tiles), then each filter in row bands, column strips or stripes. Every stage writes disjoint
/// samples and keeps the spec order where it matters, so the result is the same for every
/// `max_threads` (docs/threading.md). A frame of one tile is row pipelined: the filters of a
/// band start as soon as the tile decode has finished the rows they read.
fn decode_frame(
    mut fs: FrameState,
    ledger: &Arc<Ledger>,
    max_threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    data: &[u8],
    tiles: &[TileRange],
    cancel: &AtomicBool,
) -> Result<FrameState, Error> {
    // Every field is set by the two inits, so the CDFs of an earlier frame can be reused.
    let mut frame_cdf = fs.frame_cdf.take().unwrap_or_else(CdfContext::new_boxed);
    frame_cdf.init_non_coeff_cdfs();
    frame_cdf.init_coeff_cdfs(h.base_q_idx);
    let threads = team_size(max_threads, &fs, tiles.len());
    if threads <= 1 {
        let tile_cols = h.tile_info.tile_cols.max(1);
        for t in tiles {
            let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
            decode_tile(
                &mut fs,
                &frame_cdf,
                data,
                t,
                row,
                col,
                h,
                seq,
                cancel,
                &mut |_, _| Ok(()),
            )?;
        }
        fs.lr_frame = wrapup(&mut fs, seq, h, cancel)?;
        fs.frame_cdf = Some(frame_cdf);
        return Ok(fs);
    }
    let w = Wrap {
        h,
        seq,
        data,
        tiles,
        cancel,
        ledger,
        frame_cdf,
        threads,
        tiles_fs: Mutex::new(Some(fs)),
        fs: OnceLock::new(),
        curr: RwLock::new(Vec::new()),
        cdef: RwLock::new(Vec::new()),
        up_cdef: RwLock::new(Vec::new()),
        up_curr: RwLock::new(Vec::new()),
        out: Mutex::new(Vec::new()),
        shadow: RwLock::new(None),
        progress: Mutex::new(Progress::default()),
        moved: Condvar::new(),
    };
    let item: &(dyn Fn(Job, usize) -> Result<(), Error> + Sync) = &|job, i| w.item(job, i);
    let planes = with_team(threads, item, |team| w.lead(team))?;
    let fs = match w.fs.into_inner() {
        Some(fs) => Some(fs),
        // The pipelined decode leaves the frame state where the tile decode wrote it.
        None => w.tiles_fs.into_inner().unwrap_or_else(|e| e.into_inner()),
    };
    let mut fs = fs.ok_or(Error::Invalid("frame decode left no state"))?;
    fs.curr_frame = w.curr.into_inner().unwrap_or_else(|e| e.into_inner());
    fs.lr_frame = planes;
    fs.frame_cdf = Some(w.frame_cdf);
    Ok(fs)
}

/// The threads of a frame decode: `max_threads` (0 counts as 1), but no more than its largest
/// stage has items (the tiles, or the 64-row bands of all planes).
fn team_size(max_threads: usize, fs: &FrameState, tiles: usize) -> usize {
    let bands = fs.mi_rows.div_ceil(16) * fs.num_planes;
    max_threads.max(1).min(tiles.max(bands))
}

/// `decode_tile()` of tile `t` (tile row `row`, column `col` of `fs`'s tiling) into `fs`,
/// calling `row_done` after each superblock row ([`TileDecoder::decode_tile`]).
fn decode_tile(
    fs: &mut FrameState,
    frame_cdf: &CdfContext,
    data: &[u8],
    t: &TileRange,
    row: usize,
    col: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    cancel: &AtomicBool,
    row_done: &mut dyn FnMut(&FrameState, usize) -> Result<(), Error>,
) -> Result<(), Error> {
    check_cancel(cancel)?;
    let bytes = data
        .get(t.start..t.end)
        .ok_or(Error::Invalid("tile data outside the temporal unit"))?;
    TileDecoder::new(bytes, row, col, frame_cdf, fs, h, seq)?.decode_tile(cancel, row_done)?;
    Ok(())
}

/// A stage of the threaded frame decode; its items are listed at [`Wrap::item`].
#[derive(Debug, Clone, Copy)]
enum Job {
    Tile,
    Pipe,
    LfRows,
    LfCols,
    Cdef,
    UpCdef,
    UpCurr,
    Lr,
}

/// The shared state of a threaded frame decode ([`decode_frame`]). Each buffer is written in
/// one stage and read in later ones. Readers hold a read lock for a whole item; writers fill a
/// private band (charged to the ledger) and copy it in under the lock, so threads only wait for
/// each other's copies. The calling thread holds no lock while it runs a stage.
struct Wrap<'a> {
    h: &'a FrameHeader,
    seq: &'a SequenceHeader,
    data: &'a [u8],
    tiles: &'a [TileRange],
    cancel: &'a AtomicBool,
    ledger: &'a Arc<Ledger>,
    frame_cdf: Box<CdfContext>,
    threads: usize,
    /// The frame state while the tiles merge into it.
    tiles_fs: Mutex<Option<FrameState>>,
    /// The frame state after the tiles: mode info, read only. Its `curr_frame` is in `curr`.
    fs: OnceLock<FrameState>,
    /// `CurrFrame`, loop filtered in place.
    curr: RwLock<Vec<WorkPlane>>,
    /// `CdefFrame`.
    cdef: RwLock<Vec<WorkPlane>>,
    /// `UpscaledCdefFrame` (exact size; a copy of `CdefFrame` without superres).
    up_cdef: RwLock<Vec<WorkPlane>>,
    /// `UpscaledCurrFrame` (superres with loop restoration only).
    up_curr: RwLock<Vec<WorkPlane>>,
    /// `LrFrame` (loop restoration only).
    out: Mutex<Vec<WorkPlane>>,
    /// A pipelined frame (one tile): the mode info the filters read, a copy of the rows the
    /// tile decode (which works in `tiles_fs`) has handed over.
    shadow: RwLock<Option<FrameState>>,
    /// A pipelined frame: the bands handed over and the filter tasks started and done.
    progress: Mutex<Progress>,
    /// Signalled when `progress` changes.
    moved: Condvar,
}

impl Wrap<'_> {
    /// The stages, in the order of [`wrapup`] and with its allocations in the same order (the
    /// private bands of the items come on top). Returns `LrFrame`.
    ///
    /// A frame of one tile is pipelined instead: one stage ([`Job::Pipe`]) in which one thread
    /// decodes the tile and the others filter the bands it has finished
    /// ([`Wrap::pipe_decode`], [`Wrap::pipe_filter`]).
    fn lead(&self, team: &Team<'_, Job>) -> Result<Vec<WorkPlane>, Error> {
        if self.tiles.len() == 1 {
            self.pipe_start()?;
            team.run(Job::Pipe, self.threads)?;
            return self.pipe_output();
        }
        let (h, seq, cancel) = (self.h, self.seq, self.cancel);
        if h.allow_intrabc || self.tiles.len() <= 1 {
            let mut g = lock(&self.tiles_fs);
            let fs = g
                .as_mut()
                .ok_or(Error::Invalid("frame decode without a frame state"))?;
            let tile_cols = h.tile_info.tile_cols.max(1);
            for t in self.tiles {
                let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
                decode_tile(
                    fs,
                    &self.frame_cdf,
                    self.data,
                    t,
                    row,
                    col,
                    h,
                    seq,
                    cancel,
                    &mut |_, _| Ok(()),
                )?;
            }
        } else {
            team.run(Job::Tile, self.tiles.len())?;
        }
        let mut fs = lock(&self.tiles_fs)
            .take()
            .ok_or(Error::Invalid("frame decode without a frame state"))?;
        *write_lock(&self.curr) = std::mem::take(&mut fs.curr_frame);
        let fs = self.fs.get_or_init(|| fs);
        let bands = fs.mi_rows.div_ceil(16);
        if lf_on(h) {
            team.run(Job::LfRows, fs.num_planes * bands)?;
            team.run(Job::LfCols, fs.num_planes * self.strips(fs))?;
        }
        let mut cdef = cdef_planes(&read_lock(&self.curr), self.ledger)?;
        let on = cdef_on(fs, &read_lock(&self.curr), seq, h, &cdef);
        if !on {
            // CDEF off: `CdefFrame` is a copy of `CurrFrame`.
            cdef_frame(fs, &read_lock(&self.curr), seq, h, &mut cdef, cancel)?;
        }
        *write_lock(&self.cdef) = cdef;
        if on {
            team.run(Job::Cdef, bands)?;
        }
        let row_bands = fs.num_planes * bands.max(1);
        if h.frame_width != h.upscaled_width {
            *write_lock(&self.up_cdef) = exact_planes(fs, None)?;
            team.run(Job::UpCdef, row_bands)?;
            write_lock(&self.cdef).clear();
            if h.uses_lr {
                *write_lock(&self.up_curr) = exact_planes(fs, None)?;
                team.run(Job::UpCurr, row_bands)?;
            }
        } else {
            *write_lock(&self.up_cdef) = exact_planes(fs, Some(read_lock(&self.cdef).as_slice()))?;
        }
        if !h.uses_lr {
            return Ok(std::mem::take(&mut *write_lock(&self.up_cdef)));
        }
        *lock(&self.out) = exact_planes(fs, Some(read_lock(&self.up_cdef).as_slice()))?;
        team.run(Job::Lr, fs.num_planes.min(3) * stripes(fs))?;
        Ok(std::mem::take(&mut *lock(&self.out)))
    }

    /// Item `i` of stage `job`:
    /// - `Tile`: tile `i` of `tiles`.
    /// - `Pipe`: item 0 decodes the one tile, the others run filter tasks.
    /// - `LfRows`: the vertical edges of plane `i / bands`, row band `i % bands` (16 mode info
    ///   rows; the last band runs to the plane end).
    /// - `LfCols`: the horizontal edges of plane `i / strips`, column strip `i % strips`.
    /// - `Cdef`: row band `i` of every plane.
    /// - `UpCdef`, `UpCurr`: plane `i / bands`, row band `i % bands` of the plane.
    /// - `Lr`: plane `i / stripes`, stripe `i % stripes`.
    ///
    /// An index past the bands of its plane does nothing.
    fn item(&self, job: Job, i: usize) -> Result<(), Error> {
        match job {
            Job::Tile => self.tile(i),
            Job::Pipe if i == 0 => self.pipe_decode(),
            Job::Pipe => self.pipe_filter(),
            Job::LfRows => self.lf_rows(self.state()?, i),
            Job::LfCols => self.lf_cols(i),
            Job::Cdef => self.cdef_band(self.state()?, i, true),
            Job::UpCdef => self.upscale_band(self.state()?, &self.cdef, &self.up_cdef, i),
            Job::UpCurr => self.upscale_band(self.state()?, &self.curr, &self.up_curr, i),
            Job::Lr => self.lr_band(self.state()?, i),
        }
    }

    /// Allocates the buffers of a pipelined frame, all at once: the shadow state (whose planes
    /// become the filtered `CurrFrame`), `CdefFrame`, `UpscaledCdefFrame`, `UpscaledCurrFrame`
    /// (superres with loop restoration) and `LrFrame` (loop restoration), and sets up the
    /// tasks.
    fn pipe_start(&self) -> Result<(), Error> {
        let (h, seq) = (self.h, self.seq);
        let mut sh = FrameState::new(h, seq, self.ledger)?;
        let curr = std::mem::take(&mut sh.curr_frame);
        let cdef = cdef_planes(&curr, self.ledger)?;
        let on = cdef_on(&sh, &curr, seq, h, &cdef);
        *write_lock(&self.up_cdef) = exact_planes(&sh, None)?;
        if h.uses_lr {
            if h.frame_width != h.upscaled_width {
                *write_lock(&self.up_curr) = exact_planes(&sh, None)?;
            }
            *lock(&self.out) = exact_planes(&sh, None)?;
        }
        *lock(&self.progress) = Progress::new(&sh, lf_on(h), h.uses_lr, on)?;
        *write_lock(&self.curr) = curr;
        *write_lock(&self.cdef) = cdef;
        *write_lock(&self.shadow) = Some(sh);
        Ok(())
    }

    /// `LrFrame` of a pipelined frame once every task is done: a plane without loop
    /// restoration is the `UpscaledCdefFrame` plane, which its (empty) tasks did not copy.
    fn pipe_output(&self) -> Result<Vec<WorkPlane>, Error> {
        let mut up_cdef = std::mem::take(&mut *write_lock(&self.up_cdef));
        if !self.h.uses_lr {
            return Ok(up_cdef);
        }
        let mut out = std::mem::take(&mut *lock(&self.out));
        let g = read_lock(&self.shadow);
        let fs = g
            .as_ref()
            .ok_or(Error::Invalid("frame filters before the tiles"))?;
        let curr = if self.h.frame_width != self.h.upscaled_width {
            read_lock(&self.up_curr)
        } else {
            read_lock(&self.curr)
        };
        for (plane, (o, c)) in out.iter_mut().zip(up_cdef.iter_mut()).enumerate() {
            let u = curr.get(plane).ok_or(Error::Invalid("plane missing"))?;
            let size = exact_size(fs, plane);
            if lr_plane(fs, self.seq, self.h, plane, [&*c, u], size).is_none() {
                std::mem::swap(o, c);
            }
        }
        Ok(out)
    }

    /// Item 0 of a pipelined frame: decodes the tile into `tiles_fs` and, after each superblock
    /// row, hands the bands it finished to the filters ([`Wrap::hand_over`]; when a filter
    /// holds a lock it goes on and hands them over after the next row, waiting for the lock
    /// then). Stops early, without an error of its own, when a filter task failed.
    fn pipe_decode(&self) -> Result<(), Error> {
        let abort = Abort {
            w: self,
            done: false,
        };
        let mut g = lock(&self.tiles_fs);
        let fs = g
            .as_mut()
            .ok_or(Error::Invalid("frame decode without a frame state"))?;
        let t = self
            .tiles
            .first()
            .ok_or(Error::Invalid("tile index out of range"))?;
        let (h, seq) = (self.h, self.seq);
        let tile_cols = h.tile_info.tile_cols.max(1);
        let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let (mut given, mut missed, mut stopped) = (0, false, false);
        let r = decode_tile(
            fs,
            &self.frame_cdf,
            self.data,
            t,
            row,
            col,
            h,
            seq,
            self.cancel,
            &mut |dec, end| {
                let done = if end >= dec.mi_rows { bands } else { end / 16 };
                if done > given {
                    // After one miss it waits, so the filters lag at most one more row.
                    missed = !self.hand_over(dec, given, done, missed);
                    if !missed {
                        given = done;
                        lock(&self.progress).ready = done;
                        self.moved.notify_all();
                    }
                }
                if lock(&self.progress).failed {
                    stopped = true;
                    return Err(Error::Cancelled);
                }
                Ok(())
            },
        );
        if stopped {
            // The failed filter task's error is the frame's.
            return Ok(());
        }
        r?;
        if given < bands {
            self.hand_over(fs, given, bands, true);
        }
        let mut p = lock(&self.progress);
        p.ready = bands;
        p.decoded = true;
        drop(p);
        abort.disarm();
        Ok(())
    }

    /// Copies bands `from..to` (`from < to`) of the tile decode's state `dec` into the shared
    /// state: the mode info rows into `shadow`, the samples into `CurrFrame`. With `wait` false
    /// it does nothing and returns false when a filter holds either lock.
    fn hand_over(&self, dec: &FrameState, from: usize, to: usize, wait: bool) -> bool {
        let guards = if wait {
            Some((write_lock(&self.shadow), write_lock(&self.curr)))
        } else {
            try_write(&self.shadow).zip(try_write(&self.curr))
        };
        let Some((mut sh, mut curr)) = guards else {
            return false;
        };
        let bands = dec.mi_rows.div_ceil(16).max(1);
        if let Some(sh) = sh.as_mut() {
            sh.copy_rows(dec, 16 * from..(16 * to).min(dec.mi_rows));
        }
        for (plane, (d, s)) in curr.iter_mut().zip(&dec.curr_frame).enumerate() {
            let (_, sy) = subsampling(dec, plane);
            let y0 = band_rows(from, bands, sy, s.height).0;
            let y1 = band_rows(to - 1, bands, sy, s.height).1;
            if let (Some(d), Some(s)) = (
                d.px.get_mut(y0 * d.stride..y1 * d.stride),
                s.px.get(y0 * s.stride..y1 * s.stride),
            ) && d.len() == s.len()
            {
                d.copy_from_slice(s);
            }
        }
        true
    }

    /// Items 1.. of a pipelined frame: run filter tasks as they become ready, until every task
    /// is done or a thread failed.
    fn pipe_filter(&self) -> Result<(), Error> {
        loop {
            let task = {
                let mut p = lock(&self.progress);
                loop {
                    if p.failed || p.complete() {
                        return Ok(());
                    }
                    if let Some(t) = p.next() {
                        p.running += 1;
                        break t;
                    }
                    if p.decoded && p.running == 0 {
                        // Cannot happen: every band is in, so some task is ready.
                        p.failed = true;
                        drop(p);
                        self.moved.notify_all();
                        return Err(Error::Invalid("frame filters stalled"));
                    }
                    p = self.moved.wait(p).unwrap_or_else(|e| e.into_inner());
                }
            };
            let abort = Abort {
                w: self,
                done: false,
            };
            self.task(task)?;
            lock(&self.progress).finish(task);
            abort.disarm();
        }
    }

    /// Runs filter task `t` of a pipelined frame on the shadow state's mode info.
    fn task(&self, t: Task) -> Result<(), Error> {
        let g = read_lock(&self.shadow);
        let fs = g
            .as_ref()
            .ok_or(Error::Invalid("frame filters before the tiles"))?;
        let bands = fs.mi_rows.div_ceil(16).max(1);
        match t {
            Task::V(plane, b) => self.lf_rows(fs, plane * bands + b),
            Task::H(plane, b) => self.lf_band(fs, plane, b),
            Task::C(b) => {
                let on = lock(&self.progress).cdef;
                self.cdef_band(fs, b, on)
            }
            Task::U(b) if self.h.frame_width == self.h.upscaled_width => self.copy_band(fs, b),
            Task::U(b) => {
                for plane in 0..fs.num_planes {
                    let i = plane * bands + b;
                    self.upscale_band(fs, &self.cdef, &self.up_cdef, i)?;
                    if self.h.uses_lr {
                        self.upscale_band(fs, &self.curr, &self.up_curr, i)?;
                    }
                }
                Ok(())
            }
            Task::L(plane, n) => self.lr_band(fs, plane * stripes(fs) + n),
        }
    }

    /// Horizontal edges of row band `b` of one plane, after those of the band above: an edge
    /// reaches at most 7 rows above it, and none past the band's end (it sits at a multiple of
    /// its filter size, which is at least its reach), so the band is copied with 8 rows above.
    fn lf_band(&self, fs: &FrameState, plane: usize, b: usize) -> Result<(), Error> {
        if plane >= fs.num_planes || !lf_plane_on(self.h, plane) {
            return Ok(());
        }
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let (_, sy) = subsampling(fs, plane);
        let (mut buf, y0, y1) = {
            let curr = read_lock(&self.curr);
            let p = curr.get(plane).ok_or(Error::Invalid("plane missing"))?;
            let (top, y1) = band_rows(b, bands, sy, p.height);
            let y0 = top.saturating_sub(8);
            let mut buf = WorkPlane::new(p.stride, y1 - y0, self.ledger)?;
            buf.px.copy_from_slice(rows_of(p, y0, y1)?);
            (buf, y0, y1)
        };
        let rows = 16 * b..(16 * b + 16).min(fs.mi_rows);
        let mut v = View {
            stride: buf.stride,
            px: buf.px.as_mut_slice(),
            x0: 0,
            y0,
        };
        loop_filter_rows(
            fs,
            self.h,
            plane,
            1,
            rows,
            0..fs.mi_cols,
            &mut v,
            self.cancel,
        )?;
        let mut curr = write_lock(&self.curr);
        let p = curr.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        rows_of_mut(p, y0, y1)?.copy_from_slice(&buf.px);
        Ok(())
    }

    /// `UpscaledCdefFrame` without superres: row band `b` of every plane of `CdefFrame`, cut
    /// to the exact size.
    fn copy_band(&self, fs: &FrameState, b: usize) -> Result<(), Error> {
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let mut bufs = Vec::with_capacity(fs.num_planes);
        {
            let cdef = read_lock(&self.cdef);
            for (plane, p) in cdef.iter().enumerate().take(fs.num_planes) {
                let (w, ht) = exact_size(fs, plane);
                let (_, sy) = subsampling(fs, plane);
                let (y0, y1) = band_rows(b, bands, sy, ht);
                let mut buf = WorkPlane::new(w, y1 - y0, self.ledger)?;
                for (y, row) in (y0..y1).zip(buf.px.chunks_exact_mut(w.max(1))) {
                    row.copy_from_slice(row_of(p, y, w)?);
                }
                bufs.push((buf, y0, y1));
            }
        }
        let mut out = write_lock(&self.up_cdef);
        for (plane, (buf, y0, y1)) in bufs.iter().enumerate() {
            let p = out.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
            rows_of_mut(p, *y0, *y1)?.copy_from_slice(&buf.px);
        }
        Ok(())
    }

    /// The frame state after the tile stage.
    fn state(&self) -> Result<&FrameState, Error> {
        self.fs
            .get()
            .ok_or(Error::Invalid("frame filters before the tiles"))
    }

    /// Column strips of the horizontal edges: one per thread, whole 16-unit columns each.
    fn strips(&self, fs: &FrameState) -> usize {
        self.threads.min(fs.mi_cols.div_ceil(16)).max(1)
    }

    fn tile(&self, i: usize) -> Result<(), Error> {
        let (h, seq) = (self.h, self.seq);
        let t = self
            .tiles
            .get(i)
            .ok_or(Error::Invalid("tile index out of range"))?;
        let tile_cols = h.tile_info.tile_cols.max(1);
        let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
        let mut local = FrameState::new_tile(h, seq, self.ledger, row, col)?;
        decode_tile(
            &mut local,
            &self.frame_cdf,
            self.data,
            t,
            0,
            0,
            h,
            seq,
            self.cancel,
            &mut |_, _| Ok(()),
        )?;
        let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
        if let Some(fs) = lock(&self.tiles_fs).as_mut() {
            fs.merge_tile(&local, sb_size);
        }
        Ok(())
    }

    /// Vertical edges of one row band: each edge filters samples of its own rows only.
    fn lf_rows(&self, fs: &FrameState, i: usize) -> Result<(), Error> {
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let (plane, b) = (i / bands, i % bands);
        if plane >= fs.num_planes || !lf_plane_on(self.h, plane) {
            return Ok(());
        }
        let (_, sy) = subsampling(fs, plane);
        let (mut buf, y0, y1) = {
            let curr = read_lock(&self.curr);
            let p = curr.get(plane).ok_or(Error::Invalid("plane missing"))?;
            let (y0, y1) = band_rows(b, bands, sy, p.height);
            let mut buf = WorkPlane::new(p.stride, y1 - y0, self.ledger)?;
            buf.px.copy_from_slice(rows_of(p, y0, y1)?);
            (buf, y0, y1)
        };
        let rows = 16 * b..(16 * b + 16).min(fs.mi_rows);
        let mut v = View {
            stride: buf.stride,
            px: buf.px.as_mut_slice(),
            x0: 0,
            y0,
        };
        loop_filter_rows(
            fs,
            self.h,
            plane,
            0,
            rows,
            0..fs.mi_cols,
            &mut v,
            self.cancel,
        )?;
        let mut curr = write_lock(&self.curr);
        let p = curr.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        rows_of_mut(p, y0, y1)?.copy_from_slice(&buf.px);
        Ok(())
    }

    /// Horizontal edges of one column strip: each edge filters samples of its own columns only.
    fn lf_cols(&self, i: usize) -> Result<(), Error> {
        let fs = self.state()?;
        let strips = self.strips(fs);
        let (plane, s) = (i / strips, i % strips);
        if plane >= fs.num_planes || !lf_plane_on(self.h, plane) {
            return Ok(());
        }
        let (sx, _) = subsampling(fs, plane);
        // Strip `s` has the mode info columns of units `units * s / strips` up to the next
        // strip's, 16 columns a unit; the last strip runs to the plane end.
        let units = fs.mi_cols.div_ceil(16);
        let c0 = 16 * (s * units / strips);
        let c1 = (16 * ((s + 1) * units / strips)).min(fs.mi_cols);
        if c0 >= c1 {
            return Ok(());
        }
        let (mut buf, x0, x1, height) = {
            let curr = read_lock(&self.curr);
            let p = curr.get(plane).ok_or(Error::Invalid("plane missing"))?;
            let x0 = ((c0 * MI_SIZE) >> sx).min(p.stride);
            let x1 = if s + 1 == strips {
                p.stride
            } else {
                ((c1 * MI_SIZE) >> sx).clamp(x0, p.stride)
            };
            let w = x1 - x0;
            let mut buf = WorkPlane::new(w, p.height, self.ledger)?;
            for (y, row) in buf.px.chunks_exact_mut(w.max(1)).enumerate() {
                let start = y * p.stride;
                row.copy_from_slice(
                    p.px.get(start + x0..start + x1)
                        .ok_or(Error::Invalid("plane smaller than its size"))?,
                );
            }
            (buf, x0, x1, p.height)
        };
        let mut v = View {
            stride: buf.stride,
            px: buf.px.as_mut_slice(),
            x0,
            y0: 0,
        };
        loop_filter_rows(
            fs,
            self.h,
            plane,
            1,
            0..fs.mi_rows,
            c0..c1,
            &mut v,
            self.cancel,
        )?;
        let mut curr = write_lock(&self.curr);
        let p = curr.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        let w = x1 - x0;
        for y in 0..height {
            let start = y * p.stride;
            p.px.get_mut(start + x0..start + x1)
                .ok_or(Error::Invalid("plane smaller than its size"))?
                .copy_from_slice(&buf.px[y * w..(y + 1) * w]);
        }
        Ok(())
    }

    /// CDEF of one row band of 8x8 blocks, every plane: reads `CurrFrame` (also outside the
    /// band), writes the band's rows of `CdefFrame`.
    fn cdef_band(&self, fs: &FrameState, b: usize, on: bool) -> Result<(), Error> {
        let (h, seq) = (self.h, self.seq);
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let curr = read_lock(&self.curr);
        // `CdefFrame` has a plane per `CurrFrame` plane, same sizes.
        let num_planes = usize::from(seq.num_planes).min(curr.len());
        let mut bufs = Vec::with_capacity(num_planes);
        for (plane, p) in curr.iter().enumerate().take(num_planes) {
            let (_, sy) = subsampling(fs, plane);
            let (y0, y1) = band_rows(b, bands, sy, p.height);
            // The band starts as the copy of `CurrFrame` that unfiltered blocks keep.
            let mut buf = WorkPlane::new(p.stride, y1 - y0, self.ledger)?;
            buf.px.copy_from_slice(rows_of(p, y0, y1)?);
            bufs.push((buf, y0, y1));
        }
        // With `on` false (CDEF off), the band stays the copy.
        if on {
            let mut views: Vec<View<'_>> = bufs
                .iter_mut()
                .map(|(buf, y0, _)| View {
                    stride: buf.stride,
                    px: buf.px.as_mut_slice(),
                    x0: 0,
                    y0: *y0,
                })
                .collect();
            let rows = 16 * b..(16 * b + 16).min(fs.mi_rows);
            cdef_rows(fs, &curr, seq, h, rows, &mut views, self.cancel)?;
        }
        drop(curr);
        let mut out = write_lock(&self.cdef);
        for (plane, (buf, y0, y1)) in bufs.iter().enumerate() {
            let p = out.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
            rows_of_mut(p, *y0, *y1)?.copy_from_slice(&buf.px);
        }
        Ok(())
    }

    /// Superres of one row band (64 luma rows) of one plane, from `src` into the exact-size
    /// planes `dst`.
    fn upscale_band(
        &self,
        fs: &FrameState,
        src: &RwLock<Vec<WorkPlane>>,
        dst: &RwLock<Vec<WorkPlane>>,
        i: usize,
    ) -> Result<(), Error> {
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let (plane, b) = (i / bands, i % bands);
        let input = read_lock(src);
        // `dst` has `num_planes` planes of `exact_size`.
        if plane >= fs.num_planes.min(input.len()) {
            return Ok(());
        }
        let (w, ht) = exact_size(fs, plane);
        let (_, sy) = subsampling(fs, plane);
        let (y0, y1) = band_rows(b, bands, sy, ht);
        if y0 == y1 {
            return Ok(());
        }
        let mut buf = WorkPlane::new(w, y1 - y0, self.ledger)?;
        let mut v = View {
            stride: w,
            px: buf.px.as_mut_slice(),
            x0: 0,
            y0,
        };
        let (seq, h) = (self.seq, self.h);
        upscale_rows(
            &input[plane],
            plane,
            seq,
            h,
            (w, ht),
            y0..y1,
            &mut v,
            self.cancel,
        )?;
        drop(input);
        let mut out = write_lock(dst);
        let p = out.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        rows_of_mut(p, y0, y1)?.copy_from_slice(&buf.px);
        Ok(())
    }

    /// Loop restoration of one stripe of one plane: reads `UpscaledCdefFrame` and
    /// `UpscaledCurrFrame` (also outside the stripe), writes the stripe's rows of `LrFrame`.
    fn lr_band(&self, fs: &FrameState, i: usize) -> Result<(), Error> {
        check_cancel(self.cancel)?;
        let ns = stripes(fs);
        let (plane, n) = (i / ns, i % ns);
        let cdef = read_lock(&self.up_cdef);
        let curr = if self.h.frame_width != self.h.upscaled_width {
            read_lock(&self.up_curr)
        } else {
            read_lock(&self.curr)
        };
        // `LrFrame` has `num_planes` planes of `exact_size`.
        if plane >= fs.num_planes.min(3).min(cdef.len()).min(curr.len()) {
            return Ok(());
        }
        let (w, ht) = exact_size(fs, plane);
        let (c, u) = (&cdef[plane], &curr[plane]);
        let Some(g) = lr_plane(fs, self.seq, self.h, plane, [c, u], (w, ht)) else {
            return Ok(());
        };
        let Some((y0, y1)) = g.stripe_rows(n) else {
            return Ok(());
        };
        // `LrFrame` starts as `UpscaledCdefFrame` (exact size too).
        let mut buf = WorkPlane::new(w, y1 + 1 - y0, self.ledger)?;
        buf.px.copy_from_slice(rows_of(c, y0, y1 + 1)?);
        let mut v = View {
            stride: w,
            px: buf.px.as_mut_slice(),
            x0: 0,
            y0,
        };
        lr_stripe(fs, self.seq, &g, n, c, u, &mut v);
        drop((cdef, curr));
        let mut out = lock(&self.out);
        let p = out.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        rows_of_mut(p, y0, y1 + 1)?.copy_from_slice(&buf.px);
        Ok(())
    }
}

/// A filter task of a pipelined frame, on row band `b` (64 luma rows) or stripe `n`:
/// `V( plane, b )` and `H( plane, b )` the vertical and horizontal loop filter edges, `C( b )`
/// CDEF (every plane), `U( b )` superres (or the exact-size copy), `L( plane, n )` loop
/// restoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    V(usize, usize),
    H(usize, usize),
    C(usize),
    U(usize),
    L(usize, usize),
}

/// The tasks of one kind (and plane) of a pipelined frame, started in index order.
#[derive(Debug, Default)]
struct Lane {
    /// Tasks started.
    started: usize,
    /// Tasks done from 0 on, without a gap.
    prefix: usize,
    done: Vec<bool>,
}

impl Lane {
    fn new(n: usize) -> Result<Lane, Error> {
        let mut done = Vec::new();
        done.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        done.resize(n, false);
        Ok(Lane {
            started: 0,
            prefix: 0,
            done,
        })
    }

    /// Starts the next task when `ready` (and one is left); returns its index.
    fn take(&mut self, ready: bool) -> Option<usize> {
        if !ready || self.started >= self.done.len() {
            return None;
        }
        self.started += 1;
        Some(self.started - 1)
    }

    fn finish(&mut self, i: usize) {
        if let Some(d) = self.done.get_mut(i) {
            *d = true;
        }
        while self.done.get(self.prefix) == Some(&true) {
            self.prefix += 1;
        }
    }

    fn complete(&self) -> bool {
        self.prefix == self.done.len()
    }
}

/// The progress of a pipelined frame (docs/threading.md, "Row pipelining"): the bands the tile
/// decode has handed over, and the filter tasks. A task starts when what it reads is final:
///
/// - `V( p, b )`: band `b` handed over.
/// - `H( p, b )`: `V( p, b )` and `H( p, b - 1 )` done (the edges of a plane go down in order).
/// - `C( b )`: `H( p, b + 1 )` done for every plane (CDEF reads 2 rows below the band, which
///   the edges at the top of the next band write); without the loop filter, band `b + 1`
///   handed over.
/// - `U( b )`: `C( b )` done.
/// - `L( p, n )`: `U( 0 ..= n )` done (stripe `n` and the 2 rows below it lie in band `n`).
#[derive(Debug, Default)]
struct Progress {
    /// Bands handed over by the tile decode.
    ready: usize,
    /// The tile decode is over and every band handed over.
    decoded: bool,
    /// A thread failed: no task starts any more.
    failed: bool,
    /// Tasks running.
    running: usize,
    /// CDEF filters (otherwise `C` only copies).
    cdef: bool,
    v: Vec<Lane>,
    h: Vec<Lane>,
    c: Lane,
    u: Lane,
    l: Vec<Lane>,
}

impl Progress {
    /// The tasks of the frame of state `fs`: loop filter lanes with `lf`, loop restoration
    /// lanes with `lr`.
    fn new(fs: &FrameState, lf: bool, lr: bool, cdef: bool) -> Result<Progress, Error> {
        let bands = fs.mi_rows.div_ceil(16).max(1);
        let lanes = |planes: usize, n: usize| -> Result<Vec<Lane>, Error> {
            (0..planes).map(|_| Lane::new(n)).collect()
        };
        let lf_planes = if lf { fs.num_planes } else { 0 };
        Ok(Progress {
            cdef,
            v: lanes(lf_planes, bands)?,
            h: lanes(lf_planes, bands)?,
            c: Lane::new(bands)?,
            u: Lane::new(bands)?,
            l: lanes(if lr { fs.num_planes.min(3) } else { 0 }, stripes(fs))?,
            ..Progress::default()
        })
    }

    /// Starts a task whose inputs are final, the last stages first.
    fn next(&mut self) -> Option<Task> {
        let bands = self.c.done.len();
        let up = self.u.prefix;
        for (p, l) in self.l.iter_mut().enumerate() {
            let ok = up >= (l.started + 1).min(bands);
            if let Some(n) = l.take(ok) {
                return Some(Task::L(p, n));
            }
        }
        let ok = self.c.done.get(self.u.started) == Some(&true);
        if let Some(b) = self.u.take(ok) {
            return Some(Task::U(b));
        }
        let need = (self.c.started + 2).min(bands);
        let ok = if self.h.is_empty() {
            self.ready >= need
        } else {
            self.h.iter().all(|l| l.prefix >= need)
        };
        if let Some(b) = self.c.take(ok) {
            return Some(Task::C(b));
        }
        for (p, (h, v)) in self.h.iter_mut().zip(&self.v).enumerate() {
            let ok = v.prefix > h.started && h.prefix == h.started;
            if let Some(b) = h.take(ok) {
                return Some(Task::H(p, b));
            }
        }
        for (p, v) in self.v.iter_mut().enumerate() {
            let ok = v.started < self.ready;
            if let Some(b) = v.take(ok) {
                return Some(Task::V(p, b));
            }
        }
        None
    }

    /// Marks task `t` done.
    fn finish(&mut self, t: Task) {
        self.running = self.running.saturating_sub(1);
        let (lane, i) = match t {
            Task::V(p, b) => (self.v.get_mut(p), b),
            Task::H(p, b) => (self.h.get_mut(p), b),
            Task::C(b) => (Some(&mut self.c), b),
            Task::U(b) => (Some(&mut self.u), b),
            Task::L(p, n) => (self.l.get_mut(p), n),
        };
        if let Some(lane) = lane {
            lane.finish(i);
        }
    }

    /// Every task done.
    fn complete(&self) -> bool {
        self.v
            .iter()
            .chain(&self.h)
            .chain(&self.l)
            .chain([&self.c, &self.u])
            .all(Lane::complete)
    }
}

/// Marks a pipelined frame failed when dropped before [`Abort::disarm`] (its thread returned
/// an error or panicked), so no thread waits for a band or a task that will not come.
struct Abort<'w, 'a> {
    w: &'w Wrap<'a>,
    done: bool,
}

impl Abort<'_, '_> {
    fn disarm(mut self) {
        self.done = true;
    }
}

impl Drop for Abort<'_, '_> {
    fn drop(&mut self) {
        if !self.done {
            lock(&self.w.progress).failed = true;
        }
        self.w.moved.notify_all();
    }
}

/// The write lock of `l`, or `None` while another thread holds it.
fn try_write<T>(l: &RwLock<T>) -> Option<RwLockWriteGuard<'_, T>> {
    match l.try_write() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// Loop restoration stripes of a frame's planes, at most (64 luma rows each, offset by 8).
fn stripes(fs: &FrameState) -> usize {
    fs.frame_height.div_ceil(64) + 1
}

/// `(subsampling_x, subsampling_y)` of plane `plane`.
fn subsampling(fs: &FrameState, plane: usize) -> (usize, usize) {
    if plane == 0 {
        (0, 0)
    } else {
        (fs.subsampling_x, fs.subsampling_y)
    }
}

/// The plane rows `y0..y1` of row band `b` of `bands` (16 mode info rows, 64 luma rows, each):
/// the last band runs to the plane end, `height` rows.
fn band_rows(b: usize, bands: usize, sy: usize, height: usize) -> (usize, usize) {
    let y0 = ((64 * b) >> sy).min(height);
    let y1 = if b + 1 >= bands {
        height
    } else {
        ((64 * (b + 1)) >> sy).min(height)
    };
    (y0, y1.max(y0))
}

/// The samples of rows `y0..y1` of `p`.
fn rows_of(p: &WorkPlane, y0: usize, y1: usize) -> Result<&[u16], Error> {
    p.px.get(y0 * p.stride..y1 * p.stride)
        .ok_or(Error::Invalid("plane smaller than its size"))
}

/// The samples of rows `y0..y1` of `p`, to write.
fn rows_of_mut(p: &mut WorkPlane, y0: usize, y1: usize) -> Result<&mut [u16], Error> {
    p.px.get_mut(y0 * p.stride..y1 * p.stride)
        .ok_or(Error::Invalid("plane smaller than its size"))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn read_lock<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

fn write_lock<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}

/// The size of plane `plane` of `LrFrame`: `UpscaledWidth` x `FrameHeight`, chroma
/// `(size + ss) >> ss`.
fn exact_size(fs: &FrameState, plane: usize) -> (usize, usize) {
    let (sx, sy) = subsampling(fs, plane);
    ((fs.upscaled_width + sx) >> sx, (fs.frame_height + sy) >> sy)
}

/// Output planes of exactly `UpscaledWidth` x `FrameHeight` (chroma `(size + ss) >> ss`), each a
/// copy of the top left of the matching plane of `src` when given.
fn exact_planes(fs: &FrameState, src: Option<&[WorkPlane]>) -> Result<Vec<WorkPlane>, Error> {
    let mut out = fs.ledger.planes(fs.num_planes)?;
    for i in 0..fs.num_planes {
        let (w, ht) = exact_size(fs, i);
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

/// `Err(Cancelled)` once the flag is set. The wrapup filters and film grain check it per row band.
pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<(), Error> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    Ok(())
}

/// The frame wrapup of spec 7.4 after the last tile: deblocking (7.14), CDEF (7.15), superres
/// (7.16) on the CDEF output and, for loop restoration, on the deblocked frame, then loop
/// restoration (7.17). Returns `LrFrame`.
fn wrapup(
    fs: &mut FrameState,
    seq: &SequenceHeader,
    h: &FrameHeader,
    cancel: &AtomicBool,
) -> Result<Vec<WorkPlane>, Error> {
    loop_filter_frame(fs, seq, h, cancel)?;
    let cdef = cdef_alloc(fs, seq, h, cancel)?;
    let (up_cdef, up_curr) = if h.frame_width != h.upscaled_width {
        let mut up_cdef = exact_planes(fs, None)?;
        upscale(&cdef, fs, seq, h, &mut up_cdef, cancel)?;
        fs.ledger.put_planes(cdef);
        let up_curr = if h.uses_lr {
            let mut p = exact_planes(fs, None)?;
            upscale(&fs.curr_frame, fs, seq, h, &mut p, cancel)?;
            Some(p)
        } else {
            None
        };
        (up_cdef, up_curr)
    } else {
        let p = exact_planes(fs, Some(cdef.as_slice()))?;
        fs.ledger.put_planes(cdef);
        (p, None)
    };
    if !h.uses_lr {
        return Ok(up_cdef);
    }
    let mut out = exact_planes(fs, Some(up_cdef.as_slice()))?;
    match &up_curr {
        Some(c) => lr_frame(fs, seq, h, &up_cdef, c, &mut out, cancel)?,
        None => lr_frame(fs, seq, h, &up_cdef, &fs.curr_frame, &mut out, cancel)?,
    }
    fs.ledger.put_planes(up_cdef);
    if let Some(c) = up_curr {
        fs.ledger.put_planes(c);
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

/// The payload of a reduced still picture sequence header (64x32, 8-bit 4:2:0, CDEF and loop
/// restoration on), for [`wrapup_frame`] (`seq::tests::still_sequence_header` writes the same).
#[cfg(any(test, feature = "fuzzing"))]
const STILL_SEQUENCE_HEADER: [u8; 6] = [0x19, 0x1d, 0x8f, 0xcf, 0xb6, 0x01];

/// A frame whose tiles are done, drawn from `data`, through [`decode_frame`] with `threads`
/// threads and `max_alloc`; the ledger is returned to read its peak. `data[ 0 ]` picks the
/// format (4:2:0, 4:2:2, 4:4:4, mono), `data[ 1 ]` the bit depth (8, 10, 12), `data[ 2 ]`
/// superres off or on; a hash of all of `data` seeds the rest: the size (up to 200x200), loop
/// filter, CDEF and loop restoration parameters, the mode info and the `CurrFrame` samples
/// (8x8 blocks of a level plus noise, so the filters have edges to work on). The mode info is
/// random, not the result of a decode: the wrapup must not rely on it being consistent.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn wrapup_frame(
    data: &[u8],
    threads: usize,
    max_alloc: usize,
) -> (Result<FrameState, Error>, Arc<Ledger>) {
    let ledger = Arc::new(Ledger::default());
    let _ = ledger.reset(max_alloc, 0);
    let sel = |i: usize| u64::from(data.get(i).copied().unwrap_or(0));
    let mut x = data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    }) | 1;
    // A value in 0..n.
    let mut next = move |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let Ok(mut seq) = SequenceHeader::parse(&STILL_SEQUENCE_HEADER, 0) else {
        return (Err(Error::Invalid("sequence header")), ledger);
    };
    seq.bit_depth = [8, 10, 12][(sel(1) % 3) as usize];
    (
        seq.mono_chrome,
        seq.num_planes,
        seq.subsampling_x,
        seq.subsampling_y,
    ) = match sel(0) % 4 {
        0 => (false, 3, 1, 1),
        1 => (false, 3, 1, 0),
        2 => (false, 3, 0, 0),
        _ => (true, 1, 1, 1),
    };
    seq.enable_superres = true;
    seq.enable_cdef = next(8) != 0;
    let upscaled = 16 + next(185) as u32;
    let height = 8 + next(193) as u32;
    let denom = if sel(2) % 2 == 1 {
        9 + next(8) as u32
    } else {
        8
    };
    let width = (upscaled * 8 + denom / 2) / denom;
    let mut h = FrameHeader {
        frame_width: width,
        frame_height: height,
        upscaled_width: upscaled,
        use_superres: denom != 8,
        superres_denom: denom,
        mi_cols: 2 * ((width + 7) >> 3),
        mi_rows: 2 * ((height + 7) >> 3),
        loop_filter_sharpness: next(8) as u8,
        loop_filter_delta_enabled: next(2) == 1,
        delta_lf_multi: next(2) == 1,
        cdef_damping: 3 + next(4) as u8,
        cdef_bits: 3,
        ..FrameHeader::default()
    };
    for l in h.loop_filter_level.iter_mut() {
        *l = next(64) as u8;
    }
    for d in h.loop_filter_ref_deltas.iter_mut() {
        *d = next(127) as i8 - 63;
    }
    for i in 0..8 {
        h.cdef_y_pri_strength[i] = next(16) as u8;
        h.cdef_y_sec_strength[i] = [0, 1, 2, 4][next(4) as usize];
        h.cdef_uv_pri_strength[i] = next(16) as u8;
        h.cdef_uv_sec_strength[i] = [0, 1, 2, 4][next(4) as usize];
    }
    let lr_size = [64u16, 128, 256][next(3) as usize];
    let uv_shift = if seq.subsampling_x == 1 && seq.subsampling_y == 1 {
        next(2) as u16
    } else {
        0
    };
    for plane in 0..usize::from(seq.num_planes) {
        h.frame_restoration_type[plane] = next(4) as u8;
        h.loop_restoration_size[plane] = if plane == 0 {
            lr_size
        } else {
            lr_size >> uv_shift
        };
    }
    h.uses_lr = h.frame_restoration_type.iter().any(|&t| t != RESTORE_NONE);
    let mut fs = match FrameState::new(&h, &seq, &ledger) {
        Ok(fs) => fs,
        Err(e) => return (Err(e), ledger),
    };
    for i in 0..fs.mi_rows * fs.mi_cols {
        fs.mi_sizes[i] = next(BLOCK_SIZES as u64) as u8;
        fs.skips[i] = next(2) == 1;
        fs.cdef_idx[i] = next(9) as i8 - 1;
        for d in fs.delta_lfs[i].iter_mut() {
            *d = next(33) as i8 - 16;
        }
        for plane in 0..fs.num_planes {
            fs.loopfilter_tx_sizes[plane][i] = next(TX_SIZES_ALL as u64) as u8;
        }
    }
    for plane in 0..3 {
        for u in 0..fs.lr_type[plane].len() {
            fs.lr_type[plane][u] = next(3) as u8;
            for pass in 0..2 {
                fs.lr_wiener[plane][u][pass] =
                    [next(16) as i8 - 5, next(32) as i8 - 23, next(64) as i8 - 17];
            }
            fs.lr_sgr_set[plane][u] = next(16) as u8;
            fs.lr_sgr_xqd[plane][u] = [next(128) as i8 - 96, next(128) as i8 - 32];
        }
    }
    let max = (1u64 << seq.bit_depth) - 1;
    let noise = 4u64 << (seq.bit_depth - 8);
    for p in fs.curr_frame.iter_mut() {
        let mut levels = vec![0u64; p.width.div_ceil(8)];
        for y in 0..p.height {
            if y.is_multiple_of(8) {
                levels.iter_mut().for_each(|l| *l = next(max + 1));
            }
            for (x, v) in p.px[y * p.stride..y * p.stride + p.width]
                .iter_mut()
                .enumerate()
            {
                *v = (levels[x / 8] + next(noise)).min(max) as u16;
            }
        }
    }
    let r = decode_frame(
        fs,
        &ledger,
        threads,
        &h,
        &seq,
        &[],
        &[],
        &AtomicBool::new(false),
    );
    (r, ledger)
}

/// [`wrapup_frame`] of `data` with each of `threads` against one thread: the same error, or
/// the same `LrFrame` and loop filtered `CurrFrame`.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn check_threads(data: &[u8], threads: &[usize]) {
    let (serial, _) = wrapup_frame(data, 1, 64 << 20);
    for &t in threads {
        let (par, _) = wrapup_frame(data, t, 64 << 20);
        match (&serial, &par) {
            (Ok(a), Ok(b)) => {
                for (p, q) in [(&a.lr_frame, &b.lr_frame), (&a.curr_frame, &b.curr_frame)] {
                    assert_eq!(p.len(), q.len());
                    for (pp, qq) in p.iter().zip(q.iter()) {
                        assert!(pp.px == qq.px, "{t} threads differ from 1");
                    }
                }
            }
            _ => assert_eq!(serial.as_ref().err(), par.as_ref().err()),
        }
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
                d.take_frames(); // hands out (and releases) the planes of a half-built unit
                assert_eq!(d.ledger.left(), before);
            }
        }
    }

    #[test]
    fn reuse_and_recycle_keep_the_output_and_the_ledger_peak() {
        let data: Vec<u8> = still_obus().concat();
        let one = Limits {
            max_threads: 1,
            ..Limits::default()
        };
        let want = Decoder::new(Options::default(), one).decode_temporal_unit(
            &data,
            Format::Section5,
            &cancel(),
        );
        for max_alloc in [1024, 40_000, 100_000, 150_000, 200_000, 400_000, 1 << 20] {
            let mut d = Decoder::new(Options::default(), Limits { max_alloc, ..one });
            let mut ok = None;
            for _ in 0..3 {
                let r = d.decode_temporal_unit(&data, Format::Section5, &cancel());
                assert!(d.ledger.peak() <= max_alloc as u64);
                // Reuse never makes the same unit fail where it fitted, nor the reverse.
                assert_eq!(*ok.get_or_insert(r.is_ok()), r.is_ok());
                match r {
                    Ok(frames) => {
                        assert_eq!(Ok(&frames), want.as_ref());
                        d.recycle(frames);
                    }
                    Err(_) => {
                        d.take_frames();
                    }
                }
            }
        }
        // The decoder that reuses nothing (the fuzz oracle) gives the same frames.
        let mut fresh = Decoder::new(Options::default(), one);
        fresh.reuse = false;
        for _ in 0..2 {
            let r = fresh.decode_temporal_unit(&data, Format::Section5, &cancel());
            assert_eq!(r.as_ref(), want.as_ref());
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

    // On a shared ledger the kept metadata is booked in it, so many decoders (or many frames of
    // one unit) cannot keep `max_metadata` each on top of `max_alloc`.
    #[test]
    fn shared_ledger_books_the_metadata() {
        let m = obu(OBU_METADATA, &[1, 0x03, 0xe8, 0x01, 0x90, 0x80]);
        let shared = |extra: usize| {
            let ledger = Arc::new(Ledger::default());
            ledger.reset(crate::ledger::SMALL + extra, 0).unwrap();
            ledger
        };
        let ledger = shared(4);
        let mut d = Decoder::with_ledger(Options::default(), Limits::default(), ledger.clone());
        assert_eq!(
            d.decode_temporal_unit(&m, Format::Section5, &cancel()),
            Err(Error::AllocLimit)
        );
        let ledger = shared(1 << 20);
        let mut d = Decoder::with_ledger(Options::default(), Limits::default(), ledger.clone());
        d.decode_temporal_unit(&m, Format::Section5, &cancel())
            .unwrap();
        assert!(ledger.left() < 1 << 20);
        // The next unit drops the kept metadata and gives its bytes back.
        d.decode_temporal_unit(&[], Format::Section5, &cancel())
            .unwrap();
        assert_eq!(ledger.left(), 1 << 20);
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

    /// A 4:2:0 frame header with every in-loop filter on (loop filter, CDEF, loop restoration)
    /// and `cols` x `rows` tiles of 64x64 (`size` the upscaled width and the height), and the
    /// bytes of its tiles: pseudo-random data, `bytes` per tile, which the symbol decoder reads
    /// as some valid syntax (or a syntax error, the same for every thread count).
    fn filtered_frame(
        seed: u64,
        (width, height): (u32, u32),
        superres_denom: u32,
        (cols, rows): (usize, usize),
        bytes: usize,
    ) -> (FrameHeader, Vec<u8>, Vec<TileRange>) {
        let frame_width = (width * 8 + superres_denom / 2) / superres_denom;
        let (mi_cols, mi_rows) = (2 * ((frame_width + 7) >> 3), 2 * ((height + 7) >> 3));
        let starts = |n: usize, end: u32| {
            let v: Vec<u32> = (0..n as u32).map(|i| 16 * i).chain([end]).collect();
            crate::header::Starts::of(&v)
        };
        let mut h = FrameHeader {
            frame_width,
            frame_height: height,
            upscaled_width: width,
            use_superres: superres_denom != 8,
            superres_denom,
            mi_cols,
            mi_rows,
            base_q_idx: 60,
            tx_mode: TX_MODE_SELECT,
            loop_filter_level: [24, 18, 12, 9],
            loop_filter_sharpness: 2,
            cdef_damping: 4,
            cdef_bits: 2,
            frame_restoration_type: [RESTORE_WIENER, RESTORE_SGRPROJ, RESTORE_SWITCHABLE],
            loop_restoration_size: [64, 32, 32],
            uses_lr: true,
            tile_info: crate::header::TileInfo {
                tile_cols: cols,
                tile_rows: rows,
                mi_col_starts: starts(cols, mi_cols),
                mi_row_starts: starts(rows, mi_rows),
                ..Default::default()
            },
            ..FrameHeader::default()
        };
        for i in 0..4 {
            h.cdef_y_pri_strength[i] = [0, 3, 7, 15][i];
            h.cdef_y_sec_strength[i] = [1, 0, 2, 4][i];
            h.cdef_uv_pri_strength[i] = [2, 0, 5, 9][i];
            h.cdef_uv_sec_strength[i] = [0, 2, 1, 4][i];
        }
        let mut x = seed | 1;
        let (mut data, mut tiles) = (Vec::new(), Vec::new());
        for tile_num in 0..cols * rows {
            let start = data.len();
            for _ in 0..bytes {
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

    /// 192x128 in 3 x 2 tiles of 64x64, every filter on.
    fn six_tiles(seed: u64) -> (FrameHeader, Vec<u8>, Vec<TileRange>) {
        filtered_frame(seed, (192, 128), 8, (3, 2), 2000)
    }

    /// 200x136 (3 row bands) in one tile, every filter on; superres (denominator 12) for odd
    /// seeds.
    fn one_tile(seed: u64) -> (FrameHeader, Vec<u8>, Vec<TileRange>) {
        let denom = if seed % 2 == 1 { 12 } else { 8 };
        filtered_frame(seed, (200, 136), denom, (1, 1), 16_000)
    }

    /// The tiles and the wrapup of `six_tiles( seed )` with `threads` threads and `max_alloc`; the ledger is
    /// returned to read its peak.
    fn run_tiles(
        seed: u64,
        threads: usize,
        max_alloc: usize,
    ) -> (Result<FrameState, Error>, Arc<Ledger>) {
        run_frame(six_tiles(seed), false, threads, max_alloc, &cancel())
    }

    /// The pipelined decode of `one_tile( seed )` (128x128 superblocks for seeds 2 and 3 mod 4)
    /// with `threads` threads, `max_alloc` and `stop`.
    fn run_one_tile(
        seed: u64,
        threads: usize,
        max_alloc: usize,
        stop: &AtomicBool,
    ) -> (Result<FrameState, Error>, Arc<Ledger>) {
        run_frame(one_tile(seed), seed % 4 >= 2, threads, max_alloc, stop)
    }

    /// [`decode_frame`] of a frame from [`filtered_frame`].
    fn run_frame(
        (h, data, tiles): (FrameHeader, Vec<u8>, Vec<TileRange>),
        sb128: bool,
        threads: usize,
        max_alloc: usize,
        stop: &AtomicBool,
    ) -> (Result<FrameState, Error>, Arc<Ledger>) {
        let mut seq = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        seq.use_128x128_superblock = sb128;
        let ledger = Arc::new(Ledger::default());
        ledger.reset(max_alloc, 0).unwrap();
        let r = FrameState::new(&h, &seq, &ledger)
            .and_then(|fs| decode_frame(fs, &ledger, threads, &h, &seq, &data, &tiles, stop));
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
        for (x, y) in [(&a.curr_frame, &b.curr_frame), (&a.lr_frame, &b.lr_frame)] {
            assert_eq!(x.len(), y.len());
            for (p, q) in x.iter().zip(y) {
                assert_eq!(p.px, q.px);
            }
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

    /// One tile with every filter on, superres and 128x128 superblocks by seed: the pipelined
    /// decode with 2, 3 and 8 threads gives the state of 1 thread.
    #[test]
    fn pipelined_tile_decodes_the_same_for_every_thread_count() {
        let mut decoded = 0;
        for seed in 1..=8 {
            let (serial, _) = run_one_tile(seed, 1, 64 << 20, &cancel());
            for threads in [2, 3, 8] {
                let (par, ledger) = run_one_tile(seed, threads, 64 << 20, &cancel());
                match (&serial, &par) {
                    (Ok(a), Ok(b)) => assert_same_state(a, b),
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    _ => panic!("seed {seed}: serial and {threads} threads differ in Ok or Err"),
                }
                assert!(ledger.peak() <= 64 << 20);
                drop(par);
                assert_eq!(ledger.left(), (64 << 20) - crate::ledger::SMALL);
            }
            decoded += usize::from(serial.is_ok());
        }
        assert!(decoded > 0);
    }

    #[test]
    fn pipelined_tile_stays_within_max_alloc() {
        for max_alloc in [1024, 100 << 10, 300 << 10, 600 << 10, 1 << 20, 4 << 20] {
            for threads in [1, 2, 3, 8] {
                let (r, ledger) = run_one_tile(1, threads, max_alloc, &cancel());
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

    /// The task order of a pipelined frame: whatever order the bands come in and the tasks
    /// finish, every task runs once and only after the tasks it reads from.
    #[test]
    fn pipeline_tasks_respect_their_inputs() {
        let seq = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        let (h, _, _) = one_tile(0);
        let ledger = Arc::new(Ledger::default());
        ledger.reset(64 << 20, 0).unwrap();
        let fs = FrameState::new(&h, &seq, &ledger).unwrap();
        let bands = fs.mi_rows.div_ceil(16);
        for (lf, lr) in [(true, true), (false, true), (true, false)] {
            for lifo in [false, true] {
                let mut p = Progress::new(&fs, lf, lr, true).unwrap();
                let mut running: Vec<Task> = Vec::new();
                let mut done: Vec<Task> = Vec::new();
                while !p.complete() {
                    while let Some(t) = p.next() {
                        let last = bands - 1;
                        let ok = match t {
                            Task::V(_, b) => b < p.ready,
                            Task::H(q, b) => {
                                done.contains(&Task::V(q, b))
                                    && (b == 0 || done.contains(&Task::H(q, b - 1)))
                            }
                            Task::C(b) if lf => (0..fs.num_planes)
                                .all(|q| done.contains(&Task::H(q, (b + 1).min(last)))),
                            Task::C(b) => p.ready > (b + 1).min(last),
                            Task::U(b) => done.contains(&Task::C(b)),
                            Task::L(_, n) => (0..=n.min(last)).all(|b| done.contains(&Task::U(b))),
                        };
                        assert!(ok, "{t:?} started before its inputs");
                        running.push(t);
                    }
                    // Finish one task (the newest or the oldest), or hand over a band.
                    let t = if lifo {
                        running.pop()
                    } else {
                        (!running.is_empty()).then(|| running.remove(0))
                    };
                    match t {
                        Some(t) => {
                            p.finish(t);
                            done.push(t);
                        }
                        None => {
                            assert!(p.ready < bands, "stalled with every band in");
                            p.ready += 1;
                        }
                    }
                }
                let lanes = if lf { 2 * fs.num_planes * bands } else { 0 };
                let lr_tasks = if lr { fs.num_planes * stripes(&fs) } else { 0 };
                assert_eq!(done.len(), lanes + 2 * bands + lr_tasks);
            }
        }
    }

    #[test]
    fn still_sequence_header_constant_matches_the_writer() {
        assert_eq!(STILL_SEQUENCE_HEADER.as_slice(), still_sequence_header());
    }

    /// Every format (4:2:0, 4:2:2, 4:4:4, mono) and bit depth, superres off and on, with the
    /// loop filter, CDEF and loop restoration drawn per seed: 2, 3 and 8 threads give the
    /// samples of 1 thread.
    #[test]
    fn wrapup_is_the_same_for_every_thread_count() {
        let mut parallel = 0;
        for format in 0..4u8 {
            for depth in 0..3u8 {
                for superres in 0..2u8 {
                    for seed in 0..2u8 {
                        let data = [format, depth, superres, seed];
                        check_threads(&data, &[2, 3, 8]);
                        // Most frames have more than one band: the threaded path did run.
                        let (r, _) = wrapup_frame(&data, 1, 64 << 20);
                        let fs = r.unwrap();
                        parallel += usize::from(team_size(8, &fs, 0) > 1);
                    }
                }
            }
        }
        assert!(parallel > 24);
    }

    #[test]
    fn wrapup_threads_stay_within_max_alloc() {
        // 4:4:4 10-bit with superres: the most buffers.
        let data = [2u8, 1, 1, 5];
        let mut fitted = 0;
        for max_alloc in [1024, 64 << 10, 256 << 10, 512 << 10, 1 << 20, 4 << 20] {
            for threads in [1, 2, 3, 8] {
                let (r, ledger) = wrapup_frame(&data, threads, max_alloc);
                assert!(ledger.peak() <= max_alloc as u64);
                assert!(r.is_ok() || r.as_ref().err() == Some(&Error::AllocLimit));
                fitted += usize::from(r.is_ok());
                drop(r);
                assert_eq!(
                    ledger.left(),
                    max_alloc.saturating_sub(crate::ledger::SMALL)
                );
            }
        }
        assert!(fitted >= 4);
    }

    #[test]
    fn team_size_stays_within_max_threads() {
        let (r, _) = wrapup_frame(&[0, 0, 0, 0], 1, 64 << 20);
        let fs = r.unwrap();
        for max_threads in [0, 1, 2, 3, 8, 64] {
            for tiles in [0, 1, 6, 100] {
                let n = team_size(max_threads, &fs, tiles);
                assert!(n >= 1 && n <= max_threads.max(1));
            }
        }
    }

    #[test]
    fn wrapup_threads_honour_the_cancel_flag() {
        let seq = SequenceHeader::parse(&still_sequence_header(), 0).unwrap();
        let (h, data, tiles) = six_tiles(3);
        let ledger = Arc::new(Ledger::default());
        ledger.reset(64 << 20, 0).unwrap();
        for threads in [1, 3] {
            let fs = FrameState::new(&h, &seq, &ledger).unwrap();
            let stop = AtomicBool::new(true);
            let r = decode_frame(fs, &ledger, threads, &h, &seq, &data, &tiles, &stop);
            assert_eq!(r.err(), Some(Error::Cancelled));
            // One tile: the pipelined decode.
            let (r, _) = run_one_tile(2, threads, 64 << 20, &stop);
            assert_eq!(r.err(), Some(Error::Cancelled));
        }
    }
}
