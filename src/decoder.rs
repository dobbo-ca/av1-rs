//! The decoder state machine: OBU dispatch, the sequence and frame header handling, tile group
//! collection, the reference slots (spec 7.20, 7.21), frame output (7.18) and the limit checks.
//!
//! The pixel work of a frame is [`Decoder::decode_frame_tiles`]: the tiles, then the in-loop
//! filters. [`probe`](crate::probe) runs the same path without it.
#![allow(dead_code)]

use crate::bits::BitReader;
use crate::cdef::{cdef_alloc, cdef_on, cdef_planes, cdef_rows};
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::error::Error;
use crate::frame::{FrameState, Plane as WorkPlane, View};
use crate::grain::{GrainInput, apply_film_grain};
use crate::header::{FrameHeader, RefInfo, RefState, TileRange, parse_tile_group};
use crate::ledger::Ledger;
use crate::lf::{lf_on, lf_plane_on, loop_filter_frame, loop_filter_rows};
use crate::limits::Limits;
use crate::lr::{LrScratch, lr_frame, lr_plane, lr_stripe};
use crate::obu::{Obu, ObuIter, parse_metadata};
use crate::parallel::{Team, run_parallel, with_team};
use crate::seq::SequenceHeader;
use crate::superres::{upscale, upscale_rows};
use crate::tile::TileDecoder;
use crate::{ColorInfo, Format, Frame, FrameInfo, Metadata, Options, Plane, PlaneData};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A frame whose header was read and whose tile groups are still coming.
#[derive(Debug)]
struct Pending {
    header: FrameHeader,
    /// The tiles received so far, in order.
    tiles: Vec<TileRange>,
    /// `TileNum` the next tile group must start at.
    next_tile: usize,
}

/// A frame of a batch ([`Decoder::decode_temporal_units`]) whose header is parsed and whose
/// reference update (the header half) is done; its pixels are decoded at the next flush.
#[derive(Debug)]
struct QueuedFrame {
    header: FrameHeader,
    seq: SequenceHeader,
    /// Index of its temporal unit in the batch; `tiles` point into that unit.
    unit: usize,
    tiles: Vec<TileRange>,
    /// The reference state after this frame's header was parsed and before its update: what a
    /// failed decode leaves behind.
    refs: RefState,
    /// The metadata an output frame takes (empty for a hidden frame).
    metadata: Vec<Metadata>,
    metadata_bytes: usize,
}

/// What one [`Decoder::decode_temporal_units`] call has collected so far.
#[derive(Default)]
struct Batch {
    /// The output frames of the finished units, in order.
    out: Vec<Frame>,
    /// Ledger bytes held by `out` until the call returns.
    held: usize,
    /// The unit whose frames are in `Decoder::ready`.
    commit_unit: usize,
    /// The unit of the frame that failed to decode.
    stop_unit: Option<usize>,
}

/// Frames in flight and threads each for a flush of `queued` frames, with `left` bytes of
/// `max_alloc` free and `est` bytes at most per frame: `w * t <= max_threads`, and `w` is 1 when
/// the budget fits one frame only.
fn plan(max_threads: usize, queued: usize, left: usize, est: usize) -> (usize, usize) {
    let max_threads = max_threads.max(1);
    let w = max_threads.min(queued).min(left / est.max(1)).max(1);
    (w, (max_threads / w).max(1))
}

/// An upper estimate of the working memory of one frame decode: three sets of 16-bit planes
/// (`CurrFrame`, `CdefFrame`, `LrFrame`) of three times the luma area, with a margin.
fn frame_estimate(h: &FrameHeader) -> usize {
    let w = (h.frame_width.max(h.upscaled_width) as usize + 127) & !127;
    let rows = (h.frame_height as usize + 127) & !127;
    w.saturating_mul(rows).saturating_mul(18)
}

/// Most plane lists of recycled frames a decoder keeps.
const SPARE_LISTS: usize = 16;

/// Heap an output frame holds besides its samples, booked with them: two `Frame` slots (a
/// growing `Vec<Frame>` doubles), its list of `planes` entries, and an allocator chunk (32 bytes)
/// for the list and for each sample buffer, so a stream of tiny frames cannot outgrow
/// `max_alloc`.
fn frame_overhead(planes: usize) -> usize {
    const CHUNK: usize = 32;
    2 * size_of::<Frame>() + planes.saturating_mul(size_of::<Plane>() + CHUNK) + CHUNK
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
    /// True while [`Decoder::decode_temporal_units`] runs frames in parallel: a finished frame
    /// is queued in `jobs` instead of decoded.
    batching: bool,
    /// The frames queued since the last flush, in stream order.
    jobs: Vec<QueuedFrame>,
    /// The batch unit being parsed.
    parse_unit: usize,
    /// The frames of the finished units of a batch that failed, for [`Decoder::take_frames`].
    batch_done: Vec<Frame>,
    /// The most frames a flush had decoding at once.
    #[cfg(test)]
    jobs_peak: usize,
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

    /// Books `bytes` of the caller's own buffers on the ledger for the life of the decoder.
    pub(crate) fn book(&self, bytes: usize) -> Result<(), Error> {
        self.ledger.reserve(bytes)
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
            batching: false,
            jobs: Vec::new(),
            parse_unit: 0,
            batch_done: Vec::new(),
            #[cfg(test)]
            jobs_peak: 0,
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
        self.batch_done.clear();
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

    /// Decodes `units`, consecutive temporal units of one stream, and returns their output
    /// frames in order: the frames of one [`Decoder::decode_temporal_unit`] call per unit.
    /// With `Limits::max_threads` 1 that is exactly a loop of those calls.
    ///
    /// With more threads the headers are parsed and the reference slots updated in stream order
    /// on the calling thread, while the pixels of up to `W` frames decode at once, each with
    /// `max_threads / W` threads (`W * threads <= max_threads`, and `W` is lowered while the
    /// frames in flight would not fit in `max_alloc`). A frame never reads another frame's
    /// pixels (every frame is intra with default CDFs), so the output is the same for every
    /// thread count. A `show_existing_frame` and a sequence header that changes the sequence
    /// wait for the frames before them. Frames are decoded in groups of at most `max_threads`.
    ///
    /// The output frames stay charged against `Limits::max_alloc` until the call returns, so
    /// pass a few units at a time to stream a long file.
    ///
    /// # Errors
    /// The error the first failing unit would give in a loop of
    /// [`Decoder::decode_temporal_unit`] calls (the lowest stream position wins, whatever the
    /// timing). The frames output before it, those of the earlier units included, are returned by
    /// [`Decoder::take_frames`], and the reference slots are as that loop leaves them.
    pub fn decode_temporal_units(
        &mut self,
        units: &[&[u8]],
        format: Format,
        cancel: &AtomicBool,
    ) -> Result<Vec<Frame>, Error> {
        self.pending = None;
        self.ready.clear();
        self.release_ready();
        self.clear_metadata();
        self.batch_done.clear();
        let mut b = Batch::default();
        let r = if self.limits.max_threads <= 1 || self.probe_only || self.charge_metadata {
            self.units_in_turn(units, format, cancel, &mut b)
        } else {
            self.batching = true;
            let r = self.units_batched(units, format, cancel, &mut b);
            self.batching = false;
            self.jobs.clear();
            r
        };
        self.ledger.release(b.held);
        match r {
            Ok(()) => Ok(b.out),
            Err(e) => {
                self.batch_done = b.out;
                Err(e)
            }
        }
    }

    /// [`Decoder::decode_temporal_units`] as a loop of [`Decoder::decode_temporal_unit`].
    fn units_in_turn(
        &mut self,
        units: &[&[u8]],
        format: Format,
        cancel: &AtomicBool,
        b: &mut Batch,
    ) -> Result<(), Error> {
        for unit in units {
            match self.decode_temporal_unit(unit, format, cancel) {
                Ok(frames) => self.hold(b, frames)?,
                Err(e) => {
                    let partial = self.select_output();
                    if b.out.try_reserve(partial.len()).is_ok() {
                        b.out.extend(partial);
                    }
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Adds the output frames of a finished unit to the batch and books them on the ledger until
    /// the call returns.
    fn hold(&self, b: &mut Batch, frames: Vec<Frame>) -> Result<(), Error> {
        for f in &frames {
            let n = crate::avif::frame_bytes(f).saturating_add(frame_overhead(f.planes.capacity()));
            self.ledger.reserve(n)?;
            b.held += n;
        }
        b.out
            .try_reserve(frames.len())
            .map_err(|_| Error::OutOfMemory)?;
        b.out.extend(frames);
        Ok(())
    }

    /// Finishes the units before `to`: their frames, all committed by now, get the layer
    /// selection (with the sequence header they were decoded under) and go to the batch.
    fn advance_unit(&mut self, b: &mut Batch, to: usize) -> Result<(), Error> {
        while b.commit_unit < to {
            let frames = self.select_output();
            b.commit_unit += 1;
            self.hold(b, frames)?;
        }
        Ok(())
    }

    /// The parallel half of [`Decoder::decode_temporal_units`]: parses every OBU in order and
    /// queues the frames, flushing the queue when it holds `max_threads` frames and before
    /// anything that reads a decoded frame.
    fn units_batched(
        &mut self,
        units: &[&[u8]],
        format: Format,
        cancel: &AtomicBool,
        b: &mut Batch,
    ) -> Result<(), Error> {
        for (u, &unit) in units.iter().enumerate() {
            self.parse_unit = u;
            self.pending = None;
            self.clear_metadata();
            for obu in ObuIter::new(unit, format) {
                let r = if cancel.load(Ordering::Relaxed) {
                    Err(Error::Cancelled)
                } else {
                    obu.and_then(|o| self.batch_obu(units, unit, o, cancel, b))
                };
                if let Err(e) = r {
                    return self.batch_failed(units, cancel, b, u, e);
                }
            }
        }
        if let Err(e) = self.flush(units, cancel, b) {
            return self.batch_failed(units, cancel, b, units.len().saturating_sub(1), e);
        }
        self.advance_unit(b, units.len())
    }

    /// One OBU of [`Decoder::units_batched`].
    fn batch_obu(
        &mut self,
        units: &[&[u8]],
        unit: &[u8],
        obu: Obu<'_>,
        cancel: &AtomicBool,
        b: &mut Batch,
    ) -> Result<(), Error> {
        if self.is_barrier(&obu) {
            self.flush(units, cancel, b)?;
            self.advance_unit(b, self.parse_unit)?;
        }
        self.handle_obu(unit, obu, cancel)?;
        if self.jobs.len() >= self.limits.max_threads {
            self.flush(units, cancel, b)?;
        }
        Ok(())
    }

    /// True for an OBU that needs the decoded frames before it: a sequence header that starts a
    /// new coded video sequence (it empties the slots), and a frame header that may be a
    /// `show_existing_frame` (it reads a slot's planes). A false alarm only flushes early.
    fn is_barrier(&self, obu: &Obu<'_>) -> bool {
        match obu.header.obu_type {
            OBU_SEQUENCE_HEADER => SequenceHeader::parse(obu.payload, self.options.operating_point)
                .is_ok_and(|new| {
                    self.seq
                        .as_ref()
                        .is_some_and(|old| !same_sequence(old, &new))
                }),
            // A redundant frame header with no frame pending parses as a frame header too.
            OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER => {
                self.seq
                    .as_ref()
                    .is_some_and(|s| !s.reduced_still_picture_header)
                    && obu.payload.first().is_some_and(|byte| byte & 0x80 != 0)
            }
            _ => false,
        }
    }

    /// An error stopped the batch while parsing unit `u`: decodes the frames queued before it (a
    /// frame that fails there is the lower stream position, so its error wins), keeps the frames
    /// of the units before the failing one and its own partial output, and returns the error.
    fn batch_failed(
        &mut self,
        units: &[&[u8]],
        cancel: &AtomicBool,
        b: &mut Batch,
        u: usize,
        e: Error,
    ) -> Result<(), Error> {
        self.spare = None;
        let e = match self.flush(units, cancel, b) {
            Err(first) => first,
            Ok(()) => e,
        };
        let last = b.stop_unit.unwrap_or(u);
        self.advance_unit(b, last)?;
        let partial = self.select_output();
        if b.out.try_reserve(partial.len()).is_ok() {
            b.out.extend(partial);
        }
        Err(e)
    }

    /// Queues a frame whose tiles are all in: the header half of the reference update (7.20)
    /// happens now, in stream order, the pixels at the next flush.
    fn queue_frame(
        &mut self,
        header: FrameHeader,
        seq: SequenceHeader,
        tiles: Vec<TileRange>,
    ) -> Result<(), Error> {
        self.jobs.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        let refs = self.refs.clone();
        self.refs.update(&header, &seq);
        // A shown frame takes the metadata before it; a hidden one leaves it for the next.
        let (metadata, metadata_bytes) = if header.show_frame {
            (
                std::mem::take(&mut self.metadata),
                std::mem::take(&mut self.metadata_bytes),
            )
        } else {
            (Vec::new(), 0)
        };
        self.jobs.push(QueuedFrame {
            header,
            seq,
            unit: self.parse_unit,
            tiles,
            refs,
            metadata,
            metadata_bytes,
        });
        Ok(())
    }

    /// Decodes the queued frames and commits them in stream order: the slots, the output frames
    /// and the spare state, as [`Decoder::finish_frame`] does one by one.
    ///
    /// The pixels of up to `W` frames decode at once ([`plan`]). When a buffer does not fit in
    /// `max_alloc` the results are dropped and the frames decode one at a time, each with all the
    /// threads, so a refusal is the one a loop of single decodes gives. At the first frame that
    /// fails (the lowest in stream order) the reference state is the one a loop leaves, and the
    /// frames after it are dropped.
    fn flush(&mut self, units: &[&[u8]], cancel: &AtomicBool, b: &mut Batch) -> Result<(), Error> {
        if self.jobs.is_empty() {
            return Ok(());
        }
        let jobs = std::mem::take(&mut self.jobs);
        let threads = self.limits.max_threads.max(1);
        let est = jobs
            .iter()
            .map(|j| frame_estimate(&j.header))
            .max()
            .unwrap_or(1);
        let (w, t) = plan(threads, jobs.len(), self.ledger.left(), est);
        let mut done: Vec<Option<Result<FrameState, Error>>> = Vec::new();
        if w > 1 {
            let spare = Mutex::new(self.spare.take().filter(|_| self.reuse));
            let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let ledger = &self.ledger;
            let results = run_parallel(jobs.len(), w, |i| {
                let Some(job) = jobs.get(i) else {
                    return Err(Error::Invalid("batch job out of range"));
                };
                peak.fetch_max(live.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
                let data = units.get(job.unit).copied().unwrap_or_default();
                let spare = spare.lock().unwrap_or_else(|e| e.into_inner()).take();
                let r = decode_job(ledger, t, job, data, spare, cancel);
                live.fetch_sub(1, Ordering::Relaxed);
                Ok(r)
            })?;
            #[cfg(test)]
            {
                self.jobs_peak = self.jobs_peak.max(peak.load(Ordering::Relaxed));
            }
            let refused = results
                .iter()
                .any(|r| matches!(r, Err(Error::AllocLimit | Error::OutOfMemory)));
            if !refused {
                done = results.into_iter().map(Some).collect();
            }
        }
        let mut done = done.into_iter();
        for mut job in jobs {
            // The slots this frame refreshes are emptied first, as in `finish_frame`.
            let spare_arc = self.empty_slots(&job.header);
            let fs = match done.next().flatten() {
                Some(r) => r,
                None => {
                    let data = units.get(job.unit).copied().unwrap_or_default();
                    let spare = self.spare.take().filter(|_| self.reuse);
                    decode_job(&self.ledger, threads, &job, data, spare, cancel)
                }
            };
            let mut state = match fs {
                Ok(s) if s.lr_frame.len() == usize::from(job.seq.num_planes) => s,
                Ok(_) => {
                    let e = Error::Invalid("frame decode produced no output planes");
                    return Err(self.job_failed(&job, e, false, b));
                }
                Err(e) => return Err(self.job_failed(&job, e, false, b)),
            };
            let planes = std::mem::take(&mut state.lr_frame);
            let stored = match spare_arc {
                Some(mut a) => match Arc::get_mut(&mut a) {
                    Some(v) => {
                        *v = planes;
                        a
                    }
                    None => Arc::new(planes),
                },
                None => Arc::new(planes),
            };
            state.curr_frame.clear();
            self.spare = Some(state);
            let mut tiles = std::mem::take(&mut job.tiles);
            tiles.clear();
            self.tile_buf = tiles;
            for (i, slot) in self.store.iter_mut().enumerate() {
                if (job.header.refresh_frame_flags >> i) & 1 != 0 {
                    *slot = Some(stored.clone());
                }
            }
            if job.header.show_frame
                && let Err(e) = self.commit_output(&mut job, &stored, cancel, b, &mut done)
            {
                return Err(self.job_failed(&job, e, true, b));
            }
        }
        Ok(())
    }

    /// The output frame of a committed job: its unit's earlier frames are finished first.
    ///
    /// A buffer that does not fit may only be short of room because `in_flight` (the decoded
    /// states of the frames after this one) still holds memory. A loop of single decodes does not
    /// have those, so they are dropped (those frames then decode one at a time) and the frame is
    /// made again; the error of the batch is then the one of the loop.
    fn commit_output(
        &mut self,
        job: &mut QueuedFrame,
        planes: &[WorkPlane],
        cancel: &AtomicBool,
        b: &mut Batch,
        in_flight: &mut std::vec::IntoIter<Option<Result<FrameState, Error>>>,
    ) -> Result<(), Error> {
        self.advance_unit(b, job.unit)?;
        let mut r = self.make_shown(job, planes, cancel);
        if matches!(r, Err(Error::AllocLimit | Error::OutOfMemory)) && in_flight.len() > 0 {
            *in_flight = Vec::new().into_iter();
            r = self.make_shown(job, planes, cancel);
        }
        self.ready.push(r?);
        Ok(())
    }

    /// `make_frame` for a queued job. On an error the job keeps its metadata and the ledger bytes
    /// the failed attempt booked for the unit's output are given back, so it can run again.
    fn make_shown(
        &mut self,
        job: &mut QueuedFrame,
        planes: &[WorkPlane],
        cancel: &AtomicBool,
    ) -> Result<Frame, Error> {
        // `make_frame` takes the metadata of the frame, not the live one of the parse position.
        let live = (
            std::mem::replace(&mut self.metadata, std::mem::take(&mut job.metadata)),
            std::mem::replace(&mut self.metadata_bytes, job.metadata_bytes),
        );
        let booked = self.ready_bytes;
        let s = Shown::of_header(&job.header, &job.seq);
        let r = self.make_frame(&s, &job.header, planes, cancel);
        if r.is_err() {
            job.metadata = std::mem::take(&mut self.metadata);
            job.metadata_bytes = self.metadata_bytes;
            self.ledger.release(self.ready_bytes - booked);
            self.ready_bytes = booked;
        }
        (self.metadata, self.metadata_bytes) = live;
        r
    }

    /// A frame failed in a flush: the references are as a loop of single decodes leaves them
    /// (`updated` once the frame went into the slots), and nothing after it runs.
    fn job_failed(&mut self, job: &QueuedFrame, e: Error, updated: bool, b: &mut Batch) -> Error {
        self.refs = job.refs.clone();
        if updated {
            self.refs.update(&job.header, &job.seq);
        }
        b.stop_unit = Some(job.unit);
        self.spare = None;
        e
    }

    /// Empties the slots a frame with `h` refreshes (a frame that fails leaves them empty, as in
    /// dav1d), so the planes no slot keeps any more are free for it. Returns the allocation of
    /// the last one nobody else holds, for the new frame's planes.
    fn empty_slots(&mut self, h: &FrameHeader) -> Option<Arc<Vec<WorkPlane>>> {
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
        spare_arc
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

    /// The frames of the current temporal unit that were output before an error stopped it,
    /// after those of the earlier units of a failed [`Decoder::decode_temporal_units`]. After a
    /// successful call there are none left.
    pub fn take_frames(&mut self) -> Vec<Frame> {
        let mut frames = std::mem::take(&mut self.batch_done);
        if frames.is_empty() {
            return self.select_output();
        }
        let partial = self.select_output();
        if frames.try_reserve(partial.len()).is_ok() {
            frames.extend(partial);
        }
        frames
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
        if self.batching {
            return self.queue_frame(p.header, seq, p.tiles);
        }
        let h = p.header;
        let mut tiles = p.tiles;
        let mut stored = None;
        if !self.probe_only {
            // The slots this frame refreshes are emptied first, as dav1d does (a frame that fails
            // leaves them empty), so the planes of a frame no slot keeps any more are free for this
            // one, and its `Arc` holds the new frame.
            let mut spare_arc = self.empty_slots(&h);
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
        // The frame's slot in `ready` and its plane list, booked until the unit is handed out.
        let overhead = frame_overhead(planes.len());
        self.ledger.reserve(overhead)?;
        self.ready_bytes += overhead;
        self.ready.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
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
/// `max_threads` (docs/threading.md).
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
            decode_tile(&mut fs, &frame_cdf, data, t, row, col, h, seq, cancel)?;
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
    };
    let item: &(dyn Fn(Job, usize) -> Result<(), Error> + Sync) = &|job, i| w.item(job, i);
    let planes = with_team(threads, item, |team| w.lead(team))?;
    let mut fs =
        w.fs.into_inner()
            .ok_or(Error::Invalid("frame decode left no state"))?;
    fs.curr_frame = w.curr.into_inner().unwrap_or_else(|e| e.into_inner());
    fs.lr_frame = planes;
    fs.frame_cdf = Some(w.frame_cdf);
    Ok(fs)
}

/// The pixel work of a queued frame: its frame state (on the arrays of `spare` when the shape
/// fits) and [`decode_frame`] with `threads` threads. `data` is the frame's temporal unit.
fn decode_job(
    ledger: &Arc<Ledger>,
    threads: usize,
    job: &QueuedFrame,
    data: &[u8],
    spare: Option<FrameState>,
    cancel: &AtomicBool,
) -> Result<FrameState, Error> {
    let fs = FrameState::reuse(spare, &job.header, &job.seq, ledger)?;
    decode_frame(
        fs,
        ledger,
        threads,
        &job.header,
        &job.seq,
        data,
        &job.tiles,
        cancel,
    )
}

/// The threads of a frame decode: `max_threads` (0 counts as 1), but no more than its largest
/// stage has items (the tiles, or the 64-row bands of all planes).
fn team_size(max_threads: usize, fs: &FrameState, tiles: usize) -> usize {
    let bands = fs.mi_rows.div_ceil(16) * fs.num_planes;
    max_threads.max(1).min(tiles.max(bands))
}

/// `decode_tile()` of tile `t` (tile row `row`, column `col` of `fs`'s tiling) into `fs`.
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
) -> Result<(), Error> {
    check_cancel(cancel)?;
    let bytes = data
        .get(t.start..t.end)
        .ok_or(Error::Invalid("tile data outside the temporal unit"))?;
    TileDecoder::new(bytes, row, col, frame_cdf, fs, h, seq)?.decode_tile(cancel)?;
    Ok(())
}

/// A stage of the threaded frame decode; its items are listed at [`Wrap::item`].
#[derive(Debug, Clone, Copy)]
enum Job {
    Tile,
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
    /// `UpscaledCdefFrame` (exact size; superres only, `CdefFrame` is read as it is without).
    up_cdef: RwLock<Vec<WorkPlane>>,
    /// `UpscaledCurrFrame` (superres with loop restoration only).
    up_curr: RwLock<Vec<WorkPlane>>,
    /// `LrFrame` (loop restoration only).
    out: Mutex<Vec<WorkPlane>>,
}

impl Wrap<'_> {
    /// The stages, in the order of [`wrapup`] and with its allocations in the same order (the
    /// private bands of the items come on top). Returns `LrFrame`.
    fn lead(&self, team: &Team<'_, Job>) -> Result<Vec<WorkPlane>, Error> {
        let (h, seq, cancel) = (self.h, self.seq, self.cancel);
        if h.allow_intrabc || self.tiles.len() <= 1 {
            let mut g = lock(&self.tiles_fs);
            let fs = g
                .as_mut()
                .ok_or(Error::Invalid("frame decode without a frame state"))?;
            let tile_cols = h.tile_info.tile_cols.max(1);
            for t in self.tiles {
                let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
                decode_tile(fs, &self.frame_cdf, self.data, t, row, col, h, seq, cancel)?;
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
        // With CDEF off `CdefFrame` is `CurrFrame` itself (`cdef` stays empty, see
        // `cdef_src`), so no copy is made.
        let on = {
            let curr = read_lock(&self.curr);
            cdef_on(fs, &curr, seq, h, &curr)
        };
        if on {
            // `CdefFrame` starts as a copy of `CurrFrame`, which the unfiltered blocks keep.
            let cdef = cdef_planes(&read_lock(&self.curr), self.ledger)?;
            *write_lock(&self.cdef) = cdef;
            team.run(Job::Cdef, bands)?;
        }
        let lr_items = fs.num_planes.min(3) * stripes(fs);
        if h.frame_width == h.upscaled_width {
            if !h.uses_lr {
                // Nothing reads the planes after this: they are the output, cropped.
                let planes = if on {
                    std::mem::take(&mut *write_lock(&self.cdef))
                } else {
                    let mut curr = write_lock(&self.curr);
                    let mut v = self.ledger.planes(curr.len())?;
                    v.append(&mut curr);
                    v
                };
                return Ok(crop_planes(fs, planes));
            }
            // No superres: loop restoration reads `CdefFrame` and `CurrFrame` as they are.
            *lock(&self.out) = exact_planes(fs, Some(read_lock(self.cdef_src()).as_slice()))?;
            team.run(Job::Lr, lr_items)?;
            return Ok(std::mem::take(&mut *lock(&self.out)));
        }
        let row_bands = fs.num_planes * fs.frame_height.div_ceil(64);
        *write_lock(&self.up_cdef) = exact_planes(fs, None)?;
        team.run(Job::UpCdef, row_bands)?;
        write_lock(&self.cdef).clear();
        if !h.uses_lr {
            return Ok(std::mem::take(&mut *write_lock(&self.up_cdef)));
        }
        *write_lock(&self.up_curr) = exact_planes(fs, None)?;
        team.run(Job::UpCurr, row_bands)?;
        *lock(&self.out) = exact_planes(fs, Some(read_lock(&self.up_cdef).as_slice()))?;
        team.run(Job::Lr, lr_items)?;
        Ok(std::mem::take(&mut *lock(&self.out)))
    }

    /// `CdefFrame`: `cdef`, or `CurrFrame` itself when CDEF did not run (`cdef` is empty).
    fn cdef_src(&self) -> &RwLock<Vec<WorkPlane>> {
        if read_lock(&self.cdef).is_empty() {
            &self.curr
        } else {
            &self.cdef
        }
    }

    /// Item `i` of stage `job`:
    /// - `Tile`: tile `i` of `tiles`.
    /// - `LfRows`: the vertical edges of plane `i / bands`, row band `i % bands` (16 mode info
    ///   rows; the last band runs to the plane end).
    /// - `LfCols`: the horizontal edges of plane `i / strips`, column strip `i % strips`.
    /// - `Cdef`: row band `i` of every plane.
    /// - `UpCdef`, `UpCurr`: plane `i / bands`, 64-row band `i % bands` of the plane.
    /// - `Lr`: plane `i / stripes`, stripe `i % stripes`.
    ///
    /// An index past the bands of its plane does nothing.
    fn item(&self, job: Job, i: usize) -> Result<(), Error> {
        match job {
            Job::Tile => self.tile(i),
            Job::LfRows => self.lf_rows(i),
            Job::LfCols => self.lf_cols(i),
            Job::Cdef => self.cdef_band(i),
            Job::UpCdef => self.upscale_band(self.cdef_src(), &self.up_cdef, i),
            Job::UpCurr => self.upscale_band(&self.curr, &self.up_curr, i),
            Job::Lr => self.lr_band(i),
        }
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
        )?;
        let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
        if let Some(fs) = lock(&self.tiles_fs).as_mut() {
            fs.merge_tile(&local, sb_size);
        }
        Ok(())
    }

    /// Vertical edges of one row band: each edge filters samples of its own rows only.
    fn lf_rows(&self, i: usize) -> Result<(), Error> {
        let fs = self.state()?;
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
            let buf = WorkPlane::copy_of(
                rows_of(p, y0, y1)?,
                p.stride,
                p.stride,
                y1 - y0,
                self.ledger,
            )?;
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
            let src = p.px.get(x0..).unwrap_or_default();
            let buf = WorkPlane::copy_of(src, p.stride, x1 - x0, p.height, self.ledger)?;
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
    fn cdef_band(&self, b: usize) -> Result<(), Error> {
        let fs = self.state()?;
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
            let buf = WorkPlane::copy_of(
                rows_of(p, y0, y1)?,
                p.stride,
                p.stride,
                y1 - y0,
                self.ledger,
            )?;
            bufs.push((buf, y0, y1));
        }
        {
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

    /// Superres of one 64-row band of one plane, from `src` into the exact-size planes `dst`.
    fn upscale_band(
        &self,
        src: &RwLock<Vec<WorkPlane>>,
        dst: &RwLock<Vec<WorkPlane>>,
        i: usize,
    ) -> Result<(), Error> {
        let fs = self.state()?;
        let bands = fs.frame_height.div_ceil(64).max(1);
        let (plane, b) = (i / bands, i % bands);
        let input = read_lock(src);
        // `dst` has `num_planes` planes of `exact_size`.
        if plane >= fs.num_planes.min(input.len()) {
            return Ok(());
        }
        let (w, ht) = exact_size(fs, plane);
        let (y0, y1) = ((64 * b).min(ht), (64 * b + 64).min(ht));
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
    fn lr_band(&self, i: usize) -> Result<(), Error> {
        check_cancel(self.cancel)?;
        let fs = self.state()?;
        let ns = stripes(fs);
        let (plane, n) = (i / ns, i % ns);
        // Without superres the upscaled frames are `CdefFrame` and `CurrFrame` themselves.
        let (cdef, curr) = if self.h.frame_width != self.h.upscaled_width {
            (read_lock(&self.up_cdef), read_lock(&self.up_curr))
        } else {
            (read_lock(self.cdef_src()), read_lock(&self.curr))
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
        // `LrFrame` starts as `UpscaledCdefFrame`, and only this item writes the stripe's rows.
        let mut buf = {
            let out = lock(&self.out);
            let p = out.get(plane).ok_or(Error::Invalid("plane missing"))?;
            WorkPlane::copy_of(rows_of(p, y0, y1 + 1)?, w, w, y1 + 1 - y0, self.ledger)?
        };
        let mut v = View {
            stride: w,
            px: buf.px.as_mut_slice(),
            x0: 0,
            y0,
        };
        let mut scratch = LrScratch::new(self.ledger)?;
        lr_stripe(fs, self.seq, &g, n, c, u, &mut v, &mut scratch);
        drop((cdef, curr));
        let mut out = lock(&self.out);
        let p = out.get_mut(plane).ok_or(Error::Invalid("plane missing"))?;
        rows_of_mut(p, y0, y1 + 1)?.copy_from_slice(&buf.px);
        Ok(())
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
        let p = match src.and_then(|s| s.get(i)) {
            Some(s) => WorkPlane::copy_of(&s.px, s.stride, w, ht, &fs.ledger)?,
            None => WorkPlane::new(w, ht, &fs.ledger)?,
        };
        out.push(p);
    }
    Ok(out)
}

/// `planes` as output planes of exactly `UpscaledWidth` x `FrameHeight` (chroma `(size + ss) >>
/// ss`) without copying: the same buffers with the size cropped and the stride kept (the output
/// reads rows with the stride, see `row_of`).
fn crop_planes(fs: &FrameState, mut planes: Vec<WorkPlane>) -> Vec<WorkPlane> {
    for (i, p) in planes.iter_mut().enumerate() {
        let (w, ht) = exact_size(fs, i);
        p.width = p.width.min(w);
        p.height = p.height.min(ht);
    }
    planes
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
    // With CDEF off `CdefFrame` is `CurrFrame` itself (`cdef_on` reads `out` only for its
    // strides and sizes, the same as `curr`'s), so no copy is made.
    let cdef = if cdef_on(fs, &fs.curr_frame, seq, h, &fs.curr_frame) {
        Some(cdef_alloc(fs, seq, h, cancel)?)
    } else {
        None
    };
    let cdef_ref = cdef.as_deref().unwrap_or(&fs.curr_frame);
    if h.frame_width == h.upscaled_width {
        if !h.uses_lr {
            // Nothing reads the planes after this: they are the output, cropped.
            let planes = match cdef {
                Some(c) => c,
                None => {
                    // Into a pooled list, so the next frame's `CurrFrame` list needs no allocation.
                    let mut v = fs.ledger.planes(fs.curr_frame.len())?;
                    v.append(&mut fs.curr_frame);
                    v
                }
            };
            return Ok(crop_planes(fs, planes));
        }
        // No superres: `UpscaledCdefFrame` and `UpscaledCurrFrame` are `CdefFrame` and
        // `CurrFrame`, which loop restoration reads as they are.
        let mut out = exact_planes(fs, Some(cdef_ref))?;
        lr_frame(fs, seq, h, cdef_ref, &fs.curr_frame, &mut out, cancel)?;
        if let Some(c) = cdef {
            fs.ledger.put_planes(c);
        }
        return Ok(out);
    }
    let mut up_cdef = exact_planes(fs, None)?;
    upscale(cdef_ref, fs, seq, h, &mut up_cdef, cancel)?;
    if let Some(c) = cdef {
        fs.ledger.put_planes(c);
    }
    if !h.uses_lr {
        return Ok(up_cdef);
    }
    let mut up_curr = exact_planes(fs, None)?;
    upscale(&fs.curr_frame, fs, seq, h, &mut up_curr, cancel)?;
    let mut out = exact_planes(fs, Some(up_cdef.as_slice()))?;
    lr_frame(fs, seq, h, &up_cdef, &up_curr, &mut out, cancel)?;
    fs.ledger.put_planes(up_cdef);
    fs.ledger.put_planes(up_curr);
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
                mi_col_starts: crate::header::Starts::of(&[0, 16, 32, 48]),
                mi_row_starts: crate::header::Starts::of(&[0, 16, 32]),
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

    /// The tiles and the wrapup of `six_tiles( seed )` with `threads` threads and `max_alloc`; the ledger is
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
            .and_then(|fs| decode_frame(fs, &ledger, threads, &h, &seq, &data, &tiles, &cancel()));
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
        }
    }

    /// A temporal unit with one key frame (odd seeds: tile data known to decode; even seeds:
    /// random data, valid syntax or a syntax error) and the sequence header when `with_seq`.
    fn frame_unit(seed: u64, with_seq: bool) -> Vec<u8> {
        let mut w = BitWriter::default();
        still_frame_header(&mut w);
        w.align();
        let mut payload = w.bytes;
        if seed % 2 == 1 {
            payload.extend_from_slice(&[1, 2, 3, 4]); // tile data known to decode (still_obus)
        } else {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for _ in 0..120 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                payload.push((x >> 24) as u8);
            }
        }
        let mut v = obu(OBU_TEMPORAL_DELIMITER, &[]);
        if with_seq {
            v.extend(obu(OBU_SEQUENCE_HEADER, &still_sequence_header()));
        }
        v.extend(obu(OBU_FRAME, &payload));
        v
    }

    fn units_of(n: u64) -> Vec<Vec<u8>> {
        (0..n).map(|i| frame_unit(i + 1, i % 3 == 0)).collect()
    }

    /// `decode_temporal_units` with `threads` threads: the frames (the ones before an error
    /// too), the error and the decoder.
    fn run_batch(
        units: &[Vec<u8>],
        threads: usize,
        max_alloc: usize,
    ) -> (Vec<Frame>, Option<Error>, Decoder) {
        let limits = Limits {
            max_threads: threads,
            max_alloc,
            ..Limits::default()
        };
        let mut d = Decoder::new(Options::default(), limits);
        let units: Vec<&[u8]> = units.iter().map(Vec::as_slice).collect();
        match d.decode_temporal_units(&units, Format::Section5, &cancel()) {
            Ok(f) => (f, None, d),
            Err(e) => {
                let f = d.take_frames();
                (f, Some(e), d)
            }
        }
    }

    /// The reference: one `decode_temporal_unit` call per unit, one thread.
    fn run_loop(units: &[Vec<u8>], max_alloc: usize) -> (Vec<Frame>, Option<Error>, Decoder) {
        let limits = Limits {
            max_threads: 1,
            max_alloc,
            ..Limits::default()
        };
        let mut d = Decoder::new(Options::default(), limits);
        let mut frames = Vec::new();
        for u in units {
            match d.decode_temporal_unit(u, Format::Section5, &cancel()) {
                Ok(f) => frames.extend(f),
                Err(e) => {
                    frames.extend(d.take_frames());
                    return (frames, Some(e), d);
                }
            }
        }
        (frames, None, d)
    }

    #[test]
    fn batch_is_the_same_as_a_loop_of_units_for_every_thread_count() {
        let ok = units_of(10);
        let mut truncated = ok.clone();
        truncated[6] = [obu(OBU_TEMPORAL_DELIMITER, &[]), obu(OBU_FRAME, &[0x00])].concat();
        let mut no_frame = ok.clone();
        no_frame[4] = obu(OBU_TEMPORAL_DELIMITER, &[]);
        let (mut frames, mut errors) = (0, 0);
        for units in [&ok, &truncated, &no_frame] {
            let (want, want_err, want_d) = run_loop(units, 64 << 20);
            frames += want.len();
            errors += usize::from(want_err.is_some());
            for threads in [0, 1, 2, 3, 8] {
                let (got, err, d) = run_batch(units, threads, 64 << 20);
                assert_eq!(got, want, "{threads} threads");
                assert_eq!(err, want_err, "{threads} threads");
                assert_eq!(d.refs, want_d.refs, "{threads} threads");
                let held = |d: &Decoder| d.store.each_ref().map(|s| s.is_some());
                assert_eq!(held(&d), held(&want_d), "{threads} threads");
            }
        }
        // The streams decode some frames and the broken one fails: the test compares something.
        assert!(frames > 0);
        assert!(errors > 0);
    }

    /// The fuzz seed of `frames_threads`: a full (not reduced) sequence header with a hidden
    /// showable key frame in the first unit, and an `OBU_REDUNDANT_FRAME_HEADER` with
    /// `show_existing_frame` of its slot in the second.
    fn show_existing_units() -> Vec<Vec<u8>> {
        let seed = include_bytes!("../tests/fuzz/seeds/regress-show-existing-redundant-header.bin");
        let stream = &seed[..seed.len() - 1]; // without the fuzz option byte
        let units = crate::split_temporal_units(stream, Format::Section5, &Limits::default());
        units.unwrap().into_iter().map(<[u8]>::to_vec).collect()
    }

    #[test]
    fn a_hidden_frame_shown_by_a_redundant_header_is_the_same_for_every_thread_count() {
        let units = show_existing_units();
        assert_eq!(units.len(), 2);
        let (want, want_err, _) = run_loop(&units, 64 << 20);
        assert_eq!((want.len(), want_err), (1, None));
        for threads in [2, 3, 8] {
            let (got, err, _) = run_batch(&units, threads, 64 << 20);
            assert_eq!(err, None, "{threads} threads");
            assert_eq!(got, want, "{threads} threads");
        }
    }

    #[test]
    fn batch_refusals_are_the_ones_of_a_loop_at_every_max_alloc() {
        let units = units_of(8);
        let mut refused = 0;
        for max_alloc in (1 << 10..(1 << 10) + (80 << 10)).step_by(512) {
            let (want, want_err, _) = run_loop(&units, max_alloc);
            refused += usize::from(want_err == Some(Error::AllocLimit));
            for threads in [2, 3, 8] {
                let (got, err, _) = run_batch(&units, threads, max_alloc);
                assert_eq!(err, want_err, "{threads} threads at {max_alloc}");
                assert_eq!(got, want, "{threads} threads at {max_alloc}");
            }
        }
        // The sweep covers refusals: it compares something.
        assert!(refused > 0);
    }

    #[test]
    fn show_existing_in_either_header_obu_is_a_barrier() {
        let (_, _, mut d) = run_batch(&units_of(1), 2, 64 << 20);
        // The sequence is a still one, which has no show_existing_frame: make it a full one.
        d.seq.as_mut().unwrap().reduced_still_picture_header = false;
        for obu_type in [OBU_FRAME_HEADER, OBU_REDUNDANT_FRAME_HEADER] {
            let bytes = obu(obu_type, &[0x80]);
            let o = ObuIter::new(&bytes, Format::Section5)
                .next()
                .unwrap()
                .unwrap();
            assert!(d.is_barrier(&o), "type {obu_type}");
            let bytes = obu(obu_type, &[0x00]);
            let o = ObuIter::new(&bytes, Format::Section5)
                .next()
                .unwrap()
                .unwrap();
            assert!(!d.is_barrier(&o), "type {obu_type}");
        }
    }

    #[test]
    fn plan_stays_within_max_threads() {
        for max_threads in 0..=17 {
            for queued in 0..=20 {
                for left in [0, 1 << 10, 5 << 20, usize::MAX] {
                    let (w, t) = plan(max_threads, queued, left, 2 << 20);
                    assert!(w >= 1 && t >= 1);
                    assert!(w * t <= max_threads.max(1));
                    assert!(w <= queued.max(1));
                }
            }
        }
        assert_eq!(plan(16, 16, usize::MAX, 1), (16, 1));
        assert_eq!(plan(16, 8, usize::MAX, 1), (8, 2));
        assert_eq!(plan(16, 16, 3 << 20, 2 << 20), (1, 16));
        assert_eq!(plan(1, 16, usize::MAX, 1), (1, 1));
    }

    #[test]
    fn batch_frames_in_flight_stay_within_max_threads_and_max_alloc() {
        let units = units_of(8);
        for threads in [2, 3, 8] {
            let (_, _, d) = run_batch(&units, threads, 64 << 20);
            assert!(d.jobs_peak <= threads, "{threads} threads");
        }
        // W frames in flight hold W frame states: the peak is above one frame's at a time.
        let (_, one_err, one) = run_batch(&units, 1, 64 << 20);
        let (_, eight_err, eight) = run_batch(&units, 8, 64 << 20);
        if one_err.is_none() && eight_err.is_none() {
            assert!(eight.ledger.peak() > one.ledger.peak());
        }
        // Whatever the limit, the peak stays within it, and a refusal is `AllocLimit`.
        for max_alloc in [2 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20] {
            for threads in [2, 8] {
                let (_, e, d) = run_batch(&units, threads, max_alloc);
                assert!(d.ledger.peak() <= max_alloc as u64, "{max_alloc} {threads}");
                assert_ne!(e, Some(Error::OutOfMemory));
            }
        }
    }
}
