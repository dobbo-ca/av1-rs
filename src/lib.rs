//! An AV1 decoder for intra frames: AVIF stills and key frames, bit-exact to the AV1
//! Bitstream & Decoding Process Specification. A stream with inter frames is decoded up to the
//! first one, which stops the decode with [`Error::InterFrame`]; use [`Decoder`] to keep the
//! frames before it.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use av1::{Format, Limits, Options};
//! use std::sync::atomic::AtomicBool;
//!
//! let data = std::fs::read("still.obu")?;
//! let limits = Limits {
//!     max_threads: 1,
//!     max_alloc: 64 << 20,
//!     ..Limits::default()
//! };
//! let cancel = AtomicBool::new(false);
//! let frames = av1::decode(&data, Format::Section5, &Options::default(), &limits, &cancel)?;
//! println!("{}x{}", frames[0].width, frames[0].height);
//! # Ok(())
//! # }
//! ```
//!
//! # Resource limits
//!
//! Every entry point takes a [`Limits`]. The sizes in the bitstream are untrusted: a frame wider
//! than `max_width`, taller than `max_height` or with more than `max_pixels` luma samples is
//! refused before anything is allocated ([`Error::WidthLimit`], [`Error::HeightLimit`],
//! [`Error::PixelLimit`]). The working memory of a decoder, or of one `decode_avif` call, all
//! threads together (frame buffers, film grain buffers, the reference slots, the AVIF canvas,
//! the output frames of a temporal unit until it returns) is charged against `max_alloc` before
//! it is allocated
//! ([`Error::AllocLimit`], [`Error::OutOfMemory`] when the system refuses); returned frames are
//! not counted ([`decode`] counts the frames it collects until it returns). `max_metadata`
//! bounds the metadata OBUs kept with a frame, on top of `max_alloc`
//! ([`Error::MetadataLimit`]). `max_threads` caps the worker threads of one call (1 spawns
//! none).
//!
//! # Cancelling
//!
//! [`decode`] and [`Decoder::decode_temporal_unit`] take an `&AtomicBool`. Setting it from
//! another thread makes the decode return [`Error::Cancelled`] at the next OBU or tile.

#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod avif;
mod bits;
mod cdef;
mod cdf;
mod cdf_tables;
mod coef;
mod consts;
mod decoder;
mod error;
mod frame;
mod grain;
mod header;
mod intrabc;
mod ledger;
mod lf;
mod limits;
mod lr;
mod obu;
mod parallel;
mod predict;
mod quant;
mod recon;
mod scan;
mod seq;
// The SIMD tiers; `unsafe` is allowed only in its load/store helpers and in the dispatch fns.
#[allow(dead_code)] // the kernel families join one per PR
mod simd;
mod superres;
mod symbol;
mod tables;
mod tile;
mod transform;

use obu::TemporalUnits;
use std::sync::atomic::AtomicBool;

pub use avif::{AvifImage, decode_avif, decode_avif_item};
pub use decoder::Decoder;
pub use error::Error;
pub use isobmff;
pub use limits::Limits;
pub use obu::split_temporal_units;

/// How the OBUs of a stream are framed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// The low-overhead bitstream format of spec 5.2 (`.obu` files, IVF frame payloads, AVIF
    /// items): a run of OBUs that each carry `obu_size` (`obu_has_size_field` is 1).
    Section5,
    /// The length-delimited format of spec Annex B: `temporal_unit_size`, `frame_unit_size` and
    /// `obu_length` prefixes.
    AnnexB,
}

/// What to decode and what to output.
#[derive(Debug, Clone)]
pub struct Options {
    /// The operating point to decode (default 0), clamped to the sequence header's
    /// `operating_points_cnt_minus_1`. It selects which spatial and temporal layers are decoded.
    pub operating_point: usize,
    /// Output every shown frame of every layer (default `false`: only the highest spatial layer
    /// frame of each temporal unit when the stream uses layers).
    pub all_layers: bool,
    /// Apply film grain synthesis to the output when the frame has film grain parameters
    /// (default `true`).
    pub film_grain: bool,
    /// Keep the metadata OBUs (HDR CLL / MDCV, ITU-T T.35, timecode, scalability and any other
    /// type) raw in [`Frame::metadata`] (default `true`). `false` drops them. The colour
    /// configuration is always in [`Frame::color`].
    pub keep_metadata: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            operating_point: 0,
            all_layers: false,
            film_grain: true,
            keep_metadata: true,
        }
    }
}

/// The color description of a sequence (`color_config()`, spec 5.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorInfo {
    /// `color_primaries` (H.273 code, 2 = unspecified).
    pub color_primaries: u8,
    /// `transfer_characteristics` (H.273 code, 2 = unspecified).
    pub transfer_characteristics: u8,
    /// `matrix_coefficients` (H.273 code, 2 = unspecified).
    pub matrix_coefficients: u8,
    /// `color_range`: `true` for full range, `false` for studio range.
    pub color_range: bool,
    /// `chroma_sample_position`: 0 unknown, 1 vertical, 2 colocated.
    pub chroma_sample_position: u8,
}

/// The samples of one plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaneData {
    /// 8-bit samples.
    U8(Vec<u8>),
    /// 10 and 12-bit samples.
    U16(Vec<u16>),
}

/// One plane of an output frame: `width * height` samples, rows in order, `stride == width`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub width: u32,
    pub height: u32,
    pub data: PlaneData,
}

/// A metadata OBU (spec 5.8) kept with a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// The `metadata_type`: 1 HDR content light level, 2 HDR mastering display colour volume,
    /// 3 scalability, 4 ITU-T T.35, 5 timecode; other values are kept raw too.
    pub metadata_type: u64,
    /// The raw payload after `metadata_type`, up to the trailing bits.
    pub payload: Vec<u8>,
}

/// One output frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Width in luma samples: `UpscaledWidth`.
    pub width: u32,
    /// Height in luma samples: `FrameHeight`.
    pub height: u32,
    /// The render size, a display hint that does not change the decoded samples.
    pub render_width: u32,
    pub render_height: u32,
    /// 8, 10 or 12.
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub color: ColorInfo,
    /// One plane for monochrome, else three (Y, U, V). 8-bit frames use [`PlaneData::U8`],
    /// 10 and 12-bit frames [`PlaneData::U16`].
    pub planes: Vec<Plane>,
    /// The metadata OBUs that came before this frame in its temporal unit.
    pub metadata: Vec<Metadata>,
    /// Whether film grain synthesis was applied to `planes`.
    pub film_grain_applied: bool,
    /// `frame_type`: 0 key frame, 2 intra only frame (the type of the frame; for a
    /// `show_existing_frame` the type of the frame shown).
    pub frame_type: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
}

/// The header facts of one frame, from [`probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameInfo {
    pub frame_type: u8,
    pub show_frame: bool,
    pub show_existing_frame: bool,
    /// `UpscaledWidth`, the output width.
    pub width: u32,
    /// `FrameHeight`.
    pub height: u32,
    /// `FrameWidth`, the coded width (before superres upscaling).
    pub frame_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub temporal_id: u8,
    pub spatial_id: u8,
}

/// The result of [`probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// The frames whose headers parsed, in stream order.
    pub frames: Vec<FrameInfo>,
    /// Why the probe stopped before the end of the stream, if it did: [`Error::InterFrame`] at
    /// the first inter frame, or a parse error.
    pub stopped: Option<Error>,
}

/// Decodes a whole stream and returns its output frames.
///
/// Decoding stops at the first inter frame with [`Error::InterFrame`] and the frames before it
/// are not returned: use [`Decoder`] to keep them.
/// The frames it collects stay charged against `max_alloc` until it returns, so a stream whose
/// output is larger than `max_alloc` is `AllocLimit`: use [`Decoder`] to stream it.
///
/// # Errors
/// `Truncated` or `Invalid` for a broken stream, `Unsupported` for a feature this decoder does
/// not handle, `InterFrame`, the limit errors of `limits` (`WidthLimit`, `HeightLimit`,
/// `PixelLimit`, `AllocLimit`, `MetadataLimit`), `OutOfMemory`, and `Cancelled` when `cancel`
/// was set.
pub fn decode(
    data: &[u8],
    format: Format,
    options: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<Frame>, Error> {
    let mut decoder = Decoder::new(options.clone(), *limits);
    let mut frames = Vec::new();
    // Framing is checked up front (without allocating), so a broken stream decodes nothing.
    for unit in TemporalUnits::new(data, format) {
        unit?;
    }
    for unit in TemporalUnits::new(data, format) {
        let unit = unit?;
        let out = decoder.decode_temporal_unit(unit, format, cancel)?;
        // Every frame collected so far stays booked: the output of the call counts too.
        decoder.hold_output(&out)?;
        frames.extend(out);
    }
    Ok(frames)
}

/// Parses every OBU header, sequence header and frame header of a stream without decoding any
/// tile, and returns the facts of each frame. Stops at the first inter frame (the `stopped`
/// field of the result is then `Some(Error::InterFrame)`).
///
/// The frame size limits of `limits` apply to every frame header (a frame over them stops the
/// probe with the limit error); nothing is allocated for pixels.
///
/// # Errors
/// Only a broken framing of the whole stream (`Truncated`, `Invalid` from
/// [`split_temporal_units`]). Errors inside a temporal unit end up in `Probe::stopped`.
pub fn probe(data: &[u8], format: Format, limits: &Limits) -> Result<Probe, Error> {
    let cancel = AtomicBool::new(false);
    let mut decoder = Decoder::probing(*limits);
    let mut stopped = None;
    let mut units = TemporalUnits::new(data, format);
    for unit in units.by_ref() {
        let unit = unit?;
        if let Err(e) = decoder.decode_temporal_unit(unit, format, &cancel) {
            stopped = Some(e);
            break;
        }
    }
    // Broken framing after the stop is still an error.
    for unit in units {
        unit?;
    }
    Ok(Probe {
        frames: decoder.take_infos(),
        stopped,
    })
}
