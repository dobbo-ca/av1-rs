//! The AVIF layer: the item graph comes from `isobmff`, the pixels from [`Decoder`].
//!
//! Coded items are decoded with their own decoder, grid tiles (after tile 0) on up to `max_threads` threads; tiles are pasted into
//! the output as they are decoded. Colour and the alpha plane are passed through raw. Transforms
//! are returned raw too, unless `Options::apply_transforms` is on: then `isobmff::apply_transforms`
//! runs on each grid tile and identity input, on the derived result and on the alpha image.

use crate::ledger::{Held, Ledger};
use crate::parallel::run_each;
use crate::{Decoder, Error, Format, Frame, Limits, Options, Plane, PlaneData};
use isobmff::{
    Av1c, Chroma, Codec, CodecConfig, Grid, Heif, Image, ImageKind, PlaneSamples, Transform,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A decoded AVIF still.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvifImage {
    /// The primary image; a grid is assembled and cropped to its output size.
    pub image: Frame,
    /// The alpha auxiliary image, decoded the same way.
    pub alpha: Option<Frame>,
    /// The transformative properties of the primary image (`clap`, `irot`, `imir`), raw, in
    /// `ipma` order, whatever `Options::apply_transforms` is. Applied to `image` and `alpha` only
    /// when that option is on.
    pub transforms: Vec<isobmff::Transform>,
    /// The alpha plane is premultiplied.
    pub premultiplied: bool,
    /// Every `colr` property of the primary image, raw.
    pub colour: Vec<isobmff::Colour>,
}

fn container(e: isobmff::Error) -> Error {
    Error::Container(e.class())
}

struct Ctx<'a> {
    heif: &'a Heif,
    file: &'a [u8],
    options: &'a Options,
    limits: &'a Limits,
    ilimits: &'a isobmff::Limits,
    /// One budget for the whole call: every decoder, the grid canvas, the item copy and the
    /// decoded primary image (while the alpha item decodes).
    ledger: Arc<Ledger>,
    cancel: &'a AtomicBool,
}

/// Decodes the primary image of an AVIF file, and its alpha image when there is one.
///
/// A coded item is decoded as one Section 5 temporal unit: the `av1C` configOBUs followed by the
/// item data. A grid is assembled from its tiles; an identity image is its input. Overlays and
/// other item types are [`Error::Unsupported`]. The frame size limits of `limits` apply to every
/// decoded frame and to the assembled grid. `max_alloc` is the peak of the call: an eighth is
/// given to the `isobmff` parse, an eighth to its reads of the item bytes, and the rest to one
/// ledger for every decoder, the assembled grid, the item copies and the primary image while the
/// alpha item decodes. `cancel`
/// is checked between tiles and inside every decode.
///
/// With `Options::apply_transforms` the `clap`, `irot` and `imir` properties are applied: to each
/// grid tile and identity input first, then to the derived image, and to the alpha image as to its
/// master; the output is charged to `max_alloc`. `AvifImage::transforms` is raw either way.
///
/// Exif, XMP, gain map, depth and thumbnail items are not returned; use `isobmff::parse` for them.
///
/// # Errors
/// [`Error::Container`] (the class of the `isobmff` error) for a broken file, otherwise the
/// errors of [`Decoder::decode_temporal_unit`], and `Invalid` for tiles that do not fit together.
/// Item bytes that do not fit their eighth of `max_alloc` are [`Error::AllocLimit`].
pub fn decode_avif(
    file: &[u8],
    options: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<AvifImage, Error> {
    // An eighth of `max_alloc` for the parse and an eighth for the item reads (both outside the
    // ledger), the rest for the ledger.
    let ilimits = isobmff::Limits {
        max_alloc: limits.max_alloc / 8,
        max_threads: limits.max_threads,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let heif = isobmff::parse(file, &ilimits, cancel).map_err(container)?;
    let ledger = Arc::new(Ledger::default());
    // A `max_alloc` below the small-allocation allowance only means that every charge fails.
    let _ = ledger.reset(limits.max_alloc - 2 * (limits.max_alloc / 8), 0);
    let ctx = Ctx {
        heif: &heif,
        file,
        options,
        limits,
        ilimits: &ilimits,
        ledger,
        cancel,
    };
    let primary = heif.primary_image().map_err(container)?;
    let _list = ctx.book_image(&primary)?;
    let image = decode_image(&ctx, &primary, false)?;
    let image = ctx.transformed_booked(&primary.transforms, image)?;
    let alpha = match primary.alpha {
        Some(id) => {
            let item = heif.image(id).map_err(container)?;
            let _list = ctx.book_image(&item)?;
            let alpha = decode_image(&ctx, &item, false)?;
            // The alpha image gets the transforms of its master.
            Some(ctx.transformed_booked(&primary.transforms, alpha)?)
        }
        None => None,
    };
    Ok(AvifImage {
        image,
        alpha,
        transforms: primary.transforms,
        premultiplied: primary.premultiplied,
        colour: primary.colour,
    })
}

/// Decodes image item `id` of `heif`, the parse of `file`, as [`decode_avif`] decodes the primary
/// image: a coded `av01` item, a grid or an identity image. There is no alpha image, the
/// transforms of the item are applied only with `Options::apply_transforms`, and nothing is
/// parsed again.
///
/// `limits.max_alloc` is the peak of this call: an eighth is given to the `isobmff` reads of the
/// item bytes and the rest to one ledger for every decoder, the assembled grid, the item copies
/// and the output. `limits.max_metadata` caps the metadata of one frame; what every decoder of
/// the call keeps is also booked in that ledger. The frame size limits apply to every decoded
/// frame and to the grid; `cancel` is checked between tiles and inside every decode.
///
/// # Errors
/// [`Error::Container`] when `id` is not an image item of `heif` or the item is broken, otherwise
/// the errors of [`decode_avif`] for the primary image.
pub fn decode_avif_item(
    heif: &Heif,
    file: &[u8],
    id: u32,
    options: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Frame, Error> {
    let ilimits = isobmff::Limits {
        max_alloc: limits.max_alloc / 8,
        max_threads: limits.max_threads,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let ledger = Arc::new(Ledger::default());
    ledger.reset(limits.max_alloc - limits.max_alloc / 8, 0)?;
    let ctx = Ctx {
        heif,
        file,
        options,
        limits,
        ilimits: &ilimits,
        ledger,
        cancel,
    };
    let image = heif.image(id).map_err(container)?;
    let _list = ctx.book_image(&image)?;
    let frame = decode_image(&ctx, &image, false)?;
    ctx.transformed_booked(&image.transforms, frame)
}

impl Ctx<'_> {
    /// Books the lists an [`Image`] owns while it is held: `Heif::image` builds them outside any
    /// budget, and their length comes from the file (a grid's tile ids, up to 65536; an overlay's
    /// inputs; the thumbnail, Exif, XMP and auxiliary references; the transforms and colour
    /// properties). Codec configs and ICC data are shared or are extents of the file, not copies.
    fn book_image(&self, image: &Image) -> Result<Held<'_>, Error> {
        fn bytes<T>(capacity: usize) -> usize {
            capacity.saturating_mul(size_of::<T>())
        }
        let kind = match &image.kind {
            ImageKind::Grid(g) => bytes::<u32>(g.tiles.capacity()),
            ImageKind::Overlay(o) => bytes::<isobmff::OverlayInput>(o.inputs.capacity()),
            _ => 0,
        };
        let ids = image.thumbnails.capacity()
            + image.exif.capacity()
            + image.xmp.capacity()
            + image.gain_map.as_ref().map_or(0, |g| g.xmp.capacity());
        let n = [
            kind,
            bytes::<u32>(ids),
            bytes::<Transform>(image.transforms.capacity()),
            bytes::<isobmff::Colour>(image.colour.capacity()),
            bytes::<isobmff::Aux>(image.aux.capacity()),
            image.pixi.as_ref().map_or(0, Vec::capacity),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add);
        self.ledger.charge(n)
    }

    /// [`transform_frame`] for a frame the ledger holds (a decoded image); the result is booked
    /// in its place.
    fn transformed_booked(&self, ts: &[Transform], frame: Frame) -> Result<Frame, Error> {
        transform_frame(
            self.options,
            self.limits,
            &self.ledger,
            self.cancel,
            ts,
            frame,
        )
    }
}

/// Applies `ts` to `frame` with `isobmff::apply_transforms` when `options.apply_transforms` is
/// on. `frame` comes booked in `ledger` for [`held_bytes`] and the result goes back booked for
/// its own; on an error the input is freed and its whole booking, the metadata included, given
/// back. What the transforms may allocate ([`transform_budget`]) is booked before they run and
/// passed to `isobmff` as its `max_alloc`, so the grid tiles that transform at once stay within
/// the ledger together. The frame size limits apply to the result.
fn transform_frame(
    options: &Options,
    limits: &Limits,
    ledger: &Ledger,
    cancel: &AtomicBool,
    ts: &[Transform],
    frame: Frame,
) -> Result<Frame, Error> {
    if !options.apply_transforms || ts.is_empty() {
        return Ok(frame);
    }
    // The metadata moves to the result unchanged and stays booked apart from the samples.
    let metadata = metadata_bytes(&frame);
    let mut booked = frame_bytes(&frame);
    let r = match apply(limits, ledger, cancel, ts, frame, &mut booked) {
        Ok(f) => rebook(ledger, booked, frame_bytes(&f)).map(|()| f),
        Err(e) => {
            ledger.release(booked);
            Err(e)
        }
    };
    if r.is_err() {
        ledger.release(metadata);
    }
    r
}

/// What `isobmff::apply_transforms` may allocate for `ts` on `planes` planes of `bytes` sample
/// bytes: each crop and quarter rotation makes a new set of planes no larger than the input (one
/// and a half times it when a 4:2:2 rotation makes 4:4:4) and its plane list, and the set before
/// it is freed only once the next is made, so at most two sets are live. Mirrors and 180 degree
/// rotations work in place.
fn transform_budget(bytes: usize, planes: usize, chroma: Chroma, ts: &[Transform]) -> usize {
    let fresh = ts
        .iter()
        .filter(|t| matches!(t, Transform::Rotate(90 | 270) | Transform::Clap(_)))
        .count();
    let quarter = ts.iter().any(|t| matches!(t, Transform::Rotate(90 | 270)));
    let set = if chroma == Chroma::Yuv422 && quarter {
        bytes.saturating_add(bytes / 2 + 1)
    } else {
        bytes
    };
    set.saturating_add(planes.saturating_mul(size_of::<isobmff::Plane>()))
        .saturating_mul(fresh.min(2))
}

/// Turns a booking of `have` bytes into one of `want`; on an error nothing stays booked.
fn rebook(ledger: &Ledger, have: usize, want: usize) -> Result<(), Error> {
    if have >= want {
        ledger.release(have - want);
        return Ok(());
    }
    ledger
        .reserve(want - have)
        .inspect_err(|_| ledger.release(have))
}

/// The work of [`transform_frame`]; `booked` grows by the transform budget once it is booked.
fn apply(
    limits: &Limits,
    ledger: &Ledger,
    cancel: &AtomicBool,
    ts: &[Transform],
    mut frame: Frame,
    booked: &mut usize,
) -> Result<Frame, Error> {
    let chroma = match (frame.mono_chrome, frame.subsampling_x, frame.subsampling_y) {
        (true, _, _) => Chroma::Mono,
        (false, true, true) => Chroma::Yuv420,
        (false, true, false) => Chroma::Yuv422,
        (false, false, false) => Chroma::Yuv444,
        (false, false, true) => return Err(Error::Unsupported("avif transform of 4:4:0")),
    };
    let ilimits = isobmff::Limits {
        max_alloc: 0,
        max_threads: limits.max_threads,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let (w, h) = isobmff::transformed_size(frame.width, frame.height, ts, &ilimits)
        .map_err(isobmff_error)?;
    if w > limits.max_width {
        return Err(Error::WidthLimit);
    }
    if h > limits.max_height {
        return Err(Error::HeightLimit);
    }
    // The input stays booked while the output is made; the transforms get exactly their budget.
    let budget = transform_budget(*booked, frame.planes.len(), chroma, ts);
    ledger.reserve(budget)?;
    *booked += budget;
    let ilimits = isobmff::Limits {
        max_alloc: budget,
        ..ilimits
    };
    let planes = std::mem::take(&mut frame.planes)
        .into_iter()
        .map(|p| isobmff::Plane {
            width: p.width,
            height: p.height,
            samples: match p.data {
                PlaneData::U8(v) => PlaneSamples::U8(v),
                PlaneData::U16(v) => PlaneSamples::U16(v),
            },
        })
        .collect();
    let planes = isobmff::Planes {
        width: frame.width,
        height: frame.height,
        chroma,
        planes,
    };
    let out = isobmff::apply_transforms(planes, ts, &ilimits, cancel).map_err(isobmff_error)?;
    (frame.width, frame.height) = (out.width, out.height);
    (frame.render_width, frame.render_height) = (out.width, out.height);
    (frame.subsampling_x, frame.subsampling_y) = match out.chroma {
        Chroma::Mono => (frame.subsampling_x, frame.subsampling_y),
        Chroma::Yuv420 => (true, true),
        Chroma::Yuv422 => (true, false),
        Chroma::Yuv444 => (false, false),
    };
    frame.planes = out
        .planes
        .into_iter()
        .map(|p| Plane {
            width: p.width,
            height: p.height,
            data: match p.samples {
                PlaneSamples::U8(v) => PlaneData::U8(v),
                PlaneSamples::U16(v) => PlaneData::U16(v),
            },
        })
        .collect();
    Ok(frame)
}

fn isobmff_error(e: isobmff::Error) -> Error {
    match e {
        isobmff::Error::LimitExceeded(isobmff::Limit::Alloc) => Error::AllocLimit,
        isobmff::Error::LimitExceeded(isobmff::Limit::Pixels) => Error::PixelLimit,
        isobmff::Error::OutOfMemory => Error::OutOfMemory,
        isobmff::Error::Cancelled => Error::Cancelled,
        e => container(e),
    }
}

fn decode_image(ctx: &Ctx<'_>, image: &Image, nested: bool) -> Result<Frame, Error> {
    match &image.kind {
        ImageKind::Coded { .. } => decode_coded(ctx, image),
        ImageKind::Grid(grid) => decode_grid(ctx, grid),
        // The input of an identity image is a coded image or a grid, not another identity.
        ImageKind::Identity { input } if !nested => {
            let inner = ctx.heif.image(*input).map_err(container)?;
            let _list = ctx.book_image(&inner)?;
            // The input gets its own transforms before the identity image's.
            let frame = decode_image(ctx, &inner, true)?;
            ctx.transformed_booked(&inner.transforms, frame)
        }
        ImageKind::Identity { .. } => Err(Error::Unsupported("avif nested identity")),
        ImageKind::Overlay(_) => Err(Error::Unsupported("avif overlay")),
        ImageKind::Unsupported(_) => Err(Error::Unsupported("avif item")),
    }
}

/// One coded AV1 item: configOBUs + item data, decoded by a fresh decoder; the last output
/// frame is the image.
fn decode_coded(ctx: &Ctx<'_>, image: &Image) -> Result<Frame, Error> {
    let decoder = Decoder::with_ledger(ctx.options.clone(), *ctx.limits, Arc::clone(&ctx.ledger));
    // The frame stays booked for the rest of the call (the alpha item decodes while it is held).
    decode_item(ctx, image, decoder)
}

/// [`decode_coded`] with the decoder given (a grid tile shares its ledger with the others). The
/// frame is booked for [`held_bytes`] from the moment the decoder makes it; the caller gives
/// that back when it frees the frame.
fn decode_item(ctx: &Ctx<'_>, image: &Image, mut decoder: Decoder) -> Result<Frame, Error> {
    let ImageKind::Coded {
        codec: Codec::Av1,
        config: CodecConfig::Av1(av1c),
    } = &image.kind
    else {
        return Err(Error::Unsupported("avif codec"));
    };
    let (unit, _held) = temporal_unit(ctx, image.id, av1c)?;
    let (mut frames, booked) = decoder.decode_unit_booked(&unit, Format::Section5, ctx.cancel)?;
    let frame = frames.pop();
    drop(frames);
    let Some(frame) = frame else {
        ctx.ledger.release(booked);
        return Err(Error::Invalid("avif item has no shown frame"));
    };
    rebook(&ctx.ledger, booked, held_bytes(&frame))?;
    Ok(frame)
}

/// The Section 5 stream of one item: the `av1C` configOBUs, then the item data.
fn temporal_unit<'a>(ctx: &'a Ctx<'_>, id: u32, av1c: &Av1c) -> Result<(Vec<u8>, Held<'a>), Error> {
    let config = Heif::slice(ctx.file, av1c.config_obus).ok_or(Error::Truncated)?;
    // A copy over the item-read share is refused as an allocation (`AllocLimit`), so a grid
    // retries it like any other refusal.
    let data = ctx
        .heif
        .item_bytes(ctx.file, id, ctx.ilimits, ctx.cancel)
        .map_err(isobmff_error)?;
    let total = config
        .len()
        .checked_add(data.len())
        .ok_or(Error::AllocLimit)?;
    // Booked until the item has decoded (the caller keeps the guard).
    let held = ctx.ledger.charge(total)?;
    let mut unit = Vec::new();
    unit.try_reserve_exact(total)
        .map_err(|_| Error::OutOfMemory)?;
    unit.extend_from_slice(config);
    unit.extend_from_slice(&data);
    Ok((unit, held))
}

fn decode_grid(ctx: &Ctx<'_>, grid: &Grid) -> Result<Frame, Error> {
    let count = u64::from(grid.rows) * u64::from(grid.columns);
    if count == 0 || count != grid.tiles.len() as u64 {
        return Err(Error::Invalid("avif grid tile count"));
    }
    // Tiles decode on up to `max_threads` threads. The tiles decoding at once share one budget
    // (`max_alloc`) and the threads: each gets `max_threads / workers` of them, so one thread
    // when there are at least as many tiles as threads.
    let max_threads = ctx.limits.max_threads.max(1);
    let workers = max_threads.min(grid.tiles.len());
    // Every worker's `item_bytes` call has its own budget, and a multi-extent item is copied
    // outside the ledger: in the parallel phase the eighth of `max_alloc` given to the reads (by
    // `decode_avif` and `decode_avif_item`) is split between the workers, so the copies that are
    // live at once stay within it. Tile 0 and the serial retry read alone, with the whole eighth.
    let ilimits = isobmff::Limits {
        max_alloc: ctx.ilimits.max_alloc / workers,
        ..*ctx.ilimits
    };
    let split = &Ctx {
        heif: ctx.heif,
        file: ctx.file,
        options: ctx.options,
        limits: ctx.limits,
        ilimits: &ilimits,
        ledger: Arc::clone(&ctx.ledger),
        cancel: ctx.cancel,
    };
    let inner = Limits {
        max_threads: (max_threads / workers).max(1),
        ..*ctx.limits
    };
    let canvas: Mutex<Option<Canvas>> = Mutex::new(None);
    // Decodes tile `i`, pastes it and frees it. The frame is booked from when its decoder makes
    // it, through its transforms (`transform_frame`), until it is pasted and freed. Tile 0 makes
    // the canvas.
    let put = |ctx: &Ctx<'_>, i: usize| -> Result<(), Error> {
        if ctx.cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let id = grid
            .tiles
            .get(i)
            .ok_or(Error::Invalid("avif grid tile count"))?;
        let tile = ctx.heif.image(*id).map_err(container)?;
        let _list = ctx.book_image(&tile)?;
        let decoder = Decoder::with_ledger(ctx.options.clone(), inner, Arc::clone(&ctx.ledger));
        let frame = decode_item(ctx, &tile, decoder)?;
        // Each tile gets its own transforms before it is pasted at its new size.
        let frame = ctx.transformed_booked(&tile.transforms, frame)?;
        let mut guard = canvas.lock().unwrap_or_else(|e| e.into_inner());
        let r = paste(&mut guard, &frame, i, grid, ctx);
        drop(guard);
        let n = held_bytes(&frame);
        drop(frame);
        ctx.ledger.release(n);
        r
    };
    // Tile 0 decodes first, alone: it sizes the canvas and gives the output frame its
    // properties, and every other tile then decodes with the canvas booked, as in a serial
    // decode.
    put(ctx, 0)?;
    let rest = grid.tiles.len() - 1;
    let refused = AtomicUsize::new(usize::MAX);
    let r = run_each(rest, max_threads, |k| {
        let r = put(split, k + 1);
        if matches!(r, Err(Error::AllocLimit | Error::OutOfMemory)) {
            refused.fetch_min(k + 1, Ordering::Relaxed);
        }
        r
    });
    match r {
        // A tile refused while others decoded beside it, or while it read with a split item-read
        // share (`workers > 1`, even when it is the only tile after tile 0), may fit alone. As
        // `decode` does for a batch, the tiles from the lowest refused one decode again one at a time, each with only
        // the canvas booked beside it and the whole item-read share, as in a serial decode (a
        // tile that was pasted already is pasted again with the same samples), so the result is
        // the serial one. The error of `run_each` is the one of its lowest failing tile: a
        // refusal there is `refused`.
        Err(Error::AllocLimit | Error::OutOfMemory) if workers > 1 => {
            for i in refused.load(Ordering::Relaxed)..grid.tiles.len() {
                put(ctx, i)?;
            }
        }
        r => {
            r?;
        }
    }
    canvas
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .map(|c| c.frame)
        .ok_or(Error::Invalid("avif grid tile count"))
}

/// Pastes `frame`, tile `i` of `grid`, into the canvas; tile 0 makes it.
fn paste(
    canvas: &mut Option<Canvas>,
    frame: &Frame,
    i: usize,
    grid: &Grid,
    ctx: &Ctx<'_>,
) -> Result<(), Error> {
    if canvas.is_none() {
        *canvas = Some(Canvas::new(frame, grid, ctx.limits, &ctx.ledger)?);
    }
    let Some(c) = canvas.as_mut() else {
        return Err(Error::Invalid("avif grid tile count"));
    };
    let i = u32::try_from(i).map_err(|_| Error::Invalid("avif grid tile count"))?;
    c.paste(frame, i % grid.columns, i / grid.columns)
}

/// The output frame of a grid and the tile size it was sized for.
struct Canvas {
    frame: Frame,
    tile_w: u32,
    tile_h: u32,
}

/// The ledger bytes an item frame is booked for while the call holds it: its sample buffers
/// and the metadata its decoder booked. Its plane list (three `Plane` headers) is not booked; it
/// is part of the uncharged slack of each decoder.
fn held_bytes(f: &Frame) -> usize {
    frame_bytes(f).saturating_add(metadata_bytes(f))
}

/// The heap bytes of a frame's sample buffers: their capacity, as an output plane taken from
/// the pool can be larger than its samples.
pub(crate) fn frame_bytes(f: &Frame) -> usize {
    f.planes.iter().fold(0usize, |n, p| {
        n.saturating_add(match &p.data {
            PlaneData::U8(v) => v.capacity(),
            PlaneData::U16(v) => v.capacity().saturating_mul(2),
        })
    })
}

/// What a decoder books for the metadata OBUs of a frame.
fn metadata_bytes(f: &Frame) -> usize {
    f.metadata.iter().fold(0usize, |n, m| {
        n.saturating_add(m.payload.len() + size_of::<crate::Metadata>())
    })
}

fn zeroed<T: Clone + Default>(n: u64) -> Result<Vec<T>, Error> {
    let n = usize::try_from(n).map_err(|_| Error::AllocLimit)?;
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, T::default());
    Ok(v)
}

impl Canvas {
    /// Sizes the output for `grid` from its first decoded tile, checking the size against
    /// `limits` and the planes against `max_alloc` before allocating them.
    fn new(first: &Frame, grid: &Grid, limits: &Limits, ledger: &Ledger) -> Result<Canvas, Error> {
        let (w, h) = (grid.width, grid.height);
        let (tw, th) = (first.width, first.height);
        if w == 0 || h == 0 {
            return Err(Error::Invalid("avif grid size"));
        }
        if w > limits.max_width {
            return Err(Error::WidthLimit);
        }
        if h > limits.max_height {
            return Err(Error::HeightLimit);
        }
        if u64::from(w) * u64::from(h) > limits.max_pixels {
            return Err(Error::PixelLimit);
        }
        if u64::from(w) > u64::from(tw) * u64::from(grid.columns)
            || u64::from(h) > u64::from(th) * u64::from(grid.rows)
        {
            return Err(Error::Invalid("avif grid larger than its tiles"));
        }
        // Every tile must reach into the output (HEIF 6.6.2.3.1): the tiles decoded stay within
        // about 4x the output size.
        if u64::from(w) <= u64::from(tw) * u64::from(grid.columns - 1)
            || u64::from(h) <= u64::from(th) * u64::from(grid.rows - 1)
        {
            return Err(Error::Invalid("avif grid tile outside the output"));
        }
        let (sx, sy) = (
            u32::from(first.subsampling_x),
            u32::from(first.subsampling_y),
        );
        if !first.mono_chrome
            && ((grid.columns > 1 && sx == 1 && !tw.is_multiple_of(2))
                || (grid.rows > 1 && sy == 1 && !th.is_multiple_of(2)))
        {
            return Err(Error::Invalid("avif odd tile size"));
        }
        let count = if first.mono_chrome { 1 } else { 3 };
        let bytes_per_sample: u64 = if first.bit_depth > 8 { 2 } else { 1 };
        let dims: Vec<(u32, u32)> = (0..count)
            .map(|i| {
                let (dx, dy) = if i == 0 { (0, 0) } else { (sx, sy) };
                (w.div_ceil(1 << dx), h.div_ceil(1 << dy))
            })
            .collect();
        let total: u64 = dims
            .iter()
            .map(|&(pw, ph)| u64::from(pw) * u64::from(ph) * bytes_per_sample)
            .sum();
        // Booked for the rest of the call: the canvas is the image the caller gets.
        ledger.reserve(usize::try_from(total).map_err(|_| Error::AllocLimit)?)?;
        // The canvas holds a copy of the first tile's metadata.
        ledger.reserve(metadata_bytes(first))?;
        let mut planes = Vec::with_capacity(dims.len());
        for (width, height) in dims {
            let n = u64::from(width) * u64::from(height);
            let data = if first.bit_depth > 8 {
                PlaneData::U16(zeroed(n)?)
            } else {
                PlaneData::U8(zeroed(n)?)
            };
            planes.push(Plane {
                width,
                height,
                data,
            });
        }
        Ok(Canvas {
            frame: Frame {
                width: w,
                height: h,
                render_width: w,
                render_height: h,
                bit_depth: first.bit_depth,
                mono_chrome: first.mono_chrome,
                subsampling_x: first.subsampling_x,
                subsampling_y: first.subsampling_y,
                color: first.color,
                planes,
                metadata: first.metadata.clone(),
                film_grain_applied: first.film_grain_applied,
                frame_type: first.frame_type,
                temporal_id: first.temporal_id,
                spatial_id: first.spatial_id,
            },
            tile_w: tw,
            tile_h: th,
        })
    }

    /// Copies `tile` to tile position (`col`, `row`), clipped to the output.
    fn paste(&mut self, tile: &Frame, col: u32, row: u32) -> Result<(), Error> {
        let f = &mut self.frame;
        if (tile.width, tile.height) != (self.tile_w, self.tile_h)
            || tile.bit_depth != f.bit_depth
            || tile.mono_chrome != f.mono_chrome
            || tile.subsampling_x != f.subsampling_x
            || tile.subsampling_y != f.subsampling_y
        {
            return Err(Error::Invalid("avif tiles differ in size or format"));
        }
        let (sx, sy) = (u32::from(f.subsampling_x), u32::from(f.subsampling_y));
        for (i, (dst, src)) in f.planes.iter_mut().zip(&tile.planes).enumerate() {
            let (px, py) = if i == 0 { (0, 0) } else { (sx, sy) };
            let x0 = (u64::from(col) * u64::from(self.tile_w)) >> px;
            let y0 = (u64::from(row) * u64::from(self.tile_h)) >> py;
            let (x0, y0) = (
                usize::try_from(x0).map_err(|_| Error::AllocLimit)?,
                usize::try_from(y0).map_err(|_| Error::AllocLimit)?,
            );
            let (dw, sw) = (dst.width as usize, src.width as usize);
            let done = match (&mut dst.data, &src.data) {
                (PlaneData::U8(d), PlaneData::U8(s)) => blit(d, dw, s, sw, x0, y0),
                (PlaneData::U16(d), PlaneData::U16(s)) => blit(d, dw, s, sw, x0, y0),
                _ => None,
            };
            done.ok_or(Error::Invalid("avif tile plane mismatch"))?;
        }
        Ok(())
    }
}

/// Copies the rows of `src` (rows of `sw` samples) into `dst` (rows of `dw` samples) at
/// (`x0`, `y0`), dropping what falls outside `dst`. `None` when the buffers do not match their
/// widths.
fn blit<T: Copy>(
    dst: &mut [T],
    dw: usize,
    src: &[T],
    sw: usize,
    x0: usize,
    y0: usize,
) -> Option<()> {
    if dw == 0 || sw == 0 {
        return Some(());
    }
    let n = sw.min(dw.saturating_sub(x0));
    let rows = dst.len() / dw;
    for (r, row) in src.chunks_exact(sw).enumerate() {
        if y0 + r >= rows || n == 0 {
            break;
        }
        let start = (y0 + r) * dw + x0;
        dst.get_mut(start..start + n)?
            .copy_from_slice(row.get(..n)?);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorInfo;

    /// A 4:2:0 8-bit frame whose luma is `y` and whose chroma is `c` everywhere.
    fn frame(w: u32, h: u32, y: u8, c: u8) -> Frame {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let plane = |width: u32, height: u32, v: u8| Plane {
            width,
            height,
            data: PlaneData::U8(vec![v; (width * height) as usize]),
        };
        Frame {
            width: w,
            height: h,
            render_width: w,
            render_height: h,
            bit_depth: 8,
            mono_chrome: false,
            subsampling_x: true,
            subsampling_y: true,
            color: ColorInfo {
                color_primaries: 2,
                transfer_characteristics: 2,
                matrix_coefficients: 2,
                color_range: false,
                chroma_sample_position: 0,
            },
            planes: vec![plane(w, h, y), plane(cw, ch, c), plane(cw, ch, c)],
            metadata: Vec::new(),
            film_grain_applied: false,
            frame_type: 0,
            temporal_id: 0,
            spatial_id: 0,
        }
    }

    fn grid(columns: u32, rows: u32, width: u32, height: u32) -> Grid {
        Grid {
            rows,
            columns,
            width,
            height,
            tiles: vec![0; (rows * columns) as usize],
        }
    }

    fn led(l: &Limits) -> Ledger {
        let ledger = Ledger::default();
        let _ = ledger.reset(l.max_alloc, 0);
        ledger
    }

    fn bytes(p: &Plane) -> &[u8] {
        match &p.data {
            PlaneData::U8(v) => v,
            PlaneData::U16(_) => &[],
        }
    }

    #[test]
    fn grid_of_two_tiles_is_cropped() {
        let (a, b) = (frame(4, 2, 1, 10), frame(4, 2, 2, 20));
        // Two 4x2 tiles side by side, output cropped from 8x2 to 7x2.
        let mut canvas = Canvas::new(
            &a,
            &grid(2, 1, 7, 2),
            &Limits::default(),
            &led(&Limits::default()),
        )
        .unwrap();
        canvas.paste(&a, 0, 0).unwrap();
        canvas.paste(&b, 1, 0).unwrap();
        let f = canvas.frame;
        assert_eq!((f.width, f.height), (7, 2));
        assert_eq!(
            bytes(&f.planes[0]),
            &[1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2]
        );
        assert_eq!((f.planes[1].width, f.planes[1].height), (4, 1));
        assert_eq!(bytes(&f.planes[1]), &[10, 10, 20, 20]);
        assert_eq!(bytes(&f.planes[2]), &[10, 10, 20, 20]);
    }

    #[test]
    fn mismatched_tiles_are_an_error() {
        let a = frame(4, 2, 1, 10);
        let mut canvas = Canvas::new(
            &a,
            &grid(2, 1, 8, 2),
            &Limits::default(),
            &led(&Limits::default()),
        )
        .unwrap();
        canvas.paste(&a, 0, 0).unwrap();
        let narrow = frame(2, 2, 2, 20);
        assert!(matches!(
            canvas.paste(&narrow, 1, 0),
            Err(Error::Invalid(_))
        ));
        let mut deep = frame(4, 2, 2, 20);
        deep.bit_depth = 10;
        assert!(matches!(canvas.paste(&deep, 1, 0), Err(Error::Invalid(_))));
    }

    #[test]
    fn grid_size_is_checked() {
        let a = frame(4, 2, 1, 10);
        let l = Limits::default();
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 9, 2), &l, &led(&l)),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 0, 2), &l, &led(&l)),
            Err(Error::Invalid(_))
        ));
        // The second tile would lie wholly outside a 4 wide output.
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 4, 2), &l, &led(&l)),
            Err(Error::Invalid(_))
        ));
        let small = Limits { max_alloc: 8, ..l };
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 8, 2), &small, &led(&small)),
            Err(Error::AllocLimit)
        ));
        let narrow = Limits { max_width: 7, ..l };
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 8, 2), &narrow, &led(&narrow)),
            Err(Error::WidthLimit)
        ));
    }

    /// The 4:2:0 test frame (4x2) with luma 1..=8.
    fn numbered() -> Frame {
        let mut f = frame(4, 2, 0, 10);
        f.planes[0].data = PlaneData::U8((1..=8).collect());
        f
    }

    fn transform(
        options: &Options,
        limits: &Limits,
        cancel: bool,
        ts: &[Transform],
    ) -> Result<Frame, Error> {
        let cancel = AtomicBool::new(cancel);
        let ledger = led(limits);
        let f = numbered();
        ledger.reserve(frame_bytes(&f))?;
        transform_frame(options, limits, &ledger, &cancel, ts, f)
    }

    #[test]
    fn transform_books_its_budget_and_then_only_its_output() {
        let p = size_of::<isobmff::Plane>();
        let quarter = [Transform::Rotate(90), Transform::Rotate(270)];
        assert_eq!(
            transform_budget(100, 3, Chroma::Yuv422, &quarter),
            2 * (151 + 3 * p)
        );
        let mirror = [Transform::Mirror(isobmff::Axis::LeftRight)];
        assert_eq!(transform_budget(100, 3, Chroma::Yuv420, &mirror), 0);
        let rot = [Transform::Rotate(90)];
        let input = frame_bytes(&numbered());
        let budget = transform_budget(input, 3, Chroma::Yuv420, &rot);
        assert_eq!(budget, input + 3 * p);
        let cancel = AtomicBool::new(false);
        for (room, ok) in [(input + budget, true), (input + budget - 1, false)] {
            let ledger = Ledger::default();
            ledger.reset(crate::ledger::SMALL + room, 0).unwrap();
            ledger.reserve(input).unwrap();
            let r = transform_frame(&ON, &Limits::default(), &ledger, &cancel, &rot, numbered());
            assert_eq!(r.is_ok(), ok, "room {room}");
            // The output stays booked in place of the input; a refusal gives the input back.
            let held = r.as_ref().map_or(0, frame_bytes);
            assert_eq!(ledger.left(), room - held, "room {room}");
        }
    }

    // A grid tile comes booked with its metadata: a refused transform gives all of it back, so a
    // refused tile leaves nothing booked for the serial retry.
    #[test]
    fn refused_transform_gives_back_the_metadata_booking() {
        let mut f = numbered();
        f.metadata.push(crate::Metadata {
            metadata_type: 4,
            payload: vec![0; 100],
        });
        let held = held_bytes(&f);
        let ledger = Ledger::default();
        ledger.reset(crate::ledger::SMALL + held, 0).unwrap();
        ledger.reserve(held).unwrap();
        let cancel = AtomicBool::new(false);
        let rot = [Transform::Rotate(90)];
        let r = transform_frame(&ON, &Limits::default(), &ledger, &cancel, &rot, f);
        assert_eq!(r, Err(Error::AllocLimit));
        assert_eq!(ledger.left(), held);
    }

    const ON: Options = Options {
        operating_point: 0,
        all_layers: false,
        film_grain: true,
        keep_metadata: true,
        apply_transforms: true,
    };

    #[test]
    fn transforms_are_raw_by_default() {
        let ts = [
            Transform::Rotate(90),
            Transform::Mirror(isobmff::Axis::LeftRight),
        ];
        let l = Limits::default();
        let out = transform(&Options::default(), &l, false, &ts).unwrap();
        assert_eq!(out, numbered());
    }

    #[test]
    fn transforms_apply_when_asked() {
        let l = Limits::default();
        let mirror = [Transform::Mirror(isobmff::Axis::LeftRight)];
        let f = transform(&ON, &l, false, &mirror).unwrap();
        assert_eq!(bytes(&f.planes[0]), &[4, 3, 2, 1, 8, 7, 6, 5]);
        let f = transform(&ON, &l, false, &[Transform::Rotate(90)]).unwrap();
        assert_eq!((f.width, f.height), (2, 4));
        assert_eq!((f.render_width, f.render_height), (2, 4));
        assert_eq!(bytes(&f.planes[0]), &[4, 8, 3, 7, 2, 6, 1, 5]);
        assert_eq!((f.planes[1].width, f.planes[1].height), (1, 2));
    }

    #[test]
    fn transform_obeys_the_limits_and_cancel() {
        let rot = [Transform::Rotate(90)];
        let small = Limits {
            max_alloc: 8,
            ..Limits::default()
        };
        assert!(matches!(
            transform(&ON, &small, false, &rot),
            Err(Error::AllocLimit)
        ));
        let narrow = Limits {
            max_width: 1,
            ..Limits::default()
        };
        assert!(matches!(
            transform(&ON, &narrow, false, &rot),
            Err(Error::WidthLimit)
        ));
        assert!(matches!(
            transform(&ON, &Limits::default(), true, &rot),
            Err(Error::Cancelled)
        ));
    }
}
