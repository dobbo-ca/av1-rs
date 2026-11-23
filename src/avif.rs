//! The AVIF layer: the item graph comes from `isobmff`, the pixels from [`Decoder`].
//!
//! Coded items are decoded with their own decoder, grid tiles on up to `max_threads` threads; tiles are pasted into
//! the output as they are decoded. Transforms, colour and the alpha plane are passed through
//! raw: nothing is applied.

use crate::ledger::Ledger;
use crate::parallel::run_parallel;
use crate::{Decoder, Error, Format, Frame, Limits, Options, Plane, PlaneData};
use isobmff::{Av1c, Codec, CodecConfig, Grid, Heif, Image, ImageKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A decoded AVIF still.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvifImage {
    /// The primary image; a grid is assembled and cropped to its output size.
    pub image: Frame,
    /// The alpha auxiliary image, decoded the same way.
    pub alpha: Option<Frame>,
    /// The transformative properties of the primary image (`clap`, `irot`, `imir`), raw, in
    /// `ipma` order. Not applied to `image`.
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
    cancel: &'a AtomicBool,
}

/// Decodes the primary image of an AVIF file, and its alpha image when there is one.
///
/// A coded item is decoded as one Section 5 temporal unit: the `av1C` configOBUs followed by the
/// item data. A grid is assembled from its tiles; an identity image is its input. Overlays and
/// other item types are [`Error::Unsupported`]. The frame size limits of `limits` apply to every
/// decoded frame and to the assembled grid, and the assembled grid is charged against
/// `max_alloc`; each item decode gets `limits` whole. `cancel` is checked between tiles and
/// inside every decode.
///
/// # Errors
/// [`Error::Container`] (the class of the `isobmff` error) for a broken file, otherwise the
/// errors of [`Decoder::decode_temporal_unit`], and `Invalid` for tiles that do not fit together.
pub fn decode_avif(
    file: &[u8],
    options: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<AvifImage, Error> {
    let ilimits = isobmff::Limits {
        max_alloc: limits.max_alloc,
        max_threads: limits.max_threads,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let heif = isobmff::parse(file, &ilimits, cancel).map_err(container)?;
    let ctx = Ctx {
        heif: &heif,
        file,
        options,
        limits,
        ilimits: &ilimits,
        cancel,
    };
    let primary = heif.primary_image().map_err(container)?;
    let image = decode_image(&ctx, &primary, false)?;
    let alpha = match primary.alpha {
        Some(id) => {
            let item = heif.image(id).map_err(container)?;
            Some(decode_image(&ctx, &item, false)?)
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

fn decode_image(ctx: &Ctx<'_>, image: &Image, nested: bool) -> Result<Frame, Error> {
    match &image.kind {
        ImageKind::Coded { .. } => decode_coded(ctx, image),
        ImageKind::Grid(grid) => decode_grid(ctx, grid),
        // The input of an identity image is a coded image or a grid, not another identity.
        ImageKind::Identity { input } if !nested => {
            let inner = ctx.heif.image(*input).map_err(container)?;
            decode_image(ctx, &inner, true)
        }
        ImageKind::Identity { .. } => Err(Error::Unsupported("avif nested identity")),
        ImageKind::Overlay(_) => Err(Error::Unsupported("avif overlay")),
        ImageKind::Unsupported(_) => Err(Error::Unsupported("avif item")),
    }
}

/// One coded AV1 item: configOBUs + item data, decoded by a fresh decoder; the last output
/// frame is the image.
fn decode_coded(ctx: &Ctx<'_>, image: &Image) -> Result<Frame, Error> {
    decode_item(ctx, image, Decoder::new(ctx.options.clone(), *ctx.limits))
}

/// [`decode_coded`] with the decoder given (a grid tile shares its ledger with the others).
fn decode_item(ctx: &Ctx<'_>, image: &Image, mut decoder: Decoder) -> Result<Frame, Error> {
    let ImageKind::Coded {
        codec: Codec::Av1,
        config: CodecConfig::Av1(av1c),
    } = &image.kind
    else {
        return Err(Error::Unsupported("avif codec"));
    };
    let unit = temporal_unit(ctx, image.id, av1c)?;
    decoder
        .decode_temporal_unit(&unit, Format::Section5, ctx.cancel)?
        .pop()
        .ok_or(Error::Invalid("avif item has no shown frame"))
}

/// The Section 5 stream of one item: the `av1C` configOBUs, then the item data.
fn temporal_unit(ctx: &Ctx<'_>, id: u32, av1c: &Av1c) -> Result<Vec<u8>, Error> {
    let config = Heif::slice(ctx.file, av1c.config_obus).ok_or(Error::Truncated)?;
    let data = ctx
        .heif
        .item_bytes(ctx.file, id, ctx.ilimits, ctx.cancel)
        .map_err(container)?;
    let total = config
        .len()
        .checked_add(data.len())
        .ok_or(Error::AllocLimit)?;
    if total > ctx.limits.max_alloc {
        return Err(Error::AllocLimit);
    }
    let mut unit = Vec::new();
    unit.try_reserve_exact(total)
        .map_err(|_| Error::OutOfMemory)?;
    unit.extend_from_slice(config);
    unit.extend_from_slice(&data);
    Ok(unit)
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
    let inner = Limits {
        max_threads: (max_threads / workers).max(1),
        ..*ctx.limits
    };
    let ledger = Arc::new(Ledger::default());
    // A `max_alloc` below the small-allocation allowance only means that every charge fails.
    let _ = ledger.reset(ctx.limits.max_alloc, 0);
    let canvas: Mutex<Option<Canvas>> = Mutex::new(None);
    run_parallel(grid.tiles.len(), max_threads, |i| {
        if ctx.cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let id = grid
            .tiles
            .get(i)
            .ok_or(Error::Invalid("avif grid tile count"))?;
        let tile = ctx.heif.image(*id).map_err(container)?;
        let decoder = Decoder::with_ledger(ctx.options.clone(), inner, Arc::clone(&ledger));
        let frame = decode_item(ctx, &tile, decoder)?;
        let i = u32::try_from(i).map_err(|_| Error::Invalid("avif grid tile count"))?;
        let mut guard = canvas.lock().unwrap_or_else(|e| e.into_inner());
        // The canvas is sized by whichever tile finishes first; tile 0 then sets the properties
        // of the output frame, as in a serial decode.
        if guard.is_none() {
            let mut c = Canvas::new(&frame, grid, ctx.limits)?;
            c.first = i;
            *guard = Some(c);
        }
        let Some(c) = guard.as_mut() else {
            return Err(Error::Invalid("avif grid tile count"));
        };
        if i == 0 && c.first != 0 {
            c.take_properties(&frame);
        }
        c.paste(&frame, i % grid.columns, i / grid.columns)
    })?;
    canvas
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .map(|c| c.frame)
        .ok_or(Error::Invalid("avif grid tile count"))
}

/// The output frame of a grid and the tile size it was sized for.
struct Canvas {
    frame: Frame,
    tile_w: u32,
    tile_h: u32,
    /// Index of the tile the canvas was sized from (0 in a serial decode).
    first: u32,
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
    fn new(first: &Frame, grid: &Grid, limits: &Limits) -> Result<Canvas, Error> {
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
        if total > limits.max_alloc as u64 {
            return Err(Error::AllocLimit);
        }
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
            first: 0,
        })
    }

    /// Takes the output frame's properties (everything but the size, format and samples) from
    /// `tile`, tile 0 of the grid.
    fn take_properties(&mut self, tile: &Frame) {
        let f = &mut self.frame;
        f.color = tile.color;
        f.metadata = tile.metadata.clone();
        f.film_grain_applied = tile.film_grain_applied;
        f.frame_type = tile.frame_type;
        f.temporal_id = tile.temporal_id;
        f.spatial_id = tile.spatial_id;
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
        let mut canvas = Canvas::new(&a, &grid(2, 1, 7, 2), &Limits::default()).unwrap();
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
        let mut canvas = Canvas::new(&a, &grid(2, 1, 8, 2), &Limits::default()).unwrap();
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
            Canvas::new(&a, &grid(2, 1, 9, 2), &l),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 0, 2), &l),
            Err(Error::Invalid(_))
        ));
        let small = Limits { max_alloc: 8, ..l };
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 8, 2), &small),
            Err(Error::AllocLimit)
        ));
        let narrow = Limits { max_width: 7, ..l };
        assert!(matches!(
            Canvas::new(&a, &grid(2, 1, 8, 2), &narrow),
            Err(Error::WidthLimit)
        ));
    }
}
