//! The threaded frame decode on the shared rayon pool (`rayon` feature, docs/threading.md).
//!
//! The lead is plain sequential code on the calling thread; each stage is one
//! [`run_parallel`] over its items, so nothing holds the frame behind a lock. The filter stages
//! write straight into disjoint row views of their output planes ([`split_rows`], no band copy
//! and no copy back); the one exception is the horizontal loop filter edges, whose column
//! strips are not contiguous memory (`lf_cols_strip`). The order of the stages and of the
//! allocations is that of the serial `wrapup`.

use super::*;
use crate::lr::LrPlane;

/// A view an item of a stage takes (once) from its slot.
type Slot<T> = Mutex<Option<T>>;

/// The threaded counterpart of the serial branch of `decode_frame`: the tiles of `tiles` into
/// `fs`, then the wrapup of 7.4 on `threads` threads, with the same result for every `threads`.
pub(super) fn decode_frame(
    mut fs: FrameState,
    ledger: &Arc<Ledger>,
    threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    data: &[u8],
    tiles: &[TileRange],
    cancel: &AtomicBool,
    frame_cdf: Box<CdfContext>,
) -> Result<FrameState, Error> {
    let tile_cols = h.tile_info.tile_cols.max(1);
    if h.allow_intrabc || tiles.len() <= 1 {
        // Intra block copy reads other tiles: the tiles decode in order into the frame state.
        for t in tiles {
            let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
            decode_tile(&mut fs, &frame_cdf, data, t, row, col, h, seq, cancel)?;
        }
    } else {
        let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
        let merged = Mutex::new(&mut fs);
        let r = run_parallel(tiles.len(), threads, |i| {
            let t = tiles
                .get(i)
                .ok_or(Error::Invalid("tile index out of range"))?;
            let (row, col) = (t.tile_num / tile_cols, t.tile_num % tile_cols);
            let mut local = FrameState::new_tile(h, seq, ledger, row, col)?;
            decode_tile(&mut local, &frame_cdf, data, t, 0, 0, h, seq, cancel)?;
            lock(&merged).merge_tile(&local, sb_size);
            Ok(())
        });
        drop(merged);
        r?;
    }
    let mut curr = std::mem::take(&mut fs.curr_frame);
    let planes = filters(&fs, ledger, threads, h, seq, cancel, &mut curr)?;
    fs.curr_frame = curr;
    fs.lr_frame = planes;
    fs.frame_cdf = Some(frame_cdf);
    Ok(fs)
}

/// Loop filter, CDEF, superres and loop restoration of `curr` (`CurrFrame`, filtered in place);
/// returns `LrFrame`.
fn filters(
    fs: &FrameState,
    ledger: &Arc<Ledger>,
    threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    cancel: &AtomicBool,
    curr: &mut Vec<WorkPlane>,
) -> Result<Vec<WorkPlane>, Error> {
    let bands = fs.mi_rows.div_ceil(16);
    if lf_on(h) {
        lf_rows(fs, ledger, threads, h, cancel, curr)?;
        let strips = threads.min(fs.mi_cols.div_ceil(16)).max(1);
        let planes = RwLock::new(std::mem::take(curr));
        let r = run_parallel(fs.num_planes * strips, threads, |i| {
            lf_cols_strip(fs, h, ledger, cancel, &planes, strips, i)
        });
        *curr = planes.into_inner().unwrap_or_else(|e| e.into_inner());
        r?;
    }
    let curr: &[WorkPlane] = curr;
    // `CdefFrame` starts as a copy of `CurrFrame`, which it stays with CDEF off.
    let mut cdef = cdef_planes(curr, ledger)?;
    if cdef_on(fs, curr, seq, h, &cdef) {
        let num_planes = usize::from(seq.num_planes).min(curr.len());
        let _held = charge_views(ledger, num_planes * bands)?;
        let mut per_plane = Vec::new();
        for (plane, p) in cdef.iter_mut().enumerate().take(num_planes) {
            let (_, sy) = subsampling(fs, plane);
            let height = p.height;
            let ranges = (0..bands).map(|b| Some(band_rows(b, bands, sy, height)));
            per_plane.push(split_rows(p, ranges)?.into_iter());
        }
        // One slot per band: the views of that band of every plane.
        let slots: Vec<Slot<Vec<View<'_>>>> = (0..bands)
            .map(|_| {
                let band = per_plane.iter_mut().filter_map(Iterator::next).collect();
                Mutex::new(Some(band))
            })
            .collect();
        run_parallel(bands, threads, |b| {
            let Some(mut views) = take(&slots, b) else {
                return Ok(());
            };
            let rows = 16 * b..(16 * b + 16).min(fs.mi_rows);
            cdef_rows(fs, curr, seq, h, rows, &mut views, cancel)
        })?;
    }
    if h.frame_width == h.upscaled_width {
        // No superres: loop restoration reads `CdefFrame` and `CurrFrame` as they are.
        let mut out = exact_planes(fs, Some(cdef.as_slice()))?;
        if h.uses_lr {
            lr_stage(
                fs,
                ledger,
                threads,
                h,
                seq,
                cancel,
                [cdef.as_slice(), curr],
                &mut out,
            )?;
        }
        fs.ledger.put_planes(cdef);
        return Ok(out);
    }
    let mut up_cdef = exact_planes(fs, None)?;
    upscale_stage(
        fs,
        ledger,
        threads,
        h,
        seq,
        cancel,
        cdef.as_slice(),
        &mut up_cdef,
    )?;
    fs.ledger.put_planes(cdef);
    if !h.uses_lr {
        return Ok(up_cdef);
    }
    let mut up_curr = exact_planes(fs, None)?;
    upscale_stage(fs, ledger, threads, h, seq, cancel, curr, &mut up_curr)?;
    let mut out = exact_planes(fs, Some(up_cdef.as_slice()))?;
    lr_stage(
        fs,
        ledger,
        threads,
        h,
        seq,
        cancel,
        [up_cdef.as_slice(), up_curr.as_slice()],
        &mut out,
    )?;
    fs.ledger.put_planes(up_cdef);
    fs.ledger.put_planes(up_curr);
    Ok(out)
}

/// Vertical edges: item `plane * bands + b` filters the rows of band `b` of `plane` (16 mode
/// info rows; the last band runs to the plane end), in place.
fn lf_rows(
    fs: &FrameState,
    ledger: &Arc<Ledger>,
    threads: usize,
    h: &FrameHeader,
    cancel: &AtomicBool,
    curr: &mut [WorkPlane],
) -> Result<(), Error> {
    let bands = fs.mi_rows.div_ceil(16);
    let n = fs.num_planes * bands;
    let _held = charge_views(ledger, n)?;
    let mut views = Vec::new();
    views.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    for (plane, p) in curr.iter_mut().enumerate().take(fs.num_planes) {
        if lf_plane_on(h, plane) {
            let (_, sy) = subsampling(fs, plane);
            let height = p.height;
            let ranges = (0..bands).map(|b| Some(band_rows(b, bands, sy, height)));
            views.extend(split_rows(p, ranges)?.into_iter().map(Some));
        } else {
            views.extend((0..bands).map(|_| None));
        }
    }
    let slots: Vec<Slot<View<'_>>> = views.into_iter().map(Mutex::new).collect();
    run_parallel(slots.len(), threads, |i| {
        let Some(mut v) = take(&slots, i) else {
            return Ok(());
        };
        let (plane, b) = (i / bands, i % bands);
        let rows = 16 * b..(16 * b + 16).min(fs.mi_rows);
        loop_filter_rows(fs, h, plane, 0, rows, 0..fs.mi_cols, &mut v, cancel)
    })?;
    Ok(())
}

/// Superres from `src` into the exact-size planes `dst`: item `plane * bands + b` is the 64-row
/// band `b` of `plane`.
fn upscale_stage(
    fs: &FrameState,
    ledger: &Arc<Ledger>,
    threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    cancel: &AtomicBool,
    src: &[WorkPlane],
    dst: &mut [WorkPlane],
) -> Result<(), Error> {
    let bands = fs.frame_height.div_ceil(64);
    let num_planes = fs.num_planes.min(src.len());
    let _held = charge_views(ledger, num_planes * bands)?;
    let mut views = Vec::new();
    views
        .try_reserve_exact(num_planes * bands)
        .map_err(|_| Error::OutOfMemory)?;
    for (plane, p) in dst.iter_mut().enumerate().take(num_planes) {
        let (_, ht) = exact_size(fs, plane);
        let ranges = (0..bands).map(|b| Some(((64 * b).min(ht), (64 * b + 64).min(ht))));
        views.extend(split_rows(p, ranges)?);
    }
    let slots: Vec<Slot<View<'_>>> = views.into_iter().map(|v| Mutex::new(Some(v))).collect();
    run_parallel(slots.len(), threads, |i| {
        let Some(mut v) = take(&slots, i) else {
            return Ok(());
        };
        let (plane, b) = (i / bands, i % bands);
        let (w, ht) = exact_size(fs, plane);
        let (y0, y1) = ((64 * b).min(ht), (64 * b + 64).min(ht));
        if y0 == y1 {
            return Ok(());
        }
        let input = src.get(plane).ok_or(Error::Invalid("plane missing"))?;
        upscale_rows(input, plane, seq, h, (w, ht), y0..y1, &mut v, cancel)
    })?;
    Ok(())
}

/// Loop restoration of `[ UpscaledCdefFrame, UpscaledCurrFrame ]` into `out` (which starts as
/// the first): item `plane * stripes + n` is stripe `n` of `plane`. The stripes of a plane cover
/// its rows once each, in order, so their views are disjoint.
fn lr_stage(
    fs: &FrameState,
    ledger: &Arc<Ledger>,
    threads: usize,
    h: &FrameHeader,
    seq: &SequenceHeader,
    cancel: &AtomicBool,
    [cdef, curr]: [&[WorkPlane]; 2],
    out: &mut [WorkPlane],
) -> Result<(), Error> {
    let ns = stripes(fs);
    let num_planes = fs.num_planes.min(3).min(cdef.len()).min(curr.len());
    let _held = charge_views(ledger, num_planes * ns)?;
    let mut planes: Vec<Option<LrPlane>> = Vec::new();
    let mut views = Vec::new();
    views
        .try_reserve_exact(num_planes * ns)
        .map_err(|_| Error::OutOfMemory)?;
    for (plane, p) in out.iter_mut().enumerate().take(num_planes) {
        let (Some(c), Some(u)) = (cdef.get(plane), curr.get(plane)) else {
            return Err(Error::Invalid("plane missing"));
        };
        let g = lr_plane(fs, seq, h, plane, [c, u], exact_size(fs, plane));
        let ranges = (0..ns).map(|n| {
            let (y0, y1) = g.as_ref()?.stripe_rows(n)?;
            Some((y0, y1 + 1))
        });
        views.extend(split_rows(p, ranges)?);
        planes.push(g);
    }
    let slots: Vec<Slot<View<'_>>> = views.into_iter().map(|v| Mutex::new(Some(v))).collect();
    run_parallel(slots.len(), threads, |i| {
        check_cancel(cancel)?;
        let (plane, n) = (i / ns, i % ns);
        let Some(mut v) = take(&slots, i) else {
            return Ok(());
        };
        if let (Some(Some(g)), Some(c), Some(u)) =
            (planes.get(plane), cdef.get(plane), curr.get(plane))
        {
            lr_stripe(fs, seq, g, n, c, u, &mut v);
        }
        Ok(())
    })?;
    Ok(())
}

/// The rows `y0..y1` of each range of `p`, as views in range order (`None`: no rows, at the end
/// of the one before). The ranges ascend and do not overlap; rows between them are left out.
fn split_rows<'a>(
    p: &'a mut WorkPlane,
    ranges: impl ExactSizeIterator<Item = Option<(usize, usize)>>,
) -> Result<Vec<View<'a>>, Error> {
    let stride = p.stride;
    let mut views = Vec::new();
    views
        .try_reserve_exact(ranges.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut rest = p.px.as_mut_slice();
    // The plane row `rest` starts at.
    let mut at = 0;
    for r in ranges {
        let (y0, y1) = r.unwrap_or((at, at));
        let skip = y0
            .checked_sub(at)
            .ok_or(Error::Invalid("loop filter rows overlap"))?;
        let rows = y1
            .checked_sub(y0)
            .ok_or(Error::Invalid("loop filter rows reversed"))?;
        let (_, tail) = std::mem::take(&mut rest)
            .split_at_mut_checked(skip.saturating_mul(stride))
            .ok_or(Error::Invalid("plane smaller than its size"))?;
        let (px, tail) = tail
            .split_at_mut_checked(rows.saturating_mul(stride))
            .ok_or(Error::Invalid("plane smaller than its size"))?;
        rest = tail;
        at = y1;
        views.push(View {
            px,
            stride,
            x0: 0,
            y0,
        });
    }
    Ok(views)
}

/// Books the slots of `n` views in the ledger until the guard drops.
fn charge_views(ledger: &Ledger, n: usize) -> Result<impl Sized + '_, Error> {
    ledger.charge(n.saturating_mul(size_of::<Slot<View<'static>>>()))
}

/// Takes the value of slot `i`, once.
fn take<T>(slots: &[Slot<T>], i: usize) -> Option<T> {
    lock(slots.get(i)?).take()
}
