# Limits

`Limits` (src/limits.rs) caps one call; every public entry point takes it.

In short (moved from the README): `max_alloc` is the enforced peak working memory of one decoder
or one `decode_avif` call, all threads included: frame buffers, film grain buffers and the output
frames of a temporal unit (and every frame `decode` collects) are charged before allocation and
released on drop. Not counted: frames already returned by `Decoder` (buffers handed back with
`Decoder::recycle` are charged again while the decoder keeps them, and dropped when they do not
fit), the item bytes inside `isobmff` (an eighth of `max_alloc` for the parse and one for the
reads), and the stacks of the up to `max_threads - 1` threads a call spawns (std default size);
`max_metadata` is on top. `max_threads = 1` spawns no thread; otherwise the tiles and in-loop
filters of a frame (one scope of threads per frame, no pool kept between calls) and the tiles of
an AVIF grid decode on up to `max_threads` threads, and film grain runs in bands of 32 rows on up
to `max_threads` threads, with the same output. `decode` and `Decoder::decode_temporal_units`
decode the frames of several temporal units at once (up to `max_threads` frames, each with
`max_threads` divided by that many threads), and charge their output until they return.

- `max_width`, `max_height`, `max_pixels`: checked on the frame size of each frame header (and
  the sequence header maximums) before any frame buffer is allocated.
- `max_alloc`: a ledger (src/ledger.rs) charges every buffer that scales with the input (frame
  planes, reference slots, per-block state, output planes) before it is allocated and releases
  it on drop. Allocation uses `try_reserve_exact`; a failure is `Error::OutOfMemory`.
- `max_metadata`: the total payload of the metadata OBUs kept with a frame.
- `max_threads`: the most threads of one call, the calling thread included. The tiles and the
  in-loop filters of a frame and the tiles of an AVIF grid run in parallel within it
  (docs/threading.md); 1 spawns no thread, and a frame with `allow_intrabc` decodes its tiles
  serially.


Charged per frame, all before allocation and released when the frame state drops: `CurrFrame`
(planes rounded up to 128, `u16` samples); the mode info arrays of `FrameState` (about 20 bytes
per 4x4 block plus palette colours when screen content tools are on, and the loop restoration
unit arrays); per tile, the above context arrays (`mi_cols` rounded to 32 per plane); with more than one
thread and tile, per worker one tile-sized state (the same arrays and planes for the tile's
rectangle, plus the loop restoration unit arrays of the frame), all in the same ledger; at the
wrapup, `CdefFrame` (same size as `CurrFrame`), the upscaled planes (exact size) when superres
is on, and the `LrFrame` planes (exact size); with more than one thread, per running filter item
its private band (rows or a column strip of a plane, docs/threading.md). The peak is
`CurrFrame` + `CdefFrame` + up to two exact-size planes sets + the arrays (+ the bands of the
running filter items); a frame over `max_alloc` fails with `AllocLimit` before the
tile decode starts or at the wrapup, never after exceeding it. Film grain charges, per band of 32
luma rows while the band runs, its noise stripe and the one above it (34 rows of about `width + 34`
`i32` per plane each) plus one row: at most one band per worker. Fixed-size tile state (the CDF
copy, the colour maps, the coefficient and level buffers, about 28 KB per tile decoder) is not
charged. The output planes are charged
from when they are built until the
temporal unit returns; `decode` keeps every frame it collects charged (for the capacity of its
planes, which may be pooled buffers larger than the samples, plus a fixed per-frame overhead for
the `Frame`, its plane list and allocator chunks) until it returns. AVIF: one
ledger per `decode_avif` call covers every item decoder, the grid canvas, the lists each held
`isobmff::Image` owns (a grid's tile ids, 4 bytes a tile; thumbnail, Exif, XMP and auxiliary
references; transforms; colour properties), the tiles scheduled with no per-tile state, the item copy, every
decoded tile until it is pasted, the transforms (output and scratch, booked before they run) and
the primary image (docs/threading.md); it gets three quarters of `max_alloc`, and the `isobmff`
parse and item reads an eighth each, so the three sum to `max_alloc`, not twice it
(`tests/avif_alloc.rs` counts the heap of every thread for small items and grids, a grid of
two-extent tiles, and an item and a grid with transforms larger than the uncharged slack, at
many `max_alloc` and `max_threads` values, and checks the peak against the ledger's share plus
what the parse and the item copies held and the uncharged fixed buffers of each decoder; it
checks `Decoder::decode_temporal_units` at 2 and 4 threads against `max_alloc` plus those
buffers).

Not charged, because their size does not depend on the input: per tile decoder, the colour map
buffers and the 17.25 KiB reconstruction scratch (`recon::ItxScratch`, with the intra edge
arrays). The loop restoration source
window and the kernels' working arrays (about 100 KB, `lr::LrScratch`) are charged: one set per
`lr_frame` call, or per stripe item when threaded, taken from the ledger pool and not cleared
between blocks.

## Batches

`Decoder::decode_temporal_units` (and so `decode`) keeps the output frames of all its units
charged until it returns (`decode` always did). With `max_threads` above 1 it also has up to
`W` frames decoding at once, each holding its own `FrameState`, `CurrFrame`, `CdefFrame` and
`LrFrame`, so the ledger peak is about `W` times that of one frame (`W x threads <= max_threads`,
docs/threading.md). `W` is lowered until `W` times an estimate of one frame (three sets of
16-bit planes of the padded luma area times three, with a margin) fits in what is left of
`max_alloc`; if a buffer is refused anyway, the whole group is dropped and decoded one frame at a
time. A stream of large frames therefore runs one frame at a time when `max_alloc` is small, and
the output limit applies as before: pass a few units at a time (the oracle passes `max_threads`
units) to stream a long file. The slots a frame refreshes are emptied when the frame is
committed, so in a batch the planes they held stay charged while the group decodes.

## Buffer reuse

A `Decoder` keeps its buffers from one frame to the next, so a stream of frames of one shape
stops allocating after the first frames:

- **The ledger's pool.** A freed sample plane (`CurrFrame`, `CdefFrame`, the upscaled and
  `LrFrame` planes, film grain planes, a tile state's planes), a film grain stripe or row
  buffer, an output plane handed back with `Decoder::recycle` and an empty plane list go to a
  pool in the ledger instead of the allocator. The next request takes the smallest pooled buffer
  that is large enough. A pooled buffer stays charged, once, at the size it was charged when it
  was allocated (or, for a recycled output plane, its capacity); it is still memory the decoder
  holds. When a charge does not fit in `max_alloc`, the ledger first drops every pooled buffer
  and releases its bytes, then tries again, so the pool never makes a decode fail that fits
  without it.
- **The frame state.** After a frame, its `FrameState` (the mode-info and loop restoration
  arrays, the tile decoder's context arrays, CDF copy and colour maps, the frame CDFs) waits for
  the next frame; its planes go to the pool. A next frame of the same shape (frame and
  mode-info size, planes, subsampling, screen content and intra block copy flags, loop
  restoration units) clears and reuses it with its charge unchanged; another shape drops it
  before allocating. These arrays (about 20 bytes per 4x4 block, plus the context arrays) are
  the one thing the decoder now holds that it did not before: they count against `max_alloc`
  from the end of a frame (output and film grain included) until the next frame takes them or
  an error drops them.
- **Reference slots.** The slots a frame refreshes are emptied before its tiles decode, as
  dav1d does (a frame that fails leaves them empty), so the planes of a frame that no slot keeps
  any more are free for the new one, and its `Arc` is reused for the new frame.
- **Output frames.** `Decoder::recycle(frames)` takes back returned frames: their sample buffers
  go to the pool (charged; a buffer that does not fit is dropped instead), their plane lists and
  the `Vec` itself are reused for the next frames. It cannot fail.
- **Per block, per transform, per superblock** the decode allocates nothing: the block decode
  (coefficients, transforms, prediction) works in fixed-size buffers, the loop restoration
  windows and filter buffers come from the ledger pool (about 100 KB, once per frame or stripe
  item), and the film grain templates (72 KB) are on the stack.

With `max_threads = 1`, a `decode_temporal_unit` of the same shape as the previous one makes no
heap allocation once the previous unit's frames were recycled (`tests/alloc.rs` counts them
with a global allocator and checks the heap peak against `max_alloc`). What still allocates:
the first frames, a change of shape (the frame state is rebuilt, the pool fills with the new
sizes), a frame of several tiles without intra block copy (its whole-frame tile decoder
buffers are dropped after the frame at every thread count, so that what stays booked does not
depend on `max_threads`, and are made again for the next frame), metadata OBUs, film grain (its list of output bands, one per frame), the layer
selection of a layered stream, probe mode, and threads: a call with more than one thread spawns
the frame's team (the worker stacks, the scheduler's result slots), builds one tile-sized state
per tile of a multi-tile frame (its planes come from the pool, its arrays and tile buffers are
new) and gives each wrapup item its private band (from the pool when it holds one of the size)
in new buffer lists (docs/threading.md, "Memory").

## Compute is not limited

No limit bounds the work of a call. Tile data past the end of an OBU reads as zeros, so a few
bytes per `OBU_FRAME` (hidden frames included) can make the decoder run the full tile decode and
wrapup of a frame up to the size limits, and one temporal unit may hold any number of them. The
work is bounded by the input length times the cost of one maximum-size frame. The only stop is
the `cancel` flag, checked per OBU, per superblock row of a tile, and per row band in each wrapup
stage and in film grain. A caller that decodes untrusted data needs a watchdog that sets it.
