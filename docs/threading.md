# Threading

`Limits::max_threads` is the most OS threads one call uses, the calling thread included. 1 spawns
no thread: the calling thread does all the work. Output is byte-identical for every thread count.
All threading is safe Rust (`std::thread::scope`, a `Mutex`, atomics); the crate forbids `unsafe`.

## What runs in parallel

- **Tiles of one frame** (`decoder.rs`, `decode_tiles`). A frame with several tiles decodes them
  on `min(max_threads, tiles)` threads. The loop filter, CDEF, superres and loop restoration of
  the frame (the wrapup) still run on the calling thread after the last tile.
- **Tiles of an AVIF grid** (`avif.rs`, `decode_grid`). Each grid tile is a coded item with its own
  `Decoder`; they run on `min(max_threads, tiles)` threads and are pasted into the output as they
  finish. Each inner decode gets `max_threads / workers` threads (so one when the grid has at least
  as many tiles as threads), and all inner decoders share one ledger (the one of the whole
  `decode_avif` call), so `max_alloc` covers them
  together. The assembled canvas, the temporal-unit copy of each item and the primary image held
  while the alpha item decodes are charged to it too. Item bytes inside `isobmff` and a decoded
  tile waiting for the canvas lock are not.

Not parallel: the OBU parse, the wrapup, film grain, output. Frames of a stream decode one after
another (a later frame may reference an earlier one).

## The scheduler

`parallel::run_parallel( n, max_threads, f )` runs `f( 0 ) .. f( n - 1 )` and returns the results in
item order. One `std::thread::scope` per call (per frame, per grid); `workers = min(max_threads, n)`,
the calling thread is one of them. Items are taken from an `AtomicUsize` counter in increasing
order. With one worker nothing is spawned and the items run in order.

Errors are deterministic: after a failure at index `k` no item above `k` starts, items already
running finish, and every item below `k` still runs, so the error of the lowest failing index is
returned, and a serial run would have stopped at that same item. (`Cancelled` from the
cancel flag and `AllocLimit` from the shared ledger depend on timing by nature.) The cancel flag is
checked before every tile and once per superblock row inside a tile, in every worker, and on the
calling thread once per 64-row band in every wrapup stage (loop filter, CDEF, superres, loop
restoration) and in film grain.

## Tile decode and the byte-identity argument

A tile decode reads and writes only inside its own tile rectangle: `is_inside` and the availability
flags use the tile bounds, the intra edge pixels are guarded by `AvailU`/`AvailL` and `BlockDecoded`
(which `clear_block_decoded_flags` limits to the tile), and the above/left contexts are cleared at
each tile start. Tiles start on superblock boundaries and a block never crosses a superblock, so a
tile also never writes into another tile's samples; only the frame edge padding is written past the
last superblock row and column. The in-loop filters, which do cross tile edges, run after all tiles.

So a tile decode can run on private storage:

- `FrameState::new_tile` makes a `FrameState` for one tile: `mi_rows`/`mi_cols`, the mode-info
  arrays and the planes (rounded up to 128) are the tile's own, the tile starts at (0, 0) of it, and
  `mi_row0`/`mi_col0` hold its place in the frame. `TileDecoder` runs on it unchanged, as tile row 0,
  tile column 0 (`read_lr` adds `mi_row0`/`mi_col0`, because loop restoration units are numbered in
  the frame). Everything the decode takes from frame position is the same in tile coordinates:
  tile origins are multiples of 16 mode-info units, so parities, 64x64 alignments and subsampled
  positions do not change, and an interior tile edge is a superblock edge, so `has_rows`, `has_cols`,
  `onscreen` sizes and the `max_x`/`max_y` clamps only differ where no block reaches them. The
  loop restoration arrays are the frame's (a few bytes per unit) with every unit marked unset.
- `FrameState::merge_tile` copies the tile's rectangle of each mode-info array, the loop restoration
  units the tile set, and the samples of the tile's superblocks into the frame state, under a
  `Mutex` (a copy of a few rows per array; the decode itself runs unlocked). Rectangles are
  disjoint, so the order of the merges does not matter.

One implementation: the serial path (`max_threads == 1`, a single tile) calls the same
`TileDecoder` and writes straight into the frame state, which is the tile state with origin (0, 0);
it only skips the spawn, the tile-local state and the merge.

**IntraBC frames are serial.** A frame with `allow_intrabc` decodes its tiles in order, directly on
the frame state: intra block copy reads samples and mode info by motion vector, and this decoder does
not enforce `is_mv_valid`, so a stream may point outside its tile. Reading the frame state in tile
order keeps that behaviour exactly as before.

## Memory

Every buffer of a tile state is charged to the shared ledger (`Arc<Ledger>`, atomics) before it is
allocated and released when the state drops, so the ledger peak covers all threads. A worker holds at
most one tile state at a time, so the extra memory is at most `workers` tile-sized states on top of
the frame state. That is below the peak of the wrapup (`CurrFrame` + `CdefFrame` + ...) for frames
with more tiles than threads; a frame that does not fit fails with `AllocLimit`, exactly like one
that does not fit the wrapup. `max_threads = 1` and single-tile frames allocate nothing extra.

## Tests

- `parallel.rs`: the scheduler in isolation: peak live items and distinct threads for
  `max_threads` 1, 2, 3 (never above it), result order, lowest failing index wins, `max_threads = 1`
  runs on the caller's thread only, charged allocations stay within the ledger and a refusal is
  `AllocLimit`.
- `decoder.rs`: `tiles_decode_the_same_for_every_thread_count` runs the real tile phase over a
  3 x 2 tile frame (pseudo-random tile data, loop restoration on) with 1, 2, 3, 6, 8 threads and
  compares every array and plane; `tile_threads_stay_within_max_alloc` checks the ledger peak for
  several `max_alloc` including a refusal that only the threaded path hits.
- There is no tile corpus in the unit tests (no fixtures in `tests/`). The corpus check is the parity
  gate (docs/parity.md): it compares `--threads 1` with the default threads on every stream,
  including the non-uniform tiling ones, against dav1d.
