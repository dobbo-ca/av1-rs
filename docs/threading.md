# Threading

`Limits::max_threads` is the most OS threads one call uses, the calling thread included. 1 spawns
no thread: the calling thread does all the work. Output is byte-identical for every thread count.
All threading is safe Rust (`std::thread::scope`, `Mutex`, `RwLock`, `Condvar`, atomics); the
crate forbids `unsafe`.

## What runs in parallel

- **The pixel work of one frame** (`decoder.rs`, `decode_frame`): the tiles, then the wrapup
  (loop filter, CDEF, superres, loop restoration), as stages of one team of threads (below).
  The team has `min(max_threads, max(tiles, 64-row bands x planes))` threads; with one, the
  frame decodes on the calling thread exactly as before (`wrapup` on whole planes). A frame of
  one tile (most AVIF stills, the `allintra` streams) is row pipelined instead: one thread
  decodes the tile while the others filter the rows it has finished ("Row pipelining").
- **Tiles of an AVIF grid** (`avif.rs`, `decode_grid`). Each grid tile is a coded item with its own
  `Decoder`; they run on `min(max_threads, tiles)` threads and are pasted into the output as they
  finish. Each inner decode gets `max_threads / workers` threads (so one when the grid has at least
  as many tiles as threads), and all inner decoders share one ledger (the one of the whole
  `decode_avif` call), so `max_alloc` covers them
  together. The assembled canvas, the temporal-unit copy of each item and the primary image held
  while the alpha item decodes are charged to it too. Item bytes inside `isobmff` and a decoded
  tile waiting for the canvas lock are not.

- **Film grain** (`grain.rs`, `apply_film_grain`). The frame is cut into bands of 32 luma rows
  (one noise stripe each), run by `run_parallel` on `min(max_threads, bands)` threads. A band
  builds its own stripe and, with overlap, the stripe above it: the generator is seeded per
  stripe, so a band reads only the grain templates, the input frame and its stripes, and writes
  only its own rows of the output (each band owns its `chunks_mut` of every plane, behind a
  `Mutex` the band locks once). The output is the same for every thread count; the unit test
  `every_tier_and_thread_count_matches_the_reference` checks 1, 2, 3 threads against the
  whole-frame noise image of the spec.

Not parallel: the OBU parse, output. Frames of a stream decode one after another (a
later frame may reference an earlier one). The stages of a frame of several tiles run one after
another: a stage starts when every item of the one before has finished.

## The schedulers

`parallel::run_parallel( n, max_threads, f )` (the AVIF grid, film grain) runs `f( 0 ) .. f( n - 1 )` and
returns the results in item order. One `std::thread::scope` per call;
`workers = min(max_threads, n)`, the calling thread is one of them. Items are taken from an
`AtomicUsize` counter in increasing order. With one worker nothing is spawned and the items run in
order.

`parallel::with_team( threads, item, lead )` (a frame) spawns `threads - 1` workers once, in one
`std::thread::scope`, and runs `lead` on the calling thread. Each `team.run( job, n )` of `lead`
is one stage: it wakes the workers (a `Condvar`), the calling thread works too, items are taken
from an `AtomicUsize` counter in increasing order as above, and `run` returns when every thread
has left the stage. Between stages the workers wait on the condvar, so the threads of a frame are
spawned once for all its stages. With one thread nothing is spawned and the items of a stage run
in order. There is no thread pool that outlives a call: a pool shared between calls would need its
jobs to borrow the caller's buffers past the scope, which safe Rust cannot express, so the lazy
shared pool of the conventions does not apply; each call starts its own scope.

Errors are deterministic, in both: after a failure at index `k` no item above `k` starts, items
already running finish, and every item below `k` still runs, so the error of the lowest failing
index is returned, and a serial run would have stopped at that same item. (`Cancelled` from the
cancel flag and `AllocLimit` from the shared ledger depend on timing by nature.) The cancel flag is
checked before every tile, once per superblock row inside a tile, and by every wrapup item (once
per 64-row band, column strip or stripe), on whichever thread runs it, and in film grain.

## The wrapup stages and why they are exact

Every stage reads buffers that no item of that stage writes, and its items write disjoint
samples, so the order of the items does not matter; where the spec has an order inside a stage,
an item keeps it. The stages and their items (`Wrap::item`):

| Stage | Items | Reads | Writes |
|---|---|---|---|
| Tiles | one per tile | tile data | `CurrFrame`, mode info (merged per tile, below) |
| Loop filter, vertical edges | plane x 64-row band | mode info | the band's rows of `CurrFrame` |
| Loop filter, horizontal edges | plane x column strip (one per thread, 64-column units) | mode info | the strip's columns of `CurrFrame` |
| CDEF | 64-row band (all planes) | `CurrFrame` | the band's rows of `CdefFrame` |
| Superres | plane x 64-row band, `CdefFrame` then (with loop restoration) `CurrFrame` | the input plane | the band's rows of the upscaled plane |
| Loop restoration | plane x stripe | upscaled `CdefFrame` and `CurrFrame` | the stripe's rows of `LrFrame` |

- **Loop filter** (7.14): the spec filters, per plane, all vertical edges, then all horizontal
  edges, in raster order. A vertical edge reads and writes the samples of its own (four) rows
  only, and a horizontal edge those of its own (four) columns only, at most 7 samples across the
  edge: the edge position is a multiple of the filter size, which is at least the number of
  samples it reaches, so a vertical edge never reaches past the start or the end of a row (and
  the bounds check in `sample_filter` never fires). So the vertical edges of different rows are
  independent, and so are the horizontal edges of different columns: row bands of vertical
  edges, then column strips of horizontal edges, each in the spec order inside the item, give the
  spec's samples. This holds for any mode info (it does not rely on the transform sizes forming a
  consistent grid). Horizontal edges are not split into row bands: an edge at a band boundary
  reads samples that the edges above it wrote and writes samples that the edges below it read.
- **CDEF** (7.15) reads `CurrFrame` around a block (the direction search and the taps) and writes
  only the block's samples of `CdefFrame`.
- **Superres** (7.16) works row by row.
- **Loop restoration** (7.17) works per stripe, reading the stripe's rows of the upscaled
  `CdefFrame` and 3 rows above and below it (`CurrFrame` rows across stripe boundaries), and
  writes only the stripe's rows. Stripes cover each plane's rows once.

The buffers live in `RwLock`s (`Wrap`). An item holds read locks on its inputs while it works and
computes into its own band (a copy of its rows or columns, charged to the ledger), then copies
the band in under the output's lock; the loop filter, which works in place, copies its band out
of `CurrFrame` first. So threads only wait for each other's copies, and no thread takes a lock it
already holds (the calling thread holds none while it runs a stage). The buffers are allocated by
the calling thread between stages, in the order of the serial `wrapup`.

## Row pipelining

A frame of one tile decodes serially (one symbol decoder), so the stages above would leave every
other thread idle until the tile is done. Such a frame (`decode_frame` with more than one thread
and one tile) runs one stage, `Job::Pipe`, with one item per thread: item 0 decodes the tile
(`Wrap::pipe_decode`), the others take filter tasks as their inputs become final
(`Wrap::pipe_filter`). Output and errors are those of the serial path.

- **Hand over.** The tile decode works on its own frame state, as in the serial path, because
  intra prediction reads the unfiltered samples above. After each superblock row
  (`TileDecoder::decode_tile` calls back) it copies the 64-row bands it has finished into the
  shared state: the mode info rows (`FrameState::copy_rows`, plus all loop restoration units)
  into a second frame state the filters read (`shadow`), the samples into the shared
  `CurrFrame` the loop filter works on. It takes both write locks with `try_write`; when a
  filter task holds one it decodes on and hands over after the next row, waiting for the locks
  then, so the filters lag at most one more superblock row. After the last row it hands over
  the rest. A band is never written by the tile decode after it is handed over: blocks do not
  cross superblocks, and the mode info rows of a band (the chroma `LoopfilterTxSizes` at their
  subsampled rows, the loop restoration units of the rows read so far) are final when the
  superblock row ends.
- **Tasks** (`Task`, `Progress`), per 64-row band `b` (16 mode info rows; chroma rows shifted
  by the subsampling, the last band to the plane end), each started when what it reads is final:

  | Task | Starts after | Reads | Writes |
  |---|---|---|---|
  | `V( plane, b )` vertical edges | band `b` handed over | its rows | its rows of `CurrFrame` |
  | `H( plane, b )` horizontal edges | `V( plane, b )`, `H( plane, b - 1 )` | its rows and 8 above | the same rows |
  | `C( b )` CDEF, every plane | `H( p, b + 1 )` for every plane (no loop filter: band `b + 1` handed over) | `CurrFrame` rows `b` +- 2 | its rows of `CdefFrame` |
  | `U( b )` superres, or the exact-size copy | `C( b )` | its rows of `CdefFrame` (and `CurrFrame`) | its rows of the upscaled planes |
  | `L( plane, n )` loop restoration stripe | `U( 0 ..= n )` | the stripe, 3 rows around it | its rows of `LrFrame` |

  Horizontal edges go down a plane in band order, which is the spec's raster order: an edge
  reaches at most 7 rows above it (into the band above, so `H` copies 8 rows above its band) and
  none below its band (an edge sits at a multiple of its filter size, which is at least its
  reach). They run after the vertical edges of their band and the band above, which is all the
  spec's "all vertical edges first" needs, since a vertical edge touches its own row only.
  CDEF reads 2 rows below its band, which the horizontal edges at the top of the next band
  write, hence `H( b + 1 )`; after that no task writes those rows (`H( b + 2 )` starts 8 rows
  above band `b + 2`). Stripe `n` covers luma rows `64n - 8 .. 64n + 56` and reads 2 rows below
  (from `UpscaledCurrFrame`), all in bands `<= n`; its loop restoration units start at or above
  the stripe, so their parameters are in the bands handed over. A plane without loop
  restoration takes its `UpscaledCdefFrame` plane as `LrFrame` at the end.
- **Locks.** The buffers are the `RwLock`s of the stages above, used the same way (an item copies
  its band out and in; readers hold the lock for the item). A task takes the `shadow` read lock
  first, for its whole run, and the sample locks inside; the tile decode takes `shadow` and then
  `CurrFrame` for writing, so it never waits on a task that waits on it. The tasks of a lane
  (kind and plane) start in index order; `Progress` (a `Mutex` and a `Condvar`) holds which are
  started and done. A task error or a failed tile decode sets `failed`: the threads stop taking
  tasks and the tile decode stops after the row; the frame returns the tile decode's error when
  it failed by itself, the failed task's otherwise. The guard `Abort`
  sets `failed` when a thread leaves early, so no thread waits for a task that will not come.
  The cancel flag is checked by the tile decode once per superblock row and by every filter
  task.

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
that does not fit the wrapup. `max_threads = 1` allocates nothing extra. A pipelined single-tile
frame allocates, before its tile decode starts, the shadow frame state (mode info arrays and a
second `CurrFrame`, the one the filters work on) and all the wrapup buffers at once (`CdefFrame`,
the exact-size planes), since every filter runs from the first band on; its tasks hold private
bands as the wrapup items do.

The wrapup allocates its frame buffers in the same order as the serial path. On top, each
wrapup item holds its private band while it runs, charged to the same ledger: 64 rows of one
plane (CDEF: of every plane; loop restoration: one stripe), or for the horizontal loop filter
edges a column strip of one plane (the plane width divided by the threads, full height). So the
extra memory is at most `threads` bands, a plane of the largest size per stage at worst. The
filters' small scratch buffers (the loop restoration windows, a few KiB per 64-sample block) are
not charged, as in the serial path. One thread allocates no band.

A tile state's planes come from the ledger's pool and go back to it when the state drops, still
charged (docs/limits.md, "Buffer reuse"); the pool gives way to any charge that does not fit, so
the peak argument above is unchanged. What a call with more than one thread allocates that a
`max_threads = 1` call does not: the team's worker threads (their stacks, `std::thread::scope`)
and its result slots, the wrapup's shared buffer lists, the private band of every wrapup item, and
per tile the mode-info arrays and tile decoder buffers of its tile-sized state (the bands' sample
buffers and the tile states' planes come from the pool when it holds one of the size). So only a
`max_threads = 1` call of the same shape as the previous one allocates nothing (`tests/alloc.rs`).

## Tests

- `parallel.rs`: the schedulers in isolation: peak live items and distinct threads for
  `max_threads` 1, 2, 3 (and 8 for the team; never above it), result order, every item of a
  stage done (and none of the next) when `run` returns, lowest failing index wins and the next
  stage starts clean, `max_threads = 1` runs on the caller's thread only, charged allocations stay
  within the ledger and a refusal is `AllocLimit`.
- `decoder.rs`: `tiles_decode_the_same_for_every_thread_count` runs the real tile phase and
  wrapup over a 3 x 2 tile frame (pseudo-random tile data, the loop filter, CDEF and loop
  restoration on) with 1, 2, 3, 6, 8
  threads and compares every array and plane; `tile_threads_stay_within_max_alloc` checks the
  ledger peak for several `max_alloc` including a refusal that only the threaded path hits.
  `wrapup_is_the_same_for_every_thread_count` builds frames whose tiles are done
  (`wrapup_frame`: 4:2:0, 4:2:2, 4:4:4 and mono, 8, 10 and 12 bits, superres off and on, the loop
  filter, CDEF and loop restoration on with random parameters, random mode info and samples) and
  compares `LrFrame` and the filtered `CurrFrame` of 2, 3 and 8 threads with 1;
  `wrapup_threads_stay_within_max_alloc` checks the ledger peak for `max_alloc` from 1 KiB up and
  1, 2, 3, 8 threads, and that every byte is released; `team_size_stays_within_max_threads`;
  `wrapup_threads_honour_the_cancel_flag` (also on a pipelined frame).
  Row pipelining: `pipelined_tile_decodes_the_same_for_every_thread_count` decodes a one-tile
  200x136 frame (3 bands; the loop filter, CDEF and loop restoration on, superres and 128x128
  superblocks by seed) with 1, 2, 3, 8 threads and compares every array and plane;
  `pipelined_tile_stays_within_max_alloc` checks the ledger peak and the release for `max_alloc`
  from 1 KiB up; `pipeline_tasks_respect_their_inputs` drives `Progress` alone, handing over
  bands and finishing tasks in first-in and last-in order, and checks that every task starts
  only after the tasks of the table above and that all of them run.
- The fuzz target `simd_threads` (docs/fuzzing.md) runs `wrapup_frame` on any input with 1, 2
  and 3 threads and asserts the same output.
- There is no tile corpus in the unit tests (no fixtures in `tests/`). The corpus check is the parity
  gate (docs/parity.md): it compares `--threads 1` with the default threads on every stream,
  including the non-uniform tiling ones, against dav1d.
