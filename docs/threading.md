# Threading

`Limits::max_threads` is the most OS threads one call uses, the calling thread included. 1 spawns
no thread: the calling thread does all the work. Output is byte-identical for every thread count.
All threading is safe Rust (`std::thread::scope`, `Mutex`, `RwLock`, `Condvar`, atomics); the
crate denies `unsafe` outside `src/simd` (docs/safety.md).

## What runs in parallel

- **The pixel work of one frame** (`decoder.rs`, `decode_frame`): the tiles, then the wrapup
  (loop filter, CDEF, superres, loop restoration), as stages of one team of threads (below).
  The team has `min(max_threads, max(tiles, 64-row bands x planes))` threads; with one, the
  frame decodes on the calling thread exactly as before (`wrapup` on whole planes).
- **Tiles of an AVIF grid** (`avif.rs`, `decode_grid`). Each grid tile is a coded item with its own
  `Decoder`. Tile 0 decodes first, alone: it sizes the canvas and gives the output its
  properties. The other tiles then run on `min(max_threads, tiles - 1)` threads and are pasted
  into the output as they finish. Each inner decode gets `max_threads / workers` threads
  (`workers = min(max_threads, tiles)`, so one thread when the grid has at least as many tiles as
  threads, tile 0 included), and all inner decoders share one ledger (the one of the whole
  `decode_avif` call), so `max_alloc` covers them together. The assembled canvas, the
  temporal-unit copy of each item, every decoded tile from when its decoder makes it until it is
  pasted and freed, the transforms of each tile (their output and the scratch `isobmff` may
  allocate, booked before they run and passed to `isobmff` as its exact limit) and the primary
  image held while the alpha item decodes are charged to it too. Item bytes inside `isobmff` (an
  eighth of `max_alloc`, kept out of the ledger's share; split between the workers while tiles
  decode in parallel, whole for tile 0 and the retry) are not; a copy over its share is
  `AllocLimit`. A refused tile gives back its whole booking, metadata included, and a tile's
  decoder gives back metadata that follows its last frame.
  A tile refused (`AllocLimit`, `OutOfMemory`) while others decoded beside it is retried as
  `decode` retries a batch: the tiles from the lowest refused one decode again one at a time,
  each with only the canvas booked beside it, as in a serial decode. So for a grid with at least
  as many tiles as threads the result, the image or the error, is the same for every thread count
  (`tests/avif_alloc.rs`).

- **Film grain** (`grain.rs`, `apply_film_grain`). The frame is cut into bands of 32 luma rows
  (one noise stripe each), run by `run_parallel` on `min(max_threads, bands)` threads. A band
  builds its own stripe and, with overlap, the stripe above it: the generator is seeded per
  stripe, so a band reads only the grain templates, the input frame and its stripes, and writes
  only its own rows of the output (each band owns its `chunks_mut` of every plane, behind a
  `Mutex` the band locks once). The output is the same for every thread count; the unit test
  `every_tier_and_thread_count_matches_the_reference` checks 1, 2, 3 threads against the
  whole-frame noise image of the spec.

- **Frames of a batch** (`decoder.rs`, `Decoder::decode_temporal_units`, used by `decode` and the
  oracle). Every frame this crate decodes is intra with `primary_ref_frame` NONE: its CDFs are the
  defaults for `base_q_idx`, the loop filter deltas and segmentation start from the defaults (no
  segment map or feature data is loaded from a slot), film grain parameters are always updated
  (never loaded), and intra block copy reads only its own frame. So the pixels of a frame are a
  function of the sequence header, its own header and its tile bytes. Only header state crosses
  frames (`RefState`: valid flags, order hints, frame ids, the sizes `show_existing_frame` reads).
  The batch parses OBUs in stream order on the calling thread, runs the header half of the
  reference update right after each frame header (`queue_frame`), and queues the frame. When
  `max_threads` frames are queued (or a barrier comes, or the batch ends) `flush` decodes them:
  `W = min(max_threads, queued, free max_alloc / estimate of one frame)` frames at a time
  (`plan`), each with `max(1, max_threads / W)` threads, so `W x threads <= max_threads`
  (`run_parallel`, with every frame's result kept so the frames below a failure still commit).
  Results are then committed in stream order: slots, output frame (film grain, the `u8`
  conversion), the layer selection per unit. Barriers flush first: a `show_existing_frame` (it
  reads a slot's planes) and a sequence header that changes the sequence (it empties the slots).
  The frame that fails first in stream order wins: its error is returned, the `RefState` is the
  copy taken right after its header was parsed (what a loop of `decode_temporal_unit` calls
  leaves), the slots it refreshes are empty, and the frames after it are dropped; the frames
  before it, those of earlier units included, are in `take_frames`. When any frame of a flush
  hits `AllocLimit` or `OutOfMemory`, the results are dropped and the flush decodes its frames
  one at a time with all the threads, which is exactly the single decode path, so a refusal is
  not caused by the frames that decoded beside it. It can still come at an earlier unit than in
  a loop, or where a loop is not refused: the batch keeps every unit's frames booked until it
  returns, while a loop hands them back unit by unit. With `max_threads = 1` the batch decodes
  the frames as a loop of `decode_temporal_unit` does, with the same caveat about held frames.
  Differences from the loop that are not visible in the output: the slots a frame refreshes are
  emptied when the frame is committed, not before its tiles decode, so the old planes are held a
  little longer; and `W` frame states are alive at once (docs/limits.md).

Not parallel: the OBU parse, the commit of the output. Frames decode one after another in
`decode_temporal_unit` (only `decode_temporal_units` overlaps frames). The stages of a frame run one after another: a stage
starts when every item of the one before has finished (no row pipelining between stages).

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
| Loop restoration | plane x stripe | upscaled `CdefFrame` and `CurrFrame` (without superres the two frames themselves) | the stripe's rows of `LrFrame` |

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

Not done: row pipelining (starting the filters of a superblock row while later rows still decode,
as dav1d does). It needs the loop filter's horizontal edges in row order across bands, which
the stage split above avoids; the stages are parallel each, with a barrier between them.

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

## Tried and not kept: a rayon pool (2026-11-25)

The owner decision of 2026-11-24 allows a thread-pool crate if a CI bench shows it measurably
faster than `std::thread::scope`. PR 38 (closed unmerged, branch deleted, head ed71c7b) built both
from one commit and the bench timed them side by side.

What the trial was:

- `rayon-core = "=1.13.0"` behind an optional feature `rayon`; one lazy `OnceLock` pool of
  `available_parallelism - 1` threads (the calling thread works too); `ThreadPool::in_place_scope`
  with at most `max_threads - 1` spawned jobs per call, so `max_threads = 1` never touches the
  pool; a caller already on a rayon thread uses its own pool (`current_thread_index`).
- The atomic-counter work loop of `run_parallel` was kept, so errors, cancel and the ledger behave
  as before.
- The frame decode did not use the team. Each stage was one `run_parallel`, and the loop filter
  (vertical edges), CDEF, superres and loop restoration wrote straight into disjoint
  `split_at_mut` row views of their output planes: no band copy, no copy back, no frame lock.
  Only the horizontal loop filter edges kept their copy (a column strip is not contiguous).
- A test with calls nested inside calls checked that the threads busy at once stay within
  `max_threads` (1, 2, 3, 8), with and without the feature. The whole suite also passed with the
  feature, with the best SIMD tier and with `AV1_SIMD=scalar`.

The bench (`tests/parity/bench.sh`, table "pools") built the std and the rayon variant of the same
commit, default and v4 builds, and timed them A, B, A, B on the same machine, keeping the
minimum of two best-of-5 per side. The rule, fixed before the run: with default threads the "all"
row of rayon must be at most 97% of std in both builds in both of two bench runs; no class slower
than max(3%, 1 ms); the 1 thread table is an A/A control (that path never touches a pool) and must
stay within max(3%, 1 ms).

Result, `rayon / std` of the "all" row (16 threads on the ARC runner; the 1 thread control is
within 1% in every run):

| run | threads | default build | v4 build |
|---|---|---|---|
| bench 45576777672 | default | 302 vs 297 ms, 98% | 294 vs 290 ms, 98% |
| bench 45577649764 | default | 304 vs 301 ms, 99% | 299 vs 289 ms, 96% |
| bench 45576777672 | 1 (control) | 891 vs 884 ms, 99% | 860 vs 857 ms, 99% |
| bench 45577649764 | 1 (control) | 891 vs 888 ms, 99% | 855 vs 858 ms, 100% |

Default threads, per class (std ms / rayon ms, default build; run 1, run 2):

| class | run 1 | run 2 |
|---|---|---|
| q8 | 15 / 14 | 15 / 15 |
| q10 | 32 / 31 | 33 / 31 |
| allintra | 53 / 53 | 53 / 54 |
| size | 11 / 10 | 11 / 10 |
| grain | 6 / 7 | 6 / 7 |
| avif8 | 42 / 41 | 42 / 41 |
| avifhi | 97 / 95 | 97 / 96 |
| tiles8 | 24 / 24 | 24 / 24 |
| tiles10 | 22 / 22 | 23 / 23 |
| all | 302 / 297 | 304 / 301 |

Why it loses: the rule needs 97% or less in all four default-thread cells, and only one cell
(v4, run 2, 96%) gets there; the others are 98%, 98% and 99%. The total gain is 3 to 5 ms in 300.
The 1 thread control, which runs identical code, differs by up to 1% (and one class cell by 10%
at 1 ms resolution), so a gain of this size is within noise. The classes that spawn most per
frame (`size`, `allintra`, `tiles8`, `tiles10`) do not move. That is an inference, not measured:
since the frame-parallel batch ("Frames of a batch" above) the frames of a unit overlap, so the
spawn cost per frame is already small, and the band copies the pool removes are small next to
the filters. `grain` is 1 ms slower in every run (6 vs 7 ms), at the limit of the timer.

So the crate stays on `std::thread::scope` and adds no dependency. Revisit it if the frame
decode gets row pipelining, where a persistent pool matters more than one scope per stage.

## Tests

- `parallel.rs`: the schedulers in isolation: peak live items and distinct threads for
  `max_threads` 1, 2, 3 (and 8 for the team; never above it), result order, every item of a
  stage done (and none of the next) when `run` returns, lowest failing index wins and the next
  stage starts clean, `max_threads = 1` runs on the caller's thread only, charged allocations stay
  within the ledger and a refusal is `AllocLimit`.
- `decoder.rs`: `tiles_decode_the_same_for_every_thread_count` runs the real tile phase and
  wrapup over a 3 x 2 tile frame (pseudo-random tile data, loop restoration on) with 1, 2, 3, 6, 8
  threads and compares every array and plane; `tile_threads_stay_within_max_alloc` checks the
  ledger peak for several `max_alloc` including a refusal that only the threaded path hits.
  `wrapup_is_the_same_for_every_thread_count` builds frames whose tiles are done
  (`wrapup_frame`: 4:2:0, 4:2:2, 4:4:4 and mono, 8, 10 and 12 bits, superres off and on, the loop
  filter, CDEF and loop restoration on with random parameters, random mode info and samples) and
  compares `LrFrame` and the filtered `CurrFrame` of 2, 3 and 8 threads with 1;
  `wrapup_threads_stay_within_max_alloc` checks the ledger peak for `max_alloc` from 1 KiB up and
  1, 2, 3, 8 threads, and that every byte is released; `team_size_stays_within_max_threads`;
  `wrapup_threads_honour_the_cancel_flag`.
- `decoder.rs`, batches: `batch_is_the_same_as_a_loop_of_units_for_every_thread_count` decodes
  streams of key frames (valid, with a truncated unit, with an empty unit) with
  `decode_temporal_units` at 0, 1, 2, 3 and 8 threads and compares the frames, the error, the
  `RefState` and which slots hold a frame with a loop of `decode_temporal_unit`;
  `plan_stays_within_max_threads` (`W x threads <= max_threads`);
  `batch_frames_in_flight_stay_within_max_threads_and_max_alloc` (frames decoding at once, the
  ledger peak above one frame's with 8 threads and within `max_alloc` for several limits).
- The fuzz targets `simd_threads` and `frames_threads` (docs/fuzzing.md): `simd_threads` runs
  `wrapup_frame` on any input with 1, 2 and 3 threads and asserts the same output;
  `frames_threads` runs the batch at 1, 2, 3 and 8 threads against a loop of units.
- There is no tile corpus in the unit tests (no fixtures in `tests/`). The corpus check is the parity
  gate (docs/parity.md): it compares `--threads 1` with the default threads on every stream,
  including the non-uniform tiling ones, against dav1d.
