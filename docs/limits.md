# Limits

`Limits` (src/limits.rs) caps one call; every public entry point takes it.

- `max_width`, `max_height`, `max_pixels`: checked on the frame size of each frame header (and
  the sequence header maximums) before any frame buffer is allocated.
- `max_alloc`: a ledger (src/ledger.rs) charges every buffer that scales with the input (frame
  planes, reference slots, per-block state, output planes) before it is allocated and releases
  it on drop. Allocation uses `try_reserve_exact`; a failure is `Error::OutOfMemory`.
- `max_metadata`: the total payload of the metadata OBUs kept with a frame.
- `max_threads`: the most threads of one call, the calling thread included. The tiles of a frame
  and the tiles of an AVIF grid decode in parallel within it (docs/threading.md); 1 spawns no
  thread, and a frame with `allow_intrabc` decodes its tiles serially.


Charged per frame, all before allocation and released when the frame state drops: `CurrFrame`
(planes rounded up to 128, `u16` samples); the mode info arrays of `FrameState` (about 20 bytes
per 4x4 block plus palette colours when screen content tools are on, and the loop restoration
unit arrays); per tile, the above context arrays (`mi_cols` rounded to 32 per plane); with more than one
thread and tile, per worker one tile-sized state (the same arrays and planes for the tile's
rectangle, plus the loop restoration unit arrays of the frame), all in the same ledger; at the
wrapup, `CdefFrame` (same size as `CurrFrame`), the upscaled planes (exact size) when superres
is on, and the `LrFrame` planes (exact size). The peak is `CurrFrame` + `CdefFrame` + up to two
exact-size planes sets + the arrays; a frame over `max_alloc` fails with `AllocLimit` before the
tile decode starts or at the wrapup, never after exceeding it. Film grain charges, per band of 32
luma rows while the band runs, its noise stripe and the one above it (34 rows of about `width + 34`
`i32` per plane each) plus one row: at most one band per worker. The output planes are charged
from when they are built until the
temporal unit returns; `decode` keeps every frame it collects charged until it returns. AVIF: one
ledger per `decode_avif` call covers every item decoder, the grid canvas, the item copy and the
primary image (docs/threading.md).

## Compute is not limited

No limit bounds the work of a call. Tile data past the end of an OBU reads as zeros, so a few
bytes per `OBU_FRAME` (hidden frames included) can make the decoder run the full tile decode and
wrapup of a frame up to the size limits, and one temporal unit may hold any number of them. The
work is bounded by the input length times the cost of one maximum-size frame. The only stop is
the `cancel` flag, checked per OBU, per superblock row of a tile, and per row band in each wrapup
stage and in film grain. A caller that decodes untrusted data needs a watchdog that sets it.
