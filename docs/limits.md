# Limits

`Limits` (src/limits.rs) caps one call; every public entry point takes it.

- `max_width`, `max_height`, `max_pixels`: checked on the frame size of each frame header (and
  the sequence header maximums) before any frame buffer is allocated.
- `max_alloc`: a ledger (src/ledger.rs) charges every buffer that scales with the input (frame
  planes, reference slots, per-block state, output planes) before it is allocated and releases
  it on drop. Allocation uses `try_reserve_exact`; a failure is `Error::OutOfMemory`.
- `max_metadata`: the total payload of the metadata OBUs kept with a frame.
- `max_threads`: tiles decode in parallel within it (after parity; tiles are serial for now); 1 spawns no thread.


Charged per frame, all before allocation and released when the frame state drops: `CurrFrame`
(planes rounded up to 128, `u16` samples); the mode info arrays of `FrameState` (about 20 bytes
per 4x4 block plus palette colours when screen content tools are on, and the loop restoration
unit arrays); per tile, the above context arrays (`mi_cols` rounded to 32 per plane); at the
wrapup, `CdefFrame` (same size as `CurrFrame`), the upscaled planes (exact size) when superres
is on, and the `LrFrame` planes (exact size). The peak is `CurrFrame` + `CdefFrame` + up to two
exact-size planes sets + the arrays; a frame over `max_alloc` fails with `AllocLimit` before the
tile decode starts or at the wrapup, never after exceeding it. The output planes are charged
while they are built.
