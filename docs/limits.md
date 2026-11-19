# Limits

`Limits` (src/limits.rs) caps one call; every public entry point takes it.

- `max_width`, `max_height`, `max_pixels`: checked on the frame size of each frame header (and
  the sequence header maximums) before any frame buffer is allocated.
- `max_alloc`: a ledger (src/ledger.rs) charges every buffer that scales with the input (frame
  planes, reference slots, per-block state, output planes) before it is allocated and releases
  it on drop. Allocation uses `try_reserve_exact`; a failure is `Error::OutOfMemory`.
- `max_metadata`: the total payload of the metadata OBUs kept with a frame.
- `max_threads`: tiles decode in parallel within it (after parity); 1 spawns no thread.

The tests that measure the peaks of `max_alloc` and `max_threads` land with the tile decoder.
