# Fuzzing

`tests/fuzz` is a cargo-fuzz crate (`cargo +nightly fuzz run --fuzz-dir tests/fuzz <target>`).
The last input byte holds the options; an input that starts with an IVF header is split into
its temporal units. Seeds: the parity corpus (`run.sh seeds`).

| Target | Entry point |
|---|---|
| `probe` | `av1::probe`: OBU framing, sequence, frame and tile group headers |
| `avif` | `av1::decode_avif`: isobmff-rs item graph, av1C, grid assembly, alpha |
| `decode` | `Decoder::decode_temporal_unit` on every temporal unit (option bits: Annex B, 2 threads, no film grain) |
| `simd_lf` | `av1::fuzzing::lf`: one loop filter edge run, every SIMD tier against scalar |
| `simd_grain` | `av1::fuzzing::grain`: the film grain blend kernels (`simd::grain`), every tier against scalar |
| `simd_lr` | `av1::fuzzing::lr`: one loop restoration block (bit depth, size, taps, self guided set and weights, window), every tier against scalar |
| `simd_cdef` | `av1::fuzzing::cdef`: CDEF direction and filter kernels, every tier against scalar |
| `simd_ipred` | `av1::fuzzing::ipred`: intra prediction, palette, CFL and the intra edge filter on every SIMD tier against scalar (first byte picks the kernel) |
| `simd_itx` | `av1::fuzzing::itx`: one inverse transform + add block (size, type, bit depth, coded region, coefficients, samples) on every tier against scalar |
| `simd_threads` | `av1::fuzzing::threads`: the threaded frame wrapup on a frame built from the input (format, bit depth, superres, filter parameters, mode info, samples) with 1, 2 and 3 threads; the outputs must be equal |
| `simd_entropy` | `av1::fuzzing::entropy`: the symbol reader against the spec's bit by bit reader on every tier, and the CDF adaptation of every tier against scalar |
| `simd_alloc` | `av1::fuzzing::alloc`: a `Decoder` that reuses its buffers and gets its frames back with `recycle`, against one that keeps no frame state (family `alloc` has no SIMD tiers) |

Each SIMD kernel family adds a cross-tier target `simd_<family>`: it calls
`av1::fuzzing::<family>` (feature `fuzzing`, enabled only by the fuzz crate), which builds kernel
inputs from the bytes, runs every available tier and asserts that each output equals the scalar
output.

`.github/workflows/fuzz.yml`: weekly (Thursday 12:23 UTC) and on dispatch, two jobs at a time,
`-rss_limit_mb=2048 -malloc_limit_mb=512`, crashers uploaded as artifacts.

## Runs on main

| Date | Commit | max_total_time | Targets | Fuzz hours | Crashes | Runs (probe / decode / avif) |
|---|---|---|---|---|---|---|
| 2026-11-23 | a5cb385 | 900 s | 3 | 0.75 | 0 | 9,965,781 / 7,930 / 4,341 |

`decode` and `avif` run whole frames under ASan, so they do few runs per second.
