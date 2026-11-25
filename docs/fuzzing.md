# Fuzzing

`tests/fuzz` is a cargo-fuzz crate (`cargo +nightly fuzz run --fuzz-dir tests/fuzz <target>`).
The last input byte holds the options; an input that starts with an IVF header is split into
its temporal units. Seeds: the parity corpus (`run.sh seeds`).

| Target | Entry point |
|---|---|
| `probe` | `av1::probe`: OBU framing, sequence, frame and tile group headers |
| `avif` | `av1::decode_avif`: isobmff-rs item graph, av1C, grid assembly, alpha (option bits: 2 threads, `apply_transforms`) |
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

Job `fuzz-arm64` (same workflow, after the x86_64 matrix, two jobs at a time): the `simd_*`
targets built for `aarch64-unknown-linux-gnu` (Dockerfile stage `fuzz-arm64`,
`cargo +nightly fuzz build --target aarch64-unknown-linux-gnu -s none`, libFuzzer compiled with
`aarch64-linux-gnu-g++`) and run with `qemu-aarch64 -L /usr/aarch64-linux-gnu`, so the NEON tier
is compared with scalar. No sanitizer: ASan does not run under qemu-user, so `-malloc_limit_mb`
has no malloc hook there and only `-rss_limit_mb` (of the qemu process) holds. The dispatch input
`only` picks the target of both jobs; a target that is not `simd_*` skips `fuzz-arm64`. Its corpus
cache (`fuzz-corpus-arm64-<target>`) starts from the x86_64 corpus of the same target; artifacts
are `crashers-arm64-<target>` and `corpus-arm64-<target>`.

`simd_threads` under qemu runs as processes of at most 1800 execs (`-runs=1800`, counting the
corpus load), restarted on the same corpus until `max_total_time` is used up; the other targets
run as one process. qemu-user 7.2 (bookworm's `qemu-user`, 7.2.22) keeps about 280 KB of RSS for
every guest thread after the thread exits, and the target spawns three threads per exec (one
scope of 2 threads, one of 3). Fuzz run 45490215220 hit `-rss_limit_mb=2048` after 2514 execs
(about 0.84 MB per exec from the first exec on). Run 45519136759 on the same corpus, 3000 execs
per mode: a target that only spawns the same 1 + 2 empty scoped threads (no av1 code) went from
45 MB to 2363 MB, the wrapup with 1 thread only stayed at 64 MB, the real target went to 2315 MB;
the x86_64 job (native) stays at about 440 MB over 5800 execs. So the growth is qemu's, not the
crate's. 1800 execs end near 1.55 GB. If the arm64 corpus of this target grows past about 1500
files, the corpus load alone fills a process: minimize the corpus (`-merge=1`) then.

## Runs on main

| Date | Commit | max_total_time | Targets | Fuzz hours | Crashes | Runs (probe / decode / avif) |
|---|---|---|---|---|---|---|
| 2026-11-23 | a5cb385 | 900 s | 3 | 0.75 | 0 | 9,965,781 / 7,930 / 4,341 |

`decode` and `avif` run whole frames under ASan, so they do few runs per second.
