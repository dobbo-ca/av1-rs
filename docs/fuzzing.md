# Fuzzing

`tests/fuzz` is a cargo-fuzz crate (`cargo +nightly fuzz run --fuzz-dir tests/fuzz <target>`).
The last input byte holds the options; an input that starts with an IVF header is split into
its temporal units. Seeds: the parity corpus (`run.sh seeds`).

| Target | Entry point |
|---|---|
| `probe` | `av1::probe`: OBU framing, sequence, frame and tile group headers |
| `avif` | `av1::decode_avif`: isobmff-rs item graph, av1C, grid assembly, alpha (option bits: 2 threads, `apply_transforms`) |
| `decode` | `Decoder::decode_temporal_unit` on every temporal unit (option bits: Annex B, 2 threads, no film grain) |
| `frames_threads` | `Decoder::decode_temporal_units` with 1, 2, 3 and 8 threads against a loop of `decode_temporal_unit` on one thread: the same frames and the same error, or, when the ledger refused either side (`AllocLimit`: the batch books more), the same frames up to the refusal (option bits: Annex B, no film grain) |
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

`.github/workflows/fuzz.yml`: weekly (Thursday 12:23 UTC) and on dispatch, on GitHub-hosted
runners, up to eight jobs at a time, `-rss_limit_mb=2048 -malloc_limit_mb=512`, crashers
uploaded as artifacts.

Every target runs on x86_64 (`ubuntu-24.04`), and each `simd_*` target runs again natively on
aarch64 (`ubuntu-24.04-arm`; the same Dockerfile stage `fuzz`, ASan included), so the NEON tier
is compared with scalar. The dispatch input `only` picks one target. A hosted x86_64 runner does
not always have AVX-512, and a `simd_*` target compares the AVX-512 tier only where the CPU has
it: each job prints the CPU first, and the dispatch input `avx512` stops an x86_64 job at once
on a runner without AVX-512 (dispatch again). The arm64 corpus cache
(`fuzz-corpus-arm64-<target>`) starts from the x86_64 corpus of the same target; its artifacts
are `crashers-arm64-<target>` and `corpus-arm64-<target>`.

Before the move to hosted runners the arm64 jobs ran under qemu-user, without a sanitizer, and
`simd_threads` restarted every 1800 execs because qemu-user 7.2 kept about 280 KB of RSS per
exited guest thread (fuzz runs 45490215220 and 45519136759). Native runs need neither.

## Runs on main

| Date | Commit | max_total_time | Targets | Fuzz hours | Crashes | Runs |
|---|---|---|---|---|---|---|
| 2026-11-23 | a5cb385 | 900 s | 3 | 0.75 | 0 | probe 9,965,781 / decode 7,930 / avif 4,341 |
| 2026-11-26 | 70a9212 | 900 s | 2, x86_64 and arm64 | 1.0 | 0 | simd_cdef 4.6M x86_64 / 4.4M arm64, simd_lf 15.5M x86_64 / 18.7M arm64 (runs 45907052285, 45907055157) |
| 2026-11-27 | 028d393 | 900 s | all 13, 22 jobs (simd_* also arm64) | 5.5 | 0 | run 45956868631, x86_64 / arm64: probe 2.32M, decode 5,698, frames_threads 2,314, avif 2,878, simd_lf 10.9M / 18.8M, simd_grain 14.7M / 9.0M, simd_lr 490K / 581K, simd_cdef 5.29M / 4.55M, simd_ipred 5.53M / 6.30M, simd_itx 3.46M / 5.03M, simd_threads 23,785 / 30,323, simd_entropy 37,804 / 96,678, simd_alloc 197,728 / 189,529 |
| 2026-11-27 | ac5e3c4 | 900 s | all 13, 22 jobs (simd_* also arm64), after the speed PRs #67-#73 | 5.5 | 0 | run 45998086602, x86_64 / arm64: probe 2.86M, decode 7,057, frames_threads 4,389, avif 3,664, simd_lf 14.3M / 16.3M, simd_grain 13.0M / 9.13M, simd_lr 539K / 595K, simd_cdef 5.18M / 4.37M, simd_ipred 4.18M / 4.45M, simd_itx 2.48M / 5.07M, simd_threads 11,321 / 33,194, simd_entropy 87,485 / 124,984, simd_alloc 173,188 / 185,551 |

`decode` and `avif` run whole frames under ASan, so they do few runs per second.
