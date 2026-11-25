# Benchmarks

Information only: there is no speed bar. `.github/workflows/bench.yml` runs
`tests/parity/bench.sh` in the `bench` image (push to `main` and dispatch). dav1d 1.0.0, aomdec
3.6.0 and av1-rs (`examples/oracle.rs --null`) decode the same frames of each bench class, once
with one thread and once with the default threads (`nproc`). av1-rs runs twice: the default
release build (runtime SIMD dispatch) and a release-binary build
(`-C target-cpu=x86-64-v4`, fat LTO,
`panic=abort`), and a third table times the default build with each SIMD tier forced
(`AV1_SIMD`). The figure is the wall time of the whole class, best of 5. No decoder writes
samples (`--null`, `-o /dev/null`).

## The v4 build

`-C target-cpu=x86-64-v4` makes `rustc 1.99.1` enable these target features
(`rustc -C target-cpu=x86-64-v4 --print cfg`): avx, avx2, avx512bw, avx512cd, avx512dq, avx512f,
avx512vl, bmi1, bmi2, cmpxchg16b, f16c, fma, fxsr, lzcnt, movbe, popcnt, sse, sse2, sse3, sse4.1,
sse4.2, ssse3, xsave.

An earlier build also passed `-C target-feature=+false-deps-tzcnt,+false-deps-bls`. Both
names are real LLVM tuning features (`rustc --print target-features` lists them with
`false-deps-lzcnt`, `-popcnt`, `-perm`, `-range`, `-getmant`, `-mulc`, `-mullq`, `-compress`
and `-expand`); `false-deps-lzcnt-tzcnt` is not one. rustc does not know them, so it warns
"unknown and unstable feature specified for `-Ctarget-feature`" and passes them to LLVM, which
accepts them. They work: with them, tzcnt and blsr in a loop get an `xor` of the destination
first (the same as `-C target-cpu=znver5`; `znver4` and plain v4 have none). On this decoder
they changed nothing measurable (v4 column of the 1 thread and default threads tables, same
to within 1 to 2 ms per class and the same 894/898 ms total), so the build dropped them and has no
warning. `tests/parity/v4-diag.sh` (workflow `v4-diag`, dispatch only) prints the toolchain's
list and the asm, to recheck on a new rustc.

## Classes

`tests/parity/bench-lib.sh` builds them from the parity corpus:

| Class | Inputs |
|---|---|
| q8 | libaom `av1-1-b8-00-quantizer-{00,16,31,48,63}` (352x288, 8-bit) |
| q10 | libaom `av1-1-b10-00-quantizer-{00,16,31,48,63}` (10-bit) |
| allintra | libaom `av1-1-b8-02-allintra` |
| size | libaom `av1-1-b8-01-size-{196,200,208,226}x{196,200,208,226}` |
| grain | libaom `av1-1-b{8,10}-23-film_grain-50` (film grain on) |
| avif8 | the AV1 items of `fox.profile0.8bpc.yuv420`, `kimono`, `kodim03`, `kodim23` |
| avifhi | the AV1 items of `fox.profile0.10bpc.yuv420`, `fox.profile2.12bpc.yuv444`, `cosmos1650_yuv444_10bpc_p3pq`, `hato.profile0.10bpc.yuv420.no-cdef.no-restoration` |
| tiles8, tiles10 | the fox picture (8-bit, 10-bit) 8 times, every frame a key frame, 4 x 2 tiles, encoded in CI by aomenc (the corpus has no large multi-tile intra stream) |

Each decoder decodes the frames that av1-rs outputs (the leading intra frames of a stream).

The result is the job summary of the `bench` run (`summary.md`). A command that exits non-zero
is listed in the summary as a warning, and the times of that run mean nothing. The table of the
latest run on `main` is the record; the speed PRs copy theirs into the PR description.

## Profile

`.github/workflows/profile.yml` (dispatch only, input `simd` sets `AV1_SIMD`) runs
`tests/parity/profile.sh` on the same classes with one thread: an oracle built with line tables
and frame pointers under `perf record` (in a `--privileged` container, because a BuildKit RUN
cannot open perf events), or under callgrind when perf cannot record. The job summary lists the
self time by source file per class; the artifact has the per-symbol and per-line reports. Never
profile on a workstation: CI only.

## Earlier runs

2026-11-23, main 49f2864 (old bench, 24 streams, 100 frames, `.yuv` written by av1-rs only):
dav1d 200 ms / av1-rs 1005 ms with one thread (5.0x); dav1d 115 ms / av1-rs 1134 ms with
default threads (9.9x; these streams have one tile).
