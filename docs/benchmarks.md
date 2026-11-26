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

## Runners

Both jobs run on GitHub-hosted runners: `bench` on `ubuntu-24.04` (x86_64, 4 vCPU, so "default
threads" is 4) and `bench-arm64` on `ubuntu-24.04-arm` (aarch64, 4 vCPU). The summary's first
line names the CPU model and `nproc`. Hosted runners are shared VMs and noisy: every figure is
the best of 5 runs, and a difference of a few percent between two runs means nothing. A hosted
x86_64 runner does not always have AVX-512; on one without it the v4 build cannot run, so the v4
columns and the avx512 tier show `-`. A dispatch with the input `avx512` stops the x86_64 job at
once on such a runner (dispatch it again) and skips `bench-arm64`. The numbers before the move (the self-hosted runner, 16
threads, AVX-512) are not comparable with these: compare runs on the same runner type only.

`bench-arm64` measures the NEON tier: dav1d 1.0.0 (bookworm's arm64 build, with its NEON asm;
`dav1d C` is the same binary with `--cpumask none`) and av1-rs with `AV1_SIMD=scalar` and with
NEON, with one thread and with the default threads. No aomdec and no v4 build there.

## Baseline on hosted runners

Bench run 45705147876 (dispatch, commit d86861a, the branch that moved CI to hosted runners).
`bench`: AMD EPYC 7763, 4 vCPU, no AVX-512, so this run has no v4 times and no avx512 tier.
Bench run 45709141819 (push to `main`, 523df8b) landed on an Intel Xeon 6975P-C with AVX-512
and has the v4 and avx512 figures in its job summary; they are not copied here.
`bench-arm64`: Neoverse-N2, 4 vCPU.

x86_64, 1 thread:

| class | streams | frames | dav1d ms | aomdec ms | av1-rs ms | av1-rs v4 ms | av1-rs / dav1d | av1-rs / aomdec |
|---|---|---|---|---|---|---|---|---|
| q8 | 5 | 5 | 27 | 49 | 33 | - | 122% | 67% |
| q10 | 5 | 5 | 46 | 94 | 74 | - | 160% | 78% |
| allintra | 2 | 78 | 293 | 570 | 681 | - | 232% | 119% |
| size | 16 | 16 | 52 | 88 | 24 | - | 46% | 27% |
| grain | 2 | 2 | 11 | 22 | 14 | - | 127% | 63% |
| avif8 | 4 | 4 | 45 | 106 | 97 | - | 215% | 91% |
| avifhi | 4 | 4 | 109 | 293 | 256 | - | 234% | 87% |
| tiles8 | 1 | 8 | 76 | 172 | 273 | - | 359% | 158% |
| tiles10 | 1 | 8 | 84 | 213 | 280 | - | 333% | 131% |
| all | | | 743 | 1607 | 1732 | - | 233% | 107% |

x86_64, default threads (4):

| class | streams | frames | dav1d ms | aomdec ms | av1-rs ms | av1-rs v4 ms | av1-rs / dav1d | av1-rs / aomdec |
|---|---|---|---|---|---|---|---|---|
| q8 | 5 | 5 | 35 | 51 | 33 | - | 94% | 64% |
| q10 | 5 | 5 | 57 | 93 | 71 | - | 124% | 76% |
| allintra | 2 | 78 | 170 | 568 | 317 | - | 186% | 55% |
| size | 16 | 16 | 73 | 88 | 30 | - | 41% | 34% |
| grain | 2 | 2 | 14 | 22 | 15 | - | 107% | 68% |
| avif8 | 4 | 4 | 50 | 102 | 88 | - | 176% | 86% |
| avifhi | 4 | 4 | 128 | 292 | 220 | - | 171% | 75% |
| tiles8 | 1 | 8 | 39 | 85 | 133 | - | 341% | 156% |
| tiles10 | 1 | 8 | 46 | 104 | 132 | - | 286% | 126% |
| all | | | 612 | 1405 | 1039 | - | 169% | 73% |

x86_64 SIMD tiers (default build, 1 thread; AVX2 is the best tier on this CPU):

| class | scalar ms | avx2 ms | avx512 ms | avx2 / scalar |
|---|---|---|---|---|
| q8 | 69 | 34 | - | 49% |
| q10 | 172 | 75 | - | 43% |
| allintra | 1243 | 684 | - | 55% |
| size | 84 | 29 | - | 34% |
| grain | 33 | 14 | - | 42% |
| avif8 | 226 | 97 | - | 42% |
| avifhi | 567 | 256 | - | 45% |
| tiles8 | 934 | 273 | - | 29% |
| tiles10 | 963 | 277 | - | 28% |

aarch64 NEON, 1 thread:

| class | dav1d ms | dav1d C ms | av1-rs scalar ms | av1-rs neon ms | neon / scalar | neon / dav1d |
|---|---|---|---|---|---|---|
| q8 | 27 | 38 | 48 | 27 | 56% | 100% |
| q10 | 48 | 78 | 121 | 60 | 49% | 125% |
| allintra | 273 | 356 | 840 | 486 | 57% | 178% |
| size | 56 | 75 | 60 | 28 | 46% | 50% |
| grain | 11 | 16 | 23 | 13 | 56% | 118% |
| avif8 | 48 | 88 | 157 | 80 | 50% | 166% |
| avifhi | 123 | 235 | 381 | 215 | 56% | 174% |
| tiles8 | 82 | 230 | 631 | 235 | 37% | 286% |
| tiles10 | 93 | 243 | 644 | 240 | 37% | 258% |
| all | 761 | 1359 | 2905 | 1384 | 47% | 181% |

aarch64 NEON, default threads (4):

| class | dav1d ms | dav1d C ms | av1-rs scalar ms | av1-rs neon ms | neon / scalar | neon / dav1d |
|---|---|---|---|---|---|---|
| q8 | 30 | 39 | 35 | 27 | 77% | 90% |
| q10 | 54 | 64 | 74 | 50 | 67% | 92% |
| allintra | 137 | 156 | 249 | 155 | 62% | 113% |
| size | 71 | 83 | 40 | 24 | 60% | 33% |
| grain | 13 | 15 | 16 | 12 | 75% | 92% |
| avif8 | 44 | 58 | 94 | 62 | 65% | 140% |
| avifhi | 107 | 178 | 245 | 163 | 66% | 152% |
| tiles8 | 29 | 67 | 170 | 70 | 41% | 241% |
| tiles10 | 33 | 72 | 170 | 70 | 41% | 212% |
| all | 518 | 732 | 1093 | 633 | 57% | 122% |

NEON roughly halves the scalar time (47% over all classes with one thread) and is 1.8x dav1d
with one thread, 1.2x with four. dav1d's own NEON asm saves less here (dav1d / dav1d C 55% with
one thread), so the gap to dav1d with one thread is smaller on aarch64 (1.8x) than on x86_64
with AVX2 (2.3x). `dav1d C` shows that the NEON asm is in use: the same dav1d with
`--cpumask none` is slower in every class.

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
they changed nothing measurable (v4 builds with and without them on the self-hosted runner used
before the move to hosted runners: the same to within 1 to 2 ms per class and the same 894/898 ms
totals), so the build dropped them and has no
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

The job summary starts with the runner's CPU model and AVX-512 subsets. With `simd=avx512`
the job stops before the image build on a runner without AVX-512; dispatch it again.

### The SIMD kernels before the AVX-512 kernels

Profile run 45889645675 (`simd=avx512`, main e0275cf plus the CPU line), AMD EPYC 9V45 (Zen 5,
4 vCPU, AVX-512 F, BW, VL, DQ, CD, VBMI, VBMI2, VNNI, IFMA, BITALG, VPOPCNTDQ, BF16). The
AVX-512 tier then ran the AVX2 kernels. Self time over all classes, inlining off
(`all.noinline.symbol.txt`), one thread:

| kernel family | AVX2 symbols | share |
|---|---|---|
| itx | `inverse_transform_add` 5.5%, `transform_1d` 4.0% (N = 8: 1.8%, 16: 1.4%, 32: 0.6%, 64: 0.2%) | 9.5% |
| cdef | `filter` 4.8%, `dir_costs` 0.7% | 5.5% |
| lf | `filter` | 1.8% |
| lr | `box_pass` 1.0%, `weigh` 0.2% (and `wiener`, 1.3% in avifhi) | 1.2% |
| ipred | `smooth` 0.8%, `cfl` 0.4%, `filter_edge` 0.2% | 1.4% |
| entropy | `adapt` | 0.5% |

The rest is scalar: coefficient and symbol decoding (`coeffs` 27%, `transform_block` 17%,
`read_levels`, `read_symbol`), the CDEF and loop filter drivers (`cdef::filter` 4.5%,
`lf::filter_run`, `lf::edge_params`) and `memmove`/`memset`. By class, CDEF is the largest
kernel in tiles8 (9% + 8% for its driver) and q10; itx in allintra (9%) and avifhi; the loop
filter is 4% in avifhi and tiles8.

## Earlier runs

2026-11-23, main 49f2864 (old bench, 24 streams, 100 frames, `.yuv` written by av1-rs only):
dav1d 200 ms / av1-rs 1005 ms with one thread (5.0x); dav1d 115 ms / av1-rs 1134 ms with
default threads (9.9x; these streams have one tile).

The push benches of the #32 merge (3686fe1, run 45534769417) and the #33 merge (ab67402, run
45544772498) were cancelled. 10c1f47 (run 45550778331) is the first clean post-merge record after
both.
