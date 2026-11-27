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

A dispatch with the input `base` (a branch, a tag or a full commit SHA: `actions/checkout`
cannot fetch a short SHA) also builds the default build of that
commit in the same image and puts an A/B table first in the summary: the base and this commit,
one run of each after the other in each of 7 rounds, best of the 7, with one thread and with the
default threads, the ratio in tenths of a percent. Both builds run on the same machine minutes
apart, so the A/B ratio is the figure for a change of a few percent, where two separate runs
differ by more than the change. The base must have the same `Cargo.lock` (the build is offline).

The result is the job summary of the `bench` run (`summary.md`). A command that exits non-zero
is listed in the summary as a warning, and the times of that run mean nothing. The table of the
latest run on `main` is the record; the speed PRs copy theirs into the PR description.

## Profile

`.github/workflows/profile.yml` (dispatch only, input `simd` sets `AV1_SIMD`) runs
`tests/parity/profile.sh` on the same classes with one thread: an oracle built with line tables
and frame pointers under `perf record` (in a `--privileged` container, because a BuildKit RUN
cannot open perf events), or under callgrind when perf cannot record. The job summary lists the
self time by source file per class; the artifact has the per-symbol and per-line reports. Never
profile on a workstation: CI only. dav1d (bookworm's build, its asm on, symbols from
`libdav1d6-dbgsym`) is profiled on the same frames with one thread: the "dav1d, all" list in the
summary and `all.dav1d.symbol.txt`.

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

## AVX-512 kernels

Each candidate was benched on its branch (`bench` dispatch with `avx512`): the SIMD tiers table
times the same build with `AV1_SIMD=avx2` and with the AVX-512 tier in one run on one machine.
The figure is the sum of the nine classes, one thread; a kernel stays only if it is faster in
every run. No run showed a class that got slower with the AVX-512 tier (no sign of a
frequency drop on these CPUs).

| kernel | change | runs: CPU, avx2 ms, avx512 ms, avx512 / avx2 | result |
|---|---|---|---|
| cdef `filter` | 32 samples per vector (8x8, 8x4, 4x8 blocks), AVX-512 BW | 45895171894: Xeon Platinum 8573C, 1618, 1584, 97.9%; 45898261819: Xeon 6975P-C, 1274, 1263, 99.1% | kept |
| lf `filter` | 32 samples along the edge per vector for runs longer than 16, mask registers, AVX-512 BW | 45895175485: AMD EPYC 9V74, 1366, 1358, 99.4%; 45896617239: Xeon Platinum 8573C, 1753, 1742, 99.4%; 45898056586: Xeon 6975P-C, 1271, 1259, 99.1% | kept: the gain is in avifhi, tiles8 and tiles10 (2% to 3% there) |
| itx | `itx_kernels!` on 16 `i32` lanes for blocks at least 16 wide, AVX-512 F (commit f13d71f) | 45892560939: Xeon 6975P-C, 1266, 1267, 100% | dropped: most of the itx time is in blocks up to 8 wide (`transform_1d::<_, 8>`), which 16 lanes cannot fill |
| lr Wiener, self guided | 16 `i32` lanes for blocks at least 16 wide, AVX-512 F (commit 993c03a) | 45896614027: AMD EPYC 9V45, 1077, 1076, 100% | dropped: 1.2% of the time, below what a bench can show |

Not tried (each below 1% of the time in the profile): ipred `smooth`, `cfl`, `filter_edge`,
cdef `dir_costs`, entropy `adapt`.

## Speed work of 2026-11-27 (entropy, coefficients and the filter drivers)

Profile run 45968771362 (main 427b329 plus the dav1d profile, Intel Xeon 6975P-C, one thread, all
classes): av1-rs and dav1d on the same frames. Self time, av1-rs with inlining off
(`all.noinline.symbol.txt`) against dav1d (`all.dav1d.symbol.txt`; dav1d takes about 0.43 of
the av1-rs time):

| av1-rs | share | dav1d | share |
|---|---|---|---|
| `TileDecoder::coeffs` | 26.4% | `decode_coefs` (two copies) | 15.5% |
| `TileDecoder::transform_block` (intra edges, prediction, dequant inlined) | 14.9% | `msac_decode_symbol_adapt4` with its renorm labels | 19.7% |
| `simd::cdef::filter` (the driver) | 5.6% | `recon_b_intra` (8 and 16 bpc) | 6.1% |
| `itx::avx2::inverse_transform_add` | 4.9% | `msac_decode_bool_equi` | 2.7% |
| `coef::read_levels` br closure | 4.5% | `decode_b` | 3.0% |
| `SymbolDecoder::read_symbol` | 4.0% | `prepare_intra_edges` (8 and 16 bpc) | 2.8% |
| `lf::filter_run` | 3.0% | `msac_decode_symbol_adapt8`, `adapt16` | 3.8% |
| `lf::edge_params` | 2.9% | `decode_sb` | 1.5% |
| `cdef::avx512::filter` | 2.5% | `msac_decode_bool_adapt` | 1.3% |
| `memmove` | 2.4% | `selfguided_filter` | 1.3% |
| `decode_block` | 2.2% | `create_lf_mask_intra` | 1.2% |
| `cdef::avx2::filter` | 2.0% | `cdef_filter_8x8_16bpc_avx2` | 1.0% |
| `itx::avx2::transform_1d::<8>` | 1.9% | `memmove` | 0.9% |
| `itx::avx2::transform_1d::<16>` | 1.5% | `msac_decode_hi_tok` | 0.7% |
| `tile::fill_rows` | 1.4% | `cdef_filter_4x4_16bpc_avx2` | 0.6% |

Where av1-rs spends time that dav1d does not: the coefficient and symbol decoding (about 53% of
the av1-rs time against 43% of the dav1d time, so about 2.8 times the cycles), the CDEF driver
(a padded copy of every 8x8 block and no skip of zero-strength work: the driver alone took about
4.7 times all of dav1d's CDEF), the loop filter driver (`edge_params` at every 4x4 position) and
the zeroed intra edge arrays.

Each change was measured with the A/B table (bench input `base`) against the change before it,
one thread, all classes; a change stayed only if it was faster on x86_64 by about 1% or more, or
faster on both runners.

| change | A/B run | x86_64 CPU | x86_64 1 thread | aarch64 1 thread | result |
|---|---|---|---|---|---|
| branchless search of rows of up to 4 symbols (`read_cdf`) | 45974090122 | Xeon 6975P-C | 99.7% | 99.9% | dropped |
| dequantize in the sign pass, only the nonzero coefficients | 45978308736 | Xeon 6975P-C | 98.4% | 98.7% | kept (#67) |
| `read_cdf` at every fixed-size CDF site | 45978313092 | Xeon 6975P-C | 99.2% | 100.0% | kept (#68) |
| CDEF: skip zero-strength work, as dav1d | 45979960691 | Xeon 6975P-C | 93.9% | 91.3% | kept (#69) |
| CDEF: interior blocks read straight from the plane | 45979964804 | Xeon 6975P-C | 99.5% | 98.7% | kept (#70) |
| intra edges: persistent scratch, only the entries read | 45981476078 | Xeon Platinum 8573C | 99.1% | 97.7% | kept (#71) |
| loop filter: transform block segment walk | 45982986040 | EPYC 7763 | 97.3% | 99.1% | kept (#72) |
| symbol decoder: inline adaptation of every row, window not inverted | 45982990244 | Xeon 6973P-C | 100.5% | 99.5% | dropped |
| coefficients: one context byte, branchless list link | 45984571106 | EPYC 9V74 | 100.0% | 100.3% | dropped |
| CDEF: primary-only and secondary-only kernels | 45984575440 | EPYC 9V45 | 101.3% | 97.8% | dropped |
| loop filter: vertical edges by transposed loads | 45986169840 | Xeon 6975P-C | 98.3% | 97.1% | kept (#73) |
| ipred: AVX2/NEON directional zones 2 and 3 | 45986174416 | EPYC 7763 | 100.3% | 99.6% | dropped |

The dropped symbol decoder changes confirm the earlier result: the scalar search with early exit
and the inline adaptation of rows of up to 4 symbols are as fast as dav1d's shapes here. The
work stopped after three changes in a row below 1% on x86_64 (the symbol decoder, the context
byte and the CDEF kernel variants).

All seven kept changes against main a4a1167, one machine (A/B run 46003957795, the base and
main ac5e3c4 with the same dav1d times in one run):

| runner | av1-rs, this / base, 1 thread | 4 threads | av1-rs / dav1d before, 1 thread | after | before, 4 threads | after |
|---|---|---|---|---|---|---|
| x86_64 Intel Xeon 6975P-C | 1232 / 1071 ms, 86.9% | 724 / 644 ms, 88.9% | 2.14x | 1.86x | 1.64x | 1.46x |
| aarch64 Neoverse-N2 (NEON) | 1385 / 1166 ms, 84.1% | 677 / 610 ms, 90.1% | 1.81x | 1.55x | 1.27x | 1.15x |

The largest single gain was the CDEF zero-strength skip (most bench streams code zero
strengths for many blocks), then the loop filter segment walk and transposed loads, and the
dequantization in the sign pass.

After the changes (profile run 45998091759, main ac5e3c4, AMD EPYC 9V74, one thread, inlining
off): `coeffs` 34.2%, `transform_block` 12.9%, `itx::avx2::inverse_transform_add` 6.4%,
`read_levels` br closure 5.1%, `decode_block` 3.5%, `intra_frame_mode_info` 2.8%,
`cdef::avx512::filter` 2.8%, `memmove` 2.7%, `lf::Lines::line` 2.4%,
`itx::avx2::transform_1d::<8>` 1.9%, `<16>` 1.7%, `cdef::avx2::filter` 1.5%,
`lf::avx2::kernel` 1.4%, `tile::fill_rows` 1.4%, `lf::avx2::filter_v` 1.3%. The CDEF driver
(5.6% before) and `lf::edge_params` (2.9%) left the list. The gap to dav1d is now mostly the
coefficient and symbol decoding in `coeffs` and `read_levels`; the symbol decoder shapes tried
here (dav1d's branchless search, inline vector adaptation, the non-inverted window) did not
make it faster.

## Speed work, round 3 (codegen and bounds checks)

Start: main 5b0f0d1, bench run 46017403904. av1-rs / dav1d, all classes: x86_64 AMD EPYC 9V45
941 / 492 ms with 1 thread (1.91x), 573 / 406 ms with 4 threads (1.41x); aarch64 Neoverse-N2
1153 / 754 ms (1.53x) and 567 / 501 ms (1.13x).

### Where the cycles of the hot functions go

Profile run 46032212917 (main 5b0f0d1 plus `perf annotate` of the 10 hottest functions of the
build with inlining off, `all.noinline.annotate.<n>.txt`; AMD EPYC 7763, no AVX-512, one
thread). Self time: `coeffs` 34.4%, `transform_block` 12.8%, `itx::avx2::inverse_transform_add`
6.5%, `read_levels` br closure 4.9%, `cdef::avx2::filter` 3.9%, `memmove` 3.5%, `decode_block`
3.3%, `lf::Lines::line` 2.5%, `intra_frame_mode_info` 2.4%. dav1d on the same frames:
`decode_coefs` 16.2%, the `msac_decode_symbol_adapt4` entry and its renorm labels 22.8%,
`decode_b` 3.9%, `recon_b_intra` 4.4%.

The instructions of `coeffs` (with `read_levels`, `read_cdf`, `norm` and `refill` inlined), by
kind, as a share of its samples:

| kind | share | what it is |
|---|---|---|
| ALU (the `cur` products, the adaptation, the context sums) | 50% | the work itself |
| loads and stores of the stack frame | 10% | spilled locals and the reload of the `SymbolDecoder` pointer |
| other loads and stores | 10% | levels, CDFs, and `dif`, `rng`, `cnt` of the `SymbolDecoder` (offsets 0x168 to 0x178 from its pointer) |
| compares and branches | 19% | the search, the loop, the bounds checks |
| shifts | 9% | `shl %cl` of the renorm (`dif`, `rng`) and the class shifts of `row`, `col` |
| `bsr` + `xor $0x1f` (`leading_zeros`) | 0.5% | the renorm count; one more instruction than `lzcnt` |

- Decoder state: every symbol loads `dif` and `rng` from memory, and stores `dif`,
  `rng` and `cnt` back, and reloads `disable_cdf_update`, because the br closure is a call
  (`read_levels::<0>::{closure#0}`, 4.9% on its own) that takes the decoder by reference. The
  store of `dif` and the load of the next symbol are on the loop-carried chain.
- Bounds checks: per coefficient, five compares against the levels length (`cmp $0x510`, one for
  each neighbour), one against the 42 base CDFs and one for the level store. They are well
  predicted and off the dependency chain: about 3% of the `coeffs` samples, so about 1% of the
  run. No panic path is hot.
- The hottest single instructions follow the load of the CDF value whose address comes from the
  context (neighbours, sum, context, CDF load, `cur`, compare): the load latency, not the check.
- Renorm: `leading_zeros` compiles to `bsr` + `xor` and the shifts by a variable count to
  `shl %cl` (three micro-ops on Intel, one with BMI2 `shlx`) in the baseline x86-64 build.
- `transform_block`: 22% of its samples are stack loads and stores; the rest is the scalar parts
  of the intra prediction (edge preparation, the directional interpolation `edge[ix(base)]`,
  filter intra, `cfl_ac`), spread over many lines, with no single hot spot.

### The v4 build against the default build

Bench run 46017403904 (main 5b0f0d1, AMD EPYC 9V45 with AVX-512), 1 thread, default / v4
(`-C target-cpu=x86-64-v4`, fat LTO, panic=abort): q8 19 / 19 ms, q10 45 / 45, allintra
392 / 385, size 13 / 18, grain 8 / 9, avif8 56 / 55, avifhi 140 / 137, tiles8 132 / 130,
tiles10 136 / 133, all 941 / 931 ms (98.9%). The coefficient-heavy classes (allintra, avif,
tiles) gain 1.5 to 2.2%; the v4 build is slower on the small `size` and `grain` classes. On
Intel hosts the v4 build gains more, default / v4, 1 thread, all classes: 1108 / 1047 ms
(94.5%) on a Xeon 6973P-C (run 46035612706, the build of change A below) and 1064 / 1035 ms
(97.3%) on a Xeon 6975P-C (run 46040173047, the build of change C below, whose default build
already has the v3 twins), where `shl %cl` costs more micro-ops than on AMD. The v4 build
changes every function (and adds fat LTO and panic=abort), so these are upper bounds for a v3
build of the tile decode loop only.

### Changes tried

Each change was measured with the A/B table (bench input `base` = main 5b0f0d1, full SHA)
against the change before it, one thread, all classes; the rule was the same as in the round
before (about 1% or more on x86_64, or faster on both runners). The A/B runs of A and B ran from
their branches without a PR (to keep within two CI runs at a time); C had PR #77.

| change | branch, head | A/B run | x86_64 CPU | x86_64 1 thread | aarch64 1 thread | result |
|---|---|---|---|---|---|---|
| A: the `SymbolDecoder` as a local copy in `read_levels` and the sign pass, the br reads inlined (no call, so `dif`, `rng`, `cnt` can stay in registers) | `speed3-sdloc-2b8e`, 6ba88ee | 46035612706 | Xeon 6973P-C | 1095 / 1095 ms, 100.0% | 1129 / 1138 ms, 100.7% | dropped |
| B: levels and `Quant[ ]` indexed masked (levels buffer of 2048 bytes, a power of two), no bounds checks on them | `speed3-bchk-9e1a`, 5fd4bf9 | 46037506121 | Xeon 6975P-C | 1072 / 1090 ms, 101.6% | 1160 / 1162 ms, 100.1% | dropped |
| C: `coeffs` and `intra_frame_mode_info` built a second time for x86-64-v3 (`avx2,bmi1,bmi2,lzcnt,popcnt,fma`), chosen by the tier (`simd::codegen`) | `speed3-v3-5f1c`, b34db13 | 46038594929 | EPYC 7763 | 1459 / 1455 ms, 99.7% | 1148 / 1147 ms, 99.9% | dropped |
| C again, a second x86_64 host | same | 46040173047 | Xeon 6975P-C | 1073 / 1066 ms, 99.3% | 1140 / 1139 ms, 99.9% | dropped |

- A: the copy did not make the loop faster. Either LLVM keeps the window in memory anyway
  (the `refill` path), or the store and load forward fast enough that they are not the limit.
- B: slower. The `and` of the mask sits on the context path (neighbour loads, context, CDF
  load), which is the dependency chain of the loop, while the bounds checks it replaces are
  predicted branches off that chain. So the bounds checks cost less than a mask.
- C: below 1% on both x86_64 hosts. The "faster on both runners" part of the rule does not
  apply: C changes no aarch64 code (`simd::codegen` runs the baseline build there), so the
  aarch64 99.9% (1 ms) is noise, not a gain.

The work stopped after these three changes in a row below 1% on x86_64.

PGO (bench column from PR #78, CI only; the crate does not change; trained on the bench
classes, so this is the best case), default / PGO build, all classes, bench run 46035159115
(the percent is PGO / default): x86_64 AMD EPYC 7763 1497 / 1493 ms (99.7%) with 1 thread,
936 / 921 ms (98.3%) with 4 threads; aarch64 Neoverse-N2 1166 / 1148 ms (98.4%) and
619 / 614 ms (99.1%). Bench run 46041900099 (main after PR #78): x86_64 Intel Xeon 6975P-C
1050 / 1023 ms (97.4%) with 1 thread, 635 / 613 ms (96.5%) with 4 threads; aarch64 1143 / 1124
ms (98.3%) and 596 / 594 ms (99.6%). A user who builds
with PGO gets 0 to 3% (most on Intel x86_64, least on AMD); the small `size` and `grain` classes get slower (up to 26% on `size` on
aarch64), tiles8 and tiles10 get slower on the EPYC.

Unsafe (owner rule of this round: allowed in the entropy and coefficient code only where a
bounds check costs time and safe Rust cannot remove it): not tried. The bounds checks of the
coefficient loop take about 3% of the `coeffs` samples (about 1% of the run) and sit off the
dependency chain, so unchecked indexing can give at most about 1%; the stop rule came first.

## Earlier runs

2026-11-23, main 49f2864 (old bench, 24 streams, 100 frames, `.yuv` written by av1-rs only):
dav1d 200 ms / av1-rs 1005 ms with one thread (5.0x); dav1d 115 ms / av1-rs 1134 ms with
default threads (9.9x; these streams have one tile).

The push benches of the #32 merge (3686fe1, run 45534769417) and the #33 merge (ab67402, run
45544772498) were cancelled. 10c1f47 (run 45550778331) is the first clean post-merge record after
both.
