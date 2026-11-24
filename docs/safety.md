# Safety

`src/lib.rs` has `#![deny(unsafe_code, unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]`.
`unsafe` is allowed in two places only, each with a `// SAFETY:` comment:

1. The load/store helpers, `src/simd/x86.rs` and `src/simd/arm.rs` (`#![allow(unsafe_code)]`).
   Each helper takes an array reference (`&[u16; 16]`, `&mut [u8; 32]`, ...), so the type
   proves that every byte of the load or store is in bounds, and each is a
   `#[target_feature]` fn, so a caller must have the feature too.
2. The dispatch fn of each kernel family (`#[allow(unsafe_code)]` on that fn only). It matches
   on a `simd::Simd` and calls the `#[target_feature]` kernel of that tier. A `Simd` has a
   private field and only `simd::available()` and `simd::best()` make one, after runtime
   detection, so a dispatch fn can only reach a kernel the CPU has.

The kernels themselves are safe `#[target_feature]` fns: inside them the non-pointer
`std::arch` intrinsics are safe calls, and memory is reached only through the helpers above
and safe slice code.

There is no inline assembly and no unsafe threading (docs/threading.md).

## SIMD tiers

`src/simd/mod.rs` detects the tiers once: scalar (always), AVX2 (with FMA, BMI1, BMI2) and
AVX-512 (F, BW, VL, DQ, CD) on x86_64, NEON on aarch64. `AV1_SIMD=scalar|avx2|avx512|neon`
(read once; empty or unset = best) caps the tier; a tier the CPU lacks falls back to the best
one below it. This variable is the only one the crate reads.

The scalar code of every kernel stays and is the oracle: each SIMD kernel must give its output
bit for bit. CI checks this four ways:

- Unit tests per kernel family compare every available tier with scalar on many
  pseudo-random inputs (own xorshift) and on the edge sizes and values.
- `cargo test` runs three times in `test` (best tier, `AV1_SIMD=scalar`, `AV1_SIMD=avx2`) and
  twice under qemu in `arm64` (NEON, scalar).
- The parity gate decodes the corpus with the best tier against dav1d and aomdec, and with the
  scalar and AVX2 tiers against the best tier (gate `tiers`); parity-arm64 compares NEON and
  scalar on aarch64 with x86_64.
- A cross-tier fuzz target per kernel family (docs/fuzzing.md).

### Layout of a kernel family

- `src/simd/<family>/mod.rs`: the dispatch fn(s), `pub(crate) fn <kernel>(simd: Simd, ...)`,
  with the bounds checks in safe code before the match, and the cross-tier tests.
- `src/simd/<family>/{avx2,avx512,neon}.rs`: the kernels, no `unsafe`.
- The scalar oracle stays where it was (`src/cdef.rs`, `src/transform.rs`, ...); the scalar
  arm of the dispatch calls it.

Production code passes `simd::best()`; tests loop over `simd::available()`.

### Kernel families

| Family | Oracle | Kernels | Tiers |
|---|---|---|---|
| `lf` | `lf::sample_filter` | One run of samples along an edge (7.14.6): masks, narrow filter, 6/8/16 wide filters, vertical and horizontal, 8/10/12-bit; lanes along the edge, the samples across it gathered into a 14-row buffer in safe code | AVX2 (16 lanes), NEON (8 lanes); AVX-512 runs the AVX2 kernel |
| `grain` | `grain::blend_luma_row`, `grain::blend_chroma_row` | `blend_luma`, `blend_chroma`: the film grain noise blend of one row (scaling lookup, `round2( scale * noise, ScalingShift )`, clip, chroma merge and chroma scaling from luma), 8/10/12-bit, all subsamplings. The scale is looked up in `grain::scale_table` (`scale_lut` at every sample value), so the high bit depth interpolation is exact by construction; the x86 helper `gather256_i32` masks every index to the 4096-entry table before the gather. The grain template generation (auto-regression) and the stripe building stay scalar (once per frame, and a copy) | AVX2 (8 lanes), NEON (4 lanes); AVX-512 runs the AVX2 kernel |
| `lr` | `lr::wiener_filter`, `lr::self_guided_filter` | `wiener`, `self_guided`: loop restoration of one block of at most 64 x 64 samples inside one stripe, from the source window that `lr::source_window` gathers (stripe edge rules unchanged): the Wiener filter and the self guided filter (box sums r = 1 and r = 2, the a/b computation, the weighted combination). The arithmetic is in 32-bit lanes; the module doc gives the bounds that make it exact. Blocks narrower than one vector use the oracle | AVX2 (8 x i32), NEON (4 x i32); AVX-512 runs the AVX2 kernel |
| `cdef` | `cdef::cdef_direction`, `cdef::cdef_filter` | `direction` (7.15.2, one 8x8 luma block), `filter` (7.15.3, one 8x8, 4x4, 4x8 or 8x4 block), 8/10/12-bit. The filter copies the block and a 2-sample border into a padded 16-bit block with the sentinel `0x8000` for unavailable taps (dav1d's design, see NOTICE), so the kernel has no per-tap check. The dispatch takes the oracle for a block whose premises do not hold (block not fully inside `MiRows` x `MiCols`, a sample of 4096 or more, strengths above 255, a plane too short); decoded frames always meet them (`MiRows` and `MiCols` are even, samples fit the bit depth) | AVX2 (16 lanes), NEON (8 lanes); AVX-512 runs the AVX2 kernels |
| `ipred` | `src/predict.rs` | Paeth; Smooth, Smooth_V, Smooth_H; directional zone 1 without upsampling; intra edge filter; palette (rows of 8); CFL alpha step | AVX2 (8 x i32), NEON (4 x i32); AVX-512 runs the AVX2 kernels |
| `itx` | `transform::inverse_transform_add` | 2D inverse transform + add/clip into the samples: DCT 4..64, ADST 4/8/16, identity 4..32, flips, rect scaling, row/column shifts and clamps | AVX2 (8 x i32), NEON (4 x i32); AVX-512 runs the AVX2 kernel (the 16-lane kernel was dropped, see below) |
| `entropy` | `symbol::adapt_cdf` | `adapt_cdf`: the CDF update of `read_symbol` for 4 to 16 symbols | AVX2, NEON; AVX-512 runs the AVX2 kernel (at most 16 lanes of `u16`, nothing for wider vectors to do) |

Tiers not written:

- `lf` AVX-512: a run is one transform edge (4 to 64 samples, mostly 4 to 16), so 32 lanes
  would be mostly padding, and the gather and scatter around the kernel cost the same per sample
  at either width.
- `grain` AVX-512: the 8-lane gather of the scale bounds the loop, and a 16-lane gather is not
  faster per sample.
- `lr` AVX-512: a block row is at most 64 samples, so 16-lane vectors would save at most half of
  the vector ops while the fixed per-block work (window gather, the stack buffers) stays; it is
  worth writing only if the bench shows loop restoration still high in the profile after AVX2.
- `cdef` AVX-512: a block is 16 to 64 samples and each tap row is a separate 8- or 4-sample
  load, so 512-bit vectors would only add lane inserts (the direction search is one 8x8 block,
  128-bit rows).

`ipred` notes:

- Every SIMD tier also predicts V_PRED and H_PRED (pAngle 90 and 180) as row copies and fills
  instead of the per-sample directional loop, and computes the CFL luma average once per
  sample (the oracle computes it twice) before the alpha kernel. Same arithmetic, so the same
  output.
- Not written, so the scalar oracle runs on every tier: DC (DC_TOP / LEFT / 128 included: one
  sum and a `fill` per row, already memset speed), directional zones 2 and 3 (per-sample
  source positions; they need gathers or a transpose, left for a later round), upsampled edges
  (at most 16 entries, only for blocks with w + h <= 16), the upsample itself, recursive filter
  intra (7-tap dependent cells; not in the profile).
- AVX-512: not written; the Avx512 tier runs the AVX2 kernels. Rows are 4 to 64 samples and
  most intra blocks are 4 to 16 wide, so 16-lane vectors would only help the widest blocks.
- Code written for this crate; dav1d's ipred was not ported.

`itx` notes: all tiers use `i32` lanes for 8, 10 and 12 bit. At 8 and 10 bit every rotation sum
fits `i32` (the dequantizer clamp bounds the inputs); at 12 bit each rotation splits its inputs
into `hi * 4096 + lo` so the wrapped result equals the scalar `i64` result (`src/simd/itx/mod.rs`
has the bounds). The lossless Walsh-Hadamard transform stays scalar on every tier (lossless 4x4
blocks only). The DC-only DCT_DCT fast path (`transform::dc_only`, used by `recon`) is scalar
and tier-independent. The 1D transforms of all tiers come from one macro (`itx_kernels!`) over
per-tier vector primitives, written from the spec code (not ported from dav1d).

- `threads` (speed round S): not a SIMD kernel family. It runs the frame wrapup (loop filter,
  CDEF, superres, loop restoration) in parallel bands (docs/threading.md); no tier is written
  (there is no arithmetic of its own: it calls the scalar filters, or whatever tier their own
  families dispatch to, on bands), and it has no `unsafe`. Its "cross-tier" checks compare
  thread counts instead: the unit tests in `decoder.rs` and the fuzz target `simd_threads`.

- `coefhot` (speed round S): not a SIMD kernel family. Scalar, safe-code changes to the
  coefficient path (`src/coef.rs`, `src/recon.rs`) with the same arithmetic and the same symbol
  reads: `Quant[ ]` and the padded levels stay all zero between transform blocks (the levels are
  cleared as the sign loop reads them, `Quant[ ]` over the coded region after the
  reconstruction) instead of being cleared whole per block; the coefficient reader tracks the
  coded region, so `recon::reconstruct_region` does not scan `Quant[ ]` again; the context
  offsets, CDF rows and scan bounds are set up once per transform block; the above / left
  context runs are slices; `BlockDecoded` and the loop filter transform sizes are filled by
  rows. No tier and no new kernel boundary, so no fuzz target: the unit tests compare the
  helpers with the spec loops (`coef::tests::context_runs_match_the_spec_loops`,
  `padded_contexts_match_the_oracle`, `recon::tests::tracked_region_matches_the_scan`) and the
  decode parity tests cover the rest.

`entropy` also holds the scalar symbol reader (`src/symbol.rs`, after dav1d's msac: a 64-bit
window, renormalisation by `leading_zeros`, a branchless symbol search) and the padded levels
of the coefficient contexts (`src/coef.rs`); neither has a SIMD tier (the reader is serial, one
symbol at a time). The spec's bit by bit reader stays as `symbol::oracle` (tests and the
`simd_entropy` fuzz target only).

## Kernels kept and dropped

A kernel stays only when the bench (docs/benchmarks.md) shows that it wins. Kernels and tiers
that were tried and dropped are listed here with their numbers.

| Kernel | Tier | Result |
|---|---|---|
| `grain` blend | AVX2, AVX-512 (runs AVX2) | Kept, not slower than scalar. Only the bench class `grain` has film grain (2 frames, about 17 ms, near the bench's 1 ms resolution). Tier table: scalar 18, avx2 17, avx512 17 ms (main before the change: 18, 17, 18). The gain is below what the bench can resolve. |
| `lr` wiener, self guided | AVX2, AVX-512 (runs AVX2) | Kept. Tier table (bench run 45324834732), scalar / avx2 / avx512 ms: avif8 144 / 115 / 115, avifhi 356 / 276 / 276, allintra 858 / 821 / 818, tiles8 565 / 535 / 535, tiles10 579 / 553 / 555 (main 68e575a before the change: avif8 138 / 131 / 131, avifhi 338 / 308 / 308, allintra 824 / 826 / 825, tiles8 537 / 511 / 552, tiles10 553 / 532 / 537). avx2 / scalar went from 94% to 79% on avif8 and from 91% to 77% on avifhi; the other classes are within the runner noise (dav1d itself ran about 3% slower on this run). NEON is not benched (the bench runs on x86_64 only). |
| `cdef` direction, filter | AVX2, AVX-512 (runs AVX2) | Kept. Tier table (bench run 45331809157), scalar / avx2 / avx512 ms: allintra 837 / 593 / 591, avif8 139 / 83 / 83, avifhi 343 / 231 / 235, tiles8 542 / 233 / 235, tiles10 558 / 235 / 236 (the `lr` bench before the change, scalar / avx2: allintra 858 / 821, avif8 144 / 115, avifhi 356 / 276, tiles8 565 / 535, tiles10 579 / 553). avx2 / scalar went from 96% to 71% on allintra, 80% to 60% on avif8, 78% to 67% on avifhi 95% to 43% on tiles8 and 96% to 42% on tiles10. AVX-512 runs the same kernel and is within the noise of AVX2. NEON is not benched (the bench runs on x86_64 only). |
| `ipred` Paeth, Smooth, zone 1, edge filter, palette, CFL; V/H fills | AVX2, AVX-512 (runs AVX2) | Kept, not slower than scalar; the gain is at the edge of what the bench resolves. Tier table (bench run 45345035593), scalar / avx2 / avx512 ms: allintra 824 / 583 / 584, avif8 138 / 82 / 82, avifhi 340 / 226 / 226, tiles8 538 / 230 / 229, tiles10 554 / 233 / 232 (the `cdef` bench before the change: allintra 837 / 593 / 591, avif8 139 / 83 / 83, avifhi 343 / 231 / 235, tiles8 542 / 233 / 235, tiles10 558 / 235 / 236). avx2 went down 1 to 2% on every class (allintra 593 to 583, avifhi 231 to 226), but the scalar tier, which runs the unchanged oracle, moved by the same amount, so avx2 / scalar stays within 1 point (allintra 71% / 71%, avifhi 67% / 66%). The kernels cover the cheap intra modes; DC and directional zones 2 and 3 still run the oracle. NEON is not benched (the bench runs on x86_64 only). |
| `itx` inverse transforms + add/clip; scalar coded-region recon and DC-only path | AVX2 | Kept. Tier table (bench run 45355824675), scalar / avx2 / avx512 ms: allintra 782 / 496 / 497, avif8 130 / 61 / 61, avifhi 317 / 149 / 148, tiles8 518 / 167 / 168, tiles10 530 / 169 / 168 (the `ipred` bench before the change: allintra 824 / 583 / 584, avif8 138 / 82 / 82, avifhi 340 / 226 / 226, tiles8 538 / 230 / 229, tiles10 554 / 233 / 232). The scalar tier (coded region only, no 32 KiB memsets, DC-only path) went down 3 to 6% (allintra 824 to 782, avifhi 340 to 317); avx2 went down 15 to 34% (allintra 583 to 496, avif8 82 to 61, avifhi 226 to 149, tiles8 230 to 167, tiles10 233 to 169). NEON is not benched (the bench runs on x86_64 only). |
| `itx` 16-lane kernel (blocks of at least 16x16) | AVX-512 | Dropped: no measurable win over AVX2. In bench run 45355824675 the AVX-512 tier (16-lane kernel for blocks of at least 16x16, AVX2 kernel for the rest) against AVX2 ms: allintra 497 / 496, avif8 61 / 61, avifhi 148 / 149, tiles8 168 / 167, tiles10 168 / 169. The Avx512 tier now runs the AVX2 kernel. |
| `entropy` CDF adaptation; scalar msac-style reader and padded coefficient levels | AVX2, AVX-512 (runs AVX2) | Kept. Tier table (bench run 45383238337, repeated within 1 ms of run 45379934968), scalar / avx2 / avx512 ms: allintra 719 / 414 / 414, avif8 125 / 55 / 55, avifhi 309 / 138 / 137, tiles8 513 / 153 / 154, tiles10 529 / 154 / 155 (the `threads` bench before the change, run 45371888091: allintra 790 / 499 / 499, avif8 130 / 61 / 61, avifhi 317 / 148 / 147, tiles8 519 / 168 / 167, tiles10 535 / 168 / 168). The scalar tier (new reader and levels, scalar adaptation) went down 1 to 9% (allintra 790 to 719); avx2 went down 7 to 17% (allintra 499 to 414, avif8 61 to 55, avifhi 148 to 138, tiles8 168 to 153, tiles10 168 to 154), avx2 / scalar from 63% to 57% on allintra. The bench does not separate the SIMD adaptation from the reader; AVX-512 runs the AVX2 kernel and is within the noise of it. NEON is not benched (the bench runs on x86_64 only). |
| `alloc` (buffer reuse, no kernel) | none | No SIMD tier: the family is allocation and buffer reuse (docs/limits.md, "Buffer reuse"), safe code only. Kept: needed for the zero-allocation guarantee (`tests/alloc.rs`) and neutral to slightly faster. 1 thread (bench run 45394312617 against the `entropy` bench 45383238337), av1-rs ms: allintra 410 / 414, avif8 55 / 56, avifhi 136 / 137, tiles8 152 / 153, tiles10 155 / 156; 16 threads: allintra 400 / 400, avif8 48 / 47, avifhi 111 / 113, tiles8 36 / 38, tiles10 38 / 38. Tiers scalar / avx2 / avx512 ms: allintra 720 / 409 / 410, avif8 126 / 55 / 55, avifhi 303 / 134 / 135, tiles8 505 / 152 / 152, tiles10 521 / 154 / 154. |

## Hostile input

Outside tests, `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::todo` and
`clippy::unimplemented` are denied (`src/lib.rs`). A bitstream error is an `Error`, never a panic.
Sizes read from a stream are checked against `Limits` before anything is allocated, and buffers
that scale with the input are charged to the ledger and allocated with `try_reserve_exact`
(docs/limits.md). The fuzz targets are described in docs/fuzzing.md.
