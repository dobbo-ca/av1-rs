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

Tiers not written:

- `lf` AVX-512: a run is one transform edge (4 to 64 samples, mostly 4 to 16), so 32 lanes
  would be mostly padding, and the gather and scatter around the kernel cost the same per sample
  at either width.
- `grain` AVX-512: the 8-lane gather of the scale bounds the loop, and a 16-lane gather is not
  faster per sample.
- `lr` AVX-512: a block row is at most 64 samples, so 16-lane vectors would save at most half of
  the vector ops while the fixed per-block work (window gather, the stack buffers) stays; it is
  worth writing only if the bench shows loop restoration still high in the profile after AVX2.

## Kernels kept and dropped

A kernel stays only when the bench (docs/benchmarks.md) shows that it wins. Kernels and tiers
that were tried and dropped are listed here with their numbers.

| Kernel | Tier | Result |
|---|---|---|
| `grain` blend | AVX2, AVX-512 (runs AVX2) | Kept, not slower than scalar. Only the bench class `grain` has film grain (2 frames, about 17 ms, near the bench's 1 ms resolution). Tier table: scalar 18, avx2 17, avx512 17 ms (main before the change: 18, 17, 18). The gain is below what the bench can resolve. |

## Hostile input

Outside tests, `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::todo` and
`clippy::unimplemented` are denied (`src/lib.rs`). A bitstream error is an `Error`, never a panic.
Sizes read from a stream are checked against `Limits` before anything is allocated, and buffers
that scale with the input are charged to the ledger and allocated with `try_reserve_exact`
(docs/limits.md). The fuzz targets are described in docs/fuzzing.md.
