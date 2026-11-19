# Design

av1-rs is an AV1 intra-only decoder. The AV1 specification (AV1 Bitstream & Decoding Process
Specification, v1.0.0 with errata, `github.com/AOMediaCodec/av1-spec`) is the source of truth.
The decoder is bit-exact to it; dav1d and libaom are the CI oracles (docs/parity.md).

## Method: spec-literal first

Parity comes first, speed later. The code follows the spec's processes one to one, with the
spec's names, so a reviewer can put a function next to its spec section:

- A syntax element keeps its spec name (`frame_type`, `delta_q_y_dc`, `cdef_bits`).
- A spec CamelCase variable becomes snake_case (`MiCols` -> `mi_cols`, `CdefDamping` ->
  `cdef_damping`, `TxSz` -> `tx_sz`).
- A spec constant keeps its name and value in `src/consts.rs` (`BLOCK_64X64`, `TX_16X16`,
  `DCT_DCT`, `DC_PRED`, `NUM_REF_FRAMES`, ...). Enumerations are `u8`/`usize` constants with the
  spec's numbers, not Rust enums, so spec tables index them directly.
- A spec table keeps its name in upper snake case (`Default_Scan_4x4` -> `DEFAULT_SCAN_4X4`,
  `Dc_Qlookup` -> `DC_QLOOKUP`) and is transcribed from the spec, section 10 or 9.
- Each function's doc comment names its spec section (`/// Spec 7.11.2.4`).

Speed work (SIMD, threads, dav1d-style restructuring) comes after parity holds, with this scalar
code kept as the oracle.

## Modules

```
lib.rs        public API: decode(), Decoder, Frame, Plane, Options, Format; crate docs
error.rs      Error (one enum; limit errors are distinct variants)
limits.rs     Limits { max_alloc, max_threads, max_width, max_height, max_pixels, ... }
ledger.rs     max_alloc ledger (charge on allocate, release on drop), as png-rs
consts.rs     spec constants (section 3) and enum values
bits.rs       BitReader: f(n), uvlc, le(n), leb128, su(n), ns(n), byte alignment (spec 4.10, 8.1)
obu.rs        OBU framing: low-overhead (section 5) and Annex B; OBU header, extension,
              operating point drop rule, metadata OBUs (HDR CLL / MDCV kept raw), padding
seq.rs        SequenceHeader (5.5) incl. color_config, timing/decoder model, operating points
header.rs     FrameHeader (5.9): uncompressed_header for KEY_FRAME and INTRA_ONLY_FRAME,
              show_existing_frame; the ref state it reads (RefValid, RefFrameType, ...); an inter
              frame stops here with Error::InterFrame
symbol.rs     symbol decoder (8.2): init, read_symbol + CDF adaptation, read_bool, read_literal,
              exit process (padding / trailing check)
cdf.rs        CdfContext: every CDF array; default init (spec "Default CDF tables"), coefficient
              CDFs by qindex, load/save/averaging per spec 7.20 and 8.2
cdf_tables.rs default CDF tables transcribed from the spec
tile.rs       tile group decode (5.11): partition, intra frame mode info, palette, filter intra,
              tx size, residual, read_cdef, read_lr, delta q/lf, segmentation ids; calls predict
              and recon in spec order (decode_block -> compute_prediction / residual)
coef.rs       coefficients syntax (5.11.39), transform type (5.11.47), contexts (8.3.2)
scan.rs       scan tables
predict.rs    intra prediction (7.11.2) incl. directional + edge filter/upsample, smooth,
              paeth, DC, filter intra; palette (7.11.4); CFL (7.11.5)
recon.rs      dequant (7.12.2) and reconstruct (7.12.3)
transform.rs  1D and 2D inverse transforms (7.13): DCT 4..64, ADST 4/8/16, flipADST, identity,
              WHT, with the spec's intermediate rounding and clamping
quant.rs      Dc_Qlookup / Ac_Qlookup (8/10/12-bit), quantizer matrices (10, "Quantizer matrix")
lf.rs         deblocking loop filter (7.14)
cdef.rs       CDEF (7.15)
superres.rs   upscaling (7.16)
lr.rs         loop restoration (7.17): Wiener, self-guided
grain.rs      film grain synthesis (7.18.3), applied to a copy at output when enabled
frame.rs      FrameState: the spec's frame-level arrays; Plane buffers
decoder.rs    the frame decode process (7.x "decode frame wrapup"), reference slots (7.20/7.21),
              show_existing_frame, output (7.18), operating point / layer selection
```

## Shared state (src/frame.rs)

The spec keeps decode state in global arrays. Here they are fields of `FrameState`, named by
the rule above, each a `Vec` of `mi_rows * mi_cols` entries in row-major order
(index `row * mi_cols + col`), charged to the ledger before allocation. Examples:
`mi_sizes`, `skips`, `segment_ids`, `y_modes`, `uv_modes`, `ref_frames` (always INTRA_FRAME),
`is_inters`, `palette_sizes`, `palette_colors`, `tx_sizes`, `inter_tx_sizes`,
`loopfilter_tx_sizes[plane]`, `delta_lfs`, `cdef_idx` (per 64x64), `lr_type`/`lr_wiener`/
`lr_sgr_set`/`lr_sgr_xqd` (per restoration unit and plane).

Pixel planes (`CurrFrame`, `CdefFrame`, `UpscaledCdefFrame`, `LrFrame`) are
`Plane { width, height, stride, px: Vec<u16> }`: one sample type (`u16`) for every bit depth.
A decode plane is allocated with the luma width and height rounded up to a multiple of 128
(shifted by the chroma subsampling), so every block write of a superblock is in bounds; the
spec's own clamps (`maxX`, `maxY`) decide which samples prediction reads.

Function boundaries pass slices plus explicit parameters, not the whole state, where a process
is self-contained (transforms, intra prediction, filters, film grain). Each module documents its
entry point and the spec section it implements.

## Pipeline per frame

```
OBUs -> seq header -> frame header -> tile groups -> (per tile) blocks:
        mode info -> prediction -> residual (dequant + inverse transform) -> CurrFrame
     -> deblock (7.14) -> CDEF (7.15) -> superres (7.16) -> loop restoration (7.17)
     -> reference update (7.20) -> output (7.18) [-> film grain, optional]
```

Frames that are not KEY_FRAME or INTRA_ONLY_FRAME stop the decode with `Error::InterFrame`;
frames decoded before it are returned by the streaming `Decoder`.

## Untrusted input

Every size from the bitstream is checked against `Limits` before an allocation (ledger); every
read goes through the bit reader or the symbol decoder, which never read past the end (the spec
pads with zeros; a read past the end of a tile is an error after the tile, as the spec's
trailing check). Non-test code denies unwrap/expect/panic and indexing that can fail on input
values is guarded by earlier validation (sizes, mi bounds). Arithmetic on input values uses
checked or saturating forms where the spec does not bound it.
