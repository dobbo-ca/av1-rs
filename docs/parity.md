# Parity

The AV1 spec requires a bit-exact decoder. The C reference decoders run in CI only, as oracle
tools, never as crate dependencies (.github/ci/Dockerfile, stages `oracles`, `parity`):

- dav1d 1.0.0 (Debian bookworm `dav1d`) on every stream: `dav1d --threads 1 --filmgrain 1
  --limit N -o <s>.yuv`.
- aomdec 3.6.0 (bookworm `aom-tools`) on the libaom test vectors: `aomdec --rawvideo --limit=N`.
- The per-frame md5 that libaom publishes with each test vector (`<vector>.md5`, one line per
  frame; 4:2:0 only, as the md5 always covers three planes).

`N` is the number of frames av1-rs outputs: the leading intra frames of a stream (a stream with
inter frames stops at the first one with `Error::InterFrame`, class `inter`). Any other error on
a corpus stream fails the gate. The gate also compares `--threads 1` with the default threads.

`examples/oracle.rs` writes `<stem>.yuv` (frames concatenated, planes Y U V, visible size,
8-bit as bytes and higher depths as little-endian u16; the dav1d `.yuv` layout), `<stem>.frames`
(one line per frame: `W H bitdepth layout`) and `<stem>.err` (the error class).

`examples/oracle.rs` also takes `.avif` files (`av1::decode_avif`; the alpha image goes to `<stem>.alpha.yuv`).
`examples/avifx.rs` is a CI-only extraction tool, not part of the library: `avifx <out-dir> <file.avif>...` writes every AV1 coded item reachable from the primary image (itself, its grid tiles, its alpha item and the alpha's tiles) as `<out-dir>/<stem>.item<id>.obu`, the `av1C` configOBUs followed by the item data, a Section 5 stream that dav1d and aomdec read.

## Modes

`tests/parity/mode` holds the gate mode. `info` (phase 1, headers only): the frame size and
count from `probe()` against dav1d's y4m output. `pixels` (from phase 3): every plane byte-exact.

## Corpus

`tests/parity/corpus.txt`, fetched once into the BuildKit cache mount `id=av1-rs-data` and
checked against pinned upstream hashes (no hash is computed by us):

| Set | Files | Pin | Licence |
|---|---|---|---|
| libaom test vectors (`storage.googleapis.com/aom-test-data`) | 240 streams + their `.md5` | sha1 from libaom `test/test-data.sha1` at aom `04fc2a69e977` | BSD-2-Clause + AOM Patent License 1.0 (libaom) |
| dav1d-test-data (features, film grain, intra, issues, resize, svc, 10/12-bit features) | 49 streams | git blob id at `61afa1ce` | dav1d project, BSD-2-Clause |

Not used: the AV1 Argon streams (no licence is stated: the generator repository says "All
rights reserved" and AOM's announcement gives no terms), the two libaom `.mkv` vectors (dav1d has
no Matroska input; dav1d-test-data has the same streams as IVF), and AVIF samples (phase 6).

## Results

Filled per phase from the `parity` job on `main`.

## Known oracle differences

- Layered streams (`svc-L2T1-2`, `svc-L2T2-2`, and the dav1d-test-data `svc-L2*` copies): the upper
  spatial layer predicts from the base layer, so it is an inter frame and av1-rs stops there;
  dav1d (all layers) and av1-rs output the base layer, libaom outputs the upper layer. These
  streams are compared with dav1d only.
- Superres floor: like libaom and dav1d, the coded width is never below Min(16, UpscaledWidth),
  and superres applies only when the coded and upscaled widths differ (dav1d issue 322; the spec
  text has no floor).
- AVIF (`avifdec` is libavif 0.11.1 from bookworm): `sofa_grid1x5_420_reversed_dimg_order`
  is not compared with avifdec, because libavif before 1.1.1 ignores the `dimg` order (HEIF says
  the `dimg` order is the tile order; av1-rs follows it). `color_grid_alpha_grid_tile_shared_in_dimg`
  uses one tile item twice in `dimg`; av1-rs refuses it, as libavif 1.1.1 and later do. The
  libavif `poc_*` files are security reproducers: fuzz seeds, not compared. When an image has
  alpha, avifdec's y4m holds the color planes then the alpha plane, so the gate compares it with
  `.yuv` followed by `.alpha.yuv`.
