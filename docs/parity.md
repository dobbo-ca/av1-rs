# Parity

The AV1 spec requires a bit-exact decoder. The C reference decoders run in CI only, as oracle
tools, never as crate dependencies (.github/ci/Dockerfile, stages `oracles`, `parity`):

- dav1d 1.0.0 (Debian bookworm `dav1d`) on every stream: `dav1d --threads 1 --filmgrain 1
  --limit N -o <s>.yuv`.
- aomdec 3.6.0 (bookworm `aom-tools`) on the libaom test vectors: `aomdec --rawvideo --limit=N`.
- The per-frame md5 that libaom publishes with each test vector (`<vector>.md5`, one line per
  frame; 4:2:0 only, as the md5 always covers three planes).
- avifdec 0.11.1 (bookworm `libavif-bin`, libavif with dav1d) on the AVIF files:
  `avifdec --jobs 1 <file> <s>.y4m`.
- avifdec-new: avifdec of libavif v1.4.2 (commit `c5240fc79fe5c2407e10afd35f5505ef6333ea49`),
  built from github.com/AOMediaCodec/libavif in the Dockerfile stage `avifdec-new` with bookworm's
  `libdav1d-dev` (dav1d 1.0.0) as its only codec, on the same AVIF files with the same command.
  libavif 1.1.1 and later follow the `dimg` order and refuse an item used in two `dimg` boxes.

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
| AVIF samples: av1-avif (Link-U, Xiph) and libavif test files (`tests/parity/corpus.txt`) | 151 files | pins in `corpus.txt` | av1-avif: Link-U CC-BY-SA-4.0, Xiph CC-BY-SA-3.0; libavif `tests/data`: BSD-2-Clause, Kodak unrestricted, CC-BY-3.0 (per file, see the pinned repositories) |

Not used: the AV1 Argon streams (no licence is stated: the generator repository says "All
rights reserved" and AOM's announcement gives no terms), the two libaom `.mkv` vectors (dav1d has
no Matroska input; dav1d-test-data has the same streams as IVF).

## Results

Last green run on `main` (a5cb385, 2026-11-23): 679 inputs, 604 frames; 289 streams and 238 AVIF items bit-exact against dav1d, 518 inputs against aomdec, 236 libaom vectors against the published md5, 133 AVIF files against avifdec; every gate green, `--threads 1` equal to the default threads, aarch64 equal to x86_64 on the subset.

The `parity` job on `main` gates: 289 streams (240 libaom + 49 dav1d-test-data) against dav1d, and against aomdec and the published md5 where those apply, and 151 AVIF files against avifdec and avifdec-new (a file an avifdec cannot decode must be listed in `tests/parity/avifdec-known-failures.txt` or `tests/parity/avifdec-new-known-failures.txt`, and a listed file that decodes fails the gate as a stale entry) plus every coded item against dav1d and aomdec. `run.sh` fails the job if fewer than 289 streams or 151 AVIF files are present, or if fewer than 289 streams, 238 extracted AVIF items, 604 frames, 133 AVIF files against avifdec or 133 against avifdec-new were actually compared (items are counted apart from streams). The avifdec-new floor of 133 is provisional (the floor of avifdec 0.11): raise it to the count of the first green run. The exact counts of compared frames, streams and files are the first lines of the job `summary.md`; the known differences are listed below. Pin the floors in `run.sh` from the latest green run.

## Known oracle differences

- Layered streams (`svc-L2T1-2`, `svc-L2T2-2`, and the dav1d-test-data `svc-L2*` copies): the upper
  spatial layer predicts from the base layer, so it is an inter frame and av1-rs stops there;
  dav1d (all layers) and av1-rs output the base layer, libaom outputs the upper layer. These
  streams are compared with dav1d only.
- Superres floor: like libaom and dav1d, the coded width is never below Min(16, UpscaledWidth),
  and superres applies only when the coded and upscaled widths differ (dav1d issue 322; the spec
  text has no floor).
- AVIF (`avifdec` is libavif 0.11.1 from bookworm, `avifdec-new` libavif 1.4.2):
  `sofa_grid1x5_420_reversed_dimg_order` is compared with avifdec-new only, because libavif
  before 1.1.1 ignores the `dimg` order (HEIF says the `dimg` order is the tile order; av1-rs and
  libavif 1.1.1 follow it). `color_grid_alpha_grid_tile_shared_in_dimg` uses one tile item twice
  in `dimg`: av1-rs and avifdec-new must both refuse it (a decode by either fails the gate).
  The libavif `poc_*` files are security reproducers: fuzz seeds, not compared. When an image has
  alpha, avifdec's y4m holds the color planes then the alpha plane, so the gate compares it with
  `.yuv` followed by `.alpha.yuv`. Files an avifdec refuses and av1-rs decodes are listed in
  `tests/parity/avifdec-known-failures.txt` (0.11.1) and `tests/parity/avifdec-new-known-failures.txt`
  (1.4.2: `alpha_noispe`, refused by libavif's strict mode; `color_grid_alpha_grid_shared_tiles`,
  a tile shared by two grids; `sofa_grid1x5_420_dimg_repeat`, a tile repeated in one grid).
- The Xiph AVIF files are layered (`a1lx`, 2 or 3 spatial layers in one item): as with the
  `svc-L2*` streams, av1-rs and dav1d output the base layer and aomdec the top layer, so these
  items are compared with dav1d only. `animated_alpha_sample_without_shown_frame`: the second
  sample of its alpha track has no shown frame; av1-rs returns no alpha plane, while both
  avifdecs decode the first frame of the track with its alpha plane, so the file is compared
  through its items only.
