# av1-rs

A pure-Rust AV1 **intra-only** decoder for AVIF stills and AV1 key frames, bit-exact to the AV1
Bitstream & Decoding Process Specification. Inter frames are refused with `Error::InterFrame`
(a `Decoder` keeps the frames before them; `av1::decode` returns only the error). No C code, no `unsafe` outside the SIMD load/store helpers and
dispatch fns (docs/safety.md), every allocation bounded by `Limits`.

## Quick start

```toml
av1 = { git = "https://github.com/dobbo-ca/av1-rs", rev = "<sha>" }
```

```rust,no_run
use av1::{Format, Limits, Options};
use std::sync::atomic::AtomicBool;

let data = std::fs::read("still.obu")?;
let limits = Limits { max_threads: 2, max_alloc: 256 << 20, ..Limits::default() };
let frames = av1::decode(&data, Format::Section5, &Options::default(), &limits, &AtomicBool::new(false))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Resource limits

| Field | Default | Error |
|---|---|---|
| `max_alloc` (bytes) | 1 GiB | `AllocLimit` |
| `max_threads` | `available_parallelism()` | - |
| `max_width`, `max_height` (pixels) | 65536 | `WidthLimit`, `HeightLimit` |
| `max_pixels` | 2^28 | `PixelLimit` |
| `max_metadata` (bytes) | 16 MiB | `MetadataLimit` |

`max_alloc` is the enforced peak working memory of one decoder or one `decode_avif` call, all
threads included, charged before each allocation. A frame size is checked before anything is
allocated; `max_threads = 1` spawns no thread, and the output is the same for every thread
count. The cancel flag returns `Error::Cancelled` at the next OBU, superblock row or filter row.
What is and is not counted: [docs/limits.md](docs/limits.md).

## AVIF and metadata

`av1::decode_avif` decodes the primary image and its alpha image (grids assembled);
`av1::decode_avif_item` decodes any image item of a parsed file. `clap`, `irot` and `imir` come
back raw in `AvifImage::transforms`; `Options { apply_transforms: true, .. }` applies them
(docs/design.md). Metadata OBUs are kept raw in `Frame::metadata` unless
`Options::keep_metadata = false`; the parsed colour configuration is always in `Frame::color`.

## Docs

- [docs/design.md](docs/design.md): method (spec-literal), modules, pipeline
- [docs/parity.md](docs/parity.md): oracles (dav1d, aomdec), corpus, gates, results
- [docs/fuzzing.md](docs/fuzzing.md), [docs/dependencies.md](docs/dependencies.md)
- [docs/threading.md](docs/threading.md), [docs/limits.md](docs/limits.md)
- [docs/safety.md](docs/safety.md), [docs/benchmarks.md](docs/benchmarks.md)
