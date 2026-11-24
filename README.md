# av1-rs

A pure-Rust AV1 **intra-only** decoder for AVIF stills and AV1 key frames, bit-exact to the AV1
Bitstream & Decoding Process Specification. Inter frames are refused with `Error::InterFrame`
(the frames before them are kept). No C code, no `unsafe` outside the SIMD load/store helpers and
dispatch fns (docs/safety.md), every allocation bounded by
`Limits`. AV1 is royalty-free (AOM patent licence); the owner reviews licensing before Dobbo
Drive goes public.

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

AVIF stills are decoded with `av1::decode_avif` (the primary image and its alpha image, grids assembled).
`av1::decode_avif_item` decodes any image item of an already parsed file (coded, grid or identity; no alpha, transforms not applied).

## Resource limits

| Field | Default | Error |
|---|---|---|
| `max_alloc` (bytes) | 1 GiB | `AllocLimit` |
| `max_threads` | `available_parallelism()` | - |
| `max_width`, `max_height` (pixels) | 65536 | `WidthLimit`, `HeightLimit` |
| `max_pixels` | 2^28 | `PixelLimit` |
| `max_metadata` (bytes) | 16 MiB | `MetadataLimit` |

`max_alloc` is the enforced peak working memory of one decoder or one `decode_avif` call, all
threads included: frame buffers, film grain buffers and the output frames of a temporal unit
(and every frame `decode` collects) are charged before allocation and released on drop. Not
counted: frames already returned by `Decoder`, the item bytes inside `isobmff`; `max_metadata`
is on top. `max_threads = 1` spawns no thread; otherwise the tiles and in-loop filters of a frame
(one scope of threads per frame, no pool kept between calls) and the tiles of
an AVIF grid decode on up to `max_threads` threads, and film grain runs in bands of 32 rows on
up to `max_threads` threads, with the same output. A frame size is
checked before anything is allocated. The cancel flag returns `Error::Cancelled` at the next
OBU, superblock row or filter row. Details: [docs/limits.md](docs/limits.md).

Metadata OBUs (HDR CLL / MDCV, ITU-T T.35, any type) and the colour configuration are kept raw
by default (`Frame::metadata`, `Frame::color`); `Options::keep_metadata = false` drops them.

## Docs

- [docs/design.md](docs/design.md): method (spec-literal), modules, pipeline
- [docs/parity.md](docs/parity.md): oracles (dav1d, aomdec), corpus, gates, results
- [docs/fuzzing.md](docs/fuzzing.md), [docs/dependencies.md](docs/dependencies.md)
- [docs/threading.md](docs/threading.md), [docs/limits.md](docs/limits.md)
- [docs/safety.md](docs/safety.md), [docs/benchmarks.md](docs/benchmarks.md)
