# av1-rs

A pure-Rust AV1 **intra-only** decoder for AVIF stills and AV1 key frames, bit-exact to the AV1
Bitstream & Decoding Process Specification. Inter frames are refused with `Error::InterFrame`
(the frames before them are kept). No C code, no unsafe code, every allocation bounded by
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

## Resource limits

| Field | Default | Error |
|---|---|---|
| `max_alloc` (bytes) | 1 GiB | `AllocLimit` |
| `max_threads` | `available_parallelism()` | - |
| `max_width`, `max_height` (pixels) | 65536 | `WidthLimit`, `HeightLimit` |
| `max_pixels` | 2^28 | `PixelLimit` |
| `max_metadata` (bytes) | 16 MiB | `MetadataLimit` |

`max_alloc` is the enforced peak working memory of one call (all threads; charged before each
allocation, released on drop). `max_threads = 1` spawns no thread. A frame size is checked
before anything is allocated. Setting the cancel flag returns `Error::Cancelled` at the next OBU
or tile. Details: [docs/limits.md](docs/limits.md).

## Docs

- [docs/design.md](docs/design.md): method (spec-literal), modules, pipeline
- [docs/parity.md](docs/parity.md): oracles (dav1d, aomdec), corpus, gates, results
- [docs/fuzzing.md](docs/fuzzing.md), [docs/dependencies.md](docs/dependencies.md)
