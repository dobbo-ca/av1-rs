# Safety

av1-rs contains no `unsafe` code: `src/lib.rs` has `#![forbid(unsafe_code)]`, so the compiler
rejects it. The unsafe inventory is empty. There is no SIMD and no inline assembly; every kernel
is scalar Rust that follows the spec text.

Outside tests, `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::todo` and
`clippy::unimplemented` are denied (`src/lib.rs`). A bitstream error is an `Error`, never a panic.
Sizes read from a stream are checked against `Limits` before anything is allocated, and buffers
that scale with the input are charged to the ledger and allocated with `try_reserve_exact`
(docs/limits.md). The fuzz targets are described in docs/fuzzing.md.
