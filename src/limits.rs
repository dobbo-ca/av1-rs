//! Caps on one decode.

/// Caps on one decode. The defaults are finite. Every entry point takes `&Limits` and, for
/// decoding, a cancel flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Peak working memory of one decoder, in bytes, all threads together (default 1 GiB). Every
    /// buffer that scales with the frame size is charged before it is allocated; the frames a
    /// decoder returns are not counted once returned (`decode` counts the frames it collects). More is
    /// [`Error::AllocLimit`](crate::Error::AllocLimit).
    pub max_alloc: usize,
    /// Most OS threads of one call, the calling thread included (default
    /// `std::thread::available_parallelism`). 1 spawns no thread; 0 is treated as 1.
    pub max_threads: usize,
    /// Most luma samples per row of a frame, upscaled (default 65536). More is
    /// [`Error::WidthLimit`](crate::Error::WidthLimit).
    pub max_width: u32,
    /// Most luma rows of a frame (default 65536). More is
    /// [`Error::HeightLimit`](crate::Error::HeightLimit).
    pub max_height: u32,
    /// Most `width * height` luma samples of a frame (default `1 << 28`). More is
    /// [`Error::PixelLimit`](crate::Error::PixelLimit).
    pub max_pixels: u64,
    /// Sum of the metadata OBU payloads kept for one frame, in bytes (default 16 MiB), counted in
    /// addition to `max_alloc`. More is
    /// [`Error::MetadataLimit`](crate::Error::MetadataLimit).
    pub max_metadata: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_alloc: 1 << 30,
            max_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_width: 65536,
            max_height: 65536,
            max_pixels: 1 << 28,
            max_metadata: 16 << 20,
        }
    }
}
