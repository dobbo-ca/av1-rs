//! The one error type of the crate.

use std::fmt;

/// Why a decode, probe or parse stopped. Limit errors are distinct variants so a caller can tell
/// "the stream is bad" from "the stream is bigger than the caller allows".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The data ended in the middle of a syntax element, OBU or tile.
    Truncated,
    /// The data breaks a requirement of the specification.
    Invalid(&'static str),
    /// Valid data that this decoder does not handle (for example large scale tile decoding).
    Unsupported(&'static str),
    /// The stream contains an inter frame (`INTER_FRAME` or `SWITCH_FRAME`). This decoder only
    /// handles intra frames. [`Decoder`](crate::Decoder) keeps the frames before it.
    InterFrame,
    /// The working memory would pass [`Limits::max_alloc`](crate::Limits::max_alloc).
    AllocLimit,
    /// The system refused an allocation.
    OutOfMemory,
    /// The frame is wider than [`Limits::max_width`](crate::Limits::max_width).
    WidthLimit,
    /// The frame is taller than [`Limits::max_height`](crate::Limits::max_height).
    HeightLimit,
    /// The frame has more pixels than [`Limits::max_pixels`](crate::Limits::max_pixels).
    PixelLimit,
    /// The metadata would pass [`Limits::max_metadata`](crate::Limits::max_metadata).
    MetadataLimit,
    /// The cancel flag was set.
    Cancelled,
}

impl Error {
    /// A short stable token for the kind of error, for logs and test oracles: `"truncated"`,
    /// `"invalid"`, `"unsupported"`, `"inter"`, `"alloc"`, `"oom"`, `"width"`, `"height"`,
    /// `"pixels"`, `"metadata"` or `"cancelled"`.
    pub fn class(&self) -> &'static str {
        match self {
            Error::Truncated => "truncated",
            Error::Invalid(_) => "invalid",
            Error::Unsupported(_) => "unsupported",
            Error::InterFrame => "inter",
            Error::AllocLimit => "alloc",
            Error::OutOfMemory => "oom",
            Error::WidthLimit => "width",
            Error::HeightLimit => "height",
            Error::PixelLimit => "pixels",
            Error::MetadataLimit => "metadata",
            Error::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => f.write_str("truncated data"),
            Error::Invalid(why) => write!(f, "invalid stream: {why}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::InterFrame => f.write_str("inter frames are not supported"),
            Error::AllocLimit => f.write_str("memory limit exceeded"),
            Error::OutOfMemory => f.write_str("out of memory"),
            Error::WidthLimit => f.write_str("frame width limit exceeded"),
            Error::HeightLimit => f.write_str("frame height limit exceeded"),
            Error::PixelLimit => f.write_str("frame pixel limit exceeded"),
            Error::MetadataLimit => f.write_str("metadata limit exceeded"),
            Error::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_and_display() {
        assert_eq!(Error::InterFrame.class(), "inter");
        assert_eq!(Error::Invalid("x").class(), "invalid");
        assert_eq!(Error::Invalid("bad").to_string(), "invalid stream: bad");
    }
}
