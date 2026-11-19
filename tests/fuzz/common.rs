//! Shared by the fuzz targets: the last input byte holds the options, and an input that starts
//! with an IVF header ("DKIF") is unwrapped into its temporal units (the corpus seeds are IVF).

use av1::Format;

/// Splits `data` into (stream, options byte).
pub fn split(data: &[u8]) -> Option<(&[u8], u8)> {
    let (&opt, stream) = data.split_last()?;
    Some((stream, opt))
}

pub fn format(opt: u8) -> Format {
    if opt & 1 == 1 {
        Format::AnnexB
    } else {
        Format::Section5
    }
}

/// The temporal units of an IVF file, or None if `data` is not IVF.
pub fn ivf_units(data: &[u8]) -> Option<Vec<&[u8]>> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return None;
    }
    let hdr = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut rest = data.get(hdr.max(32)..)?;
    let mut units = Vec::new();
    while rest.len() >= 12 {
        let n = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        let body = &rest[12..];
        let take = n.min(body.len());
        units.push(&body[..take]);
        rest = &body[take..];
    }
    Some(units)
}
