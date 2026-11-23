//! OBU framing: the low-overhead format of spec 5.2 and the length-delimited format of Annex B,
//! the OBU header (5.3.2, 5.3.3), the operating point drop rule (5.3.1) and metadata OBUs (5.8).
#![allow(dead_code)]

use crate::Format;
use crate::Metadata;
use crate::bits::BitReader;
use crate::consts::*;
use crate::error::Error;
use crate::limits::Limits;

/// `obu_header()` and `obu_extension_header()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObuHeader {
    pub obu_type: u8,
    pub obu_extension_flag: bool,
    pub obu_has_size_field: bool,
    /// 0 when there is no extension header.
    pub temporal_id: u8,
    /// 0 when there is no extension header.
    pub spatial_id: u8,
}

impl ObuHeader {
    /// True when the OBU is not part of the operating point whose `OperatingPointIdc` is `idc`
    /// and must be dropped (the check at the top of `open_bitstream_unit()`).
    pub(crate) fn dropped_by(&self, idc: u16) -> bool {
        if self.obu_type == OBU_SEQUENCE_HEADER
            || self.obu_type == OBU_TEMPORAL_DELIMITER
            || idc == 0
            || !self.obu_extension_flag
        {
            return false;
        }
        let in_temporal_layer = (idc >> self.temporal_id) & 1 != 0;
        let in_spatial_layer = (idc >> (self.spatial_id + 8)) & 1 != 0;
        !in_temporal_layer || !in_spatial_layer
    }
}

/// One OBU cut out of a temporal unit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Obu<'a> {
    pub header: ObuHeader,
    /// The `obu_size` bytes after the header and the size field.
    pub payload: &'a [u8],
    /// Offset of `payload` from the start of the temporal unit data.
    pub payload_start: usize,
}

/// Reads one OBU from the start of `data` (`open_bitstream_unit(sz)` of spec 5.3.1). `base` is
/// the offset of `data` in the temporal unit. `sz` is `Some(obu_length)` in the Annex B format
/// (the OBU is then exactly the first `sz` bytes) and `None` in the low-overhead format.
///
/// Without `obu_has_size_field` the OBU runs to the end of the available bytes (the spec derives
/// `obu_size` from `sz`; the low-overhead format requires the field, this is the lenient reading
/// for a last OBU). Returns the OBU and the bytes it took, header included.
fn read_obu(data: &[u8], base: usize, sz: Option<usize>) -> Result<(Obu<'_>, usize), Error> {
    let data = match sz {
        Some(n) => data.get(..n).ok_or(Error::Truncated)?,
        None => data,
    };
    let mut r = BitReader::new(data);
    if r.f(1)? != 0 {
        return Err(Error::Invalid("obu_forbidden_bit is set"));
    }
    let obu_type = r.f(4)? as u8;
    let obu_extension_flag = r.flag()?;
    let obu_has_size_field = r.flag()?;
    r.f(1)?; // obu_reserved_1bit
    let (mut temporal_id, mut spatial_id) = (0, 0);
    if obu_extension_flag {
        temporal_id = r.f(3)? as u8;
        spatial_id = r.f(2)? as u8;
        r.f(3)?; // extension_header_reserved_3bits
    }
    let obu_size = if obu_has_size_field {
        r.leb128()?.0 as usize
    } else {
        r.bits_left() / 8
    };
    let start = r.get_position() / 8;
    let end = start.checked_add(obu_size).ok_or(Error::Truncated)?;
    let payload = data.get(start..end).ok_or(Error::Truncated)?;
    let header = ObuHeader {
        obu_type,
        obu_extension_flag,
        obu_has_size_field,
        temporal_id,
        spatial_id,
    };
    let obu = Obu {
        header,
        payload,
        payload_start: base + start,
    };
    Ok((obu, end))
}

/// The OBUs of one temporal unit, in order. In the low-overhead format `data` is a run of OBUs;
/// in Annex B it is the payload of one `temporal_unit( sz )`, without its leading
/// `temporal_unit_size`. After an error the iterator ends.
pub(crate) struct ObuIter<'a> {
    data: &'a [u8],
    pos: usize,
    format: Format,
    /// Annex B: where the current frame unit ends.
    frame_unit_end: usize,
    failed: bool,
}

impl<'a> ObuIter<'a> {
    pub(crate) fn new(data: &'a [u8], format: Format) -> Self {
        Self {
            data,
            pos: 0,
            format,
            frame_unit_end: 0,
            failed: false,
        }
    }

    fn read_next(&mut self) -> Result<Option<Obu<'a>>, Error> {
        match self.format {
            Format::Section5 => {
                if self.pos >= self.data.len() {
                    return Ok(None);
                }
                let (obu, used) = read_obu(&self.data[self.pos..], self.pos, None)?;
                self.pos += used;
                Ok(Some(obu))
            }
            Format::AnnexB => {
                // Skip to the next frame unit with bytes in it.
                while self.pos >= self.frame_unit_end {
                    if self.pos >= self.data.len() {
                        return Ok(None);
                    }
                    let mut r = BitReader::new(&self.data[self.pos..]);
                    let (frame_unit_size, n) = r.leb128()?;
                    self.pos += n;
                    self.frame_unit_end = self
                        .pos
                        .checked_add(frame_unit_size as usize)
                        .filter(|&e| e <= self.data.len())
                        .ok_or(Error::Truncated)?;
                }
                let mut r = BitReader::new(&self.data[self.pos..self.frame_unit_end]);
                let (obu_length, n) = r.leb128()?;
                let start = self.pos + n;
                let end = start
                    .checked_add(obu_length as usize)
                    .filter(|&e| e <= self.frame_unit_end)
                    .ok_or(Error::Invalid("obu_length passes the frame unit"))?;
                let (obu, _) = read_obu(
                    &self.data[start..self.frame_unit_end],
                    start,
                    Some(obu_length as usize),
                )?;
                self.pos = end;
                Ok(Some(obu))
            }
        }
    }
}

impl<'a> Iterator for ObuIter<'a> {
    type Item = Result<Obu<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.read_next() {
            Ok(Some(obu)) => Some(Ok(obu)),
            Ok(None) => None,
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}
/// The temporal units of a stream, one at a time and without allocating (see
/// [`split_temporal_units`]). After a framing error the iterator is finished.
pub(crate) struct TemporalUnits<'a> {
    data: &'a [u8],
    format: Format,
    start: usize,
    pos: usize,
    failed: bool,
}

impl<'a> TemporalUnits<'a> {
    pub(crate) fn new(data: &'a [u8], format: Format) -> Self {
        TemporalUnits {
            data,
            format,
            start: 0,
            pos: 0,
            failed: false,
        }
    }

    fn cut(&mut self) -> Result<Option<&'a [u8]>, Error> {
        let data = self.data;
        match self.format {
            Format::Section5 => {
                while self.pos < data.len() {
                    let pos = self.pos;
                    let (obu, used) = read_obu(&data[pos..], pos, None)?;
                    self.pos += used;
                    if obu.header.obu_type == OBU_TEMPORAL_DELIMITER && pos > self.start {
                        let unit = &data[self.start..pos];
                        self.start = pos;
                        return Ok(Some(unit));
                    }
                }
                if self.pos > self.start {
                    let unit = &data[self.start..self.pos];
                    self.start = self.pos;
                    return Ok(Some(unit));
                }
                Ok(None)
            }
            Format::AnnexB => {
                if self.pos >= data.len() {
                    return Ok(None);
                }
                let mut r = BitReader::new(&data[self.pos..]);
                let (size, n) = r.leb128()?;
                let start = self.pos + n;
                let end = start
                    .checked_add(size as usize)
                    .filter(|&e| e <= data.len())
                    .ok_or(Error::Truncated)?;
                self.pos = end;
                Ok(Some(&data[start..end]))
            }
        }
    }
}

impl<'a> Iterator for TemporalUnits<'a> {
    type Item = Result<&'a [u8], Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let unit = self.cut().transpose();
        self.failed = matches!(unit, Some(Err(_)));
        unit
    }
}

/// Cuts a whole stream into temporal units. In the low-overhead format a unit starts at each
/// temporal delimiter OBU (bytes before the first one form a unit of their own); in Annex B a
/// unit is one `temporal_unit( sz )` and the slice is its payload without `temporal_unit_size`.
/// [`Decoder::decode_temporal_unit`](crate::Decoder::decode_temporal_unit) takes one slice.
///
/// # Errors
/// `Truncated` or `Invalid` when the framing (OBU sizes, Annex B lengths) is broken. The slices
/// before the broken part are not returned; the streaming `Decoder` users can split by hand.
/// `AllocLimit` when the slice list alone would pass `limits.max_alloc`, `OutOfMemory` when it
/// cannot be allocated.
pub fn split_temporal_units<'a>(
    data: &'a [u8],
    format: Format,
    limits: &Limits,
) -> Result<Vec<&'a [u8]>, Error> {
    let mut units = Vec::new();
    for unit in TemporalUnits::new(data, format) {
        let unit = unit?;
        if (units.len() + 1).saturating_mul(size_of::<&[u8]>()) > limits.max_alloc {
            return Err(Error::AllocLimit);
        }
        units.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        units.push(unit);
    }
    Ok(units)
}

/// `metadata_obu()` (spec 5.8.1): the `metadata_type` and the raw payload after it. The payload
/// is the bytes up to the trailing bits: trailing zero bytes are cut, then a last byte equal to
/// `0x80` (the trailing one bit of a byte aligned payload) is cut too. Every `metadata_type` is
/// kept raw, including types this decoder does not interpret (owner rule: keep metadata by
/// default; `Options::keep_metadata` drops it).
///
/// # Errors
/// `Truncated` if `metadata_type` is cut off, `Invalid` if it is not a valid leb128.
pub(crate) fn parse_metadata(payload: &[u8]) -> Result<Metadata, Error> {
    let mut r = BitReader::new(payload);
    let (metadata_type, _) = r.leb128()?;
    let metadata_type = u64::from(metadata_type);
    let mut rest = r.rest();
    while let [head @ .., 0] = rest {
        rest = head;
    }
    if let [head @ .., 0x80] = rest {
        rest = head;
    }
    let mut kept = Vec::new();
    kept.try_reserve_exact(rest.len())
        .map_err(|_| Error::OutOfMemory)?;
    kept.extend_from_slice(rest);
    Ok(Metadata {
        metadata_type,
        payload: kept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::tests::still_sequence_header;

    /// A temporal delimiter and a sequence header OBU, low-overhead format.
    fn td_and_seq() -> Vec<u8> {
        let seq = still_sequence_header();
        let mut v = vec![0x12, 0x00, 0x0a, seq.len() as u8];
        v.extend_from_slice(&seq);
        v
    }

    #[test]
    fn section5_obus() {
        let data = td_and_seq();
        let obus: Vec<_> = ObuIter::new(&data, Format::Section5)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].header.obu_type, OBU_TEMPORAL_DELIMITER);
        assert!(obus[0].payload.is_empty());
        assert_eq!(obus[1].header.obu_type, OBU_SEQUENCE_HEADER);
        assert_eq!(obus[1].payload, &data[4..]);
        assert_eq!(obus[1].payload_start, 4);
        assert!(!obus[1].header.obu_extension_flag);
    }

    #[test]
    fn truncated_and_forbidden() {
        let mut data = td_and_seq();
        data.pop();
        let r: Result<Vec<_>, _> = ObuIter::new(&data, Format::Section5).collect();
        assert_eq!(r.unwrap_err(), Error::Truncated);
        let r: Result<Vec<_>, _> = ObuIter::new(&[0x82, 0x00], Format::Section5).collect();
        assert!(matches!(r, Err(Error::Invalid(_))));
    }

    #[test]
    fn extension_header_and_drop_rule() {
        // OBU_FRAME_HEADER with extension: temporal_id 1, spatial_id 2, size 0.
        let data = [0x1e, (1 << 5) | (2 << 3), 0x00];
        let obu = ObuIter::new(&data, Format::Section5)
            .next()
            .unwrap()
            .unwrap();
        assert_eq!((obu.header.temporal_id, obu.header.spatial_id), (1, 2));
        // idc 0 keeps everything; temporal layers 0..1 + spatial layers 0..2 keeps it.
        assert!(!obu.header.dropped_by(0));
        assert!(!obu.header.dropped_by(0x0703));
        // No spatial layer 2, or no temporal layer 1: dropped.
        assert!(obu.header.dropped_by(0x0303));
        assert!(obu.header.dropped_by(0x0701));
        // Sequence headers and temporal delimiters are never dropped.
        let mut h = obu.header;
        h.obu_type = OBU_SEQUENCE_HEADER;
        assert!(!h.dropped_by(0x0101));
    }

    #[test]
    fn size_field_absent_takes_the_rest() {
        let data = [0x08, 0xaa, 0xbb];
        let obu = ObuIter::new(&data, Format::Section5)
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(obu.payload, &[0xaa, 0xbb]);
    }

    #[test]
    fn splits_section5_units() {
        let mut data = td_and_seq();
        let first = data.len();
        data.extend_from_slice(&[0x12, 0x00, 0x7a, 0x01, 0x80]);
        let units = split_temporal_units(&data, Format::Section5, &Limits::default()).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].len(), first);
        assert_eq!(units[1], &[0x12, 0x00, 0x7a, 0x01, 0x80]);
    }

    #[test]
    fn annexb_units_and_obus() {
        // One temporal unit with one frame unit holding a temporal delimiter (obu_length 1,
        // header only, no size field) and a padding OBU with 2 payload bytes.
        let frame_unit = [1u8, 0x10, 3, 0x78, 0xaa, 0xbb];
        let mut stream = vec![(1 + frame_unit.len()) as u8, frame_unit.len() as u8];
        stream.extend_from_slice(&frame_unit);
        let units = split_temporal_units(&stream, Format::AnnexB, &Limits::default()).unwrap();
        assert_eq!(units.len(), 1);
        let obus: Vec<_> = ObuIter::new(units[0], Format::AnnexB)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].header.obu_type, OBU_TEMPORAL_DELIMITER);
        assert!(obus[0].payload.is_empty());
        assert_eq!(obus[1].header.obu_type, OBU_PADDING);
        assert_eq!(obus[1].payload, &[0xaa, 0xbb]);
        // A length that passes the frame unit.
        let bad = [3u8, 9, 0x10, 0x00];
        let r: Result<Vec<_>, _> = ObuIter::new(&bad, Format::AnnexB).collect();
        assert!(r.is_err());
    }

    #[test]
    fn metadata_payload_and_trailing_bits() {
        // HDR CLL: type 1, max_cll, max_fall, trailing byte 0x80, then padding zeros.
        let m = parse_metadata(&[1, 0x03, 0xe8, 0x01, 0x90, 0x80, 0, 0]).unwrap();
        assert_eq!(m.metadata_type, METADATA_TYPE_HDR_CLL);
        assert_eq!(m.payload, [0x03, 0xe8, 0x01, 0x90]);
        // A type this decoder does not interpret is kept raw too.
        let u = parse_metadata(&[0x20, 0x07, 0x80]).unwrap();
        assert_eq!((u.metadata_type, u.payload), (0x20, vec![0x07]));
        assert_eq!(parse_metadata(&[]), Err(Error::Truncated));
    }
}
