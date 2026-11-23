//! The bit reader of the syntax tables: the descriptors f(n), uvlc(), le(n), leb128(), su(n) and
//! ns(n) of spec 4.10, over a byte slice (the parsing process of spec 8.1).
#![allow(dead_code)]

use crate::error::Error;

/// Reads bits from the most significant bit of each byte first. A read past the end of the data
/// is [`Error::Truncated`]; nothing here panics.
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// `get_position()` of the spec: the number of bits read so far.
    pub(crate) fn get_position(&self) -> usize {
        self.pos
    }

    /// The bits that are left.
    pub(crate) fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    /// One bit as `0` or `1`.
    fn bit(&mut self) -> Result<u32, Error> {
        let byte = *self.data.get(self.pos >> 3).ok_or(Error::Truncated)?;
        let b = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(b))
    }

    /// f(n): an unsigned n-bit number, high bit first. `n` is at most 32 (0 reads nothing and
    /// returns 0).
    pub(crate) fn f(&mut self, n: u32) -> Result<u32, Error> {
        if n > 32 {
            return Err(Error::Invalid("bit count above 32"));
        }
        if self.bits_left() < n as usize {
            return Err(Error::Truncated);
        }
        let mut x: u64 = 0;
        for _ in 0..n {
            x = (x << 1) | u64::from(self.bit()?);
        }
        Ok(x as u32)
    }

    /// f(1) as a flag.
    pub(crate) fn flag(&mut self) -> Result<bool, Error> {
        Ok(self.bit()? != 0)
    }

    /// uvlc(): a variable length unsigned number.
    pub(crate) fn uvlc(&mut self) -> Result<u32, Error> {
        let mut leading_zeros = 0u32;
        while !self.flag()? {
            leading_zeros += 1;
        }
        if leading_zeros >= 32 {
            return Ok(u32::MAX);
        }
        let value = self.f(leading_zeros)?;
        Ok(value + ((1u32 << leading_zeros) - 1))
    }

    /// le(n): an unsigned little-endian number of `n` bytes (at most 4), at a byte boundary.
    pub(crate) fn le(&mut self, n: u32) -> Result<u32, Error> {
        if n > 4 {
            return Err(Error::Invalid("le byte count above 4"));
        }
        let mut t = 0u32;
        for i in 0..n {
            t |= self.f(8)? << (i * 8);
        }
        Ok(t)
    }

    /// leb128(): an unsigned little-endian base 128 number of at most 8 bytes. A value above
    /// `u32::MAX` (the spec's conformance limit) or a ninth byte is `Error::Invalid`. Returns
    /// the value and `Leb128Bytes`, the bytes it took.
    pub(crate) fn leb128(&mut self) -> Result<(u32, usize), Error> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.f(8)?;
            value |= u64::from(byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                let value = u32::try_from(value).map_err(|_| Error::Invalid("leb128 too large"))?;
                return Ok((value, i + 1));
            }
        }
        Err(Error::Invalid("leb128 longer than 8 bytes"))
    }

    /// su(n): a signed number from its `n` low bits (1 to 32).
    pub(crate) fn su(&mut self, n: u32) -> Result<i32, Error> {
        if n == 0 {
            return Err(Error::Invalid("su of zero bits"));
        }
        let value = self.f(n)?;
        let sign_mask = 1u32 << (n - 1);
        if value & sign_mask != 0 {
            Ok((value as i32).wrapping_sub((sign_mask as i32).wrapping_mul(2)))
        } else {
            Ok(value as i32)
        }
    }

    /// ns(n): a non-symmetric unsigned number in `0..n`; `n` is in `1..=1 << 30`.
    pub(crate) fn ns(&mut self, n: u32) -> Result<u32, Error> {
        if n == 0 || n > 1 << 30 {
            return Err(Error::Invalid("ns range"));
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// byte_alignment(): skips zero bits up to the next byte boundary. A one bit there is
    /// `Error::Invalid`.
    pub(crate) fn byte_alignment(&mut self) -> Result<(), Error> {
        while self.pos & 7 != 0 {
            if self.bit()? != 0 {
                return Err(Error::Invalid("nonzero bit in byte alignment"));
            }
        }
        Ok(())
    }

    /// The bytes from the current position on; the position must be byte aligned.
    pub(crate) fn rest(&self) -> &'a [u8] {
        self.data.get(self.pos >> 3..).unwrap_or(&[])
    }
}

/// FloorLog2(x) of spec 4.7 for x > 0 (0 for x = 0).
pub(crate) fn floor_log2(x: u32) -> u32 {
    31u32.saturating_sub(x.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_reads_high_bit_first_and_truncates() {
        let mut r = BitReader::new(&[0b1010_0110, 0xff]);
        assert_eq!(r.f(3).unwrap(), 0b101);
        assert_eq!(r.f(5).unwrap(), 0b00110);
        assert_eq!(r.get_position(), 8);
        assert_eq!(r.f(0).unwrap(), 0);
        assert_eq!(r.f(9), Err(Error::Truncated));
        // A failed read consumes nothing.
        assert_eq!(r.f(8).unwrap(), 0xff);
        assert_eq!(r.flag(), Err(Error::Truncated));
    }

    #[test]
    fn f32_full_width() {
        let mut r = BitReader::new(&[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(r.f(32).unwrap(), 0xdead_beef);
    }

    #[test]
    fn uvlc_values() {
        // 1 -> 0; 010 -> 1; 011 -> 2; 00100 -> 3; 00111 -> 6.
        let mut r = BitReader::new(&[0b1010_0110, 0b0100_0011, 0b1000_0000]);
        assert_eq!(r.uvlc().unwrap(), 0);
        assert_eq!(r.uvlc().unwrap(), 1);
        assert_eq!(r.uvlc().unwrap(), 2);
        assert_eq!(r.uvlc().unwrap(), 3);
        assert_eq!(r.uvlc().unwrap(), 6);
        // 32 zeros then a one: the maximum, with no value bits.
        let mut r = BitReader::new(&[0, 0, 0, 0, 0x80]);
        assert_eq!(r.uvlc().unwrap(), u32::MAX);
        // Only zeros: truncated.
        let mut r = BitReader::new(&[0, 0]);
        assert_eq!(r.uvlc(), Err(Error::Truncated));
    }

    #[test]
    fn le_reads_little_endian() {
        let mut r = BitReader::new(&[0x34, 0x12, 0x00]);
        assert_eq!(r.le(2).unwrap(), 0x1234);
        assert_eq!(r.le(2), Err(Error::Truncated));
    }

    #[test]
    fn leb128_values_and_limits() {
        let mut r = BitReader::new(&[0xe5, 0x8e, 0x26, 0x05]);
        assert_eq!(r.leb128().unwrap(), (624_485, 3));
        assert_eq!(r.leb128().unwrap(), (5, 1));
        // Padded encoding of 1.
        let mut r = BitReader::new(&[0x81, 0x80, 0x00]);
        assert_eq!(r.leb128().unwrap(), (1, 3));
        // 2^32 does not fit.
        let mut r = BitReader::new(&[0x80, 0x80, 0x80, 0x80, 0x10]);
        assert_eq!(r.leb128(), Err(Error::Invalid("leb128 too large")));
        // u32::MAX does.
        let mut r = BitReader::new(&[0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert_eq!(r.leb128().unwrap(), (u32::MAX, 5));
        // Nine bytes.
        let mut r = BitReader::new(&[0x80; 9]);
        assert!(matches!(r.leb128(), Err(Error::Invalid(_))));
        let mut r = BitReader::new(&[0x80]);
        assert_eq!(r.leb128(), Err(Error::Truncated));
    }

    #[test]
    fn su_sign_extends() {
        // su(4): 0111 = 7, 1000 = -8, 1111 = -1.
        let mut r = BitReader::new(&[0b0111_1000, 0b1111_0000]);
        assert_eq!(r.su(4).unwrap(), 7);
        assert_eq!(r.su(4).unwrap(), -8);
        assert_eq!(r.su(4).unwrap(), -1);
        // su(1+6) as used by the delta quantizers: 1000000 = -64.
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.su(7).unwrap(), -64);
    }

    #[test]
    fn ns_matches_the_spec_table() {
        // ns(5): 00 01 10 110 111 -> 0 1 2 3 4.
        let bits = [0b0001_1011, 0b0111_0000];
        let mut r = BitReader::new(&bits);
        for want in 0..5 {
            assert_eq!(r.ns(5).unwrap(), want);
        }
        // ns(1) reads no bits.
        let mut r = BitReader::new(&[]);
        assert_eq!(r.ns(1).unwrap(), 0);
        // ns(4) is a plain 2 bit number.
        let mut r = BitReader::new(&[0b1100_0000]);
        assert_eq!(r.ns(4).unwrap(), 3);
    }

    #[test]
    fn byte_alignment_and_rest() {
        let mut r = BitReader::new(&[0b1010_0000, 0x77]);
        r.f(3).unwrap();
        r.byte_alignment().unwrap();
        assert_eq!(r.get_position(), 8);
        assert_eq!(r.rest(), &[0x77]);
        let mut r = BitReader::new(&[0b1011_0000]);
        r.f(3).unwrap();
        assert!(r.byte_alignment().is_err());
    }
}
