//! Symbol decoder, spec 8.2: `init_symbol`, `read_symbol` with the CDF adaptation process,
//! `read_bool`, `read_literal` (`L(n)`), `NS(n)` and `exit_symbol`.
//!
//! The decoder follows the spec bit by bit: `SymbolValue` is a 15-bit window kept in inverted
//! form and refilled from the tile data on every renormalisation. Reads past the end of the
//! data return zero bits, as the spec requires, and never panic.

#![allow(dead_code)]

use crate::Error;

const EC_PROB_SHIFT: u32 = crate::consts::EC_PROB_SHIFT as u32;
const EC_MIN_PROB: u32 = crate::consts::EC_MIN_PROB as u32;

/// Arithmetic decoder over the bytes of one tile (spec 8.2).
pub(crate) struct SymbolDecoder<'a> {
    data: &'a [u8],
    /// Read position in bits from the start of `data` (the spec's `get_position()`).
    pos: usize,
    /// `SymbolValue`.
    symbol_value: u32,
    /// `SymbolRange`.
    symbol_range: u32,
    /// `SymbolMaxBits`; negative once padding zero bits are in use.
    symbol_max_bits: i64,
    disable_cdf_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// Spec 8.2.2, `init_symbol( sz )` with `sz = data.len()`.
    ///
    /// Fails with `Error::Invalid` when `data` is empty.
    pub(crate) fn new(
        data: &'a [u8],
        disable_cdf_update: bool,
    ) -> Result<SymbolDecoder<'a>, Error> {
        if data.is_empty() {
            return Err(Error::Invalid("empty tile data"));
        }
        let sz_bits = data.len().saturating_mul(8);
        let mut dec = SymbolDecoder {
            data,
            pos: 0,
            symbol_value: 0,
            symbol_range: 1 << 15,
            symbol_max_bits: i64::try_from(sz_bits).unwrap_or(i64::MAX) - 15,
            disable_cdf_update,
        };
        let num_bits = sz_bits.min(15) as u32;
        let buf = dec.read_bits(num_bits);
        let padded_buf = buf << (15 - num_bits);
        dec.symbol_value = ((1 << 15) - 1) ^ padded_buf;
        Ok(dec)
    }

    /// The `f( n )` descriptor on the tile data (n <= 15); bits past the end read as zero.
    fn read_bits(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            let byte = self.data.get(self.pos >> 3).copied().unwrap_or(0);
            x = (x << 1) | u32::from((byte >> (7 - (self.pos & 7))) & 1);
            self.pos += 1;
        }
        x
    }

    /// Spec 8.2.6 without the CDF update: decodes one symbol from `cdf` (N + 1 entries).
    /// Returns 0 for a slice shorter than 3 entries, which no caller passes.
    fn decode(&mut self, cdf: &[u16]) -> usize {
        let n = cdf.len().saturating_sub(1);
        if n < 2 {
            return 0;
        }
        let mut cur = self.symbol_range;
        let mut prev;
        let mut symbol = 0;
        loop {
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol]);
            cur = (((self.symbol_range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - symbol - 1) as u32;
            // A valid cdf ends in 32768, which makes cur 0 and stops the loop at symbol n - 1.
            if self.symbol_value >= cur || symbol + 1 >= n {
                break;
            }
            symbol += 1;
        }
        self.symbol_range = prev.saturating_sub(cur);
        self.symbol_value = self.symbol_value.saturating_sub(cur);

        // Renormalisation.
        let bits = self.symbol_range.leading_zeros().saturating_sub(16);
        self.symbol_range <<= bits;
        let num_bits = i64::from(bits).min(self.symbol_max_bits.max(0)) as u32;
        let new_data = self.read_bits(num_bits);
        let padded_data = new_data << (bits - num_bits);
        self.symbol_value = padded_data ^ (((self.symbol_value + 1) << bits) - 1);
        self.symbol_max_bits -= i64::from(bits);
        symbol
    }

    /// Spec 8.2.6 `read_symbol( cdf )`. `cdf` has N + 1 entries: N cumulative values in the
    /// spec's representation (the last one is 32768) followed by the adaptation counter.
    /// The cdf is adapted unless `disable_cdf_update` was set.
    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let symbol = self.decode(cdf);
        if !self.disable_cdf_update {
            adapt_cdf(cdf, symbol);
        }
        symbol
    }

    /// Spec 8.2.3 `read_bool( )`: a symbol with the fixed cdf { 1 << 14, 1 << 15, 0 }.
    pub(crate) fn read_bool(&mut self) -> bool {
        // The update is skipped: the spec rebuilds this cdf on every call.
        self.decode(&[1 << 14, 1 << 15, 0]) != 0
    }

    /// Spec 8.2.5 `read_literal( n )`, the `L(n)` descriptor: n bools, most significant first.
    /// `n` must be at most 32.
    pub(crate) fn read_literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | u32::from(self.read_bool());
        }
        x
    }

    /// The `NS(n)` descriptor (spec 4.10): a value in `0..n` coded with `L()` bits.
    /// Returns 0 for `n == 0`.
    pub(crate) fn read_ns(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let w = 32 - n.leading_zeros();
        let m = ((1u64 << w) - u64::from(n)) as u32;
        let v = self.read_literal(w - 1);
        if v < m {
            return v;
        }
        let extra_bit = self.read_literal(1);
        (v << 1) - m + extra_bit
    }

    /// Spec 8.2.4 exit process for the symbol decoder. Returns true when the trailing bit is 1
    /// and every following padding bit is 0 and `SymbolMaxBits >= -14`, as bitstream
    /// conformance requires; the caller decides what to do about a false result.
    ///
    /// Saving the tile's CDFs for the frame end update is done by the caller (the CDF arrays are
    /// not owned by the decoder).
    pub(crate) fn exit(self) -> bool {
        if self.symbol_max_bits < -14 {
            return false;
        }
        let pos = self.pos as i64;
        let trailing_bit_position = pos - (self.symbol_max_bits + 15).min(15);
        let padding_end_position = pos + self.symbol_max_bits.max(0);
        let bit = |i: i64| -> Option<u8> {
            let i = usize::try_from(i).ok()?;
            self.data.get(i >> 3).map(|b| (b >> (7 - (i & 7))) & 1)
        };
        if bit(trailing_bit_position) != Some(1) {
            return false;
        }
        (trailing_bit_position + 1..padding_end_position).all(|i| bit(i) == Some(0))
    }
}

/// The CDF update of spec 8.2.6 for a decoded `symbol`.
fn adapt_cdf(cdf: &mut [u16], symbol: usize) {
    if cdf.len() < 3 {
        return;
    }
    let n = cdf.len() - 1;
    let count = cdf[n];
    let rate = 3
        + u32::from(count > 15)
        + u32::from(count > 31)
        + (31 - (n as u32).leading_zeros()).min(2);
    for (i, c) in cdf[..n - 1].iter_mut().enumerate() {
        if i >= symbol {
            *c += ((1 << 15) - *c) >> rate;
        } else {
            *c -= *c >> rate;
        }
    }
    cdf[n] += u16::from(count < 32);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_data_is_invalid() {
        assert!(SymbolDecoder::new(&[], false).is_err());
    }

    #[test]
    fn zero_bytes_decode_zero_bools_and_never_panic_past_the_end() {
        // The value register is the inverse of the data, so zero data always selects symbol 0.
        let mut dec = SymbolDecoder::new(&[0, 0, 0, 0], false).unwrap();
        for _ in 0..1000 {
            assert!(!dec.read_bool());
        }
        assert_eq!(dec.read_literal(32), 0);
        assert_eq!(dec.read_ns(5), 0);
    }

    #[test]
    fn all_one_bytes_decode_true_bools() {
        // value = 0x7fff ^ 0x7fff = 0: every bool takes symbol 1 and refills one 1 bit.
        let mut dec = SymbolDecoder::new(&[0xff; 8], false).unwrap();
        for _ in 0..40 {
            assert!(dec.read_bool());
        }
        let mut dec = SymbolDecoder::new(&[0xff; 8], false).unwrap();
        assert_eq!(dec.read_literal(16), 0xffff);
    }

    #[test]
    fn read_symbol_adapts_a_binary_cdf() {
        // Symbol 0 on zero data: rate = 3 + 0 + 0 + min(FloorLog2(2), 2) = 4, so
        // cdf[0] = 16384 + ((32768 - 16384) >> 4) = 17408, and the counter becomes 1.
        let mut dec = SymbolDecoder::new(&[0, 0], false).unwrap();
        let mut cdf = [1 << 14, 1 << 15, 0];
        assert_eq!(dec.read_symbol(&mut cdf), 0);
        assert_eq!(cdf, [17408, 32768, 1]);
    }

    #[test]
    fn read_symbol_leaves_cdf_alone_when_update_disabled() {
        let mut dec = SymbolDecoder::new(&[0, 0], true).unwrap();
        let mut cdf = [1 << 14, 1 << 15, 0];
        assert_eq!(dec.read_symbol(&mut cdf), 0);
        assert_eq!(cdf, [1 << 14, 1 << 15, 0]);
    }

    #[test]
    fn counter_saturates_at_32() {
        let mut dec = SymbolDecoder::new(&[0; 16], false).unwrap();
        let mut cdf = [1 << 14, 1 << 15, 0];
        for _ in 0..40 {
            dec.read_symbol(&mut cdf);
        }
        assert_eq!(cdf[2], 32);
        assert_eq!(cdf[1], 32768);
        assert!(cdf[0] < 32768);
    }

    #[test]
    fn read_ns_stays_in_range() {
        for n in 1..=40 {
            let mut dec = SymbolDecoder::new(&[0xa5, 0x5a, 0xc3, 0x3c], false).unwrap();
            for _ in 0..8 {
                assert!(dec.read_ns(n) < n);
            }
        }
    }

    #[test]
    fn exit_checks_trailing_bit_and_padding() {
        // One byte: the decoder has read all 8 bits, so the trailing bit is bit 0.
        assert!(SymbolDecoder::new(&[0x80], false).unwrap().exit());
        assert!(!SymbolDecoder::new(&[0x00], false).unwrap().exit());
        assert!(!SymbolDecoder::new(&[0x81], false).unwrap().exit());
    }
}
