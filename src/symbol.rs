//! Symbol decoder, spec 8.2: `init_symbol`, `read_symbol` with the CDF adaptation process,
//! `read_bool`, `read_literal` (`L(n)`), `NS(n)` and `exit_symbol`.
//!
//! The window design is ported from dav1d's `src/msac.c` (Copyright 2018-2019, VideoLAN and
//! dav1d authors, Copyright 2018-2019, Two Orioles, LLC; BSD-2-Clause, see NOTICE and
//! docs/dependencies.md): `SymbolValue` is the top 16 bits of a 64-bit window whose lower bits
//! hold the next bits of the tile data, inverted, refilled a byte at a time; the renormalisation
//! shifts by `leading_zeros`; the symbol search stops at the first partition point at or below
//! the value (or, on the SIMD tiers, compares it with all of them at once and adapts the row in
//! the same pass, `simd::entropy::decode_symbol`); `read_bool` has its own fast path. The CDFs
//! keep this crate's convention (the spec's values, ending in 32768, then the counter).
//!
//! The arithmetic is the spec's, so every symbol and every state equal those of the bit by bit
//! reader of the spec, which stays in `oracle` (tests and the fuzz target compare the two).
//! The data is read as if followed by zero bytes, as the spec requires, and nothing panics
//! past the end.

#![allow(dead_code)]

use crate::Error;
use crate::simd::{self, Simd};

const EC_PROB_SHIFT: u32 = crate::consts::EC_PROB_SHIFT as u32;
const EC_MIN_PROB: u32 = crate::consts::EC_MIN_PROB as u32;

/// Arithmetic decoder over the bytes of one tile (spec 8.2).
pub(crate) struct SymbolDecoder<'a> {
    data: &'a [u8],
    /// The next byte of `data` to load into the window.
    pos: usize,
    /// The window (dav1d's `dif`): `SymbolValue` in the top 16 bits, then the next `cnt` bits
    /// of the data, inverted; every bit below them is 1 (an inverted zero bit, which is what
    /// the data reads as past its end). While `cnt` is negative the low `-cnt` bits of
    /// `SymbolValue` still wait for their data.
    dif: u64,
    /// `SymbolRange`.
    rng: u32,
    /// The number of data bits in the window below `SymbolValue` (see `dif`). It stays exact
    /// past the end of the data: `8 * pos - cnt` is always 15 plus the number of bits shifted
    /// out by the renormalisations.
    cnt: i64,
    disable_cdf_update: bool,
    /// The tier of the CDF adaptation.
    simd: Simd,
}

impl<'a> SymbolDecoder<'a> {
    /// Spec 8.2.2, `init_symbol( sz )` with `sz = data.len()`.
    ///
    /// Fails with `Error::Invalid` when `data` is empty.
    pub(crate) fn new(
        data: &'a [u8],
        disable_cdf_update: bool,
    ) -> Result<SymbolDecoder<'a>, Error> {
        Self::with_simd(data, disable_cdf_update, simd::best())
    }

    /// [`SymbolDecoder::new`] with the CDF adaptation of tier `simd`.
    pub(crate) fn with_simd(
        data: &'a [u8],
        disable_cdf_update: bool,
        simd: Simd,
    ) -> Result<SymbolDecoder<'a>, Error> {
        if data.is_empty() {
            return Err(Error::Invalid("empty tile data"));
        }
        // SymbolValue = 0 with 15 bits waiting: the refill XORs the first 15 data bits into
        // them, which gives the spec's `((1 << 15) - 1) ^ f( 15 )`.
        let mut dec = SymbolDecoder {
            data,
            pos: 0,
            dif: (1 << 63) - 1,
            rng: 1 << 15,
            cnt: -15,
            disable_cdf_update,
            simd,
        };
        dec.refill();
        Ok(dec)
    }

    /// Loads whole bytes below the valid bits of the window until it is full or the data ends.
    /// Called only with `cnt` in -16..0 while data is left (a renormalisation shifts by at most
    /// 16), so no byte is shifted by more than 56.
    #[inline]
    fn refill(&mut self) {
        let mut c = 40 - self.cnt;
        while c >= 0 {
            let Some(&b) = self.data.get(self.pos) else {
                break;
            };
            self.dif ^= u64::from(b) << c;
            self.pos += 1;
            c -= 8;
        }
        self.cnt = 40 - c;
    }

    /// Sets the window to `dif` and `SymbolRange` to `rng`, then renormalises (spec 8.2.6):
    /// shifts both left until the range has its bit 15 set, shifting in ones (inverted data
    /// bits), and refills when the window runs short.
    #[inline]
    fn norm(&mut self, dif: u64, rng: u32) {
        // 16 for a zero range, which only a CDF that breaks the spec rules gives.
        let d = rng.leading_zeros().saturating_sub(16);
        self.dif = (dif << d) | ((1 << d) - 1);
        self.rng = rng << d;
        self.cnt -= i64::from(d);
        if self.cnt < 0 {
            self.refill();
        }
    }

    /// Spec 8.2.6 `read_symbol( cdf )`. `cdf` has N + 1 entries: N cumulative values in the
    /// spec's representation (the last one is 32768) followed by the adaptation counter.
    /// The cdf is adapted unless `disable_cdf_update` was set. Returns 0 and reads nothing for
    /// a slice shorter than 3 entries, which no caller passes.
    ///
    /// The search and the update run in `simd::entropy::decode_symbol`: for a CDF of the
    /// spec's form (non-decreasing, at most 32768, the last value 32768) the partition point
    /// `cur` falls strictly with the symbol; any other CDF gives some symbol below N and no
    /// panic.
    #[inline]
    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        if cdf.len() < 3 {
            return 0;
        }
        let c = (self.dif >> 48) as u32;
        let (symbol, prev, v) =
            simd::entropy::decode_symbol(self.simd, cdf, c, self.rng, !self.disable_cdf_update);
        self.norm(
            self.dif.wrapping_sub(u64::from(v) << 48),
            prev.wrapping_sub(v) & 0xffff,
        );
        symbol
    }

    /// Spec 8.2.3 `read_bool( )`: a symbol with the fixed cdf { 1 << 14, 1 << 15, 0 }, whose
    /// `cur` for symbol 0 is `((SymbolRange >> 8) << 7) + EC_MIN_PROB` (dav1d's
    /// `decode_bool_equi`). No update: the spec rebuilds this cdf on every call.
    #[inline]
    pub(crate) fn read_bool(&mut self) -> bool {
        let v = ((self.rng >> 8) << 7) + EC_MIN_PROB;
        let vw = u64::from(v) << 48;
        // Symbol 0 when SymbolValue >= cur.
        let zero = self.dif >= vw;
        let (dif, rng) = if zero {
            (self.dif - vw, self.rng.wrapping_sub(v) & 0xffff)
        } else {
            (self.dif, v)
        };
        self.norm(dif, rng);
        !zero
    }

    /// Spec 8.2.5 `read_literal( n )`, the `L(n)` descriptor: n bools, most significant first.
    /// `n` must be at most 32. A loop of the equiprobable [`SymbolDecoder::read_bool`].
    #[inline]
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

    /// The spec's `get_position( )` (bits read from the data, at most all of them) and
    /// `SymbolMaxBits`.
    fn spec_position(&self) -> (i64, i64) {
        let len = 8 * self.data.len() as i64;
        let bits = 8 * self.pos as i64 - self.cnt;
        (bits.min(len), len - bits)
    }

    /// `SymbolValue`, `SymbolRange`, `get_position( )` and `SymbolMaxBits`, to compare with
    /// `oracle::SymbolDecoder::state`.
    #[cfg(any(test, feature = "fuzzing"))]
    fn state(&self) -> (u32, u32, i64, i64) {
        let (pos, max_bits) = self.spec_position();
        ((self.dif >> 48) as u32, self.rng, pos, max_bits)
    }

    /// Spec 8.2.4 exit process for the symbol decoder. Returns true when the trailing bit is 1
    /// and every following padding bit is 0 and `SymbolMaxBits >= -14`, as bitstream
    /// conformance requires; the caller decides what to do about a false result.
    ///
    /// Saving the tile's CDFs for the frame end update is done by the caller (the CDF arrays are
    /// not owned by the decoder).
    pub(crate) fn exit(self) -> bool {
        let (pos, max_bits) = self.spec_position();
        exit_ok(self.data, pos, max_bits)
    }
}

/// The checks of the exit process (spec 8.2.4) at bit position `pos` with `SymbolMaxBits`
/// `symbol_max_bits`.
fn exit_ok(data: &[u8], pos: i64, symbol_max_bits: i64) -> bool {
    if symbol_max_bits < -14 {
        return false;
    }
    let trailing_bit_position = pos - (symbol_max_bits + 15).min(15);
    let padding_end_position = pos + symbol_max_bits.max(0);
    let bit = |i: i64| -> Option<u8> {
        let i = usize::try_from(i).ok()?;
        data.get(i >> 3).map(|b| (b >> (7 - (i & 7))) & 1)
    };
    if bit(trailing_bit_position) != Some(1) {
        return false;
    }
    (trailing_bit_position + 1..padding_end_position).all(|i| bit(i) == Some(0))
}

/// `cur` of spec 8.2.6 for symbol `i` of an N-symbol CDF whose value `i` is `x`, with
/// `r = SymbolRange >> 8`. A value above 32768 (not of the spec's form) counts as 32768, as in
/// the SIMD tiers. At most 65340, so it fits 16 bits.
#[inline]
fn cur(r: u32, x: u16, n: usize, i: usize) -> u32 {
    let f = 32768u32.saturating_sub(u32::from(x));
    ((r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - 1 - i) as u32
}

/// The symbol search of spec 8.2.6 on `cdf` (N values, then the counter) for
/// `SymbolValue` = `c` and `SymbolRange` = `rng` (the scalar tier of
/// `simd::entropy::decode_symbol`; dav1d's `msac_decode_symbol_adapt`): the first symbol below
/// N - 1 whose `cur` is at most `c`, else N - 1, stopping there (most symbols are small).
/// Returns the symbol, the `cur` before it (`rng` for symbol 0) and its own `cur`; `(0, rng,
/// 0)` for N = 0.
#[inline]
pub(crate) fn search(cdf: &[u16], c: u32, rng: u32) -> (usize, u32, u32) {
    let n = cdf.len().saturating_sub(1);
    let r = rng >> 8;
    let mut prev = rng;
    for (i, &x) in cdf[..n].iter().enumerate() {
        let v = cur(r, x, n, i);
        if c >= v || i + 1 == n {
            return (i, prev, v);
        }
        prev = v;
    }
    (0, rng, 0)
}

/// The adaptation rate of spec 8.2.6 for a CDF of `n` symbols whose counter is `count`.
#[inline]
pub(crate) fn adapt_rate(count: u16, n: usize) -> u32 {
    3 + u32::from(count > 15) + u32::from(count > 31) + (31 - (n as u32).leading_zeros()).min(2)
}

/// The CDF update of spec 8.2.6 for a decoded `symbol` (the scalar tier of
/// `simd::entropy::adapt_cdf`). The arithmetic wraps, so any values give some result and the
/// SIMD tiers give the same; for a CDF of the spec's form nothing wraps.
pub(crate) fn adapt_cdf(cdf: &mut [u16], symbol: usize) {
    let Some((count, values)) = cdf.split_last_mut() else {
        return;
    };
    let n = values.len();
    if n < 2 {
        return;
    }
    let rate = adapt_rate(*count, n);
    let (below, above) = values[..n - 1].split_at_mut(symbol.min(n - 1));
    for c in below {
        *c -= *c >> rate;
    }
    for c in above {
        *c = c.wrapping_add(32768u16.wrapping_sub(*c) >> rate);
    }
    *count += u16::from(*count < 32);
}

/// The bit by bit reader of the spec that [`SymbolDecoder`] replaced: the oracle of the tests
/// and of the `simd_entropy` fuzz target.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) mod oracle {
    use super::{EC_MIN_PROB, EC_PROB_SHIFT, exit_ok};
    use crate::Error;

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

        /// Spec 8.2.6 without the CDF update.
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

        /// Spec 8.2.6 `read_symbol( cdf )`.
        pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
            let symbol = self.decode(cdf);
            if !self.disable_cdf_update {
                adapt_cdf(cdf, symbol);
            }
            symbol
        }

        /// Spec 8.2.3 `read_bool( )`.
        pub(crate) fn read_bool(&mut self) -> bool {
            self.decode(&[1 << 14, 1 << 15, 0]) != 0
        }

        /// Spec 8.2.5 `read_literal( n )`.
        pub(crate) fn read_literal(&mut self, n: u32) -> u32 {
            let mut x = 0;
            for _ in 0..n {
                x = (x << 1) | u32::from(self.read_bool());
            }
            x
        }

        /// The `NS(n)` descriptor (spec 4.10).
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

        /// `SymbolValue`, `SymbolRange`, `get_position( )` and `SymbolMaxBits`.
        pub(crate) fn state(&self) -> (u32, u32, i64, i64) {
            (
                self.symbol_value,
                self.symbol_range,
                self.pos as i64,
                self.symbol_max_bits,
            )
        }

        /// Spec 8.2.4 exit process.
        pub(crate) fn exit(self) -> bool {
            exit_ok(self.data, self.pos as i64, self.symbol_max_bits)
        }
    }

    /// The CDF update of spec 8.2.6 for a decoded `symbol`.
    pub(crate) fn adapt_cdf(cdf: &mut [u16], symbol: usize) {
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
}

/// A xorshift generator for the tests and the fuzz targets (no dependency).
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) struct XorShift(pub(crate) u64);

#[cfg(any(test, feature = "fuzzing"))]
impl XorShift {
    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut x = self.0.max(1);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

/// A CDF of the spec's form with 2 to 16 symbols and a counter in 0..=32: sorted random values
/// (some 0, some 32768), the last one 32768.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn random_cdf(rng: &mut XorShift) -> Vec<u16> {
    let n = 2 + rng.next_u32() as usize % 15;
    let mut cdf: Vec<u16> = (0..n - 1)
        .map(|_| match rng.next_u32() % 8 {
            0 => 0,
            1 => 32768,
            _ => (rng.next_u32() % 32769) as u16,
        })
        .collect();
    cdf.sort_unstable();
    cdf.push(32768);
    cdf.push((rng.next_u32() % 33) as u16);
    cdf
}

/// Runs `ops` random reads (`seed` picks them and the CDFs) on `data` with [`SymbolDecoder`]
/// at tier `simd` and with [`oracle::SymbolDecoder`], and asserts that every result, every
/// state, the adapted CDFs and the exit result agree.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn check_against_oracle(
    data: &[u8],
    disable_cdf_update: bool,
    seed: u64,
    ops: usize,
    simd: Simd,
) {
    let (mut a, mut b) = match (
        SymbolDecoder::with_simd(data, disable_cdf_update, simd),
        oracle::SymbolDecoder::new(data, disable_cdf_update),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            assert!(a.is_err() && b.is_err());
            return;
        }
    };
    assert_eq!(a.state(), b.state());
    let mut rng = XorShift(seed);
    let mut cdfs: Vec<Vec<u16>> = (0..8).map(|_| random_cdf(&mut rng)).collect();
    let mut old_cdfs = cdfs.clone();
    for _ in 0..ops {
        let r = rng.next_u32();
        let arg = r >> 8;
        match r % 8 {
            0 => assert_eq!(a.read_bool(), b.read_bool()),
            1 => assert_eq!(a.read_literal(arg % 33), b.read_literal(arg % 33)),
            2 => assert_eq!(a.read_ns(arg % 300), b.read_ns(arg % 300)),
            _ => {
                let i = arg as usize % cdfs.len();
                assert_eq!(a.read_symbol(&mut cdfs[i]), b.read_symbol(&mut old_cdfs[i]));
            }
        }
        assert_eq!(a.state(), b.state());
    }
    assert_eq!(cdfs, old_cdfs);
    assert_eq!(a.exit(), b.exit());
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

    #[test]
    fn adapt_cdf_matches_the_oracle() {
        let mut rng = XorShift(7);
        for _ in 0..2000 {
            let mut cdf = random_cdf(&mut rng);
            let mut old = cdf.clone();
            let symbol = rng.next_u32() as usize % (cdf.len() - 1);
            adapt_cdf(&mut cdf, symbol);
            oracle::adapt_cdf(&mut old, symbol);
            assert_eq!(cdf, old);
        }
    }

    /// Long random streams with random CDFs: every symbol and state equal the spec reader's,
    /// on every tier, with and without CDF updates, including far past the end of the data.
    #[test]
    fn matches_the_oracle_on_random_streams() {
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        for &simd in &simd::available() {
            for round in 0..60 {
                let len = match round {
                    0..=15 => 1 + round,
                    _ => 1 + rng.next_u32() as usize % 600,
                };
                let data: Vec<u8> = (0..len)
                    .map(|_| match rng.next_u32() % 6 {
                        0 => 0,
                        1 => 0xff,
                        _ => rng.next_u32() as u8,
                    })
                    .collect();
                // Enough reads to run well past the end of the data.
                let ops = 200 + 8 * len;
                check_against_oracle(&data, round % 5 == 4, rng.next_u32().into(), ops, simd);
            }
        }
    }

    #[test]
    fn matches_the_oracle_on_constant_streams() {
        for &simd in &simd::available() {
            for b in [0x00, 0xff, 0x80, 0x7f, 0x01] {
                for len in [1, 2, 3, 7, 8, 9, 64] {
                    check_against_oracle(&vec![b; len], false, 99 + len as u64, 400, simd);
                }
            }
        }
    }
}
