//! Kernel family `entropy`: the symbol search and CDF adaptation of spec 8.2.6
//! (`read_symbol`) for CDFs of 4 to 16 symbols, one lane per cumulative value, in one pass over
//! the row (dav1d's `msac_decode_symbol_adapt16`). The scalar oracle is `symbol::search` then
//! `symbol::adapt_cdf`; the rest of the symbol reader (`src/symbol.rs`) is scalar.
//!
//! The kernels load the first N values of the row with two overlapping vectors of 4 or 8 lanes,
//! compute `cur` of every lane, take the first lane whose `cur` is at most `SymbolValue` (N - 1
//! if none), then update the same vectors (the last value is 32768 in a valid CDF; the dispatch
//! puts it back afterwards). Both vectors are computed from the original values before either
//! is stored, so the overlap is harmless. AVX-512 has nothing to add for at most 16 lanes: its
//! tier runs the AVX2 kernel.
//!
//! The kernels follow the design of dav1d's `msac_decode_symbol_adapt16` (`src/x86/msac.asm`,
//! `src/arm/64/msac.S`; Copyright 2019, VideoLAN and dav1d authors, Copyright 2019, Two Orioles,
//! LLC; BSD-2-Clause, see NOTICE and docs/dependencies.md).

use super::{Simd, Tier};
use crate::symbol;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Decodes one symbol from `cdf` (N cumulative values, then the counter; spec 8.2.6) for
/// `SymbolValue` = `c` and `SymbolRange` = `rng`, and adapts `cdf` when `update` is set, as
/// `symbol::search` then `symbol::adapt_cdf` do, with the kernel of `simd` when N is 4 to 16.
/// Returns the symbol, the `cur` before it (`rng` for symbol 0) and its own `cur`.
#[inline]
#[allow(unsafe_code)]
pub(crate) fn decode_symbol(
    simd: Simd,
    cdf: &mut [u16],
    c: u32,
    rng: u32,
    update: bool,
) -> (usize, u32, u32) {
    let n = cdf.len().wrapping_sub(1);
    if simd.tier() == Tier::Scalar || !(4..=16).contains(&n) {
        return scalar(cdf, c, rng, update);
    }
    let count = cdf[n];
    let last = cdf[n - 1];
    let rate = update.then(|| symbol::adapt_rate(count, n));
    let values = &mut cdf[..n];
    let out = match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: `simd` is the proof (simd::available / simd::best) that the CPU has this
            // tier; AVX-512 implies AVX2.
            unsafe { avx2::decode(values, c, rng, rate) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `simd` is the proof (simd::available / simd::best) that the CPU has NEON.
            unsafe { neon::decode(values, c, rng, rate) }
        }
        _ => return scalar(cdf, c, rng, update),
    };
    if update {
        cdf[n - 1] = last;
        cdf[n] = count + u16::from(count < 32);
    }
    out
}

/// The scalar tier of [`decode_symbol`].
#[inline]
fn scalar(cdf: &mut [u16], c: u32, rng: u32, update: bool) -> (usize, u32, u32) {
    let out = symbol::search(cdf, c, rng);
    if update {
        symbol::adapt_cdf(cdf, out.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::XorShift;

    /// Every tier against scalar on random rows (any u16 values, counters up to 39) of every
    /// length, at unaligned starts, with edge and random `SymbolValue` and `SymbolRange`, with
    /// and without the update.
    #[test]
    fn decode_symbol_matches_scalar() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for len in 3..=18 {
            for round in 0..300 {
                let off = round % 3;
                let mut buf: Vec<u16> = (0..len + off)
                    .map(|i| match (round + i) % 7 {
                        0 => 0,
                        1 => 32768,
                        2 => u16::MAX,
                        3 => 1,
                        _ => rng.next_u32() as u16,
                    })
                    .collect();
                // Sorted rows are CDFs of the spec's form (but for the u16::MAX values).
                if round % 2 == 0 {
                    buf[off..len + off - 1].sort_unstable();
                }
                if let Some(c) = buf.last_mut() {
                    *c = (rng.next_u32() % 40) as u16;
                }
                let c = match round % 5 {
                    0 => 0,
                    1 => 0xffff,
                    _ => rng.next_u32() >> 16,
                };
                let range = match round % 4 {
                    0 => 0x8000,
                    1 => 0xffff,
                    2 => rng.next_u32() >> 16,
                    _ => 0x8000 | (rng.next_u32() >> 17),
                };
                let update = round % 3 != 2;
                let mut want = buf.clone();
                let want_out = scalar(&mut want[off..], c, range, update);
                for &simd in &crate::simd::available() {
                    let mut got = buf.clone();
                    let got_out = decode_symbol(simd, &mut got[off..], c, range, update);
                    assert_eq!(
                        got_out,
                        want_out,
                        "{:?} len {len} round {round}",
                        simd.tier()
                    );
                    assert_eq!(got, want, "{:?} len {len} round {round}", simd.tier());
                }
            }
        }
    }

    /// Valid CDFs decoded and adapted many times in a row stay equal on every tier.
    #[test]
    fn decode_symbol_sequences_match_scalar() {
        let mut rng = XorShift(5);
        for _ in 0..300 {
            let start = symbol::random_cdf(&mut rng);
            let steps: Vec<(u32, u32)> = (0..50)
                .map(|_| (rng.next_u32() >> 16, 0x8000 | (rng.next_u32() >> 17)))
                .collect();
            let mut want = start.clone();
            let want_out: Vec<_> = steps
                .iter()
                .map(|&(c, r)| scalar(&mut want, c, r, true))
                .collect();
            for &simd in &crate::simd::available() {
                let mut got = start.clone();
                let got_out: Vec<_> = steps
                    .iter()
                    .map(|&(c, r)| decode_symbol(simd, &mut got, c, r, true))
                    .collect();
                assert_eq!(got_out, want_out, "{:?}", simd.tier());
                assert_eq!(got, want, "{:?}", simd.tier());
            }
        }
    }
}
