//! AVX2 symbol search and CDF adaptation (see `super::decode_symbol`). It needs only SSE4.1
//! instructions on 128-bit vectors; the AVX2 target feature gives them the VEX encoding.
//! After dav1d's `msac_decode_symbol_adapt16` (`src/x86/msac.asm`; BSD-2-Clause, see NOTICE).

use crate::simd::x86::{load64_u16, load128_u16, store64_u16, store128_u16};
use std::arch::x86_64::*;

/// The update of spec 8.2.6 on the lanes of `c`, whose first lane is value `first` of the
/// row: lanes at or above `sym` move up towards 32768, the others down towards 0, by `rate`
/// (a shift count vector). The arithmetic wraps as the scalar code does.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(c: __m128i, first: i16, sym: i16, rate: __m128i) -> __m128i {
    let idx = _mm_add_epi16(
        _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7),
        _mm_set1_epi16(first),
    );
    let up = _mm_cmpgt_epi16(idx, _mm_set1_epi16(sym - 1));
    let down = _mm_sub_epi16(c, _mm_srl_epi16(c, rate));
    // 0x8000 - c is 32768 - c, wrapped.
    let gap = _mm_sub_epi16(_mm_set1_epi16(i16::MIN), c);
    let up_value = _mm_add_epi16(c, _mm_srl_epi16(gap, rate));
    _mm_blendv_epi8(down, up_value, up)
}

/// `symbol::cur` of the lanes of `x`, whose first lane is value `first` of an `n`-value row,
/// with `r` = `SymbolRange >> 8` in every lane: `((r * ((32768 - min(x, 32768)) >> 6)) >> 1) +
/// 4 * (n - 1 - i)`. The product needs 17 bits, so it is put together from its low and high
/// halves; the result fits 16 bits.
#[inline]
#[target_feature(enable = "avx2")]
fn cur(x: __m128i, r: __m128i, first: usize, n: usize) -> __m128i {
    let top = _mm_set1_epi16(i16::MIN);
    let g = _mm_srli_epi16::<6>(_mm_sub_epi16(top, _mm_min_epu16(x, top)));
    let lo = _mm_mullo_epi16(g, r);
    let hi = _mm_mulhi_epu16(g, r);
    let half = _mm_or_si128(_mm_slli_epi16::<15>(hi), _mm_srli_epi16::<1>(lo));
    let min_prob = _mm_sub_epi16(
        _mm_set1_epi16((4 * (n - 1 - first)) as i16),
        _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28),
    );
    _mm_add_epi16(half, min_prob)
}

/// The movemask of the lanes of `v` that are at most `cv` (unsigned: `max(v, cv) == cv`), two
/// bits per lane.
#[inline]
#[target_feature(enable = "avx2")]
fn ge(v: __m128i, cv: __m128i) -> u32 {
    _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_max_epu16(v, cv), cv)) as u32
}

/// Loads the first 8 values of `row`, or the first 4 (upper lanes zero) when it is shorter.
#[inline]
#[target_feature(enable = "avx2")]
fn load(row: &[u16]) -> __m128i {
    if let Some(a) = row.first_chunk::<8>() {
        load128_u16(a)
    } else if let Some(a) = row.first_chunk::<4>() {
        load64_u16(a)
    } else {
        _mm_setzero_si128()
    }
}

/// Stores the lanes of `v` to the first 8 values of `row`, or the first 4 when it is shorter.
#[inline]
#[target_feature(enable = "avx2")]
fn store(row: &mut [u16], v: __m128i) {
    if let Some(d) = row.first_chunk_mut::<8>() {
        store128_u16(d, v);
    } else if let Some(d) = row.first_chunk_mut::<4>() {
        store64_u16(d, v);
    }
}

/// Decodes the symbol of the 4 to 16 values of `cdf` for `SymbolValue` = `c` and
/// `SymbolRange` = `rng` and, with `Some(rate)`, adapts the values for it. Returns the symbol,
/// the `cur` before it (`rng` for symbol 0) and its own `cur`.
#[target_feature(enable = "avx2")]
pub(super) fn decode(cdf: &mut [u16], c: u32, rng: u32, rate: Option<u32>) -> (usize, u32, u32) {
    let n = cdf.len();
    // Lanes per vector: the second vector starts at value n - w.
    let w = if n >= 8 { 8 } else { 4 };
    let r = _mm_set1_epi16((rng >> 8) as i16);
    let a = load(cdf);
    let b = load(&cdf[n - w..]);
    let cur_a = cur(a, r, 0, n);
    let cur_b = cur(b, r, n - w, n);
    // Two mask bits per lane, value i at bits 2i and 2i + 1; value n - 1 always counts.
    let lane_bits = (1u32 << (2 * w)) - 1;
    let cv = _mm_set1_epi16(c as i16);
    let m = (ge(cur_a, cv) & lane_bits)
        | ((ge(cur_b, cv) & lane_bits) << (2 * (n - w)))
        | (1 << (2 * (n - 1)));
    let sym = (m.trailing_zeros() / 2) as usize;
    // The `cur` of every value: the second store overwrites the upper lanes of the first one
    // that are not values (when w = 4) with the right ones.
    let mut curs = [0u16; 16];
    if let Some(d) = curs.first_chunk_mut::<8>() {
        store128_u16(d, cur_a);
    }
    if let Some(d) = curs[n - w..].first_chunk_mut::<8>() {
        store128_u16(d, cur_b);
    }
    let v = u32::from(curs[sym]);
    let prev = if sym == 0 {
        rng
    } else {
        u32::from(curs[sym - 1])
    };
    if let Some(rate) = rate {
        let rate = _mm_cvtsi32_si128(rate as i32);
        let a = lanes(a, 0, sym as i16, rate);
        let b = lanes(b, (n - w) as i16, sym as i16, rate);
        store(cdf, a);
        store(&mut cdf[n - w..], b);
    }
    (sym, prev, v)
}
