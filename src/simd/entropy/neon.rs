//! NEON symbol search and CDF adaptation (see `super::decode_symbol`). After dav1d's
//! `msac_decode_symbol_adapt16` (`src/arm/64/msac.S`; BSD-2-Clause, see NOTICE).

use crate::simd::arm::{load_u16x4, load_u16x8, store_u16x4, store_u16x8};
use std::arch::aarch64::*;

/// The update of spec 8.2.6 on 8 lanes whose first lane is value `first` of the row: lanes at
/// or above `sym` move up towards 32768, the others down towards 0; `rate` holds minus the
/// shift. The arithmetic wraps as the scalar code does.
#[inline]
#[target_feature(enable = "neon")]
fn lanes8(c: uint16x8_t, first: u16, sym: u16, rate: int16x8_t) -> uint16x8_t {
    let idx = vaddq_u16(load_u16x8(&[0, 1, 2, 3, 4, 5, 6, 7]), vdupq_n_u16(first));
    let up = vcgeq_u16(idx, vdupq_n_u16(sym));
    let down = vsubq_u16(c, vshlq_u16(c, rate));
    let gap = vsubq_u16(vdupq_n_u16(32768), c);
    vbslq_u16(up, vaddq_u16(c, vshlq_u16(gap, rate)), down)
}

/// `symbol::cur` of the lanes of `x`, whose first lane is value `first` of an `n`-value row:
/// `((r * ((32768 - min(x, 32768)) >> 6)) >> 1) + 4 * (n - 1 - i)`. The product is widened to
/// 32 bits; the result fits 16 bits.
#[inline]
#[target_feature(enable = "neon")]
fn cur(x: uint16x8_t, r: u16, first: usize, n: usize) -> uint16x8_t {
    let top = vdupq_n_u16(32768);
    let g = vshrq_n_u16::<6>(vsubq_u16(top, vminq_u16(x, top)));
    let rv = vdupq_n_u16(r);
    let lo = vmull_u16(vget_low_u16(g), vget_low_u16(rv));
    let hi = vmull_high_u16(g, rv);
    let half = vcombine_u16(vshrn_n_u32::<1>(lo), vshrn_n_u32::<1>(hi));
    let min_prob = vsubq_u16(
        vdupq_n_u16((4 * (n - 1 - first)) as u16),
        load_u16x8(&[0, 4, 8, 12, 16, 20, 24, 28]),
    );
    vaddq_u16(half, min_prob)
}

/// The smallest value index `first + i` over the `w` first lanes `i` of `cur` that are at
/// most `c`, or 16 if none.
#[inline]
#[target_feature(enable = "neon")]
fn first_ge(cur: uint16x8_t, c: u16, first: usize, w: usize) -> usize {
    let lane = load_u16x8(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let hit = vandq_u16(
        vcgeq_u16(vdupq_n_u16(c), cur),
        vcltq_u16(lane, vdupq_n_u16(w as u16)),
    );
    let idx = vaddq_u16(lane, vdupq_n_u16(first as u16));
    usize::from(vminvq_u16(vbslq_u16(hit, idx, vdupq_n_u16(16))))
}

/// Loads the first 8 values of `row`, or the first 4 (upper lanes zero) when it is shorter.
#[inline]
#[target_feature(enable = "neon")]
fn load(row: &[u16]) -> uint16x8_t {
    if let Some(a) = row.first_chunk::<8>() {
        load_u16x8(a)
    } else if let Some(a) = row.first_chunk::<4>() {
        vcombine_u16(load_u16x4(a), vdup_n_u16(0))
    } else {
        vdupq_n_u16(0)
    }
}

/// Stores the lanes of `v` to the first 8 values of `row`, or the first 4 when it is shorter.
#[inline]
#[target_feature(enable = "neon")]
fn store(row: &mut [u16], v: uint16x8_t) {
    if let Some(d) = row.first_chunk_mut::<8>() {
        store_u16x8(d, v);
    } else if let Some(d) = row.first_chunk_mut::<4>() {
        store_u16x4(d, vget_low_u16(v));
    }
}

/// Decodes the symbol of the 4 to 16 values of `cdf` for `SymbolValue` = `c` and
/// `SymbolRange` = `rng` and, with `Some(rate)`, adapts the values for it. Returns the symbol,
/// the `cur` before it (`rng` for symbol 0) and its own `cur`.
#[target_feature(enable = "neon")]
pub(super) fn decode(cdf: &mut [u16], c: u32, rng: u32, rate: Option<u32>) -> (usize, u32, u32) {
    let n = cdf.len();
    // Lanes per vector: the second vector starts at value n - w.
    let w = if n >= 8 { 8 } else { 4 };
    let r = (rng >> 8) as u16;
    let a = load(cdf);
    let b = load(&cdf[n - w..]);
    let cur_a = cur(a, r, 0, n);
    let cur_b = cur(b, r, n - w, n);
    let c16 = c as u16;
    // Value n - 1 always counts.
    let sym = first_ge(cur_a, c16, 0, w)
        .min(first_ge(cur_b, c16, n - w, w))
        .min(n - 1);
    // The `cur` of every value: the second store overwrites the upper lanes of the first one
    // that are not values (when w = 4) with the right ones.
    let mut curs = [0u16; 16];
    if let Some(d) = curs.first_chunk_mut::<8>() {
        store_u16x8(d, cur_a);
    }
    if let Some(d) = curs[n - w..].first_chunk_mut::<8>() {
        store_u16x8(d, cur_b);
    }
    let v = u32::from(curs[sym]);
    let prev = if sym == 0 {
        rng
    } else {
        u32::from(curs[sym - 1])
    };
    if let Some(rate) = rate {
        let rate = vdupq_n_s16(-(rate as i16));
        let a = lanes8(a, 0, sym as u16, rate);
        let b = lanes8(b, (n - w) as u16, sym as u16, rate);
        store(cdf, a);
        store(&mut cdf[n - w..], b);
    }
    (sym, prev, v)
}
