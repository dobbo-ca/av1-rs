//! NEON CDF adaptation (see `super::adapt_cdf`).

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

/// [`lanes8`] on 4 lanes.
#[inline]
#[target_feature(enable = "neon")]
fn lanes4(c: uint16x4_t, first: u16, sym: u16, rate: int16x4_t) -> uint16x4_t {
    let idx = vadd_u16(load_u16x4(&[0, 1, 2, 3]), vdup_n_u16(first));
    let up = vcge_u16(idx, vdup_n_u16(sym));
    let down = vsub_u16(c, vshl_u16(c, rate));
    let gap = vsub_u16(vdup_n_u16(32768), c);
    vbsl_u16(up, vadd_u16(c, vshl_u16(gap, rate)), down)
}

/// Adapts the 4 to 16 values of `cdf` for `sym` (below the length) at `rate`.
#[target_feature(enable = "neon")]
pub(super) fn adapt(cdf: &mut [u16], sym: usize, rate: u32) {
    let n = cdf.len();
    let sym = sym as u16;
    let shift = -(rate as i16);
    if n >= 8 {
        let (Some(a), Some(b)) = (cdf.first_chunk::<8>(), cdf.last_chunk::<8>()) else {
            return;
        };
        let rate = vdupq_n_s16(shift);
        let a = lanes8(load_u16x8(a), 0, sym, rate);
        let b = lanes8(load_u16x8(b), (n - 8) as u16, sym, rate);
        if let Some(d) = cdf.first_chunk_mut::<8>() {
            store_u16x8(d, a);
        }
        if let Some(d) = cdf.last_chunk_mut::<8>() {
            store_u16x8(d, b);
        }
    } else {
        let (Some(a), Some(b)) = (cdf.first_chunk::<4>(), cdf.last_chunk::<4>()) else {
            return;
        };
        let rate = vdup_n_s16(shift);
        let a = lanes4(load_u16x4(a), 0, sym, rate);
        let b = lanes4(load_u16x4(b), (n - 4) as u16, sym, rate);
        if let Some(d) = cdf.first_chunk_mut::<4>() {
            store_u16x4(d, a);
        }
        if let Some(d) = cdf.last_chunk_mut::<4>() {
            store_u16x4(d, b);
        }
    }
}
