//! NEON `itx` kernels: 4 `i32` lanes (see `itx_kernels!` in mod.rs).

use crate::simd::arm::{load_i32x4, load_u16x4, store_i32x4, store_u16x4};
use std::arch::aarch64::*;

type V = int32x4_t;
const L: usize = 4;

#[inline]
#[target_feature(enable = "neon")]
fn vsplat(x: i32) -> V {
    vdupq_n_s32(x)
}
#[inline]
#[target_feature(enable = "neon")]
fn vadd(a: V, b: V) -> V {
    vaddq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn vsub(a: V, b: V) -> V {
    vsubq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn vmul(a: V, b: V) -> V {
    vmulq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn vand(a: V, b: V) -> V {
    vandq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn vmin(a: V, b: V) -> V {
    vminq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn vmax(a: V, b: V) -> V {
    vmaxq_s32(a, b)
}
/// A negative count makes `vshlq_s32` an arithmetic (truncating) shift right.
#[inline]
#[target_feature(enable = "neon")]
fn vsra(a: V, n: u32) -> V {
    vshlq_s32(a, vdupq_n_s32(-(n as i32)))
}
#[inline]
#[target_feature(enable = "neon")]
fn vload(a: &[i32; L]) -> V {
    load_i32x4(a)
}
#[inline]
#[target_feature(enable = "neon")]
fn vstore(a: &mut [i32; L], v: V) {
    store_i32x4(a, v)
}
#[inline]
#[target_feature(enable = "neon")]
fn px_load(a: &[u16; L]) -> V {
    vreinterpretq_s32_u32(vmovl_u16(load_u16x4(a)))
}
/// `v` holds values in `0..=65535`, so the truncating narrow is exact.
#[inline]
#[target_feature(enable = "neon")]
fn px_store(a: &mut [u16; L], v: V) {
    store_u16x4(a, vmovn_u32(vreinterpretq_u32_s32(v)))
}

itx_kernels!("neon");
