//! aarch64 NEON load/store helpers for the SIMD kernels (docs/safety.md). The only `unsafe`
//! here is the load or store itself: the array reference proves that every byte it touches is
//! in bounds. Kernels get the arrays from slices in safe code.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

macro_rules! io {
    ($load:ident, $store:ident, $t:ty, $n:literal, $v:ty, $ld:ident, $st:ident) => {
        #[inline]
        #[target_feature(enable = "neon")]
        pub(crate) fn $load(a: &[$t; $n]) -> $v {
            // SAFETY: `a` is exactly the bytes of one vector (the array type).
            unsafe { $ld(a.as_ptr()) }
        }
        #[inline]
        #[target_feature(enable = "neon")]
        pub(crate) fn $store(a: &mut [$t; $n], v: $v) {
            // SAFETY: `a` is exactly the bytes of one vector (the array type).
            unsafe { $st(a.as_mut_ptr(), v) }
        }
    };
}

io!(
    load_u8x16,
    store_u8x16,
    u8,
    16,
    uint8x16_t,
    vld1q_u8,
    vst1q_u8
);
io!(load_u8x8, store_u8x8, u8, 8, uint8x8_t, vld1_u8, vst1_u8);
io!(
    load_u16x8,
    store_u16x8,
    u16,
    8,
    uint16x8_t,
    vld1q_u16,
    vst1q_u16
);
io!(
    load_u16x4,
    store_u16x4,
    u16,
    4,
    uint16x4_t,
    vld1_u16,
    vst1_u16
);
io!(
    load_i16x8,
    store_i16x8,
    i16,
    8,
    int16x8_t,
    vld1q_s16,
    vst1q_s16
);
io!(
    load_i16x4,
    store_i16x4,
    i16,
    4,
    int16x4_t,
    vld1_s16,
    vst1_s16
);
io!(
    load_i32x4,
    store_i32x4,
    i32,
    4,
    int32x4_t,
    vld1q_s32,
    vst1q_s32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let src: [u16; 8] = std::array::from_fn(|i| i as u16 * 1000 + 7);
        let mut dst = [0u16; 8];
        // SAFETY: test only; NEON is mandatory on aarch64.
        unsafe { rt(&src, &mut dst) };
        assert_eq!(src, dst);
    }

    #[target_feature(enable = "neon")]
    fn rt(src: &[u16; 8], dst: &mut [u16; 8]) {
        store_u16x8(dst, load_u16x8(src));
    }
}
