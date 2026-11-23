//! x86_64 load/store helpers for the SIMD kernels (docs/safety.md). The only `unsafe` here is
//! the unaligned load or store itself: the array reference proves that every byte it touches
//! is in bounds, and the `#[target_feature]` attribute that the caller (a kernel) must also
//! carry proves the instruction exists. Kernels get the arrays from slices in safe code
//! (`chunks_exact`, `try_into`, `first_chunk`).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// `load128_<t>(&[T; N]) -> __m128i` and `store128_<t>(&mut [T; N], __m128i)` for a 16-byte
/// array, and the same for 32 bytes (256) and 64 bytes (512).
macro_rules! io {
    ($load:ident, $store:ident, $t:ty, $n:literal, $v:ty, $feat:literal, $ld:ident, $st:ident) => {
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) fn $load(a: &[$t; $n]) -> $v {
            // SAFETY: `a` is `size_of::<$v>()` readable bytes (the array type); the load is
            // unaligned; the target feature is enabled on this fn.
            unsafe { $ld(a.as_ptr().cast()) }
        }
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) fn $store(a: &mut [$t; $n], v: $v) {
            // SAFETY: `a` is `size_of::<$v>()` writable bytes (the array type); the store is
            // unaligned; the target feature is enabled on this fn.
            unsafe { $st(a.as_mut_ptr().cast(), v) }
        }
    };
}

io!(
    load128_u8,
    store128_u8,
    u8,
    16,
    __m128i,
    "sse2",
    _mm_loadu_si128,
    _mm_storeu_si128
);
io!(
    load128_u16,
    store128_u16,
    u16,
    8,
    __m128i,
    "sse2",
    _mm_loadu_si128,
    _mm_storeu_si128
);
io!(
    load128_i16,
    store128_i16,
    i16,
    8,
    __m128i,
    "sse2",
    _mm_loadu_si128,
    _mm_storeu_si128
);
io!(
    load128_i32,
    store128_i32,
    i32,
    4,
    __m128i,
    "sse2",
    _mm_loadu_si128,
    _mm_storeu_si128
);
io!(
    load256_u8,
    store256_u8,
    u8,
    32,
    __m256i,
    "avx",
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
io!(
    load256_u16,
    store256_u16,
    u16,
    16,
    __m256i,
    "avx",
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
io!(
    load256_i16,
    store256_i16,
    i16,
    16,
    __m256i,
    "avx",
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
io!(
    load256_i32,
    store256_i32,
    i32,
    8,
    __m256i,
    "avx",
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
io!(
    load512_u8,
    store512_u8,
    u8,
    64,
    __m512i,
    "avx512f",
    _mm512_loadu_si512,
    _mm512_storeu_si512
);
io!(
    load512_u16,
    store512_u16,
    u16,
    32,
    __m512i,
    "avx512f",
    _mm512_loadu_si512,
    _mm512_storeu_si512
);
io!(
    load512_i16,
    store512_i16,
    i16,
    32,
    __m512i,
    "avx512f",
    _mm512_loadu_si512,
    _mm512_storeu_si512
);
io!(
    load512_i32,
    store512_i32,
    i32,
    16,
    __m512i,
    "avx512f",
    _mm512_loadu_si512,
    _mm512_storeu_si512
);

/// The low 8 bytes: `load64_u8(&[u8; 8])` (upper lane zero), `store64_u8`.
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn load64_u8(a: &[u8; 8]) -> __m128i {
    // SAFETY: `a` is 8 readable bytes; `_mm_loadl_epi64` reads exactly 8, unaligned.
    unsafe { _mm_loadl_epi64(a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn store64_u8(a: &mut [u8; 8], v: __m128i) {
    // SAFETY: `a` is 8 writable bytes; `_mm_storel_epi64` writes exactly 8, unaligned.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn load64_u16(a: &[u16; 4]) -> __m128i {
    // SAFETY: `a` is 8 readable bytes; `_mm_loadl_epi64` reads exactly 8, unaligned.
    unsafe { _mm_loadl_epi64(a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn store64_u16(a: &mut [u16; 4], v: __m128i) {
    // SAFETY: `a` is 8 writable bytes; `_mm_storel_epi64` writes exactly 8, unaligned.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_every_width() {
        let src: [u8; 64] = std::array::from_fn(|i| i as u8 * 3 + 1);
        let mut dst = [0u8; 64];
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: test only; avx512f (and so sse2/avx) was just detected.
            unsafe { rt512(&src, &mut dst) };
            assert_eq!(src, dst);
        }
    }

    #[target_feature(enable = "avx512f,avx")]
    fn rt512(src: &[u8; 64], dst: &mut [u8; 64]) {
        let v = load512_u8(src);
        store512_u8(dst, v);
        let mut d32 = [0u8; 32];
        store256_u8(
            &mut d32,
            load256_u8(src[..32].try_into().unwrap_or(&[0; 32])),
        );
        assert_eq!(&d32[..], &src[..32]);
        let mut d8 = [0u8; 8];
        store64_u8(&mut d8, load64_u8(src[..8].try_into().unwrap_or(&[0; 8])));
        assert_eq!(&d8[..], &src[..8]);
    }
}
