//! Cross-tier checks for the fuzz targets in `tests/fuzz` (feature `fuzzing`, not for users).
//!
//! Each kernel family adds one `pub fn <family>(data: &[u8])`: it builds valid kernel inputs
//! from the bytes (any bytes give some input), runs the kernel on every tier of
//! `simd::available()` and asserts that each output equals the scalar output.

/// The number of SIMD tiers this CPU has, scalar included (a fuzz target can report it).
pub fn tiers() -> usize {
    crate::simd::available().len()
}

/// Loop filter (`simd::lf`): one edge run (bit depth, plane, filter size, direction, length,
/// level, sharpness, start and sample pattern from the bytes) on every tier against scalar.
pub fn lf(data: &[u8]) {
    crate::simd::lf::check(data);
}

/// The next input byte, 0 past the end.
fn byte(it: &mut impl Iterator<Item = u8>) -> u8 {
    it.next().unwrap_or(0)
}

/// The next two input bytes, little endian.
fn word(it: &mut impl Iterator<Item = u8>) -> u16 {
    u16::from_le_bytes([byte(it), byte(it)])
}

/// Family `grain` (`simd::grain`): one luma and one chroma row of the film grain blend from the
/// bytes (bit depth, clip range, scaling shift, chroma merge, scaling table, samples in range,
/// noise in the grain range), every tier against scalar.
pub fn grain(data: &[u8]) {
    use crate::grain::{Blend, scale_table};
    use crate::simd::grain::{blend_chroma, blend_luma};
    let mut it = data.iter().copied();
    let bit_depth = [8u32, 10, 12][usize::from(byte(&mut it) % 3)];
    let sh = bit_depth - 8;
    let pixel_max = (1i32 << bit_depth) - 1;
    let (min, max) = match byte(&mut it) % 3 {
        0 => (16 << sh, 235 << sh),
        1 => (16 << sh, 240 << sh),
        _ => (0, pixel_max),
    };
    let flags = byte(&mut it);
    let b = Blend {
        bit_depth,
        pixel_max,
        scaling_shift: 8 + u32::from(flags & 3),
        min,
        max,
        sx: usize::from((flags >> 2) & 1),
        cfl: (flags >> 3) & 1 == 1,
        luma_mult: i32::from(byte(&mut it)) - 128,
        mult: i32::from(byte(&mut it)) - 128,
        offset: (((i32::from(byte(&mut it)) << 1) | i32::from((flags >> 4) & 1)) - 256) << sh,
    };
    let n = usize::from(byte(&mut it) % 80);
    // With subsampling the luma row is 2n or 2n - 1 samples (odd widths read one twice).
    let luma_len = (n << b.sx).saturating_sub(b.sx * usize::from((flags >> 5) & 1));
    let mut lut = [0i32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        // A few bytes spread over the table: the fuzzer reaches the samples sooner.
        *v = i32::from(byte(&mut it).wrapping_add((i as u8).wrapping_mul(flags | 1)));
    }
    let table = scale_table(&lut, bit_depth);
    let mask = pixel_max as u16;
    let src: Vec<u16> = (0..n).map(|_| word(&mut it) & mask).collect();
    let luma: Vec<u16> = (0..luma_len).map(|_| word(&mut it) & mask).collect();
    let noise: Vec<i32> = (0..n)
        .map(|_| i32::from(word(&mut it) & mask) - (128 << sh))
        .collect();

    let scalar = crate::simd::Simd::scalar();
    let mut want_y = vec![0u16; n];
    blend_luma(scalar, &mut want_y, &src, &noise, &lut, &table, &b);
    let mut want_c = vec![0u16; n];
    blend_chroma(scalar, &mut want_c, &src, &luma, &noise, &lut, &table, &b);
    for simd in crate::simd::available() {
        let mut got = vec![0u16; n];
        blend_luma(simd, &mut got, &src, &noise, &lut, &table, &b);
        assert_eq!(got, want_y, "luma {simd:?} {b:?}");
        let mut got = vec![0u16; n];
        blend_chroma(simd, &mut got, &src, &luma, &noise, &lut, &table, &b);
        assert_eq!(got, want_c, "chroma {simd:?} {b:?}");
    }
}

/// Loop restoration (`simd::lr`): one block from the bytes (bit depth, size, column, Wiener taps,
/// self guided set and weights, then the window samples, cycled), both filters on every tier
/// against scalar.
pub fn lr(data: &[u8]) {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0);
    let bd = [8, 10, 12][usize::from(byte(0) % 3)];
    let w = usize::from(byte(1) % 64) + 1;
    let h = usize::from(byte(2) % 64) + 1;
    let x = usize::from(byte(3) % 16);
    let set = usize::from(byte(4) % 16);
    let coef = [
        [byte(5) as i8, byte(6) as i8, byte(7) as i8],
        [byte(8) as i8, byte(9) as i8, byte(10) as i8],
    ];
    let xqd = [byte(11) as i8, byte(12) as i8];
    // Bit 0 set: every sample is 0 or the maximum (from bit 0 of its bytes).
    let extremes = byte(13) & 1 == 1;
    let rest = data.get(14..).unwrap_or(&[]);
    let max = (1i32 << bd) - 1;
    let win: Vec<i32> = (0..(w + 6) * (h + 6))
        .map(|k| {
            let v = match rest.len() {
                0 => 0,
                n => i32::from(u16::from_le_bytes([rest[2 * k % n], rest[(2 * k + 1) % n]])),
            };
            match (extremes, v & 1) {
                (true, 1) => max,
                (true, _) => 0,
                _ => v & max,
            }
        })
        .collect();
    crate::simd::lr::cross_check(&win, w, h, bd, &coef, set, &xqd, x);
}
