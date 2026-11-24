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
/// CDEF (`src/simd/cdef`): the direction search and the filter of one block, every tier against
/// scalar. Bytes 0..8: bit depth, plane, subsampling, mi grid, block position, strengths,
/// damping, direction, stride padding; the rest: the samples, two bytes each, repeated.
pub fn cdef(data: &[u8]) {
    use crate::cdef::{FilterCtx, cdef_direction, cdef_filter};
    use crate::frame::{Plane, View};
    use crate::ledger::Ledger;
    use crate::simd::{available, cdef as k};
    use std::sync::Arc;

    let Some((head, samples)) = data.split_first_chunk::<8>() else {
        return;
    };
    let n = samples.len();
    if n == 0 {
        return;
    }
    let bd = [8u32, 10, 12, 8][usize::from(head[0] & 3)];
    let cs = bd - 8;
    let plane = usize::from((head[0] >> 2) & 3) % 3;
    let sub = [(0, 0), (1, 1), (1, 0), (0, 1)][usize::from((head[0] >> 4) & 3)];
    let mi_rows = 2 + 2 * usize::from(head[1] & 3);
    let mi_cols = 2 + usize::from((head[1] >> 2) & 7);
    let r = (usize::from(head[2] & 15) % mi_rows) & !1;
    let c = (usize::from(head[2] >> 4) % mi_cols) & !1;
    let pri = match head[3] >> 6 {
        0 => 0,
        // The largest luma strength after the variance adjustment.
        1 => ((15 << cs) * 16 + 8) >> 4,
        _ => i32::from(head[3] & 15) << cs,
    };
    let sec = [0, 1, 2, 4][usize::from(head[4] & 3)] << cs;
    let damping = 3 + i32::from((head[4] >> 2) & 3) + cs as i32 - i32::from(plane > 0);
    let dir = usize::from(head[5] & 7);
    let pad = usize::from(head[5] >> 3);
    let (sx, sy) = if plane == 0 { (0, 0) } else { sub };
    let (w, h) = ((mi_cols * 4) >> sx, (mi_rows * 4) >> sy);
    let ledger = Arc::new(Ledger::default());
    if ledger.reset(1 << 20, 0).is_err() {
        return;
    }
    let mk = || -> Option<Plane> {
        let mut p = Plane::new(w + pad, h, &ledger).ok()?;
        p.width = w;
        Some(p)
    };
    let Some(mut src) = mk() else {
        return;
    };
    let max = (1u16 << bd) - 1;
    for (i, v) in src.px.iter_mut().enumerate() {
        *v = u16::from_le_bytes([samples[(2 * i) % n], samples[(2 * i + 1) % n]]) & max;
    }

    let want = cdef_direction(&src, r, c, bd);
    for s in available() {
        assert_eq!(k::direction(s, &src, r, c, bd), want, "{s:?}");
    }

    let ctx = FilterCtx {
        mi_rows,
        mi_cols,
        coeff_shift: cs,
    };
    let (Some(mut want), Some(mut got)) = (mk(), mk()) else {
        return;
    };
    want.px.copy_from_slice(&src.px);
    cdef_filter(
        &ctx,
        &src,
        &mut View::of(&mut want),
        plane,
        sub,
        r,
        c,
        (pri, sec),
        damping,
        dir,
    );
    for s in available() {
        got.px.copy_from_slice(&src.px);
        k::filter(
            s,
            &ctx,
            &src,
            &mut View::of(&mut got),
            plane,
            sub,
            r,
            c,
            (pri, sec),
            damping,
            dir,
        );
        assert_eq!(got.px, want.px, "{s:?}");
    }
}

/// Kernel family `ipred` (src/simd/ipred): the first byte picks the kernel (intra prediction,
/// palette, CFL, edge filter), the next bytes (two per choice, then a xorshift seeded by the
/// input length) give its input. Every tier must equal scalar.
pub fn ipred(data: &[u8]) {
    use crate::simd::ipred::check;
    let Some((&which, rest)) = data.split_first() else {
        return;
    };
    let mut bytes = rest.iter();
    let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ rest.len() as u64;
    let mut next = move || -> u32 {
        match (bytes.next(), bytes.next()) {
            (Some(&a), Some(&b)) => u32::from(a) | (u32::from(b) << 8),
            _ => {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 32) as u32
            }
        }
    };
    match which % 4 {
        0 => check::intra(&mut next),
        1 => check::palette(&mut next),
        2 => check::cfl(&mut next),
        _ => check::filter_edge(&mut next),
    }
}

/// Kernel family `itx`: one block (size, type, bit depth, coded region, coefficients within
/// the dequantizer clamp, samples) built from the bytes, every tier against scalar. Bytes past
/// the end read as zero.
pub fn itx(data: &[u8]) {
    let mut bytes = data.iter().copied();
    let mut next = move || {
        let mut v = 0u32;
        for _ in 0..4 {
            v = (v << 8) | u32::from(bytes.next().unwrap_or(0));
        }
        v
    };
    crate::simd::itx::check_random(&mut next);
}

/// Family `threads` (no SIMD tiers: it compares thread counts). Builds a frame whose tiles are
/// done from `data` (format, bit depth, superres, size, filter parameters, mode info and
/// samples) and runs the threaded wrapup (loop filter, CDEF, superres, loop restoration) with
/// 2 and 3 threads: the samples or the error must equal those of 1 thread.
pub fn threads(data: &[u8]) {
    crate::decoder::check_threads(data, &[2, 3]);
}

/// `simd_entropy`: the symbol reader against the spec's bit by bit reader
/// (`symbol::oracle`), with the symbol search and CDF adaptation of every tier, and the search
/// and adaptation of every tier against scalar on rows of any values. The first 8 bytes seed
/// the reads and the CDFs (bit 0: CDF updates disabled); the rest is the tile data and the
/// rows.
pub fn entropy(data: &[u8]) {
    use crate::symbol::{XorShift, adapt_cdf, check_against_oracle, search};
    // The codec seeds are large; 4 KiB of tile data is plenty and keeps each run short.
    let data = &data[..data.len().min(8 + 4096)];
    let (seed, stream) = data.split_at(data.len().min(8));
    let mut s = [0u8; 8];
    s[..seed.len()].copy_from_slice(seed);
    let seed = u64::from_le_bytes(s);
    for &simd in &crate::simd::available() {
        check_against_oracle(stream, seed & 1 != 0, seed, 64 + 16 * stream.len(), simd);
    }
    let values: Vec<u16> = stream
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect();
    let mut rng = XorShift(seed);
    for len in 3..=17 {
        let Some(row) = values.get(..len) else {
            break;
        };
        for _ in 0..len {
            let c = rng.next_u32() >> 16;
            let range = rng.next_u32() >> 16;
            let update = rng.next_u32() & 1 != 0;
            let mut want = row.to_vec();
            let want_out = search(&want, c, range);
            if update {
                adapt_cdf(&mut want, want_out.0);
            }
            for &simd in &crate::simd::available() {
                let mut got = row.to_vec();
                let got_out = crate::simd::entropy::decode_symbol(simd, &mut got, c, range, update);
                assert_eq!(got_out, want_out, "{:?} len {len}", simd.tier());
                assert_eq!(got, want, "{:?} len {len}", simd.tier());
            }
        }
    }
}

/// Family `alloc` (buffer reuse; no SIMD tiers, so the oracle is a decoder that reuses
/// nothing). Decodes the temporal units of `data` (Section 5) with a decoder that reuses its
/// frame state and gets every output back through [`Decoder::recycle`](crate::Decoder::recycle),
/// and with one that keeps no frame state between frames; every unit must give the same frames
/// or the same error.
pub fn alloc(data: &[u8]) {
    use crate::{Decoder, Format, Limits, Options};
    use std::sync::atomic::AtomicBool;
    let limits = Limits {
        max_alloc: 256 << 20,
        max_threads: 1,
        max_width: 1024,
        max_height: 1024,
        ..Limits::default()
    };
    let Ok(units) = crate::split_temporal_units(data, Format::Section5, &limits) else {
        return;
    };
    // Work budget, as in the `decode` target: tile data past the end reads as zeros, so a few
    // bytes can decode whole frames.
    let Ok(p) = crate::probe(data, Format::Section5, &limits) else {
        return;
    };
    let work: u64 = p
        .frames
        .iter()
        .map(|f| u64::from(f.width) * u64::from(f.height))
        .sum();
    if work > 16 << 20 {
        return;
    }
    let cancel = AtomicBool::new(false);
    let mut reuse = Decoder::new(Options::default(), limits);
    let mut fresh = Decoder::new(Options::default(), limits);
    fresh.reuse = false;
    for u in units {
        let a = reuse.decode_temporal_unit(u, Format::Section5, &cancel);
        let b = fresh.decode_temporal_unit(u, Format::Section5, &cancel);
        assert_eq!(a, b);
        assert_eq!(reuse.take_frames(), fresh.take_frames());
        match a {
            Ok(frames) => reuse.recycle(frames),
            Err(_) => break,
        }
    }
}
