//! Kernel family `ipred`: intra prediction (spec 7.11.2), palette (7.11.4) and the alpha step of
//! chroma from luma (7.11.5). The scalar oracle is src/predict.rs.
//!
//! SIMD kernels: Paeth, Smooth / Smooth_V / Smooth_H, directional zone 1 without upsampling,
//! the intra edge filter, palette (rows of 8) and the CFL alpha step. On every SIMD tier V_PRED
//! and H_PRED (pAngle 90 / 180) are row copies and fills instead of the per-sample directional
//! loop, and the CFL luma average is computed once per sample instead of twice. The rest (DC,
//! directional zones 2 and 3, upsampled edges, filter intra) runs the scalar oracle on every
//! tier (docs/safety.md says why).

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use super::{Simd, Tier};
use crate::predict::{self, Edge, IntraParams, OFF};

/// Whether a SIMD kernel may write the `w x h` block at `(x, y)`: `w` is 4..=64 and a multiple
/// of 4, `h` is 1..=64 and every row is in `len` samples.
fn block_ok(len: usize, stride: usize, x: usize, y: usize, w: usize, h: usize) -> bool {
    w.is_multiple_of(4)
        && (4..=64).contains(&w)
        && (1..=64).contains(&h)
        && y.checked_add(h - 1)
            .and_then(|r| r.checked_mul(stride))
            .and_then(|r| r.checked_add(x))
            .and_then(|r| r.checked_add(w))
            .is_some_and(|end| end <= len)
}

/// Paeth (spec 7.11.2.2).
#[allow(unsafe_code)]
pub(crate) fn paeth(
    simd: Simd,
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    left: &Edge,
) {
    let ok = block_ok(px.len(), stride, x, y, w, h);
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if ok => {
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe { avx2::paeth(px, stride, x, y, w, h, above, left) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if ok => {
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe { neon::paeth(px, stride, x, y, w, h, above, left) }
        }
        _ => predict::predict_paeth(px, stride, x, y, w, h, above, left),
    }
}

/// SMOOTH_PRED, SMOOTH_V_PRED and SMOOTH_H_PRED (spec 7.11.2.6).
#[allow(unsafe_code)]
pub(crate) fn smooth(
    simd: Simd,
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    above: &Edge,
    left: &Edge,
    p: &IntraParams,
) {
    let w = 1usize << p.log2w;
    let h = 1usize << p.log2h;
    let ok = block_ok(px.len(), stride, x, y, w, h);
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if ok => {
            let (wx, wy) = (predict::sm_weights(p.log2w), predict::sm_weights(p.log2h));
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe { avx2::smooth(px, stride, x, y, w, h, p.mode, wx, wy, above, left) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if ok => {
            let (wx, wy) = (predict::sm_weights(p.log2w), predict::sm_weights(p.log2h));
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe { neon::smooth(px, stride, x, y, w, h, p.mode, wx, wy, above, left) }
        }
        _ => predict::predict_smooth(px, stride, x, y, above, left, p),
    }
}

/// The prediction loop of the directional predictor (spec 7.11.2.4, after the edge filter and
/// upsampling); the arguments are those of [`predict::directional_ref`].
#[allow(unsafe_code)]
pub(crate) fn directional(
    simd: Simd,
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    above: &Edge,
    left: &Edge,
    p_angle: i32,
    dx: i32,
    dy: i32,
    upsample_above: u32,
    upsample_left: u32,
) {
    let ok = simd.tier() != Tier::Scalar && block_ok(px.len(), stride, x, y, w, h);
    if ok && p_angle == 90 {
        let mut row = [0u16; 64];
        for (d, &a) in row.iter_mut().zip(&above[OFF..OFF + w]) {
            *d = a as u16;
        }
        for i in 0..h {
            px[(y + i) * stride + x..][..w].copy_from_slice(&row[..w]);
        }
        return;
    }
    if ok && p_angle == 180 {
        for i in 0..h {
            px[(y + i) * stride + x..][..w].fill(left[OFF + i] as u16);
        }
        return;
    }
    let z1 = ok && p_angle < 90 && upsample_above == 0;
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if z1 => {
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe { avx2::dr_z1(px, stride, x, y, w, h, above, dx) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if z1 => {
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe { neon::dr_z1(px, stride, x, y, w, h, above, dx) }
        }
        _ => predict::directional_ref(
            px,
            stride,
            x,
            y,
            w,
            h,
            above,
            left,
            p_angle,
            dx,
            dy,
            upsample_above,
            upsample_left,
        ),
    }
}

/// Length of the padded edge copy the edge filter kernels read: entries -2..=sz+11 of the edge.
const PAD: usize = 152;

/// The edge filter input `edge[ clamp( i - 2, 0, sz - 1 ) ]` at `p[ i ]` for every `i` in
/// `PAD` (spec 7.11.2.12 reads `edge[ i + j - 2 ]` clamped). `sz` is 2..=129.
fn pad_edge(buf: &Edge, sz: usize) -> [i32; PAD] {
    let mut p = [buf[OFF + sz - 2]; PAD];
    p[0] = buf[OFF - 1];
    p[1] = buf[OFF - 1];
    p[2..sz + 2].copy_from_slice(&buf[OFF - 1..OFF - 1 + sz]);
    p
}

/// The intra edge filter (spec 7.11.2.12) of [`predict::filter_edge`].
#[allow(unsafe_code)]
pub(crate) fn filter_edge(simd: Simd, buf: &mut Edge, sz: usize, strength: usize) {
    let ok = (1..=3).contains(&strength) && (2..=129).contains(&sz);
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if ok => {
            let p = pad_edge(buf, sz);
            let k = &predict::INTRA_EDGE_KERNEL[strength - 1];
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe { avx2::filter_edge(buf, &p, sz, k) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if ok => {
            let p = pad_edge(buf, sz);
            let k = &predict::INTRA_EDGE_KERNEL[strength - 1];
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe { neon::filter_edge(buf, &p, sz, k) }
        }
        _ => predict::filter_edge(buf, sz, strength),
    }
}

/// Palette prediction (spec 7.11.4) of [`predict::predict_palette`].
#[allow(unsafe_code)]
pub(crate) fn palette(
    simd: Simd,
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    palette: &[u16],
    map: &[u8],
    map_stride: usize,
) {
    let ok = block_ok(px.len(), stride, x, y, w, h)
        && palette.len() <= 16
        && (h - 1)
            .checked_mul(map_stride)
            .and_then(|r| r.checked_add(w))
            .is_some_and(|end| end <= map.len());
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if ok => {
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe { avx2::palette(px, stride, x, y, w, h, palette, map, map_stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if ok => {
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe { neon::palette(px, stride, x, y, w, h, palette, map, map_stride) }
        }
        _ => predict::palette_ref(px, stride, x, y, w, h, palette, map, map_stride),
    }
}

/// `L( i, j ) - lumaAvg` of spec 7.11.5 at `ac[ i * tx_w + j ]`, with the arithmetic of
/// [`predict::cfl_ref`]. `tx_w * tx_h` is at most 1024; only `ac[..tx_w * tx_h]` is written.
fn cfl_ac(
    luma: &[u16],
    l_stride: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
    sub_x: usize,
    sub_y: usize,
    max_luma_w: usize,
    max_luma_h: usize,
    ac: &mut [i32; 1024],
) {
    let mut sum = 0i32;
    for i in 0..tx_h {
        let luma_y = ((start_y + i) << sub_y).min(max_luma_h.saturating_sub(1 << sub_y));
        for j in 0..tx_w {
            let luma_x = ((start_x + j) << sub_x).min(max_luma_w.saturating_sub(1 << sub_x));
            let mut t = 0i32;
            for dy in 0..=sub_y {
                for dx in 0..=sub_x {
                    t += i32::from(luma[(luma_y + dy) * l_stride + luma_x + dx]);
                }
            }
            let v = t << (3 - sub_x - sub_y);
            ac[i * tx_w + j] = v;
            sum += v;
        }
    }
    let avg = predict::round2(sum, tx_w.trailing_zeros() + tx_h.trailing_zeros());
    for v in &mut ac[..tx_w * tx_h] {
        *v -= avg;
    }
}

/// Chroma from luma (spec 7.11.5) of [`predict::predict_cfl`].
#[allow(unsafe_code)]
pub(crate) fn cfl(
    simd: Simd,
    chroma: &mut [u16],
    c_stride: usize,
    luma: &[u16],
    l_stride: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    max_luma_w: usize,
    max_luma_h: usize,
    alpha: i32,
    bit_depth: u32,
    ac: &mut [i32; 1024],
) {
    let ok = block_ok(chroma.len(), c_stride, start_x, start_y, tx_w, tx_h)
        && tx_w <= 32
        && tx_h <= 32
        && tx_h.is_power_of_two()
        && tx_w.is_power_of_two();
    let (sub_x, sub_y) = (usize::from(subsampling_x), usize::from(subsampling_y));
    let max = (1i32 << bit_depth) - 1;
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if ok => {
            cfl_ac(
                luma, l_stride, start_x, start_y, tx_w, tx_h, sub_x, sub_y, max_luma_w, max_luma_h,
                ac,
            );
            // SAFETY: a Simd of tier Avx2 or Avx512 is made only by simd::available() / best()
            // after detecting AVX2 (Avx512 includes it).
            unsafe {
                avx2::cfl(
                    chroma, c_stride, start_x, start_y, tx_w, tx_h, ac, alpha, max,
                )
            }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if ok => {
            cfl_ac(
                luma, l_stride, start_x, start_y, tx_w, tx_h, sub_x, sub_y, max_luma_w, max_luma_h,
                ac,
            );
            // SAFETY: a Simd of tier Neon is made only by simd::available() / best() after
            // detecting NEON.
            unsafe {
                neon::cfl(
                    chroma, c_stride, start_x, start_y, tx_w, tx_h, ac, alpha, max,
                )
            }
        }
        _ => predict::cfl_ref(
            chroma,
            c_stride,
            luma,
            l_stride,
            start_x,
            start_y,
            tx_w,
            tx_h,
            subsampling_x,
            subsampling_y,
            max_luma_w,
            max_luma_h,
            alpha,
            bit_depth,
        ),
    }
}

/// Cross-tier checks shared by the unit tests and the fuzz target (`crate::fuzzing::ipred`).
/// `next` gives the random choices; each check builds a valid input from them, runs every tier
/// of `simd::available()` and asserts that the output equals the scalar output.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) mod check {
    use super::super::{Simd, available};
    use crate::predict::{self, EDGE_LEN, Edge, IntraEdges, IntraParams};

    /// A sample of `bit_depth` bits: `kind` 0 = all zero, 1 = all max, 2 = 0 or max, else any.
    fn sample(next: &mut dyn FnMut() -> u32, kind: u32, bit_depth: u32) -> u16 {
        let max = (1u32 << bit_depth) - 1;
        (match kind {
            0 => 0,
            1 => max,
            2 => max * (next() & 1),
            _ => next() & max,
        }) as u16
    }

    fn samples(next: &mut dyn FnMut() -> u32, n: usize, kind: u32, bit_depth: u32) -> Vec<u16> {
        let mut v = vec![0u16; n];
        for s in v.iter_mut() {
            *s = sample(next, kind, bit_depth);
        }
        v
    }

    fn bit_depth(next: &mut dyn FnMut() -> u32) -> u32 {
        [8, 10, 12][(next() % 3) as usize]
    }

    /// One `predict_intra` call on every tier, bit-exact with scalar over the whole plane, with
    /// a zeroed and with a garbage edge scratch (the edge build must write every entry the mode
    /// reads, as the decoder does not clear the scratch between blocks).
    pub(crate) fn intra_case(
        px: &[u16],
        stride: usize,
        x: usize,
        y: usize,
        e: IntraEdges,
        p: &IntraParams,
    ) {
        let mut want = px.to_vec();
        let mut edges: [Edge; 2] = [[0; EDGE_LEN]; 2];
        predict::predict_intra_simd(Simd::scalar(), &mut want, stride, x, y, e, p, &mut edges);
        for simd in available() {
            for fill in [0, -77_777] {
                let mut got = px.to_vec();
                let mut edges: [Edge; 2] = [[fill; EDGE_LEN]; 2];
                predict::predict_intra_simd(simd, &mut got, stride, x, y, e, p, &mut edges);
                assert!(
                    got == want,
                    "{:?} fill {fill} x {x} y {y} {e:?} {p:?}",
                    simd.tier()
                );
            }
        }
    }

    /// A random intra block (any mode, angle, size 4x4..64x64 up to 4:1, filter intra, edge
    /// filter, availability, frame edge, bit depth) in a 128 x 128 plane.
    pub(crate) fn intra(next: &mut dyn FnMut() -> u32) {
        const S: usize = 128;
        let bd = bit_depth(next);
        let log2w = 2 + (next() % 5) as usize;
        let log2h = (log2w as i32 + (next() % 5) as i32 - 2).clamp(2, 6) as usize;
        let x = (next() % 32) as usize;
        let y = (next() % 32) as usize;
        let kind = next() % 6;
        let px = samples(next, S * S, kind, bd);
        let e = IntraEdges {
            have_left: x > 0 && next() & 1 == 1,
            have_above: y > 0 && next() & 1 == 1,
            have_above_right: next() & 1 == 1,
            have_below_left: next() & 1 == 1,
        };
        let p = IntraParams {
            mode: (next() % 13) as usize,
            angle_delta: (next() % 7) as i32 - 3,
            log2w,
            log2h,
            use_filter_intra: next().is_multiple_of(4),
            filter_intra_mode: (next() % 5) as usize,
            enable_intra_edge_filter: next() & 3 != 0,
            smooth_neighbor: next() & 1 == 1,
            max_x: x + (next() as usize % (S - x)),
            max_y: y + (next() as usize % (S - y)),
            bit_depth: bd,
        };
        intra_case(&px, S, x, y, e, &p);
    }

    /// A random palette block (2..=8 colours, 4..=64 wide and high) in an 80 x 80 plane.
    pub(crate) fn palette(next: &mut dyn FnMut() -> u32) {
        const S: usize = 80;
        let bd = bit_depth(next);
        let w = 4usize << (next() % 5);
        let h = 4usize << (next() % 5);
        let x = (next() % 16) as usize;
        let y = (next() % 16) as usize;
        let n = 2 + (next() % 7) as usize;
        let kind = next() % 6;
        let colours = samples(next, n, kind, bd);
        let map_stride = w + (next() % 8) as usize;
        let mut map = vec![0u8; map_stride * h];
        for m in map.iter_mut() {
            *m = (next() as usize % n) as u8;
        }
        let px = samples(next, S * S, 3, bd);
        let mut want = px.clone();
        predict::palette_ref(&mut want, S, x, y, w, h, &colours, &map, map_stride);
        for simd in available() {
            let mut got = px.clone();
            super::palette(simd, &mut got, S, x, y, w, h, &colours, &map, map_stride);
            assert!(got == want, "{:?} {w}x{h} at {x},{y}", simd.tier());
        }
    }

    /// A random CFL block (4..=32 wide and high, 4:2:0 / 4:2:2 / 4:4:0 / 4:4:4, the luma edge
    /// clamps, alpha -16..=16) on a 64 x 64 chroma and a 100 x 100 luma plane.
    pub(crate) fn cfl(next: &mut dyn FnMut() -> u32) {
        const CS: usize = 64;
        const LS: usize = 100;
        let bd = bit_depth(next);
        let tx_w = 4usize << (next() % 4);
        let tx_h = 4usize << (next() % 4);
        let (sub_x, sub_y) = (next() & 1 == 1, next() & 1 == 1);
        let x = (next() % 16) as usize;
        let y = (next() % 16) as usize;
        let max_luma_w = next() as usize % 97;
        let max_luma_h = next() as usize % 97;
        let alpha = (next() % 33) as i32 - 16;
        let kind = next() % 6;
        let luma = samples(next, LS * LS, kind, bd);
        let chroma = samples(next, CS * CS, 3, bd);
        let mut want = chroma.clone();
        predict::cfl_ref(
            &mut want, CS, &luma, LS, x, y, tx_w, tx_h, sub_x, sub_y, max_luma_w, max_luma_h,
            alpha, bd,
        );
        for simd in available() {
            let mut got = chroma.clone();
            // Garbage: `cfl_ac` must write every entry the kernels read.
            let mut ac = [-77_777i32; 1024];
            super::cfl(
                simd, &mut got, CS, &luma, LS, x, y, tx_w, tx_h, sub_x, sub_y, max_luma_w,
                max_luma_h, alpha, bd, &mut ac,
            );
            assert!(
                got == want,
                "{:?} {tx_w}x{tx_h} at {x},{y} alpha {alpha}",
                simd.tier()
            );
        }
    }

    /// The edge filter on one edge of `sz` entries with `strength`, every tier against scalar.
    pub(crate) fn filter_edge_case(buf: &Edge, sz: usize, strength: usize) {
        let mut want = *buf;
        predict::filter_edge(&mut want, sz, strength);
        for simd in available() {
            let mut got = *buf;
            super::filter_edge(simd, &mut got, sz, strength);
            assert!(got == want, "{:?} sz {sz} strength {strength}", simd.tier());
        }
    }

    /// A random edge filter call (sz 0..=129, strength 0..=3).
    pub(crate) fn filter_edge(next: &mut dyn FnMut() -> u32) {
        let bd = bit_depth(next);
        let kind = next() % 6;
        let mut buf: Edge = [0; EDGE_LEN];
        for v in buf.iter_mut() {
            *v = i32::from(sample(next, kind, bd));
        }
        filter_edge_case(&buf, (next() % 130) as usize, (next() % 4) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::check;
    use crate::predict::{EDGE_LEN, Edge, IntraEdges, IntraParams};

    /// xorshift64, high half.
    fn rng(seed: u64) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u32
        }
    }

    #[test]
    fn intra_random() {
        let mut next = rng(0x1234_5678_9abc_def1);
        for _ in 0..600 {
            check::intra(&mut next);
        }
    }

    /// Every mode, angle delta and block size (4:1 and 1:4 included), with and without the edge
    /// filter and filter intra, at an unaligned position, for 8, 10 and 12 bits.
    #[test]
    fn intra_every_mode_and_size() {
        const S: usize = 128;
        let mut next = rng(0x0bad_cafe_f00d_0001);
        for bd in [8u32, 10, 12] {
            let max = (1u32 << bd) - 1;
            let px: Vec<u16> = (0..S * S).map(|_| (next() & max) as u16).collect();
            let e = IntraEdges {
                have_left: true,
                have_above: true,
                have_above_right: bd != 10,
                have_below_left: bd != 12,
            };
            for mode in 0..13 {
                for delta in -3..=3 {
                    // Every delta at 10 bits; three of them at 8 and 12 (qemu time).
                    if bd != 10 && ![-3, 0, 2].contains(&delta) {
                        continue;
                    }
                    for log2w in 2usize..=6 {
                        for log2h in 2usize..=6 {
                            if log2w.abs_diff(log2h) > 2 {
                                continue;
                            }
                            for filter in [false, true] {
                                let p = IntraParams {
                                    mode,
                                    angle_delta: delta,
                                    log2w,
                                    log2h,
                                    use_filter_intra: filter && delta == 0,
                                    filter_intra_mode: mode % 5,
                                    enable_intra_edge_filter: filter,
                                    smooth_neighbor: (mode + bd as usize).is_multiple_of(2),
                                    max_x: if delta == 1 {
                                        27 + (1 << log2w) / 2
                                    } else {
                                        S - 1
                                    },
                                    max_y: if delta == -1 {
                                        21 + (1 << log2h) / 2
                                    } else {
                                        S - 1
                                    },
                                    bit_depth: bd,
                                };
                                check::intra_case(&px, S, 27, 21, e, &p);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn palette_random() {
        let mut next = rng(0x5555_aaaa_1234_0002);
        for _ in 0..400 {
            check::palette(&mut next);
        }
    }

    #[test]
    fn cfl_random() {
        let mut next = rng(0x7777_1111_cccc_0003);
        for _ in 0..400 {
            check::cfl(&mut next);
        }
    }

    /// Every size 0..=129 (lane - 1, lane, lane + 1 of both 4 and 8 lanes) and strength, with
    /// random, zero and maximum samples.
    #[test]
    fn filter_edge_every_size() {
        let mut next = rng(0x3141_5926_5358_0004);
        for max in [255i32, 1023, 4095] {
            let mut buf: Edge = [0; EDGE_LEN];
            for v in buf.iter_mut() {
                *v = (next() as i32) & max;
            }
            for sz in 0..=129 {
                for strength in 0..=3 {
                    check::filter_edge_case(&buf, sz, strength);
                    check::filter_edge_case(&[max; EDGE_LEN], sz, strength);
                    check::filter_edge_case(&[0; EDGE_LEN], sz, strength);
                }
            }
        }
        for _ in 0..300 {
            check::filter_edge(&mut next);
        }
    }
}
