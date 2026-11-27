//! NEON loop filter kernel: 8 samples along the edge per vector, one 16-bit lane each.

use super::{Edge, ROWS};
use crate::simd::arm::{load_u16x8, store_u16x8};
use std::arch::aarch64::*;

/// All ones in the lanes where `|a - b| > t`.
#[inline]
#[target_feature(enable = "neon")]
fn over(a: int16x8_t, b: int16x8_t, t: int16x8_t) -> uint16x8_t {
    vcgtq_s16(vabdq_s16(a, b), t)
}

#[inline]
#[target_feature(enable = "neon")]
fn clamp(v: int16x8_t, lo: int16x8_t, hi: int16x8_t) -> int16x8_t {
    vminq_s16(vmaxq_s16(v, lo), hi)
}

/// The wide filter of 7.14.6.4 with `n`, `n2` and `log2Size`, written to rows `7 - n` to
/// `7 + n - 1` of `o` in the lanes of `m`. The sums stay below `1 << 16` (unsigned lanes).
#[inline]
#[target_feature(enable = "neon")]
fn wide(
    r: &[uint16x8_t; ROWS],
    o: &mut [uint16x8_t; ROWS],
    n: i32,
    n2: i32,
    log2: i32,
    m: uint16x8_t,
) {
    let round = vdupq_n_u16(1 << (log2 - 1));
    let shift = vdupq_n_s16(-log2 as i16);
    for i in -n..n {
        let mut t = round;
        for j in -n..=n {
            let v = r[(7 + (i + j).clamp(-(n + 1), n)) as usize];
            t = vaddq_u16(t, v);
            if j.abs() <= n2 {
                t = vaddq_u16(t, v);
            }
        }
        let k = (7 + i) as usize;
        o[k] = vbslq_u16(m, vshlq_u16(t, shift), o[k]);
    }
}

/// `lf::sample_filter` on the 8 lanes of `s` (rows as in `super::ROWS`); `super::filter_edge`
/// checked the preconditions.
#[target_feature(enable = "neon")]
pub(super) fn filter(s: &mut [[u16; 8]; ROWS], e: &Edge) {
    let mut r = [vdupq_n_u16(0); ROWS];
    for t in 0..ROWS {
        r[t] = load_u16x8(&s[t]);
    }
    if let Some(o) = kernel(r, e) {
        for t in 0..ROWS {
            store_u16x8(&mut s[t], o[t]);
        }
    }
}

/// `lf::sample_filter` on every sample of the vertical edge `e` (`step` 1) with `m` taps, 8 at
/// a time. Each lane's samples across the edge are one row in memory: 8 (`m <= 4`) or 16
/// samples around its `q0` are loaded whole and transposed into the kernel rows, and back, as
/// dav1d's `lpf_v_sb` assembly does (dav1d, BSD-2-Clause). The samples of a row that the
/// filter does not change are stored back unchanged. `super::rows_fit` checked that the rows
/// are disjoint and inside `px`.
#[target_feature(enable = "neon")]
pub(super) fn filter_v(px: &mut [u16], e: &Edge, m: usize) {
    let zero = vdupq_n_u16(0);
    let mut k0 = 0;
    while k0 < e.n {
        let cnt = (e.n - k0).min(8);
        let q0 = e.pos + k0 * e.along;
        let mut r = [zero; ROWS];
        if m <= 4 {
            // Columns -4..4 from `q0` are rows 3..11.
            let first = q0 - 4;
            r[3..11].copy_from_slice(&load_cols(px, first, e.along, cnt));
            if let Some(o) = kernel(r, e) {
                let mut c = [zero; 8];
                c.copy_from_slice(&o[3..11]);
                store_cols(px, first, e.along, cnt, c);
            }
        } else {
            // Columns -8..8: column j is row j + 7; columns -8 and 7 are not filtered.
            let first = q0 - 8;
            let mut lo = load_cols(px, first, e.along, cnt);
            let mut hi = load_cols(px, first + 8, e.along, cnt);
            r[..7].copy_from_slice(&lo[1..]);
            r[7..].copy_from_slice(&hi[..7]);
            if let Some(o) = kernel(r, e) {
                lo[1..].copy_from_slice(&o[..7]);
                hi[..7].copy_from_slice(&o[7..]);
                store_cols(px, first, e.along, cnt, lo);
                store_cols(px, first + 8, e.along, cnt, hi);
            }
        }
        k0 += 8;
    }
}

/// The 32-bit transpose step of `transpose8`: `vtrn1q_u32` and `vtrn2q_u32` of `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
fn trn32(a: uint16x8_t, b: uint16x8_t) -> (uint32x4_t, uint32x4_t) {
    let (a, b) = (vreinterpretq_u32_u16(a), vreinterpretq_u32_u16(b));
    (vtrn1q_u32(a, b), vtrn2q_u32(a, b))
}

/// The 64-bit transpose step of `transpose8`: `vtrn1q_u64` and `vtrn2q_u64` of `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
fn trn64(a: uint32x4_t, b: uint32x4_t) -> (uint16x8_t, uint16x8_t) {
    let (a, b) = (vreinterpretq_u64_u32(a), vreinterpretq_u64_u32(b));
    (
        vreinterpretq_u16_u64(vtrn1q_u64(a, b)),
        vreinterpretq_u16_u64(vtrn2q_u64(a, b)),
    )
}

/// Transposes an 8x8 block of 16-bit samples: sample `j` of row `i` becomes sample `i` of row
/// `j`.
#[inline]
#[target_feature(enable = "neon")]
fn transpose8(a: [uint16x8_t; 8]) -> [uint16x8_t; 8] {
    // Pairs of rows: b0 holds the even samples of rows 0 and 1, b1 the odd ones, and so on.
    let (b0, b1) = (vtrn1q_u16(a[0], a[1]), vtrn2q_u16(a[0], a[1]));
    let (b2, b3) = (vtrn1q_u16(a[2], a[3]), vtrn2q_u16(a[2], a[3]));
    let (b4, b5) = (vtrn1q_u16(a[4], a[5]), vtrn2q_u16(a[4], a[5]));
    let (b6, b7) = (vtrn1q_u16(a[6], a[7]), vtrn2q_u16(a[6], a[7]));
    // Columns (0, 4), (2, 6), (1, 5) and (3, 7) of rows 0..4, then of rows 4..8.
    let (c0, c2) = trn32(b0, b2);
    let (c1, c3) = trn32(b1, b3);
    let (c4, c6) = trn32(b4, b6);
    let (c5, c7) = trn32(b5, b7);
    let (r0, r4) = trn64(c0, c4);
    let (r1, r5) = trn64(c1, c5);
    let (r2, r6) = trn64(c2, c6);
    let (r3, r7) = trn64(c3, c7);
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// The 8 columns from `px[first + l * along]` of the first `cnt` of 8 lanes: lane `l` of
/// column `j` is `px[first + l * along + j]`, zero for the lanes from `cnt` on.
#[inline]
#[target_feature(enable = "neon")]
fn load_cols(px: &[u16], first: usize, along: usize, cnt: usize) -> [uint16x8_t; 8] {
    let mut a = [vdupq_n_u16(0); 8];
    for l in 0..cnt.min(8) {
        if let Some(s) = px
            .get(first + l * along..)
            .and_then(|s| s.first_chunk::<8>())
        {
            a[l] = load_u16x8(s);
        }
    }
    transpose8(a)
}

/// Stores the columns `c` of the first `cnt` lanes back where `load_cols` read them.
#[inline]
#[target_feature(enable = "neon")]
fn store_cols(px: &mut [u16], first: usize, along: usize, cnt: usize, c: [uint16x8_t; 8]) {
    let a = transpose8(c);
    for l in 0..cnt.min(8) {
        if let Some(d) = px
            .get_mut(first + l * along..)
            .and_then(|s| s.first_chunk_mut::<8>())
        {
            store_u16x8(d, a[l]);
        }
    }
}

/// `lf::sample_filter` on the 8 lanes of the rows `r`, or `None` when no lane passes the
/// filter mask (the rows are then unchanged).
#[inline]
#[target_feature(enable = "neon")]
fn kernel(r: [uint16x8_t; ROWS], e: &Edge) -> Option<[uint16x8_t; ROWS]> {
    let sh = e.bd - 8;
    let [limit, blimit, thresh] = e.lims;
    let limit = vdupq_n_s16((limit << sh) as i16);
    let blimit = vdupq_n_s16((blimit << sh) as i16);
    let thresh = vdupq_n_s16((thresh << sh) as i16);
    let one = vdupq_n_s16(1 << sh);
    let luma = e.plane == 0;
    let filter_len = match (e.filter_size, luma) {
        (4, _) => 4,
        (_, false) => 6,
        (8, true) => 8,
        _ => 16,
    };
    let mut ri = [vdupq_n_s16(0); ROWS];
    for t in 0..ROWS {
        ri[t] = vreinterpretq_s16_u16(r[t]);
    }
    let (p0, p1, p2, p3) = (ri[6], ri[5], ri[4], ri[3]);
    let (q0, q1, q2, q3) = (ri[7], ri[8], ri[9], ri[10]);

    // 7.14.6.2 masks.
    let hev = vorrq_u16(over(p1, p0, thresh), over(q1, q0, thresh));
    let mut bad = vorrq_u16(over(p1, p0, limit), over(q1, q0, limit));
    let d0 = vabdq_s16(p0, q0);
    let edge = vaddq_s16(vaddq_s16(d0, d0), vshrq_n_s16::<1>(vabdq_s16(p1, q1)));
    bad = vorrq_u16(bad, vcgtq_s16(edge, blimit));
    if filter_len >= 6 {
        bad = vorrq_u16(bad, vorrq_u16(over(p2, p1, limit), over(q2, q1, limit)));
    }
    if filter_len >= 8 {
        bad = vorrq_u16(bad, vorrq_u16(over(p3, p2, limit), over(q3, q2, limit)));
    }
    let mask = vmvnq_u16(bad);
    if vmaxvq_u16(mask) == 0 {
        return None;
    }
    let mut flat = vdupq_n_u16(0);
    if e.filter_size >= 8 {
        let mut f = vorrq_u16(over(p1, p0, one), over(q1, q0, one));
        f = vorrq_u16(f, vorrq_u16(over(p2, p0, one), over(q2, q0, one)));
        if filter_len >= 8 {
            f = vorrq_u16(f, vorrq_u16(over(p3, p0, one), over(q3, q0, one)));
        }
        flat = vbicq_u16(mask, f);
    }
    let mut flat2 = vdupq_n_u16(0);
    if e.filter_size >= 16 {
        let mut f = vdupq_n_u16(0);
        for i in 4..=6 {
            f = vorrq_u16(
                f,
                vorrq_u16(over(ri[6 - i], p0, one), over(ri[7 + i], q0, one)),
            );
        }
        flat2 = vbicq_u16(flat, f);
    }
    let mut o = r;

    // 7.14.6.3 narrow filter, in the lanes that pass the mask and are not flat.
    let narrow = vbicq_u16(mask, flat);
    let lo = vdupq_n_s16((-(1i32 << (e.bd - 1))) as i16);
    let hi = vdupq_n_s16(((1i32 << (e.bd - 1)) - 1) as i16);
    let off = vdupq_n_s16((0x80i32 << sh) as i16);
    let (ps1, ps0) = (vsubq_s16(p1, off), vsubq_s16(p0, off));
    let (qs0, qs1) = (vsubq_s16(q0, off), vsubq_s16(q1, off));
    let mut f = vandq_s16(
        clamp(vsubq_s16(ps1, qs1), lo, hi),
        vreinterpretq_s16_u16(hev),
    );
    let d = vsubq_s16(qs0, ps0);
    f = clamp(vaddq_s16(f, vaddq_s16(vaddq_s16(d, d), d)), lo, hi);
    let f1 = vshrq_n_s16::<3>(clamp(vaddq_s16(f, vdupq_n_s16(4)), lo, hi));
    let f2 = vshrq_n_s16::<3>(clamp(vaddq_s16(f, vdupq_n_s16(3)), lo, hi));
    let nq0 = vaddq_s16(clamp(vsubq_s16(qs0, f1), lo, hi), off);
    let np0 = vaddq_s16(clamp(vaddq_s16(ps0, f2), lo, hi), off);
    let f3 = vshrq_n_s16::<1>(vaddq_s16(f1, vdupq_n_s16(1)));
    let nq1 = vaddq_s16(clamp(vsubq_s16(qs1, f3), lo, hi), off);
    let np1 = vaddq_s16(clamp(vaddq_s16(ps1, f3), lo, hi), off);
    o[7] = vbslq_u16(narrow, vreinterpretq_u16_s16(nq0), o[7]);
    o[6] = vbslq_u16(narrow, vreinterpretq_u16_s16(np0), o[6]);
    let outer = vbicq_u16(narrow, hev);
    o[8] = vbslq_u16(outer, vreinterpretq_u16_s16(nq1), o[8]);
    o[5] = vbslq_u16(outer, vreinterpretq_u16_s16(np1), o[5]);

    // 7.14.6.4 wide filters, in the flat lanes.
    if e.filter_size >= 8 {
        let (n, n2) = if luma { (3, 0) } else { (2, 1) };
        wide(&r, &mut o, n, n2, 3, vbicq_u16(flat, flat2));
    }
    if e.filter_size >= 16 {
        wide(&r, &mut o, 6, 1, 4, flat2);
    }
    Some(o)
}
