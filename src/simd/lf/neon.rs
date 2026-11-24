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
        return;
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
    for t in 0..ROWS {
        store_u16x8(&mut s[t], o[t]);
    }
}
