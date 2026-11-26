//! AVX-512 loop filter kernel: the AVX2 kernel (`avx2.rs`) on 32 samples along the edge per
//! vector, with mask registers in place of the byte blends. AVX-512 BW instructions only.

use super::{Edge, ROWS};
use crate::simd::x86::{load512_u16, store512_u16};
use std::arch::x86_64::*;

type M = __mmask32;

#[inline]
#[target_feature(enable = "avx512bw")]
fn absd(a: __m512i, b: __m512i) -> __m512i {
    _mm512_abs_epi16(_mm512_sub_epi16(a, b))
}

/// The lanes where `|a - b| > t`.
#[inline]
#[target_feature(enable = "avx512bw")]
fn over(a: __m512i, b: __m512i, t: __m512i) -> M {
    _mm512_cmpgt_epi16_mask(absd(a, b), t)
}

#[inline]
#[target_feature(enable = "avx512bw")]
fn clamp(v: __m512i, lo: __m512i, hi: __m512i) -> __m512i {
    _mm512_min_epi16(_mm512_max_epi16(v, lo), hi)
}

/// The wide filter of 7.14.6.4 with `n`, `n2` and `log2Size`, written to rows `7 - n` to
/// `7 + n - 1` of `o` in the lanes of `m`. The sums stay below `1 << 16` (unsigned lanes).
#[inline]
#[target_feature(enable = "avx512bw")]
fn wide(r: &[__m512i; ROWS], o: &mut [__m512i; ROWS], n: i32, n2: i32, log2: i32, m: M) {
    let round = _mm512_set1_epi16(1 << (log2 - 1));
    let count = _mm_cvtsi32_si128(log2);
    for i in -n..n {
        let mut t = round;
        for j in -n..=n {
            let v = r[(7 + (i + j).clamp(-(n + 1), n)) as usize];
            t = _mm512_add_epi16(t, v);
            if j.abs() <= n2 {
                t = _mm512_add_epi16(t, v);
            }
        }
        let k = (7 + i) as usize;
        o[k] = _mm512_mask_mov_epi16(o[k], m, _mm512_srl_epi16(t, count));
    }
}

/// `lf::sample_filter` on the 32 lanes of `s` (rows as in `super::ROWS`); `super::filter_edge`
/// checked the preconditions.
#[target_feature(enable = "avx512bw")]
pub(super) fn filter(s: &mut [[u16; 32]; ROWS], e: &Edge) {
    let mut r = [_mm512_setzero_si512(); ROWS];
    for t in 0..ROWS {
        r[t] = load512_u16(&s[t]);
    }
    let sh = e.bd - 8;
    let [limit, blimit, thresh] = e.lims;
    let limit = _mm512_set1_epi16((limit << sh) as i16);
    let blimit = _mm512_set1_epi16((blimit << sh) as i16);
    let thresh = _mm512_set1_epi16((thresh << sh) as i16);
    let one = _mm512_set1_epi16(1 << sh);
    let luma = e.plane == 0;
    let filter_len = match (e.filter_size, luma) {
        (4, _) => 4,
        (_, false) => 6,
        (8, true) => 8,
        _ => 16,
    };
    let (p0, p1, p2, p3) = (r[6], r[5], r[4], r[3]);
    let (q0, q1, q2, q3) = (r[7], r[8], r[9], r[10]);

    // 7.14.6.2 masks.
    let hev = over(p1, p0, thresh) | over(q1, q0, thresh);
    let mut bad = over(p1, p0, limit) | over(q1, q0, limit);
    let d0 = absd(p0, q0);
    let edge = _mm512_add_epi16(
        _mm512_add_epi16(d0, d0),
        _mm512_srli_epi16::<1>(absd(p1, q1)),
    );
    bad |= _mm512_cmpgt_epi16_mask(edge, blimit);
    if filter_len >= 6 {
        bad |= over(p2, p1, limit) | over(q2, q1, limit);
    }
    if filter_len >= 8 {
        bad |= over(p3, p2, limit) | over(q3, q2, limit);
    }
    let mask = !bad;
    if mask == 0 {
        return;
    }
    let mut flat: M = 0;
    if e.filter_size >= 8 {
        let mut f = over(p1, p0, one) | over(q1, q0, one);
        f |= over(p2, p0, one) | over(q2, q0, one);
        if filter_len >= 8 {
            f |= over(p3, p0, one) | over(q3, q0, one);
        }
        flat = !f & mask;
    }
    let mut flat2: M = 0;
    if e.filter_size >= 16 {
        let mut f: M = 0;
        for i in 4..=6 {
            f |= over(r[6 - i], p0, one) | over(r[7 + i], q0, one);
        }
        flat2 = !f & flat;
    }
    let mut o = r;

    // 7.14.6.3 narrow filter, in the lanes that pass the mask and are not flat.
    let narrow = !flat & mask;
    let lo = _mm512_set1_epi16((-(1i32 << (e.bd - 1))) as i16);
    let hi = _mm512_set1_epi16(((1i32 << (e.bd - 1)) - 1) as i16);
    let off = _mm512_set1_epi16((0x80i32 << sh) as i16);
    let (ps1, ps0) = (_mm512_sub_epi16(p1, off), _mm512_sub_epi16(p0, off));
    let (qs0, qs1) = (_mm512_sub_epi16(q0, off), _mm512_sub_epi16(q1, off));
    let mut f = _mm512_maskz_mov_epi16(hev, clamp(_mm512_sub_epi16(ps1, qs1), lo, hi));
    let d = _mm512_sub_epi16(qs0, ps0);
    f = clamp(
        _mm512_add_epi16(f, _mm512_add_epi16(_mm512_add_epi16(d, d), d)),
        lo,
        hi,
    );
    let f1 = _mm512_srai_epi16::<3>(clamp(_mm512_add_epi16(f, _mm512_set1_epi16(4)), lo, hi));
    let f2 = _mm512_srai_epi16::<3>(clamp(_mm512_add_epi16(f, _mm512_set1_epi16(3)), lo, hi));
    let nq0 = _mm512_add_epi16(clamp(_mm512_sub_epi16(qs0, f1), lo, hi), off);
    let np0 = _mm512_add_epi16(clamp(_mm512_add_epi16(ps0, f2), lo, hi), off);
    let f3 = _mm512_srai_epi16::<1>(_mm512_add_epi16(f1, _mm512_set1_epi16(1)));
    let nq1 = _mm512_add_epi16(clamp(_mm512_sub_epi16(qs1, f3), lo, hi), off);
    let np1 = _mm512_add_epi16(clamp(_mm512_add_epi16(ps1, f3), lo, hi), off);
    o[7] = _mm512_mask_mov_epi16(o[7], narrow, nq0);
    o[6] = _mm512_mask_mov_epi16(o[6], narrow, np0);
    let outer = !hev & narrow;
    o[8] = _mm512_mask_mov_epi16(o[8], outer, nq1);
    o[5] = _mm512_mask_mov_epi16(o[5], outer, np1);

    // 7.14.6.4 wide filters, in the flat lanes.
    if e.filter_size >= 8 {
        let (n, n2) = if luma { (3, 0) } else { (2, 1) };
        wide(&r, &mut o, n, n2, 3, !flat2 & flat);
    }
    if e.filter_size >= 16 {
        wide(&r, &mut o, 6, 1, 4, flat2);
    }
    for t in 0..ROWS {
        store512_u16(&mut s[t], o[t]);
    }
}
