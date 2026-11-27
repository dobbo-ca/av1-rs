//! AVX2 loop filter kernel: 16 samples along the edge per vector, one 16-bit lane each.
//! The AVX-512 tier uses it too for every vertical edge (`filter_v`) and for horizontal runs
//! of 16 or fewer samples (longer runs: `avx512.rs`).

use super::{Edge, ROWS};
use crate::simd::x86::{load128_u16, load256_u16, store128_u16, store256_u16};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn absd(a: __m256i, b: __m256i) -> __m256i {
    _mm256_abs_epi16(_mm256_sub_epi16(a, b))
}

/// All ones in the lanes where `|a - b| > t`.
#[inline]
#[target_feature(enable = "avx2")]
fn over(a: __m256i, b: __m256i, t: __m256i) -> __m256i {
    _mm256_cmpgt_epi16(absd(a, b), t)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clamp(v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_min_epi16(_mm256_max_epi16(v, lo), hi)
}

/// The wide filter of 7.14.6.4 with `n`, `n2` and `log2Size`, written to rows `7 - n` to
/// `7 + n - 1` of `o` in the lanes of `m`. The sums stay below `1 << 16` (unsigned lanes).
#[inline]
#[target_feature(enable = "avx2")]
fn wide(r: &[__m256i; ROWS], o: &mut [__m256i; ROWS], n: i32, n2: i32, log2: i32, m: __m256i) {
    let round = _mm256_set1_epi16(1 << (log2 - 1));
    let count = _mm_cvtsi32_si128(log2);
    for i in -n..n {
        let mut t = round;
        for j in -n..=n {
            let v = r[(7 + (i + j).clamp(-(n + 1), n)) as usize];
            t = _mm256_add_epi16(t, v);
            if j.abs() <= n2 {
                t = _mm256_add_epi16(t, v);
            }
        }
        let k = (7 + i) as usize;
        o[k] = _mm256_blendv_epi8(o[k], _mm256_srl_epi16(t, count), m);
    }
}

/// `lf::sample_filter` on the 16 lanes of `s` (rows as in `super::ROWS`); `super::filter_edge`
/// checked the preconditions.
#[target_feature(enable = "avx2")]
pub(super) fn filter(s: &mut [[u16; 16]; ROWS], e: &Edge) {
    let mut r = [_mm256_setzero_si256(); ROWS];
    for t in 0..ROWS {
        r[t] = load256_u16(&s[t]);
    }
    if let Some(o) = kernel(r, e) {
        for t in 0..ROWS {
            store256_u16(&mut s[t], o[t]);
        }
    }
}

/// `lf::sample_filter` on every sample of the vertical edge `e` (`step` 1) with `m` taps, 16
/// at a time. Each lane's samples across the edge are one row in memory: 8 (`m <= 4`) or 16
/// samples around its `q0` are loaded whole and transposed into the kernel rows, and back, as
/// dav1d's `lpf_v_sb` assembly does (dav1d, BSD-2-Clause). The samples of a row that the
/// filter does not change are stored back unchanged. `super::rows_fit` checked that the rows
/// are disjoint and inside `px`.
#[target_feature(enable = "avx2")]
pub(super) fn filter_v(px: &mut [u16], e: &Edge, m: usize) {
    let zero = _mm256_setzero_si256();
    let mut k0 = 0;
    while k0 < e.n {
        let cnt = (e.n - k0).min(16);
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
        k0 += 16;
    }
}

/// Transposes the 8x8 block of 16-bit samples in the low 128-bit half of `a`, and the one in
/// the high half: sample `j` of row `i` becomes sample `i` of row `j`, in each half.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose8(a: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi16(a[0], a[1]);
    let t1 = _mm256_unpackhi_epi16(a[0], a[1]);
    let t2 = _mm256_unpacklo_epi16(a[2], a[3]);
    let t3 = _mm256_unpackhi_epi16(a[2], a[3]);
    let t4 = _mm256_unpacklo_epi16(a[4], a[5]);
    let t5 = _mm256_unpackhi_epi16(a[4], a[5]);
    let t6 = _mm256_unpacklo_epi16(a[6], a[7]);
    let t7 = _mm256_unpackhi_epi16(a[6], a[7]);
    let u0 = _mm256_unpacklo_epi32(t0, t2);
    let u1 = _mm256_unpackhi_epi32(t0, t2);
    let u2 = _mm256_unpacklo_epi32(t1, t3);
    let u3 = _mm256_unpackhi_epi32(t1, t3);
    let u4 = _mm256_unpacklo_epi32(t4, t6);
    let u5 = _mm256_unpackhi_epi32(t4, t6);
    let u6 = _mm256_unpacklo_epi32(t5, t7);
    let u7 = _mm256_unpackhi_epi32(t5, t7);
    [
        _mm256_unpacklo_epi64(u0, u4),
        _mm256_unpackhi_epi64(u0, u4),
        _mm256_unpacklo_epi64(u1, u5),
        _mm256_unpackhi_epi64(u1, u5),
        _mm256_unpacklo_epi64(u2, u6),
        _mm256_unpackhi_epi64(u2, u6),
        _mm256_unpacklo_epi64(u3, u7),
        _mm256_unpackhi_epi64(u3, u7),
    ]
}

/// The 8 columns from `px[first + l * along]` of the first `cnt` of 16 lanes: lane `l` of
/// column `j` is `px[first + l * along + j]`, zero for the lanes from `cnt` on.
#[inline]
#[target_feature(enable = "avx2")]
fn load_cols(px: &[u16], first: usize, along: usize, cnt: usize) -> [__m256i; 8] {
    let mut h = [_mm_setzero_si128(); 16];
    for l in 0..cnt.min(16) {
        if let Some(s) = px
            .get(first + l * along..)
            .and_then(|s| s.first_chunk::<8>())
        {
            h[l] = load128_u16(s);
        }
    }
    // Lane l in the low half of row l, lane l + 8 in the high half.
    let mut a = [_mm256_setzero_si256(); 8];
    for l in 0..8 {
        a[l] = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(h[l]), h[l + 8]);
    }
    transpose8(a)
}

/// Stores the columns `c` of the first `cnt` lanes back where `load_cols` read them.
#[inline]
#[target_feature(enable = "avx2")]
fn store_cols(px: &mut [u16], first: usize, along: usize, cnt: usize, c: [__m256i; 8]) {
    let a = transpose8(c);
    for l in 0..cnt.min(16) {
        let v = if l < 8 {
            _mm256_castsi256_si128(a[l])
        } else {
            _mm256_extracti128_si256::<1>(a[l - 8])
        };
        if let Some(d) = px
            .get_mut(first + l * along..)
            .and_then(|s| s.first_chunk_mut::<8>())
        {
            store128_u16(d, v);
        }
    }
}

/// `lf::sample_filter` on the 16 lanes of the rows `r`, or `None` when no lane passes the
/// filter mask (the rows are then unchanged).
#[inline]
#[target_feature(enable = "avx2")]
fn kernel(r: [__m256i; ROWS], e: &Edge) -> Option<[__m256i; ROWS]> {
    let sh = e.bd - 8;
    let [limit, blimit, thresh] = e.lims;
    let limit = _mm256_set1_epi16((limit << sh) as i16);
    let blimit = _mm256_set1_epi16((blimit << sh) as i16);
    let thresh = _mm256_set1_epi16((thresh << sh) as i16);
    let one = _mm256_set1_epi16(1 << sh);
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
    let hev = _mm256_or_si256(over(p1, p0, thresh), over(q1, q0, thresh));
    let mut bad = _mm256_or_si256(over(p1, p0, limit), over(q1, q0, limit));
    let d0 = absd(p0, q0);
    let edge = _mm256_add_epi16(_mm256_add_epi16(d0, d0), _mm256_srli_epi16(absd(p1, q1), 1));
    bad = _mm256_or_si256(bad, _mm256_cmpgt_epi16(edge, blimit));
    if filter_len >= 6 {
        bad = _mm256_or_si256(
            bad,
            _mm256_or_si256(over(p2, p1, limit), over(q2, q1, limit)),
        );
    }
    if filter_len >= 8 {
        bad = _mm256_or_si256(
            bad,
            _mm256_or_si256(over(p3, p2, limit), over(q3, q2, limit)),
        );
    }
    let mask = _mm256_andnot_si256(bad, _mm256_set1_epi16(-1));
    if _mm256_movemask_epi8(mask) == 0 {
        return None;
    }
    let mut flat = _mm256_setzero_si256();
    if e.filter_size >= 8 {
        let mut f = _mm256_or_si256(over(p1, p0, one), over(q1, q0, one));
        f = _mm256_or_si256(f, _mm256_or_si256(over(p2, p0, one), over(q2, q0, one)));
        if filter_len >= 8 {
            f = _mm256_or_si256(f, _mm256_or_si256(over(p3, p0, one), over(q3, q0, one)));
        }
        flat = _mm256_andnot_si256(f, mask);
    }
    let mut flat2 = _mm256_setzero_si256();
    if e.filter_size >= 16 {
        let mut f = _mm256_setzero_si256();
        for i in 4..=6 {
            f = _mm256_or_si256(
                f,
                _mm256_or_si256(over(r[6 - i], p0, one), over(r[7 + i], q0, one)),
            );
        }
        flat2 = _mm256_andnot_si256(f, flat);
    }
    let mut o = r;

    // 7.14.6.3 narrow filter, in the lanes that pass the mask and are not flat.
    let narrow = _mm256_andnot_si256(flat, mask);
    let lo = _mm256_set1_epi16((-(1i32 << (e.bd - 1))) as i16);
    let hi = _mm256_set1_epi16(((1i32 << (e.bd - 1)) - 1) as i16);
    let off = _mm256_set1_epi16((0x80i32 << sh) as i16);
    let (ps1, ps0) = (_mm256_sub_epi16(p1, off), _mm256_sub_epi16(p0, off));
    let (qs0, qs1) = (_mm256_sub_epi16(q0, off), _mm256_sub_epi16(q1, off));
    let mut f = _mm256_and_si256(clamp(_mm256_sub_epi16(ps1, qs1), lo, hi), hev);
    let d = _mm256_sub_epi16(qs0, ps0);
    f = clamp(
        _mm256_add_epi16(f, _mm256_add_epi16(_mm256_add_epi16(d, d), d)),
        lo,
        hi,
    );
    let f1 = _mm256_srai_epi16(clamp(_mm256_add_epi16(f, _mm256_set1_epi16(4)), lo, hi), 3);
    let f2 = _mm256_srai_epi16(clamp(_mm256_add_epi16(f, _mm256_set1_epi16(3)), lo, hi), 3);
    let nq0 = _mm256_add_epi16(clamp(_mm256_sub_epi16(qs0, f1), lo, hi), off);
    let np0 = _mm256_add_epi16(clamp(_mm256_add_epi16(ps0, f2), lo, hi), off);
    let f3 = _mm256_srai_epi16(_mm256_add_epi16(f1, _mm256_set1_epi16(1)), 1);
    let nq1 = _mm256_add_epi16(clamp(_mm256_sub_epi16(qs1, f3), lo, hi), off);
    let np1 = _mm256_add_epi16(clamp(_mm256_add_epi16(ps1, f3), lo, hi), off);
    o[7] = _mm256_blendv_epi8(o[7], nq0, narrow);
    o[6] = _mm256_blendv_epi8(o[6], np0, narrow);
    let outer = _mm256_andnot_si256(hev, narrow);
    o[8] = _mm256_blendv_epi8(o[8], nq1, outer);
    o[5] = _mm256_blendv_epi8(o[5], np1, outer);

    // 7.14.6.4 wide filters, in the flat lanes.
    if e.filter_size >= 8 {
        let (n, n2) = if luma { (3, 0) } else { (2, 1) };
        wide(&r, &mut o, n, n2, 3, _mm256_andnot_si256(flat2, flat));
    }
    if e.filter_size >= 16 {
        wide(&r, &mut o, 6, 1, 4, flat2);
    }
    Some(o)
}
