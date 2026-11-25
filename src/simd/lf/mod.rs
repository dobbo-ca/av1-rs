//! Loop filter kernels (spec 7.14.6): one run of samples along an edge, several samples per
//! vector, each lane doing exactly what `lf::sample_filter` (the scalar oracle) does for one
//! sample. Like dav1d's loop filter, the lanes run along the edge; the samples across the edge
//! (`p6..p0`, `q0..q6`) are gathered into rows of a small buffer first, in safe code, and
//! written back after the kernel.
//!
//! Every lane works in 16 bits: samples are below `1 << bd` (12 bits at most), so the narrow
//! filter's signed values stay within `i16` and the wide filter's sums (taps add up to 8 or 16)
//! stay below `1 << 16`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use super::{Simd, Tier};
use crate::lf::sample_filter;

/// The gather buffer has one row per sample across the edge: row `6 - i` is `p_i`, row `7 + i`
/// is `q_i`, as `s[]` in `lf::sample_filter`.
pub(crate) const ROWS: usize = 14;

/// One run of samples along an edge: sample `k` (0 to `n - 1`) has its `q0` at
/// `px[pos + k * along]`, and `step` is the distance across the edge (1 for a vertical edge,
/// the stride for a horizontal one). The other fields are those of `lf::sample_filter`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Edge {
    pub pos: usize,
    pub step: usize,
    pub along: usize,
    pub n: usize,
    pub plane: usize,
    pub bd: u32,
    pub lims: [i32; 3],
    pub filter_size: usize,
}

/// Filters every sample of `e`, as `lf::sample_filter` on each in turn.
///
/// A SIMD tier runs only on the inputs the decoder makes: bit depth 8, 10 or 12, filter size 4,
/// 8 or 16 (4 or 8 for chroma), limits in 0 to 255, every sample of every window in `px`, and
/// windows that do not overlap. Anything else, and the scalar tier, filter sample by sample.
/// The samples must be below `1 << bd`, as every decoded sample is.
#[allow(unsafe_code)]
pub(crate) fn filter_edge(simd: Simd, px: &mut [u16], e: &Edge) {
    let Some(m) = vector_taps(px.len(), e) else {
        scalar(px, e);
        return;
    };
    match simd.tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => chunks::<16>(px, e, m, |s| {
            // SAFETY: `simd` is a `Simd` of tier AVX2 or AVX-512, which `simd::available` or
            // `simd::best` made only after detecting AVX2.
            unsafe { avx2::filter(s, e) }
        }),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => chunks::<8>(px, e, m, |s| {
            // SAFETY: `simd` is a `Simd` of tier NEON, which `simd::available` or `simd::best`
            // made only after detecting NEON.
            unsafe { neon::filter(s, e) }
        }),
        _ => scalar(px, e),
    }
}

fn scalar(px: &mut [u16], e: &Edge) {
    for k in 0..e.n {
        let Some(pos) = k.checked_mul(e.along).and_then(|d| d.checked_add(e.pos)) else {
            return;
        };
        sample_filter(px, pos, e.step, e.plane, e.bd, e.lims, e.filter_size);
    }
}

/// `m` (the samples each side of the edge that the filter reads) when the SIMD kernels may run
/// on `e` in a buffer of `len` samples (see `filter_edge`), else `None`.
fn vector_taps(len: usize, e: &Edge) -> Option<usize> {
    let m = match (e.filter_size, e.plane) {
        (4, _) => 2,
        (8, 0) => 4,
        (8, _) => 3,
        (16, 0) => 7,
        _ => return None,
    };
    let ok = matches!(e.bd, 8 | 10 | 12)
        && e.lims.iter().all(|v| (0..=255).contains(v))
        && e.n > 0
        // Disjoint windows: the window of sample k is `pos + k * along + j * step`, j in -m..m.
        && ((e.step == 1 && e.along >= 2 * m) || (e.along == 1 && e.step >= e.n))
        && e.pos >= m.checked_mul(e.step)?
        && (e.n - 1)
            .checked_mul(e.along)?
            .checked_add((m - 1).checked_mul(e.step)?)?
            .checked_add(e.pos)?
            < len;
    ok.then_some(m)
}

/// Runs `kernel` on the samples of `e` in chunks of `L` lanes (the last one padded with zero
/// lanes that are not written back). `vector_taps` checked every index.
fn chunks<const L: usize>(
    px: &mut [u16],
    e: &Edge,
    m: usize,
    kernel: impl Fn(&mut [[u16; L]; ROWS]),
) {
    let mut k0 = 0;
    while k0 < e.n {
        let cnt = L.min(e.n - k0);
        // `px[first + j * step]` (j in 0..2m) is row 7 - m + j of lane 0.
        let first = e.pos + k0 * e.along - m * e.step;
        let mut s = [[0u16; L]; ROWS];
        if e.along == 1 {
            for j in 0..2 * m {
                let b = first + j * e.step;
                let src = &px[b..b + cnt];
                // A full chunk is a fixed-size copy (no `memmove` call).
                if let Some(a) = src.first_chunk::<L>() {
                    s[7 - m + j] = *a;
                } else {
                    s[7 - m + j][..cnt].copy_from_slice(src);
                }
            }
        } else {
            // `step` is 1 here (`vector_taps`): the window of each lane is one slice.
            for l in 0..cnt {
                let b = first + l * e.along;
                for (j, &v) in px[b..b + 2 * m].iter().enumerate() {
                    s[7 - m + j][l] = v;
                }
            }
        }
        kernel(&mut s);
        if e.along == 1 {
            for j in 0..2 * m {
                let b = first + j * e.step;
                let dst = &mut px[b..b + cnt];
                if let Some(a) = dst.first_chunk_mut::<L>() {
                    *a = s[7 - m + j];
                } else {
                    dst.copy_from_slice(&s[7 - m + j][..cnt]);
                }
            }
        } else {
            for l in 0..cnt {
                let b = first + l * e.along;
                for (j, d) in px[b..b + 2 * m].iter_mut().enumerate() {
                    *d = s[7 - m + j][l];
                }
            }
        }
        k0 += L;
    }
}

/// Builds one edge run from `data` (any bytes), filters it with every available tier and
/// asserts that each output equals the scalar output. Used by the tests and `fuzzing::lf`.
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn check(data: &[u8]) {
    let b = |i: usize| data.get(i).copied().unwrap_or(0);
    let seed = data
        .iter()
        .skip(11)
        .fold(0x9e37_79b9_7f4a_7c15u64, |h, &v| {
            (h ^ u64::from(v)).wrapping_mul(0x100_0000_01b3)
        });
    check_case(&Case {
        bd: [8, 10, 12][usize::from(b(0)) % 3],
        plane: usize::from(b(1) & 1),
        filter_size: [4, 8, 16][usize::from(b(2)) % 3],
        vertical: b(3) & 1 != 0,
        n: usize::from(b(4)) % 70,
        lvl: 1 + i32::from(b(5)) % 63,
        sharp: b(6) % 8,
        extra: usize::from(b(7)) % 24,
        start: usize::from(b(8)) % 24,
        mode: b(9) % 5,
        delta: usize::from(b(10)),
        seed: seed | 1,
    });
}

#[cfg(any(test, feature = "fuzzing"))]
struct Case {
    bd: u32,
    plane: usize,
    filter_size: usize,
    vertical: bool,
    /// Samples along the edge.
    n: usize,
    lvl: i32,
    sharp: u8,
    /// Spare stride beyond what the run needs.
    extra: usize,
    /// Offset of the run along the edge (unaligned starts).
    start: usize,
    /// How the samples are drawn: 0 random, 1 near flat, 2 step at the edge, 3 min/max only,
    /// 4 flat each side with a step of `delta`.
    mode: u8,
    delta: usize,
    seed: u64,
}

#[cfg(any(test, feature = "fuzzing"))]
fn check_case(c: &Case) {
    let mut rng = c.seed.max(1);
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let filter_size = if c.plane != 0 {
        c.filter_size.min(8)
    } else {
        c.filter_size
    };
    let max = (1u32 << c.bd) - 1;
    // The plane: 16 samples each side across the edge, `start + n` along it, plus spare.
    let (w, h) = if c.vertical {
        (32 + c.extra, c.start + c.n)
    } else {
        (c.start + c.n + c.extra, 32)
    };
    let (pos, step, along) = if c.vertical {
        (c.start * w + 16, 1, w)
    } else {
        (16 * w + c.start, w, 1)
    };
    let base = (next() as u32) & max;
    let step_px = (c.delta as u32) << (c.bd - 8);
    let mut px = vec![0u16; w * h];
    for (i, v) in px.iter_mut().enumerate() {
        let across = if c.vertical { i % w } else { i / w };
        let r = next() as u32;
        let s = match c.mode {
            0 => r & max,
            1 => (base + (r % 3) * (1 << (c.bd - 8))).min(max),
            2 => {
                let near = (base + (r % 5)).min(max);
                if across >= 16 {
                    (near + step_px).min(max)
                } else {
                    near
                }
            }
            3 => {
                if r & 1 == 0 {
                    0
                } else {
                    max
                }
            }
            _ => {
                if across >= 16 {
                    (base + step_px).min(max)
                } else {
                    base
                }
            }
        };
        *v = s as u16;
    }
    let e = Edge {
        pos,
        step,
        along,
        n: c.n,
        plane: c.plane,
        bd: c.bd,
        lims: crate::lf::limits(c.lvl, c.sharp),
        filter_size,
    };
    let mut want = px.clone();
    filter_edge(Simd::scalar(), &mut want, &e);
    for simd in super::available() {
        let mut got = px.clone();
        filter_edge(simd, &mut got, &e);
        assert_eq!(got, want, "{:?} {e:?} mode {}", simd.tier(), c.mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn random_runs_match_scalar() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let mut data = [0u8; 16];
        for _ in 0..3000 {
            for d in &mut data {
                *d = xorshift(&mut st) as u8;
            }
            check(&data);
        }
    }

    #[test]
    fn edge_sizes_and_values_match_scalar() {
        let mut seed = 1u64;
        for bd in [8, 10, 12] {
            for (plane, filter_size) in [(0, 4), (0, 8), (0, 16), (1, 4), (1, 8)] {
                for vertical in [false, true] {
                    for n in [0, 1, 4, 7, 8, 9, 15, 16, 17, 33] {
                        for (start, mode) in [(0, 1), (3, 2), (5, 3), (1, 4), (0, 0)] {
                            for lvl in [1, 32, 63] {
                                seed += 1;
                                check_case(&Case {
                                    bd,
                                    plane,
                                    filter_size,
                                    vertical,
                                    n,
                                    lvl,
                                    sharp: (seed % 8) as u8,
                                    extra: (seed % 3) as usize,
                                    start,
                                    mode,
                                    delta: (seed % 40) as usize,
                                    seed,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Runs that break a SIMD precondition still match scalar (they take the scalar path).
    #[test]
    fn unsupported_runs_fall_back() {
        let mut px: Vec<u16> = (0..64 * 8).map(|i| (i * 7 % 256) as u16).collect();
        for e in [
            // Overlapping windows (along < 2 m).
            Edge {
                pos: 8,
                step: 1,
                along: 3,
                n: 4,
                plane: 0,
                bd: 8,
                lims: [10, 40, 1],
                filter_size: 8,
            },
            // Runs past the end of the buffer.
            Edge {
                pos: 64 * 7 + 8,
                step: 64,
                along: 1,
                n: 16,
                plane: 0,
                bd: 8,
                lims: [10, 40, 1],
                filter_size: 4,
            },
            // Starts too close to the start.
            Edge {
                pos: 1,
                step: 1,
                along: 64,
                n: 8,
                plane: 0,
                bd: 8,
                lims: [10, 40, 1],
                filter_size: 16,
            },
        ] {
            let mut want = px.clone();
            filter_edge(Simd::scalar(), &mut want, &e);
            for simd in super::super::available() {
                let mut got = px.clone();
                filter_edge(simd, &mut got, &e);
                assert_eq!(got, want, "{:?} {e:?}", simd.tier());
            }
            px = want;
        }
    }
}
