//! Loop restoration kernels (spec 7.17): the Wiener filter (7.17.4) and the self guided filter
//! (7.17.2, 7.17.3) of one block of at most `BLOCK_W` x `BLOCK_H` samples, from the source window
//! that `lr::source_window` gathered (the stripe edge rules are already applied there). The
//! scalar oracle is `lr::wiener_filter` and `lr::self_guided_filter`.
//!
//! The SIMD kernels compute in 32-bit lanes. With samples below 2^12 every intermediate of the
//! spec fits: Wiener sums stay below 2^26 for any `i8` taps, box sums below 2^29, and `p * s`
//! below 2^31 because `p` is clipped at a value from which `z >= 255` (`a2` is 256 from there
//! on). `(256 - a2) * b * oneOverN` is below 2^32 (the bound dav1d relies on too): it is computed
//! with wrapping multiplies and a logical shift. `a2 = ((z << 8) + z / 2) / (z + 1)` uses an f32
//! division, which is exact after truncation for these operands (the quotient is below 256 and at
//! least 1/256 away from the next integer, far more than the f32 error). Blocks narrower than
//! one vector go to the oracle; the last vector of a row overlaps the one before it instead of
//! running past the row, and computes the same values again.

use super::{Simd, Tier};
use crate::consts::{FILTER_BITS, SGRPROJ_MTABLE_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_RECIP_BITS};
use crate::frame::Plane;
use crate::lr::{BLOCK_W, SGR_PARAMS, self_guided_filter, wiener_coefficients, wiener_filter};

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Tallest block: one stripe.
const BLOCK_H: usize = 64;

/// The Wiener constants of a block (spec 7.17.4, 7.11.3.2), as the oracle derives them.
struct WienerK {
    /// Taps 0..=3 of the horizontal and the vertical filter (taps 4..=6 mirror them).
    hf: [i32; 4],
    vf: [i32; 4],
    /// `InterRound0`, `InterRound1`.
    round0: i32,
    round1: i32,
    /// The clip of the horizontal pass: `-offset` and `limit - offset`.
    lo: i32,
    hi: i32,
    /// `(1 << BitDepth) - 1`.
    max: i32,
}

impl WienerK {
    fn new(bd: usize, coef: &[[i8; 3]; 2]) -> Self {
        let (round0, round1) = if bd == 12 { (5, 9) } else { (3, 11) };
        let offset = 1i32 << (bd + FILTER_BITS - round0 - 1);
        let limit = (1i32 << (bd + 1 + FILTER_BITS - round0)) - 1;
        let hf = wiener_coefficients(&coef[1]);
        let vf = wiener_coefficients(&coef[0]);
        Self {
            hf: [hf[0] as i32, hf[1] as i32, hf[2] as i32, hf[3] as i32],
            vf: [vf[0] as i32, vf[1] as i32, vf[2] as i32, vf[3] as i32],
            round0: round0 as i32,
            round1,
            lo: -offset,
            hi: limit - offset,
            max: (1 << bd) - 1,
        }
    }
}

/// The constants of one box filter pass (spec 7.17.3), as the oracle derives them.
struct BoxK {
    r: usize,
    pass0: bool,
    /// `n`, `s` and `oneOverN` of the spec.
    n: i32,
    s: i32,
    one_over_n: i32,
    /// The least `p` with `p * s >= 255 << SGRPROJ_MTABLE_BITS`, so `z >= 255` from there on.
    pmax: i32,
    /// `2 * (BitDepth - 8)` and `BitDepth - 8`.
    sh_a: i32,
    sh_b: i32,
}

impl BoxK {
    fn new(bd: usize, r: usize, eps: i64, pass0: bool) -> Self {
        let n = ((2 * r + 1) * (2 * r + 1)) as i64;
        let n2e = n * n * eps;
        let s = ((1i64 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
        let one_over_n = ((1i64 << SGRPROJ_RECIP_BITS) + n / 2) / n;
        let pmax = ((255i64 << SGRPROJ_MTABLE_BITS) + s - 1) / s;
        Self {
            r,
            pass0,
            n: n as i32,
            s: s as i32,
            one_over_n: one_over_n as i32,
            pmax: pmax as i32,
            sh_a: 2 * (bd as i32 - 8),
            sh_b: bd as i32 - 8,
        }
    }
}

/// The self guided constants of a block (spec 7.17.2): the box filter passes that run (radius
/// not 0) and the projection weights.
struct SgrK {
    pass: [Option<BoxK>; 2],
    w0: i32,
    w1: i32,
    w2: i32,
    max: i32,
}

impl SgrK {
    fn new(bd: usize, set: usize, xqd: &[i8; 2]) -> Option<Self> {
        let p = SGR_PARAMS.get(set)?;
        let (r0, r1) = (usize::from(p[0]), usize::from(p[2]));
        let (w0, w1) = (i32::from(xqd[0]), i32::from(xqd[1]));
        Some(Self {
            pass: [
                (r0 != 0).then(|| BoxK::new(bd, r0, i64::from(p[1]), true)),
                (r1 != 0).then(|| BoxK::new(bd, r1, i64::from(p[3]), false)),
            ],
            w0,
            w1,
            w2: (1 << SGRPROJ_PRJ_BITS) - w0 - w1,
            max: (1 << bd) - 1,
        })
    }
}

/// Starts of the `lanes`-wide chunks that cover `0..n` (`n >= lanes`): the last chunk ends at `n`
/// and may overlap the one before it.
fn starts(n: usize, lanes: usize) -> impl Iterator<Item = usize> {
    (0..n.div_ceil(lanes)).map(move |k| (k * lanes).min(n - lanes))
}

/// The output of the block at (`x`, `y`) of `dst` (its samples from (`x`, `y`) to the end of its
/// last row) if the kernels with `lanes`-wide vectors take the block: `lanes <= w <= BLOCK_W`,
/// `1 <= h <= BLOCK_H`, a bit depth of 8, 10 or 12, a whole source window, every row in `dst`.
fn block_out<'a>(
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    lanes: usize,
    dst: &'a mut Plane,
    x: usize,
    y: usize,
) -> Option<&'a mut [u16]> {
    if !(lanes..=BLOCK_W).contains(&w)
        || !(1..=BLOCK_H).contains(&h)
        || !matches!(bd, 8 | 10 | 12)
        || win.len() < (w + 6) * (h + 6)
        || x.checked_add(w)? > dst.stride
    {
        return None;
    }
    let start = y.checked_mul(dst.stride)?.checked_add(x)?;
    let end = (h - 1).checked_mul(dst.stride)?.checked_add(w)?;
    dst.px.get_mut(start..start.checked_add(end)?)
}

/// The Wiener filter (`lr::wiener_filter`) of the `w` x `h` block at (`x`, `y`) of `dst`, from
/// its source window `win`; `coef` is `LrWiener[ plane ][ unitRow ][ unitCol ]`.
#[allow(unsafe_code)]
pub(crate) fn wiener(
    simd: Simd,
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    coef: &[[i8; 3]; 2],
    dst: &mut Plane,
    x: usize,
    y: usize,
) {
    let stride = dst.stride;
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(simd.tier(), Tier::Avx2 | Tier::Avx512)
            && let Some(out) = block_out(win, w, h, bd, 8, dst, x, y)
        {
            // SAFETY: `simd` proves that this CPU has AVX2 (only `simd::available` and
            // `simd::best` make a `Simd`, after runtime detection; the Avx512 tier includes it).
            // `block_out` checked the sizes the kernel indexes with.
            unsafe { avx2::wiener(win, w, h, &WienerK::new(bd, coef), out, stride) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd.tier() == Tier::Neon
            && let Some(out) = block_out(win, w, h, bd, 4, dst, x, y)
        {
            // SAFETY: `simd` proves that this CPU has NEON (only `simd::available` and
            // `simd::best` make a `Simd`, after runtime detection). `block_out` checked the
            // sizes the kernel indexes with.
            unsafe { neon::wiener(win, w, h, &WienerK::new(bd, coef), out, stride) };
            return;
        }
    }
    wiener_filter(win, w, h, bd, coef, dst, x, y);
}

/// The self guided filter (`lr::self_guided_filter`) of the `w` x `h` block at (`x`, `y`) of
/// `dst`, from its source window `win`, with `LrSgrSet` `set` and `LrSgrXqd` `xqd`.
#[allow(unsafe_code)]
pub(crate) fn self_guided(
    simd: Simd,
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    set: usize,
    xqd: &[i8; 2],
    dst: &mut Plane,
    x: usize,
    y: usize,
) {
    let stride = dst.stride;
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(simd.tier(), Tier::Avx2 | Tier::Avx512)
            && let Some(k) = SgrK::new(bd, set, xqd)
            && let Some(out) = block_out(win, w, h, bd, 8, dst, x, y)
        {
            // SAFETY: `simd` proves that this CPU has AVX2 (only `simd::available` and
            // `simd::best` make a `Simd`, after runtime detection; the Avx512 tier includes it).
            // `block_out` checked the sizes the kernel indexes with.
            unsafe { avx2::self_guided(win, w, h, &k, out, stride) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd.tier() == Tier::Neon
            && let Some(k) = SgrK::new(bd, set, xqd)
            && let Some(out) = block_out(win, w, h, bd, 4, dst, x, y)
        {
            // SAFETY: `simd` proves that this CPU has NEON (only `simd::available` and
            // `simd::best` make a `Simd`, after runtime detection). `block_out` checked the
            // sizes the kernel indexes with.
            unsafe { neon::self_guided(win, w, h, &k, out, stride) };
            return;
        }
    }
    self_guided_filter(win, w, h, bd, set, xqd, dst, x, y);
}

/// Runs both filters on the `w` x `h` block at (`x`, 0) of a plane `x + w` samples wide, from the
/// source window `win`, on every tier, and asserts that each output equals the scalar one (the
/// unit tests below and `fuzzing::lr`).
#[cfg(any(test, feature = "fuzzing"))]
pub(crate) fn cross_check(
    win: &[i32],
    w: usize,
    h: usize,
    bd: usize,
    coef: &[[i8; 3]; 2],
    set: usize,
    xqd: &[i8; 2],
    x: usize,
) {
    use crate::ledger::Ledger;
    use std::sync::Arc;
    let ledger = Arc::new(Ledger::default());
    if ledger.reset(1 << 20, 0).is_err() {
        return;
    }
    let (Ok(mut want), Ok(mut got)) =
        (Plane::new(x + w, h, &ledger), Plane::new(x + w, h, &ledger))
    else {
        return;
    };
    for sgr in [false, true] {
        let run = |simd: Simd, p: &mut Plane| {
            p.px.fill(0x5555);
            if sgr {
                self_guided(simd, win, w, h, bd, set, xqd, p, x, 0);
            } else {
                wiener(simd, win, w, h, bd, coef, p, x, 0);
            }
        };
        run(Simd::scalar(), &mut want);
        for simd in super::available() {
            run(simd, &mut got);
            assert_eq!(
                got.px,
                want.px,
                "{:?} sgr {sgr} {w}x{h} bd {bd} x {x} set {set} coef {coef:?} xqd {xqd:?}",
                simd.tier()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }

        fn in_range(&mut self, lo: i8, hi: i8) -> i8 {
            (i32::from(lo) + self.below((i32::from(hi) - i32::from(lo) + 1) as usize) as i32) as i8
        }
    }

    /// How a test window is filled.
    #[derive(Clone, Copy)]
    enum Fill {
        /// Uniform samples (high variance: `z` saturates).
        Random,
        /// Mid grey plus noise of this amplitude at 8 bits (`z` in its middle range).
        Noise(i32),
        Zero,
        Max,
        /// 0 and max alternating: the largest variance.
        Checker,
    }

    const BDS: [usize; 3] = [8, 10, 12];
    /// `Wiener_Taps_Min`, `Wiener_Taps_Max`, `Sgrproj_Xqd_Min`, `Sgrproj_Xqd_Max`.
    const TAPS_MIN: [i8; 3] = [-5, -23, -17];
    const TAPS_MAX: [i8; 3] = [10, 8, 46];
    const XQD_MIN: [i8; 2] = [-96, -32];
    const XQD_MAX: [i8; 2] = [31, 95];

    /// One block: taps and weights in the spec ranges or anywhere in `i8`.
    fn case(rng: &mut Rng, w: usize, h: usize, bd: usize, x: usize, fill: Fill, set: usize) {
        let max = (1i32 << bd) - 1;
        let ww = w + 6;
        let mut win = Vec::with_capacity(ww * (h + 6));
        for k in 0..ww * (h + 6) {
            win.push(match fill {
                Fill::Random => (rng.next_u64() as i32) & max,
                Fill::Noise(amp) => {
                    let amp = amp << (bd - 8);
                    let d = rng.below(2 * amp as usize + 1) as i32 - amp;
                    ((max + 1) / 2 + d).clamp(0, max)
                }
                Fill::Zero => 0,
                Fill::Max => max,
                Fill::Checker => {
                    if (k % ww + k / ww) & 1 == 0 {
                        0
                    } else {
                        max
                    }
                }
            });
        }
        let full = rng.next_u64() & 1 == 0;
        let mut coef = [[0i8; 3]; 2];
        for row in &mut coef {
            for t in 0..3 {
                row[t] = if full {
                    rng.next_u64() as i8
                } else {
                    rng.in_range(TAPS_MIN[t], TAPS_MAX[t])
                };
            }
        }
        let mut xqd = [0i8; 2];
        for t in 0..2 {
            xqd[t] = if full {
                rng.next_u64() as i8
            } else {
                rng.in_range(XQD_MIN[t], XQD_MAX[t])
            };
        }
        cross_check(&win, w, h, bd, &coef, set, &xqd, x);
    }

    #[test]
    fn starts_cover_the_row() {
        assert_eq!(starts(8, 8).collect::<Vec<_>>(), [0]);
        assert_eq!(starts(9, 8).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(starts(17, 8).collect::<Vec<_>>(), [0, 8, 9]);
        assert_eq!(starts(16, 4).collect::<Vec<_>>(), [0, 4, 8, 12]);
    }

    #[test]
    fn edge_sizes() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut k = 0;
        for bd in BDS {
            for w in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64] {
                for h in [1, 2, 3, 8, 64] {
                    case(&mut rng, w, h, bd, k % 13, Fill::Noise(12), k % 16);
                    k += 1;
                }
            }
        }
    }

    #[test]
    fn extreme_samples() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for bd in BDS {
            for fill in [Fill::Zero, Fill::Max, Fill::Checker, Fill::Random] {
                for (w, h) in [(64, 16), (9, 5), (17, 2)] {
                    for set in 0..16 {
                        case(&mut rng, w, h, bd, set % 5, fill, set);
                    }
                }
            }
        }
    }

    #[test]
    fn random_blocks() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..300 {
            let bd = BDS[rng.below(3)];
            let (w, h) = (1 + rng.below(64), 1 + rng.below(64));
            let fill = if rng.below(4) == 0 {
                Fill::Random
            } else {
                Fill::Noise(1 + rng.below(40) as i32)
            };
            let (x, set) = (rng.below(16), rng.below(16));
            case(&mut rng, w, h, bd, x, fill, set);
        }
    }
}
