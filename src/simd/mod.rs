//! Runtime SIMD tier selection (docs/safety.md).
//!
//! The scalar code of each kernel is the oracle and the fallback: it is always compiled, and
//! every SIMD kernel must give the same output, bit for bit. The tiers the CPU has are detected
//! once. `AV1_SIMD` (`scalar`, `avx2`, `avx512`, `neon`; read once, empty or unset = best)
//! forces a lower tier; a tier the CPU does not have falls back to the best one below it.
//!
//! A [`Simd`] value is the proof that its tier was detected: its field is private, and only
//! [`available`] and [`best`] make one. The dispatch fn of each kernel family matches on
//! [`Simd::tier`] and calls the `#[target_feature]` kernel of that tier in an `unsafe` block
//! whose SAFETY comment names this proof. Apart from those dispatch fns, `unsafe` lives only in
//! the load/store helpers of [`x86`] and [`arm`], whose array reference arguments prove the
//! bounds. The kernels themselves are safe `#[target_feature]` fns.

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm;
pub(crate) mod lf;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::sync::OnceLock;

/// A SIMD tier, lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Tier {
    Scalar,
    /// x86_64 AVX2 with FMA, BMI1 and BMI2.
    Avx2,
    /// x86_64 AVX-512 F, BW, VL, DQ, CD (Skylake-X, Zen 4 and later), on top of `Avx2`.
    Avx512,
    /// aarch64 NEON.
    Neon,
}

/// A tier that this CPU has (see the module doc): the only way to reach a SIMD kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Simd(Tier);

impl Simd {
    pub(crate) fn tier(self) -> Tier {
        self.0
    }

    /// The scalar tier, which every CPU has.
    pub(crate) fn scalar() -> Self {
        Self(Tier::Scalar)
    }
}

/// Every tier this CPU can run, lowest first (`Scalar` always).
pub(crate) fn available() -> Vec<Simd> {
    static AVAIL: OnceLock<Vec<Simd>> = OnceLock::new();
    AVAIL.get_or_init(detect).clone()
}

fn detect() -> Vec<Simd> {
    let mut v = vec![Simd(Tier::Scalar)];
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        if has!("avx2") && has!("fma") && has!("bmi1") && has!("bmi2") {
            v.push(Simd(Tier::Avx2));
            if has!("avx512f")
                && has!("avx512bw")
                && has!("avx512vl")
                && has!("avx512dq")
                && has!("avx512cd")
            {
                v.push(Simd(Tier::Avx512));
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push(Simd(Tier::Neon));
        }
    }
    v
}

/// The tier named by `AV1_SIMD`, if any.
fn requested() -> Option<Tier> {
    match std::env::var("AV1_SIMD").ok()?.trim() {
        "scalar" => Some(Tier::Scalar),
        "avx2" => Some(Tier::Avx2),
        "avx512" => Some(Tier::Avx512),
        "neon" => Some(Tier::Neon),
        _ => None,
    }
}

/// The tier the decoder uses: the best available one, capped by `AV1_SIMD`. Resolved once.
pub(crate) fn best() -> Simd {
    static BEST: OnceLock<Simd> = OnceLock::new();
    *BEST.get_or_init(|| {
        let avail = available();
        let cap = requested();
        avail
            .into_iter()
            .rfind(|s| cap.is_none_or(|r| s.0 <= r))
            .unwrap_or(Simd(Tier::Scalar))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CI runners have AVX-512 (x86_64) and NEON (aarch64 under qemu); a lost detection
    /// would silently test only the scalar code.
    #[test]
    fn simd_is_detected() {
        let a: Vec<Tier> = available().iter().map(|s| s.tier()).collect();
        assert_eq!(a[0], Tier::Scalar);
        #[cfg(target_arch = "x86_64")]
        assert!(
            a.contains(&Tier::Avx2) && a.contains(&Tier::Avx512),
            "{a:?}"
        );
        #[cfg(target_arch = "aarch64")]
        assert!(a.contains(&Tier::Neon), "{a:?}");
    }

    #[test]
    fn best_is_available_and_follows_av1_simd() {
        let b = best();
        assert!(available().contains(&b));
        if let Some(r) = requested() {
            assert!(b.tier() <= r);
        }
    }
}
