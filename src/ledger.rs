//! The working memory of one decoder: every buffer that scales with the frame size is charged
//! before it is allocated, by all threads together, and released when its guard drops.
#![allow(dead_code)]

use crate::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};

/// Booked at the start for the small allocations that do not scale with the input and are not
/// charged one by one (headers, tile lists). A `max_alloc` below this
/// is `Error::AllocLimit` at once.
pub(crate) const SMALL: usize = 1024;

/// Bytes booked against `Limits::max_alloc`.
#[derive(Debug, Default)]
pub(crate) struct Ledger {
    max: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl Ledger {
    /// Starts a decode: `max` bytes allowed, `preload` of them already taken by the caller, plus
    /// [`SMALL`] for the small allocations that are not charged one by one.
    pub(crate) fn reset(&self, max: usize, preload: usize) -> Result<(), Error> {
        self.max.store(max as u64, Ordering::Relaxed);
        self.used.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.reserve(preload.checked_add(SMALL).ok_or(Error::AllocLimit)?)
    }

    /// What may still be charged.
    pub(crate) fn left(&self) -> usize {
        let free = self
            .max
            .load(Ordering::Relaxed)
            .saturating_sub(self.used.load(Ordering::Relaxed));
        usize::try_from(free).unwrap_or(usize::MAX)
    }

    /// The most bytes booked at once.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Books `n` bytes until they are given back with [`Ledger::release`].
    pub(crate) fn reserve(&self, n: usize) -> Result<(), Error> {
        let n = n as u64;
        let max = self.max.load(Ordering::Relaxed);
        let r = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |u| {
                u.checked_add(n).filter(|&t| t <= max)
            });
        match r {
            Ok(u) => {
                self.peak.fetch_max(u + n, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => Err(Error::AllocLimit),
        }
    }

    /// Books `n` bytes until the returned guard drops.
    #[allow(dead_code)]
    pub(crate) fn charge(&self, n: usize) -> Result<Held<'_>, Error> {
        self.reserve(n)?;
        Ok(Held { l: self, n })
    }

    /// Gives back `n` bytes booked with [`Ledger::reserve`], for a buffer that was freed.
    pub(crate) fn release(&self, n: usize) {
        self.used.fetch_sub(n as u64, Ordering::AcqRel);
    }
}

/// Bytes booked in a [`Ledger`], released on drop.
#[derive(Debug)]
pub(crate) struct Held<'a> {
    l: &'a Ledger,
    n: usize,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.l.release(self.n);
    }
}

/// Sets `v` to `n` zero elements and books their bytes for the rest of the call (the caller
/// releases them when the buffer is freed). A buffer that is already big enough allocates
/// nothing. Works for `Vec<u8>` and `Vec<u16>` alike.
pub(crate) fn zeroed<T: Copy + Default>(
    v: &mut Vec<T>,
    n: usize,
    ledger: &Ledger,
) -> Result<(), Error> {
    let bytes = n.checked_mul(size_of::<T>()).ok_or(Error::AllocLimit)?;
    ledger.reserve(bytes)?;
    v.clear();
    if n > v.capacity() && v.try_reserve_exact(n).is_err() {
        ledger.release(bytes);
        return Err(Error::OutOfMemory);
    }
    v.resize(n, T::default());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_release_and_peak() {
        let l = Ledger::default();
        l.reset(SMALL + 100, 0).unwrap();
        let a = l.charge(60).unwrap();
        assert_eq!(l.charge(50).unwrap_err(), Error::AllocLimit);
        assert_eq!(l.left(), 40);
        drop(a);
        assert_eq!(l.left(), 100);
        let _b = l.charge(100).unwrap();
        assert_eq!(l.peak(), SMALL as u64 + 100);
    }

    #[test]
    fn reset_preload_and_zeroed() {
        let l = Ledger::default();
        assert_eq!(l.reset(SMALL + 10, 11).unwrap_err(), Error::AllocLimit);
        l.reset(SMALL + 12, 4).unwrap();
        assert_eq!(l.left(), 8);
        let mut v: Vec<u16> = Vec::new();
        assert_eq!(zeroed(&mut v, 5, &l).unwrap_err(), Error::AllocLimit);
        zeroed(&mut v, 4, &l).unwrap();
        assert_eq!(v, [0u16; 4]);
        assert_eq!(l.left(), 0);
        let mut b: Vec<u8> = vec![9; 3];
        l.release(8);
        zeroed(&mut b, 3, &l).unwrap();
        assert_eq!(b, [0u8; 3]);
    }
}
