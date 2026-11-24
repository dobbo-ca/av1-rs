//! The working memory of one decoder: every buffer that scales with the frame size is charged
//! before it is allocated, by all threads together, and released when its guard drops.
//!
//! The ledger also keeps a pool of freed buffers (sample planes, output planes, film grain
//! buffers and empty plane lists), so the next frame of the same shape reuses them instead of
//! allocating. A pooled buffer stays charged once, as it is still memory the decoder holds; a
//! charge that does not fit empties the pool first (releasing its bytes) and tries again, so the
//! pool never makes a decode fail that would fit without it.
#![allow(dead_code)]

use crate::error::Error;
use crate::frame::Plane;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Booked at the start for the small allocations that do not scale with the input and are not
/// charged one by one (headers, tile lists). A `max_alloc` below this
/// is `Error::AllocLimit` at once.
pub(crate) const SMALL: usize = 1024;

/// Entries a pool list reserves on first use, so returning buffers does not grow it frame by
/// frame (a frame uses about 20 buffers).
const POOL_SLOTS: usize = 64;

/// Freed buffers kept for reuse, each with the bytes it is charged for.
#[derive(Debug, Default)]
pub(crate) struct Pool {
    u8s: Vec<(Vec<u8>, usize)>,
    u16s: Vec<(Vec<u16>, usize)>,
    i32s: Vec<(Vec<i32>, usize)>,
    /// Empty plane lists (not charged: a few words each).
    sets: Vec<Vec<Plane>>,
    /// Bytes charged for the buffers above.
    bytes: u64,
}

/// An element type the pool keeps buffers of.
pub(crate) trait Pooled: Copy + Default {
    fn bin(p: &mut Pool) -> &mut Vec<(Vec<Self>, usize)>;
}

impl Pooled for u8 {
    fn bin(p: &mut Pool) -> &mut Vec<(Vec<u8>, usize)> {
        &mut p.u8s
    }
}

impl Pooled for u16 {
    fn bin(p: &mut Pool) -> &mut Vec<(Vec<u16>, usize)> {
        &mut p.u16s
    }
}

impl Pooled for i32 {
    fn bin(p: &mut Pool) -> &mut Vec<(Vec<i32>, usize)> {
        &mut p.i32s
    }
}

/// Bytes booked against `Limits::max_alloc`.
#[derive(Debug, Default)]
pub(crate) struct Ledger {
    max: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
    pool: Mutex<Pool>,
}

impl Ledger {
    /// Starts a decode: `max` bytes allowed, `preload` of them already taken by the caller, plus
    /// [`SMALL`] for the small allocations that are not charged one by one.
    pub(crate) fn reset(&self, max: usize, preload: usize) -> Result<(), Error> {
        let _old = std::mem::take(&mut *self.lock());
        self.max.store(max as u64, Ordering::Relaxed);
        self.used.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.reserve(preload.checked_add(SMALL).ok_or(Error::AllocLimit)?)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What may still be charged (pooled buffers give way to a charge, so they count as free).
    pub(crate) fn left(&self) -> usize {
        let pooled = self.lock().bytes;
        let used = self.used.load(Ordering::Relaxed).saturating_sub(pooled);
        let free = self.max.load(Ordering::Relaxed).saturating_sub(used);
        usize::try_from(free).unwrap_or(usize::MAX)
    }

    /// The most bytes booked at once.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Books `n` bytes until they are given back with [`Ledger::release`]. When they do not fit,
    /// the pool is emptied first.
    pub(crate) fn reserve(&self, n: usize) -> Result<(), Error> {
        match self.book(n) {
            Err(_) if self.evict() => self.book(n),
            r => r,
        }
    }

    /// Drops every pooled buffer and releases its bytes; false when there were none.
    fn evict(&self) -> bool {
        let old = std::mem::take(&mut *self.lock());
        if old.bytes == 0 {
            return false;
        }
        self.used.fetch_sub(old.bytes, Ordering::AcqRel);
        true
    }

    /// An empty `Vec` with room for `n` elements and the bytes charged for it: a pooled buffer
    /// (the smallest that is large enough) or a new one, charged first.
    pub(crate) fn take<T: Pooled>(&self, n: usize) -> Result<(Vec<T>, usize), Error> {
        if n == 0 {
            return Ok((Vec::new(), 0));
        }
        let found = {
            let mut p = self.lock();
            let bin = T::bin(&mut p);
            let best = (0..bin.len())
                .filter(|&i| bin[i].0.capacity() >= n)
                .min_by_key(|&i| bin[i].0.capacity());
            let found = best.map(|i| bin.swap_remove(i));
            if let Some((_, c)) = &found {
                p.bytes -= *c as u64;
            }
            found
        };
        if let Some((mut v, charged)) = found {
            v.clear();
            return Ok((v, charged));
        }
        let bytes = n.checked_mul(size_of::<T>()).ok_or(Error::AllocLimit)?;
        self.reserve(bytes)?;
        let mut v = Vec::new();
        if v.try_reserve_exact(n).is_err() {
            self.release(bytes);
            return Err(Error::OutOfMemory);
        }
        Ok((v, bytes))
    }

    /// `n` zeros in a buffer from [`Ledger::take`], back in the pool when it drops.
    pub(crate) fn zeros<T: Pooled>(&self, n: usize) -> Result<Buf<'_, T>, Error> {
        let (mut v, charged) = self.take(n)?;
        v.resize(n, T::default());
        Ok(Buf {
            v,
            charged,
            ledger: self,
        })
    }

    /// Keeps `v`, charged `charged` bytes, for a later [`Ledger::take`].
    pub(crate) fn give<T: Pooled>(&self, v: Vec<T>, charged: usize) {
        if v.capacity() == 0 {
            self.release(charged);
            return;
        }
        let mut p = self.lock();
        p.bytes += charged as u64;
        let bin = T::bin(&mut p);
        if bin.capacity() == 0 {
            let _ = bin.try_reserve(POOL_SLOTS);
        }
        bin.push((v, charged));
    }

    /// Charges `v` (a buffer handed back by the caller) and keeps it; drops it when it does not
    /// fit in `max_alloc` (without emptying the pool for it).
    pub(crate) fn adopt<T: Pooled>(&self, v: Vec<T>) {
        let bytes = v.capacity().saturating_mul(size_of::<T>());
        if v.capacity() > 0 && self.book(bytes).is_ok() {
            self.give(v, bytes);
        }
    }

    /// An empty plane list with room for `n` planes, pooled or new.
    pub(crate) fn planes(&self, n: usize) -> Result<Vec<Plane>, Error> {
        let mut v = self.lock().sets.pop().unwrap_or_default();
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        Ok(v)
    }

    /// Frees the planes of `v` (into the pool) and keeps the empty list.
    pub(crate) fn put_planes(&self, mut v: Vec<Plane>) {
        v.clear();
        let mut p = self.lock();
        if p.sets.capacity() == 0 {
            let _ = p.sets.try_reserve(POOL_SLOTS);
        }
        p.sets.push(v);
    }

    /// Books `n` bytes, or `AllocLimit` when they do not fit.
    fn book(&self, n: usize) -> Result<(), Error> {
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

/// A charged buffer from [`Ledger::zeros`]; it reads as a slice and goes back to the pool on drop.
pub(crate) struct Buf<'a, T: Pooled> {
    v: Vec<T>,
    charged: usize,
    ledger: &'a Ledger,
}

impl<T: Pooled> std::ops::Deref for Buf<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.v
    }
}

impl<T: Pooled> std::ops::DerefMut for Buf<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.v
    }
}

impl<T: Pooled> Drop for Buf<'_, T> {
    fn drop(&mut self) {
        self.ledger.give(std::mem::take(&mut self.v), self.charged);
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

    #[test]
    fn pooled_buffers_are_reused_stay_charged_and_give_way() {
        let l = Ledger::default();
        l.reset(SMALL + 100, 0).unwrap();
        let (v, c) = l.take::<u16>(20).unwrap();
        assert_eq!(c, 40);
        l.give(v, c);
        // Pooled bytes are still booked, but count as free.
        assert_eq!(l.left(), 100);
        assert_eq!(l.peak(), SMALL as u64 + 40);
        // A smaller request reuses the buffer and its charge.
        let (v, c) = l.take::<u16>(10).unwrap();
        assert!(v.capacity() >= 20 && v.is_empty() && c == 40);
        let z = l.zeros::<u16>(5).unwrap();
        assert_eq!(*z, [0u16; 5]);
        drop(z);
        l.give(v, c);
        // A charge that only fits without the pool empties it.
        l.reserve(100).unwrap();
        assert_eq!(l.left(), 0);
        l.release(100);
        // A handed back buffer is charged; one that does not fit is dropped.
        l.adopt(vec![7u8; 60]);
        assert_eq!(l.left(), 100);
        l.adopt(vec![7u8; 60]);
        let (v, c) = l.take::<u8>(60).unwrap();
        assert_eq!((v.capacity(), c), (60, 60));
        assert_eq!(l.take::<u8>(41).unwrap_err(), Error::AllocLimit);
        l.give(v, c);
        // Plane lists come back empty.
        let p = l.planes(3).unwrap();
        assert!(p.capacity() >= 3);
        l.put_planes(p);
        assert!(l.planes(2).unwrap().capacity() >= 3);
    }
}
