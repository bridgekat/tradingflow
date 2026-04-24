//! Versioned ring buffer for operator output values.
//!
//! Each operator node stores its per-tick output in a
//! [`VersionedRing`].  Entries are pushed as they're computed and
//! popped from the front when the global low-water mark advances
//! past them (no downstream node will ever read them again).

use std::collections::VecDeque;

/// One entry in a versioned ring.
struct VersionedEntry {
    tick: usize,
    ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

/// A ring buffer of per-tick output values for one operator node.
pub struct VersionedRing {
    entries: VecDeque<VersionedEntry>,
}

impl VersionedRing {
    /// Create an empty ring.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    /// Push a value for the given tick.
    pub fn push(&mut self, tick: usize, ptr: *mut u8, drop_fn: unsafe fn(*mut u8)) {
        self.entries.push_back(VersionedEntry {
            tick,
            ptr,
            drop_fn,
        });
    }

    /// Look up the value for a tick.  Linear scan; the ring is
    /// kept small by `pop_below`.
    pub fn get(&self, tick: usize) -> Option<*const u8> {
        for entry in &self.entries {
            if entry.tick == tick {
                return Some(entry.ptr);
            }
        }
        None
    }

    /// Drop and remove all entries with `tick < cutoff`.
    pub fn pop_below(&mut self, cutoff: usize) {
        while let Some(front) = self.entries.front() {
            if front.tick < cutoff {
                let entry = self.entries.pop_front().unwrap();
                unsafe { (entry.drop_fn)(entry.ptr) };
            } else {
                break;
            }
        }
    }
}

impl Drop for VersionedRing {
    fn drop(&mut self) {
        while let Some(entry) = self.entries.pop_front() {
            unsafe { (entry.drop_fn)(entry.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_pop() {
        let mut ring = VersionedRing::new();
        let val0 = Box::into_raw(Box::new(42u32));
        let val1 = Box::into_raw(Box::new(99u32));
        unsafe fn drop_u32(p: *mut u8) {
            unsafe { drop(Box::from_raw(p as *mut u32)) };
        }
        ring.push(0, val0 as *mut u8, drop_u32);
        ring.push(1, val1 as *mut u8, drop_u32);

        assert!(ring.get(0).is_some());
        assert!(ring.get(1).is_some());
        assert!(ring.get(2).is_none());

        ring.pop_below(1);
        assert!(ring.get(0).is_none());
        assert!(ring.get(1).is_some());

        ring.pop_below(2);
        assert!(ring.get(1).is_none());
    }
}
