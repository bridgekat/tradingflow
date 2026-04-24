//! Auto-managed output queue with Arc-refcount garbage collection.
//!
//! An [`OutputQueue<T>`] is a bounded `VecDeque<Arc<Slot<T>>>` that
//! collects values committed by a single writer and served to any number
//! of concurrent readers.  Readers clone the `Arc` to hold a slot alive
//! for the duration of a downstream compute; the queue reclaims front
//! slots whose `Arc::strong_count` has returned to 1 (no live readers)
//! whenever the writer next pushes.
//!
//! # Read semantics
//!
//! [`OutputQueue::latest_at_or_before`] returns the newest slot whose
//! timestamp is `<= requested_ts`.  This matches the legacy runtime's
//! semantic of "read the upstream's current value," generalised so that
//! concurrent `t`s in flight don't clobber each other's observations.
//!
//! # Retention invariant
//!
//! The queue always keeps at least its most recent slot, even if its
//! `Arc::strong_count == 1` — future readers (at later `ts`) may still
//! need to read "the latest value up to now."  Only slots with a newer
//! sibling behind them are eligible for retirement.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use super::data::Instant;

/// One committed value in the output queue.
pub struct Slot<T> {
    pub ts: Instant,
    pub value: T,
}

impl<T> Slot<T> {
    #[inline]
    pub fn new(ts: Instant, value: T) -> Self {
        Self { ts, value }
    }
}

struct Inner<T> {
    slots: VecDeque<Arc<Slot<T>>>,
    /// Closed flag set on scenario shutdown so parked writers can bail.
    closed: bool,
}

/// Bounded, Arc-refcounted output queue with auto-retire.
pub struct OutputQueue<T: Send + Sync + 'static> {
    inner: Mutex<Inner<T>>,
    /// Signalled on slot retirement (writer park-on-full) and on push
    /// (reader park-on-miss if implemented later).
    cv: Condvar,
    cap: usize,
}

impl<T: Send + Sync + 'static> OutputQueue<T> {
    /// Create a queue with the given capacity (`cap >= 1`).  Capacity
    /// bounds the in-flight pipeline depth: the writer parks when the
    /// queue is full and readers have not released enough slots.
    pub fn new(cap: usize) -> Self {
        assert!(cap >= 1, "OutputQueue capacity must be >= 1");
        Self {
            inner: Mutex::new(Inner {
                slots: VecDeque::with_capacity(cap),
                closed: false,
            }),
            cv: Condvar::new(),
            cap,
        }
    }

    /// Seed the queue with an initial slot at `Instant::MIN`.  Used at
    /// scenario init time so downstreams can always find *some* value to
    /// read (typically the init-returned output template).
    pub fn seed_initial(&self, value: T) {
        let mut g = self.inner.lock().unwrap();
        debug_assert!(
            g.slots.is_empty(),
            "seed_initial called on non-empty queue"
        );
        g.slots
            .push_back(Arc::new(Slot::new(Instant::MIN, value)));
    }

    /// Commit a new value at `ts`.  Blocks while the queue is at capacity
    /// **and** the oldest slot still has a live reader; otherwise evicts
    /// the oldest slot and pushes.  Returns `true` on commit, `false` if
    /// the queue was closed while blocked.
    ///
    /// Called by the single writer for this output.
    pub fn push(&self, ts: Instant, value: T) -> bool {
        let mut g = self.inner.lock().unwrap();
        loop {
            if g.closed {
                return false;
            }
            if g.slots.len() < self.cap {
                break;
            }
            // At cap: evict the oldest if no live reader still holds it.
            let can_evict = g
                .slots
                .front()
                .map(|a| Arc::strong_count(a) == 1)
                .unwrap_or(false);
            if can_evict {
                g.slots.pop_front();
                // Loop again — len may still be at cap.
            } else {
                // Park until a reader releases the front slot.
                g = self.cv.wait(g).unwrap();
            }
        }
        g.slots.push_back(Arc::new(Slot::new(ts, value)));
        drop(g);
        self.cv.notify_all();
        true
    }

    /// Return the newest slot whose `ts <= requested_ts`, cloning the
    /// `Arc` to keep it alive for the reader's compute.  Returns `None`
    /// if no such slot exists (should not happen post-[`seed_initial`]).
    pub fn latest_at_or_before(&self, requested_ts: Instant) -> Option<Arc<Slot<T>>> {
        let g = self.inner.lock().unwrap();
        // Scan from the back (newest first).
        for slot in g.slots.iter().rev() {
            if slot.ts <= requested_ts {
                return Some(Arc::clone(slot));
            }
        }
        None
    }

    /// Current number of slots held by the queue (for tests / backpressure
    /// introspection).
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().slots.len()
    }

    /// Capacity of the queue.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// No-op retire kept for API compatibility with the scheduler's
    /// lifecycle hooks.  Eviction happens lazily inside [`push`] when
    /// the queue reaches capacity (oldest slot evicted if it has no live
    /// reader, else writer parks).  Proactive refcount-driven retirement
    /// was abandoned because it could drop a slot that downstream
    /// readers would shortly look up via
    /// [`latest_at_or_before`](Self::latest_at_or_before).
    pub fn retire(&self) {}

    #[cfg(test)]
    fn force_retire_for_tests(&self) {
        let mut g = self.inner.lock().unwrap();
        while g.slots.len() > 1 {
            let front_refs = match g.slots.front() {
                Some(a) => Arc::strong_count(a),
                None => break,
            };
            if front_refs == 1 {
                g.slots.pop_front();
            } else {
                break;
            }
        }
        drop(g);
        self.cv.notify_all();
    }

    /// Mark the queue closed; wakes any parked writer so they can exit.
    pub fn close(&self) {
        let mut g = self.inner.lock().unwrap();
        g.closed = true;
        drop(g);
        self.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn seed_and_read() {
        let q: OutputQueue<i32> = OutputQueue::new(4);
        q.seed_initial(7);
        let got = q.latest_at_or_before(ts(100)).unwrap();
        assert_eq!(got.value, 7);
        assert_eq!(got.ts, Instant::MIN);
    }

    #[test]
    fn push_and_read_latest() {
        let q: OutputQueue<i32> = OutputQueue::new(4);
        q.seed_initial(0);
        assert!(q.push(ts(10), 10));
        assert!(q.push(ts(20), 20));
        assert_eq!(q.latest_at_or_before(ts(5)).unwrap().value, 0);
        assert_eq!(q.latest_at_or_before(ts(10)).unwrap().value, 10);
        assert_eq!(q.latest_at_or_before(ts(15)).unwrap().value, 10);
        assert_eq!(q.latest_at_or_before(ts(100)).unwrap().value, 20);
    }

    #[test]
    fn retire_is_no_op() {
        // retire() under the lazy-retire policy is a no-op; eviction
        // happens inside push() only under capacity pressure.
        let q: OutputQueue<i32> = OutputQueue::new(8);
        q.seed_initial(0);
        q.push(ts(10), 10);
        q.push(ts(20), 20);
        q.retire();
        assert_eq!(q.len(), 3);
        q.force_retire_for_tests();
        assert_eq!(q.len(), 1);
        // Only slot 20 remains — ts(5) now has no at-or-before match.
        assert!(q.latest_at_or_before(ts(5)).is_none());
        assert_eq!(q.latest_at_or_before(ts(30)).unwrap().value, 20);
    }

    #[test]
    fn push_evicts_oldest_at_cap_when_no_live_reader() {
        let q: OutputQueue<i32> = OutputQueue::new(3);
        q.seed_initial(0);
        q.push(ts(10), 10);
        q.push(ts(20), 20);
        // Now at cap=3.  Next push evicts seed (refcount==1).
        assert_eq!(q.len(), 3);
        q.push(ts(30), 30);
        assert_eq!(q.len(), 3);
        // Seed is gone, so ts(5) has no match.
        assert!(q.latest_at_or_before(ts(5)).is_none());
        assert_eq!(q.latest_at_or_before(ts(15)).unwrap().value, 10);
        assert_eq!(q.latest_at_or_before(ts(30)).unwrap().value, 30);
    }

    #[test]
    fn writer_parks_on_full_and_wakes_after_retire() {
        let q = Arc::new(OutputQueue::<i32>::new(2));
        q.seed_initial(0);
        q.push(ts(10), 10);
        // Now len == 2 == cap.  A reader on slot 10 pins it alive.
        let r = q.latest_at_or_before(ts(10)).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let q2 = Arc::clone(&q);
        let b2 = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            b2.wait();
            // This should block until `r` is dropped below.
            assert!(q2.push(ts(20), 20));
        });
        barrier.wait();
        // Give the writer a moment to block.
        thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(q.len(), 2);
        drop(r);
        // Dropping `r` alone doesn't retire — retire is invoked on push.
        // The writer's push will trigger a retire that frees the front
        // slot and admits the new one.
        writer.join().unwrap();
        assert!(q.len() <= 2);
        assert_eq!(q.latest_at_or_before(ts(25)).unwrap().value, 20);
    }

    #[test]
    fn out_of_order_reader_drops() {
        let q = Arc::new(OutputQueue::<i32>::new(8));
        q.seed_initial(-1);
        for t in 1..=5 {
            q.push(ts(t), (t as i32) * 10);
        }
        // Three readers, finishing in reverse order.
        let r1 = q.latest_at_or_before(ts(1)).unwrap();
        let r2 = q.latest_at_or_before(ts(3)).unwrap();
        let r3 = q.latest_at_or_before(ts(5)).unwrap();
        drop(r3);
        q.force_retire_for_tests();
        // r1 pins slot 1; retire pops seed (refcount 1) then stops at slot 1.
        // Remaining: 5 slots.
        assert_eq!(q.len(), 5);
        drop(r2);
        q.force_retire_for_tests();
        assert_eq!(q.len(), 5); // still pinned by r1
        drop(r1);
        q.force_retire_for_tests();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn close_wakes_blocked_writer() {
        let q = Arc::new(OutputQueue::<i32>::new(1));
        q.seed_initial(0);
        // cap=1, seeded → full.  A live reader keeps the slot.
        let _r = q.latest_at_or_before(ts(0)).unwrap();

        let q2 = Arc::clone(&q);
        let handle = thread::spawn(move || q2.push(ts(1), 1));
        thread::sleep(std::time::Duration::from_millis(30));
        q.close();
        let pushed = handle.join().unwrap();
        assert!(!pushed, "push should return false after close()");
    }
}
