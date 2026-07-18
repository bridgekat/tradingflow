//! Stress tests for the work-stealing [`Pool`]: every seeded or spawned task
//! must run **exactly once**, and `run` must always return (no lost wakeups, no
//! hangs), across serial / wide / fan-out / mixed shapes with the pool reused
//! many times. These hammer the lazy-recruit + nap/unpark transitions that the
//! graph-level tests only exercise incidentally.
//!
//! Excluded under Miri: the reuse loops spend nearly all interpreted time
//! inside crossbeam's deque/epoch internals (whose reports are third-party
//! noise), and the schedule fuzzing these tests exist for needs real
//! parallelism anyway -- native runs cover it.
#![cfg(not(miri))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use tradingflow_graph::pool::Pool;

fn pool() -> Pool {
    Pool::new(thread::available_parallelism().map_or(1, |n| n.get()))
}

fn zeroed(n: usize) -> Vec<AtomicUsize> {
    (0..n).map(|_| AtomicUsize::new(0)).collect()
}

fn assert_counts(counts: &[AtomicUsize], expected: impl Fn(usize) -> usize) {
    for (i, c) in counts.iter().enumerate() {
        assert_eq!(
            c.load(Ordering::Relaxed),
            expected(i),
            "task {i} ran the wrong number of times"
        );
    }
}

#[test]
fn wide_flat_each_task_runs_once() {
    let mut pool = pool();
    let n = 10_000;
    let counts = zeroed(n);
    for _ in 0..50 {
        counts.iter().for_each(|c| c.store(0, Ordering::Relaxed));
        pool.run(0..n, |i, _| {
            counts[i].fetch_add(1, Ordering::Relaxed);
        });
        assert_counts(&counts, |_| 1);
    }
}

#[test]
fn serial_chain_runs_each_once() {
    let mut pool = pool();
    let n = 5_000usize;
    let counts = zeroed(n);
    for _ in 0..50 {
        counts.iter().for_each(|c| c.store(0, Ordering::Relaxed));
        pool.run([0], |i, scope| {
            counts[i].fetch_add(1, Ordering::Relaxed);
            if i + 1 < n {
                scope.spawn(i + 1);
            }
        });
        assert_counts(&counts, |_| 1);
    }
}

#[test]
fn fanout_tree_runs_each_once() {
    // A single root that fans out: every index in `0..n` is reachable from 0
    // via the binary-tree edges, so each must run exactly once.
    let mut pool = pool();
    let n = 8_000usize;
    let counts = zeroed(n);
    for _ in 0..30 {
        counts.iter().for_each(|c| c.store(0, Ordering::Relaxed));
        pool.run([0], |i, scope| {
            counts[i].fetch_add(1, Ordering::Relaxed);
            let (l, r) = (2 * i + 1, 2 * i + 2);
            if l < n {
                scope.spawn(l);
            }
            if r < n {
                scope.spawn(r);
            }
        });
        assert_counts(&counts, |_| 1);
    }
}

#[test]
fn empty_seed_returns() {
    // No seed: `run` must return immediately without waking anyone.
    let mut pool = pool();
    for _ in 0..1000 {
        pool.run(std::iter::empty(), |_, _: &_| {
            panic!("handler must not run")
        });
    }
}

#[test]
fn handler_panic_propagates_and_pool_survives() {
    // A panicking handler must not deadlock `run` (the pending count is still
    // settled) and must re-raise on the caller; the pool stays reusable after.
    let mut pool = pool();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(0..100, |i, _| {
            assert_ne!(i, 42, "boom");
        });
    }));
    assert!(panicked.is_err(), "panic must propagate out of run");

    // Reuse the same pool: a subsequent clean run still runs every task once.
    let n = 1_000;
    let counts = zeroed(n);
    pool.run(0..n, |i, _| {
        counts[i].fetch_add(1, Ordering::Relaxed);
    });
    assert_counts(&counts, |_| 1);
}

#[test]
fn mixed_shapes_reuse_pool() {
    // Alternate single / wide / serial / fan-out scopes on one pool to stress
    // the recruit/nap transitions between back-to-back scopes of different
    // widths. A double-run (>1) would expose a duplicate-dispatch race; a hang
    // would expose a lost wakeup.
    let mut pool = pool();
    let n = 2_000usize;
    let counts = zeroed(n);
    for round in 0..400usize {
        counts.iter().for_each(|c| c.store(0, Ordering::Relaxed));
        match round % 4 {
            0 => pool.run([0], |i, _| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            }),
            1 => pool.run(0..n, |i, _| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            }),
            2 => pool.run([0], |i, s| {
                counts[i].fetch_add(1, Ordering::Relaxed);
                if i + 1 < n {
                    s.spawn(i + 1);
                }
            }),
            _ => pool.run([0], |i, s| {
                counts[i].fetch_add(1, Ordering::Relaxed);
                let (l, r) = (2 * i + 1, 2 * i + 2);
                if l < n {
                    s.spawn(l);
                }
                if r < n {
                    s.spawn(r);
                }
            }),
        }
        // Round 0 (single seed, no fan-out) touches only task 0; the rest cover
        // every task. Either way, no task may run more than once.
        if round % 4 == 0 {
            assert_counts(&counts, |i| usize::from(i == 0));
        } else {
            assert_counts(&counts, |_| 1);
        }
    }
}
