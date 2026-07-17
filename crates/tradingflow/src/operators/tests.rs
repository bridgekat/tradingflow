//! Engine tests: each graph asserts a known-good output (the constants were
//! originally lifted from the retired reference engine's test suite and are now
//! the canonical expected values), plus a parallel-execution gate that runs the
//! gating / clock / concurrent-write paths under real worker threads.
//!
//! [array-view-refactor] Migrated to the const-rank strided view currency: every
//! array-shaped edge is a `ArrayPort<T, N>` carrying an
//! `ArrayView<'a, T, N>` by value. Sources are owned [`array_cell`]s
//! lending that currency directly (poke via `state_mut` on the source handle,
//! wire via the paired `ArrayPort` handle); outputs are read with
//! `g.view(h).as_slice()` (view edges) or `g.view(h)` (`SeriesView`
//! edges). Arithmetic uses the lowercase free constructors (`add`/`negate`/…);
//! the rank-changers carry explicit out-rank generics.

use super::*;
use crate::data::{Array, ArrayView, Duration, Instant, Retention, SeriesView};
use crate::graph::core::Pool;
use crate::graph::typed::{Builder, NodeHandle, PortHandle, RefSource, ViewSource};
use crate::operators::op::ArrayPort;

fn ts(n: i64) -> Instant {
    Instant::from_nanos(n)
}

/// A rank-`N` view port — the array edge currency. A slice of these handles
/// feeds the carry-join combine operators ([`Stack`] / [`Concat`], whose
/// inputs are `ArrayPorts`) directly — no bridging.
type Vp<const N: usize> = ArrayPort<f64, N>;

/// Push a scalar-array [`array_cell`]; return the source handle (for
/// `state_mut`) and its `ArrayPort` view handle (for wiring).
fn scalar_src(
    b: &mut Builder<Instant>,
    v: f64,
) -> (
    NodeHandle<ViewSource<ArrayValue<f64, 0>, Instant>>,
    PortHandle<Vp<0>>,
) {
    b.source(array_cell(Array::scalar(v)))
}

/// Push a rank-1 array [`array_cell`]; handles as in [`scalar_src`].
fn vec_src(
    b: &mut Builder<Instant>,
    v: Vec<f64>,
) -> (
    NodeHandle<ViewSource<ArrayValue<f64, 1>, Instant>>,
    PortHandle<Vp<1>>,
) {
    b.source(array_cell(Array::from_vec([v.len()], v)))
}

// ===========================================================================
// Sequential execution (Pool::new(0): single-threaded stabilization)
// ===========================================================================

/// `scenario_simple_add`: 10 + 3 == 13.
#[test]
fn simple_add() {
    let mut b = Builder::new(Instant::MIN);
    let (ha, hav) = scalar_src(&mut b, 0.0);
    let (hb, hbv) = scalar_src(&mut b, 0.0);
    let hc = b.segment(add(), (hav, hbv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(ha) = Array::scalar(10.0);
    *g.state_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(hc).as_slice().unwrap(), &[13.0]);
}

/// `scenario_chain`: (2 + 3) * 2 == 10. Exercises a 3-deep cone where the leaf
/// is read by two operators (add and multiply both read `a`).
#[test]
fn chain_add_then_mul() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = scalar_src(&mut b, 0.0);
    let (bb, bv) = scalar_src(&mut b, 0.0);
    let ab = b.segment(add(), (av, bv));
    let out = b.segment(multiply(), (ab, av));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::scalar(2.0);
    *g.state_mut(bb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(out).as_slice().unwrap(), &[10.0]);
}

/// `scenario_record`: record (10+3), (20+7) → series [13, 27] @ [1, 2].
#[test]
fn record_series() {
    let mut b = Builder::new(Instant::MIN);
    let (ha, hav) = scalar_src(&mut b, 0.0);
    let (hb, hbv) = scalar_src(&mut b, 0.0);
    let sum = b.segment(add(), (hav, hbv));
    let rec = b.segment(record(), sum);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.context_mut() = ts(1);
    *g.state_mut(ha) = Array::scalar(10.0);
    *g.state_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    *g.context_mut() = ts(2);
    *g.state_mut(ha) = Array::scalar(20.0);
    *g.state_mut(hb) = Array::scalar(7.0);
    g.stabilize(&mut pool);

    let s: SeriesView<f64, 0> = g.view(rec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.timestamps(), &[ts(1), ts(2)]);
    assert_eq!(s.data(), &[13.0, 27.0]);
}

/// `scenario_run_filter`: source [1,5,2,10]@[1,2,3,4], keep >3 → recorded
/// (2,5),(4,10). THE cutoff proof: a dropped Filter must suppress Record.
#[test]
fn filter_gates_record() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let flt = b.segment(
        filter(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 3.0),
        srcv,
    );
    let rec = b.segment(record(), flt);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 5.0), (3, 2.0), (4, 10.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    let s: SeriesView<f64, 0> = g.view(rec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.data(), &[5.0, 10.0]);
    assert_eq!(s.timestamps(), &[ts(2), ts(4)]);
}

/// `Last(Record(x))` recovers the latest array value.
#[test]
fn last_of_record() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.segment(record(), srcv);
    let lst = b.segment(last(0.0_f64), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.view(lst).as_slice().unwrap(), &[20.0]);
}

/// A retention-bounded `Record` feeding `Lag` and `RollingMean` produces the
/// same outputs as an unbounded one, across front-compaction. Exercises the
/// window-relative addressing: the rolling accumulator and `Lag`'s offset
/// address the `SeriesView` relative to its newest row while the front is
/// dropped underneath them.
#[test]
fn bounded_record_feeds_rolling_and_lag() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    // RollingMean(5) reads back 6 on eviction, Lag(3) back 4 → retain 8 covers both.
    let rec = b.segment(record_bounded(Retention::count(8)), srcv);
    let lag = b.segment(lag_series(3, f64::NAN), rec);
    let rmean = b.segment(rolling_mean(Window::Count(5)), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let m = 30_i64;
    for t in 1..=m {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(t as f64);
        g.stabilize(&mut pool);
        if t >= 5 {
            // Mean of the last 5 values {t-4..t} == t - 2.
            assert!(
                (g.view(rmean).as_slice().unwrap()[0] - (t as f64 - 2.0)).abs() < 1e-9,
                "rmean@{t} = {}",
                g.view(rmean).as_slice().unwrap()[0],
            );
        }
        if t > 3 {
            // Lag(3): the value from 3 steps ago == t - 3.
            assert!(
                (g.view(lag).as_slice().unwrap()[0] - (t as f64 - 3.0)).abs() < 1e-9,
                "lag@{t} = {}",
                g.view(lag).as_slice().unwrap()[0],
            );
        }
    }

    let s: SeriesView<f64, 0> = g.view(rec);
    assert!(
        (s.len() as i64) < m,
        "expected front-compaction (window={} of {m})",
        s.len()
    );
    assert!(s.len() <= 16, "physical storage unbounded: {}", s.len());
    assert_eq!(s.last().unwrap().data(), &[m as f64], "latest value intact");
    assert_eq!(s.last_timestamp(), Some(ts(m)), "latest timestamp intact");
}

/// `scenario_run_periodic_single_input`: data [10,20,30]@[1,2,3], clock @2 only
/// → record (2,20). Exercises the `Clocked` Segment + clock gating.
#[test]
fn clocked_periodic() {
    let mut b = Builder::new(Instant::MIN);
    let (data, datav) = scalar_src(&mut b, 0.0);
    let (tick_cell, tick) = b.source(RefSource::new(()));
    let gated = b.segment(
        Clocked::<_, ()>::new(filter(|_: ArrayView<f64, 0>| true)),
        (tick, datav),
    );
    let rec = b.segment(record(), gated);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let clock_ticks = [2_i64];
    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(data) = Array::scalar(v);
        if clock_ticks.contains(&t) {
            let _ = g.state_mut(tick_cell);
        }
        g.stabilize(&mut pool);
    }

    let s: SeriesView<f64, 0> = g.view(rec);
    assert_eq!(s.len(), 1);
    assert_eq!(s.data(), &[20.0]);
    assert_eq!(s.timestamps(), &[ts(2)]);
}

/// `scenario_run_coalescing`: two sources fire at the same timestamp → one
/// stabilize over the union of cones → 110, 220.
#[test]
fn coalesced_two_source_add() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = scalar_src(&mut b, 0.0);
    let (bb, bv) = scalar_src(&mut b, 0.0);
    let sum = b.segment(add(), (av, bv));
    let rec = b.segment(record(), sum);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, va, vb) in [(1_i64, 10.0, 100.0), (2, 20.0, 200.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(a) = Array::scalar(va);
        *g.state_mut(bb) = Array::scalar(vb);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.view(rec).data(), &[110.0, 220.0]);
}

/// Per-element notify: gen1 all fire → [1,2,3]; gen2 only s1 → Stack keeps stale
/// [1,20,3]; StackSync NaN-fills [NaN,20,NaN]. The three scalar (rank-0) sources
/// stack along a new axis into a rank-1 cross-section.
#[test]
fn slice_stack_and_sync() {
    let mut b = Builder::new(Instant::MIN);
    let (s0, s0v) = scalar_src(&mut b, 0.0);
    let (s1, s1v) = scalar_src(&mut b, 0.0);
    let (s2, s2v) = scalar_src(&mut b, 0.0);
    let stacked = b.segment(stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let synced = b.segment(stack_sync::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(s0) = Array::scalar(1.0);
    *g.state_mut(s1) = Array::scalar(2.0);
    *g.state_mut(s2) = Array::scalar(3.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(stacked).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(g.view(synced).as_slice().unwrap(), &[1.0, 2.0, 3.0]);

    *g.state_mut(s1) = Array::scalar(20.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(stacked).as_slice().unwrap(), &[1.0, 20.0, 3.0]);
    let v = g.view(synced).as_slice().unwrap();
    assert!(v[0].is_nan());
    assert_eq!(v[1], 20.0);
    assert!(v[2].is_nan());
}

/// `rolling::mean_basic`: window 3 over [1,2,3,6] → false,false, mean 2.0,
/// mean 11/3. Validates Series-input rolling + warm-up gating (returns false
/// until the window is full, suppressing downstream).
#[test]
fn rolling_mean_count_warmup_and_value() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.segment(record(), srcv);
    let rm = b.segment(rolling_mean(Window::Count(3)), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(rm).as_slice().unwrap(), &[2.0]); // mean(1,2,3)

    *g.context_mut() = ts(4);
    *g.state_mut(src) = Array::scalar(6.0);
    g.stabilize(&mut pool);
    assert!((g.view(rm).as_slice().unwrap()[0] - 11.0 / 3.0).abs() < 1e-12); // mean(2,3,6)
}

// -- Batch 2: arithmetic / rolling / structural parity ----------------------

/// Unary `negate` + binary `subtract`/`divide` (values from `arithmetic` tests).
#[test]
fn arith_unary_and_binary() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, -2.0, 3.0]);
    let neg = b.segment(negate(), av);
    let (x, xv) = scalar_src(&mut b, 20.0);
    let (y, yv) = scalar_src(&mut b, 4.0);
    let sub = b.segment(subtract(), (xv, yv));
    let div = b.segment(divide(), (xv, yv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, -2.0, 3.0]);
    *g.state_mut(x) = Array::scalar(20.0);
    *g.state_mut(y) = Array::scalar(4.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(neg).as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    assert_eq!(g.view(sub).as_slice().unwrap(), &[16.0]);
    assert_eq!(g.view(div).as_slice().unwrap(), &[5.0]);
}

/// `min`/`max`/`pow` (values from `arithmetic` tests).
#[test]
fn arith_min_max_pow() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 5.0, 3.0]);
    let (bb, bv) = vec_src(&mut b, vec![2.0_f64, 4.0, 6.0]);
    let mn = b.segment(min(), (av, bv));
    let mx = b.segment(max(), (av, bv));
    let p = b.segment(pow(2.0), av);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 5.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec([3], vec![2.0, 4.0, 6.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(mn).as_slice().unwrap(), &[1.0, 4.0, 3.0]);
    assert_eq!(g.view(mx).as_slice().unwrap(), &[2.0, 5.0, 6.0]);
    assert_eq!(g.view(p).as_slice().unwrap(), &[1.0, 25.0, 9.0]);
}

/// `RollingSum`/`RollingVariance` window-3 (values from rolling tests).
#[test]
fn rolling_sum_and_variance() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.segment(record(), srcv);
    let rsum = b.segment(rolling_sum(Window::Count(3)), rec);
    let rvar = b.segment(rolling_variance(Window::Count(3)), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(rsum).as_slice().unwrap(), &[6.0]);
    assert!((g.view(rvar).as_slice().unwrap()[0] - 2.0 / 3.0).abs() < 1e-10);

    *g.context_mut() = ts(4);
    *g.state_mut(src) = Array::scalar(4.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(rsum).as_slice().unwrap(), &[9.0]); // 2+3+4
}

/// `RollingCovariance` on a `[2]` vector with `y = 2x` (values from cov tests).
#[test]
fn rolling_covariance_2d() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = vec_src(&mut b, vec![0.0_f64; 2]);
    let rec = b.segment(record(), srcv);
    let cov = b.segment(rolling_covariance(Window::Count(3)), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, x) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::from_vec([2], vec![x, 2.0 * x]);
        g.stabilize(&mut pool);
    }
    let cb = g.view(cov).to_contiguous(); // [2,2]
    let c: &[f64] = &cb;
    assert!((c[0] - 2.0 / 3.0).abs() < 1e-10); // Var(x)
    assert!((c[1] - 4.0 / 3.0).abs() < 1e-10); // Cov(x,y)
    assert!((c[3] - 8.0 / 3.0).abs() < 1e-10); // Var(y)
}

/// `Ema` window-2 (value from ema_two_values).
#[test]
fn ema_two_values() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.segment(record(), srcv);
    let e = b.segment(ema_series(0.5, 2), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
    assert!((g.view(e).as_slice().unwrap()[0] - expected).abs() < 1e-10);
}

/// `Where` / `Cast` / `Id` (values from their legacy tests).
#[test]
fn structural_where_cast_id() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 5.0, 2.0]);
    let w = b.segment(keep_where(|v: f64| v > 3.0, 0.0_f64), av);
    // `Id` is the whole-value `RefPort` identity — exercise it on a scalar
    // cell (array edges are always `ArrayPort` views, never `RefPort`).
    let (k_cell, k) = b.source(RefSource::new(7_i64));
    let i = b.segment(Id::<i64>::new(), k);
    let (ci_cell, ci) = b.source(array_cell(Array::from_vec([3], vec![1_i32, 2, 3])));
    let c = b.segment(cast::<i32, f64, 1>(), ci);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 5.0, 2.0]);
    *g.state_mut(k_cell) = 9;
    *g.state_mut(ci_cell) = Array::from_vec([3], vec![1_i32, 2, 3]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(w).as_slice().unwrap(), &[0.0, 5.0, 0.0]);
    assert_eq!(*g.view(i), 9);
    assert_eq!(g.view(c).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

// -- Batch 3: num tail (element-wise / cross-tick / cross-sectional) --------

/// `Clamp`, `Fillna`, `ForwardFill` (single-shot, values from legacy tests).
#[test]
fn num_clamp_fillna_ffill() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 3.0, 7.0]);
    let clamp = b.segment(clamp(2.0, 5.0), av);
    let (na, nav) = vec_src(&mut b, vec![1.0_f64, f64::NAN, 3.0]);
    let fill = b.segment(fillna(0.0), nav);
    let ff = b.segment(forward_fill(), nav);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 3.0, 7.0]);
    *g.state_mut(na) = Array::from_vec([3], vec![1.0, f64::NAN, 3.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(clamp).as_slice().unwrap(), &[2.0, 3.0, 5.0]);
    assert_eq!(g.view(fill).as_slice().unwrap(), &[1.0, 0.0, 3.0]);
    let v = g.view(ff).as_slice().unwrap();
    assert_eq!(v[0], 1.0);
    assert!(v[1].is_nan());
    assert_eq!(v[2], 3.0);
}

/// `Diff` / `PctChange` across ticks (NaN on first, then differences/returns).
#[test]
fn num_diff_and_pct_change() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let d = b.segment(diff(), srcv);
    let pc = b.segment(pct_change(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::scalar(100.0);
    g.stabilize(&mut pool);
    assert!(g.view(d).as_slice().unwrap()[0].is_nan());
    assert!(g.view(pc).as_slice().unwrap()[0].is_nan());

    *g.state_mut(src) = Array::scalar(110.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(d).as_slice().unwrap()[0], 10.0);
    assert!((g.view(pc).as_slice().unwrap()[0] - 0.1).abs() < 1e-12);
}

/// Cross-sectional `Gaussianize` / `Percentile` / `Standardize` / `Winsorize`
/// (values from their legacy tests).
#[test]
fn num_cross_sectional() {
    let mut b = Builder::new(Instant::MIN);
    let (five, fivev) = vec_src(&mut b, vec![30.0_f64, 10.0, 50.0, 20.0, 40.0]);
    let gau = b.segment(gaussianize(), fivev);
    let pct = b.segment(percentile(), fivev);
    let (std_in, std_inv) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
    let zsc = b.segment(standardize(), std_inv);
    let (win_in, win_inv) = vec_src(&mut b, (0..10).map(|i| i as f64).collect());
    let win = b.segment(winsorize(0.1), win_inv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(five) = Array::from_vec([5], vec![30.0, 10.0, 50.0, 20.0, 40.0]);
    *g.state_mut(std_in) = Array::from_vec([5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    *g.state_mut(win_in) = Array::from_vec([10], (0..10).map(|i| i as f64).collect());
    g.stabilize(&mut pool);

    // Gaussianize: middle (30) → Φ⁻¹(0.5) = 0; symmetric ranks sum to 0.
    let gvb = g.view(gau).to_contiguous();
    let gv: &[f64] = &gvb;
    assert!((gv[0] - 0.0).abs() < 1e-9);
    assert!(gv.iter().sum::<f64>().abs() < 1e-9);
    // Percentile: (0.5,1.5,2.5,3.5,4.5)/5 by rank.
    let pvb = g.view(pct).to_contiguous();
    let pv: &[f64] = &pvb;
    assert!(
        (pv[1] - 0.1).abs() < 1e-12 && (pv[0] - 0.5).abs() < 1e-12 && (pv[2] - 0.9).abs() < 1e-12
    );
    // Standardize: zero mean, unit pop-variance.
    let zvb = g.view(zsc).to_contiguous();
    let zv: &[f64] = &zvb;
    assert!((zv.iter().sum::<f64>() / 5.0).abs() < 1e-12);
    assert!((zv.iter().map(|&x| x * x).sum::<f64>() / 5.0 - 1.0).abs() < 1e-12);
    // Winsorize p=0.1 over [0..9] → clip to [1, 8].
    assert_eq!(
        g.view(win).as_slice().unwrap(),
        &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]
    );
}

// -- Batch 4: transform / reshape (Map, Apply, Select, Lag, Concat) ---------

/// `Map` (allocating SI→SO) doubling a scalar.
#[test]
fn map_doubles() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let m = b.segment(
        map(|a: ArrayView<f64, 0>| {
            let mut o = a.to_array();
            o[[]] *= 2.0;
            o
        }),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::scalar(5.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(m).as_slice().unwrap(), &[10.0]);
}

/// `Apply` (two-input add) and `Select` (flat index pick).
#[test]
fn apply_add_and_select() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 2.0, 3.0]);
    let (bb, bv) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0]);
    let ap = b.segment(
        apply(|(a, b): (ArrayView<f64, 1>, ArrayView<f64, 1>)| {
            let mut out = a.to_array();
            for (o, v) in out.data_mut().iter_mut().zip(b.to_contiguous().iter()) {
                *o += *v;
            }
            out
        }),
        (av, bv),
    );
    let (five, fivev) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
    let sel = b.segment(select_flat::<f64, 1, 1>(vec![1, 3]), fivev);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec([3], vec![10.0, 20.0, 30.0]);
    *g.state_mut(five) = Array::from_vec([5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(ap).as_slice().unwrap(), &[11.0, 22.0, 33.0]);
    assert_eq!(g.view(sel).as_slice().unwrap(), &[20.0, 40.0]);
}

/// `Lag` (offset 2 over a recorded series).
#[test]
fn lag_offset_two() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.segment(record(), srcv);
    let lag = b.segment(lag_series(2, f64::NAN), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        *g.context_mut() = ts(t);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(lag).as_slice().unwrap(), &[10.0]); // value from 2 steps ago
}

/// `Concat` axis-0 of two `[2]` arrays → `[4]`.
#[test]
fn concat_axis0() {
    let mut b = Builder::new(Instant::MIN);
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 2.0]);
    let (bb, bv) = vec_src(&mut b, vec![3.0_f64, 4.0]);
    let cc = b.segment(concat(0), &[av, bv][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([2], vec![1.0, 2.0]);
    *g.state_mut(bb) = Array::from_vec([2], vec![3.0, 4.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(cc).as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

// -- Batch 5/6: metrics (clock-gated) + stocks -----------------------------

/// Clock-gated `CompoundReturn` / `AverageReturn` / `Volatility` over a price
/// path, firing the clock each tick (values from the metrics tests).
#[test]
fn metrics_clock_gated() {
    let mut b = Builder::new(Instant::MIN);
    let (data, datav) = scalar_src(&mut b, 0.0);
    let (tick_cell, tick) = b.source(RefSource::new(()));
    let cr = b.segment(compound_return(), (tick, datav));
    let ar = b.segment(average_return(), (tick, datav));
    let vol = b.segment(volatility(), (tick, datav));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 100.0), (2, 110.0)] {
        let _ = t;
        *g.state_mut(data) = Array::scalar(v);
        let _ = g.state_mut(tick_cell);
        g.stabilize(&mut pool);
    }
    assert!((g.view(cr).as_slice().unwrap()[0] - 0.10).abs() < 1e-10);
    assert!((g.view(ar).as_slice().unwrap()[0] - 0.10).abs() < 1e-10);
    assert_eq!(g.view(vol).as_slice().unwrap()[0], 0.0); // single return → zero std

    *g.state_mut(data) = Array::scalar(99.0);
    let _ = g.state_mut(tick_cell);
    g.stabilize(&mut pool);
    assert!(g.view(ar).as_slice().unwrap()[0].abs() < 1e-10); // 0.10, -0.10 → 0
    assert!((g.view(vol).as_slice().unwrap()[0] - 0.10).abs() < 1e-10); // std 0.10
}

/// `Drawdown` (single input, no clock) from the running maximum.
#[test]
fn metrics_drawdown() {
    let mut b = Builder::new(Instant::MIN);
    let (data, datav) = scalar_src(&mut b, 0.0);
    let dd = b.segment(drawdown(), datav);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (v, e) in [(100.0, 0.0), (120.0, 0.0), (90.0, -0.25)] {
        *g.state_mut(data) = Array::scalar(v);
        g.stabilize(&mut pool);
        assert!((g.view(dd).as_slice().unwrap()[0] - e).abs() < 1e-10);
    }
}

/// `Annualize`: YTD [2024, day 91, 100, 20] → annualized × 365/91.
#[test]
fn stocks_annualize() {
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = vec_src(&mut b, vec![2024.0_f64, 91.0, 100.0, 20.0]);
    let ann = b.segment(annualize(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::from_vec([4], vec![2024.0, 91.0, 100.0, 20.0]);
    g.stabilize(&mut pool);
    let o = g.view(ann).as_slice().unwrap();
    assert!((o[0] - 100.0 * 365.0 / 91.0).abs() < 1e-10);
    assert!((o[1] - 20.0 * 365.0 / 91.0).abs() < 1e-10);
}

/// `ForwardAdjust`: price-only tick, then a cash dividend (message-passing on
/// the two inputs). 9.5 with a 0.5 cash dividend forward-adjusts back to 10.0.
#[test]
fn stocks_forward_adjust() {
    let mut b = Builder::new(Instant::MIN);
    let (price, pricev) = scalar_src(&mut b, 10.0);
    let (divd, divdv) = vec_src(&mut b, vec![0.0_f64, 0.0]);
    let fa = b.segment(forward_adjust(), (pricev, divdv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // gen1: price only.
    *g.state_mut(price) = Array::scalar(10.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(fa).as_slice().unwrap(), &[10.0]);

    // gen2: price 9.5 + cash dividend 0.5 → adjusted back to 10.0.
    *g.state_mut(price) = Array::scalar(9.5);
    *g.state_mut(divd) = Array::from_vec([2], vec![0.0, 0.5]);
    g.stabilize(&mut pool);
    assert!((g.view(fa).as_slice().unwrap()[0] - 10.0).abs() < 1e-12);
}

// ===========================================================================
// Parallel-execution gate (Pool::new(N>0))
// ===========================================================================

/// One source fans out to K independent Filter(>3)→Record chains across 8
/// worker threads; every branch must record exactly the passing values.
#[test]
fn parallel_fanout_matches_sequential() {
    const K: usize = 16;
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let recs: Vec<_> = (0..K)
        .map(|_| {
            let f = b.segment(
                filter(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 3.0),
                srcv,
            );
            b.segment(record(), f)
        })
        .collect();
    let mut g = b.build();
    let mut pool = Pool::new(8);

    let seq = [1.0_f64, 5.0, 2.0, 10.0, 4.0, 0.5, 7.0, 3.0, 9.0];
    let expected: Vec<f64> = seq.iter().copied().filter(|&v| v > 3.0).collect();
    for (i, &v) in seq.iter().enumerate() {
        *g.context_mut() = ts(i as i64 + 1);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &rec in &recs {
        assert_eq!(
            g.view(rec).data(),
            &expected[..],
            "a parallel branch diverged"
        );
    }
}

/// K concurrent stateful Count chains behind per-branch gates over 500
/// generations on 8 workers; each Count must equal the number of passing gens.
#[test]
fn parallel_stress_stateful_counts() {
    const K: usize = 12;
    const GENS: usize = 500;
    let mut b = Builder::new(Instant::MIN);
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let cnts: Vec<_> = (0..K)
        .map(|_| {
            let f = b.segment(
                filter(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 0.0),
                srcv,
            );
            b.segment(Count::<0>, f)
        })
        .collect();
    let mut g = b.build();
    let mut pool = Pool::new(8);

    let mut passes = 0.0_f64;
    for i in 0..GENS {
        let v = if i % 3 == 0 { 1.0 } else { -1.0 };
        if v > 0.0 {
            passes += 1.0;
        }
        *g.context_mut() = ts(i as i64 + 1);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &c in &cnts {
        assert_eq!(
            g.view(c).as_slice().unwrap(),
            &[passes],
            "a parallel Count raced"
        );
    }
}

// ===========================================================================
// Split + segment fusion
// ===========================================================================

/// `Split` fans a `[3, 2]` panel into per-row **zero-copy view** ports holding
/// the row values; rows notify exactly when the panel does. The rows feed a
/// `Stack` that rebuilds the panel, and a `Record` on the stack counts how many
/// times the carry join recomputed — it advances only on panel pokes, not on an
/// unrelated generation.
///
/// [array-view-refactor] Reinterpreted: a `Split` row is an ordinary by-value
/// `ArrayPort` (since the by-value group migration it feeds `Gate` / `Select` /
/// `Count` directly again); this test keeps the notify-tracking formulation —
/// counting `Stack` recomputes via a clock-stamped `Record` length — which
/// pins the same intent (rows recompute with the panel, not on unrelated
/// pokes).
#[test]
fn split_rows_notify_with_panel() {
    let mut b = Builder::new(Instant::MIN);
    let (panel_cell, panel) = b.source(array_cell(Array::from_vec([3, 2], vec![0.0_f64; 6])));
    let (other_cell, other) = b.source(array_cell(Array::scalar(0.0_f64)));
    let rows = b.segment(split(3), panel);
    assert_eq!(rows.len(), 3);
    // The rows feed a carry `Stack` that rebuilds the `[3, 2]` panel; a `Record`
    // on the stacked output advances exactly once per recompute of the join.
    let stacked = b.segment(stack::<f64, 1, 2>(0), &rows[..]);
    let rec = b.segment(record(), stacked);
    let _sink = b.segment(Count::<0>, other); // unrelated cone
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Build values: row views hold the initial panel rows; the record is empty.
    assert_eq!(g.view(rows[0]).to_contiguous().as_ref(), &[0.0, 0.0]);
    assert_eq!(g.view(rows[0]).extents(), [2]);
    assert_eq!(g.view(rec).len(), 0);

    *g.context_mut() = ts(1);
    *g.state_mut(panel_cell) = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(rows[0]).to_contiguous().as_ref(), &[1.0, 2.0]);
    assert_eq!(g.view(rows[1]).to_contiguous().as_ref(), &[3.0, 4.0]);
    assert_eq!(g.view(rows[2]).to_contiguous().as_ref(), &[5.0, 6.0]);
    assert_eq!(
        g.view(stacked).as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(g.view(rec).len(), 1); // join recomputed once (on the panel poke)

    // Poking an unrelated source must not advance the carry-join record.
    *g.context_mut() = ts(2);
    *g.state_mut(other_cell) = Array::scalar(1.0);
    g.stabilize(&mut pool);
    assert_eq!(
        g.view(rec).len(),
        1,
        "unrelated poke must not recompute the join"
    );
}

/// The declared axis size is validated against the build-time input shape.
#[test]
#[should_panic(expected = "Split: input axis-0 size")]
fn split_axis_size_mismatch_panics() {
    let mut b = Builder::new(Instant::MIN);
    let (_, panel) = b.source(array_cell(Array::from_vec([3, 2], vec![0.0_f64; 6])));
    let _ = b.segment(split::<f64, 2, 1>(2), panel);
    let _g = b.build();
}

/// The view chain (retaining `Gate` -> view-input `SliceView` / `ForwardAdjust`)
/// is tick-for-tick bit-identical to the owned chain (`Filter` -> owned `Select`
/// / `ForwardAdjust`) over the same source pokes, including the NaN cutoff and
/// the price/dividend message-passing. `Gate` and `Filter` honour the
/// no-notify⟹unchanged contract by retaining the last passed row, so their
/// downstream cones must agree bit-for-bit.
///
/// [array-view-refactor] Reinterpreted: the per-stock row is sourced as a
/// direct view (a per-stock `[2]` `array_cell` lending its `ArrayPort`)
/// instead of a panel split. (A `Split` row is an ordinary by-value
/// `ArrayPort` again, so a panel-split formulation would also work; this
/// variant keeps the simpler source boundary.) The data stream and every
/// asserted value are unchanged.
#[test]
fn view_chain_matches_owned_chain() {
    fn any_finite(a: ArrayView<'_, f64, 1>) -> bool {
        a.to_contiguous().iter().any(|x| x.is_finite())
    }
    fn bits(a: ArrayView<'_, f64, 0>) -> Vec<u64> {
        a.to_contiguous().iter().map(|x| x.to_bits()).collect()
    }

    let nan = f64::NAN;
    let mut b = Builder::new(Instant::MIN);
    // Stock 0's price/dividend rows as direct `[2]` view sources.
    let (prices, prices_view) = vec_src(&mut b, vec![nan; 2]);
    let (div, div_view) = vec_src(&mut b, vec![nan; 2]);

    // Owned reference chain (materializes at the row Selects).
    let p_f = {
        let m = b.segment(select_flat(vec![0, 1]), prices_view);
        b.segment(filter(any_finite), m)
    };
    let d_f = {
        let m = b.segment(select_flat(vec![0, 1]), div_view);
        b.segment(filter(any_finite), m)
    };
    // Squeeze the single close out to a scalar (rank-0) price.
    let close = b.segment(select(vec![0], 0, true), p_f);
    let adj = b.segment(forward_adjust().with_output_prices(false), (close, d_f));
    let adjusted = b.segment(multiply(), (close, adj));

    // Zero-copy view chain (materializes at SliceView).
    let p_g = b.segment(gate(any_finite), prices_view);
    let d_g = b.segment(gate(any_finite), div_view);
    let v_close = b.segment(slice_view(vec![0], 0, true), p_g);
    let v_adj = b.segment(
        ForwardAdjust::<0, 1>::default().with_output_prices(false),
        (v_close, d_g),
    );
    let v_adjusted = b.segment(multiply(), (v_close, v_adj));

    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (price row, dividend row) for stock 0 per tick; None = source not poked.
    type Tick = (Option<[f64; 2]>, Option<[f64; 2]>);
    let ticks: &[Tick] = &[
        (Some([10.0, 100.0]), None),
        (Some([12.0, 110.0]), Some([0.0, 2.0])), // cash dividend
        (None, None),                            // idle generation
        (Some([11.0, 90.0]), None),
        (Some([nan, nan]), Some([0.5, 0.0])), // share dividend on a no-data tick
        (Some([13.0, 95.0]), None),
    ];
    for (i, (p, d)) in ticks.iter().enumerate() {
        if let Some(p) = p {
            *g.state_mut(prices) = Array::from_vec([2], p.to_vec());
        }
        if let Some(d) = d {
            *g.state_mut(div) = Array::from_vec([2], d.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(
            bits(g.view(adjusted)),
            bits(g.view(v_adjusted)),
            "tick {i}: adjusted"
        );
        assert_eq!(bits(g.view(adj)), bits(g.view(v_adj)), "tick {i}: adjusts");
    }
}

/// A carry `Stack` over zero-copy `Split` rows honours the no-notify⟹unchanged
/// contract: the join reads **every** input each generation (incl. un-notified
/// ones), and an idle generation (the panel not poked) leaves the stacked
/// cross-section byte-identical to the last poked value. Two equivalent joins
/// (`Stack` and its `Stack` alias) over the same rows agree bit-for-bit.
///
/// [array-view-refactor] Reinterpreted: the retained core — the carry join
/// re-reads un-notified inputs and freezes its output across idle
/// generations — is tested by stacking the `Split` rows directly and asserting
/// the idle-generation carry. (`Stack` is now a type alias of `Stack`, so
/// the owned-vs-view comparison is the same operator over the same inputs.)
#[test]
fn view_join_carry_matches_owned_join() {
    fn bits(a: ArrayView<'_, f64, 2>) -> Vec<u64> {
        a.to_contiguous().iter().map(|x| x.to_bits()).collect()
    }

    let n = 3usize;
    let mut b = Builder::new(Instant::MIN);
    let (panel_cell, panel) = b.source(array_cell(Array::from_vec([n, 2], vec![0.0; n * 2])));
    let rows = b.segment(split::<f64, 2, 1>(n), panel);

    // Two equivalent carry joins over the same `Split` rows (the only buildable
    // carry-join input): `Stack` and its `Stack` alias.
    let owned_join = b.segment(stack(0), &rows[..]);
    let view_join = b.segment(stack(0), &rows[..]);

    let mut g = b.build();
    let mut pool = Pool::new(0);

    // `None` = idle generation (panel not poked → the join must carry the last
    // stacked cross-section unchanged).
    type Tick = Option<[f64; 6]>;
    let ticks: &[Tick] = &[
        Some([1.0, 10.0, 2.0, 20.0, 3.0, 30.0]),
        None, // idle generation → carry
        Some([4.0, 40.0, 5.0, 50.0, 6.0, 60.0]),
        None, // idle generation → carry
        None, // still idle → carry
        Some([7.0, 70.0, 8.0, 80.0, 9.0, 90.0]),
    ];
    let mut last = [0.0f64; 6];
    for (i, t) in ticks.iter().enumerate() {
        if let Some(p) = t {
            *g.state_mut(panel_cell) = Array::from_vec([n, 2], p.to_vec());
            last = *p;
        }
        g.stabilize(&mut pool);
        // The two joins are the same operator over the same rows.
        assert_eq!(
            bits(g.view(view_join)),
            bits(g.view(owned_join)),
            "tick {i}: joins agree"
        );
        // The stacked cross-section is the last poked panel (carried across idle
        // generations) — the no-notify⟹unchanged carry contract.
        assert_eq!(
            g.view(view_join).as_slice().unwrap(),
            &last,
            "tick {i}: carry"
        );
    }
}

/// A fused `segment!` chain (Filter -> Selects -> ForwardAdjust -> Multiply)
/// is tick-for-tick bit-identical to the same operators as separate nodes,
/// including the NaN cutoff and the price/dividend message-passing.
#[test]
fn fused_segment_matches_unfused_nodes() {
    fn any_finite(a: ArrayView<'_, f64, 1>) -> bool {
        a.to_contiguous().iter().any(|x| x.is_finite())
    }
    fn bits(a: ArrayView<'_, f64, 0>) -> Vec<u64> {
        a.to_contiguous().iter().map(|x| x.to_bits()).collect()
    }

    let fused = tradingflow_graph::segment!(|prices_row: Vp<1>, div_row: Vp<1>|
        -> (Vp<0>, Vp<0>) {
        let prices = filter(any_finite) @ prices_row;
        let dividends = filter(any_finite) @ div_row;
        let close = select(vec![0], 0, true) @ prices;
        let adjusts = forward_adjust().with_output_prices(false) @ (close, dividends);
        let adjusted = multiply() @ (close, adjusts);
        (adjusted, adjusts)
    });

    let nan = f64::NAN;
    let mut b = Builder::new(Instant::MIN);
    let (prices, pricesv) = vec_src(&mut b, vec![nan; 2]);
    let (div, divv) = vec_src(&mut b, vec![nan; 2]);

    // Reference: the same chain as separate nodes.
    let p_f = b.segment(filter(any_finite), pricesv);
    let d_f = b.segment(filter(any_finite), divv);
    let close = b.segment(select(vec![0], 0, true), p_f);
    let adj = b.segment(forward_adjust().with_output_prices(false), (close, d_f));
    let adjusted = b.segment(multiply(), (close, adj));

    let (f_adjusted, f_adj) = b.segment(fused, (pricesv, divv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (prices_row, dividend_row) per tick; None = source not poked.
    type Tick = (Option<[f64; 2]>, Option<[f64; 2]>);
    let ticks: &[Tick] = &[
        (Some([10.0, 100.0]), None),
        (Some([12.0, 110.0]), Some([0.0, 2.0])), // cash dividend
        (None, None),                            // idle generation
        (Some([11.0, 90.0]), None),
        (Some([nan, nan]), Some([0.5, 0.0])), // share dividend on a no-data tick
        (Some([13.0, 95.0]), None),
    ];
    for (i, (p, d)) in ticks.iter().enumerate() {
        if let Some(p) = p {
            *g.state_mut(prices) = Array::from_vec([2], p.to_vec());
        }
        if let Some(d) = d {
            *g.state_mut(div) = Array::from_vec([2], d.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(
            bits(g.view(adjusted)),
            bits(g.view(f_adjusted)),
            "tick {i}: adjusted"
        );
        assert_eq!(bits(g.view(adj)), bits(g.view(f_adj)), "tick {i}: adjusts");
    }
}

// ===========================================================================
// Comparison / logical operators
// ===========================================================================

#[test]
fn comparisons_follow_ieee_nan_semantics() {
    // Ordering predicates are false for NaN; `!=` is true. `is_finite` is the
    // explicit missing-data test.
    let nan = f64::NAN;
    let mut b = Builder::new(Instant::MIN);
    let (_s, x) = vec_src(&mut b, vec![1.0, -1.0, 0.0, nan]);
    let gt0 = b.segment(greater_than(0.0), x);
    let ne0 = b.segment(not_equal_to(0.0), x);
    let fin = b.segment(is_finite(), x);
    let g = b.build();

    assert_eq!(g.view(gt0).to_vec(), vec![true, false, false, false]);
    assert_eq!(g.view(ne0).to_vec(), vec![true, true, false, true]); // NaN != 0 is true
    assert_eq!(g.view(fin).to_vec(), vec![true, true, true, false]);
}

#[test]
fn logical_connectives_and_mask_readout() {
    let mut b = Builder::new(Instant::MIN);
    let (_sa, a) = vec_src(&mut b, vec![1.0, 1.0, 0.0, 0.0]);
    let (_sb, bb) = vec_src(&mut b, vec![1.0, 0.0, 1.0, 0.0]);
    let am = b.segment(greater_than(0.5), a);
    let bm = b.segment(greater_than(0.5), bb);
    let both = b.segment(and(), (am, bm));
    let either = b.segment(or(), (am, bm));
    let neither = b.segment(not(), either);
    let onehot = b.segment(xor(), (am, bm));
    // Mask -> numeric currency, and the three-input selector.
    let ind = b.segment(indicator(1.0, f64::NAN), both);
    let pick = b.segment(Choose::<f64, 1>::new(), (onehot, a, bb));
    let g = b.build();

    assert_eq!(g.view(both).to_vec(), vec![true, false, false, false]);
    assert_eq!(g.view(either).to_vec(), vec![true, true, true, false]);
    assert_eq!(g.view(neither).to_vec(), vec![false, false, false, true]);
    assert_eq!(g.view(onehot).to_vec(), vec![false, true, true, false]);
    let iv = g.view(ind).to_vec();
    assert_eq!(iv[0], 1.0);
    assert!(iv[1..].iter().all(|x| x.is_nan()));
    // `onehot` selects from `a` where exactly one mask is set, else from `bb`:
    // [bb[0], a[1], a[2], bb[3]] = [1, 1, 0, 0].
    assert_eq!(g.view(pick).to_vec(), vec![1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn comparison_of_two_arrays_and_strided_inputs() {
    // `Compare` over a strided (column) view exercises the strided-slow path of
    // the shared elementwise core at a bool output type.
    let mut b = Builder::new(Instant::MIN);
    let (_, s) = b.source(array_cell(Array::from_vec(
        [2, 2],
        vec![1.0, 5.0, 4.0, 2.0], // rows: [1,5], [4,2]
    )));
    let m = s;
    // `SliceView`'s output rank is not determined by its input rank (`squeeze`
    // drops the sliced axis), so it is the one operator here that must be told.
    let col0 = b.segment(slice_view::<f64, 2, 1>(vec![0], 1, true), m); // [1, 4]
    let col1 = b.segment(slice_view::<f64, 2, 1>(vec![1], 1, true), m); // [5, 2]
    let lt = b.segment(less(), (col0, col1));
    let g = b.build();
    assert_eq!(g.view(lt).to_vec(), vec![true, false]);
}

/// The signal from the design brief, as one fused node:
/// `MA(x,10) - MA(x,5) > 0 AND NOT LAG(MA(x,10) - MA(x,5), 1) > 0`
/// i.e. the crossover *event* — the spread is positive now and was not before.
#[test]
fn ma_crossover_signal_fuses_into_one_node() {
    let ret = Retention::count(4);
    let (fast, slow) = (2usize, 3usize);

    // The result is a single application in tail position; the crossover is a
    // rank-0 boolean edge.
    let seg = tradingflow_graph::segment!(|xs: crate::operators::SeriesPort<f64, 0>| -> ArrayPort<bool, 0> {
        let d = subtract() @ (
            rolling_mean(Window::Count(fast)) @ xs,
            rolling_mean(Window::Count(slow)) @ xs,
        );
        let up = greater_than(0.0) @ d;
        let prev = lag_series(1, f64::NAN)
            @ record_bounded(ret)
            @ d;
        and() @ (up, not() @ (greater_than(0.0) @ prev))
    });

    let mut b = Builder::new(Instant::MIN);
    let (src, view) = scalar_src(&mut b, 0.0);
    let series = b.segment(record(), view);
    let signal = b.segment(seg, series);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // A V-shaped path: the fast MA crosses ABOVE the slow MA exactly once.
    // fast=MA2, slow=MA3; the spread turns positive on the first rising tick.
    let path = [10.0, 9.0, 8.0, 7.0, 12.0, 14.0, 16.0];
    let mut fired = Vec::new();
    for (i, &x) in path.iter().enumerate() {
        *g.context_mut() = ts(i as i64 + 1);
        *g.state_mut(src) = Array::scalar(x);
        g.stabilize(&mut pool);
        fired.push(g.view(signal).to_vec()[0]);
    }
    // Warm-up (windows not full) yields no signal; the crossover fires once,
    // and stays quiet while the spread merely remains positive.
    assert_eq!(
        fired,
        vec![false, false, false, false, true, false, false],
        "crossover must be an edge, not a level"
    );
}

// ===========================================================================
// Formula constructors (self-recording windowed segments)
// ===========================================================================

/// Reference model for the crossover signal, mirroring the fused graph's
/// notify semantics: the spread `d` first computes when the shorter window
/// fills (the longer MA still reads `NaN`), and its 1-tick lag reads the
/// previously *computed* spread.
// `!(prev > 0.0)` is the `NOT LAG(..) > 0` of the formula, and the negation is
// load-bearing: `prev` is `NaN` before the spread first computes, and
// `!(NaN > 0.0)` is `true` (matching `not() @ greater_than(0.0)` on a NaN input)
// whereas `prev <= 0.0` would be `false`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn crossover_reference(path: &[f64], fast: usize, slow: usize) -> Vec<bool> {
    let mean = |t: usize, w: usize| -> f64 {
        if t >= w {
            path[t - w..t].iter().sum::<f64>() / w as f64
        } else {
            f64::NAN
        }
    };
    let mut d_hist: Vec<f64> = Vec::new();
    let mut out = Vec::new();
    for t in 1..=path.len() {
        if t < fast.min(slow) {
            out.push(false); // the spread has never been computed
            continue;
        }
        let d = mean(t, fast) - mean(t, slow);
        let prev = d_hist.last().copied().unwrap_or(f64::NAN);
        d_hist.push(d);
        out.push(d > 0.0 && !(prev > 0.0));
    }
    out
}

/// A deterministic quarter-valued pseudo-random path. Quarters keep every
/// running sum exactly representable, so the incremental accumulators match a
/// freshly-summed reference bit-for-bit and `assert_eq!` on floats is sound.
fn quarter_path(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as f64 / 4.0
        })
        .collect()
}

/// The design-brief signal in **formula style** — self-recording [`ma`] /
/// [`lag`] over the live array handle, no hoisted record, no retention
/// arithmetic, no turbofish:
/// `MA(x,2) − MA(x,3) > 0 AND NOT LAG(MA(x,2) − MA(x,3), 1) > 0`.
/// Runs 60 ticks on a two-element cross-section so the private records
/// compact under the windows (retention = window + margin ≪ 60).
#[test]
fn formula_ma_crossover_signal() {
    let (fast, slow) = (2usize, 3usize);

    let mut path0 = vec![10.0, 9.0, 8.0, 7.0, 12.0, 14.0, 16.0];
    path0.extend(quarter_path(42, 53));
    let path1: Vec<f64> = path0.iter().map(|&x| 200.25 - x).collect();

    let mut b = Builder::new(Instant::MIN);
    let (src, xv) = vec_src(&mut b, vec![0.0, 0.0]);
    let signal = b.segment(
        tradingflow_graph::segment!(|x: Vp<1>| -> ArrayPort<bool, 1> {
            let d = subtract() @ (ma(fast) @ x, ma(slow) @ x);
            and() @ (
                greater_than(0.0) @ d,
                not() @ (greater_than(0.0) @ lag(1) @ d),
            )
        }),
        xv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let expect0 = crossover_reference(&path0, fast, slow);
    let expect1 = crossover_reference(&path1, fast, slow);
    let mut fired = 0usize;
    for t in 0..path0.len() {
        *g.context_mut() = ts(t as i64 + 1);
        *g.state_mut(src) = Array::from_vec([2], vec![path0[t], path1[t]]);
        g.stabilize(&mut pool);
        assert_eq!(
            g.view(signal).to_vec(),
            vec![expect0[t], expect1[t]],
            "tick {t}"
        );
        fired += usize::from(expect0[t]) + usize::from(expect1[t]);
    }
    // The V-shaped prefix alone fires element 0 exactly once (tick 5); the
    // random tail keeps both elements crossing, so the assertion above is not
    // vacuously comparing all-false vectors.
    assert!(
        expect0[4] && fired > 5,
        "path must exercise real crossovers"
    );
}

/// [`change`] / [`growth`] equal `x − x₋ₙ` and `(x − x₋ₙ) / x₋ₙ`, with a
/// `NaN` warm-up while the lag is unavailable — pushed directly (no
/// `segment!`), inferring `T`/`N` from the wiring.
#[test]
fn formula_change_and_growth() {
    let mut b = Builder::new(Instant::MIN);
    let (src, xv) = scalar_src(&mut b, 0.0);
    let chg = b.segment(change(2), xv);
    let pct = b.segment(growth(2), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // 40 ticks over a 10-row private record: crosses front compaction.
    let path: Vec<f64> = (1..=40).map(|i| (i * i) as f64).collect();
    for (i, &v) in path.iter().enumerate() {
        *g.context_mut() = ts(i as i64 + 1);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
        let c = g.view(chg).to_vec()[0];
        let p = g.view(pct).to_vec()[0];
        if i < 2 {
            assert!(c.is_nan() && p.is_nan(), "tick {i}: warm-up must be NaN");
        } else {
            let base = path[i - 2];
            assert_eq!(c, path[i] - base, "tick {i}");
            assert_eq!(p, (path[i] - base) / base, "tick {i}");
        }
    }
}

/// [`ma_time`] means all ticks within the trailing time window (daily ticks,
/// 2-day window → the last 3 ticks), across 50 days of duration-bounded
/// compaction.
#[test]
fn formula_ma_time_window() {
    let mut b = Builder::new(Instant::MIN);
    let (src, xv) = scalar_src(&mut b, 0.0);
    let m = b.segment(ma_time(Duration::from_days(2)), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let day = |i: i64| ts(i * 86_400 * 1_000_000_000);
    let vals = quarter_path(7, 50);
    for (i, &v) in vals.iter().enumerate() {
        *g.context_mut() = day(i as i64 + 1);
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
        let k = (i + 1).min(3); // ticks stamped within 2 days of the latest
        let want = vals[i + 1 - k..=i].iter().sum::<f64>() / k as f64;
        assert_eq!(g.view(m).to_vec(), vec![want], "day {i}");
    }
}

/// [`mstd`] (self-recording, variance → sqrt fused) matches the hoisted
/// spelling — a shared bounded [`Record`] feeding [`RollingVariance`] then
/// [`sqrt`] — tick for tick.
#[test]
fn formula_mstd_matches_hoisted() {
    let mut b = Builder::new(Instant::MIN);
    let (src, xv) = vec_src(&mut b, vec![0.0, 0.0]);
    let fused = b.segment(mstd(4), xv);
    let series = b.segment(record_bounded(Retention::count(16)), xv);
    let var = b.segment(rolling_variance(Window::Count(4)), series);
    let hoisted = b.segment(sqrt(), var);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let (p0, p1) = (quarter_path(1, 40), quarter_path(2, 40));
    for t in 0..40 {
        *g.context_mut() = ts(t as i64 + 1);
        *g.state_mut(src) = Array::from_vec([2], vec![p0[t], p1[t]]);
        g.stabilize(&mut pool);
        let (f, h) = (g.view(fused).to_vec(), g.view(hoisted).to_vec());
        for e in 0..2 {
            assert!(
                f[e] == h[e] || (f[e].is_nan() && h[e].is_nan()),
                "tick {t} element {e}: fused {} != hoisted {}",
                f[e],
                h[e]
            );
        }
    }
}
