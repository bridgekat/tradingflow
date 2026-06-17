//! Engine tests: each graph asserts a known-good output (the constants were
//! originally lifted from the retired reference engine's test suite and are now
//! the canonical expected values), plus a parallel-execution gate that runs the
//! gating / clock / concurrent-write paths under real worker threads.

use flowgraph::core::Pool;
use flowgraph::typed::{Graph, GraphBuilder, RefPort, RefSource};

use super::*;
use crate::{Array, ArraySlice, Instant, Retention, Series};

fn ts(n: i64) -> Instant {
    Instant::from_nanos(n)
}

// ===========================================================================
// Sequential execution (Pool::new(0): single-threaded stabilization)
// ===========================================================================

/// `scenario_simple_add`: 10 + 3 == 13.
#[test]
fn simple_add() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let ha = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let hb = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let hc = b.push(Add::<f64>::new(), (*ha, *hb));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(ha) = Array::scalar(10.0);
    *g.state_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(hc).as_slice(), &[13.0]);
}

/// `scenario_chain`: (2 + 3) * 2 == 10. Exercises a 3-deep cone where the leaf
/// is read by two operators (Add and Mul both read `a`).
#[test]
fn chain_add_then_mul() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let bb = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let ab = b.push(Add::<f64>::new(), (*a, *bb));
    let out = b.push(Multiply::<f64>::new(), (ab, *a));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::scalar(2.0);
    *g.state_mut(bb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(out).as_slice(), &[10.0]);
}

/// `scenario_record`: record (10+3), (20+7) → series [13, 27] @ [1, 2].
#[test]
fn record_series() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let ha = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let hb = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let sum = b.push(Add::<f64>::new(), (*ha, *hb));
    let rec = b.push(Record::new(clock.clone()), sum);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(ha) = Array::scalar(10.0);
    *g.state_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    clock.set(ts(2));
    *g.state_mut(ha) = Array::scalar(20.0);
    *g.state_mut(hb) = Array::scalar(7.0);
    g.stabilize(&mut pool);

    let s: &Series<f64> = g.ref_view(rec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.timestamps(), &[ts(1), ts(2)]);
    assert_eq!(s.values(), &[13.0, 27.0]);
}

/// `scenario_run_filter`: source [1,5,2,10]@[1,2,3,4], keep >3 → recorded
/// (2,5),(4,10). THE cutoff proof: a dropped Filter must suppress Record.
#[test]
fn filter_gates_record() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let flt = b.push(Filter(|a: &Array<f64>| a[0] > 3.0), *src);
    let rec = b.push(Record::new(clock.clone()), flt);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 5.0), (3, 2.0), (4, 10.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    let s: &Series<f64> = g.ref_view(rec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.values(), &[5.0, 10.0]);
    assert_eq!(s.timestamps(), &[ts(2), ts(4)]);
}

/// `Last(Record(x))` recovers the latest array value.
#[test]
fn last_of_record() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rec = b.push(Record::new(clock.clone()), *src);
    let lst = b.push(Last::new(0.0_f64), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.ref_view(lst).as_slice(), &[20.0]);
}

/// A retention-bounded `Record` feeding `Lag` and `RollingMean` produces the
/// same outputs as an unbounded one, across front-compaction. Exercises the
/// engine path for logical indexing: the rolling window's `start` and `Lag`'s
/// offset address the series by absolute (logical) index while the front is
/// dropped underneath them.
#[test]
fn bounded_record_feeds_rolling_and_lag() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    // RollingMean(5) reads back 6 on eviction, Lag(3) back 4 → retain 8 covers both.
    let rec = b.push(Record::with_retention(clock.clone(), Retention::count(8)), *src);
    let lag = b.push(Lag::<f64>::new(3, f64::NAN), rec);
    let rmean = b.push(RollingMean::<f64>::count(5), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    let m = 30_i64;
    for t in 1..=m {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(t as f64);
        g.stabilize(&mut pool);
        if t >= 5 {
            // Mean of the last 5 values {t-4..t} == t - 2.
            assert!(
                (g.ref_view(rmean).as_slice()[0] - (t as f64 - 2.0)).abs() < 1e-9,
                "rmean@{t} = {}",
                g.ref_view(rmean).as_slice()[0],
            );
        }
        if t > 3 {
            // Lag(3): the value from 3 steps ago == t - 3.
            assert!(
                (g.ref_view(lag).as_slice()[0] - (t as f64 - 3.0)).abs() < 1e-9,
                "lag@{t} = {}",
                g.ref_view(lag).as_slice()[0],
            );
        }
    }

    let s: &Series<f64> = g.ref_view(rec);
    assert_eq!(s.len(), m as usize, "logical length preserved across compaction");
    assert!(s.base() > 0, "expected front-compaction (base={})", s.base());
    assert!(s.retained_len() <= 16, "physical storage unbounded: {}", s.retained_len());
    assert_eq!(s.last(), Some([m as f64].as_slice()), "latest value intact");
}

/// `scenario_run_periodic_single_input`: data [10,20,30]@[1,2,3], clock @2 only
/// → record (2,20). Exercises the `Clocked` Segment + clock gating.
#[test]
fn clocked_periodic() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let tick = b.push_source(RefSource::new(()));
    let gated = b.push(
        Clocked::<_, ()>::new(Filter(|_: &Array<f64>| true)),
        (*tick, *data),
    );
    let rec = b.push(Record::new(clock.clone()), gated);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    let clock_ticks = [2_i64];
    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        clock.set(ts(t));
        *g.state_mut(data) = Array::scalar(v);
        if clock_ticks.contains(&t) {
            let _ = g.state_mut(tick);
        }
        g.stabilize(&mut pool);
    }

    let s: &Series<f64> = g.ref_view(rec);
    assert_eq!(s.len(), 1);
    assert_eq!(s.values(), &[20.0]);
    assert_eq!(s.timestamps(), &[ts(2)]);
}

/// `scenario_run_coalescing`: two sources fire at the same timestamp → one
/// stabilize over the union of cones → 110, 220.
#[test]
fn coalesced_two_source_add() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let bb = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let sum = b.push(Add::<f64>::new(), (*a, *bb));
    let rec = b.push(Record::new(clock.clone()), sum);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, va, vb) in [(1_i64, 10.0, 100.0), (2, 20.0, 200.0)] {
        clock.set(ts(t));
        *g.state_mut(a) = Array::scalar(va);
        *g.state_mut(bb) = Array::scalar(vb);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.ref_view(rec).values(), &[110.0, 220.0]);
}

/// `RefPorts` input + per-element notify: gen1 all fire → [1,2,3]; gen2 only s1 →
/// Stack keeps stale [1,20,3]; StackSync NaN-fills [NaN,20,NaN].
#[test]
fn slice_stack_and_sync() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let s0 = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let s1 = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let s2 = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let stacked = b.push(Stack::<f64>::new(0), &[*s0, *s1, *s2][..]);
    let synced = b.push(StackSync::<f64>::new(0), &[*s0, *s1, *s2][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(s0) = Array::scalar(1.0);
    *g.state_mut(s1) = Array::scalar(2.0);
    *g.state_mut(s2) = Array::scalar(3.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(stacked).as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(g.ref_view(synced).as_slice(), &[1.0, 2.0, 3.0]);

    clock.set(ts(2));
    *g.state_mut(s1) = Array::scalar(20.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(stacked).as_slice(), &[1.0, 20.0, 3.0]);
    let v = g.ref_view(synced).as_slice();
    assert!(v[0].is_nan());
    assert_eq!(v[1], 20.0);
    assert!(v[2].is_nan());
}

/// `rolling::mean_basic`: window 3 over [1,2,3,6] → false,false, mean 2.0,
/// mean 11/3. Validates Series-input rolling + warm-up gating (returns false
/// until the window is full, suppressing downstream).
#[test]
fn rolling_mean_count_warmup_and_value() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rec = b.push(Record::new(clock.clone()), *src);
    let rm = b.push(RollingMean::<f64>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.ref_view(rm).as_slice(), &[2.0]); // mean(1,2,3)

    clock.set(ts(4));
    *g.state_mut(src) = Array::scalar(6.0);
    g.stabilize(&mut pool);
    assert!((g.ref_view(rm).as_slice()[0] - 11.0 / 3.0).abs() < 1e-12); // mean(2,3,6)
}

// -- Batch 2: arithmetic / rolling / structural parity ----------------------

/// Unary `Negate` + binary `Subtract`/`Divide` (values from `arithmetic` tests).
#[test]
fn arith_unary_and_binary() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, -2.0, 3.0])));
    let neg = b.push(Negate::<f64>::new(), *a);
    let x = b.push_source(RefSource::new(Array::scalar(20.0_f64)));
    let y = b.push_source(RefSource::new(Array::scalar(4.0_f64)));
    let sub = b.push(Subtract::<f64>::new(), (*x, *y));
    let div = b.push(Divide::<f64>::new(), (*x, *y));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, -2.0, 3.0]);
    *g.state_mut(x) = Array::scalar(20.0);
    *g.state_mut(y) = Array::scalar(4.0);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(neg).as_slice(), &[-1.0, 2.0, -3.0]);
    assert_eq!(g.ref_view(sub).as_slice(), &[16.0]);
    assert_eq!(g.ref_view(div).as_slice(), &[5.0]);
}

/// `Min`/`Max`/`Pow` (values from `arithmetic` tests).
#[test]
fn arith_min_max_pow() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 5.0, 3.0])));
    let bb = b.push_source(RefSource::new(Array::from_vec(&[3], vec![2.0_f64, 4.0, 6.0])));
    let mn = b.push(Min::<f64>::new(), (*a, *bb));
    let mx = b.push(Max::<f64>::new(), (*a, *bb));
    let p = b.push(Pow::new(2.0_f64), *a);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 5.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec(&[3], vec![2.0, 4.0, 6.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(mn).as_slice(), &[1.0, 4.0, 3.0]);
    assert_eq!(g.ref_view(mx).as_slice(), &[2.0, 5.0, 6.0]);
    assert_eq!(g.ref_view(p).as_slice(), &[1.0, 25.0, 9.0]);
}

/// `RollingSum`/`RollingVariance` window-3 (values from rolling tests).
#[test]
fn rolling_sum_and_variance() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rec = b.push(Record::new(clock.clone()), *src);
    let rsum = b.push(RollingSum::<f64>::count(3), rec);
    let rvar = b.push(RollingVariance::<f64>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.ref_view(rsum).as_slice(), &[6.0]);
    assert!((g.ref_view(rvar).as_slice()[0] - 2.0 / 3.0).abs() < 1e-10);

    clock.set(ts(4));
    *g.state_mut(src) = Array::scalar(4.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(rsum).as_slice(), &[9.0]); // 2+3+4
}

/// `RollingCovariance` on a `[2]` vector with `y = 2x` (values from cov tests).
#[test]
fn rolling_covariance_2d() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::<f64>::zeros(&[2])));
    let rec = b.push(Record::new(clock.clone()), *src);
    let cov = b.push(RollingCovariance::<f64>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, x) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::from_vec(&[2], vec![x, 2.0 * x]);
        g.stabilize(&mut pool);
    }
    let c = g.ref_view(cov).as_slice(); // [2,2]
    assert!((c[0] - 2.0 / 3.0).abs() < 1e-10); // Var(x)
    assert!((c[1] - 4.0 / 3.0).abs() < 1e-10); // Cov(x,y)
    assert!((c[3] - 8.0 / 3.0).abs() < 1e-10); // Var(y)
}

/// `Ema` window-2 (value from ema_two_values).
#[test]
fn ema_two_values() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rec = b.push(Record::new(clock.clone()), *src);
    let e = b.push(Ema::<f64>::new(0.5, 2), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
    assert!((g.ref_view(e).as_slice()[0] - expected).abs() < 1e-10);
}

/// `Where` / `Cast` / `Id` (values from their legacy tests).
#[test]
fn structural_where_cast_id() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 5.0, 2.0])));
    let w = b.push(Where::new(|v: f64| v > 3.0, 0.0_f64), *a);
    let i = b.push(Id::<Array<f64>>::new(), *a);
    let ci = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1_i32, 2, 3])));
    let c = b.push(Cast::<i32, f64>::new(), *ci);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 5.0, 2.0]);
    *g.state_mut(ci) = Array::from_vec(&[3], vec![1_i32, 2, 3]);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(w).as_slice(), &[0.0, 5.0, 0.0]);
    assert_eq!(g.ref_view(i).as_slice(), &[1.0, 5.0, 2.0]);
    assert_eq!(g.ref_view(c).as_slice(), &[1.0, 2.0, 3.0]);
}

// -- Batch 3: num tail (element-wise / cross-tick / cross-sectional) --------

/// `Clamp`, `Fillna`, `ForwardFill` (single-shot, values from legacy tests).
#[test]
fn num_clamp_fillna_ffill() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 3.0, 7.0])));
    let clamp = b.push(Clamp::new(2.0_f64, 5.0), *a);
    let na = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, f64::NAN, 3.0])));
    let fill = b.push(Fillna::new(0.0_f64), *na);
    let ff = b.push(ForwardFill::<f64>::new(), *na);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 3.0, 7.0]);
    *g.state_mut(na) = Array::from_vec(&[3], vec![1.0, f64::NAN, 3.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.ref_view(clamp).as_slice(), &[2.0, 3.0, 5.0]);
    assert_eq!(g.ref_view(fill).as_slice(), &[1.0, 0.0, 3.0]);
    let v = g.ref_view(ff).as_slice();
    assert_eq!(v[0], 1.0);
    assert!(v[1].is_nan());
    assert_eq!(v[2], 3.0);
}

/// `Diff` / `PctChange` across ticks (NaN on first, then differences/returns).
#[test]
fn num_diff_and_pct_change() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let d = b.push(Diff::<f64>::new(), *src);
    let pc = b.push(PctChange::<f64>::new(), *src);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(src) = Array::scalar(100.0);
    g.stabilize(&mut pool);
    assert!(g.ref_view(d).as_slice()[0].is_nan());
    assert!(g.ref_view(pc).as_slice()[0].is_nan());

    clock.set(ts(2));
    *g.state_mut(src) = Array::scalar(110.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(d).as_slice()[0], 10.0);
    assert!((g.ref_view(pc).as_slice()[0] - 0.1).abs() < 1e-12);
}

/// Cross-sectional `Gaussianize` / `Percentile` / `Standardize` / `Winsorize`
/// (values from their legacy tests).
#[test]
fn num_cross_sectional() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let five = b.push_source(RefSource::new(Array::from_vec(&[5], vec![30.0_f64, 10.0, 50.0, 20.0, 40.0])));
    let gau = b.push(Gaussianize::<f64>::new(), *five);
    let pct = b.push(Percentile::<f64>::new(), *five);
    let std_in = b.push_source(RefSource::new(Array::from_vec(&[5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0])));
    let zsc = b.push(Standardize::<f64>::new(), *std_in);
    let win_in = b.push_source(RefSource::new(Array::from_vec(&[10], (0..10).map(|i| i as f64).collect())));
    let win = b.push(Winsorize::<f64>::new(0.1), *win_in);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(five) = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0]);
    *g.state_mut(std_in) = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    *g.state_mut(win_in) = Array::from_vec(&[10], (0..10).map(|i| i as f64).collect());
    g.stabilize(&mut pool);

    // Gaussianize: middle (30) → Φ⁻¹(0.5) = 0; symmetric ranks sum to 0.
    let gv = g.ref_view(gau).as_slice();
    assert!((gv[0] - 0.0).abs() < 1e-9);
    assert!(gv.iter().sum::<f64>().abs() < 1e-9);
    // Percentile: (0.5,1.5,2.5,3.5,4.5)/5 by rank.
    let pv = g.ref_view(pct).as_slice();
    assert!((pv[1] - 0.1).abs() < 1e-12 && (pv[0] - 0.5).abs() < 1e-12 && (pv[2] - 0.9).abs() < 1e-12);
    // Standardize: zero mean, unit pop-variance.
    let zv = g.ref_view(zsc).as_slice();
    assert!((zv.iter().sum::<f64>() / 5.0).abs() < 1e-12);
    assert!((zv.iter().map(|&x| x * x).sum::<f64>() / 5.0 - 1.0).abs() < 1e-12);
    // Winsorize p=0.1 over [0..9] → clip to [1, 8].
    assert_eq!(g.ref_view(win).as_slice(), &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]);
}

// -- Batch 4: transform / reshape (Map, Apply, Select, Lag, Concat) ---------

/// `Map` (allocating S→T) doubling a scalar.
#[test]
fn map_doubles() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let m = b.push(
        Map::new(|a: &Array<f64>| {
            let mut o = a.clone();
            o[0] *= 2.0;
            o
        }),
        *src,
    );
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(src) = Array::scalar(5.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(m).as_slice(), &[10.0]);
}

/// `Apply` (two-input add) and `Select` (flat index pick).
#[test]
fn apply_add_and_select() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0])));
    let bb = b.push_source(RefSource::new(Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0])));
    let ap = b.push(
        Apply::<(RefPort<Array<f64>>, RefPort<Array<f64>>), _, _>::new(
            |(a, b): (&Array<f64>, &Array<f64>)| {
                let mut out = a.clone();
                for (o, &v) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
                    *o += v;
                }
                out
            },
        ),
        (*a, *bb),
    );
    let five = b.push_source(RefSource::new(Array::from_vec(&[5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0])));
    let sel = b.push(Select::<f64>::flat(vec![1, 3]), *five);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
    *g.state_mut(five) = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(ap).as_slice(), &[11.0, 22.0, 33.0]);
    assert_eq!(g.ref_view(sel).as_slice(), &[20.0, 40.0]);
}

/// `Lag` (offset 2 over a recorded series).
#[test]
fn lag_offset_two() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rec = b.push(Record::new(clock.clone()), *src);
    let lag = b.push(Lag::new(2, f64::NAN), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.ref_view(lag).as_slice(), &[10.0]); // value from 2 steps ago
}

/// `Concat` axis-0 of two `[2]` arrays → `[4]`.
#[test]
fn concat_axis0() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push_source(RefSource::new(Array::from_vec(&[2], vec![1.0_f64, 2.0])));
    let bb = b.push_source(RefSource::new(Array::from_vec(&[2], vec![3.0_f64, 4.0])));
    let cc = b.push(Concat::<f64>::new(0), &[*a, *bb][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(a) = Array::from_vec(&[2], vec![1.0, 2.0]);
    *g.state_mut(bb) = Array::from_vec(&[2], vec![3.0, 4.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(cc).as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

// -- Batch 5/6: metrics (clock-gated) + stocks -----------------------------

/// Clock-gated `CompoundReturn` / `AverageReturn` / `Volatility` over a price
/// path, firing the clock each tick (values from the metrics tests).
#[test]
fn metrics_clock_gated() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let tick = b.push_source(RefSource::new(()));
    let cr = b.push(CompoundReturn::<f64>::new(), (*data, *tick));
    let ar = b.push(AverageReturn::<f64>::new(), (*data, *tick));
    let vol = b.push(Volatility::<f64>::new(), (*data, *tick));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 100.0), (2, 110.0)] {
        clock.set(ts(t));
        *g.state_mut(data) = Array::scalar(v);
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool);
    }
    assert!((g.ref_view(cr).as_slice()[0] - 0.10).abs() < 1e-10);
    assert!((g.ref_view(ar).as_slice()[0] - 0.10).abs() < 1e-10);
    assert_eq!(g.ref_view(vol).as_slice()[0], 0.0); // single return → zero std

    clock.set(ts(3));
    *g.state_mut(data) = Array::scalar(99.0);
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool);
    assert!(g.ref_view(ar).as_slice()[0].abs() < 1e-10); // 0.10, -0.10 → 0
    assert!((g.ref_view(vol).as_slice()[0] - 0.10).abs() < 1e-10); // std 0.10
}

/// `Drawdown` (single input, no clock) from the running maximum.
#[test]
fn metrics_drawdown() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let dd = b.push(Drawdown::<f64>::new(), *data);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v, e) in [(1_i64, 100.0, 0.0), (2, 120.0, 0.0), (3, 90.0, -0.25)] {
        clock.set(ts(t));
        *g.state_mut(data) = Array::scalar(v);
        g.stabilize(&mut pool);
        assert!((g.ref_view(dd).as_slice()[0] - e).abs() < 1e-10);
    }
}

/// `Annualize`: YTD [2024, day 91, 100, 20] → annualized × 365/91.
#[test]
fn stocks_annualize() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::from_vec(&[4], vec![2024.0_f64, 91.0, 100.0, 20.0])));
    let ann = b.push(Annualize::new(), *src);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.state_mut(src) = Array::from_vec(&[4], vec![2024.0, 91.0, 100.0, 20.0]);
    g.stabilize(&mut pool);
    let o = g.ref_view(ann).as_slice();
    assert!((o[0] - 100.0 * 365.0 / 91.0).abs() < 1e-10);
    assert!((o[1] - 20.0 * 365.0 / 91.0).abs() < 1e-10);
}

/// `ForwardAdjust`: price-only tick, then a cash dividend (message-passing on
/// the two inputs). 9.5 with a 0.5 cash dividend forward-adjusts back to 10.0.
#[test]
fn stocks_forward_adjust() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let price = b.push_source(RefSource::new(Array::scalar(10.0_f64)));
    let divd = b.push_source(RefSource::new(Array::from_vec(&[2], vec![0.0_f64, 0.0])));
    let fa = b.push(ForwardAdjust::new(), (*price, *divd));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // gen1: price only.
    clock.set(ts(1));
    *g.state_mut(price) = Array::scalar(10.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(fa).as_slice(), &[10.0]);

    // gen2: price 9.5 + cash dividend 0.5 → adjusted back to 10.0.
    clock.set(ts(2));
    *g.state_mut(price) = Array::scalar(9.5);
    *g.state_mut(divd) = Array::from_vec(&[2], vec![0.0, 0.5]);
    g.stabilize(&mut pool);
    assert!((g.ref_view(fa).as_slice()[0] - 10.0).abs() < 1e-12);
}

// ===========================================================================
// Parallel-execution gate (Pool::new(N>0))
// ===========================================================================

/// One source fans out to K independent Filter(>3)→Record chains across 8
/// worker threads; every branch must record exactly the passing values.
#[test]
fn parallel_fanout_matches_sequential() {
    const K: usize = 16;
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let recs: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Filter(|a: &Array<f64>| a[0] > 3.0), *src);
            b.push(Record::new(clock.clone()), f)
        })
        .collect();
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(8);

    let seq = [1.0_f64, 5.0, 2.0, 10.0, 4.0, 0.5, 7.0, 3.0, 9.0];
    let expected: Vec<f64> = seq.iter().copied().filter(|&v| v > 3.0).collect();
    for (i, &v) in seq.iter().enumerate() {
        clock.set(ts(i as i64 + 1));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &rec in &recs {
        assert_eq!(g.ref_view(rec).values(), &expected[..], "a parallel branch diverged");
    }
}

/// K concurrent stateful Count chains behind per-branch gates over 500
/// generations on 8 workers; each Count must equal the number of passing gens.
#[test]
fn parallel_stress_stateful_counts() {
    const K: usize = 12;
    const GENS: usize = 500;
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let cnts: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Filter(|a: &Array<f64>| a[0] > 0.0), *src);
            b.push(Count, f)
        })
        .collect();
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(8);

    let mut passes = 0.0_f64;
    for i in 0..GENS {
        let v = if i % 3 == 0 { 1.0 } else { -1.0 };
        if v > 0.0 {
            passes += 1.0;
        }
        clock.set(ts(i as i64 + 1));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &c in &cnts {
        assert_eq!(g.ref_view(c).as_slice(), &[passes], "a parallel Count raced");
    }
}

// ===========================================================================
// Split + segment fusion
// ===========================================================================

/// `Split` fans a `[3, 2]` panel into per-row **zero-copy view** ports holding
/// the row values; rows notify exactly when the panel does (a downstream
/// `FilterView → Count` chain advances only on panel pokes, not on unrelated
/// generations).
#[test]
fn split_rows_notify_with_panel() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec(&[3, 2], vec![0.0_f64; 6])));
    let other = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let rows = b.push(Split::<f64>::new(3), *panel);
    assert_eq!(rows.len(), 3);
    fn pass(_: ArraySlice<'_, f64>) -> bool {
        true
    }
    let counts: Vec<_> = rows
        .iter()
        .map(|&r| {
            let gated = b.push(Gate(pass), r);
            let mat = b.push(SelectView::<f64>::flat(vec![0, 1]), gated);
            b.push(Count, mat)
        })
        .collect();
    let _sink = b.push(Count, *other); // unrelated cone
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // Build values: row views hold the initial panel rows; counters untouched.
    assert_eq!(g.ref_view(rows[0]).as_slice(), &[0.0, 0.0]);
    assert_eq!(g.ref_view(rows[0]).shape(), &[2]);
    for &c in &counts {
        assert_eq!(g.ref_view(c).as_slice(), &[0.0]);
    }

    clock.set(ts(1));
    *g.state_mut(panel) = Array::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(rows[0]).as_slice(), &[1.0, 2.0]);
    assert_eq!(g.ref_view(rows[1]).as_slice(), &[3.0, 4.0]);
    assert_eq!(g.ref_view(rows[2]).as_slice(), &[5.0, 6.0]);
    for &c in &counts {
        assert_eq!(g.ref_view(c).as_slice(), &[1.0]);
    }

    // Poking an unrelated source must not advance the per-row counters.
    clock.set(ts(2));
    *g.state_mut(other) = Array::scalar(1.0);
    g.stabilize(&mut pool);
    for &c in &counts {
        assert_eq!(g.ref_view(c).as_slice(), &[1.0]);
    }
}

/// The declared axis size is validated against the build-time input shape.
#[test]
#[should_panic(expected = "Split: input shape")]
fn split_axis_size_mismatch_panics() {
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec(&[3, 2], vec![0.0_f64; 6])));
    let _ = b.push(Split::<f64>::new(2), *panel);
}

/// The view chain (`Split` rows -> retaining `Gate` -> view-input `Select` /
/// `ForwardAdjust`) is tick-for-tick bit-identical to the owned chain
/// (`Filter` -> owned `Select` / `ForwardAdjust`) over the same source pokes,
/// including the NaN cutoff and the price/dividend message-passing. `Gate` and
/// `Filter` are the same operator over different edge kinds — both honour the
/// no-notify⟹unchanged contract by retaining the last passed row — so their
/// downstream cones must agree bit-for-bit.
#[test]
fn view_chain_matches_owned_chain() {
    fn any_finite_owned(a: &Array<f64>) -> bool {
        a.as_slice().iter().any(|x| x.is_finite())
    }
    fn any_finite_view(a: ArraySlice<'_, f64>) -> bool {
        a.as_slice().iter().any(|x| x.is_finite())
    }
    fn bits(a: &Array<f64>) -> Vec<u64> {
        a.as_slice().iter().map(|x| x.to_bits()).collect()
    }

    let nan = f64::NAN;
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    // A [2, 2] two-stock panel per data kind, split into per-stock rows.
    let prices_panel = b.push_source(RefSource::new(Array::from_vec(&[2, 2], vec![nan; 4])));
    let div_panel = b.push_source(RefSource::new(Array::from_vec(&[2, 2], vec![nan; 4])));
    let prices_rows = b.push(Split::<f64>::new(2), *prices_panel);
    let div_rows = b.push(Split::<f64>::new(2), *div_panel);

    // Owned reference chain for stock 0 (materializes at the row Selects).
    type RowSelect = Select<f64, flowgraph::typed::RefViewPort<ArrayValue<f64>>>;
    let p_f = {
        let m = b.push(RowSelect::flat(vec![0, 1]), prices_rows[0]);
        b.push(Filter(any_finite_owned), m)
    };
    let d_f = {
        let m = b.push(RowSelect::flat(vec![0, 1]), div_rows[0]);
        b.push(Filter(any_finite_owned), m)
    };
    let close = b.push(Select::<f64>::new(vec![0], 0, true), p_f);
    let adj = b.push(ForwardAdjust::new().with_output_prices(false), (close, d_f));
    let adjusted = b.push(Multiply::<f64>::new(), (close, adj));

    // Zero-copy view chain for stock 0 (materializes at Select).
    let p_g = b.push(Gate(any_finite_view), prices_rows[0]);
    let d_g = b.push(Gate(any_finite_view), div_rows[0]);
    let v_close = b.push(SelectView::<f64>::new(vec![0], 0, true), p_g);
    let v_adj = b.push(
        ForwardAdjustViewDiv::default().with_output_prices(false),
        (v_close, d_g),
    );
    let v_adjusted = b.push(Multiply::<f64>::new(), (v_close, v_adj));

    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // (prices rows, dividend rows) per tick for both stocks; None = not poked.
    type Tick = (Option<[f64; 4]>, Option<[f64; 4]>);
    let ticks: &[Tick] = &[
        (Some([10.0, 100.0, 7.0, 70.0]), None),
        (Some([12.0, 110.0, nan, nan]), Some([0.0, 2.0, nan, nan])), // cash div, stock1 idle
        (None, None),                                                // idle generation
        (Some([11.0, 90.0, 8.0, 80.0]), None),
        (Some([nan, nan, 9.0, 90.0]), Some([0.5, 0.0, nan, nan])), // share div on no-data tick
        (Some([13.0, 95.0, nan, nan]), None),
    ];
    for (i, (p, d)) in ticks.iter().enumerate() {
        clock.set(ts(i as i64 + 1));
        if let Some(p) = p {
            *g.state_mut(prices_panel) = Array::from_vec(&[2, 2], p.to_vec());
        }
        if let Some(d) = d {
            *g.state_mut(div_panel) = Array::from_vec(&[2, 2], d.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(bits(g.ref_view(adjusted)), bits(g.ref_view(v_adjusted)), "tick {i}: adjusted");
        assert_eq!(bits(g.ref_view(adj)), bits(g.ref_view(v_adj)), "tick {i}: adjusts");
    }
}

/// A carry `StackView` over zero-copy `SliceView` rows is tick-for-tick
/// bit-identical to an owned `Stack` over `Select` rows, even as stocks go
/// silent for several generations. This is the regression test for the
/// view-edge carry join: `StackView` reads **every** input each generation
/// (incl. un-notified ones), and the contract guarantees an un-notified
/// `SliceView` reads the retaining `Gate`'s frozen last value — exactly what
/// the owned `Select`'s retained buffer holds. Both gate off the same `Gate`.
#[test]
fn view_join_carry_matches_owned_join() {
    fn any_finite_view(a: ArraySlice<'_, f64>) -> bool {
        a.as_slice().iter().any(|x| x.is_finite())
    }
    fn bits(a: &Array<f64>) -> Vec<u64> {
        a.as_slice().iter().map(|x| x.to_bits()).collect()
    }

    let nan = f64::NAN;
    let n = 3usize;
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec(&[n, 2], vec![nan; n * 2])));
    let rows = b.push(Split::<f64>::new(n), *panel);

    // Each stock's retaining `Gate` fans out to BOTH an owned `Select` (→ owned
    // carry `Stack`) and a zero-copy `SliceView` (→ view carry `StackView`).
    let mut owned_v = Vec::with_capacity(n);
    let mut view_v = Vec::with_capacity(n);
    for &r in rows.iter() {
        let g = b.push(Gate(any_finite_view), r);
        owned_v.push(b.push(SelectView::<f64>::new(vec![0], 0, true), g));
        view_v.push(b.push(SliceView::new(vec![0], 0, true), g));
    }
    let owned_join = b.push(Stack::<f64>::new(0), &owned_v[..]);
    let view_join = b.push(StackView::<f64>::new(0), &view_v[..]);

    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // Full `[3, 2]` panels per tick (column 0 is the value); `None` = idle gen
    // (panel not poked). An all-NaN row is a stock with no data that tick.
    type Tick = Option<[f64; 6]>;
    let ticks: &[Tick] = &[
        Some([1.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        Some([nan, nan, 5.0, 0.0, 6.0, 0.0]), // stock 0 silent → carries 1
        None,                                  // idle generation → all carry
        Some([7.0, 0.0, nan, nan, 8.0, 0.0]), // stock 1 silent → carries 5
        Some([nan, nan, nan, nan, nan, nan]),  // all silent → all carry
        Some([9.0, 0.0, nan, nan, nan, nan]),  // only stock 0 ticks
    ];
    for (i, t) in ticks.iter().enumerate() {
        clock.set(ts(i as i64 + 1));
        if let Some(p) = t {
            *g.state_mut(panel) = Array::from_vec(&[n, 2], p.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(
            bits(g.ref_view(view_join)),
            bits(g.ref_view(owned_join)),
            "tick {i}: view carry join must match owned carry join",
        );
    }
    // Sanity: the final carried cross-section is what we expect (s0=9 fresh,
    // s1=5 last-seen at tick 1, s2=8 last-seen at tick 3).
    assert_eq!(g.ref_view(view_join).as_slice(), &[9.0, 5.0, 8.0]);
}

/// A fused `segment!` chain (Filter -> Selects -> ForwardAdjust -> Multiply)
/// is tick-for-tick bit-identical to the same operators as separate nodes,
/// including the NaN cutoff and the price/dividend message-passing.
#[test]
fn fused_segment_matches_unfused_nodes() {
    fn any_finite(a: &Array<f64>) -> bool {
        a.as_slice().iter().any(|x| x.is_finite())
    }
    fn bits(a: &Array<f64>) -> Vec<u64> {
        a.as_slice().iter().map(|x| x.to_bits()).collect()
    }

    let fused = flowgraph::segment!(|prices_row: RefPort<Array<f64>>, div_row: RefPort<Array<f64>>|
        -> (RefPort<Array<f64>>, RefPort<Array<f64>>) {
        let prices = prices_row => Filter(any_finite);
        let dividends = div_row => Filter(any_finite);
        let close = prices => Select::<f64>::new(vec![0], 0, true);
        let adjusts = (close, dividends) => ForwardAdjust::new().with_output_prices(false);
        let adjusted = (close, adjusts) => Multiply::<f64>::new();
        (adjusted, adjusts)
    });

    let nan = f64::NAN;
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let prices = b.push_source(RefSource::new(Array::from_vec(&[2], vec![nan; 2])));
    let div = b.push_source(RefSource::new(Array::from_vec(&[2], vec![nan; 2])));

    // Reference: the same chain as separate nodes.
    let p_f = b.push(Filter(any_finite), *prices);
    let d_f = b.push(Filter(any_finite), *div);
    let close = b.push(Select::<f64>::new(vec![0], 0, true), p_f);
    let adj = b.push(ForwardAdjust::new().with_output_prices(false), (close, d_f));
    let adjusted = b.push(Multiply::<f64>::new(), (close, adj));

    let (f_adjusted, f_adj) = b.push(fused, (*prices, *div));
    let mut g = Graph::from_builder(b);
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
        clock.set(ts(i as i64 + 1));
        if let Some(p) = p {
            *g.state_mut(prices) = Array::from_vec(&[2], p.to_vec());
        }
        if let Some(d) = d {
            *g.state_mut(div) = Array::from_vec(&[2], d.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(bits(g.ref_view(adjusted)), bits(g.ref_view(f_adjusted)), "tick {i}: adjusted");
        assert_eq!(bits(g.ref_view(adj)), bits(g.ref_view(f_adj)), "tick {i}: adjusts");
    }
}
