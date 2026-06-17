//! Engine tests: each graph asserts a known-good output (the constants were
//! originally lifted from the retired reference engine's test suite and are now
//! the canonical expected values), plus a parallel-execution gate that runs the
//! gating / clock / concurrent-write paths under real worker threads.
//!
//! [array-view-refactor] Migrated to the const-rank strided view currency: every
//! array-shaped edge is a `ViewPort<ArrayValue<T, N>>` carrying an
//! `ArrayView<'a, T, N>` by value. Sources stay whole-array `RefSource`s and are
//! bridged into the view currency with [`AsView`]; outputs are read with
//! `g.view(h).contiguous_slice()` (view edges) or `g.ref_view(h)` (Series edges).
//! Arithmetic uses the lowercase free constructors (`add`/`negate`/…); the
//! rank-changers carry explicit out-rank generics.

use flowgraph::core::Pool;
use flowgraph::typed::{Graph, GraphBuilder, Handle, RefSource, RefViewPort, SourceHandle, ViewPort};

use super::*;
use crate::operators::op::ArrayValue;
use crate::{Array, ArrayView, Instant, Retention, Series};

fn ts(n: i64) -> Instant {
    Instant::from_nanos(n)
}

/// A rank-`N` view port — the array edge currency.
type Vp<const N: usize> = ViewPort<ArrayValue<f64, N>>;
/// A rank-`N` by-reference view port — the leaf kind that feeds the carry-join
/// combine operators ([`Stack`] / [`Concat`], whose inputs are `RefViewPorts`).
type RVp<const N: usize> = RefViewPort<ArrayValue<f64, N>>;

/// Push a `RefSource` of a scalar plus an `AsView` bridge; return the source
/// handle (for `state_mut`) and the view handle (for wiring).
fn scalar_src(
    b: &mut GraphBuilder,
    v: f64,
) -> (SourceHandle<RefSource<Array<f64, 0>>>, Handle<Vp<0>>) {
    let s = b.push_source(RefSource::new(Array::scalar(v)));
    let view = b.push(AsView::<f64, 0>::new(), *s);
    (s, view)
}

/// Push a `RefSource` of a rank-1 array plus an `AsView` bridge.
fn vec_src(
    b: &mut GraphBuilder,
    v: Vec<f64>,
) -> (SourceHandle<RefSource<Array<f64, 1>>>, Handle<Vp<1>>) {
    let s = b.push_source(RefSource::new(Array::from_vec([v.len()], v)));
    let view = b.push(AsView::<f64, 1>::new(), *s);
    (s, view)
}

/// Push an independent rank-0 source whose value is exposed as a **by-reference**
/// view (`RefViewPort`) — the leaf kind the carry-join combines (`Stack` /
/// `StackSync` / `Concat`) consume. The only `RefViewPort` array producer is
/// [`Split`] (the by-reference fan-out), so the bridge wraps the scalar as a
/// `[1]` array and splits axis-0 into its single rank-0 row. Each source pokes
/// independently, so the row notifies independently — preserving the original
/// per-element notify semantics the carry tests rely on.
fn scalar_ref_src(
    b: &mut GraphBuilder,
    v: f64,
) -> (SourceHandle<RefSource<Array<f64, 1>>>, Handle<RVp<0>>) {
    let s = b.push_source(RefSource::new(Array::from_vec([1], vec![v])));
    let view = b.push(AsView::<f64, 1>::new(), *s);
    let rows = b.push(Split::<f64, 1, 0>::new(1), view);
    (s, rows[0])
}

/// Like [`scalar_ref_src`] but for a rank-1 value (wraps it as `[1, len]` and
/// splits the leading axis into its single rank-1 row — feeds `Concat`).
fn vec_ref_src(
    b: &mut GraphBuilder,
    v: Vec<f64>,
) -> (SourceHandle<RefSource<Array<f64, 2>>>, Handle<RVp<1>>) {
    let len = v.len();
    let s = b.push_source(RefSource::new(Array::from_vec([1, len], v)));
    let view = b.push(AsView::<f64, 2>::new(), *s);
    let rows = b.push(Split::<f64, 2, 1>::new(1), view);
    (s, rows[0])
}

// ===========================================================================
// Sequential execution (Pool::new(0): single-threaded stabilization)
// ===========================================================================

/// `scenario_simple_add`: 10 + 3 == 13.
#[test]
fn simple_add() {
    let mut b = GraphBuilder::new();
    let (ha, hav) = scalar_src(&mut b, 0.0);
    let (hb, hbv) = scalar_src(&mut b, 0.0);
    let hc = b.push(add::<f64, 0>(), (hav, hbv));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(ha) = Array::scalar(10.0);
    *g.state_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(hc).contiguous_slice().unwrap(), &[13.0]);
}

/// `scenario_chain`: (2 + 3) * 2 == 10. Exercises a 3-deep cone where the leaf
/// is read by two operators (add and multiply both read `a`).
#[test]
fn chain_add_then_mul() {
    let mut b = GraphBuilder::new();
    let (a, av) = scalar_src(&mut b, 0.0);
    let (bb, bv) = scalar_src(&mut b, 0.0);
    let ab = b.push(add::<f64, 0>(), (av, bv));
    let out = b.push(multiply::<f64, 0>(), (ab, av));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::scalar(2.0);
    *g.state_mut(bb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(out).contiguous_slice().unwrap(), &[10.0]);
}

/// `scenario_record`: record (10+3), (20+7) → series [13, 27] @ [1, 2].
#[test]
fn record_series() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let (ha, hav) = scalar_src(&mut b, 0.0);
    let (hb, hbv) = scalar_src(&mut b, 0.0);
    let sum = b.push(add::<f64, 0>(), (hav, hbv));
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), sum);
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let flt = b.push(Filter::<_, 0>(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 3.0), srcv);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), flt);
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), srcv);
    let lst = b.push(Last::<f64, 0>::new(0.0_f64), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.view(lst).contiguous_slice().unwrap(), &[20.0]);
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    // RollingMean(5) reads back 6 on eviction, Lag(3) back 4 → retain 8 covers both.
    let rec = b.push(
        Record::<f64, 0>::with_retention(clock.clone(), Retention::count(8)),
        srcv,
    );
    let lag = b.push(Lag::<f64, 0>::new(3, f64::NAN), rec);
    let rmean = b.push(RollingMean::<f64, 0>::count(5), rec);
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
                (g.view(rmean).contiguous_slice().unwrap()[0] - (t as f64 - 2.0)).abs() < 1e-9,
                "rmean@{t} = {}",
                g.view(rmean).contiguous_slice().unwrap()[0],
            );
        }
        if t > 3 {
            // Lag(3): the value from 3 steps ago == t - 3.
            assert!(
                (g.view(lag).contiguous_slice().unwrap()[0] - (t as f64 - 3.0)).abs() < 1e-9,
                "lag@{t} = {}",
                g.view(lag).contiguous_slice().unwrap()[0],
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
    let (data, datav) = scalar_src(&mut b, 0.0);
    let tick = b.push_source(RefSource::new(()));
    let gated = b.push(
        Clocked::<_, ()>::new(Filter::<_, 0>(|_: ArrayView<f64, 0>| true)),
        (*tick, datav),
    );
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), gated);
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
    let (a, av) = scalar_src(&mut b, 0.0);
    let (bb, bv) = scalar_src(&mut b, 0.0);
    let sum = b.push(add::<f64, 0>(), (av, bv));
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), sum);
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

/// Per-element notify: gen1 all fire → [1,2,3]; gen2 only s1 → Stack keeps stale
/// [1,20,3]; StackSync NaN-fills [NaN,20,NaN]. The three scalar (rank-0) sources
/// stack along a new axis into a rank-1 cross-section.
#[test]
fn slice_stack_and_sync() {
    let mut b = GraphBuilder::new();
    let (s0, s0v) = scalar_ref_src(&mut b, 0.0);
    let (s1, s1v) = scalar_ref_src(&mut b, 0.0);
    let (s2, s2v) = scalar_ref_src(&mut b, 0.0);
    let stacked = b.push(Stack::<f64, 0, 1>::new(0), &[s0v, s1v, s2v][..]);
    let synced = b.push(StackSync::<f64, 0, 1>::new(0), &[s0v, s1v, s2v][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(s0) = Array::from_vec([1], vec![1.0]);
    *g.state_mut(s1) = Array::from_vec([1], vec![2.0]);
    *g.state_mut(s2) = Array::from_vec([1], vec![3.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(stacked).contiguous_slice().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(g.view(synced).contiguous_slice().unwrap(), &[1.0, 2.0, 3.0]);

    *g.state_mut(s1) = Array::from_vec([1], vec![20.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(stacked).contiguous_slice().unwrap(), &[1.0, 20.0, 3.0]);
    let v = g.view(synced).contiguous_slice().unwrap();
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), srcv);
    let rm = b.push(RollingMean::<f64, 0>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(rm).contiguous_slice().unwrap(), &[2.0]); // mean(1,2,3)

    clock.set(ts(4));
    *g.state_mut(src) = Array::scalar(6.0);
    g.stabilize(&mut pool);
    assert!((g.view(rm).contiguous_slice().unwrap()[0] - 11.0 / 3.0).abs() < 1e-12); // mean(2,3,6)
}

// -- Batch 2: arithmetic / rolling / structural parity ----------------------

/// Unary `negate` + binary `subtract`/`divide` (values from `arithmetic` tests).
#[test]
fn arith_unary_and_binary() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_src(&mut b, vec![1.0_f64, -2.0, 3.0]);
    let neg = b.push(negate::<f64, 1>(), av);
    let (x, xv) = scalar_src(&mut b, 20.0);
    let (y, yv) = scalar_src(&mut b, 4.0);
    let sub = b.push(subtract::<f64, 0>(), (xv, yv));
    let div = b.push(divide::<f64, 0>(), (xv, yv));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, -2.0, 3.0]);
    *g.state_mut(x) = Array::scalar(20.0);
    *g.state_mut(y) = Array::scalar(4.0);
    g.stabilize(&mut pool);

    assert_eq!(g.view(neg).contiguous_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    assert_eq!(g.view(sub).contiguous_slice().unwrap(), &[16.0]);
    assert_eq!(g.view(div).contiguous_slice().unwrap(), &[5.0]);
}

/// `min`/`max`/`pow` (values from `arithmetic` tests).
#[test]
fn arith_min_max_pow() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 5.0, 3.0]);
    let (bb, bv) = vec_src(&mut b, vec![2.0_f64, 4.0, 6.0]);
    let mn = b.push(min::<f64, 1>(), (av, bv));
    let mx = b.push(max::<f64, 1>(), (av, bv));
    let p = b.push(pow::<f64, 1>(2.0), av);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 5.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec([3], vec![2.0, 4.0, 6.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(mn).contiguous_slice().unwrap(), &[1.0, 4.0, 3.0]);
    assert_eq!(g.view(mx).contiguous_slice().unwrap(), &[2.0, 5.0, 6.0]);
    assert_eq!(g.view(p).contiguous_slice().unwrap(), &[1.0, 25.0, 9.0]);
}

/// `RollingSum`/`RollingVariance` window-3 (values from rolling tests).
#[test]
fn rolling_sum_and_variance() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), srcv);
    let rsum = b.push(RollingSum::<f64, 0>::count(3), rec);
    let rvar = b.push(RollingVariance::<f64, 0>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(rsum).contiguous_slice().unwrap(), &[6.0]);
    assert!((g.view(rvar).contiguous_slice().unwrap()[0] - 2.0 / 3.0).abs() < 1e-10);

    clock.set(ts(4));
    *g.state_mut(src) = Array::scalar(4.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(rsum).contiguous_slice().unwrap(), &[9.0]); // 2+3+4
}

/// `RollingCovariance` on a `[2]` vector with `y = 2x` (values from cov tests).
#[test]
fn rolling_covariance_2d() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let (src, srcv) = vec_src(&mut b, vec![0.0_f64; 2]);
    let rec = b.push(Record::<f64, 1>::new(clock.clone()), srcv);
    let cov = b.push(RollingCovariance::<f64>::count(3), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, x) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
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
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), srcv);
    let e = b.push(Ema::<f64, 0>::new(0.5, 2), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
    assert!((g.view(e).contiguous_slice().unwrap()[0] - expected).abs() < 1e-10);
}

/// `Where` / `Cast` / `Id` (values from their legacy tests).
#[test]
fn structural_where_cast_id() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 5.0, 2.0]);
    let w = b.push(Where::<f64, _, 1>::new(|v: f64| v > 3.0, 0.0_f64), av);
    // `Id` is currency-agnostic — exercise it on the owned-array source cell.
    let i = b.push(Id::<Array<f64, 1>>::new(), *a);
    let ci = b.push_source(RefSource::new(Array::from_vec([3], vec![1_i32, 2, 3])));
    let civ = b.push(AsView::<i32, 1>::new(), *ci);
    let c = b.push(Cast::<i32, f64, 1>::new(), civ);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 5.0, 2.0]);
    *g.state_mut(ci) = Array::from_vec([3], vec![1_i32, 2, 3]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(w).contiguous_slice().unwrap(), &[0.0, 5.0, 0.0]);
    assert_eq!(g.ref_view(i).as_slice(), &[1.0, 5.0, 2.0]);
    assert_eq!(g.view(c).contiguous_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

// -- Batch 3: num tail (element-wise / cross-tick / cross-sectional) --------

/// `Clamp`, `Fillna`, `ForwardFill` (single-shot, values from legacy tests).
#[test]
fn num_clamp_fillna_ffill() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 3.0, 7.0]);
    let clamp = b.push(Clamp::<f64, 1>::new(2.0, 5.0), av);
    let (na, nav) = vec_src(&mut b, vec![1.0_f64, f64::NAN, 3.0]);
    let fill = b.push(Fillna::<f64, 1>::new(0.0), nav);
    let ff = b.push(ForwardFill::<f64, 1>::new(), nav);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 3.0, 7.0]);
    *g.state_mut(na) = Array::from_vec([3], vec![1.0, f64::NAN, 3.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.view(clamp).contiguous_slice().unwrap(), &[2.0, 3.0, 5.0]);
    assert_eq!(g.view(fill).contiguous_slice().unwrap(), &[1.0, 0.0, 3.0]);
    let v = g.view(ff).contiguous_slice().unwrap();
    assert_eq!(v[0], 1.0);
    assert!(v[1].is_nan());
    assert_eq!(v[2], 3.0);
}

/// `Diff` / `PctChange` across ticks (NaN on first, then differences/returns).
#[test]
fn num_diff_and_pct_change() {
    let mut b = GraphBuilder::new();
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let d = b.push(Diff::<f64, 0>::new(), srcv);
    let pc = b.push(PctChange::<f64, 0>::new(), srcv);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::scalar(100.0);
    g.stabilize(&mut pool);
    assert!(g.view(d).contiguous_slice().unwrap()[0].is_nan());
    assert!(g.view(pc).contiguous_slice().unwrap()[0].is_nan());

    *g.state_mut(src) = Array::scalar(110.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(d).contiguous_slice().unwrap()[0], 10.0);
    assert!((g.view(pc).contiguous_slice().unwrap()[0] - 0.1).abs() < 1e-12);
}

/// Cross-sectional `Gaussianize` / `Percentile` / `Standardize` / `Winsorize`
/// (values from their legacy tests).
#[test]
fn num_cross_sectional() {
    let mut b = GraphBuilder::new();
    let (five, fivev) = vec_src(&mut b, vec![30.0_f64, 10.0, 50.0, 20.0, 40.0]);
    let gau = b.push(Gaussianize::<f64, 1>::new(), fivev);
    let pct = b.push(Percentile::<f64, 1>::new(), fivev);
    let (std_in, std_inv) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
    let zsc = b.push(Standardize::<f64, 1>::new(), std_inv);
    let (win_in, win_inv) = vec_src(&mut b, (0..10).map(|i| i as f64).collect());
    let win = b.push(Winsorize::<f64, 1>::new(0.1), win_inv);
    let mut g = Graph::from_builder(b);
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
    assert!((pv[1] - 0.1).abs() < 1e-12 && (pv[0] - 0.5).abs() < 1e-12 && (pv[2] - 0.9).abs() < 1e-12);
    // Standardize: zero mean, unit pop-variance.
    let zvb = g.view(zsc).to_contiguous();
    let zv: &[f64] = &zvb;
    assert!((zv.iter().sum::<f64>() / 5.0).abs() < 1e-12);
    assert!((zv.iter().map(|&x| x * x).sum::<f64>() / 5.0 - 1.0).abs() < 1e-12);
    // Winsorize p=0.1 over [0..9] → clip to [1, 8].
    assert_eq!(
        g.view(win).contiguous_slice().unwrap(),
        &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]
    );
}

// -- Batch 4: transform / reshape (Map, Apply, Select, Lag, Concat) ---------

/// `Map` (allocating SI→SO) doubling a scalar.
#[test]
fn map_doubles() {
    let mut b = GraphBuilder::new();
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let m = b.push(
        Map::new(|a: ArrayView<f64, 0>| {
            let mut o = a.to_array();
            o[0] *= 2.0;
            o
        }),
        srcv,
    );
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::scalar(5.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(m).contiguous_slice().unwrap(), &[10.0]);
}

/// `Apply` (two-input add) and `Select` (flat index pick).
#[test]
fn apply_add_and_select() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_src(&mut b, vec![1.0_f64, 2.0, 3.0]);
    let (bb, bv) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0]);
    let ap = b.push(
        Apply::<(Vp<1>, Vp<1>), _, 1, _>::new(
            |(a, b): (ArrayView<f64, 1>, ArrayView<f64, 1>)| {
                let mut out = a.to_array();
                for (o, v) in out.as_mut_slice().iter_mut().zip(b.to_contiguous().iter()) {
                    *o += *v;
                }
                out
            },
        ),
        (av, bv),
    );
    let (five, fivev) = vec_src(&mut b, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
    let sel = b.push(Select::<f64, 1, 1>::flat(vec![1, 3]), fivev);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    *g.state_mut(bb) = Array::from_vec([3], vec![10.0, 20.0, 30.0]);
    *g.state_mut(five) = Array::from_vec([5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(ap).contiguous_slice().unwrap(), &[11.0, 22.0, 33.0]);
    assert_eq!(g.view(sel).contiguous_slice().unwrap(), &[20.0, 40.0]);
}

/// `Lag` (offset 2 over a recorded series).
#[test]
fn lag_offset_two() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let rec = b.push(Record::<f64, 0>::new(clock.clone()), srcv);
    let lag = b.push(Lag::<f64, 0>::new(2, f64::NAN), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        clock.set(ts(t));
        *g.state_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.view(lag).contiguous_slice().unwrap(), &[10.0]); // value from 2 steps ago
}

/// `Concat` axis-0 of two `[2]` arrays → `[4]`.
#[test]
fn concat_axis0() {
    let mut b = GraphBuilder::new();
    let (a, av) = vec_ref_src(&mut b, vec![1.0_f64, 2.0]);
    let (bb, bv) = vec_ref_src(&mut b, vec![3.0_f64, 4.0]);
    let cc = b.push(Concat::<f64, 1>::new(0), &[av, bv][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(a) = Array::from_vec([1, 2], vec![1.0, 2.0]);
    *g.state_mut(bb) = Array::from_vec([1, 2], vec![3.0, 4.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.view(cc).contiguous_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

// -- Batch 5/6: metrics (clock-gated) + stocks -----------------------------

/// Clock-gated `CompoundReturn` / `AverageReturn` / `Volatility` over a price
/// path, firing the clock each tick (values from the metrics tests).
#[test]
fn metrics_clock_gated() {
    let mut b = GraphBuilder::new();
    let (data, datav) = scalar_src(&mut b, 0.0);
    let tick = b.push_source(RefSource::new(()));
    let cr = b.push(CompoundReturn::<f64, 0>::new(), (datav, *tick));
    let ar = b.push(AverageReturn::<f64, 0>::new(), (datav, *tick));
    let vol = b.push(Volatility::<f64, 0>::new(), (datav, *tick));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 100.0), (2, 110.0)] {
        let _ = t;
        *g.state_mut(data) = Array::scalar(v);
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool);
    }
    assert!((g.view(cr).contiguous_slice().unwrap()[0] - 0.10).abs() < 1e-10);
    assert!((g.view(ar).contiguous_slice().unwrap()[0] - 0.10).abs() < 1e-10);
    assert_eq!(g.view(vol).contiguous_slice().unwrap()[0], 0.0); // single return → zero std

    *g.state_mut(data) = Array::scalar(99.0);
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool);
    assert!(g.view(ar).contiguous_slice().unwrap()[0].abs() < 1e-10); // 0.10, -0.10 → 0
    assert!((g.view(vol).contiguous_slice().unwrap()[0] - 0.10).abs() < 1e-10); // std 0.10
}

/// `Drawdown` (single input, no clock) from the running maximum.
#[test]
fn metrics_drawdown() {
    let mut b = GraphBuilder::new();
    let (data, datav) = scalar_src(&mut b, 0.0);
    let dd = b.push(Drawdown::<f64, 0>::new(), datav);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (v, e) in [(100.0, 0.0), (120.0, 0.0), (90.0, -0.25)] {
        *g.state_mut(data) = Array::scalar(v);
        g.stabilize(&mut pool);
        assert!((g.view(dd).contiguous_slice().unwrap()[0] - e).abs() < 1e-10);
    }
}

/// `Annualize`: YTD [2024, day 91, 100, 20] → annualized × 365/91.
#[test]
fn stocks_annualize() {
    let mut b = GraphBuilder::new();
    let (src, srcv) = vec_src(&mut b, vec![2024.0_f64, 91.0, 100.0, 20.0]);
    let ann = b.push(Annualize::new(), srcv);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    *g.state_mut(src) = Array::from_vec([4], vec![2024.0, 91.0, 100.0, 20.0]);
    g.stabilize(&mut pool);
    let o = g.view(ann).contiguous_slice().unwrap();
    assert!((o[0] - 100.0 * 365.0 / 91.0).abs() < 1e-10);
    assert!((o[1] - 20.0 * 365.0 / 91.0).abs() < 1e-10);
}

/// `ForwardAdjust`: price-only tick, then a cash dividend (message-passing on
/// the two inputs). 9.5 with a 0.5 cash dividend forward-adjusts back to 10.0.
#[test]
fn stocks_forward_adjust() {
    let mut b = GraphBuilder::new();
    let (price, pricev) = scalar_src(&mut b, 10.0);
    let (divd, divdv) = vec_src(&mut b, vec![0.0_f64, 0.0]);
    let fa = b.push(ForwardAdjust::<0, 1>::new(), (pricev, divdv));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // gen1: price only.
    *g.state_mut(price) = Array::scalar(10.0);
    g.stabilize(&mut pool);
    assert_eq!(g.view(fa).contiguous_slice().unwrap(), &[10.0]);

    // gen2: price 9.5 + cash dividend 0.5 → adjusted back to 10.0.
    *g.state_mut(price) = Array::scalar(9.5);
    *g.state_mut(divd) = Array::from_vec([2], vec![0.0, 0.5]);
    g.stabilize(&mut pool);
    assert!((g.view(fa).contiguous_slice().unwrap()[0] - 10.0).abs() < 1e-12);
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let recs: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Filter::<_, 0>(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 3.0), srcv);
            b.push(Record::<f64, 0>::new(clock.clone()), f)
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
    let (src, srcv) = scalar_src(&mut b, 0.0);
    let cnts: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Filter::<_, 0>(|a: ArrayView<f64, 0>| a.to_contiguous()[0] > 0.0), srcv);
            b.push(Count::<0>, f)
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
        assert_eq!(g.view(c).contiguous_slice().unwrap(), &[passes], "a parallel Count raced");
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
/// [array-view-refactor] Reinterpreted: in the const-rank model a `Split` row is
/// a `RefViewPort` (the by-reference fan-out leaf), which the carry-join combines
/// (`Stack`) consume directly but the `ViewPort`-input operators (`Gate` /
/// `SelectView` / `Count`) cannot. The original downstream `Gate → SelectView →
/// Count` per-row chain is therefore not expressible; the notify-tracking intent
/// (rows recompute with the panel, not on unrelated pokes) is preserved by
/// counting `Stack` recomputes via a clock-stamped `Record` length instead.
#[test]
fn split_rows_notify_with_panel() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec([3, 2], vec![0.0_f64; 6])));
    let panelv = b.push(AsView::<f64, 2>::new(), *panel);
    let other = b.push_source(RefSource::new(Array::scalar(0.0_f64)));
    let otherv = b.push(AsView::<f64, 0>::new(), *other);
    let rows = b.push(Split::<f64, 2, 1>::new(3), panelv);
    assert_eq!(rows.len(), 3);
    // The rows feed a carry `Stack` that rebuilds the `[3, 2]` panel; a `Record`
    // on the stacked output advances exactly once per recompute of the join.
    let stacked = b.push(Stack::<f64, 1, 2>::new(0), &rows[..]);
    let rec = b.push(Record::<f64, 2>::new(clock.clone()), stacked);
    let _sink = b.push(Count::<0>, otherv); // unrelated cone
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // Build values: row views hold the initial panel rows; the record is empty.
    assert_eq!(g.ref_view(rows[0]).to_contiguous().as_ref(), &[0.0, 0.0]);
    assert_eq!(g.ref_view(rows[0]).extents(), [2]);
    assert_eq!(g.ref_view(rec).len(), 0);

    clock.set(ts(1));
    *g.state_mut(panel) = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(rows[0]).to_contiguous().as_ref(), &[1.0, 2.0]);
    assert_eq!(g.ref_view(rows[1]).to_contiguous().as_ref(), &[3.0, 4.0]);
    assert_eq!(g.ref_view(rows[2]).to_contiguous().as_ref(), &[5.0, 6.0]);
    assert_eq!(g.view(stacked).contiguous_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(g.ref_view(rec).len(), 1); // join recomputed once (on the panel poke)

    // Poking an unrelated source must not advance the carry-join record.
    clock.set(ts(2));
    *g.state_mut(other) = Array::scalar(1.0);
    g.stabilize(&mut pool);
    assert_eq!(g.ref_view(rec).len(), 1, "unrelated poke must not recompute the join");
}

/// The declared axis size is validated against the build-time input shape.
#[test]
#[should_panic(expected = "Split: input axis-0 size")]
fn split_axis_size_mismatch_panics() {
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec([3, 2], vec![0.0_f64; 6])));
    let panelv = b.push(AsView::<f64, 2>::new(), *panel);
    let _ = b.push(Split::<f64, 2, 1>::new(2), panelv);
    let _g = Graph::from_builder(b);
}

/// The view chain (retaining `Gate` -> view-input `SliceView` / `ForwardAdjust`)
/// is tick-for-tick bit-identical to the owned chain (`Filter` -> owned `Select`
/// / `ForwardAdjust`) over the same source pokes, including the NaN cutoff and
/// the price/dividend message-passing. `Gate` and `Filter` honour the
/// no-notify⟹unchanged contract by retaining the last passed row, so their
/// downstream cones must agree bit-for-bit.
///
/// [array-view-refactor] Reinterpreted: a `Split` row is a `RefViewPort`, which
/// cannot feed the `ViewPort`-input chain (`Gate`/`Filter`/`Select`/`SliceView`),
/// so the per-stock row is sourced as a direct `ViewPort` (an `AsView` over a
/// per-stock `[2]` `RefSource`) instead of a panel split. The data stream and
/// every asserted value are unchanged — only the source boundary differs.
#[test]
fn view_chain_matches_owned_chain() {
    fn any_finite(a: ArrayView<'_, f64, 1>) -> bool {
        a.to_contiguous().iter().any(|x| x.is_finite())
    }
    fn bits(a: ArrayView<'_, f64, 0>) -> Vec<u64> {
        a.to_contiguous().iter().map(|x| x.to_bits()).collect()
    }

    let nan = f64::NAN;
    let mut b = GraphBuilder::new();
    // Stock 0's price/dividend rows as direct `[2]` view sources.
    let (prices, prices_view) = vec_src(&mut b, vec![nan; 2]);
    let (div, div_view) = vec_src(&mut b, vec![nan; 2]);

    // Owned reference chain (materializes at the row Selects).
    let p_f = {
        let m = b.push(Select::<f64, 1, 1>::flat(vec![0, 1]), prices_view);
        b.push(Filter::<_, 1>(any_finite), m)
    };
    let d_f = {
        let m = b.push(Select::<f64, 1, 1>::flat(vec![0, 1]), div_view);
        b.push(Filter::<_, 1>(any_finite), m)
    };
    // Squeeze the single close out to a scalar (rank-0) price.
    let close = b.push(Select::<f64, 1, 0>::new(vec![0], 0, true), p_f);
    let adj = b.push(ForwardAdjust::<0, 1>::new().with_output_prices(false), (close, d_f));
    let adjusted = b.push(multiply::<f64, 0>(), (close, adj));

    // Zero-copy view chain (materializes at SliceView).
    let p_g = b.push(Gate::<_, 1>(any_finite), prices_view);
    let d_g = b.push(Gate::<_, 1>(any_finite), div_view);
    let v_close = b.push(SliceView::<f64, 1, 0>::new(vec![0], 0, true), p_g);
    let v_adj = b.push(
        ForwardAdjustViewDiv::<0, 1>::default().with_output_prices(false),
        (v_close, d_g),
    );
    let v_adjusted = b.push(multiply::<f64, 0>(), (v_close, v_adj));

    let mut g = Graph::from_builder(b);
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
        assert_eq!(bits(g.view(adjusted)), bits(g.view(v_adjusted)), "tick {i}: adjusted");
        assert_eq!(bits(g.view(adj)), bits(g.view(v_adj)), "tick {i}: adjusts");
    }
}

/// A carry `Stack` over zero-copy `Split` rows honours the no-notify⟹unchanged
/// contract: the join reads **every** input each generation (incl. un-notified
/// ones), and an idle generation (the panel not poked) leaves the stacked
/// cross-section byte-identical to the last poked value. Two equivalent joins
/// (`Stack` and its `StackView` alias) over the same rows agree bit-for-bit.
///
/// [array-view-refactor] Reinterpreted: in the const-rank model the only
/// `RefViewPort` array producer is `Split`, and the carry-join combines
/// (`Stack`) take `RefViewPorts` — but the `ViewPort`-input `Gate`/`Select`/
/// `SliceView` cannot consume a `Split` row, so the original
/// `Split → Gate → {Select, SliceView} → {Stack, StackView}` topology (per-stock
/// gated cutoff feeding the join) is not expressible. The retained core — the
/// carry join re-reads un-notified inputs and freezes its output across idle
/// generations — is tested by stacking the `Split` rows directly and asserting
/// the idle-generation carry. (`StackView` is now a type alias of `Stack`, so
/// the owned-vs-view comparison is the same operator over the same inputs.)
#[test]
fn view_join_carry_matches_owned_join() {
    fn bits(a: ArrayView<'_, f64, 2>) -> Vec<u64> {
        a.to_contiguous().iter().map(|x| x.to_bits()).collect()
    }

    let n = 3usize;
    let mut b = GraphBuilder::new();
    let panel = b.push_source(RefSource::new(Array::from_vec([n, 2], vec![0.0; n * 2])));
    let panelv = b.push(AsView::<f64, 2>::new(), *panel);
    let rows = b.push(Split::<f64, 2, 1>::new(n), panelv);

    // Two equivalent carry joins over the same `Split` rows (the only buildable
    // carry-join input): `Stack` and its `StackView` alias.
    let owned_join = b.push(Stack::<f64, 1, 2>::new(0), &rows[..]);
    let view_join = b.push(StackView::<f64, 1, 2>::new(0), &rows[..]);

    let mut g = Graph::from_builder(b);
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
            *g.state_mut(panel) = Array::from_vec([n, 2], p.to_vec());
            last = *p;
        }
        g.stabilize(&mut pool);
        // The two joins are the same operator over the same rows.
        assert_eq!(bits(g.view(view_join)), bits(g.view(owned_join)), "tick {i}: joins agree");
        // The stacked cross-section is the last poked panel (carried across idle
        // generations) — the no-notify⟹unchanged carry contract.
        assert_eq!(g.view(view_join).contiguous_slice().unwrap(), &last, "tick {i}: carry");
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

    let fused = flowgraph::segment!(|prices_row: Vp<1>, div_row: Vp<1>|
        -> (Vp<0>, Vp<0>) {
        let prices = prices_row => Filter::<_, 1>(any_finite);
        let dividends = div_row => Filter::<_, 1>(any_finite);
        let close = prices => Select::<f64, 1, 0>::new(vec![0], 0, true);
        let adjusts = (close, dividends) => ForwardAdjust::<0, 1>::new().with_output_prices(false);
        let adjusted = (close, adjusts) => multiply::<f64, 0>();
        (adjusted, adjusts)
    });

    let nan = f64::NAN;
    let mut b = GraphBuilder::new();
    let (prices, pricesv) = vec_src(&mut b, vec![nan; 2]);
    let (div, divv) = vec_src(&mut b, vec![nan; 2]);

    // Reference: the same chain as separate nodes.
    let p_f = b.push(Filter::<_, 1>(any_finite), pricesv);
    let d_f = b.push(Filter::<_, 1>(any_finite), divv);
    let close = b.push(Select::<f64, 1, 0>::new(vec![0], 0, true), p_f);
    let adj = b.push(ForwardAdjust::<0, 1>::new().with_output_prices(false), (close, d_f));
    let adjusted = b.push(multiply::<f64, 0>(), (close, adj));

    let (f_adjusted, f_adj) = b.push(fused, (pricesv, divv));
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
        if let Some(p) = p {
            *g.state_mut(prices) = Array::from_vec([2], p.to_vec());
        }
        if let Some(d) = d {
            *g.state_mut(div) = Array::from_vec([2], d.to_vec());
        }
        g.stabilize(&mut pool);
        assert_eq!(bits(g.view(adjusted)), bits(g.view(f_adjusted)), "tick {i}: adjusted");
        assert_eq!(bits(g.view(adj)), bits(g.view(f_adj)), "tick {i}: adjusts");
    }
}
