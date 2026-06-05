//! Differential tests: each graph reproduces a known-good output of the legacy
//! [`scenario`](crate::scenario) engine (the asserted values are lifted from
//! `scenario`'s own test suite), plus a parallel-execution gate that runs the
//! gating / clock / concurrent-write paths under real worker threads.

use flowgraph::core::Pool;
use flowgraph::typed::{Graph, GraphBuilder, Port};

use super::*;
use crate::{Array, Instant, Series};

fn ts(n: i64) -> Instant {
    Instant::from_nanos(n)
}

// ===========================================================================
// Differential vs legacy engine (Pool::new(0): sequential equivalence)
// ===========================================================================

/// `scenario_simple_add`: 10 + 3 == 13.
#[test]
fn simple_add() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let ha = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let hb = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let hc = b.push(Adapt::new(Add::<f64>::new()), (ha, hb));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(ha) = Array::scalar(10.0);
    *g.cell_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(hc).as_slice(), &[13.0]);
}

/// `scenario_chain`: (2 + 3) * 2 == 10. Exercises a 3-deep cone where the leaf
/// is read by two operators (Add and Mul both read `a`).
#[test]
fn chain_add_then_mul() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let bb = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let ab = b.push(Adapt::new(Add::<f64>::new()), (a, bb));
    let out = b.push(Adapt::new(Multiply::<f64>::new()), (ab, a));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::scalar(2.0);
    *g.cell_mut(bb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(out).as_slice(), &[10.0]);
}

/// `scenario_record`: record (10+3), (20+7) → series [13, 27] @ [1, 2].
#[test]
fn record_series() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let ha = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let hb = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let sum = b.push(Adapt::new(Add::<f64>::new()), (ha, hb));
    let rec = b.push(Adapt::new(Record::new(clock.clone())), sum);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(ha) = Array::scalar(10.0);
    *g.cell_mut(hb) = Array::scalar(3.0);
    g.stabilize(&mut pool);

    clock.set(ts(2));
    *g.cell_mut(ha) = Array::scalar(20.0);
    *g.cell_mut(hb) = Array::scalar(7.0);
    g.stabilize(&mut pool);

    let s: &Series<f64> = g.cell(rec);
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
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let flt = b.push(Adapt::new(Filter(|a: &Array<f64>| a[0] > 3.0)), src);
    let rec = b.push(Adapt::new(Record::new(clock.clone())), flt);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 5.0), (3, 2.0), (4, 10.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    let s: &Series<f64> = g.cell(rec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.values(), &[5.0, 10.0]);
    assert_eq!(s.timestamps(), &[ts(2), ts(4)]);
}

/// `Last(Record(x))` recovers the latest array value.
#[test]
fn last_of_record() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let lst = b.push(Adapt::new(Last::new(0.0_f64)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.cell(lst).as_slice(), &[20.0]);
}

/// `scenario_run_periodic_single_input`: data [10,20,30]@[1,2,3], clock @2 only
/// → record (2,20). Exercises the `?Sized` Clocked generic + clock gating.
#[test]
fn clocked_periodic() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let tick = b.push(Adapt::new(Const(())), ());
    let gated = b.push(
        Adapt::new(Clocked::<_, ()>::new(Filter(|_: &Array<f64>| true))),
        (tick, data),
    );
    let rec = b.push(Adapt::new(Record::new(clock.clone())), gated);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    let clock_ticks = [2_i64];
    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        clock.set(ts(t));
        *g.cell_mut(data) = Array::scalar(v);
        if clock_ticks.contains(&t) {
            let _ = g.cell_mut(tick);
        }
        g.stabilize(&mut pool);
    }

    let s: &Series<f64> = g.cell(rec);
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
    let a = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let bb = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let sum = b.push(Adapt::new(Add::<f64>::new()), (a, bb));
    let rec = b.push(Adapt::new(Record::new(clock.clone())), sum);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, va, vb) in [(1_i64, 10.0, 100.0), (2, 20.0, 200.0)] {
        clock.set(ts(t));
        *g.cell_mut(a) = Array::scalar(va);
        *g.cell_mut(bb) = Array::scalar(vb);
        g.stabilize(&mut pool);
    }

    assert_eq!(g.cell(rec).values(), &[110.0, 220.0]);
}

/// Slice input + per-element notify: gen1 all fire → [1,2,3]; gen2 only s1 →
/// Stack keeps stale [1,20,3]; StackSync NaN-fills [NaN,20,NaN].
#[test]
fn slice_stack_and_sync() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let s0 = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let s1 = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let s2 = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let stacked = b.push(Adapt::new(Stack::<f64>::new(0)), &[s0, s1, s2][..]);
    let synced = b.push(Adapt::new(StackSync::<f64>::new(0)), &[s0, s1, s2][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(s0) = Array::scalar(1.0);
    *g.cell_mut(s1) = Array::scalar(2.0);
    *g.cell_mut(s2) = Array::scalar(3.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(stacked).as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(g.cell(synced).as_slice(), &[1.0, 2.0, 3.0]);

    clock.set(ts(2));
    *g.cell_mut(s1) = Array::scalar(20.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(stacked).as_slice(), &[1.0, 20.0, 3.0]);
    let v = g.cell(synced).as_slice();
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
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let rm = b.push(Adapt::new(RollingMean::<f64>::count(3)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.cell(rm).as_slice(), &[2.0]); // mean(1,2,3)

    clock.set(ts(4));
    *g.cell_mut(src) = Array::scalar(6.0);
    g.stabilize(&mut pool);
    assert!((g.cell(rm).as_slice()[0] - 11.0 / 3.0).abs() < 1e-12); // mean(2,3,6)
}

// -- Batch 2: arithmetic / rolling / structural parity ----------------------

/// Unary `Negate` + binary `Subtract`/`Divide` (values from `arithmetic` tests).
#[test]
fn arith_unary_and_binary() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, -2.0, 3.0]))), ());
    let neg = b.push(Adapt::new(Negate::<f64>::new()), a);
    let x = b.push(Adapt::new(Const(Array::scalar(20.0_f64))), ());
    let y = b.push(Adapt::new(Const(Array::scalar(4.0_f64))), ());
    let sub = b.push(Adapt::new(Subtract::<f64>::new()), (x, y));
    let div = b.push(Adapt::new(Divide::<f64>::new()), (x, y));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, -2.0, 3.0]);
    *g.cell_mut(x) = Array::scalar(20.0);
    *g.cell_mut(y) = Array::scalar(4.0);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(neg).as_slice(), &[-1.0, 2.0, -3.0]);
    assert_eq!(g.cell(sub).as_slice(), &[16.0]);
    assert_eq!(g.cell(div).as_slice(), &[5.0]);
}

/// `Min`/`Max`/`Pow` (values from `arithmetic` tests).
#[test]
fn arith_min_max_pow() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 5.0, 3.0]))), ());
    let bb = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![2.0_f64, 4.0, 6.0]))), ());
    let mn = b.push(Adapt::new(Min::<f64>::new()), (a, bb));
    let mx = b.push(Adapt::new(Max::<f64>::new()), (a, bb));
    let p = b.push(Adapt::new(Pow::new(2.0_f64)), a);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 5.0, 3.0]);
    *g.cell_mut(bb) = Array::from_vec(&[3], vec![2.0, 4.0, 6.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(mn).as_slice(), &[1.0, 4.0, 3.0]);
    assert_eq!(g.cell(mx).as_slice(), &[2.0, 5.0, 6.0]);
    assert_eq!(g.cell(p).as_slice(), &[1.0, 25.0, 9.0]);
}

/// `RollingSum`/`RollingVariance` window-3 (values from rolling tests).
#[test]
fn rolling_sum_and_variance() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let rsum = b.push(Adapt::new(RollingSum::<f64>::count(3)), rec);
    let rvar = b.push(Adapt::new(RollingVariance::<f64>::count(3)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.cell(rsum).as_slice(), &[6.0]);
    assert!((g.cell(rvar).as_slice()[0] - 2.0 / 3.0).abs() < 1e-10);

    clock.set(ts(4));
    *g.cell_mut(src) = Array::scalar(4.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(rsum).as_slice(), &[9.0]); // 2+3+4
}

/// `RollingCovariance` on a `[2]` vector with `y = 2x` (values from cov tests).
#[test]
fn rolling_covariance_2d() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::<f64>::zeros(&[2]))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let cov = b.push(Adapt::new(RollingCovariance::<f64>::count(3)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, x) in [(1_i64, 1.0), (2, 2.0), (3, 3.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::from_vec(&[2], vec![x, 2.0 * x]);
        g.stabilize(&mut pool);
    }
    let c = g.cell(cov).as_slice(); // [2,2]
    assert!((c[0] - 2.0 / 3.0).abs() < 1e-10); // Var(x)
    assert!((c[1] - 4.0 / 3.0).abs() < 1e-10); // Cov(x,y)
    assert!((c[3] - 8.0 / 3.0).abs() < 1e-10); // Var(y)
}

/// `Ema` window-2 (value from ema_two_values).
#[test]
fn ema_two_values() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let e = b.push(Adapt::new(Ema::<f64>::new(0.5, 2)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
    assert!((g.cell(e).as_slice()[0] - expected).abs() < 1e-10);
}

/// `Where` / `Cast` / `Id` (values from their legacy tests).
#[test]
fn structural_where_cast_id() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 5.0, 2.0]))), ());
    let w = b.push(Adapt::new(Where::new(|v: f64| v > 3.0, 0.0_f64)), a);
    let i = b.push(Adapt::new(Id::<Array<f64>>::new()), a);
    let ci = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1_i32, 2, 3]))), ());
    let c = b.push(Adapt::new(Cast::<i32, f64>::new()), ci);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 5.0, 2.0]);
    *g.cell_mut(ci) = Array::from_vec(&[3], vec![1_i32, 2, 3]);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(w).as_slice(), &[0.0, 5.0, 0.0]);
    assert_eq!(g.cell(i).as_slice(), &[1.0, 5.0, 2.0]);
    assert_eq!(g.cell(c).as_slice(), &[1.0, 2.0, 3.0]);
}

// -- Batch 3: num tail (element-wise / cross-tick / cross-sectional) --------

/// `Clamp`, `Fillna`, `ForwardFill` (single-shot, values from legacy tests).
#[test]
fn num_clamp_fillna_ffill() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 3.0, 7.0]))), ());
    let clamp = b.push(Adapt::new(Clamp::new(2.0_f64, 5.0)), a);
    let na = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, f64::NAN, 3.0]))), ());
    let fill = b.push(Adapt::new(Fillna::new(0.0_f64)), na);
    let ff = b.push(Adapt::new(ForwardFill::<f64>::new()), na);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 3.0, 7.0]);
    *g.cell_mut(na) = Array::from_vec(&[3], vec![1.0, f64::NAN, 3.0]);
    g.stabilize(&mut pool);

    assert_eq!(g.cell(clamp).as_slice(), &[2.0, 3.0, 5.0]);
    assert_eq!(g.cell(fill).as_slice(), &[1.0, 0.0, 3.0]);
    let v = g.cell(ff).as_slice();
    assert_eq!(v[0], 1.0);
    assert!(v[1].is_nan());
    assert_eq!(v[2], 3.0);
}

/// `Diff` / `PctChange` across ticks (NaN on first, then differences/returns).
#[test]
fn num_diff_and_pct_change() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let d = b.push(Adapt::new(Diff::<f64>::new()), src);
    let pc = b.push(Adapt::new(PctChange::<f64>::new()), src);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(src) = Array::scalar(100.0);
    g.stabilize(&mut pool);
    assert!(g.cell(d).as_slice()[0].is_nan());
    assert!(g.cell(pc).as_slice()[0].is_nan());

    clock.set(ts(2));
    *g.cell_mut(src) = Array::scalar(110.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(d).as_slice()[0], 10.0);
    assert!((g.cell(pc).as_slice()[0] - 0.1).abs() < 1e-12);
}

/// Cross-sectional `Gaussianize` / `Percentile` / `Standardize` / `Winsorize`
/// (values from their legacy tests).
#[test]
fn num_cross_sectional() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let five = b.push(Adapt::new(Const(Array::from_vec(&[5], vec![30.0_f64, 10.0, 50.0, 20.0, 40.0]))), ());
    let gau = b.push(Adapt::new(Gaussianize::<f64>::new()), five);
    let pct = b.push(Adapt::new(Percentile::<f64>::new()), five);
    let std_in = b.push(Adapt::new(Const(Array::from_vec(&[5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]))), ());
    let zsc = b.push(Adapt::new(Standardize::<f64>::new()), std_in);
    let win_in = b.push(Adapt::new(Const(Array::from_vec(&[10], (0..10).map(|i| i as f64).collect()))), ());
    let win = b.push(Adapt::new(Winsorize::<f64>::new(0.1)), win_in);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(five) = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0]);
    *g.cell_mut(std_in) = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    *g.cell_mut(win_in) = Array::from_vec(&[10], (0..10).map(|i| i as f64).collect());
    g.stabilize(&mut pool);

    // Gaussianize: middle (30) → Φ⁻¹(0.5) = 0; symmetric ranks sum to 0.
    let gv = g.cell(gau).as_slice();
    assert!((gv[0] - 0.0).abs() < 1e-9);
    assert!(gv.iter().sum::<f64>().abs() < 1e-9);
    // Percentile: (0.5,1.5,2.5,3.5,4.5)/5 by rank.
    let pv = g.cell(pct).as_slice();
    assert!((pv[1] - 0.1).abs() < 1e-12 && (pv[0] - 0.5).abs() < 1e-12 && (pv[2] - 0.9).abs() < 1e-12);
    // Standardize: zero mean, unit pop-variance.
    let zv = g.cell(zsc).as_slice();
    assert!((zv.iter().sum::<f64>() / 5.0).abs() < 1e-12);
    assert!((zv.iter().map(|&x| x * x).sum::<f64>() / 5.0 - 1.0).abs() < 1e-12);
    // Winsorize p=0.1 over [0..9] → clip to [1, 8].
    assert_eq!(g.cell(win).as_slice(), &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]);
}

// -- Batch 4: transform / reshape (Map, Apply, Select, Lag, Concat) ---------

/// `Map` (allocating S→T) doubling a scalar.
#[test]
fn map_doubles() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let m = b.push(
        Adapt::new(Map::new(|a: &Array<f64>| {
            let mut o = a.clone();
            o[0] *= 2.0;
            o
        })),
        src,
    );
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(src) = Array::scalar(5.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(m).as_slice(), &[10.0]);
}

/// `Apply` (two-input add) and `Select` (flat index pick).
#[test]
fn apply_add_and_select() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]))), ());
    let bb = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0]))), ());
    let ap = b.push(
        Adapt::new(Apply::<(Port<Array<f64>>, Port<Array<f64>>), _, _>::new(
            |(a, b): (&Array<f64>, &Array<f64>)| {
                let mut out = a.clone();
                for (o, &v) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
                    *o += v;
                }
                out
            },
        )),
        (a, bb),
    );
    let five = b.push(Adapt::new(Const(Array::from_vec(&[5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]))), ());
    let sel = b.push(Adapt::new(Select::<f64>::flat(vec![1, 3])), five);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
    *g.cell_mut(bb) = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
    *g.cell_mut(five) = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(ap).as_slice(), &[11.0, 22.0, 33.0]);
    assert_eq!(g.cell(sel).as_slice(), &[20.0, 40.0]);
}

/// `Lag` (offset 2 over a recorded series).
#[test]
fn lag_offset_two() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let rec = b.push(Adapt::new(Record::new(clock.clone())), src);
    let lag = b.push(Adapt::new(Lag::new(2, f64::NAN)), rec);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        clock.set(ts(t));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    assert_eq!(g.cell(lag).as_slice(), &[10.0]); // value from 2 steps ago
}

/// `Concat` axis-0 of two `[2]` arrays → `[4]`.
#[test]
fn concat_axis0() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let a = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![1.0_f64, 2.0]))), ());
    let bb = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![3.0_f64, 4.0]))), ());
    let cc = b.push(Adapt::new(Concat::<f64>::new(0)), &[a, bb][..]);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(a) = Array::from_vec(&[2], vec![1.0, 2.0]);
    *g.cell_mut(bb) = Array::from_vec(&[2], vec![3.0, 4.0]);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(cc).as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

// -- Batch 5/6: metrics (clock-gated) + stocks -----------------------------

/// Clock-gated `CompoundReturn` / `AverageReturn` / `Volatility` over a price
/// path, firing the clock each tick (values from the metrics tests).
#[test]
fn metrics_clock_gated() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let tick = b.push(Adapt::new(Const(())), ());
    let cr = b.push(Adapt::new(CompoundReturn::<f64>::new()), (data, tick));
    let ar = b.push(Adapt::new(AverageReturn::<f64>::new()), (data, tick));
    let vol = b.push(Adapt::new(Volatility::<f64>::new()), (data, tick));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 100.0), (2, 110.0)] {
        clock.set(ts(t));
        *g.cell_mut(data) = Array::scalar(v);
        let _ = g.cell_mut(tick);
        g.stabilize(&mut pool);
    }
    assert!((g.cell(cr).as_slice()[0] - 0.10).abs() < 1e-10);
    assert!((g.cell(ar).as_slice()[0] - 0.10).abs() < 1e-10);
    assert_eq!(g.cell(vol).as_slice()[0], 0.0); // single return → zero std

    clock.set(ts(3));
    *g.cell_mut(data) = Array::scalar(99.0);
    let _ = g.cell_mut(tick);
    g.stabilize(&mut pool);
    assert!(g.cell(ar).as_slice()[0].abs() < 1e-10); // 0.10, -0.10 → 0
    assert!((g.cell(vol).as_slice()[0] - 0.10).abs() < 1e-10); // std 0.10
}

/// `Drawdown` (single input, no clock) from the running maximum.
#[test]
fn metrics_drawdown() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let data = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let dd = b.push(Adapt::new(Drawdown::<f64>::new()), data);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    for (t, v, e) in [(1_i64, 100.0, 0.0), (2, 120.0, 0.0), (3, 90.0, -0.25)] {
        clock.set(ts(t));
        *g.cell_mut(data) = Array::scalar(v);
        g.stabilize(&mut pool);
        assert!((g.cell(dd).as_slice()[0] - e).abs() < 1e-10);
    }
}

/// `Annualize`: YTD [2024, day 91, 100, 20] → annualized × 365/91.
#[test]
fn stocks_annualize() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let src = b.push(
        Adapt::new(Const(Array::from_vec(&[4], vec![2024.0_f64, 91.0, 100.0, 20.0]))),
        (),
    );
    let ann = b.push(Adapt::new(Annualize::new()), src);
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    clock.set(ts(1));
    *g.cell_mut(src) = Array::from_vec(&[4], vec![2024.0, 91.0, 100.0, 20.0]);
    g.stabilize(&mut pool);
    let o = g.cell(ann).as_slice();
    assert!((o[0] - 100.0 * 365.0 / 91.0).abs() < 1e-10);
    assert!((o[1] - 20.0 * 365.0 / 91.0).abs() < 1e-10);
}

/// `ForwardAdjust`: price-only tick, then a cash dividend (message-passing on
/// the two inputs). 9.5 with a 0.5 cash dividend forward-adjusts back to 10.0.
#[test]
fn stocks_forward_adjust() {
    let clock = Clock::new();
    let mut b = GraphBuilder::new();
    let price = b.push(Adapt::new(Const(Array::scalar(10.0_f64))), ());
    let divd = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![0.0_f64, 0.0]))), ());
    let fa = b.push(Adapt::new(ForwardAdjust::new()), (price, divd));
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // gen1: price only.
    clock.set(ts(1));
    *g.cell_mut(price) = Array::scalar(10.0);
    g.stabilize(&mut pool);
    assert_eq!(g.cell(fa).as_slice(), &[10.0]);

    // gen2: price 9.5 + cash dividend 0.5 → adjusted back to 10.0.
    clock.set(ts(2));
    *g.cell_mut(price) = Array::scalar(9.5);
    *g.cell_mut(divd) = Array::from_vec(&[2], vec![0.0, 0.5]);
    g.stabilize(&mut pool);
    assert!((g.cell(fa).as_slice()[0] - 10.0).abs() < 1e-12);
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
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let recs: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Adapt::new(Filter(|a: &Array<f64>| a[0] > 3.0)), src);
            b.push(Adapt::new(Record::new(clock.clone())), f)
        })
        .collect();
    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(8);

    let seq = [1.0_f64, 5.0, 2.0, 10.0, 4.0, 0.5, 7.0, 3.0, 9.0];
    let expected: Vec<f64> = seq.iter().copied().filter(|&v| v > 3.0).collect();
    for (i, &v) in seq.iter().enumerate() {
        clock.set(ts(i as i64 + 1));
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &rec in &recs {
        assert_eq!(g.cell(rec).values(), &expected[..], "a parallel branch diverged");
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
    let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
    let cnts: Vec<_> = (0..K)
        .map(|_| {
            let f = b.push(Adapt::new(Filter(|a: &Array<f64>| a[0] > 0.0)), src);
            b.push(Adapt::new(Count), f)
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
        *g.cell_mut(src) = Array::scalar(v);
        g.stabilize(&mut pool);
    }
    for &c in &cnts {
        assert_eq!(g.cell(c).as_slice(), &[passes], "a parallel Count raced");
    }
}
