//! Axis reductions: `reduce::{ops, cmp, boolean, float}`.
//!
//! Every constructor is a thin wrapper over `array::reduce_along_axis` around
//! one scalar fold, so the tests are grouped one per *family*, each running
//! its members over a single shared `[2, 3]` panel along both axes — which is
//! also what pins the axis argument, since a reduction that folded the wrong
//! one would still publish plausible-looking numbers.
//!
//! The seed each fold starts from is the part worth asserting: it is what an
//! empty axis reduces to, and what tells the two float readings apart — the
//! `Float::max` one that only passes over NaN, and the `stats` one that treats
//! everything non-finite as missing.

use tradingflow::data::{Array, Instant};
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, elem, reduce};

use crate::harness::*;

/// `[[1, 2, 3], [4, 5, 6]]` — the same panel the array tests use, where the
/// two axes reduce to visibly different numbers.
fn panel23() -> Array<f64, 2> {
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into()
}

// ---------------------------------------------------------------------------
// ops: sum, product
// ---------------------------------------------------------------------------

/// `sum`/`product` along each axis of the shared panel, and the seeds they
/// start from: an empty axis is the fold's identity, `0` and `1`.
#[test]
fn ops_arithmetic_family() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let empty = b.val(array::constant(Array::<f64, 2>::zeros([2, 0])));
    let rows = b.op(reduce::sum::<f64, 2, 1>(1), x);
    let cols = b.op(reduce::sum::<f64, 2, 1>(0), x);
    let prod_rows = b.op(reduce::product::<f64, 2, 1>(1), x);
    let prod_cols = b.op(reduce::product::<f64, 2, 1>(0), x);
    let empty_sum = b.op(reduce::sum::<f64, 2, 1>(1), empty);
    let empty_prod = b.op(reduce::product::<f64, 2, 1>(1), empty);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(rows).extents(), [2], "the folded axis is dropped");
    assert_eq!(vals(g.view(rows)), vec![6.0, 15.0]);
    assert_eq!(vals(g.view(cols)), vec![5.0, 7.0, 9.0]);
    assert_eq!(vals(g.view(prod_rows)), vec![6.0, 120.0]);
    assert_eq!(vals(g.view(prod_cols)), vec![4.0, 10.0, 18.0]);

    // An empty axis reduces to the seed, once per remaining index.
    assert_eq!(vals(g.view(empty_sum)), vec![0.0, 0.0]);
    assert_eq!(vals(g.view(empty_prod)), vec![1.0, 1.0]);
}

/// The same operators over `i32` — they are generic over the scalar type, and
/// integer folds neither saturate nor promote.
#[test]
fn ops_integer_arithmetic() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::<i32, 2>::zeros([2, 3])));
    let sums = b.op(reduce::sum::<i32, 2, 1>(1), x);
    let prods = b.op(reduce::product::<i32, 2, 1>(1), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[1, -2, 3], [4, 5, 6]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(sums)), vec![2, 15]);
    assert_eq!(vals(g.view(prods)), vec![-6, 120]);
}

// ---------------------------------------------------------------------------
// cmp: max, min
// ---------------------------------------------------------------------------

/// `max`/`min` along each axis, and the bounds they seed from — which is what
/// an empty axis reduces to. These are the [`Ord`] reductions, as in `elem`,
/// so floats go through `maxf`/`minf` below instead.
#[test]
fn cmp_extremes_family() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::<i32, 2>::zeros([2, 3])));
    let empty = b.val(array::constant(Array::<i32, 2>::zeros([2, 0])));
    let max_rows = b.op(reduce::max::<i32, 2, 1>(1), x);
    let min_rows = b.op(reduce::min::<i32, 2, 1>(1), x);
    let max_cols = b.op(reduce::max::<i32, 2, 1>(0), x);
    let empty_max = b.op(reduce::max::<i32, 2, 1>(1), empty);
    let empty_min = b.op(reduce::min::<i32, 2, 1>(1), empty);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[1, -2, 3], [4, 5, -6]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(max_rows)), vec![3, 5]);
    assert_eq!(vals(g.view(min_rows)), vec![-2, -6]);
    assert_eq!(vals(g.view(max_cols)), vec![4, 5, 3]);

    // Nothing to compare against leaves the seed: the type's own bounds.
    assert_eq!(vals(g.view(empty_max)), vec![i32::MIN, i32::MIN]);
    assert_eq!(vals(g.view(empty_min)), vec![i32::MAX, i32::MAX]);
}

// ---------------------------------------------------------------------------
// boolean: all, any, count
// ---------------------------------------------------------------------------

/// `all`/`any`/`count` over a mask, including the empty-axis identities that
/// make them vacuously true and vacuously false.
#[test]
fn boolean_family() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::<bool, 2>::zeros([2, 3])));
    let empty = b.val(array::constant(Array::<bool, 2>::zeros([2, 0])));
    let all_rows = b.op(reduce::all::<2, 1>(1), x);
    let any_rows = b.op(reduce::any::<2, 1>(1), x);
    let count_rows = b.op(reduce::count::<f64, 2, 1>(1), x);
    let count_cols = b.op(reduce::count::<f64, 2, 1>(0), x);
    let empty_all = b.op(reduce::all::<2, 1>(1), empty);
    let empty_any = b.op(reduce::any::<2, 1>(1), empty);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[true, true, true], [true, false, false]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(all_rows)), vec![true, false]);
    assert_eq!(vals(g.view(any_rows)), vec![true, true]);
    // Counting into `f64` keeps a share a plain division downstream.
    assert_eq!(vals(g.view(count_rows)), vec![3.0, 1.0]);
    assert_eq!(vals(g.view(count_cols)), vec![2.0, 1.0, 1.0]);

    assert_eq!(vals(g.view(empty_all)), vec![true, true], "vacuously true");
    assert_eq!(vals(g.view(empty_any)), vec![false, false]);
}

// ---------------------------------------------------------------------------
// float: maxf, minf, sum_finite, count_finite
// ---------------------------------------------------------------------------

/// `maxf`/`minf` are the floating-point extremes, delegating to
/// [`Float::max`]/[`Float::min`] as `elem` does: NaN is passed over, and
/// infinities compare like any other value. The `Ord` `max`/`min` above do not
/// accept floats at all.
#[test]
fn float_extremes_family() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let hi = b.op(reduce::maxf::<f64, 2, 1>(1), x);
    let lo = b.op(reduce::minf::<f64, 2, 1>(1), x);
    let hi_cols = b.op(reduce::maxf::<f64, 2, 1>(0), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Row 0 carries a NaN and an infinity; row 1 is all finite.
    *g.state_mut(src) = [[1.0, f64::NAN, f64::INFINITY], [4.0, -5.0, 6.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(&vals(g.view(hi)), &[f64::INFINITY, 6.0], "maxf");
    assert_eq!(vals(g.view(lo)), vec![1.0, -5.0]);
    assert_close(
        &vals(g.view(hi_cols)),
        &[4.0, -5.0, f64::INFINITY],
        "the NaN column takes its only finite value",
    );
}

/// `sum_finite`/`count_finite` are the other reading, the one `stats` uses:
/// non-finite entries are missing rather than arithmetic, so they neither
/// poison the sum nor count toward the sample.
#[test]
fn float_finite_family_treats_non_finite_as_missing() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let plain = b.op(reduce::sum::<f64, 2, 1>(1), x);
    let finite = b.op(reduce::sum_finite::<f64, 2, 1>(1), x);
    let counted = b.op(reduce::count_finite::<f64, 2, 1>(1), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[1.0, f64::NAN, f64::INFINITY], [4.0, -5.0, 6.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    // The plain fold propagates whatever IEEE gives it...
    assert_close(&vals(g.view(plain)), &[f64::NAN, 5.0], "sum propagates NaN");
    // ...where the finite fold drops the two missing entries.
    assert_eq!(vals(g.view(finite)), vec![1.0, 5.0]);
    assert_eq!(vals(g.view(counted)), vec![1.0, 3.0]);
}

/// With nothing on the axis to work with, each family reports it its own way:
/// the finite pair gives `0 / 0`, so a mean composed from them is NaN rather
/// than zero, and `maxf` stays at its NaN seed rather than naming a bound.
#[test]
fn float_families_report_an_all_missing_axis() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let mean = {
        let total = b.op(reduce::sum_finite::<f64, 2, 1>(1), x);
        let n = b.op(reduce::count_finite::<f64, 2, 1>(1), x);
        b.op(elem::div(), (total, n))
    };
    let hi = b.op(reduce::maxf::<f64, 2, 1>(1), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[f64::NAN; 3], [1.0, 2.0, 6.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(
        &vals(g.view(mean)),
        &[f64::NAN, 3.0],
        "sum_finite / count_finite",
    );
    assert_close(&vals(g.view(hi)), &[f64::NAN, 6.0], "maxf");
}

// ---------------------------------------------------------------------------
// Shared core
// ---------------------------------------------------------------------------

/// Every member routes through `array::reduce_along_axis`, so each inherits
/// its axis dispatch: fed a strided view, the reduction still folds the axis
/// it was told to and publishes a contiguous result.
#[test]
fn reductions_fold_the_named_axis_of_a_strided_view() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::permute_axes([1, 0]), x);
    let direct = b.op(reduce::sum::<f64, 2, 1>(1), x);
    let strided = b.op(reduce::sum::<f64, 2, 1>(0), flipped);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    // Axis 0 of the transpose is axis 1 of the panel: the same row sums.
    assert_eq!(vals(g.view(strided)), vals(g.view(direct)));
    assert_eq!(vals(g.view(strided)), vec![6.0, 15.0]);
}

/// Reductions are stateless across generations: the accumulators are reseeded
/// each time, so a running total never leaks into the next tick.
#[test]
fn reductions_reseed_each_generation() {
    let mut b = Builder::new();
    let (src, x) = b.source(array::constant(Array::zeros([2, 3])));
    let sums = b.op(reduce::sum::<f64, 2, 1>(1), x);
    let hi = b.op(reduce::maxf::<f64, 2, 1>(1), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(sums)), vec![6.0, 15.0]);
    assert_eq!(vals(g.view(hi)), vec![3.0, 6.0]);

    *g.state_mut(src) = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(sums)), vec![3.0, 6.0], "not 9 and 21");
    assert_eq!(vals(g.view(hi)), vec![1.0, 2.0], "not 3 and 6");
}
