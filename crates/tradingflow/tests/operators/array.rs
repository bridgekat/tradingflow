//! Integration tests for `operators::array`: the constant cells, the
//! element-wise and whole-array maps, axis selection, the variadic join/split
//! pairs, and the zero-copy view derivations.
//!
//! Every array-shaped edge carries an `ArrayView<'a, T, N>` by value, so these
//! tests pin both the *values* an operator publishes and its *shape* — rank,
//! extents, and whether the published view is contiguous or strided. The last
//! one matters: `to_contiguous` hides the difference from the reader, but the
//! operators below have separate fast and slow paths for it, so several tests
//! deliberately feed a strided view (a `transpose`, a stepped `slice`, an
//! `unstack` column) into the operator under test.

use tradingflow::data::{Array, ArrayView, Instant};
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::array;

use crate::harness::*;

/// A `[2, 3]` panel of consecutive integers — the smallest shape where "which
/// axis?" is a real question and where a transpose is observably strided.
fn panel23() -> Array<f64, 2> {
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into()
}

/// Whether a published view is row-major contiguous (`as_slice` succeeds) or a
/// strided window that downstream operators must walk element by element.
fn is_contiguous<const N: usize>(v: ArrayView<'_, f64, N>) -> bool {
    v.as_slice().is_some()
}

// ---------------------------------------------------------------------------
// constant
// ---------------------------------------------------------------------------

/// `scalar` is the rank-0 cell: no axes, exactly one element. Wired as
/// a source it is the pokeable entry point every other test builds on, so the
/// poke-then-stabilize round trip is pinned here once.
#[test]
fn scalar_is_a_pokeable_rank_zero_cell() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(1.5_f64));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert_eq!(g.view(srcv).ndim(), 0);
    assert_eq!(vals(g.view(srcv)), vec![1.5]);

    *g.state_mut(src) = (2.5).into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(srcv)), vec![2.5]);

    *g.state_mut(src) = (-0.25).into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(srcv)), vec![-0.25]);
}

/// `full` and `zeros` publish their requested extents filled with a single
/// value; the buffer is sized by the extent product, not by any data argument.
#[test]
fn full_and_zeros_fill_the_requested_extents() {
    let mut b = Builder::new();
    let sevens = b.val(array::constant(Array::full([2, 3], 7.0_f64)));
    let empty = b.val(array::constant(Array::<f64, 2>::zeros([2, 2])));
    let g = b.build();

    assert_eq!(g.view(sevens).extents(), [2, 3]);
    assert_eq!(vals(g.view(sevens)), vec![7.0; 6]);
    assert_eq!(g.view(empty).extents(), [2, 2]);
    assert_eq!(vals(g.view(empty)), vec![0.0; 4]);
}

/// `from_parts` and `constant` interpret their buffer row-major: the last axis
/// varies fastest, so `[1, 0]` of a `[2, 3]` panel is the fourth element.
#[test]
fn from_parts_and_constant_lay_data_out_row_major() {
    let mut b = Builder::new();
    let parts = b.val(array::constant(Array::from_parts(
        [2, 3],
        [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0].into(),
    )));
    let owned = b.val(array::constant(panel23()));
    let g = b.build();

    assert_eq!(g.view(parts).extents(), [2, 3]);
    assert_eq!(g.view(parts)[[0, 2]], 3.0);
    assert_eq!(g.view(parts)[[1, 0]], 4.0);
    assert_eq!(vals(g.view(parts)), vals(g.view(owned)));
}

/// A constant registered with `value` is dirty only at build, so later
/// generations never re-enter its cone; the same constant registered with
/// `source` recomputes its cone exactly on the generations it is poked.
#[test]
fn only_poked_constants_recompute_their_cone() {
    let mut b = Builder::new();
    let fixed = b.val(array::constant(1.0_f64));
    let (src, srcv) = b.source(array::constant(1.0_f64));
    let fixed_runs = b.op(runs::<0>(), fixed);
    let src_runs = b.op(runs::<0>(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert_eq!((g.view(fixed_runs), g.view(src_runs)), (0, 0));

    g.stabilize(&mut pool, &nano(1)); // nothing poked at all
    assert_eq!((g.view(fixed_runs), g.view(src_runs)), (0, 0));

    *g.state_mut(src) = (2.0).into();
    g.stabilize(&mut pool, &nano(2));
    assert_eq!((g.view(fixed_runs), g.view(src_runs)), (0, 1));

    *g.state_mut(src) = (3.0).into();
    g.stabilize(&mut pool, &nano(3));
    assert_eq!((g.view(fixed_runs), g.view(src_runs)), (0, 2));
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

/// `map` applies `&A -> T` position by position, keeping the extents and
/// letting the scalar type change (here `f64 -> bool`). The second poke pins
/// the recompute path, which writes into the retained buffer rather than the
/// one `init` allocated.
#[test]
fn map_is_elementwise_and_may_change_the_scalar_type() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 4]));
    let doubled = b.op(array::map(|&x: &f64| x * 2.0), srcv);
    let positive = b.op(array::map(|&x: &f64| x > 0.0), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, -2.0, 3.0, -4.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(doubled).extents(), [4]);
    assert_eq!(vals(g.view(doubled)), vec![2.0, -4.0, 6.0, -8.0]);
    assert_eq!(vals(g.view(positive)), vec![true, false, true, false]);

    *g.state_mut(src) = [-1.0, 2.0, -3.0, 4.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(doubled)), vec![-2.0, 4.0, -6.0, 8.0]);
    assert_eq!(vals(g.view(positive)), vec![false, true, false, true]);
}

/// Fed a strided view, `map` still visits elements in row-major *logical*
/// order (not buffer order) and materializes a contiguous result.
#[test]
fn map_walks_a_strided_input_in_logical_order() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::transpose([1, 0]), srcv);
    let scaled = b.op(array::map(|&x: &f64| x * 10.0), flipped);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert!(!is_contiguous(g.view(flipped)), "transpose stays a view");
    assert_eq!(vals(g.view(flipped)), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(is_contiguous(g.view(scaled)), "map owns its output");
    assert_eq!(
        vals(g.view(scaled)),
        vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]
    );
}

/// `binary_map` stretches any extent-1 axis to its partner's extent, so a
/// `[2, 1]` column broadcasts across a `[2, 3]` panel. Poking only the column
/// recomputes the join against the carried panel.
#[test]
fn binary_map_broadcasts_extent_one_axes() {
    let mut b = Builder::new();
    let (panel, panelv) = b.source(array::constant(Array::zeros([2, 3])));
    let (col, colv) = b.source(array::constant(Array::zeros([2, 1])));
    let sum = b.op(
        array::binary_map(|&x: &f64, &y: &f64| x + y),
        (panelv, colv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(panel) = panel23();
    *g.state_mut(col) = [[10.0], [20.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(sum).extents(), [2, 3]);
    assert_eq!(vals(g.view(sum)), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

    *g.state_mut(col) = [[100.0], [200.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(
        vals(g.view(sum)),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );
}

/// `ternary_map` combines three aligned inputs position by position and, like
/// the binary form, recomputes whenever *any* of them notifies.
#[test]
fn ternary_map_combines_three_aligned_inputs() {
    let mut b = Builder::new();
    let (x, xv) = b.source(array::constant(Array::zeros([3])));
    let (y, yv) = b.source(array::constant(Array::zeros([3])));
    let (z, zv) = b.source(array::constant(Array::zeros([3])));
    let fma = b.op(
        array::ternary_map(|&a: &f64, &b: &f64, &c: &f64| a * b + c),
        (xv, yv, zv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(x) = [1.0, 2.0, 3.0].into();
    *g.state_mut(y) = [10.0, 10.0, 10.0].into();
    *g.state_mut(z) = [0.5, 0.5, 0.5].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(fma)), vec![10.5, 20.5, 30.5]);

    *g.state_mut(z) = [-0.5, -0.5, -0.5].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(fma)), vec![9.5, 19.5, 29.5]);
}

/// `array_map` hands the closure the whole view, so unlike `map` it may change
/// the rank: a `[3]` cross-section reduces to a rank-0 mean.
#[test]
fn array_map_reduces_the_whole_array_to_a_lower_rank() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 3]));
    let mean = b.op(
        array::array_map(|a: ArrayView<'_, f64, 1>| {
            let n = a.extents()[0] as f64;
            (a.iter().sum::<f64>() / n).into()
        }),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, 2.0, 4.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(mean).ndim(), 0);
    assert_close(&vals(g.view(mean)), &[7.0 / 3.0], "mean of [1, 2, 4]");

    *g.state_mut(src) = [2.0, 4.0, 6.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(mean)), vec![4.0]);
}

/// `array_map` replaces its output array wholesale, so its extents may differ
/// from generation to generation — the elementwise `map`, which writes into a
/// buffer sized once at `init`, cannot do this.
#[test]
fn array_map_may_resize_its_output_between_generations() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 3]));
    let finite = b.op(
        array::array_map(|a: ArrayView<'_, f64, 1>| {
            a.iter()
                .copied()
                .filter(|x| x.is_finite())
                .collect::<Vec<_>>()
                .into()
        }),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, f64::NAN, 3.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(finite).extents(), [2]);
    assert_eq!(vals(g.view(finite)), vec![1.0, 3.0]);

    *g.state_mut(src) = [f64::NAN, f64::NAN, 5.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(finite).extents(), [1]);
    assert_eq!(vals(g.view(finite)), vec![5.0]);
}

/// `array_binary_map` may raise the rank as well as lower it: two `[n]` and
/// `[m]` vectors become an `[n, m]` outer product.
#[test]
fn array_binary_map_may_raise_the_output_rank() {
    let mut b = Builder::new();
    let (x, xv) = b.source(array::constant([0.0_f64; 2]));
    let (y, yv) = b.source(array::constant([0.0_f64; 3]));
    let outer = b.op(
        array::array_binary_map(|a: ArrayView<'_, f64, 1>, c: ArrayView<'_, f64, 1>| {
            let (n, m) = (a.extents()[0], c.extents()[0]);
            let mut out = Array::zeros([n, m]);
            for i in 0..n {
                for j in 0..m {
                    out[[i, j]] = a[[i]] * c[[j]];
                }
            }
            out
        }),
        (xv, yv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(x) = [1.0, 2.0].into();
    *g.state_mut(y) = [10.0, 20.0, 30.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(outer).extents(), [2, 3]);
    assert_eq!(
        vals(g.view(outer)),
        vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    );
}

/// `array_ternary_map` fires on any input's notification and reads the other
/// two from their carried views — poking only the middle source updates only
/// the middle slot of the assembled cross-section.
#[test]
fn array_ternary_map_assembles_carried_inputs() {
    let mut b = Builder::new();
    let (s0, s0v) = b.source(array::constant(0.0_f64));
    let (s1, s1v) = b.source(array::constant(0.0_f64));
    let (s2, s2v) = b.source(array::constant(0.0_f64));
    let row = b.op(
        array::array_ternary_map(
            |a: ArrayView<'_, f64, 0>, b: ArrayView<'_, f64, 0>, c: ArrayView<'_, f64, 0>| {
                [*a, *b, *c].into()
            },
        ),
        (s0v, s1v, s2v),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(s0) = (1.0).into();
    *g.state_mut(s1) = (2.0).into();
    *g.state_mut(s2) = (3.0).into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(row).extents(), [3]);
    assert_eq!(vals(g.view(row)), vec![1.0, 2.0, 3.0]);

    *g.state_mut(s1) = (20.0).into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(row)), vec![1.0, 20.0, 3.0]);
}

/// The `_inplace` forms split `init` (run once) from `update` (run per
/// generation) and keep the output buffer between them, so `update` can read
/// its own previous result — the difference that makes them accumulators
/// rather than pure functions.
#[test]
fn array_map_inplace_accumulates_into_a_retained_buffer() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 2]));
    let running = b.op(
        array::array_map_inplace(
            |a: ArrayView<'_, f64, 1>| Array::zeros(a.extents()),
            |out: &mut Array<f64, 1>, a: ArrayView<'_, f64, 1>| {
                for (o, x) in out.data_mut().iter_mut().zip(a.iter()) {
                    *o += *x;
                }
            },
        ),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert_eq!(vals(g.view(running)), vec![0.0, 0.0], "init only zeroes");

    for (poke, expected) in [
        (vec![1.0, 2.0], vec![1.0, 2.0]),
        (vec![10.0, 20.0], vec![11.0, 22.0]),
        (vec![100.0, 200.0], vec![111.0, 222.0]),
    ] {
        *g.state_mut(src) = poke.into();
        g.stabilize(&mut pool, &Instant::MIN);
        assert_eq!(vals(g.view(running)), expected);
    }
}

/// `array_binary_map_inplace` retains its buffer across both inputs: the
/// accumulated product advances once per generation, whichever input notified.
#[test]
fn array_binary_map_inplace_accumulates_across_both_inputs() {
    let mut b = Builder::new();
    let (x, xv) = b.source(array::constant([0.0_f64; 2]));
    let (y, yv) = b.source(array::constant([1.0_f64; 2]));
    let dot = b.op(
        array::array_binary_map_inplace(
            |a: ArrayView<'_, f64, 1>, _: ArrayView<'_, f64, 1>| Array::zeros(a.extents()),
            |out: &mut Array<f64, 1>, a: ArrayView<'_, f64, 1>, c: ArrayView<'_, f64, 1>| {
                for ((o, p), q) in out.data_mut().iter_mut().zip(a.iter()).zip(c.iter()) {
                    *o += *p * *q;
                }
            },
        ),
        (xv, yv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(x) = [1.0, 2.0].into();
    *g.state_mut(y) = [3.0, 4.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(dot)), vec![3.0, 8.0]);

    // Only `x` notifies; `y` is read from its carried view and the running
    // product keeps accumulating.
    *g.state_mut(x) = [10.0, 10.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(dot)), vec![33.0, 48.0]);
}

/// `array_ternary_map_inplace` behaves the same way at arity three: `init`
/// sizes the buffer once, `update` folds each generation into it.
#[test]
fn array_ternary_map_inplace_folds_three_inputs_into_one_buffer() {
    let mut b = Builder::new();
    let (x, xv) = b.source(array::constant([0.0_f64; 2]));
    let (_, yv) = b.source(array::constant([0.0_f64; 2]));
    let (_, zv) = b.source(array::constant([0.0_f64; 2]));
    let total = b.op(
        array::array_ternary_map_inplace(
            |a: ArrayView<'_, f64, 1>, _: ArrayView<'_, f64, 1>, _: ArrayView<'_, f64, 1>| {
                Array::zeros(a.extents())
            },
            |out: &mut Array<f64, 1>,
             a: ArrayView<'_, f64, 1>,
             c: ArrayView<'_, f64, 1>,
             d: ArrayView<'_, f64, 1>| {
                for (i, o) in out.data_mut().iter_mut().enumerate() {
                    *o += a[[i]] + c[[i]] + d[[i]];
                }
            },
        ),
        (xv, yv, zv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(x) = [1.0, 2.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(total)), vec![1.0, 2.0]);

    *g.state_mut(x) = [4.0, 8.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(total)), vec![5.0, 10.0]);
}

// ---------------------------------------------------------------------------
// reduce
// ---------------------------------------------------------------------------

/// `inner_reduce` splits the input at rank `M` and folds each sub-region into
/// its own accumulator, so a `[2, 3]` panel reduced over rank-1 rows publishes
/// a `[2]` array of row sums. Transposing first walks the columns instead —
/// strided sub-regions, which is the slow path.
#[test]
fn inner_reduce_collapses_the_trailing_axes() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::transpose([1, 0]), srcv);
    let rows = b.op(
        array::inner_reduce::<f64, 2, f64, 1, 1>(0.0, |acc, &x| *acc += x),
        srcv,
    );
    let cols = b.op(
        array::inner_reduce::<f64, 2, f64, 1, 1>(0.0, |acc, &x| *acc += x),
        flipped,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(rows).extents(), [2]);
    assert_eq!(vals(g.view(rows)), vec![6.0, 15.0]);
    assert_eq!(g.view(cols).extents(), [3]);
    assert_eq!(vals(g.view(cols)), vec![5.0, 7.0, 9.0]);
    assert!(is_contiguous(g.view(rows)), "reduce owns its output");
}

/// `outer_reduce` is the same fold with the accumulators held still and the
/// input streamed past: reducing the leading axis of the same `[2, 3]` panel
/// publishes the `[3]` column sums, which `inner_reduce` reaches only through
/// a transpose.
#[test]
fn outer_reduce_folds_the_leading_axes() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::transpose([1, 0]), srcv);
    let cols = b.op(
        array::outer_reduce::<f64, 2, f64, 1, 1>(0.0, |acc, &x| *acc += x),
        srcv,
    );
    let rows = b.op(
        array::inner_reduce::<f64, 2, f64, 1, 1>(0.0, |acc, &x| *acc += x),
        flipped,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(cols).extents(), [3]);
    assert_eq!(vals(g.view(cols)), vec![5.0, 7.0, 9.0]);
    assert_eq!(vals(g.view(cols)), vals(g.view(rows)), "same fold");
}

/// Reducing every axis leaves no accumulator axes, so the output is the rank-0
/// cell holding one scalar — either direction gets there.
#[test]
fn reduce_to_rank_zero_yields_one_scalar() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let inner = b.op(
        array::inner_reduce::<f64, 2, f64, 0, 2>(0.0, |acc, &x| *acc += x),
        srcv,
    );
    let outer = b.op(
        array::outer_reduce::<f64, 2, f64, 2, 0>(0.0, |acc, &x| *acc += x),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(inner).ndim(), 0);
    assert_eq!(vals(g.view(inner)), vec![21.0]);
    assert_eq!(vals(g.view(outer)), vec![21.0]);
}

/// The accumulators are reseeded every generation, so a fold that would keep
/// growing still publishes this generation's reduction alone — and the scalar
/// type may differ from the input's.
#[test]
fn reduce_reseeds_its_accumulators_each_generation() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let any_negative = b.op(
        array::inner_reduce::<f64, 2, bool, 1, 1>(false, |acc, &x| *acc |= x < 0.0),
        srcv,
    );
    let sums = b.op(
        array::outer_reduce::<f64, 2, f64, 1, 1>(0.0, |acc, &x| *acc += x),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(any_negative)), vec![false, false]);
    assert_eq!(vals(g.view(sums)), vec![5.0, 7.0, 9.0]);

    *g.state_mut(src) = [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(any_negative)), vec![true, false]);
    // Not [10.0, 10.0, 18.0]: the previous generation is not carried over.
    assert_eq!(vals(g.view(sums)), vec![5.0, 3.0, 9.0]);
}

/// The rank relation is checked when the operator is built, not on the first
/// generation: the reduced axes plus the kept ones must be the whole input.
#[test]
#[should_panic(expected = "must be input ndim")]
fn reduce_rank_mismatch_panics_at_build() {
    let _ = array::inner_reduce::<f64, 2, f64, 2, 1>(0.0, |acc, &x| *acc += x);
}

// ---------------------------------------------------------------------------
// select
// ---------------------------------------------------------------------------

/// `select` gathers the given indices along one axis *in the order given*,
/// repetition included — it is a gather, not a filter, so the output extent is
/// `indices.len()` regardless of the input extent.
#[test]
fn select_gathers_indices_in_order_and_may_repeat_them() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 5]));
    let picked = b.op(array::select(vec![3, 0, 3], 0), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [10.0, 20.0, 30.0, 40.0, 50.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(picked).extents(), [3]);
    assert_eq!(vals(g.view(picked)), vec![40.0, 10.0, 40.0]);

    *g.state_mut(src) = [1.0, 2.0, 3.0, 4.0, 5.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(picked)), vec![4.0, 1.0, 4.0]);
}

/// The `axis` argument picks which axis is gathered; on a `[2, 3]` panel,
/// axis 0 reorders rows and axis 1 reorders columns, and both keep the rank.
#[test]
fn select_gathers_along_the_named_axis_only() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let rows = b.op(array::select(vec![1, 0], 0), srcv);
    let cols = b.op(array::select(vec![2, 0], 1), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(rows).extents(), [2, 3]);
    assert_eq!(vals(g.view(rows)), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    assert_eq!(g.view(cols).extents(), [2, 2]);
    assert_eq!(vals(g.view(cols)), vec![3.0, 1.0, 6.0, 4.0]);
}

/// `select_mask` keeps the `true` positions in place; the output extent along
/// the axis is the number of `true` flags, so it is a filter where `select` is
/// a gather.
#[test]
fn select_mask_keeps_the_true_positions() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([4])));
    let (panel, panelv) = b.source(array::constant(Array::zeros([2, 3])));
    let kept = b.op(array::select_mask(vec![true, false, true, true], 0), srcv);
    let outer = b.op(array::select_mask(vec![true, false, true], 1), panelv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [10.0, 20.0, 30.0, 40.0].into();
    *g.state_mut(panel) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(kept).extents(), [3]);
    assert_eq!(vals(g.view(kept)), vec![10.0, 30.0, 40.0]);
    assert_eq!(g.view(outer).extents(), [2, 2]);
    assert_eq!(vals(g.view(outer)), vec![1.0, 3.0, 4.0, 6.0]);
}

/// `select_at` squeezes the indexed axis, dropping the rank by one. Which axis
/// is indexed also decides the *layout* of the result: indexing the outer axis
/// of a row-major panel yields a contiguous row, indexing the inner axis a
/// strided column.
#[test]
fn select_at_squeezes_the_indexed_axis() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let row = b.op(array::select_at::<f64, 2, 1>(1, 0), srcv);
    let col = b.op(array::select_at::<f64, 2, 1>(2, 1), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(row).extents(), [3]);
    assert_eq!(vals(g.view(row)), vec![4.0, 5.0, 6.0]);
    assert!(is_contiguous(g.view(row)), "an outer-axis row is a slice");

    assert_eq!(g.view(col).extents(), [2]);
    assert_eq!(vals(g.view(col)), vec![3.0, 6.0]);
    assert!(
        !is_contiguous(g.view(col)),
        "an inner-axis column is strided"
    );
}

/// At rank 1 the squeeze bottoms out at a rank-0 view — the idiom for reading
/// one instrument out of a cross-section — and, being a derived view rather
/// than a copy, it follows later pokes of the source.
#[test]
fn select_at_on_a_vector_yields_a_rank_zero_view() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 3]));
    let one = b.op(array::select_at::<f64, 1, 0>(1, 0), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [7.0, 8.0, 9.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(one).ndim(), 0);
    assert_eq!(vals(g.view(one)), vec![8.0]);

    *g.state_mut(src) = [70.0, 80.0, 90.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(one)), vec![80.0]);
}

/// `select` materializing from a *strided* input takes the element-by-element
/// copy path; the gathered rows must still come out in logical order.
#[test]
fn select_gathers_from_a_strided_input() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::transpose([1, 0]), srcv); // [3, 2], strided
    let picked = b.op(array::select(vec![0, 2], 0), flipped);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert!(!is_contiguous(g.view(flipped)));
    assert_eq!(g.view(picked).extents(), [2, 2]);
    assert_eq!(vals(g.view(picked)), vec![1.0, 4.0, 3.0, 6.0]);
}

// ---------------------------------------------------------------------------
// concat
// ---------------------------------------------------------------------------

/// `concat` joins along an *existing* axis, keeping the rank. Which axis is
/// named changes the interleaving, not just the extents: axis 0 appends whole
/// panels, axis 1 appends columns row by row.
#[test]
fn concat_interleaves_differently_per_axis() {
    let mut b = Builder::new();
    let (p, pv) = b.source(array::constant(Array::zeros([2, 3])));
    let (q, qv) = b.source(array::constant(Array::zeros([2, 3])));
    let down = b.op(array::concat(0), &[pv, qv][..]);
    let across = b.op(array::concat(1), &[pv, qv][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(p) = panel23();
    *g.state_mut(q) = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(down).extents(), [4, 3]);
    assert_eq!(
        vals(g.view(down)),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0
        ]
    );
    assert_eq!(g.view(across).extents(), [2, 6]);
    assert_eq!(
        vals(g.view(across)),
        vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0
        ]
    );
}

/// The join axis is the only one whose extents may differ, and the operator is
/// variadic: `[2]`, `[3]` and `[1]` vectors become one `[6]`. Poking one input
/// recomputes the join against the other two's carried views.
#[test]
fn concat_joins_a_variadic_group_of_uneven_inputs() {
    let mut b = Builder::new();
    let (a, av) = b.source(array::constant([0.0_f64; 2]));
    let (c, cv) = b.source(array::constant([0.0_f64; 3]));
    let (d, dv) = b.source(array::constant([0.0_f64; 1]));
    let joined = b.op(array::concat(0), &[av, cv, dv][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(a) = [1.0, 2.0].into();
    *g.state_mut(c) = [3.0, 4.0, 5.0].into();
    *g.state_mut(d) = [6.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(g.view(joined).extents(), [6]);
    assert_eq!(vals(g.view(joined)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    *g.state_mut(c) = [30.0, 40.0, 50.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(joined)), vec![1.0, 2.0, 30.0, 40.0, 50.0, 6.0]);
}

/// `stack` adds a *new* axis at the requested position, taking rank-`N` inputs
/// to a rank-`N + 1` output; the position decides whether the inputs land as
/// rows or as columns.
#[test]
fn stack_inserts_a_new_axis_at_the_requested_position() {
    let mut b = Builder::new();
    let (u, uv) = b.source(array::constant([0.0_f64; 3]));
    let (v, vv) = b.source(array::constant([0.0_f64; 3]));
    let as_rows = b.op(array::stack::<f64, 1, 2>(0), &[uv, vv][..]);
    let as_cols = b.op(array::stack::<f64, 1, 2>(1), &[uv, vv][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(u) = [1.0, 2.0, 3.0].into();
    *g.state_mut(v) = [4.0, 5.0, 6.0].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(as_rows).extents(), [2, 3]);
    assert_eq!(vals(g.view(as_rows)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(g.view(as_cols).extents(), [3, 2]);
    assert_eq!(vals(g.view(as_cols)), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// Stacking rank-0 sources is how per-instrument scalars become a
/// cross-section. Only the sources poked in a generation change; the rest are
/// read from their carried views, so the stack is never partially stale.
#[test]
fn stack_of_rank_zero_sources_builds_a_cross_section() {
    let mut b = Builder::new();
    let (s0, s0v) = b.source(array::constant(0.0_f64));
    let (s1, s1v) = b.source(array::constant(0.0_f64));
    let (s2, s2v) = b.source(array::constant(0.0_f64));
    let row = b.op(array::stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(s0) = (1.0).into();
    *g.state_mut(s1) = (2.0).into();
    *g.state_mut(s2) = (3.0).into();
    g.stabilize(&mut pool, &nano(1));
    assert_eq!(g.view(row).extents(), [3]);
    assert_eq!(vals(g.view(row)), vec![1.0, 2.0, 3.0]);

    *g.state_mut(s1) = (20.0).into();
    g.stabilize(&mut pool, &nano(2));
    assert_eq!(vals(g.view(row)), vec![1.0, 20.0, 3.0]);
}

/// `concat` materializes, so it accepts a strided input alongside a contiguous
/// one and produces a contiguous result.
#[test]
fn concat_accepts_strided_and_contiguous_inputs_together() {
    let mut b = Builder::new();
    let (p, pv) = b.source(array::constant(Array::zeros([2, 3])));
    let (q, qv) = b.source(array::constant(Array::zeros([3, 2])));
    let flipped = b.op(array::transpose([1, 0]), pv); // [3, 2], strided
    let joined = b.op(array::concat(1), &[flipped, qv][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(p) = panel23();
    *g.state_mut(q) = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(joined).extents(), [3, 4]);
    assert!(is_contiguous(g.view(joined)));
    assert_eq!(
        vals(g.view(joined)),
        vec![
            1.0, 4.0, 7.0, 8.0, 2.0, 5.0, 9.0, 10.0, 3.0, 6.0, 11.0, 12.0
        ]
    );
}

// ---------------------------------------------------------------------------
// split
// ---------------------------------------------------------------------------

/// `split` is `concat`'s inverse: it cuts one axis into consecutive chunks of
/// the given lengths, publishing one output port per chunk. Feeding the chunks
/// straight back into `concat` must reproduce the input, at every generation.
#[test]
fn split_cuts_an_axis_into_chunks_that_concat_back() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 6]));
    let chunks = b.op(array::split(vec![2, 3, 1], 0), srcv);
    assert_eq!(chunks.len(), 3);
    let rejoined = b.op(array::concat(0), &chunks[..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(chunks[0])), vec![1.0, 2.0]);
    assert_eq!(vals(g.view(chunks[1])), vec![3.0, 4.0, 5.0]);
    assert_eq!(vals(g.view(chunks[2])), vec![6.0]);
    assert_eq!(vals(g.view(rejoined)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    *g.state_mut(src) = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(chunks[1])), vec![30.0, 40.0, 50.0]);
    assert_eq!(
        vals(g.view(rejoined)),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}

/// Splitting the inner axis of a row-major panel produces zero-copy chunks
/// that are necessarily strided — the chunk's rows are not adjacent in the
/// backing buffer.
#[test]
fn split_along_an_inner_axis_yields_strided_chunks() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let chunks = b.op(array::split(vec![1, 2], 1), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(chunks[0]).extents(), [2, 1]);
    assert_eq!(vals(g.view(chunks[0])), vec![1.0, 4.0]);
    assert_eq!(g.view(chunks[1]).extents(), [2, 2]);
    assert_eq!(vals(g.view(chunks[1])), vec![2.0, 3.0, 5.0, 6.0]);
    assert!(!is_contiguous(g.view(chunks[0])));
    assert!(!is_contiguous(g.view(chunks[1])));
}

/// `unstack` is `stack`'s inverse: it drops the split axis, so a rank-`N`
/// panel fans out into rank-`N - 1` slices — contiguous rows along axis 0,
/// strided columns along axis 1.
#[test]
fn unstack_drops_the_split_axis() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([3, 2])));
    let (panel, panelv) = b.source(array::constant(Array::zeros([2, 3])));
    let rows = b.op(array::unstack::<f64, 2, 1>(0), srcv);
    let cols = b.op(array::unstack::<f64, 2, 1>(1), panelv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into();
    *g.state_mut(panel) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(rows.len(), 3);
    assert_eq!(g.view(rows[0]).extents(), [2]);
    assert_eq!(vals(g.view(rows[0])), vec![1.0, 2.0]);
    assert_eq!(vals(g.view(rows[2])), vec![5.0, 6.0]);
    assert!(is_contiguous(g.view(rows[0])));

    assert_eq!(cols.len(), 3);
    assert_eq!(g.view(cols[0]).extents(), [2]);
    assert_eq!(vals(g.view(cols[0])), vec![1.0, 4.0]);
    assert_eq!(vals(g.view(cols[2])), vec![3.0, 6.0]);
    assert!(!is_contiguous(g.view(cols[0])));
}

/// Fanning a panel out with `unstack` and rebuilding it with `stack` is the
/// identity. The join recomputes exactly on the generations the panel is
/// poked: an unrelated source's generation leaves it alone, still publishing
/// the last cross-section.
#[test]
fn unstack_then_stack_rebuilds_the_panel_and_carries_when_idle() {
    let mut b = Builder::new();
    let (panel, panelv) = b.source(array::constant(Array::zeros([3, 2])));
    let (other, otherv) = b.source(array::constant(0.0_f64));
    let rows = b.op(array::unstack::<f64, 2, 1>(0), panelv);
    let rebuilt = b.op(array::stack::<f64, 1, 2>(0), &rows[..]);
    let joins = b.op(runs::<2>(), rebuilt);
    let _unrelated = b.op(runs::<0>(), otherv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert_eq!(g.view(joins), 0);

    *g.state_mut(panel) = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]].into();
    g.stabilize(&mut pool, &nano(1));
    assert_eq!(g.view(rebuilt).extents(), [3, 2]);
    assert_eq!(vals(g.view(rebuilt)), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    assert_eq!(g.view(joins), 1);

    // An unrelated poke must not recompute the join, and must not disturb the
    // cross-section it already published.
    *g.state_mut(other) = (1.0).into();
    g.stabilize(&mut pool, &nano(2));
    assert_eq!(g.view(joins), 1, "unrelated poke recomputed the join");
    assert_eq!(vals(g.view(rebuilt)), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);

    *g.state_mut(panel) = [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]].into();
    g.stabilize(&mut pool, &nano(3));
    assert_eq!(vals(g.view(rebuilt)), vec![4.0, 40.0, 5.0, 50.0, 6.0, 60.0]);
    assert_eq!(g.view(joins), 2);
}

// ---------------------------------------------------------------------------
// view
// ---------------------------------------------------------------------------

/// `slice` narrows each axis to a sub-range, keeping the rank. A rank-1 window
/// stays contiguous; a two-axis window of a panel cannot, since its rows are
/// spaced by the original row length. Being a derived view, it follows pokes.
#[test]
fn slice_narrows_each_axis_and_tracks_the_source() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([5])));
    let (grid, gridv) = b.source(array::constant(Array::zeros([3, 4])));
    let middle = b.op(array::slice((1usize..4,)), srcv);
    let window = b.op(array::slice((1usize..3, 1usize..3)), gridv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [10.0, 20.0, 30.0, 40.0, 50.0].into();
    *g.state_mut(grid) = Array::from_parts([3, 4], (1..=12).map(|i| i as f64).collect());
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(middle).extents(), [3]);
    assert_eq!(vals(g.view(middle)), vec![20.0, 30.0, 40.0]);
    assert!(is_contiguous(g.view(middle)));

    assert_eq!(g.view(window).extents(), [2, 2]);
    assert_eq!(vals(g.view(window)), vec![6.0, 7.0, 10.0, 11.0]);
    assert!(!is_contiguous(g.view(window)));

    *g.state_mut(src) = [1.0, 2.0, 3.0, 4.0, 5.0].into();
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(middle)), vec![2.0, 3.0, 4.0]);
}

/// A stepped `slice` produces a non-contiguous rank-1 view — the cheapest way
/// to hand a downstream operator a strided input and check it does not assume
/// `as_slice` succeeds.
#[test]
fn a_stepped_slice_feeds_a_strided_view_downstream() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant([0.0_f64; 6]));
    let evens = b.op(array::slice([(0usize..6, 2usize)]), srcv);
    let scaled = b.op(array::map(|&x: &f64| x * 10.0), evens);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(evens).extents(), [3]);
    assert_eq!(vals(g.view(evens)), vec![1.0, 3.0, 5.0]);
    assert!(!is_contiguous(g.view(evens)));
    assert_eq!(vals(g.view(scaled)), vec![10.0, 30.0, 50.0]);
}

/// `slice_reshape` drops every axis given as a bare index, so unlike
/// `select_at` — which squeezes exactly one — it can collapse a panel all the
/// way down to a rank-0 element in a single node.
#[test]
fn slice_reshape_drops_every_indexed_axis() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let row = b.op(array::slice_reshape::<_, _, 1, _>((1usize, ..)), srcv);
    let cell = b.op(array::slice_reshape::<_, _, 0, _>((1usize, 2usize)), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(row).extents(), [3]);
    assert_eq!(vals(g.view(row)), vec![4.0, 5.0, 6.0]);
    assert_eq!(g.view(cell).ndim(), 0);
    assert_eq!(vals(g.view(cell)), vec![6.0]);
}

/// A `NewAxis` specifier inserts a unit axis, and consumes no input axis — so
/// the specifier count is independent of the input rank. `slice_reshape` can
/// therefore raise, lower and hold the rank, and mix all three in one node.
#[test]
fn slice_reshape_inserts_new_axes() {
    use tradingflow::data::layout::SliceReshape::NewAxis;

    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    // Two axes in, four out: a unit axis outside and between the originals.
    let spread = b.op(
        array::slice_reshape::<_, _, 4, _>((NewAxis, .., NewAxis, ..)),
        srcv,
    );
    // Index one axis away and insert one: rank preserved, content is a row.
    let row = b.op(
        array::slice_reshape::<_, _, 2, _>((NewAxis, 1usize, ..)),
        srcv,
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(spread).extents(), [1, 2, 1, 3]);
    assert_eq!(vals(g.view(spread)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(g.view(row).extents(), [1, 3]);
    assert_eq!(vals(g.view(row)), vec![4.0, 5.0, 6.0]);
}

/// `transpose` permutes axes by rewriting strides, never data: the permuted
/// view of a row-major panel is strided, and permuting back restores both the
/// original order and contiguity.
#[test]
fn transpose_permutes_axes_without_copying() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let flipped = b.op(array::transpose([1, 0]), srcv);
    let back = b.op(array::transpose([1, 0]), flipped);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(flipped).extents(), [3, 2]);
    assert_eq!(vals(g.view(flipped)), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(!is_contiguous(g.view(flipped)));

    assert_eq!(g.view(back).extents(), [2, 3]);
    assert_eq!(vals(g.view(back)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(is_contiguous(g.view(back)));
}

/// `derive_view` is the escape hatch the other view operators are built from:
/// any `ArrayView -> ArrayView` transform, including compositions and
/// rank changes that no single named view operator spells.
#[test]
fn derive_view_runs_an_arbitrary_view_transform() {
    fn tail_rows(a: ArrayView<'_, f64, 2>) -> ArrayView<'_, f64, 2> {
        a.slice((1usize.., ..))
    }
    fn as_row_panel(a: ArrayView<'_, f64, 1>) -> ArrayView<'_, f64, 2> {
        a.pad_ndim()
    }

    let mut b = Builder::new();
    let (panel, panelv) = b.source(array::constant(Array::zeros([3, 2])));
    let (vector, vectorv) = b.source(array::constant(Array::zeros([3])));
    let tail = b.op(array::derive_view(tail_rows), panelv);
    let padded = b.op(array::derive_view(as_row_panel), vectorv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(panel) = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into();
    *g.state_mut(vector) = [7.0, 8.0, 9.0].into();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(tail).extents(), [2, 2]);
    assert_eq!(vals(g.view(tail)), vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(g.view(padded).extents(), [1, 3]);
    assert_eq!(vals(g.view(padded)), vec![7.0, 8.0, 9.0]);
}

/// `pad_ndim` raises the rank by prepending unit axes, leaving the element
/// order and contiguity untouched — the zero-copy way to line an array up
/// against a higher-rank one. Padding to the input's own rank is the identity.
#[test]
fn pad_ndim_prepends_unit_axes() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(Array::zeros([2, 3])));
    let raised = b.op(array::pad_ndim::<_, 2, 4>(), srcv);
    let same = b.op(array::pad_ndim::<_, 2, 2>(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = panel23();
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(raised).extents(), [1, 1, 2, 3]);
    assert_eq!(vals(g.view(raised)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(is_contiguous(g.view(raised)));
    assert_eq!(g.view(same).extents(), [2, 3]);
    assert!(is_contiguous(g.view(same)));
}
