//! Prototype: const-generic-rank, strided (`std::mdspan`-like) array views as the
//! unified operator currency, validated against the **real** `flowgraph` traits.
//!
//! This is a design spike for the array-layer rework discussed in the design
//! thread, not production code. It demonstrates, end to end and type-checked
//! against `flowgraph::typed`:
//!
//! * `Array<T, const N: usize>` — owned, row-major contiguous; the backing store
//!   that lives in operator `State`.
//! * `View<'a, T, const N: usize>` ≈ `mdspan<T, dextents<usize, N>, layout_stride>`:
//!   `&[T]` + `offset` + per-axis `extents` + per-axis `strides`, all inline.
//!   `Copy`, `Send`, `Sync`, self-contained — **no** by-reference shape, so no
//!   in-state shape storage and (for single outputs) **no arena**.
//! * One `Unary` / `Binary` elementwise core taking a callable, with a
//!   **contiguous fast path** and a **strided fallback** behind a single helper —
//!   so `Add` is `Binary::new(|a, b| a + b)`, `Log` is `Unary::new(|x| x.ln())`.
//! * The two interesting structural shapes:
//!   - `Stack<T, IN, OUT>` — a **rank-changer** carrying its out-rank explicitly
//!     (`OUT == IN + 1` asserted at build), so no `generic_const_exprs`,
//!     no nightly, no rank cap. Emits a by-value `ViewPort` borrowing its state.
//!   - `SliceView<T, IN, OUT>` — a zero-copy **strided** squeeze (`OUT == IN - 1`)
//!     that lends a by-value `ViewPort` of its input with **no `State` shape and
//!     no arena** — the boilerplate the unification is meant to delete.
//!
//! The flowgraph integration is exercised by implementing `Operator` / `Segment`
//! and calling `compute` / `passthrough` directly across two "generations" — that
//! covers the load-bearing lifetime claim (a gated `Operator` may return a
//! `ViewPort` value borrowing `&'b mut`/`&'b` `State`, `'b: 'a`) without needing
//! the full `Scenario` engine.

#![allow(clippy::needless_range_loop)]

use std::marker::PhantomData;

use flowgraph::typed::{Operator, RefViewPorts, Segment, ValueView, ViewPort};

// ===========================================================================
// Scalar bound
// ===========================================================================

/// Permitted element type for the prototype. `Copy` keeps the elementwise core
/// readable; `Send + Sync + 'static` are what the engine's view wire requires.
trait Num: Copy + Default + Send + Sync + 'static {}
impl Num for f64 {}

// ===========================================================================
// Shape — inline extents + strides (the mdspan `layout_stride` mapping)
// ===========================================================================

#[derive(Clone, Copy)]
struct Shape<const N: usize> {
    extents: [usize; N],
    strides: [usize; N],
}

impl<const N: usize> Shape<N> {
    /// Canonical row-major (C-contiguous) strides for `extents`.
    fn row_major(extents: [usize; N]) -> Self {
        let mut strides = [0usize; N];
        let mut acc = 1usize;
        let mut d = N;
        while d > 0 {
            d -= 1;
            strides[d] = acc;
            acc *= extents[d];
        }
        Shape { extents, strides }
    }

    /// Element count (empty product = 1, i.e. a rank-0 scalar holds one element).
    fn len(&self) -> usize {
        self.extents.iter().product()
    }

    /// Whether the strides are exactly canonical row-major for the extents —
    /// the predicate that gates the contiguous fast path (mdspan `is_exhaustive`).
    fn is_contiguous(&self) -> bool {
        self.strides == Shape::row_major(self.extents).strides
    }
}

// ===========================================================================
// Array<T, N> — owned, row-major contiguous backing store
// ===========================================================================

struct Array<T: Num, const N: usize> {
    data: Box<[T]>,
    shape: Shape<N>,
}

impl<T: Num, const N: usize> Array<T, N> {
    fn zeros(extents: [usize; N]) -> Self {
        let shape = Shape::row_major(extents);
        Self {
            data: vec![T::default(); shape.len()].into_boxed_slice(),
            shape,
        }
    }

    fn from_elems(extents: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(data.len(), shape.len(), "from_elems: length mismatch");
        Self {
            data: data.into_boxed_slice(),
            shape,
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Lend the whole array as a contiguous view (strides row-major-derived).
    fn view(&self) -> View<'_, T, N> {
        View {
            data: &self.data,
            offset: 0,
            shape: self.shape,
        }
    }
}

// ===========================================================================
// View<'a, T, N> — strided, Copy, self-contained
// ===========================================================================

struct View<'a, T: Num, const N: usize> {
    /// The whole backing buffer; `offset` + `strides` address into it.
    data: &'a [T],
    offset: usize,
    shape: Shape<N>,
}

// Manual (not derived) `Clone`/`Copy`: the view is references + plain `usize`s,
// so it is `Copy` regardless of whether `T` is — exactly like the existing
// `ArraySlice`. (Derived `Copy` would wrongly demand `T: Copy`.)
impl<T: Num, const N: usize> Clone for View<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Num, const N: usize> Copy for View<'_, T, N> {}

impl<'a, T: Num, const N: usize> View<'a, T, N> {
    /// The contiguous fast path: `Some(flat slice)` iff the view is exhaustive.
    fn contiguous_slice(&self) -> Option<&'a [T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[self.offset..self.offset + self.shape.len()])
        } else {
            None
        }
    }
}

// ===========================================================================
// flowgraph view-kind: ArrayValue<T, N> carries View<'a, T, N> on the wire
// ===========================================================================

struct ArrayValue<T: Num, const N: usize>(PhantomData<T>);

// SAFETY: `View<'a, T, N>` holds only `&'a [T]` plus plain `usize`s, so it is
// covariant in `'a` — the sole `ValueView` obligation.
unsafe impl<T: Num, const N: usize> ValueView for ArrayValue<T, N> {
    type View<'a> = View<'a, T, N>;
}

// ===========================================================================
// Elementwise core — contiguous fast path + strided fallback, one helper each
// ===========================================================================

/// Advance a mixed-radix logical index one step in row-major order, tracking the
/// running flat offset incrementally (no per-element `O(N)` recompute).
#[inline]
fn advance<const N: usize>(idx: &mut [usize; N], ext: &[usize; N], strd: &[usize; N], off: &mut usize) {
    let mut d = N;
    while d > 0 {
        d -= 1;
        idx[d] += 1;
        *off += strd[d];
        if idx[d] < ext[d] {
            return;
        }
        idx[d] = 0;
        *off -= ext[d] * strd[d];
    }
}

fn apply_unary<T: Num, const N: usize>(out: &mut Array<T, N>, x: &View<T, N>, f: &impl Fn(T) -> T) {
    let o = out.as_mut_slice();
    if let Some(s) = x.contiguous_slice() {
        for (dst, &v) in o.iter_mut().zip(s) {
            *dst = f(v);
        }
        return;
    }
    let (ext, strd) = (x.shape.extents, x.shape.strides);
    let mut idx = [0usize; N];
    let mut off = x.offset;
    for dst in o.iter_mut() {
        *dst = f(x.data[off]);
        advance(&mut idx, &ext, &strd, &mut off);
    }
}

fn apply_binary<T: Num, const N: usize>(
    out: &mut Array<T, N>,
    a: &View<T, N>,
    b: &View<T, N>,
    f: &impl Fn(T, T) -> T,
) {
    let o = out.as_mut_slice();
    if let (Some(sa), Some(sb)) = (a.contiguous_slice(), b.contiguous_slice()) {
        for (dst, (&va, &vb)) in o.iter_mut().zip(sa.iter().zip(sb)) {
            *dst = f(va, vb);
        }
        return;
    }
    // Strided: `a` and `b` share extents; each tracks its own offset.
    let ext = a.shape.extents;
    let (stra, strb) = (a.shape.strides, b.shape.strides);
    let mut idx = [0usize; N];
    let (mut oa, mut ob) = (a.offset, b.offset);
    for dst in o.iter_mut() {
        *dst = f(a.data[oa], b.data[ob]);
        let mut d = N;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            oa += stra[d];
            ob += strb[d];
            if idx[d] < ext[d] {
                break;
            }
            idx[d] = 0;
            oa -= ext[d] * stra[d];
            ob -= ext[d] * strb[d];
        }
    }
}

/// Copy a (possibly strided) view into a contiguous destination in row-major
/// order — the materialization primitive the rank-changer needs.
fn write_row_major<T: Num, const N: usize>(dst: &mut [T], v: &View<T, N>) {
    if let Some(s) = v.contiguous_slice() {
        dst.copy_from_slice(s);
        return;
    }
    let (ext, strd) = (v.shape.extents, v.shape.strides);
    let mut idx = [0usize; N];
    let mut off = v.offset;
    for d in dst.iter_mut() {
        *d = v.data[off];
        advance(&mut idx, &ext, &strd, &mut off);
    }
}

// ===========================================================================
// Unary — one elementwise operator, parameterized by a callable
// ===========================================================================

struct Unary<T: Num, const N: usize, F> {
    f: F,
    _p: PhantomData<T>,
}

impl<T: Num, const N: usize, F: Fn(T) -> T + Send + 'static> Unary<T, N, F> {
    fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

struct UnaryState<T: Num, const N: usize, F> {
    f: F,
    out: Array<T, N>,
}

impl<T: Num, const N: usize, F: Fn(T) -> T + Send + 'static> Operator for Unary<T, N, F> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = UnaryState<T, N, F>;

    fn init(self) -> Self::State {
        UnaryState {
            f: self.f,
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, View<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, View<'a, T, N>) {
        if init {
            state.out = Array::zeros(x.shape.extents);
        }
        apply_unary(&mut state.out, &x, &state.f);
        // By-value view borrowing state's (in-place, non-realloc) buffer:
        // `'b: 'a` lets it ride out as the `'a` output. passthrough re-lends it.
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (!init, out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, View<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, View<'a, T, N>) {
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (false, out)
    }
}

// ===========================================================================
// Binary — the two-input elementwise operator
// ===========================================================================

struct Binary<T: Num, const N: usize, F> {
    f: F,
    _p: PhantomData<T>,
}

impl<T: Num, const N: usize, F: Fn(T, T) -> T + Send + 'static> Binary<T, N, F> {
    fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

struct BinaryState<T: Num, const N: usize, F> {
    f: F,
    out: Array<T, N>,
}

impl<T: Num, const N: usize, F: Fn(T, T) -> T + Send + 'static> Operator for Binary<T, N, F> {
    type Inputs = (ViewPort<ArrayValue<T, N>>, ViewPort<ArrayValue<T, N>>);
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = BinaryState<T, N, F>;

    fn init(self) -> Self::State {
        BinaryState {
            f: self.f,
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, View<'a, T, N>), (bool, View<'a, T, N>)),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, View<'a, T, N>) {
        if init {
            state.out = Array::zeros(a.shape.extents);
        }
        apply_binary(&mut state.out, &a, &b, &state.f);
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (!init, out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, View<'a, T, N>), (bool, View<'a, T, N>)),
        state: &'b Self::State,
    ) -> (bool, View<'a, T, N>) {
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (false, out)
    }
}

// ===========================================================================
// Stack<T, IN, OUT> — rank-changer with explicit out-rank (no const arithmetic)
// ===========================================================================

struct Stack<T: Num, const IN: usize, const OUT: usize> {
    _p: PhantomData<T>,
}

impl<T: Num, const IN: usize, const OUT: usize> Stack<T, IN, OUT> {
    fn new() -> Self {
        Self { _p: PhantomData }
    }
}

struct StackState<T: Num, const OUT: usize> {
    out: Array<T, OUT>,
}

impl<T: Num, const IN: usize, const OUT: usize> Operator for Stack<T, IN, OUT> {
    type Inputs = RefViewPorts<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = StackState<T, OUT>;

    fn init(self) -> Self::State {
        StackState {
            out: Array::zeros([0; OUT]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [&'a View<'a, T, IN>]),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, View<'a, T, OUT>) {
        if init {
            assert!(OUT == IN + 1, "Stack: OUT must be IN + 1");
            assert!(!views.is_empty(), "Stack requires >= 1 input");
            let ie = views[0].shape.extents;
            let mut oe = [0usize; OUT];
            oe[0] = views.len(); // new axis 0
            for d in 0..IN {
                oe[d + 1] = ie[d];
            }
            state.out = Array::zeros(oe);
        }
        let chunk = views[0].shape.len();
        let dst = state.out.as_mut_slice();
        for (k, &v) in views.iter().enumerate() {
            write_row_major(&mut dst[k * chunk..(k + 1) * chunk], v);
        }
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (!init, out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a View<'a, T, IN>]),
        state: &'b Self::State,
    ) -> (bool, View<'a, T, OUT>) {
        let out = View {
            data: state.out.as_slice(),
            offset: 0,
            shape: state.out.shape,
        };
        (false, out)
    }
}

// ===========================================================================
// SliceView<T, IN, OUT> — zero-copy strided squeeze; NO state shape, NO arena
// ===========================================================================

struct SliceView<T: Num, const IN: usize, const OUT: usize> {
    axis: usize,
    index: usize,
    _p: PhantomData<T>,
}

impl<T: Num, const IN: usize, const OUT: usize> SliceView<T, IN, OUT> {
    fn new(axis: usize, index: usize) -> Self {
        Self {
            axis,
            index,
            _p: PhantomData,
        }
    }
}

/// State is *only* the config — no `Box<[usize]>` shape, no `Arena`. The lent
/// view is re-derived by value from the fresh input each generation.
struct SliceViewState {
    axis: usize,
    index: usize,
}

impl<T: Num, const IN: usize, const OUT: usize> Segment for SliceView<T, IN, OUT> {
    type Inputs = ViewPort<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = SliceViewState;

    fn init(self) -> Self::State {
        SliceViewState {
            axis: self.axis,
            index: self.index,
        }
    }

    fn compute<'a, 'b: 'a>(
        (notified, v): (bool, View<'a, T, IN>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, View<'a, T, OUT>) {
        if init {
            assert!(IN == OUT + 1, "SliceView: OUT must be IN - 1 (squeeze)");
            assert!(state.axis < IN, "SliceView: axis out of range");
            assert!(
                state.index < v.shape.extents[state.axis],
                "SliceView: index out of range",
            );
        }
        // Drop `axis`, keeping every other axis's extent AND stride: the result
        // is a strided window into the SAME buffer — no copy, no owned shape.
        let (ie, is) = (v.shape.extents, v.shape.strides);
        let mut oe = [0usize; OUT];
        let mut os = [0usize; OUT];
        let mut j = 0;
        for d in 0..IN {
            if d == state.axis {
                continue;
            }
            oe[j] = ie[d];
            os[j] = is[d];
            j += 1;
        }
        let out = View {
            data: v.data,
            offset: v.offset + state.index * is[state.axis],
            shape: Shape {
                extents: oe,
                strides: os,
            },
        };
        (notified && !init, out)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Drive a `Unary` across init + one steady generation, returning the
/// materialized output. Generic so the (unnameable) closure type `F` is in scope
/// for the qualified `Operator::compute` call.
fn run_unary<const N: usize, F>(f: F, x: View<f64, N>) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Send + 'static,
{
    let mut st = Operator::init(Unary::<f64, N, F>::new(f));
    let _ = <Unary<f64, N, F> as Operator>::compute((true, x), &mut st, true);
    let (_, out) = <Unary<f64, N, F> as Operator>::compute((true, x), &mut st, false);
    out.contiguous_slice().unwrap().to_vec()
}

fn run_binary<const N: usize, F>(f: F, a: View<f64, N>, b: View<f64, N>) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Send + 'static,
{
    let mut st = Operator::init(Binary::<f64, N, F>::new(f));
    let _ = <Binary<f64, N, F> as Operator>::compute(((true, a), (true, b)), &mut st, true);
    let (_, out) = <Binary<f64, N, F> as Operator>::compute(((true, a), (true, b)), &mut st, false);
    out.contiguous_slice().unwrap().to_vec()
}

#[test]
fn view_is_cheap_inline_and_copy() {
    use std::mem::size_of;
    let word = size_of::<usize>();
    let fatptr = size_of::<&[f64]>(); // 2 words
    // data(&[T]) + offset(usize) + extents[N] + strides[N] — all inline, no heap.
    assert_eq!(size_of::<View<f64, 1>>(), fatptr + word * (1 + 2 * 1));
    assert_eq!(size_of::<View<f64, 2>>(), fatptr + word * (1 + 2 * 2));
    assert_eq!(size_of::<View<f64, 3>>(), fatptr + word * (1 + 2 * 3));
    fn assert_copy<T: Copy>() {}
    assert_copy::<View<f64, 1>>();
    assert_copy::<View<f64, 8>>(); // rank 8 is still just `Copy` inline data
}

#[test]
fn unary_contiguous_and_strided_agree() {
    // Add/Log-style unary via a callable, contiguous input.
    let x = Array::<f64, 1>::from_elems([3], vec![1.0, 4.0, 9.0]);
    assert_eq!(run_unary(|v: f64| v.sqrt(), x.view()), vec![1.0, 2.0, 3.0]);

    // Strided input: column 1 of a [3, 4] row-major panel (extent 3, stride 4).
    // Exercises the strided fallback in `apply_unary`.
    let panel = Array::<f64, 2>::from_elems(
        [3, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0,
        ],
    );
    let col1 = View {
        data: panel.as_slice(),
        offset: 1,
        shape: Shape {
            extents: [3],
            strides: [4],
        },
    };
    assert!(col1.contiguous_slice().is_none(), "column must be strided");
    assert_eq!(run_unary(|v: f64| v * 10.0, col1), vec![10.0, 50.0, 90.0]);
}

#[test]
fn binary_is_add_via_callable() {
    let a = Array::<f64, 2>::from_elems([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::<f64, 2>::from_elems([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(
        run_binary(|x, y| x + y, a.view(), b.view()),
        vec![11.0, 22.0, 33.0, 44.0]
    );
}

#[test]
fn binary_mixed_contiguous_and_strided() {
    // a contiguous [3], b strided (a column) [3] — forces the strided path with
    // two distinct stride sets sharing extents.
    let a = Array::<f64, 1>::from_elems([3], vec![100.0, 200.0, 300.0]);
    let panel = Array::<f64, 2>::from_elems([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bcol = View {
        data: panel.as_slice(),
        offset: 1,
        shape: Shape {
            extents: [3],
            strides: [2],
        },
    };
    assert_eq!(
        run_binary(|x, y| x + y, a.view(), bcol),
        vec![102.0, 204.0, 306.0] // 100+col[0]=2, 200+col[1]=4, 300+col[2]=6
    );
}

#[test]
fn stack_changes_rank_with_explicit_out_rank() {
    let a = Array::<f64, 1>::from_elems([4], vec![0.0, 1.0, 2.0, 3.0]);
    let b = Array::<f64, 1>::from_elems([4], vec![4.0, 5.0, 6.0, 7.0]);
    let c = Array::<f64, 1>::from_elems([4], vec![8.0, 9.0, 10.0, 11.0]);
    let (va, vb, vc) = (a.view(), b.view(), c.view());
    let refs: [&View<f64, 1>; 3] = [&va, &vb, &vc];
    let flags = [true; 3];

    let mut st = Operator::init(Stack::<f64, 1, 2>::new());
    let _ = <Stack<f64, 1, 2> as Operator>::compute((&flags, &refs), &mut st, true);
    let (notify, out) = <Stack<f64, 1, 2> as Operator>::compute((&flags, &refs), &mut st, false);

    assert!(notify);
    assert_eq!(out.shape.extents, [3, 4]);
    assert_eq!(
        out.contiguous_slice().unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn sliceview_is_zero_copy_strided_squeeze_with_no_state_shape() {
    // [3, 4] panel; take column 2 → a strided View<f64, 1> over the SAME buffer.
    let panel = Array::<f64, 2>::from_elems(
        [3, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0,
        ],
    );
    let mut st = SliceView::<f64, 2, 1>::new(/*axis=*/ 1, /*index=*/ 2).init();
    let (_, col) =
        <SliceView<f64, 2, 1> as Segment>::compute((true, panel.view()), &mut st, true);

    // Same backing buffer (zero copy), strided addressing.
    assert!(std::ptr::eq(col.data.as_ptr(), panel.as_slice().as_ptr()));
    assert_eq!(col.shape.extents, [3]);
    assert_eq!(col.shape.strides, [4]);
    assert!(col.contiguous_slice().is_none());

    // And it reads the right elements when materialized through the unary core.
    assert_eq!(run_unary(|v: f64| v, col), vec![2.0, 6.0, 10.0]);
    // State carries only the config — no Box<[usize]> shape, no Arena.
    assert_eq!((st.axis, st.index), (1, 2));
}

#[test]
fn passthrough_re_lends_stable_buffer_across_generations() {
    // The no-notify ⟹ unchanged contract: passthrough hands back a view of the
    // same in-place (never-reallocated) state buffer, equal to the last compute.
    let x = Array::<f64, 1>::from_elems([3], vec![2.0, 3.0, 4.0]);
    let mut st = Operator::init(Unary::<f64, 1, _>::new(|v: f64| v * v));

    let _ = <Unary<f64, 1, _> as Operator>::compute((true, x.view()), &mut st, true);
    let from_compute = {
        let (n, out) = <Unary<f64, 1, _> as Operator>::compute((true, x.view()), &mut st, false);
        assert!(n);
        out.contiguous_slice().unwrap().to_vec()
    };
    let from_passthrough = {
        let (n, out) = <Unary<f64, 1, _> as Operator>::passthrough((false, x.view()), &st);
        assert!(!n, "passthrough must not notify");
        out.contiguous_slice().unwrap().to_vec()
    };
    assert_eq!(from_compute, vec![4.0, 9.0, 16.0]);
    assert_eq!(from_compute, from_passthrough);
}

// ===========================================================================
// Inference experiment: can the INPUT HANDLE TYPES drive the operator's generic
// params (the rank), with NO turbofish? This mirrors the real `add_operator`.
// ===========================================================================
//
// The problem: today `add_operator<O>(op, inputs: <O::Inputs as Handles>::H)`
// derives the handle type as a *forward projection of `O`*. Inference cannot run
// that backwards, so `O`'s generics (here the rank `N`) stay ambiguous and need a
// turbofish. The fix below inverts the dependency.

/// A build-time handle to a wire of interface `I` (mock of flowgraph's `Handle`,
/// which is likewise structural in its port/interface type).
struct Handle<I>(PhantomData<I>);

// Manual (not derived) so `Handle<I>: Copy` never demands `I: Copy` — the
// interface types are zero-size markers, not `Copy` values.
impl<I> Clone for Handle<I> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<I> Copy for Handle<I> {}

/// The missing INVERSE: recover the interface *from* a handle tree. Because a
/// handle is structural in `I`, `H` is inferred directly from the `inputs`
/// argument, and `H::Interface` then normalizes forward — breaking the cycle.
trait HandleTree {
    type Interface;
}
impl<I> HandleTree for Handle<I> {
    type Interface = I;
}
impl<A: HandleTree, B: HandleTree> HandleTree for (A, B) {
    type Interface = (A::Interface, B::Interface);
}

/// Forward projection for the RETURN handles (interface -> handle tree). Forward
/// is fine: `O` is fully known by the time we name the outputs.
trait ToHandles {
    type Handles;
}
impl<T: Num, const N: usize> ToHandles for ViewPort<ArrayValue<T, N>> {
    type Handles = Handle<ViewPort<ArrayValue<T, N>>>;
}
impl<A: ToHandles, B: ToHandles> ToHandles for (A, B) {
    type Handles = (A::Handles, B::Handles);
}

/// The proposed `add_operator` shape. `H` (the handles) is a FREE type param,
/// inferred STRUCTURALLY from `inputs`; the operator's inputs are then pinned by
/// `O: Operator<Inputs = H::Interface>`, so the operator's generics flow OUT of
/// the handle types. No turbofish.
fn wire<O, H>(_op: O, _inputs: H) -> <O::Outputs as ToHandles>::Handles
where
    H: HandleTree,
    O: Operator<Inputs = H::Interface>,
    O::Outputs: ToHandles,
{
    // Never executed (the inference checks below are dead-code, compiled not run);
    // the proof is that they TYPE-CHECK with no annotations.
    unimplemented!()
}

// If `N` did not infer from the handle, each of these would fail with "type
// annotations needed". They compile ⟹ the rank flowed out of the wiring.

#[allow(dead_code)]
fn _infer_unary_rank_from_handle() {
    let h2: Handle<ViewPort<ArrayValue<f64, 2>>> = Handle(PhantomData);
    // No `Unary::<f64, 2, _>` turbofish, no output annotation.
    let _out = wire(Unary::new(|x: f64| x.sqrt()), h2);
}

#[allow(dead_code)]
fn _infer_rank_propagates_through_a_chain() {
    let h1: Handle<ViewPort<ArrayValue<f64, 1>>> = Handle(PhantomData);
    let a = wire(Unary::new(|x: f64| x.sqrt()), h1); // a: Handle<..f64, 1>
    let b = wire(Unary::new(|x: f64| x + 1.0), a); // rank 1 flows a -> b
    let _c = wire(Binary::new(|x: f64, y: f64| x * y), (a, b)); // tuple, rank 1
}

#[allow(dead_code)]
fn _infer_rank_2_tuple() {
    let p: Handle<ViewPort<ArrayValue<f64, 2>>> = Handle(PhantomData);
    let q: Handle<ViewPort<ArrayValue<f64, 2>>> = Handle(PhantomData);
    let _r = wire(Binary::new(|x: f64, y: f64| x + y), (p, q)); // N=2 from p, q
}

#[test]
fn handle_driven_inference_compiles() {
    // The real assertions are the three `_infer_*` fns above: if the rank did not
    // flow from the handle types, this test crate would not compile.
}

#[test]
fn pipeline_stack_then_strided_slice_then_unary() {
    // Materialize (Stack, IN=1→OUT=2) → zero-copy strided squeeze (SliceView,
    // 2→1) → strided elementwise (Unary). The whole chain in the unified view
    // currency, with exactly one copy (the Stack materialization).
    let a = Array::<f64, 1>::from_elems([4], vec![0.0, 1.0, 2.0, 3.0]);
    let b = Array::<f64, 1>::from_elems([4], vec![4.0, 5.0, 6.0, 7.0]);
    let c = Array::<f64, 1>::from_elems([4], vec![8.0, 9.0, 10.0, 11.0]);
    let (va, vb, vc) = (a.view(), b.view(), c.view());
    let refs: [&View<f64, 1>; 3] = [&va, &vb, &vc];
    let flags = [true; 3];

    let mut stk = Operator::init(Stack::<f64, 1, 2>::new());
    let (_, panel) = <Stack<f64, 1, 2> as Operator>::compute((&flags, &refs), &mut stk, true);
    // panel: View<f64, 2> [3, 4] borrowing `stk.out`.

    let mut slc = SliceView::<f64, 2, 1>::new(1, 3).init(); // column 3
    let (_, col) = <SliceView<f64, 2, 1> as Segment>::compute((true, panel), &mut slc, true);
    // col: strided View<f64, 1> = [3, 7, 11], still borrowing `stk.out`.

    assert_eq!(run_unary(|v: f64| v + 0.5, col), vec![3.5, 7.5, 11.5]);
}
