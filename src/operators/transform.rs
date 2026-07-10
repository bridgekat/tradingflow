//! Function / selection / lag operators over the strided [`ArrayView`]
//! currency — `Map`/`MapInplace`, `Apply`/`ApplyInplace` (closure compute),
//! `Select` (materializing selection), [`SliceView`] (zero-copy strided
//! squeeze), and `Lag` (a `Series` element from N steps ago).
//!
//! The closure operators receive the values-only views tree (notify flags
//! stripped via [`StripNotify`]) and return an owned [`Array`], which is homed
//! in `State` and lent as a `ViewPort` view — so a closure reads strided inputs
//! and the result composes as the same view currency.

use std::marker::PhantomData;

use bumpalo::Bump;
use flowgraph::typed::{Interface, Operator, RefPort, RefViewPort, Segment, ViewPort};

use super::op::{ArrayValue, StripNotify};
use crate::data::array::Shape;
use crate::{Array, ArrayView, Scalar, Series};

// ---------------------------------------------------------------------------
// Map / MapInplace (single input)
// ---------------------------------------------------------------------------

/// Allocating map: applies `Fn(ArrayView<SI, NI>) -> Array<SO, NO>` each tick,
/// homing the result and lending a view of it.
pub struct Map<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> {
    f: F,
    _phantom: PhantomData<(SI, SO)>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Map`]: the function plus the output cell, `None` only
/// until the build call runs the closure once.
pub struct MapState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Option<Array<SO, NO>>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = ViewPort<ArrayValue<SI, NI>>;
    type Outputs = ViewPort<ArrayValue<SO, NO>>;
    type State = MapState<SO, NO, F>;

    fn init(self) -> Self::State {
        MapState {
            f: self.f,
            out: None,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        // The build call runs the closure once on the build-time input to seed
        // the output — replicating the legacy `init` — but does not notify.
        state.out = Some((state.f)(x));
        (!init, state.out.as_ref().unwrap().view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.as_ref().unwrap().view())
    }
}

/// In-place map: `Fn(ArrayView<SI, NI>, &mut Array<SO, NO>) -> bool`; the bool
/// controls propagation.
pub struct MapInplace<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> {
    f: F,
    initial: Array<SO, NO>,
    _phantom: PhantomData<SI>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    pub fn new(f: F, initial: Array<SO, NO>) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`MapInplace`]: the function, the initial value, and the
/// output buffer.
pub struct MapInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
    initial: Array<SO, NO>,
    out: Array<SO, NO>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    type Inputs = ViewPort<ArrayValue<SI, NI>>;
    type Outputs = ViewPort<ArrayValue<SO, NO>>;
    type State = MapInplaceState<SO, NO, F>;

    fn init(self) -> Self::State {
        MapInplaceState {
            f: self.f,
            out: self.initial.clone(),
            initial: self.initial,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        if init {
            state.out = state.initial.clone();
            (state.f)(x, &mut state.out);
            return (false, state.out.view());
        }
        let notify = (state.f)(x, &mut state.out);
        (notify, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Apply / ApplyInplace (tuple inputs)
// ---------------------------------------------------------------------------

/// Allocating multi-input map: `Fn(views) -> Array<SO, NO>`, where `views` is
/// the values-only views tree of the input interface `I` (e.g.
/// `(ArrayView<f64, 2>, ArrayView<f64, 2>)` for two `ViewPort` ports — see
/// [`StripNotify`]).
pub struct Apply<I, SO: Scalar, const NO: usize, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<fn() -> (I, SO)>,
}

impl<I, SO: Scalar, const NO: usize, F> Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Apply`]: the function plus the output cell, `None` only
/// until the build call runs the closure once.
pub struct ApplyState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Option<Array<SO, NO>>,
}

impl<I, SO: Scalar, const NO: usize, F> Operator for Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = I;
    type Outputs = ViewPort<ArrayValue<SO, NO>>;
    type State = ApplyState<SO, NO, F>;

    fn init(self) -> Self::State {
        ApplyState {
            f: self.f,
            out: None,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        state.out = Some((state.f)(<I as StripNotify>::plain(inputs)));
        (!init, state.out.as_ref().unwrap().view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.as_ref().unwrap().view())
    }
}

/// In-place multi-input map: `Fn(views, &mut Array<SO, NO>) -> bool`.
pub struct ApplyInplace<I, SO: Scalar, const NO: usize, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    f: F,
    initial: Array<SO, NO>,
    _phantom: PhantomData<fn() -> I>,
}

impl<I, SO: Scalar, const NO: usize, F> ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    pub fn new(f: F, initial: Array<SO, NO>) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`ApplyInplace`]: the function, the initial value, and the
/// output buffer.
pub struct ApplyInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
    initial: Array<SO, NO>,
    out: Array<SO, NO>,
}

impl<I, SO: Scalar, const NO: usize, F> Operator for ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    type Inputs = I;
    type Outputs = ViewPort<ArrayValue<SO, NO>>;
    type State = ApplyInplaceState<SO, NO, F>;

    fn init(self) -> Self::State {
        ApplyInplaceState {
            f: self.f,
            out: self.initial.clone(),
            initial: self.initial,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        if init {
            state.out = state.initial.clone();
            (state.f)(<I as StripNotify>::plain(inputs), &mut state.out);
            return (false, state.out.view());
        }
        let notify = (state.f)(<I as StripNotify>::plain(inputs), &mut state.out);
        (notify, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Select (index selection along an axis — the materialization point)
// ---------------------------------------------------------------------------

/// Select elements along an axis into an owned, contiguous rank-`OUT` array
/// (`OUT == IN` for a plain selection, `OUT == IN - 1` when squeezing a single
/// index). Accepts a strided input view; the selection is the **materialization
/// point** of a view chain — it retains the last computed selection in owned
/// state, preserving the carry semantics downstream `Stack`-style readers rely
/// on. For a zero-copy alternative see [`SliceView`].
pub struct Select<T: Scalar, const IN: usize, const OUT: usize> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Clone for Select<T, IN, OUT> {
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            axis: self.axis,
            squeeze: self.squeeze,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const IN: usize, const OUT: usize> Select<T, IN, OUT> {
    pub fn new(indices: Vec<usize>, axis: usize, squeeze: bool) -> Self {
        assert!(
            !squeeze || indices.len() == 1,
            "squeeze requires exactly one index, got {}",
            indices.len(),
        );
        assert!(
            OUT == IN - squeeze as usize,
            "Select: OUT ({OUT}) must be IN ({IN}) minus {} (squeeze={squeeze})",
            squeeze as usize,
        );
        Self {
            indices,
            axis,
            squeeze,
            _phantom: PhantomData,
        }
    }

    pub fn flat(indices: Vec<usize>) -> Self {
        Self::new(indices, 0, false)
    }

    pub fn along_axis(indices: Vec<usize>, axis: usize) -> Self {
        Self::new(indices, axis, false)
    }
}

/// Runtime state for [`Select`]: the configuration, the resolved flat index map
/// (computable only on the build call once the input shape is known), and the
/// output buffer.
pub struct SelectState<T: Scalar, const OUT: usize> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    index_map: Vec<usize>,
    out: Array<T, OUT>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Select<T, IN, OUT> {
    type Inputs = ViewPort<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = SelectState<T, OUT>;

    fn init(self) -> Self::State {
        SelectState {
            indices: self.indices,
            axis: self.axis,
            squeeze: self.squeeze,
            index_map: Vec::new(),
            out: Array::zeros([0; OUT]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        if init {
            let input_extents = x.extents();
            state.index_map = compute_select_map(&input_extents, &state.indices, state.axis);
            let out_extents = select_out_extents::<OUT>(
                &input_extents,
                state.indices.len(),
                state.axis,
                state.squeeze,
            );
            // Seed the initial output with the actual selection of the build-time
            // input (NOT zeros — a fabricated finite observation leaks through
            // carry-style consumers; the faithful selection of a NaN-initialised
            // panel correctly reads "no data yet").
            state.out = Array::from_vec(
                out_extents,
                state.index_map.iter().map(|&s| src[s].clone()).collect(),
            );
            return (false, state.out.view());
        }
        let dst = state.out.as_mut_slice();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, IN>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// The row-major flat index map of selecting `indices` along `axis`.
fn compute_select_map(input_extents: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    if input_extents.is_empty() {
        return indices.to_vec();
    }
    let outer: usize = input_extents[..axis].iter().product();
    let inner: usize = input_extents[axis + 1..].iter().product();
    let axis_size = input_extents[axis];
    let mut map = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &idx in indices {
            for i in 0..inner {
                map.push(o * axis_size * inner + idx * inner + i);
            }
        }
    }
    map
}

/// The output extents of a select, as a static `[usize; OUT]`.
fn select_out_extents<const OUT: usize>(
    input_extents: &[usize],
    n_selected: usize,
    axis: usize,
    squeeze: bool,
) -> [usize; OUT] {
    let mut v = input_extents.to_vec();
    if v.is_empty() {
        v = vec![n_selected];
    } else {
        v[axis] = n_selected;
    }
    if squeeze && n_selected == 1 && v.len() > axis {
        v.remove(axis);
    }
    <[usize; OUT]>::try_from(v.as_slice())
        .unwrap_or_else(|_| panic!("Select: output rank {} != OUT {OUT}", v.len()))
}

// ---------------------------------------------------------------------------
// SliceView (zero-copy strided selection → borrowed view, no arena/no copy)
// ---------------------------------------------------------------------------

/// Zero-copy selection: re-derives a **strided** [`ArrayView`] of a contiguous
/// run along one axis (a range, or a single index with `squeeze`) **by value**,
/// over the input's own buffer — no copy, no owned storage, no per-generation
/// arena. The view-chain counterpart of [`Select`]: where `Select` materializes
/// (the retention point for a carry-style [`Stack`](super::Stack)), `SliceView`
/// keeps the data as a view into its input's storage.
///
/// It is correct precisely because every operator honours the no-notify⟹
/// unchanged contract: the lent view reads the input's stable storage (a
/// retaining [`Gate`](super::Gate) or an owned compute output), which only
/// changes when the input notifies, so a carry-style
/// [`Stack`](super::Stack) reader sees the last notified value for an
/// un-notified stock. Implements [`Segment`] directly (the by-value view is
/// derived from the fresh input, not from state).
pub struct SliceView<T: Scalar, const IN: usize, const OUT: usize> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> SliceView<T, IN, OUT> {
    pub fn new(indices: Vec<usize>, axis: usize, squeeze: bool) -> Self {
        assert!(
            !squeeze || indices.len() == 1,
            "squeeze requires exactly one index, got {}",
            indices.len(),
        );
        assert!(
            OUT == IN - squeeze as usize,
            "SliceView: OUT ({OUT}) must be IN ({IN}) minus {} (squeeze={squeeze})",
            squeeze as usize,
        );
        Self {
            indices,
            axis,
            squeeze,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`SliceView`]: just the config plus the resolved
/// `(axis, start, count)` — **no** owned shape, **no** arena.
pub struct SliceViewState {
    axis: usize,
    squeeze: bool,
    start: usize,
    count: usize,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Segment for SliceView<T, IN, OUT> {
    type Inputs = ViewPort<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = SliceViewState;

    fn init(self) -> Self::State {
        // Resolve the contiguous run `[start, start+count)` from the index list.
        assert!(self.axis < IN, "SliceView: axis {} out of range", self.axis);
        assert!(!self.indices.is_empty(), "SliceView requires >= 1 index");
        assert!(
            self.indices.windows(2).all(|w| w[1] == w[0] + 1),
            "SliceView requires a contiguous index range along the axis; got {:?}",
            self.indices,
        );
        SliceViewState {
            axis: self.axis,
            squeeze: self.squeeze,
            start: self.indices[0],
            count: self.indices.len(),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, v): (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        let data = v.buffer();
        let shape = v.shape();
        let (ext, strd) = (shape.extents(), shape.strides());
        let offset = state.start * strd[state.axis];

        // Build the output (extents, strides) by dropping (squeeze) or shrinking
        // (range) the sliced axis — purely strided, no copy.
        let mut oe = [0usize; OUT];
        let mut os = [0usize; OUT];
        if state.squeeze {
            let mut j = 0;
            for d in 0..IN {
                if d == state.axis {
                    continue;
                }
                oe[j] = ext[d];
                os[j] = strd[d];
                j += 1;
            }
        } else {
            // OUT == IN here; copy axes, shrink the sliced one to `count`.
            for d in 0..IN {
                oe[d] = if d == state.axis { state.count } else { ext[d] };
                os[d] = strd[d];
            }
        }
        let out = ArrayView::from_parts(&data[offset..], Shape::strided(oe, os));
        (notified && !init, out)
    }
}

// ---------------------------------------------------------------------------
// AsView (whole-array reference -> strided view: the source/owned bridge)
// ---------------------------------------------------------------------------

/// Bridge a `RefPort<Array<T, N>>` (a whole-array reference — e.g. a source cell
/// or a `RefSource`) into the `ViewPort<ArrayValue<T, N>>` view currency by
/// lending a contiguous [`ArrayView`] of it **by value**, zero-copy. Like
/// [`SliceView`], the view is re-derived from the fresh input each generation,
/// so it implements [`Segment`] and forwards the input's notify.
pub struct AsView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> AsView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for AsView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for AsView<T, N> {
    type Inputs = RefPort<Array<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = ();

    fn init(self) {}

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, arr): (bool, &'a Array<T, N>),
        _state: &'b mut (),
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        (notified && !init, arr.view())
    }
}

// ---------------------------------------------------------------------------
// RefArrayView / DerefArrayView (the ViewPort <-> RefViewPort bridges)
// ---------------------------------------------------------------------------

/// Bridge a by-value `ViewPort<ArrayValue<T, N>>` into a by-reference
/// `RefViewPort<ArrayValue<T, N>>` — the adapter that lets independently-produced
/// view handles (e.g. separate feature columns) be collected into the
/// `&[RefViewPort]` slice the carry-join combines ([`Stack`](super::Stack) /
/// [`StackSync`](super::StackSync) / [`Concat`](super::Concat)) consume.
///
/// There is no value↔reference adapter inside `flowgraph::segment!` (it forwards
/// values positionally), and an `Operator::State: 'static` cannot store an
/// `ArrayView<'a>` — so the engine's only *intrinsic* producer of by-reference
/// view rows is [`Split`](super::Split) (via a per-generation [`Arena`]). This is
/// the rank-preserving, single-handle counterpart: each `compute` homes the input
/// view — a `Copy` fat pointer already pointing at the upstream's stable storage,
/// valid for the generation lifetime `'a` — in its own arena and lends back
/// `&'a ArrayView`. The arena holds only the view struct, so no array data is
/// copied. It forwards the input's notify, so the no-notify⟹output-unchanged
/// contract carries: when the input is silent the lent view still points at the
/// upstream's last (unchanged) value. [`DerefArrayView`] is the exact inverse.
pub struct RefArrayView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> RefArrayView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for RefArrayView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for RefArrayView<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = RefViewPort<ArrayValue<T, N>>;
    type State = Bump;

    fn init(self) -> Bump {
        Bump::new()
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, x): (bool, ArrayView<'a, T, N>),
        arena: &'b mut Bump,
        init: bool,
    ) -> (bool, &'a ArrayView<'a, T, N>) {
        arena.reset();
        let alloc: &'a Bump = arena;
        (notified && !init, &*alloc.alloc(x))
    }
}

/// Re-derive a by-reference `RefViewPort<ArrayValue<T, N>>` (e.g. a
/// [`Split`](super::Split) row, whose leaf is a `&ArrayView`) as a by-value
/// `ViewPort<ArrayValue<T, N>>` — the zero-copy adapter that lets a by-reference
/// view feed the by-value view operators ([`Gate`](super::Gate), [`SliceView`],
/// the arithmetic ops, …). The inner `ArrayView` is `Copy`, so this forwards it
/// by value; like [`SliceView`] / [`AsView`] it is a [`Segment`] re-derived from
/// the fresh input each generation, forwarding the input's notify. It is the
/// exact inverse of [`RefArrayView`].
pub struct DerefArrayView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> DerefArrayView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for DerefArrayView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for DerefArrayView<T, N> {
    type Inputs = RefViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = ();

    fn init(self) {}

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, v): (bool, &'a ArrayView<'a, T, N>),
        _state: &'b mut (),
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        (notified && !init, *v)
    }
}

// ---------------------------------------------------------------------------
// Own (strided view -> owned whole-array reference: the inverse of AsView)
// ---------------------------------------------------------------------------

/// Materialize a strided [`ArrayView`] back into an **owned** `Array<T, N>` homed
/// in state, lent as a whole-array `RefPort<Array<T, N>>` — the inverse of
/// [`AsView`], bridging the view currency back to the whole-array reference
/// currency that whole-array consumers (e.g. [`Resample`](super::Resample), or a
/// `PyClassOperator` under the `python` feature) take. Materializes its owned
/// output each notifying tick (via [`ArrayView::to_array`]) and re-presents the
/// last value un-notified, so the no-notify⟹output-unchanged contract holds.
pub struct Own<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Own<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for Own<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Operator for Own<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = RefPort<Array<T, N>>;
    type State = Option<Array<T, N>>;

    fn init(self) -> Self::State {
        None
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, &'a Array<T, N>) {
        *state = Some(x.to_array());
        (notified && !init, state.as_ref().unwrap())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, &'a Array<T, N>) {
        (false, state.as_ref().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Lag (value from N steps ago in a Series)
// ---------------------------------------------------------------------------

/// Emits the `Series<T, N>` element from `offset` steps ago (else `fill`), as
/// a rank-`N` view of its homed buffer.
#[derive(Clone)]
pub struct Lag<T: Scalar, const N: usize> {
    offset: usize,
    fill: T,
}

impl<T: Scalar, const N: usize> Lag<T, N> {
    pub fn new(offset: usize, fill: T) -> Self {
        Self { offset, fill }
    }
}

/// Runtime state for [`Lag`]: the configuration plus the output buffer (sized
/// and seeded with the fill value on the build call).
pub struct LagState<T: Scalar, const N: usize> {
    offset: usize,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Lag<T, N> {
    type Inputs = RefPort<Series<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = LagState<T, N>;

    fn init(self) -> Self::State {
        LagState {
            offset: self.offset,
            fill: self.fill,
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, &'a Series<T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::full(series.extents(), state.fill.clone());
            return (false, state.out.view());
        }
        let len = series.len();
        let dst = state.out.as_mut_slice();
        if len > state.offset {
            dst.clone_from_slice(series.at(len - 1 - state.offset));
        } else {
            dst.fill(state.fill.clone());
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Series<T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Element-wise (well, whole-array) closure `ArrayView -> Array`.
pub fn map<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F>(
    f: F,
) -> Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    Map::new(f)
}

/// [`map`] writing into a reused output buffer; the closure returns whether to
/// notify.
pub fn map_inplace<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F>(
    f: F,
    initial: Array<SO, NO>,
) -> MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    MapInplace::new(f, initial)
}

/// A closure over a whole input *tree* (the multi-port [`map`]); the input
/// interface `I` is inferred from the wiring.
pub fn apply<I, SO: Scalar, const NO: usize, F>(f: F) -> Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    Apply::new(f)
}

/// [`apply`] writing into a reused output buffer; the closure returns whether
/// to notify.
pub fn apply_inplace<I, SO: Scalar, const NO: usize, F>(
    f: F,
    initial: Array<SO, NO>,
) -> ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    ApplyInplace::new(f, initial)
}

/// Gather `indices` along `axis` into an owned output, optionally squeezing a
/// length-1 axis.
pub fn select<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
) -> Select<T, IN, OUT> {
    Select::new(indices, axis, squeeze)
}

/// [`select`] over the flattened input: `select(indices, 0, false)`.
pub fn select_flat<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
) -> Select<T, IN, OUT> {
    Select::flat(indices)
}

/// [`select`] along `axis`, keeping the axis: `select(indices, axis, false)`.
pub fn select_along_axis<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
) -> Select<T, IN, OUT> {
    Select::along_axis(indices, axis)
}

/// Zero-copy strided [`select`]: re-derives a view of the input rather than
/// copying (contiguous `indices` along `axis`).
pub fn slice_view<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
) -> SliceView<T, IN, OUT> {
    SliceView::new(indices, axis, squeeze)
}

/// Bridge a whole-array `RefPort<Array<T, N>>` into the `ViewPort` view
/// currency the operators speak (zero-copy).
pub fn as_view<T: Scalar, const N: usize>() -> AsView<T, N> {
    AsView::new()
}

/// Bridge a by-value view into the by-reference `RefViewPort` currency the
/// carry-join combines ([`Stack`](super::Stack) / [`Concat`](super::Concat))
/// consume.
pub fn ref_array_view<T: Scalar, const N: usize>() -> RefArrayView<T, N> {
    RefArrayView::new()
}

/// The inverse of [`ref_array_view`]: re-derive a by-reference view (e.g. a
/// [`Split`](super::Split) row) as the by-value currency the elementwise
/// operators consume.
pub fn deref_array_view<T: Scalar, const N: usize>() -> DerefArrayView<T, N> {
    DerefArrayView::new()
}

/// Materialize a by-value view back into an owned whole-array
/// `RefPort<Array<T, N>>` cell — the inverse of [`as_view`], e.g. before a
/// whole-array consumer such as a `PyClassOperator`.
pub fn own<T: Scalar, const N: usize>() -> Own<T, N> {
    Own::new()
}

/// The value from `offset` ticks ago in a recorded [`Series`], `fill` until it
/// exists — the primitive behind the self-recording [`lag`](super::lag).
/// (Named `_series` because `lag` is taken by its live-array counterpart.)
pub fn lag_series<T: Scalar, const N: usize>(offset: usize, fill: T) -> Lag<T, N> {
    Lag::new(offset, fill)
}

/// Push one [`ref_array_view`] bridge per handle — the vector form, for the
/// common `push(stack(axis), &ref_array_views(sc, &columns)[..])` shape. Takes
/// any builder that derefs to a [`flowgraph::typed::Builder`] (e.g. a
/// `Scenario`).
pub fn ref_array_views<T: Scalar, const N: usize>(
    builder: &mut flowgraph::typed::Builder,
    data: &[flowgraph::typed::Handle<ViewPort<ArrayValue<T, N>>>],
) -> Vec<flowgraph::typed::Handle<RefViewPort<ArrayValue<T, N>>>> {
    data.iter()
        .map(|&h| builder.push(ref_array_view(), h))
        .collect()
}
