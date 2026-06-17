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

use flowgraph::typed::{Interface, Operator, Segment, RefPort, ViewPort};

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

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator for Map<SI, NI, SO, NO, F>
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
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    f: F,
    initial: Array<SO, NO>,
    _phantom: PhantomData<fn() -> I>,
}

impl<I, SO: Scalar, const NO: usize, F> ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
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
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
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

/// [`Select`] retained for source compatibility — the view-input variant is now
/// the only `Select` (its input is always a `ViewPort`).
pub type SelectView<T, const IN: usize, const OUT: usize> = Select<T, IN, OUT>;

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
/// [`StackView`](super::StackView) reader sees the last notified value for an
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
        let (data, base) = v.buffer();
        let shape = v.shape();
        let (ext, strd) = (shape.extents(), shape.strides());
        let offset = base + state.start * strd[state.axis];

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
        let out = ArrayView::from_parts(data, offset, Shape::strided(oe, os));
        (notified && !init, out)
    }
}

// ---------------------------------------------------------------------------
// Lag (value from N steps ago in a Series)
// ---------------------------------------------------------------------------

/// Emits the `Series<T>` element from `offset` steps ago (else `fill`), as a
/// rank-`N` view of its homed buffer.
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
    type Inputs = RefPort<Series<T>>;
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
        (_, series): (bool, &'a Series<T>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            let ext = super::gating::series_extents::<T, N>(series);
            state.out = Array::full(ext, state.fill.clone());
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
        _: (bool, &'a Series<T>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}
