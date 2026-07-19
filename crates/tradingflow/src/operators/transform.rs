//! Function / selection / lag operators over the strided [`ArrayView`]
//! currency — `Map`/`MapInplace`, `Apply`/`ApplyInplace` (closure compute),
//! `Select` (materializing selection), and `Lag` (a `Series` element from N
//! steps ago).
//!
//! The closure operators receive the values-only views tree (notify flags
//! stripped via [`StripNotify`]) and return an owned [`Array`], which is homed
//! in `State` and lent as a `ViewPort` view — so a closure reads strided inputs
//! and the result composes as the same view currency.

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar, SeriesView};
use crate::graph::typed::{Interface, Operator};
use crate::ports::{ArrayPort, SeriesPort, StripNotify};

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

/// Runtime state for [`Map`]: the function plus the output cell, seeded by
/// running the closure once on the build-time input in `init`.
pub struct MapState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = ArrayPort<SI, NI>;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = MapState<SO, NO, F>;

    fn init(self, (_, x): (bool, ArrayView<'_, SI, NI>)) -> Self::State {
        // Run the closure once on the build-time input to seed the output; the
        // initial render then goes through `passthrough` without notifying.
        let out = (self.f)(x);
        MapState { f: self.f, out }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        state.out = (state.f)(x);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
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

/// Runtime state for [`MapInplace`]: the function and the output buffer
/// (seeded from the initial value in `init`).
pub struct MapInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<SI, NI>;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = MapInplaceState<SO, NO, F>;

    fn init(self, (_, x): (bool, ArrayView<'_, SI, NI>)) -> Self::State {
        let mut out = self.initial;
        (self.f)(x, &mut out);
        MapInplaceState { f: self.f, out }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        let notify = (state.f)(x, &mut state.out);
        (notify, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
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

/// Runtime state for [`Apply`]: the function plus the output cell, seeded by
/// running the closure once on the build-time inputs in `init`.
pub struct ApplyState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<I, SO: Scalar, const NO: usize, F> Operator for Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = I;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = ApplyState<SO, NO, F>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> Self::State {
        // Run the closure once on the build-time inputs to seed the output; the
        // initial render then goes through `passthrough` without notifying.
        let out = (self.f)(<I as StripNotify>::plain(inputs));
        ApplyState { f: self.f, out }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        state.out = (state.f)(<I as StripNotify>::plain(inputs));
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
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

/// Runtime state for [`ApplyInplace`]: the function and the output buffer
/// (seeded from the initial value in `init`).
pub struct ApplyInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
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
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = ApplyInplaceState<SO, NO, F>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> Self::State {
        let mut out = self.initial;
        (self.f)(<I as StripNotify>::plain(inputs), &mut out);
        ApplyInplaceState { f: self.f, out }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        let notify = (state.f)(<I as StripNotify>::plain(inputs), &mut state.out);
        (notify, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
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
/// on.
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
}

/// Runtime state for [`Select`]: the resolved flat index map (computed in
/// `init` once the input shape is known) and the output buffer.
pub struct SelectState<T: Scalar, const OUT: usize> {
    index_map: Vec<usize>,
    out: Array<T, OUT>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Select<T, IN, OUT> {
    type Inputs = ArrayPort<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = SelectState<T, OUT>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, IN>)) -> Self::State {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let input_extents = x.layout().extents();
        let index_map = compute_select_map(&input_extents, &self.indices, self.axis);
        let out_extents =
            select_out_extents::<OUT>(&input_extents, self.indices.len(), self.axis, self.squeeze);
        // Seed the initial output with the actual selection of the build-time
        // input (NOT zeros — a fabricated finite observation leaks through
        // carry-style consumers; the faithful selection of a NaN-initialised
        // panel correctly reads "no data yet").
        let out = Array::from_parts(
            out_extents,
            index_map.iter().map(|&s| src[s].clone()).collect(),
        );
        SelectState { index_map, out }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
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
// Lag (value from N steps ago in a Series)
// ---------------------------------------------------------------------------

/// Emits the recorded-history element from `offset` steps ago (else `fill`),
/// as a rank-`N` view of its homed buffer. Consumes a [`SeriesPort`] window;
/// the look-back is relative to the window's newest row, so a
/// retention-bounded record works as long as the bound covers `offset + 1`
/// rows.
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
/// and seeded with the fill value in `init`).
pub struct LagState<T: Scalar, const N: usize> {
    offset: usize,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Lag<T, N> {
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = LagState<T, N>;

    fn init(self, (_, series): (bool, SeriesView<'_, T, N>)) -> Self::State {
        LagState {
            offset: self.offset,
            out: Array::full(series.layout().extents(), self.fill.clone()),
            fill: self.fill,
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let len = series.len();
        let dst = state.out.data_mut();
        if len > state.offset {
            dst.clone_from_slice(series.at(len - 1 - state.offset).unwrap().1.data());
        } else {
            dst.fill(state.fill.clone());
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, T, N>),
        state: &'b mut Self::State,
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

/// [`select`] at a single index along `axis`, squeezing that axis.
pub fn select_at<T: Scalar, const IN: usize, const OUT: usize>(
    index: usize,
    axis: usize,
) -> Select<T, IN, OUT> {
    Select::new(vec![index], axis, true)
}

/// [`select`] along `axis`, keeping the axis: `select(indices, axis, false)`.
pub fn select_many<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
) -> Select<T, IN, OUT> {
    Select::new(indices, axis, false)
}

/// The value from `offset` ticks ago in a recorded [`Series`](tradingflow_data::Series), `fill` until it
/// exists — the primitive behind the self-recording [`lag`](super::formula::lag).
/// (Named `_series` because `lag` is taken by its live-array counterpart.)
pub fn lag_series<T: Scalar, const N: usize>(offset: usize, fill: T) -> Lag<T, N> {
    Lag::new(offset, fill)
}
