//! Function / selection / lag operators — `Map`/`MapInplace`,
//! `Apply`/`ApplyInplace`, `Select`, `Lag`, implemented directly on
//! [`flowgraph::typed::Operator`]. The closure operators receive the
//! values-only refs tree (notify flags stripped via [`StripNotify`]), which
//! keeps the legacy closure signatures, e.g. `Fn((&Array<f64>, &Array<f64>))`
//! for a two-port input.

use std::marker::PhantomData;

use flowgraph::typed::{Interface, Operator, Port};

use super::op::StripNotify;
use crate::{Array, Scalar, Series};

// ---------------------------------------------------------------------------
// Map / MapInplace (single input)
// ---------------------------------------------------------------------------

/// Allocating map: applies `Fn(&S) -> T` each tick.
#[derive(Clone)]
pub struct Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<(S, T)>,
}

impl<S, T, F> Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Send + Sync + 'static,
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
pub struct MapState<T, F> {
    f: F,
    out: Option<T>,
}

impl<S, T, F> Operator for Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Send + Sync + 'static,
{
    type Inputs = Port<S>;
    type Outputs = Port<T>;
    type State = MapState<T, F>;

    fn init(self) -> MapState<T, F> {
        MapState {
            f: self.f,
            out: None,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a S),
        state: &'b mut MapState<T, F>,
        init: bool,
    ) -> (bool, &'a T) {
        // The build call (`init`) runs the closure once on the build-time input
        // to seed the output — replicating the legacy `init` — but does not
        // notify.
        state.out = Some((state.f)(x));
        (!init, state.out.as_ref().unwrap())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(_: (bool, &'a S), state: &'b MapState<T, F>) -> (bool, &'a T) {
        (false, state.out.as_ref().unwrap())
    }
}

/// In-place map: `Fn(&S, &mut T) -> bool`; the bool controls propagation.
#[derive(Clone)]
pub struct MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Send + Sync + 'static,
{
    f: F,
    initial: T,
    _phantom: PhantomData<S>,
}

impl<S, T, F> MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Send + Sync + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`MapInplace`]: the function, the initial value, and the
/// output buffer.
pub struct MapInplaceState<T, F> {
    f: F,
    initial: T,
    out: T,
}

impl<S, T, F> Operator for MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Send + Sync + 'static,
{
    type Inputs = Port<S>;
    type Outputs = Port<T>;
    type State = MapInplaceState<T, F>;

    fn init(self) -> MapInplaceState<T, F> {
        MapInplaceState {
            f: self.f,
            out: self.initial.clone(),
            initial: self.initial,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a S),
        state: &'b mut MapInplaceState<T, F>,
        init: bool,
    ) -> (bool, &'a T) {
        if init {
            state.out = state.initial.clone();
            (state.f)(x, &mut state.out);
            return (false, &state.out);
        }
        let notify = (state.f)(x, &mut state.out);
        (notify, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a S),
        state: &'b MapInplaceState<T, F>,
    ) -> (bool, &'a T) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Apply / ApplyInplace (tuple inputs)
// ---------------------------------------------------------------------------

/// Allocating multi-input map: `Fn(values) -> T`, where `values` is the
/// values-only refs tree of the input interface `I` (e.g. `(&A, &B)` for
/// `(Port<A>, Port<B>)` — see [`StripNotify`]).
pub struct Apply<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>) -> T + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<fn() -> (I, T)>,
}

impl<I, T, F> Clone for Apply<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>) -> T + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> Apply<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>) -> T + Send + Sync + 'static,
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
pub struct ApplyState<T, F> {
    f: F,
    out: Option<T>,
}

impl<I, T, F> Operator for Apply<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>) -> T + Send + Sync + 'static,
{
    type Inputs = I;
    type Outputs = Port<T>;
    type State = ApplyState<T, F>;

    fn init(self) -> ApplyState<T, F> {
        ApplyState {
            f: self.f,
            out: None,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Refs<'a>,
        state: &'b mut ApplyState<T, F>,
        init: bool,
    ) -> (bool, &'a T) {
        // The build call (`init`) runs the closure once on the build-time
        // inputs to seed the output — replicating the legacy `init` — but does
        // not notify.
        state.out = Some((state.f)(<I as StripNotify>::values(inputs)));
        (!init, state.out.as_ref().unwrap())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Refs<'a>,
        state: &'b ApplyState<T, F>,
    ) -> (bool, &'a T) {
        (false, state.out.as_ref().unwrap())
    }
}

/// In-place multi-input map: `Fn(values, &mut T) -> bool`, where `values` is
/// the values-only refs tree of the input interface `I` (see [`StripNotify`]);
/// the bool controls propagation.
pub struct ApplyInplace<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>, &mut T) -> bool + Send + Sync + 'static,
{
    f: F,
    initial: T,
    _phantom: PhantomData<fn() -> I>,
}

impl<I, T, F> Clone for ApplyInplace<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            initial: self.initial.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> ApplyInplace<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>, &mut T) -> bool + Send + Sync + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`ApplyInplace`]: the function, the initial value, and the
/// output buffer.
pub struct ApplyInplaceState<T, F> {
    f: F,
    initial: T,
    out: T,
}

impl<I, T, F> Operator for ApplyInplace<I, T, F>
where
    I: StripNotify + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as StripNotify>::Values<'a>, &mut T) -> bool + Send + Sync + 'static,
{
    type Inputs = I;
    type Outputs = Port<T>;
    type State = ApplyInplaceState<T, F>;

    fn init(self) -> ApplyInplaceState<T, F> {
        ApplyInplaceState {
            f: self.f,
            out: self.initial.clone(),
            initial: self.initial,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Refs<'a>,
        state: &'b mut ApplyInplaceState<T, F>,
        init: bool,
    ) -> (bool, &'a T) {
        if init {
            state.out = state.initial.clone();
            (state.f)(<I as StripNotify>::values(inputs), &mut state.out);
            return (false, &state.out);
        }
        let notify = (state.f)(<I as StripNotify>::values(inputs), &mut state.out);
        (notify, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Refs<'a>,
        state: &'b ApplyInplaceState<T, F>,
    ) -> (bool, &'a T) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Select (index selection along an axis)
// ---------------------------------------------------------------------------

/// Select elements along an axis (precomputed flat index map).
#[derive(Clone)]
pub struct Select<T: Scalar> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Select<T> {
    pub fn new(indices: Vec<usize>, axis: usize, squeeze: bool) -> Self {
        assert!(
            !squeeze || indices.len() == 1,
            "squeeze requires exactly one index, got {}",
            indices.len(),
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

/// Runtime state for [`Select`]: the configuration (the flat index map can only
/// be computed on the build call, once the input shape is known), the index
/// map, and the output buffer.
pub struct SelectState<T: Scalar> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    index_map: Vec<usize>,
    out: Array<T>,
}

impl<T: Scalar> Operator for Select<T> {
    type Inputs = Port<Array<T>>;
    type Outputs = Port<Array<T>>;
    type State = SelectState<T>;

    fn init(self) -> SelectState<T> {
        SelectState {
            indices: self.indices,
            axis: self.axis,
            squeeze: self.squeeze,
            index_map: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a Array<T>),
        state: &'b mut SelectState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let input_shape = x.shape();
            state.index_map = compute_select_map(input_shape, &state.indices, state.axis);
            let mut output_shape = input_shape.to_vec();
            if output_shape.is_empty() {
                output_shape = vec![state.indices.len()];
            } else {
                output_shape[state.axis] = state.indices.len();
            }
            if state.squeeze && state.indices.len() == 1 && output_shape.len() > state.axis {
                output_shape.remove(state.axis);
            }
            // Seed the initial output with the actual selection of the
            // build-time input — NOT zeros. A zeros build value is a fabricated
            // finite observation: it leaks through carry-style consumers
            // (`Stack`) into e.g. cross-sectional ranks for stocks with no data
            // before the window starts, whereas the faithful selection of a
            // NaN-initialised panel correctly reads "no data yet". (Also keeps
            // `Select` and `Split` agreeing on build values.)
            let src = x.as_slice();
            state.out = Array::from_vec(
                &output_shape,
                state.index_map.iter().map(|&s| src[s].clone()).collect(),
            );
            return (false, &state.out);
        }
        let src = x.as_slice();
        let dst = state.out.as_mut_slice();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b SelectState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}

fn compute_select_map(input_shape: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    if input_shape.is_empty() {
        return indices.to_vec();
    }
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let axis_size = input_shape[axis];
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

// ---------------------------------------------------------------------------
// Lag (value from N steps ago in a Series)
// ---------------------------------------------------------------------------

/// Emits the `Series<T>` element from `offset` steps ago (else `fill`).
#[derive(Clone)]
pub struct Lag<T: Scalar> {
    offset: usize,
    fill: T,
}

impl<T: Scalar> Lag<T> {
    pub fn new(offset: usize, fill: T) -> Self {
        Self { offset, fill }
    }
}

/// Runtime state for [`Lag`]: the configuration plus the output buffer (sized
/// and seeded with the fill value on the build call).
pub struct LagState<T: Scalar> {
    offset: usize,
    fill: T,
    out: Array<T>,
}

impl<T: Scalar> Operator for Lag<T> {
    type Inputs = Port<Series<T>>;
    type Outputs = Port<Array<T>>;
    type State = LagState<T>;

    fn init(self) -> LagState<T> {
        LagState {
            offset: self.offset,
            fill: self.fill,
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, &'a Series<T>),
        state: &'b mut LagState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let shape = series.shape();
            let stride: usize = shape.iter().product();
            state.out = Array::from_vec(shape, vec![state.fill.clone(); stride]);
            return (false, &state.out);
        }
        let len = series.len();
        let dst = state.out.as_mut_slice();
        if len > state.offset {
            dst.clone_from_slice(series.at(len - 1 - state.offset));
        } else {
            dst.fill(state.fill.clone());
        }
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Series<T>),
        state: &'b LagState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}
