//! Function / selection / lag operators — port of `Map`/`MapInplace`,
//! `Apply`/`ApplyInplace`, `Select`, `Lag`. Closure operators carry the
//! trait's `Sync` bound on their function.

use std::marker::PhantomData;

use flowgraph::typed::{Port, Ports};

use super::op::Operator;
use crate::{Array, Instant, Scalar, Series};

// ---------------------------------------------------------------------------
// Map / MapInplace (single input)
// ---------------------------------------------------------------------------

/// Allocating map: applies `Fn(&S) -> T` each tick.
#[derive(Clone)]
pub struct Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Clone + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<(S, T)>,
}

impl<S, T, F> Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Clone + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<S, T, F> Operator for Map<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S) -> T + Clone + Send + Sync + 'static,
{
    type State = Self;
    type Inputs = Port<S>;
    type Output = T;

    fn init(&self, inputs: &S, _ts: Instant) -> (Self, T) {
        let output = (self.f)(inputs);
        (self.clone(), output)
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: &S, output: &mut T, _ts: Instant, _produced: bool) -> bool {
        *output = (state.f)(inputs);
        true
    }
}

/// In-place map: `Fn(&S, &mut T) -> bool`; the bool controls propagation.
#[derive(Clone)]
pub struct MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    f: F,
    initial: T,
    _phantom: PhantomData<S>,
}

impl<S, T, F> MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

impl<S, T, F> Operator for MapInplace<S, T, F>
where
    S: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    F: Fn(&S, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    type State = Self;
    type Inputs = Port<S>;
    type Output = T;

    fn init(&self, inputs: &S, _ts: Instant) -> (Self, T) {
        let mut output = self.initial.clone();
        (self.f)(inputs, &mut output);
        (self.clone(), output)
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: &S, output: &mut T, _ts: Instant, _produced: bool) -> bool {
        (state.f)(inputs, output)
    }
}

// ---------------------------------------------------------------------------
// Apply / ApplyInplace (tuple inputs)
// ---------------------------------------------------------------------------

/// Allocating multi-input map: `Fn(Inputs::Refs) -> T`.
pub struct Apply<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>) -> T + Clone + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<fn() -> (I, T)>,
}

impl<I, T, F> Clone for Apply<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>) -> T + Clone + Send + Sync + 'static,
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
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>) -> T + Clone + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> Operator for Apply<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>) -> T + Clone + Send + Sync + 'static,
{
    type State = Self;
    type Inputs = I;
    type Output = T;

    fn init(&self, inputs: <I as Ports>::Refs<'_>, _ts: Instant) -> (Self, T) {
        let output = (self.f)(inputs);
        (self.clone(), output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: <I as Ports>::Refs<'_>,
        output: &mut T,
        _ts: Instant,
        _produced: <I as Ports>::Notify<'_>,
    ) -> bool {
        *output = (state.f)(inputs);
        true
    }
}

/// In-place multi-input map: `Fn(Inputs::Refs, &mut T) -> bool`.
pub struct ApplyInplace<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    f: F,
    initial: T,
    _phantom: PhantomData<fn() -> I>,
}

impl<I, T, F> Clone for ApplyInplace<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>, &mut T) -> bool + Clone + Send + Sync + 'static,
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
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> Operator for ApplyInplace<I, T, F>
where
    I: Ports + 'static,
    T: Clone + Send + Sync + 'static,
    F: for<'a> Fn(<I as Ports>::Refs<'a>, &mut T) -> bool + Clone + Send + Sync + 'static,
{
    type State = Self;
    type Inputs = I;
    type Output = T;

    fn init(&self, inputs: <I as Ports>::Refs<'_>, _ts: Instant) -> (Self, T) {
        let mut output = self.initial.clone();
        (self.f)(inputs, &mut output);
        (self.clone(), output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: <I as Ports>::Refs<'_>,
        output: &mut T,
        _ts: Instant,
        _produced: <I as Ports>::Notify<'_>,
    ) -> bool {
        (state.f)(inputs, output)
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

/// Runtime state for [`Select`].
pub struct SelectState {
    index_map: Vec<usize>,
}

impl<T: Scalar> Operator for Select<T> {
    type State = SelectState;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (SelectState, Array<T>) {
        let input_shape = inputs.shape();
        let index_map = compute_select_map(input_shape, &self.indices, self.axis);
        let mut output_shape = input_shape.to_vec();
        if output_shape.is_empty() {
            output_shape = vec![self.indices.len()];
        } else {
            output_shape[self.axis] = self.indices.len();
        }
        if self.squeeze && self.indices.len() == 1 && output_shape.len() > self.axis {
            output_shape.remove(self.axis);
        }
        (SelectState { index_map }, Array::zeros(&output_shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut SelectState,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        true
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

/// Runtime state for [`Lag`].
pub struct LagState<T: Scalar> {
    offset: usize,
    fill: T,
}

impl<T: Scalar> Operator for Lag<T> {
    type State = LagState<T>;
    type Inputs = Port<Series<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Series<T>, _ts: Instant) -> (LagState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let fill = Array::from_vec(shape, vec![self.fill.clone(); stride]);
        let state = LagState {
            offset: self.offset,
            fill: self.fill.clone(),
        };
        (state, fill)
    }

    fn compute(
        state: &mut LagState<T>,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let series = inputs;
        let len = series.len();
        let dst = output.as_mut_slice();
        if len > state.offset {
            dst.clone_from_slice(series.at(len - 1 - state.offset));
        } else {
            dst.fill(state.fill.clone());
        }
        true
    }
}
