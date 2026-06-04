//! Structural operators — port of `Id`, `Where`, `Cast`.

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use flowgraph::typed::{Port, Ports};

use super::op::Operator;
use super::ops::Clocked;
use crate::{Array, Instant, Scalar};

/// Identity passthrough: clones input to output unchanged.
#[derive(Clone)]
pub struct Id<T: Clone + Send + Sync + 'static> {
    _phantom: PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> Id<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> Default for Id<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync + 'static> Operator for Id<T> {
    type State = ();
    type Inputs = Port<T>;
    type Output = T;

    fn init(&self, inputs: &T, _ts: Instant) -> ((), T) {
        ((), inputs.clone())
    }

    #[inline(always)]
    fn compute(_state: &mut (), inputs: &T, output: &mut T, _ts: Instant, _produced: bool) -> bool {
        output.clone_from(inputs);
        true
    }
}

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace with `fill`. The closure carries the trait's `Sync` bound.
#[derive(Clone)]
pub struct Where<T: Scalar, F: Fn(T) -> bool + Clone> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone> Where<T, F> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, F: Fn(T) -> bool + Clone + Send + Sync + 'static> Operator for Where<T, F> {
    type State = Self;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (Self, Array<T>) {
        (self.clone(), inputs.clone())
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if (state.condition)(a[i].clone()) {
                a[i].clone()
            } else {
                state.fill.clone()
            };
        }
        true
    }
}

/// Element-wise type conversion `Array<S> → Array<T>` via `AsPrimitive`.
#[derive(Clone)]
pub struct Cast<S: Scalar, T: Scalar> {
    _phantom: PhantomData<(S, T)>,
}

impl<S: Scalar, T: Scalar> Cast<S, T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Scalar, T: Scalar> Default for Cast<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T> Operator for Cast<S, T>
where
    S: Scalar + Copy + AsPrimitive<T>,
    T: Scalar + Copy + 'static,
{
    type State = ();
    type Inputs = Port<Array<S>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<S>, _ts: Instant) -> ((), Array<T>) {
        let src = inputs.as_slice();
        let data: Vec<T> = src.iter().map(|&v| v.as_()).collect();
        ((), Array::from_vec(inputs.shape(), data))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: &Array<S>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        true
    }
}

/// Re-emit a data input's latest value on every clock tick:
/// `Clocked<Id<O>, C>`. The clock (`C`) and data (`O`) node types are
/// independent — only the clock's notify bit is consulted.
pub struct Resample<O, C>(Clocked<Id<O>, C>)
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static;

impl<O, C> Clone for Resample<O, C>
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O, C> Resample<O, C>
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self(Clocked::new(Id::new()))
    }
}

impl<O, C> Default for Resample<O, C>
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<O, C> Operator for Resample<O, C>
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    type State = <Clocked<Id<O>, C> as Operator>::State;
    type Inputs = <Clocked<Id<O>, C> as Operator>::Inputs;
    type Output = <Clocked<Id<O>, C> as Operator>::Output;

    fn init(
        &self,
        inputs: <Self::Inputs as Ports>::Refs<'_>,
        ts: Instant,
    ) -> (Self::State, Self::Output) {
        self.0.init(inputs, ts)
    }

    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as Ports>::Refs<'_>,
        output: &mut Self::Output,
        ts: Instant,
        produced: <Self::Inputs as Ports>::Notify<'_>,
    ) -> bool {
        <Clocked<Id<O>, C> as Operator>::compute(state, inputs, output, ts, produced)
    }
}
