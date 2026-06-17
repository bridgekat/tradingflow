//! Structural operators — port of `Id`, `Where`, `Cast`, plus the
//! [`Resample`] clock-gated identity, implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`].

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use flowgraph::typed::{Interface, Operator, RefPort, Segment, ViewPort};

use super::gating::Clocked;
use super::op::ArrayValue;
use crate::{Array, ArrayView, Scalar};

/// Identity passthrough: clones input to output unchanged. Generic over the
/// payload `T` (an owned value carried by `RefPort<T>` — an `Array<T, N>`, a
/// `Series<T>`, a scalar, …), so it is rank- and currency-agnostic.
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

// State is `Option<T>` because `init(self)` runs before any input value is
// seen and `T` carries no `Default` bound: the build (`init == true`) call
// fills the `Some` from the build-time input value, so every later call may
// unwrap it.
impl<T: Clone + Send + Sync + 'static> Operator for Id<T> {
    type Inputs = RefPort<T>;
    type Outputs = RefPort<T>;
    type State = Option<T>;

    fn init(self) -> Option<T> {
        None
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a T),
        state: &'b mut Option<T>,
        init: bool,
    ) -> (bool, &'a T) {
        if init {
            *state = Some(x.clone());
            return (false, state.as_ref().unwrap());
        }
        state.as_mut().unwrap().clone_from(x);
        (true, state.as_ref().unwrap())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(_: (bool, &'a T), state: &'b Option<T>) -> (bool, &'a T) {
        (false, state.as_ref().unwrap())
    }
}

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace with `fill`.
#[derive(Clone)]
pub struct Where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> Where<T, F, N> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Where`]: the predicate and fill plus the output buffer.
pub struct WhereState<T: Scalar, F, const N: usize> {
    condition: F,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone + Send + Sync + 'static, const N: usize> Operator
    for Where<T, F, N>
{
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = WhereState<T, F, N>;

    fn init(self) -> Self::State {
        WhereState {
            condition: self.condition,
            fill: self.fill,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let out = state.out.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if (state.condition)(src[i].clone()) {
                src[i].clone()
            } else {
                state.fill.clone()
            };
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Element-wise type conversion `ArrayView<S, N> → Array<T, N>` via
/// `AsPrimitive`.
#[derive(Clone)]
pub struct Cast<S: Scalar, T: Scalar, const N: usize> {
    _phantom: PhantomData<(S, T)>,
}

impl<S: Scalar, T: Scalar, const N: usize> Cast<S, T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Scalar, T: Scalar, const N: usize> Default for Cast<S, T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T, const N: usize> Operator for Cast<S, T, N>
where
    S: Scalar + Copy + AsPrimitive<T>,
    T: Scalar + Copy + 'static,
{
    type Inputs = ViewPort<ArrayValue<S, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, S, N>),
        out: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            *out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[S] = &xs;
        let dst = out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        (!init, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, S, N>),
        out: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// Re-emit a data input's latest value on every clock tick:
/// `Clocked<Id<O>, C>`. The clock (`C`) and data (`O`) node types are
/// independent — only the clock's notify bit is consulted. Like
/// [`Clocked`], this implements [`Segment`] directly (its gate ignores the
/// data input's notify bit) and simply delegates to the wrapped segment.
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

impl<O, C> Segment for Resample<O, C>
where
    O: Clone + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    type Inputs = <Clocked<Id<O>, C> as Segment>::Inputs; // = (RefPort<C>, RefPort<O>)
    type Outputs = <Clocked<Id<O>, C> as Segment>::Outputs; // = RefPort<O>
    type State = <Clocked<Id<O>, C> as Segment>::State;

    fn init(self) -> Self::State {
        <Clocked<Id<O>, C> as Segment>::init(self.0)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        init: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        <Clocked<Id<O>, C> as Segment>::compute(inputs, state, init)
    }
}
