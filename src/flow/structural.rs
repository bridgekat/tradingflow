//! Structural operators — port of `Id`, `Where`, `Cast`, plus the
//! [`Resample`] clock-gated identity, implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`].

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use flowgraph::typed::{Interface, Operator, RefPort, Segment};

use super::ops::Clocked;
use crate::{Array, Scalar};

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

/// Runtime state for [`Where`]: the predicate and fill plus the output buffer.
pub struct WhereState<T: Scalar, F> {
    condition: F,
    fill: T,
    out: Array<T>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone + Send + Sync + 'static> Operator for Where<T, F> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = WhereState<T, F>;

    fn init(self) -> WhereState<T, F> {
        WhereState {
            condition: self.condition,
            fill: self.fill,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut WhereState<T, F>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.out = a.clone();
            return (false, &state.out);
        }
        let src = a.as_slice();
        let out = state.out.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if (state.condition)(src[i].clone()) {
                src[i].clone()
            } else {
                state.fill.clone()
            };
        }
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b WhereState<T, F>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = RefPort<Array<S>>;
    type Outputs = RefPort<Array<T>>;
    type State = Array<T>;

    fn init(self) -> Array<T> {
        Array::zeros(&[0])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<S>),
        out: &'b mut Array<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            // The legacy build call cast the build-time values (not zeros).
            let data: Vec<T> = a.as_slice().iter().map(|&v| v.as_()).collect();
            *out = Array::from_vec(a.shape(), data);
            return (false, &*out);
        }
        let src = a.as_slice();
        let dst = out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        (true, &*out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<S>),
        out: &'b Array<T>,
    ) -> (bool, &'a Array<T>) {
        (false, out)
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
