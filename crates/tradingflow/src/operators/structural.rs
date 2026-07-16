//! Structural operators — port of `Id`, `Where`, `Cast`, plus the
//! [`Resample`] clock-gated identity and its view-currency counterparts
//! [`ResampleView`] / [`ResampleClocked`], implemented directly on
//! [`Operator`](crate::graph::Operator) / [`Segment`].

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use crate::graph::{Interface, Operator, RefPort, Segment};

use super::gating::Clocked;
use super::op::ArrayPort;
use crate::{Array, ArrayView, Instant, Scalar};

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
    type Context = Instant;
    type State = Option<T>;

    fn init(self) -> Option<T> {
        None
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a T),
        _: &Instant,
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
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a T),
        _: &Instant,
        state: &'b Option<T>,
    ) -> (bool, &'a T) {
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
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
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
        _: &Instant,
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
        _: &Instant,
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
    type Inputs = ArrayPort<S, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, S, N>),
        _: &Instant,
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
        _: &Instant,
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
    type Context = <Clocked<Id<O>, C> as Segment>::Context; // = Instant
    type State = <Clocked<Id<O>, C> as Segment>::State;

    fn init(self) -> Self::State {
        <Clocked<Id<O>, C> as Segment>::init(self.0)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        init: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        <Clocked<Id<O>, C> as Segment>::compute(inputs, context, state, init)
    }
}

/// State shared by the view-currency resamplers: the last data view
/// materialized into an owned buffer, so it survives between clock ticks while
/// the upstream view's storage may change.
pub struct ResampleViewState<T: Scalar, const N: usize> {
    out: Option<Array<T, N>>,
}

/// Clock-gated **view** passthrough whose clock is another data view (only the
/// clock's notify bit is consulted): re-emits the rank-`N` data view on every
/// tick of the rank-1 clock view, holding the last value in between. The
/// view-currency counterpart of [`Resample`] — e.g. resample a feature panel
/// onto the daily-close pulse. Stays in the [`ArrayView`] currency end-to-end,
/// so it needs no [`Own`](super::Own) bridge on either side.
#[derive(Clone)]
pub struct ResampleView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// A `Segment`, not an `Operator`: the gate ignores the data input's notify bit
// (only the clock's fires it), which the `Operator` any-notify gate cannot
// express.
impl<T: Scalar, const N: usize> Segment for ResampleView<T, N> {
    type Inputs = (ArrayPort<T, 1>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(self) -> Self::State {
        ResampleViewState { out: None }
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, ArrayView<'a, T, 1>), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init || clock_fired {
            state.out = Some(x.to_array());
            return (!init, state.out.as_ref().unwrap().view());
        }
        (false, state.out.as_ref().unwrap().view())
    }
}

/// Clock-gated **view** passthrough whose clock is a unit (`RefPort<()>`) clock
/// source (e.g. a rebalance [`pulse`](crate::sources::pulse())): re-emits the
/// rank-`N` data view on every clock tick. The unit-clock counterpart of
/// [`ResampleView`].
#[derive(Clone)]
pub struct ResampleClocked<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleClocked<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleClocked<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for ResampleClocked<T, N> {
    type Inputs = (RefPort<()>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(self) -> Self::State {
        ResampleViewState { out: None }
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, &'a ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init || clock_fired {
            state.out = Some(x.to_array());
            return (!init, state.out.as_ref().unwrap().view());
        }
        (false, state.out.as_ref().unwrap().view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// The identity operator: forwards its input unchanged.
pub fn id<T: Clone + Send + Sync + 'static>() -> Id<T> {
    Id::new()
}

/// Element-wise scalar cast `S -> T`.
pub fn cast<S: Scalar, T: Scalar, const N: usize>() -> Cast<S, T, N> {
    Cast::new()
}

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace it with `fill`. (Named `keep_where` because `where` is a keyword.)
pub fn keep_where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize>(
    condition: F,
    fill: T,
) -> Where<T, F, N> {
    Where::new(condition, fill)
}

/// Re-emit a data input's latest value on every clock tick (the whole-value
/// `RefPort` currency).
pub fn resample<O: Clone + Send + Sync + 'static, C: Send + Sync + 'static>() -> Resample<O, C> {
    Resample::new()
}

/// Re-emit an array view on every tick of a leading *view* pulse.
pub fn resample_view<T: Scalar, const N: usize>() -> ResampleView<T, N> {
    ResampleView::new()
}

/// Re-emit an array view on every tick of a leading *clock* pulse.
pub fn resample_clocked<T: Scalar, const N: usize>() -> ResampleClocked<T, N> {
    ResampleClocked::new()
}
