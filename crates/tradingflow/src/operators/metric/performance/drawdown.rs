use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SignalPort};

/// Operator signature for [`drawdown`].
pub struct Drawdown<T: Scalar + Float> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float> Drawdown<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Drawdown<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Drawdown`].
pub struct DrawdownState<T: Scalar + Float> {
    max: T,
    out: Array<T, 0>,
}

impl<T: Scalar + Float> Segment for Drawdown<T> {
    type Inputs = (SignalPort<0>, ArrayPort<T, 0>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = DrawdownState<T>;

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, T, 0>)) -> Self::State {
        DrawdownState {
            max: T::nan(),
            out: Array::scalar(T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, 0> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (signal, value): (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, 0> {
        if !*signal {
            return state.out.view();
        }
        assert!(value.is_finite(), "drawdown: input value must be finite");
        assert!(*value > T::zero(), "drawdown: input value must be positive");
        if state.max.is_nan() {
            state.max = *value;
            *state.out = T::zero();
        } else {
            state.max = state.max.max(*value);
            *state.out = (*value - state.max) / state.max;
        }
        state.out.view()
    }
}

/// Percentage drawdown from the running maximum of a net-asset-value scalar,
/// where each period is specified by a signal.
pub fn drawdown<T: Scalar + Float>()
-> impl Segment<Inputs = (SignalPort<0>, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    Drawdown::new()
}
