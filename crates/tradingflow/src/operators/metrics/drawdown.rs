//! Drawdown from the running peak (single input, no clock).

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Drawdown from the running maximum: `(current - max) / max` (non-positive).
#[derive(Clone)]
pub struct Drawdown<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Drawdown<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Drawdown<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Drawdown`]: the running maximum plus the output buffer.
pub struct DrawdownState<T: Scalar + Float> {
    running_max: T,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Drawdown<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = DrawdownState<T>;

    fn init(self, _: (bool, ArrayView<'_, T, N>)) -> Self::State {
        DrawdownState {
            running_max: T::nan(),
            out: Array::scalar(T::zero()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, 0>) {
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        if state.running_max.is_nan() || current > state.running_max {
            state.running_max = current;
        }

        state.out[[]] = if state.running_max > T::zero() {
            (current - state.running_max) / state.running_max
        } else {
            T::zero()
        };
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

/// Running drawdown from the running peak.
pub fn drawdown<T: Scalar + Float, const N: usize>() -> Drawdown<T, N> {
    Drawdown::new()
}
