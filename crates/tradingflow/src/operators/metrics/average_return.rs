//! Arithmetic mean of per-tick returns since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::{ArrayPort, UnitPort};

use super::common::ticked_value;

/// Arithmetic mean of period returns since inception.
#[derive(Clone)]
pub struct AverageReturn<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> AverageReturn<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for AverageReturn<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`AverageReturn`]: the accumulators plus the output buffer.
pub struct AverageReturnState<T: Scalar + Float> {
    prev: T,
    sum: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for AverageReturn<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = AverageReturnState<T>;

    fn init(self, _: ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        AverageReturnState {
            prev: T::nan(),
            sum: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_clock, _), (_, data)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, 0>) {
        let Some(current) = ticked_value(produced_clock, data) else {
            return (false, state.out.view());
        };

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.count += 1;
            state.out[[]] = state.sum / T::from(state.count).unwrap();
        }

        state.prev = current;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

/// Running mean of a per-tick return stream.
pub fn average_return<T: Scalar + Float, const N: usize>() -> AverageReturn<T, N> {
    AverageReturn::new()
}
