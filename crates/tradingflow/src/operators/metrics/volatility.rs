//! Population standard deviation of per-tick returns since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::{ArrayPort, UnitPort};

use super::common::{mean_var, ticked_value};

/// Population standard deviation of period returns since inception.
#[derive(Clone)]
pub struct Volatility<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Volatility<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Volatility<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Volatility`]: the accumulators plus the output buffer.
pub struct VolatilityState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Volatility<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = VolatilityState<T>;

    fn init(self, _: ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        VolatilityState {
            prev: T::nan(),
            sum: T::zero(),
            sum_sq: T::zero(),
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
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;
            let (_, var) = mean_var(state.sum, state.sum_sq, state.count);
            state.out[[]] = if var > T::zero() {
                var.sqrt()
            } else {
                T::zero()
            };
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

/// Running standard deviation of a per-tick return stream.
pub fn volatility<T: Scalar + Float, const N: usize>() -> Volatility<T, N> {
    Volatility::new()
}
