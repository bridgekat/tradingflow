//! Mean-over-standard-deviation of per-tick returns since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::{ArrayPort, UnitPort};

use super::common::{mean_var, ticked_value};

/// `mean(r) / std(r)` of period returns since inception.
#[derive(Clone)]
pub struct SharpeRatio<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> SharpeRatio<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for SharpeRatio<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`SharpeRatio`]: the accumulators plus the output buffer.
pub struct SharpeRatioState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for SharpeRatio<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = SharpeRatioState<T>;

    fn init(self, _: ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        SharpeRatioState {
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

            let (mean, var) = mean_var(state.sum, state.sum_sq, state.count);

            state.out[[]] = if var > T::zero() {
                mean / var.sqrt()
            } else {
                T::nan()
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

/// Running Sharpe ratio (mean / standard deviation) of a return stream.
pub fn sharpe_ratio<T: Scalar + Float, const N: usize>() -> SharpeRatio<T, N> {
    SharpeRatio::new()
}
