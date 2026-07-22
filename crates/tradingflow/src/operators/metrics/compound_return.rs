//! Compounded per-tick growth rate since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::{ArrayPort, UnitPort};

use super::common::ticked_value;

/// `(current / first)^(1/n) - 1` over clock ticks since inception.
#[derive(Clone)]
pub struct CompoundReturn<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> CompoundReturn<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for CompoundReturn<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`CompoundReturn`]: the accumulators plus the output
/// buffer.
pub struct CompoundReturnState<T: Scalar + Float> {
    first_value: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for CompoundReturn<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = CompoundReturnState<T>;

    fn init(self, _: ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        CompoundReturnState {
            first_value: T::nan(),
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

        state.count += 1;

        if state.first_value.is_nan() {
            state.first_value = current;
            state.out[[]] = T::zero();
            return (true, state.out.view());
        }

        if state.first_value <= T::zero() || current <= T::zero() {
            state.out[[]] = T::nan();
            return (true, state.out.view());
        }

        let ratio = current / state.first_value;
        let n = T::from(state.count - 1).unwrap();
        if n <= T::zero() {
            state.out[[]] = T::zero();
        } else {
            state.out[[]] = ratio.powf(T::one() / n) - T::one();
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

/// Cumulative compounded return of a per-tick return stream.
pub fn compound_return<T: Scalar + Float, const N: usize>() -> CompoundReturn<T, N> {
    CompoundReturn::new()
}
