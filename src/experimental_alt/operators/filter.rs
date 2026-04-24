//! Predicate-gated filter — ported from `src/operators/filter.rs`.

use std::marker::PhantomData;

use super::super::data::{Array, Input, InputTypes, Instant, Scalar};
use super::super::operator::Operator;

pub struct Filter<T: Scalar, F: Fn(&Array<T>) -> bool> {
    predicate: F,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(&Array<T>) -> bool> Filter<T, F> {
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, F: Fn(&Array<T>) -> bool + Send + 'static> Operator for Filter<T, F> {
    type State = Self;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (Self, Array<T>) {
        (self, inputs.clone())
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        if (state.predicate)(inputs) {
            output.as_mut_slice().clone_from_slice(inputs.as_slice());
            true
        } else {
            false
        }
    }
}
