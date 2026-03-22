//! Filter operator — whole-element filter by predicate.

use std::marker::PhantomData;

use crate::array::Array;
use crate::operator::Operator;
use crate::types::Scalar;

/// Filter operator: passes or drops the entire element based on a predicate.
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
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (Self, Array<T>) {
        (self, inputs.0.clone())
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        if (state.predicate)(inputs.0) {
            output.assign(inputs.0);
            true
        } else {
            false
        }
    }
}
