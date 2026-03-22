//! Where operator — element-wise conditional replacement.

use std::marker::PhantomData;

use crate::array::Array;
use crate::operator::Operator;
use crate::types::Scalar;

/// Element-wise conditional operator.
pub struct Where<T: Scalar, F: Fn(T) -> bool> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool> Where<T, F> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, F: Fn(T) -> bool + Send + 'static> Operator for Where<T, F> {
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
        let a = inputs.0.as_slice();
        let out = output.as_slice_mut();
        for i in 0..out.len() {
            out[i] = if (state.condition)(a[i].clone()) {
                a[i].clone()
            } else {
                state.fill.clone()
            };
        }
        true
    }
}
