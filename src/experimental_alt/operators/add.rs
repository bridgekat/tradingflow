//! Element-wise addition — ported from `src/operators/num/arithmetic.rs`.

use std::marker::PhantomData;
use std::ops;

use super::super::data::{Array, Input, InputTypes, Instant, Scalar};
use super::super::operator::Operator;

pub struct Add<T: Scalar>(PhantomData<T>);

impl<T: Scalar + ops::Add<Output = T>> Add<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + ops::Add<Output = T>> Default for Add<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + ops::Add<Output = T>> Operator for Add<T> {
    type State = ();
    type Inputs = (Input<Array<T>>, Input<Array<T>>);
    type Output = Array<T>;

    fn init(
        self,
        inputs: (&Array<T>, &Array<T>),
        _timestamp: Instant,
    ) -> ((), Array<T>) {
        ((), Array::zeros(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: (&Array<T>, &Array<T>),
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let a_sl = inputs.0.as_slice();
        let b_sl = inputs.1.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = a_sl[i].clone() + b_sl[i].clone();
        }
        true
    }
}
