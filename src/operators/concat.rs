//! Concat operator — concatenates N arrays along an existing axis.

use std::marker::PhantomData;

use crate::array::Array;
use crate::operator::Operator;
use crate::types::Scalar;

/// Concatenate N homogeneous arrays along an existing axis.
pub struct Concat<T: Scalar> {
    outer_count: usize,
    chunk_size: usize,
    axis: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Concat<T> {
    pub fn new(input_shape: &[usize], axis: usize) -> Self {
        if input_shape.is_empty() {
            return Self {
                outer_count: 1,
                chunk_size: 1,
                axis,
                _phantom: PhantomData,
            };
        }
        debug_assert!(axis < input_shape.len(), "axis out of bounds");
        let outer_count = input_shape[..axis].iter().product::<usize>();
        let chunk_size = input_shape[axis..].iter().product::<usize>();
        Self {
            outer_count,
            chunk_size,
            axis,
            _phantom: PhantomData,
        }
    }

    fn output_shape(&self, inputs: &[&Array<T>]) -> Vec<usize> {
        let n = inputs.len();
        let first_shape = inputs[0].shape();
        if first_shape.is_empty() {
            vec![n]
        } else {
            let mut s = first_shape.to_vec();
            s[self.axis] *= n;
            s
        }
    }
}

impl<T: Scalar> Operator for Concat<T> {
    type State = Self;
    type Inputs = [Array<T>];
    type Output = Array<T>;

    fn init(self, inputs: Box<[&Array<T>]>, _timestamp: i64) -> (Self, Array<T>) {
        let shape = self.output_shape(&inputs);
        (self, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: Box<[&Array<T>]>,
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        let out = output.as_slice_mut();
        let mut out_offset = 0;
        for outer in 0..state.outer_count {
            for arr in inputs.iter() {
                let src = arr.as_slice();
                let src_offset = outer * state.chunk_size;
                out[out_offset..out_offset + state.chunk_size]
                    .clone_from_slice(&src[src_offset..src_offset + state.chunk_size]);
                out_offset += state.chunk_size;
            }
        }
        true
    }
}
