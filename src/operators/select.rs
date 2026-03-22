//! Select operator — index selection along an axis.

use std::marker::PhantomData;

use crate::array::Array;
use crate::operator::Operator;
use crate::types::Scalar;

/// Select elements from an array along an axis.
pub struct Select<T: Scalar> {
    index_map: Vec<usize>,
    output_shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Select<T> {
    pub fn flat(indices: Vec<usize>) -> Self {
        let n = indices.len();
        Self {
            index_map: indices,
            output_shape: vec![n],
            _phantom: PhantomData,
        }
    }

    pub fn along_axis(input_shape: &[usize], indices: &[usize], axis: usize) -> Self {
        let index_map = compute_select_map(input_shape, indices, axis);
        let mut output_shape = input_shape.to_vec();
        output_shape[axis] = indices.len();
        Self {
            index_map,
            output_shape,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar> Operator for Select<T> {
    type State = Self;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: i64) -> (Self, Array<T>) {
        let output = Array::zeros(&self.output_shape);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        let src = inputs.0.as_slice();
        let dst = output.as_slice_mut();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        true
    }
}

fn compute_select_map(input_shape: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    if input_shape.is_empty() {
        return indices.to_vec();
    }
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let axis_size = input_shape[axis];
    let mut map = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &idx in indices {
            for i in 0..inner {
                map.push(o * axis_size * inner + idx * inner + i);
            }
        }
    }
    map
}
