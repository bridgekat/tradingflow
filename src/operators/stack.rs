//! Stack operator — stacks N arrays along a new axis.

use crate::{Array, Operator, Scalar};

/// Stack N homogeneous arrays along a new axis.
pub struct Stack<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Stack<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Stack`].
pub struct StackState {
    outer_count: usize,
    chunk_size: usize,
}

impl<T: Scalar> Operator for Stack<T> {
    type State = StackState;
    type Inputs = [Array<T>];
    type Output = Array<T>;

    fn init(self, inputs: Box<[&Array<T>]>, _timestamp: i64) -> (StackState, Array<T>) {
        let first = inputs[0].shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = StackState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        (state, Array::default(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut StackState,
        inputs: Box<[&Array<T>]>,
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        super::concat::interleaved_copy(output, &inputs, state.outer_count, state.chunk_size);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    // Two 2×3 matrices stacked along each possible axis.
    //
    // a = [[1,2,3],[4,5,6]]   b = [[7,8,9],[10,11,12]]
    //
    // flat(a) = [1,2,3,4,5,6]   flat(b) = [7,8,9,10,11,12]

    fn ab() -> (Array<f64>, Array<f64>) {
        let a = Array::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Array::from_vec(&[2, 3], vec![7., 8., 9., 10., 11., 12.]);
        (a, b)
    }

    #[test]
    fn matrix_axis0() {
        // shape [2,2,3]: new axis = batch
        // [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Stack::<f64>::new(0).init(inputs.clone(), i64::MIN);
        Stack::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[2, 2, 3]);
        assert_eq!(
            o.as_slice(),
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
        );
    }

    #[test]
    fn matrix_axis1() {
        // shape [2,2,3]: new axis between rows and cols
        // [[[1,2,3],[7,8,9]], [[4,5,6],[10,11,12]]]
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Stack::<f64>::new(1).init(inputs.clone(), i64::MIN);
        Stack::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[2, 2, 3]);
        assert_eq!(
            o.as_slice(),
            &[1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]
        );
    }

    #[test]
    fn matrix_axis2() {
        // shape [2,3,2]: new axis = innermost
        // [[[1,7],[2,8],[3,9]], [[4,10],[5,11],[6,12]]]
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Stack::<f64>::new(2).init(inputs.clone(), i64::MIN);
        Stack::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[2, 3, 2]);
        assert_eq!(
            o.as_slice(),
            &[1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.]
        );
    }
}
