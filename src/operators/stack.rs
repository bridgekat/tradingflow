//! Stack operator — stacks N arrays along a new axis.

use crate::data::Instant;
use crate::{Array, Input, Notify, Operator, Scalar, Slice, SliceRefs};

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
    n_inputs: usize,
}

impl<T: Scalar> Operator for Stack<T> {
    type State = StackState;
    type Inputs = Slice<Input<Array<T>>>;
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (StackState, Array<T>) {
        assert!(!inputs.is_empty(), "Stack requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = StackState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut StackState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        super::concat::interleaved_copy(
            output,
            inputs.iter(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Array;
    use crate::operator::Operator;
    use crate::data::{InputTypes, SliceShape};

    fn make_slice<'a, T: Scalar>(
        arrays: &'a [&'a Array<T>],
    ) -> (Vec<*const u8>, SliceShape<Input<Array<T>>>) {
        let ptrs: Vec<*const u8> = arrays
            .iter()
            .map(|&a| a as *const Array<T> as *const u8)
            .collect();
        let shape = SliceShape::new(vec![(); arrays.len()].into_boxed_slice());
        (ptrs, shape)
    }

    fn refs<'a, T: Scalar>(
        ptrs: &'a [*const u8],
        shape: &'a SliceShape<Input<Array<T>>>,
    ) -> SliceRefs<'a, Input<Array<T>>> {
        let mut reader = crate::data::FlatRead::new(ptrs);
        unsafe { <Slice<Input<Array<T>>> as InputTypes>::refs_from_flat(&mut reader, shape) }
    }

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
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let (ptrs, shape) = make_slice(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(0).init(refs(&ptrs, &shape), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs, &shape),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
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
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let (ptrs, shape) = make_slice(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(1).init(refs(&ptrs, &shape), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs, &shape),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
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
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let (ptrs, shape) = make_slice(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(2).init(refs(&ptrs, &shape), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs, &shape),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
        assert_eq!(o.shape(), &[2, 3, 2]);
        assert_eq!(
            o.as_slice(),
            &[1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.]
        );
    }
}
