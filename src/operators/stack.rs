//! Stack operator — stacks N stores along a new axis.
//!
//! All inputs must have the same element type and shape.  The output shape
//! has a new dimension of size `n_inputs` inserted at position `axis`.

use std::marker::PhantomData;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

/// Stack N homogeneous stores along a new axis.
///
/// Inserts a new dimension at `axis` with size = number of inputs.
/// For axis 0, the flat layout is just sequential copies (same as concat
/// axis 0 when all inputs have the same shape).
pub struct Stack<T: Scalar> {
    /// Number of outer iterations (product of dims before `axis`).
    outer_count: usize,
    /// Number of elements per input per outer iteration (product of dims
    /// from `axis` onwards in the *input* shape).
    chunk_size: usize,
    axis: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Stack<T> {
    /// Create a Stack operator.
    ///
    /// `input_shape`: the element shape of each input (all must match).
    /// `axis`: position for the new dimension (0 = outermost).
    pub fn new(input_shape: &[usize], axis: usize) -> Self {
        debug_assert!(axis <= input_shape.len(), "axis out of bounds");
        if input_shape.is_empty() {
            return Self {
                outer_count: 1,
                chunk_size: 1,
                axis,
                _phantom: PhantomData,
            };
        }
        let outer_count = input_shape[..axis].iter().product::<usize>();
        let chunk_size = input_shape[axis..].iter().product::<usize>();
        Self {
            outer_count,
            chunk_size,
            axis,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar> Operator for Stack<T> {
    type State = Self;
    type Inputs = [Store<T>];
    type Scalar = T;

    fn window_sizes(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        let n = input_shapes.len();
        vec![1; n].into()
    }

    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[T]>) {
        let n = input_shapes.len();
        let shape: Box<[usize]> = if input_shapes[0].is_empty() {
            vec![n].into()
        } else {
            let s = input_shapes[0];
            let mut v = Vec::with_capacity(s.len() + 1);
            v.extend_from_slice(&s[..self.axis]);
            v.push(n);
            v.extend_from_slice(&s[self.axis..]);
            v.into()
        };
        let stride = shape.iter().product::<usize>();
        (shape, vec![T::default(); stride].into())
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: Box<[&Store<T>]>, output: ElementViewMut<'_, T>) -> bool {
        let out = output.values;
        if state.outer_count == 1 {
            let mut offset = 0;
            for store in inputs.iter() {
                let src = store.current();
                out[offset..offset + src.len()].clone_from_slice(src);
                offset += src.len();
            }
        } else {
            let mut out_offset = 0;
            for outer in 0..state.outer_count {
                for store in inputs.iter() {
                    let src = store.current();
                    let src_offset = outer * state.chunk_size;
                    out[out_offset..out_offset + state.chunk_size]
                        .clone_from_slice(&src[src_offset..src_offset + state.chunk_size]);
                    out_offset += state.chunk_size;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn stack_scalars_axis0() {
        let a = Store::element(&[], &[1.0]);
        let b = Store::element(&[], &[2.0]);
        let c = Store::element(&[], &[3.0]);
        let mut state = Stack::<f64>::new(&[], 0);
        let mut out = Store::element(&[3], &[0.0; 3]);
        out.push_default(1);
        Stack::compute(
            &mut state,
            vec![&a, &b, &c].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        assert_eq!(out.current(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn stack_vectors_axis0() {
        let a = Store::element(&[2], &[1.0, 2.0]);
        let b = Store::element(&[2], &[3.0, 4.0]);
        let c = Store::element(&[2], &[5.0, 6.0]);
        let mut state = Stack::<f64>::new(&[2], 0);
        let mut out = Store::element(&[3, 2], &[0.0; 6]);
        out.push_default(1);
        Stack::compute(
            &mut state,
            vec![&a, &b, &c].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        assert_eq!(out.current(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_vectors_axis1() {
        let a = Store::element(&[3], &[1.0, 2.0, 3.0]);
        let b = Store::element(&[3], &[4.0, 5.0, 6.0]);
        let mut state = Stack::<f64>::new(&[3], 1);
        let mut out = Store::element(&[3, 2], &[0.0; 6]);
        out.push_default(1);
        Stack::compute(
            &mut state,
            vec![&a, &b].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        // Expected shape [3, 2]: [[1, 4], [2, 5], [3, 6]]
        // Flat row-major: [1, 4, 2, 5, 3, 6]
        assert_eq!(out.current(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn output_shape_computation() {
        let op = Stack::<f64>::new(&[], 0);
        assert_eq!(&*op.default(&[&[], &[], &[]]).0, &[3]);
        let op = Stack::<f64>::new(&[2], 0);
        assert_eq!(&*op.default(&[&[2], &[2], &[2]]).0, &[3, 2]);
        let op = Stack::<f64>::new(&[2], 1);
        assert_eq!(&*op.default(&[&[2], &[2], &[2]]).0, &[2, 3]);
        let s: &[usize] = &[2, 3];
        let op = Stack::<f64>::new(&[2, 3], 0);
        assert_eq!(&*op.default(&[s, s, s, s]).0, &[4, 2, 3]);
        let op = Stack::<f64>::new(&[2, 3], 1);
        assert_eq!(&*op.default(&[s, s, s, s]).0, &[2, 4, 3]);
        let op = Stack::<f64>::new(&[2, 3], 2);
        assert_eq!(&*op.default(&[s, s, s, s]).0, &[2, 3, 4]);
    }
}
