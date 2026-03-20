//! Stack operator — stacks N observables along a new axis.
//!
//! All inputs must have the same element type and shape.  The output shape
//! has a new dimension of size `n_inputs` inserted at position `axis`.
//!
//! Register via [`Scenario::add_slice_operator`].

use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::Operator;

/// Stack N homogeneous observables along a new axis.
///
/// Inserts a new dimension at `axis` with size = number of inputs.
/// For axis 0, the flat layout is just sequential copies (same as concat
/// axis 0 when all inputs have the same shape).
pub struct Stack<T: Copy> {
    /// Number of outer iterations (product of dims before `axis`).
    outer_count: usize,
    /// Number of elements per input per outer iteration (product of dims
    /// from `axis` onwards in the *input* shape).
    chunk_size: usize,
    axis: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Stack<T> {
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
        let outer_count = input_shape[..axis].iter().product::<usize>().max(1);
        let chunk_size = input_shape[axis..].iter().product::<usize>().max(1);
        Self {
            outer_count,
            chunk_size,
            axis,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + 'static> Operator for Stack<T> {
    type State = Self;
    type Inputs<'a>
        = Box<[&'a Observable<T>]>
    where
        Self: 'a;
    type Output<'o>
        = &'o mut Observable<T>
    where
        Self: 'o;

    fn shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        let n = input_shapes.len();
        if input_shapes[0].is_empty() {
            vec![n].into()
        } else {
            let s = input_shapes[0];
            let mut shape = Vec::with_capacity(s.len() + 1);
            shape.extend_from_slice(&s[..self.axis]);
            shape.push(n);
            shape.extend_from_slice(&s[self.axis..]);
            shape.into()
        }
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: Box<[&Observable<T>]>,
        output: &mut Observable<T>,
    ) -> bool {
        let out = output.current_mut();
        if state.outer_count == 1 {
            let mut offset = 0;
            for obs in inputs.iter() {
                let src = obs.current();
                out[offset..offset + src.len()].copy_from_slice(src);
                offset += src.len();
            }
        } else {
            let mut out_offset = 0;
            for outer in 0..state.outer_count {
                for obs in inputs.iter() {
                    let src = obs.current();
                    let src_offset = outer * state.chunk_size;
                    out[out_offset..out_offset + state.chunk_size]
                        .copy_from_slice(&src[src_offset..src_offset + state.chunk_size]);
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
    use crate::observable::Observable;

    #[test]
    fn stack_scalars_axis0() {
        let a = Observable::new(&[], &[1.0]);
        let b = Observable::new(&[], &[2.0]);
        let c = Observable::new(&[], &[3.0]);
        let mut state = Stack::<f64>::new(&[], 0);
        let mut out = Observable::new(&[3], &[0.0; 3]);
        assert!(Stack::compute(
            &mut state,
            vec![&a, &b, &c].into_boxed_slice(),
            &mut out
        ));
        assert_eq!(out.current(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn stack_vectors_axis0() {
        let a = Observable::new(&[2], &[1.0, 2.0]);
        let b = Observable::new(&[2], &[3.0, 4.0]);
        let c = Observable::new(&[2], &[5.0, 6.0]);
        let mut state = Stack::<f64>::new(&[2], 0);
        let mut out = Observable::new(&[3, 2], &[0.0; 6]);
        assert!(Stack::compute(
            &mut state,
            vec![&a, &b, &c].into_boxed_slice(),
            &mut out
        ));
        assert_eq!(out.current(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_vectors_axis1() {
        let a = Observable::new(&[3], &[1.0, 2.0, 3.0]);
        let b = Observable::new(&[3], &[4.0, 5.0, 6.0]);
        let mut state = Stack::<f64>::new(&[3], 1);
        let mut out = Observable::new(&[3, 2], &[0.0; 6]);
        assert!(Stack::compute(
            &mut state,
            vec![&a, &b].into_boxed_slice(),
            &mut out
        ));
        // Expected shape [3, 2]: [[1, 4], [2, 5], [3, 6]]
        // Flat row-major: [1, 4, 2, 5, 3, 6]
        assert_eq!(out.current(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn output_shape_computation() {
        let op = Stack::<f64>::new(&[], 0);
        assert_eq!(&*op.shape(&[&[], &[], &[]]), &[3]);
        let op = Stack::<f64>::new(&[2], 0);
        assert_eq!(&*op.shape(&[&[2], &[2], &[2]]), &[3, 2]);
        let op = Stack::<f64>::new(&[2], 1);
        assert_eq!(&*op.shape(&[&[2], &[2], &[2]]), &[2, 3]);
        let s: &[usize] = &[2, 3];
        let op = Stack::<f64>::new(&[2, 3], 0);
        assert_eq!(&*op.shape(&[s, s, s, s]), &[4, 2, 3]);
        let op = Stack::<f64>::new(&[2, 3], 1);
        assert_eq!(&*op.shape(&[s, s, s, s]), &[2, 4, 3]);
        let op = Stack::<f64>::new(&[2, 3], 2);
        assert_eq!(&*op.shape(&[s, s, s, s]), &[2, 3, 4]);
    }
}
