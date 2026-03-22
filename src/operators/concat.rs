//! Concat operator — concatenates N stores along an existing axis.
//!
//! All inputs must have the same element type and the same shape except
//! along the concatenation axis.  The output shape has the concatenation
//! axis size equal to the sum of all inputs' axis sizes.

use std::marker::PhantomData;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

/// Concatenate N homogeneous stores along an existing axis.
///
/// For axis 0 (and for scalar/1-D inputs), the flat storage layout is just
/// sequential copies.  For higher axes, blocks are interleaved.
pub struct Concat<T: Scalar> {
    /// Number of outer iterations (product of dims before `axis`).
    outer_count: usize,
    /// Number of elements per input per outer iteration (product of dims
    /// from `axis` onwards).
    chunk_size: usize,
    axis: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Concat<T> {
    /// Create a Concat operator.
    ///
    /// `input_shape`: the element shape of each input (all inputs must
    ///     have the same shape).
    /// `axis`: the axis to concatenate along.
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
}

impl<T: Scalar> Operator for Concat<T> {
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
            let mut s = input_shapes[0].to_vec();
            s[self.axis] *= n;
            s.into()
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
    fn concat_scalars() {
        let a = Store::element(&[], &[1.0]);
        let b = Store::element(&[], &[2.0]);
        let c = Store::element(&[], &[3.0]);
        let mut state = Concat::<f64>::new(&[], 0);
        let mut out = Store::element(&[3], &[0.0; 3]);
        out.push_default(1);
        Concat::compute(
            &mut state,
            vec![&a, &b, &c].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        assert_eq!(out.current(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_vectors_axis0() {
        let a = Store::element(&[2], &[1.0, 2.0]);
        let b = Store::element(&[2], &[3.0, 4.0]);
        let mut state = Concat::<f64>::new(&[2], 0);
        let mut out = Store::element(&[4], &[0.0; 4]);
        out.push_default(1);
        Concat::compute(
            &mut state,
            vec![&a, &b].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        assert_eq!(out.current(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_2d_axis1() {
        // Input shape [2, 2]: each input is a 2x2 matrix
        // a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]]
        let a = Store::element(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = Store::element(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        // Concat along axis 1 -> output shape [2, 4]
        let mut state = Concat::<f64>::new(&[2, 2], 1);
        let mut out = Store::element(&[2, 4], &[0.0; 8]);
        out.push_default(1);
        Concat::compute(
            &mut state,
            vec![&a, &b].into_boxed_slice(),
            out.current_view_mut(),
        );
        out.commit();
        // Expected: [[1, 2, 5, 6], [3, 4, 7, 8]]
        assert_eq!(out.current(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn output_shape_computation() {
        let op = Concat::<f64>::new(&[], 0);
        assert_eq!(&*op.default(&[&[], &[], &[]]).0, &[3]);
        let op = Concat::<f64>::new(&[2], 0);
        assert_eq!(&*op.default(&[&[2], &[2], &[2]]).0, &[6]);
        let op = Concat::<f64>::new(&[2, 3], 0);
        assert_eq!(&*op.default(&[&[2, 3], &[2, 3]]).0, &[4, 3]);
        let op = Concat::<f64>::new(&[2, 3], 1);
        assert_eq!(&*op.default(&[&[2, 3], &[2, 3]]).0, &[2, 6]);
    }
}
