//! Concat operator — concatenates N observables along an existing axis.
//!
//! All inputs must have the same element type and the same shape except
//! along the concatenation axis.  The output shape has the concatenation
//! axis size equal to the sum of all inputs' axis sizes.
//!
//! Register via [`Scenario::add_slice_operator`].

use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::Operator;

/// Concatenate N homogeneous observables along an existing axis.
///
/// For axis 0 (and for scalar/1-D inputs), the flat storage layout is just
/// sequential copies.  For higher axes, blocks are interleaved.
pub struct Concat<T: Copy> {
    /// Number of outer iterations (product of dims before `axis`).
    outer_count: usize,
    /// Number of elements per input per outer iteration (product of dims
    /// from `axis` onwards).
    chunk_size: usize,
    axis: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Concat<T> {
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

impl<T: Copy + 'static> Operator for Concat<T> {
    type Inputs<'a> = Box<[&'a Observable<T>]>;
    type Scalar = T;

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        let n = input_shapes.len();
        if input_shapes[0].is_empty() {
            vec![n].into()
        } else {
            let mut shape = input_shapes[0].to_vec();
            shape[self.axis] *= n;
            shape.into()
        }
    }

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Box<[&Observable<T>]>, out: &mut [T]) -> bool {
        if self.outer_count == 1 {
            // Fast path: sequential copy (axis 0 or scalar inputs).
            let mut offset = 0;
            for obs in inputs.iter() {
                let src = obs.current();
                out[offset..offset + src.len()].copy_from_slice(src);
                offset += src.len();
            }
        } else {
            // General path: interleave blocks for higher axes.
            let mut out_offset = 0;
            for outer in 0..self.outer_count {
                for obs in inputs.iter() {
                    let src = obs.current();
                    let src_offset = outer * self.chunk_size;
                    out[out_offset..out_offset + self.chunk_size]
                        .copy_from_slice(&src[src_offset..src_offset + self.chunk_size]);
                    out_offset += self.chunk_size;
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
    fn concat_scalars() {
        let a = Observable::new(&[], &[1.0]);
        let b = Observable::new(&[], &[2.0]);
        let c = Observable::new(&[], &[3.0]);
        let mut op = Concat::<f64>::new(&[], 0);
        let mut out = [0.0; 3];
        assert!(op.compute(1, vec![&a, &b, &c].into_boxed_slice(), &mut out));
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_vectors_axis0() {
        let a = Observable::new(&[2], &[1.0, 2.0]);
        let b = Observable::new(&[2], &[3.0, 4.0]);
        let mut op = Concat::<f64>::new(&[2], 0);
        let mut out = [0.0; 4];
        assert!(op.compute(1, vec![&a, &b].into_boxed_slice(), &mut out));
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_2d_axis1() {
        // Input shape [2, 2]: each input is a 2x2 matrix
        // a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]]
        let a = Observable::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = Observable::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        // Concat along axis 1 → output shape [2, 4]
        let mut op = Concat::<f64>::new(&[2, 2], 1);
        let mut out = [0.0; 8];
        assert!(op.compute(1, vec![&a, &b].into_boxed_slice(), &mut out));
        // Expected: [[1, 2, 5, 6], [3, 4, 7, 8]]
        assert_eq!(out, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn output_shape_computation() {
        let op = Concat::<f64>::new(&[], 0);
        assert_eq!(&*op.output_shape(&[&[], &[], &[]]), &[3]);
        let op = Concat::<f64>::new(&[2], 0);
        assert_eq!(&*op.output_shape(&[&[2], &[2], &[2]]), &[6]);
        let op = Concat::<f64>::new(&[2, 3], 0);
        assert_eq!(&*op.output_shape(&[&[2, 3], &[2, 3]]), &[4, 3]);
        let op = Concat::<f64>::new(&[2, 3], 1);
        assert_eq!(&*op.output_shape(&[&[2, 3], &[2, 3]]), &[2, 6]);
    }
}
