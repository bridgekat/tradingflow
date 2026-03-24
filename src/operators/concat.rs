//! Concat operator — concatenates N arrays along an existing axis.

use crate::{Array, Operator, Scalar};

/// Concatenate N homogeneous arrays along an existing axis.
pub struct Concat<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Concat<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Concat`].
pub struct ConcatState {
    outer_count: usize,
    chunk_size: usize,
}

impl<T: Scalar> Operator for Concat<T> {
    type State = ConcatState;
    type Inputs = [Array<T>];
    type Output = Array<T>;

    fn init(self, inputs: Box<[&Array<T>]>, _timestamp: i64) -> (ConcatState, Array<T>) {
        let first = inputs[0].shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = ConcatState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut ConcatState,
        inputs: Box<[&Array<T>]>,
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        interleaved_copy(output, &inputs, state.outer_count, state.chunk_size);
        true
    }
}

/// Copy data from N input arrays into an output array with interleaved
/// outer × chunk layout.  Used by [`Concat`] and [`Stack`].
#[inline(always)]
pub(super) fn interleaved_copy<T: Scalar>(
    output: &mut Array<T>,
    inputs: &[&Array<T>],
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_slice_mut();
    let mut offset = 0;
    for outer in 0..outer_count {
        for arr in inputs {
            let src = arr.as_slice();
            let src_offset = outer * chunk_size;
            out[offset..offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
            offset += chunk_size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;

    // Two 2×3×2 arrays concatenated along each axis.
    //
    // a[i][j][k] = 1 + i*6 + j*2 + k   (values 1..12)
    // b[i][j][k] = 13 + i*6 + j*2 + k   (values 13..24)
    //
    // flat(a) = [1,2,3,4,5,6,7,8,9,10,11,12]
    // flat(b) = [13,14,15,16,17,18,19,20,21,22,23,24]

    fn ab() -> (Array<f64>, Array<f64>) {
        let a = Array::from_vec(&[2, 3, 2], (1..=12).map(|x| x as f64).collect());
        let b = Array::from_vec(&[2, 3, 2], (13..=24).map(|x| x as f64).collect());
        (a, b)
    }

    #[test]
    fn array3d_axis0() {
        // [2,3,2] concat [2,3,2] along axis 0 → [4,3,2]
        // Just sequential: all of a, then all of b.
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Concat::<f64>::new(0).init(inputs.clone(), i64::MIN);
        Concat::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[4, 3, 2]);
        let expected: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        assert_eq!(o.as_slice(), &expected[..]);
    }

    #[test]
    fn array3d_axis1() {
        // [2,3,2] concat [2,3,2] along axis 1 → [2,6,2]
        // For each of the 2 outer slices, interleave 3×2 chunks.
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Concat::<f64>::new(1).init(inputs.clone(), i64::MIN);
        Concat::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[2, 6, 2]);
        // outer=0: a[0]=[1..6], b[0]=[13..18] → [1,2,3,4,5,6,13,14,15,16,17,18]
        // outer=1: a[1]=[7..12], b[1]=[19..24] → [7,8,9,10,11,12,19,20,21,22,23,24]
        assert_eq!(
            o.as_slice(),
            &[
                1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., 7., 8., 9., 10., 11., 12.,
                19., 20., 21., 22., 23., 24.
            ]
        );
    }

    #[test]
    fn array3d_axis2() {
        // [2,3,2] concat [2,3,2] along axis 2 → [2,3,4]
        // For each of the 2×3=6 outer positions, interleave 2-element chunks.
        let (a, b) = ab();
        let inputs: Box<[&Array<f64>]> = vec![&a, &b].into_boxed_slice();
        let (mut s, mut o) = Concat::<f64>::new(2).init(inputs.clone(), i64::MIN);
        Concat::compute(&mut s, inputs, &mut o, 1);
        assert_eq!(o.shape(), &[2, 3, 4]);
        // Each pair of 2-element chunks from a and b interleaved:
        // a[0][0]=[1,2], b[0][0]=[13,14] → [1,2,13,14]
        // a[0][1]=[3,4], b[0][1]=[15,16] → [3,4,15,16]
        // ...
        assert_eq!(
            o.as_slice(),
            &[
                1., 2., 13., 14., 3., 4., 15., 16., 5., 6., 17., 18., 7., 8., 19., 20., 9., 10.,
                21., 22., 11., 12., 23., 24.
            ]
        );
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn scalar_panics() {
        let a = Array::scalar(1.0_f64);
        let inputs: Box<[&Array<f64>]> = vec![&a].into_boxed_slice();
        Concat::<f64>::new(0).init(inputs, i64::MIN);
    }
}
