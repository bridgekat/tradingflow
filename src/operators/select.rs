//! Select operator — index selection along an axis.
//!
//! Selects elements from an observable along a given axis using a precomputed
//! flat index mapping.  Supports arbitrary axis and multi-index selection.
//!
//! Register via [`Scenario::add_operator`] with `(Obs<T>,)` input.

use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::Operator;

/// Select elements from an observable along an axis.
///
/// Precomputes a flat index mapping at construction time for O(1) per-element
/// access during compute.
pub struct Select<T: Copy> {
    /// For each output element, the index in the flat input buffer.
    index_map: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Select<T> {
    /// Select by flat indices (for 1-D inputs or when the caller has
    /// precomputed the mapping).
    pub fn flat(indices: Vec<usize>) -> Self {
        Self {
            index_map: indices,
            _phantom: PhantomData,
        }
    }

    /// Select along a specific axis of a multi-dimensional input.
    ///
    /// `input_shape`: the element shape of the input observable.
    /// `indices`: the indices to select along `axis`.
    /// `axis`: the axis to select along.
    pub fn along_axis(input_shape: &[usize], indices: &[usize], axis: usize) -> Self {
        let index_map = compute_select_map(input_shape, indices, axis);
        Self {
            index_map,
            _phantom: PhantomData,
        }
    }

    /// Compute the output shape for a select operation.
    ///
    /// Returns the input shape with `shape[axis]` replaced by `indices.len()`.
    pub fn output_shape(input_shape: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
        let mut shape = input_shape.to_vec();
        shape[axis] = indices.len();
        shape
    }
}

impl<T: Copy> Operator for Select<T> {
    type Inputs<'a>
        = (&'a Observable<T>,)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (obs,) = inputs;
        let input = obs.last();
        for (i, &src_idx) in self.index_map.iter().enumerate() {
            out[i] = input[src_idx];
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Index mapping helper
// ---------------------------------------------------------------------------

/// Compute flat index mapping for axis-based selection.
fn compute_select_map(input_shape: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    let ndim = input_shape.len();
    debug_assert!(axis < ndim, "axis out of bounds");

    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = indices.len();

    let output_size: usize = output_shape.iter().product::<usize>().max(1);
    let input_strides = row_major_strides(input_shape);
    let output_strides = row_major_strides(&output_shape);

    let mut flat_map = Vec::with_capacity(output_size);
    for out_flat in 0..output_size {
        let mut in_flat = 0;
        let mut remaining = out_flat;
        for d in 0..ndim {
            let stride = if d + 1 < ndim { output_strides[d] } else { 1 };
            let coord = remaining / stride;
            remaining %= stride;
            if d == axis {
                in_flat += indices[coord] * input_strides[d];
            } else {
                in_flat += coord * input_strides[d];
            }
        }
        flat_map.push(in_flat);
    }
    flat_map
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observable;

    #[test]
    fn select_flat_indices() {
        let obs = Observable::new(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut op = Select::<f64>::flat(vec![1, 3]);
        let mut out = [0.0, 0.0];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [20.0, 40.0]);
    }

    #[test]
    fn select_single_index() {
        let obs = Observable::new(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let mut op = Select::<f64>::flat(vec![2]);
        let mut out = [0.0];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [3.0]);
    }

    #[test]
    fn select_along_axis_2d() {
        // Input shape [3, 2]: [[1, 2], [3, 4], [5, 6]]
        let obs = Observable::new(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Select rows 0 and 2 along axis 0
        let mut op = Select::<f64>::along_axis(&[3, 2], &[0, 2], 0);
        let mut out = [0.0; 4];
        assert!(op.compute(1, (&obs,), &mut out));
        // Expected: [[1, 2], [5, 6]] → flat [1, 2, 5, 6]
        assert_eq!(out, [1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn select_along_axis1_2d() {
        // Input shape [2, 4]: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let obs = Observable::new(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Select columns 1 and 3 along axis 1
        let mut op = Select::<f64>::along_axis(&[2, 4], &[1, 3], 1);
        let mut out = [0.0; 4];
        assert!(op.compute(1, (&obs,), &mut out));
        // Expected: [[2, 4], [6, 8]] → flat [2, 4, 6, 8]
        assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn output_shape_computation() {
        assert_eq!(Select::<f64>::output_shape(&[5], &[1, 3], 0), vec![2]);
        assert_eq!(
            Select::<f64>::output_shape(&[3, 4], &[0, 2], 0),
            vec![2, 4]
        );
        assert_eq!(
            Select::<f64>::output_shape(&[3, 4], &[1], 1),
            vec![3, 1]
        );
    }
}
