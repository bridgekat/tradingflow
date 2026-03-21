//! Select operator — index selection along an axis.
//!
//! Selects elements from a store along a given axis using a precomputed
//! flat index mapping.  Supports arbitrary axis and multi-index selection.

use std::marker::PhantomData;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

/// Select elements from a store along an axis.
///
/// Precomputes a flat index mapping at construction time for O(1) per-element
/// access during compute.
pub struct Select<T: Copy> {
    /// For each output element, the index in the flat input buffer.
    index_map: Vec<usize>,
    /// Axis along which selection was performed (used for output_shape).
    axis: usize,
    /// Number of selected indices.
    n_selected: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Select<T> {
    /// Select by flat indices (for 1-D inputs or when the caller has
    /// precomputed the mapping).
    pub fn flat(indices: Vec<usize>) -> Self {
        let n = indices.len();
        Self {
            index_map: indices,
            axis: 0,
            n_selected: n,
            _phantom: PhantomData,
        }
    }

    /// Select along a specific axis of a multi-dimensional input.
    ///
    /// `input_shape`: the element shape of the input store.
    /// `indices`: the indices to select along `axis`.
    /// `axis`: the axis to select along.
    pub fn along_axis(input_shape: &[usize], indices: &[usize], axis: usize) -> Self {
        let index_map = compute_select_map(input_shape, indices, axis);
        Self {
            index_map,
            axis,
            n_selected: indices.len(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar> Operator for Select<T> {
    type State = Self;
    type Inputs = (Store<T>,);
    type Scalar = T;

    fn window_sizes(&self, _: &[&[usize]]) -> (usize,) {
        (1,)
    }

    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[T]>) {
        let mut shape = input_shapes[0].to_vec();
        shape[self.axis] = self.n_selected;
        let stride = shape.iter().product::<usize>();
        (shape.into(), vec![T::default(); stride].into())
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: (&Store<T>,), output: ElementViewMut<'_, T>) -> bool {
        let input = inputs.0.current();
        let out = output.values;
        for (i, &src_idx) in state.index_map.iter().enumerate() {
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

    let output_size: usize = output_shape.iter().product::<usize>();
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
    use crate::store::Store;

    #[test]
    fn select_flat_indices() {
        let store = Store::element(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut state = Select::<f64>::flat(vec![1, 3]);
        let mut out = Store::element(&[2], &[0.0, 0.0]);
        out.push_default(1);
        Select::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[20.0, 40.0]);
    }

    #[test]
    fn select_single_index() {
        let store = Store::element(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let mut state = Select::<f64>::flat(vec![2]);
        let mut out = Store::element(&[1], &[0.0]);
        out.push_default(1);
        Select::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[3.0]);
    }

    #[test]
    fn select_along_axis_2d() {
        // Input shape [3, 2]: [[1, 2], [3, 4], [5, 6]]
        let store = Store::element(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Select rows 0 and 2 along axis 0
        let mut state = Select::<f64>::along_axis(&[3, 2], &[0, 2], 0);
        let mut out = Store::element(&[2, 2], &[0.0; 4]);
        out.push_default(1);
        Select::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        // Expected: [[1, 2], [5, 6]] -> flat [1, 2, 5, 6]
        assert_eq!(out.current(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn select_along_axis1_2d() {
        // Input shape [2, 4]: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let store = Store::element(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Select columns 1 and 3 along axis 1
        let mut state = Select::<f64>::along_axis(&[2, 4], &[1, 3], 1);
        let mut out = Store::element(&[2, 2], &[0.0; 4]);
        out.push_default(1);
        Select::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        // Expected: [[2, 4], [6, 8]] -> flat [2, 4, 6, 8]
        assert_eq!(out.current(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn output_shape_computation() {
        let op = Select::<f64>::flat(vec![1, 3]);
        assert_eq!(&*op.default(&[&[5]]).0, &[2]);
        let op = Select::<f64>::along_axis(&[3, 4], &[0, 2], 0);
        assert_eq!(&*op.default(&[&[3, 4]]).0, &[2, 4]);
        let op = Select::<f64>::along_axis(&[3, 4], &[1], 1);
        assert_eq!(&*op.default(&[&[3, 4]]).0, &[3, 1]);
    }
}
