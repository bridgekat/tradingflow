//! Select operator — index selection along an axis.

use std::marker::PhantomData;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Select elements from an array along an axis.
///
/// Precomputes a flat index mapping at init time from the actual input shape.
///
/// When `squeeze` is `true` and exactly one index is selected, the
/// selected axis is removed from the output shape (e.g. selecting one
/// element from a `[5]` array yields a scalar `[]` instead of `[1]`).
pub struct Select<T: Scalar> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Select<T> {
    /// Create a new `Select` operator.
    ///
    /// * `indices` — positions to select along `axis`.
    /// * `axis` — the axis to select from.
    /// * `squeeze` — if `true` and `indices.len() == 1`, the selected
    ///   axis is removed from the output shape.
    pub fn new(indices: Vec<usize>, axis: usize, squeeze: bool) -> Self {
        assert!(
            !squeeze || indices.len() == 1,
            "squeeze requires exactly one index, got {}",
            indices.len(),
        );
        Self {
            indices,
            axis,
            squeeze,
            _phantom: PhantomData,
        }
    }

    /// Select by flat indices (axis 0, 1-D inputs, no squeeze).
    pub fn flat(indices: Vec<usize>) -> Self {
        Self::new(indices, 0, false)
    }

    /// Select along a specific axis (no squeeze).
    pub fn along_axis(indices: Vec<usize>, axis: usize) -> Self {
        Self::new(indices, axis, false)
    }
}

/// Runtime state for [`Select`].
pub struct SelectState {
    index_map: Vec<usize>,
}

impl<T: Scalar> Operator for Select<T> {
    type State = SelectState;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (SelectState, Array<T>) {
        let input_shape = inputs.shape();
        let index_map = compute_select_map(input_shape, &self.indices, self.axis);
        let mut output_shape = input_shape.to_vec();
        if output_shape.is_empty() {
            output_shape = vec![self.indices.len()];
        } else {
            output_shape[self.axis] = self.indices.len();
        }
        // Squeeze: remove the axis when selecting exactly one element.
        if self.squeeze && self.indices.len() == 1 {
            if output_shape.len() > self.axis {
                output_shape.remove(self.axis);
            }
        }
        (SelectState { index_map }, Array::zeros(&output_shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut SelectState,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;

    #[test]
    fn flat() {
        let a = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0_f64]);
        let (mut s, mut o) = Select::<f64>::flat(vec![1, 3]).init(&a, Instant::MIN);
        Select::compute(
            &mut s,
            &a,
            &mut o,
            Instant::from_nanos(1),
            false,
        );
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.as_slice(), &[20.0, 40.0]);
    }

    #[test]
    fn along_axis_columns() {
        // 2x3 matrix, select columns 0 and 2
        let a = Array::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64]);
        let (mut s, mut o) = Select::<f64>::along_axis(vec![0, 2], 1).init(&a, Instant::MIN);
        Select::compute(
            &mut s,
            &a,
            &mut o,
            Instant::from_nanos(1),
            false,
        );
        assert_eq!(o.shape(), &[2, 2]);
        assert_eq!(o.as_slice(), &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn single_element() {
        let a = Array::from_vec(&[4], vec![10.0, 20.0, 30.0, 40.0_f64]);
        let (mut s, mut o) = Select::<f64>::flat(vec![2]).init(&a, Instant::MIN);
        Select::compute(
            &mut s,
            &a,
            &mut o,
            Instant::from_nanos(1),
            false,
        );
        assert_eq!(o.as_slice(), &[30.0]);
    }

    #[test]
    fn init_reduces_axis() {
        let a = Array::from_vec(&[2, 3], vec![0.0_f64; 6]);
        let (_, o) = Select::<f64>::along_axis(vec![0], 1).init(&a, Instant::MIN);
        assert_eq!(o.shape(), &[2, 1]);
    }
}
