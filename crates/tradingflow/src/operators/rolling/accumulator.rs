use num_traits::Float;

use crate::data::Scalar;

/// Incremental computation over a rolling window of array elements.
pub trait Accumulator: Send + Sync + 'static {
    type Scalar: Scalar + Float;

    fn new(input_shape: &[usize]) -> Self;

    fn output_shape(input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn add(&mut self, element: &[Self::Scalar]);
    fn remove(&mut self, element: &[Self::Scalar]);
    fn write(&self, count: usize, output: &mut [Self::Scalar]);
}
