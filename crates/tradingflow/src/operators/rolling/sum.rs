use num_traits::Float;

use crate::data::Scalar;

use super::{Accumulator, Rolling, Window, rolling};

/// Incremental sum (non-finite values skipped + counted; NaN if any present).
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for SumAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
            }
        }
    }

    fn write(&self, _count: usize, output: &mut [T]) {
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j]
            };
        }
    }
}

/// Element-wise rolling sum (output rank `NO` = input element rank).
pub type RollingSum<T, const NO: usize> = Rolling<SumAccumulator<T>, NO, NO>;

/// Rolling sum over a recorded [`Series`](tradingflow_data::Series): `rolling_sum(Window::Count(20)) @ xs`.
pub fn rolling_sum<T: Scalar + Float, const NO: usize>(window: Window) -> RollingSum<T, NO> {
    rolling(window)
}
