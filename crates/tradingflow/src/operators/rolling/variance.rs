use num_traits::Float;

use crate::data::Scalar;

use super::{Accumulator, Rolling, Window, rolling};

/// Incremental population variance via `E[x²] − E[x]²`.
pub struct VarianceAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    sum_sq: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for VarianceAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            sum_sq: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
                self.sum_sq[j] = self.sum_sq[j] + v * v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
                self.sum_sq[j] = self.sum_sq[j] - v * v;
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] == 0 {
                let mean = self.sum[j] / n;
                self.sum_sq[j] / n - mean * mean
            } else {
                T::nan()
            };
        }
    }
}

/// Element-wise rolling population variance (output rank `NO` = input rank).
pub type RollingVariance<T, const NO: usize> = Rolling<VarianceAccumulator<T>, NO, NO>;

/// Rolling population variance over a recorded [`Series`](tradingflow_data::Series). Self-recording
/// counterpart: [`mvar`](crate::operators::formula::mvar).
pub fn rolling_variance<T: Scalar + Float, const NO: usize>(
    window: Window,
) -> RollingVariance<T, NO> {
    rolling(window)
}
