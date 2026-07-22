use num_traits::Float;

use crate::data::Scalar;

use super::{Accumulator, Rolling, Window, rolling};

/// Incremental mean.
pub struct MeanAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for MeanAccumulator<T> {
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

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j] / n
            };
        }
    }
}

/// Element-wise rolling mean (output rank `NO` = input element rank).
pub type RollingMean<T, const NO: usize> = Rolling<MeanAccumulator<T>, NO, NO>;

/// Rolling mean over a recorded [`Series`](tradingflow_data::Series). The self-recording counterpart
/// over a live array handle is [`ma`](crate::operators::formula::ma) /
/// [`ma_time`](crate::operators::formula::ma_time).
pub fn rolling_mean<T: Scalar + Float, const NO: usize>(window: Window) -> RollingMean<T, NO> {
    rolling(window)
}
