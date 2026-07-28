use num_traits::Float;

use super::base::{Accumulator, LogReturn, Return};
use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort};

/// Accumulator for [`return_mean`].
pub struct ReturnMeanAccumulator<T: Scalar + Float> {
    sum: T,
}

impl<T: Scalar + Float> ReturnMeanAccumulator<T> {
    pub fn new() -> Self {
        Self { sum: T::zero() }
    }
}

impl<T: Scalar + Float> Default for ReturnMeanAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Accumulator<T> for ReturnMeanAccumulator<T> {
    fn add(&mut self, value: T) {
        self.sum = self.sum + value;
    }

    fn output(&mut self, count: usize) -> T {
        self.sum / T::from(count).unwrap()
    }
}

/// Average per-period percentage return of a net-asset-value scalar, where
/// each period is specified by a clock signal.
pub fn return_mean<T: Scalar + Float>()
-> impl Segment<Inputs = (ClockPort, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    Return::new(ReturnMeanAccumulator::new())
}

/// Average per-period log return of a net-asset-value scalar, where
/// each period is specified by a clock signal.
pub fn log_return_mean<T: Scalar + Float>()
-> impl Segment<Inputs = (ClockPort, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    LogReturn::new(ReturnMeanAccumulator::new())
}
