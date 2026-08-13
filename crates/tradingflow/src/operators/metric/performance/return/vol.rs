use num_traits::Float;

use super::base::{Accumulator, LogReturn, Return};
use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SignalPort};

/// Accumulator for [`return_vol`].
pub struct ReturnVolAccumulator<T: Scalar + Float> {
    sum: T,
    sum2: T,
}

impl<T: Scalar + Float> ReturnVolAccumulator<T> {
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            sum2: T::zero(),
        }
    }
}

impl<T: Scalar + Float> Default for ReturnVolAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Accumulator<T> for ReturnVolAccumulator<T> {
    fn add(&mut self, value: T) {
        self.sum = self.sum + value;
        self.sum2 = self.sum2 + value * value;
    }

    fn output(&mut self, count: usize) -> T {
        let n = T::from(count).unwrap();
        let mean = self.sum / n;
        let var = (self.sum2 / n - mean * mean).max(T::zero());
        var.sqrt()
    }
}

/// Volatility (standard deviation) of per-period percentage return of a
/// net-asset-value scalar, where each period is specified by a signal.
///
/// Inputs and outputs: see [module-level docs](super::super).
pub fn return_vol<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    Return::new(ReturnVolAccumulator::new())
}

/// Volatility (standard deviation) of per-period log return of a
/// net-asset-value scalar, where each period is specified by a signal.
///
/// Inputs and outputs: see [module-level docs](super::super).
pub fn log_return_vol<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    LogReturn::new(ReturnVolAccumulator::new())
}
