use num_traits::Float;

use super::base::{Accumulator, Return};
use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort};

/// Accumulator for [`comp_return`].
pub struct CompReturnAccumulator<T: Scalar + Float> {
    ratio: T,
}

impl<T: Scalar + Float> CompReturnAccumulator<T> {
    pub fn new() -> Self {
        Self { ratio: T::one() }
    }
}

impl<T: Scalar + Float> Default for CompReturnAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Accumulator<T> for CompReturnAccumulator<T> {
    fn add(&mut self, value: T) {
        self.ratio = self.ratio * (value + T::one());
    }

    fn output(&mut self, count: usize) -> T {
        self.ratio.powf(T::one() / T::from(count).unwrap()) - T::one()
    }
}

/// Amortized compounded return of a net-asset-value scalar, where
/// each period is specified by a clock pulse.
pub fn comp_return<T: Scalar + Float>()
-> impl Segment<Inputs = (ClockPort, ArrayPort<T, 0>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    Return::new(CompReturnAccumulator::new())
}
