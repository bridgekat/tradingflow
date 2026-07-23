use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::num::Lag;
use crate::operators::series::buffer_n;
use crate::ports::ArrayPort;

/// The value from `n` ticks ago: `lag(n) @ x`. Self-recording; `NaN`
/// until more than `n` values have been seen.
pub fn lag<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    lag_or(n, T::nan())
}

/// [`lag`] with an explicit fill value — for non-float scalars, or when a
/// missing lag should read as something other than `NaN`.
pub fn lag_or<T: Scalar, const N: usize>(
    n: usize,
    fill: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(buffer_n(n), Lag::new(n, fill))
}
