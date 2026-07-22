use num_traits::Float;

use super::Windowed;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::structural::buffer;
use crate::operators::transform::Lag;

/// The value from `n` ticks ago: `lag(n) @ x`. Self-recording; `NaN`
/// until more than `n` values have been seen.
pub fn lag<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, Lag<T, N>> {
    lag_or(n, T::nan())
}

/// [`lag`] with an explicit fill value — for non-float scalars, or when a
/// missing lag should read as something other than `NaN`.
pub fn lag_or<T: Scalar, const N: usize>(n: usize, fill: T) -> Windowed<T, N, Lag<T, N>> {
    Comp(buffer(n), Lag::new(n, fill))
}
