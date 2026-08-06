use num_traits::Bounded;

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Maximum along `axis`, from [`Bounded::min_value`]: [`Ord::max`].
///
/// An empty axis reduces to [`Bounded::min_value`]. Floats are not [`Ord`] —
/// see [`maxf`](super::maxf).
pub fn max<T: Scalar + Ord + Bounded, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::min_value(), |acc: &mut T, x: &T| {
        *acc = acc.clone().max(x.clone())
    })
}

/// Minimum along `axis`, from [`Bounded::max_value`]: [`Ord::min`].
///
/// An empty axis reduces to [`Bounded::max_value`]. Floats are not [`Ord`] —
/// see [`minf`](super::minf).
pub fn min<T: Scalar + Ord + Bounded, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::max_value(), |acc: &mut T, x: &T| {
        *acc = acc.clone().min(x.clone())
    })
}
