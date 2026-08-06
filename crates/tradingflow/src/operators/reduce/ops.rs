use num_traits::{One, Zero};

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Sum along `axis`, from [`Zero::zero`]: [`Add`](std::ops::Add).
pub fn sum<T: Scalar + Zero, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::zero(), |acc: &mut T, x: &T| {
        *acc = acc.clone() + x.clone()
    })
}

/// Product along `axis`, from [`One::one`]: [`Mul`](std::ops::Mul).
pub fn product<T: Scalar + One, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::one(), |acc: &mut T, x: &T| {
        *acc = acc.clone() * x.clone()
    })
}
