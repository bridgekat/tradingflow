use num_traits::{One, Zero};

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Boolean and along `axis`: whether every scalar holds. An empty axis reduces
/// to `true`.
pub fn all<const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<bool, M>, Context = Instant> {
    array::reduce_along_axis(axis, true, |acc: &mut bool, &x| *acc &= x)
}

/// Boolean or along `axis`: whether any scalar holds. An empty axis reduces to
/// `false`.
pub fn any<const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<bool, M>, Context = Instant> {
    array::reduce_along_axis(axis, false, |acc: &mut bool, &x| *acc |= x)
}

/// Count along `axis`: how many scalars hold, as `T`, from [`Zero::zero`].
///
/// Counting into the same type the rest of the arithmetic uses keeps a ratio
/// like `count / all` a plain [`elem::div`](crate::operators::elem::div).
pub fn count<T: Scalar + Zero + One, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::zero(), |acc: &mut T, &x| {
        if x {
            *acc = acc.clone() + T::one();
        }
    })
}
