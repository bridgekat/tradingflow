use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Floating-point maximum along `axis`, from `NaN`: [`Float::max`].
///
/// `NaN` is passed over as [`Float::max`] passes it over, so an axis with
/// nothing but `NaN` on it — an empty one included — reduces to `NaN` rather
/// than to a bound as [`max`](super::max) does. Infinities compare as usual.
pub fn maxf<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::nan(), |acc: &mut T, &x: &T| *acc = acc.max(x))
}

/// Floating-point minimum along `axis`, from `NaN`: [`Float::min`].
///
/// `NaN` is passed over as [`Float::min`] passes it over, so an axis with
/// nothing but `NaN` on it — an empty one included — reduces to `NaN` rather
/// than to a bound as [`min`](super::min) does. Infinities compare as usual.
pub fn minf<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::nan(), |acc: &mut T, &x: &T| *acc = acc.min(x))
}

/// Count along `axis` of the finite scalars, as `T`: [`Float::is_finite`].
///
/// The denominator of a mean: `sum_finite(axis)` over `count_finite(axis)`
/// through [`elem::div`](crate::operators::elem::div) is the finite-sample
/// mean, and gives `NaN` where nothing was finite.
pub fn count_finite<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::zero(), |acc: &mut T, &x: &T| {
        if x.is_finite() {
            *acc = *acc + T::one();
        }
    })
}

/// Sum along `axis` over the finite scalars: [`Float::is_finite`].
///
/// Where [`sum`](super::sum) propagates whatever IEEE arithmetic gives it,
/// this treats non-finite entries (`NaN` or ±∞) as missing, as
/// [`stats`](crate::operators::stats) does. An axis with nothing finite on it
/// sums to zero — pair with [`count_finite`] to tell that from a real zero.
pub fn sum_finite<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::zero(), |acc: &mut T, &x: &T| {
        if x.is_finite() {
            *acc = *acc + x;
        }
    })
}

/// Product along `axis` over the finite scalars: [`Float::is_finite`].
///
/// Where [`product`](super::product) propagates whatever IEEE arithmetic gives
/// it, this treats non-finite entries (`NaN` or ±∞) as missing, as
/// [`stats`](crate::operators::stats) does. An axis with nothing finite on it
/// products to one — pair with [`count_finite`] to tell that from a real one.
pub fn product_finite<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    array::reduce_along_axis(axis, T::one(), |acc: &mut T, &x: &T| {
        if x.is_finite() {
            *acc = *acc * x;
        }
    })
}
