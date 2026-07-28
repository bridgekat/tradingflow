use std::cmp::{Ord, Ordering, PartialEq, PartialOrd};

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise equality comparison: [`PartialEq::eq`].
pub fn eq<T: Scalar + PartialEq, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.eq(b))
}

/// Elementwise inequality comparison: [`PartialEq::ne`].
pub fn ne<T: Scalar + PartialEq, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.ne(b))
}

/// Elementwise partial order: [`PartialOrd::partial_cmp`].
pub fn partial_cmp<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<Option<Ordering>, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.partial_cmp(b))
}

/// Elementwise less than: [`PartialOrd::lt`].
pub fn lt<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.lt(b))
}

/// Elementwise less than or equal: [`PartialOrd::le`].
pub fn le<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.le(b))
}

/// Elementwise greater than: [`PartialOrd::gt`].
pub fn gt<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.gt(b))
}

/// Elementwise greater than or equal: [`PartialOrd::ge`].
pub fn ge<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.ge(b))
}

/// Elementwise minimum: [`Ord::min`].
pub fn min<T: Scalar + Ord, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.min(b).clone())
}

/// Elementwise maximum: [`Ord::max`].
pub fn max<T: Scalar + Ord, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &T| a.max(b).clone())
}

/// Elementwise clamp: [`Ord::clamp`].
pub fn clamp<T: Scalar + Ord, const N: usize>(
    min: T,
    max: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: &T| x.clone().clamp(min.clone(), max.clone()))
}
