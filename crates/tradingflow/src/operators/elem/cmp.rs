use std::cmp::{Ord, Ordering, PartialEq, PartialOrd};

use crate::data::{Instant, Scalar};
use crate::graph::typed::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise equality comparison.
pub fn eq<T: Scalar + PartialEq, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a == b)
}

/// Elementwise inequality comparison.
pub fn ne<T: Scalar + PartialEq, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a != b)
}

/// Elementwise partial order.
pub fn partial_cmp<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<Option<Ordering>, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a.partial_cmp(&b))
}

/// Elementwise `<`.
pub fn lt<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a < b)
}

/// Elementwise `<=`.
pub fn le<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a <= b)
}

/// Elementwise `>`.
pub fn gt<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a > b)
}

/// Elementwise `>=`.
pub fn ge<T: Scalar + PartialOrd, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a >= b)
}

/// Elementwise minimum.
pub fn min<T: Scalar + Ord, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a.min(b))
}

/// Elementwise maximum.
pub fn max<T: Scalar + Ord, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: T| a.max(b))
}

/// Elementwise clamp.
pub fn clamp<T: Scalar + Ord, const N: usize>(
    min: T,
    max: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: T| x.clamp(min.clone(), max.clone()))
}
