use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use crate::data::{Instant, Scalar};
use crate::graph::typed::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise negation: [`Neg::neg`].
pub fn negate<T: Scalar + Neg<Output: Scalar>, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T::Output, N>, Context = Instant> {
    array::map(|x: T| -x)
}

/// Elementwise addition: [`Add::add`].
pub fn add<T: Scalar + Add<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a + b)
}

/// Elementwise subtraction: [`Sub::sub`].
pub fn subtract<T: Scalar + Sub<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a - b)
}

/// Elementwise multiplication: [`Mul::mul`].
pub fn multiply<T: Scalar + Mul<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a * b)
}

/// Elementwise division: [`Div::div`].
pub fn divide<T: Scalar + Div<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a / b)
}

/// Elementwise remainder: [`Rem::rem`].
pub fn remainder<T: Scalar + Rem<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a % b)
}

/// Elementwise bitwise NOT: [`Not::not`].
pub fn not<T: Scalar + Not<Output: Scalar>, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T::Output, N>, Context = Instant> {
    array::map(|x: T| !x)
}

/// Elementwise bitwise AND: [`BitAnd::bitand`].
pub fn bitand<T: Scalar + BitAnd<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a & b)
}

/// Elementwise bitwise OR: [`BitOr::bitor`].
pub fn bitor<T: Scalar + BitOr<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a | b)
}

/// Elementwise bitwise XOR: [`BitXor::bitxor`].
pub fn bitxor<T: Scalar + BitXor<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a ^ b)
}

/// Elementwise left shift: [`Shl::shl`].
pub fn shl<T: Scalar + Shl<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a << b)
}

/// Elementwise right shift: [`Shr::shr`].
pub fn shr<T: Scalar + Shr<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::map_broadcast(|a: T, b: U| a >> b)
}
