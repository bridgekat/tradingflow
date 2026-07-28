use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise negation: [`Neg::neg`].
pub fn neg<T: Scalar + Neg<Output: Scalar>, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T::Output, N>, Context = Instant> {
    array::map(|x: &T| x.clone().neg())
}

/// Elementwise addition: [`Add::add`].
pub fn add<T: Scalar + Add<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().add(b.clone()))
}

/// Elementwise subtraction: [`Sub::sub`].
pub fn sub<T: Scalar + Sub<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().sub(b.clone()))
}

/// Elementwise multiplication: [`Mul::mul`].
pub fn mul<T: Scalar + Mul<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().mul(b.clone()))
}

/// Elementwise division: [`Div::div`].
pub fn div<T: Scalar + Div<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().div(b.clone()))
}

/// Elementwise remainder: [`Rem::rem`].
pub fn rem<T: Scalar + Rem<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().rem(b.clone()))
}

/// Elementwise bitwise not: [`Not::not`].
pub fn not<T: Scalar + Not<Output: Scalar>, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T::Output, N>, Context = Instant> {
    array::map(|x: &T| x.clone().not())
}

/// Elementwise bitwise and: [`BitAnd::bitand`].
pub fn bitand<T: Scalar + BitAnd<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().bitand(b.clone()))
}

/// Elementwise bitwise or: [`BitOr::bitor`].
pub fn bitor<T: Scalar + BitOr<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().bitor(b.clone()))
}

/// Elementwise bitwise exclusive or: [`BitXor::bitxor`].
pub fn bitxor<T: Scalar + BitXor<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().bitxor(b.clone()))
}

/// Elementwise left shift: [`Shl::shl`].
pub fn shl<T: Scalar + Shl<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().shl(b.clone()))
}

/// Elementwise right shift: [`Shr::shr`].
pub fn shr<T: Scalar + Shr<U, Output: Scalar>, U: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<U, N>),
    Outputs = ArrayPort<T::Output, N>,
    Context = Instant,
> {
    array::binary_map(|a: &T, b: &U| a.clone().shr(b.clone()))
}
