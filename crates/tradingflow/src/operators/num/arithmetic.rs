//! Named constructors — arithmetic (`Unary` / `Binary`, i.e. `U = T`).

use std::ops;

use num_traits::{Float, Signed};

use super::map::{Binary, BinaryFn, Unary, UnaryFn};
use crate::data::Scalar;

macro_rules! define_unary {
    ($(#[$meta:meta])* $name:ident [$($bounds:tt)*], |$x:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + $($bounds)*, const N: usize>() -> UnaryFn<T, N> {
            Unary::new(|$x| $body)
        }
    };
}

macro_rules! define_binary {
    ($(#[$meta:meta])* $name:ident [$($bounds:tt)*], |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + $($bounds)*, const N: usize>() -> BinaryFn<T, N> {
            Binary::new(|$a, $b| $body)
        }
    };
}

define_unary!(/// Element-wise negation: `-a`.
    negate [ops::Neg<Output = T>], |x| -x);
define_unary!(/// Element-wise natural logarithm.
    log [Float], |x| x.ln());
define_unary!(/// Element-wise base-2 logarithm.
    log2 [Float], |x| x.log2());
define_unary!(/// Element-wise base-10 logarithm.
    log10 [Float], |x| x.log10());
define_unary!(/// Element-wise exponential.
    exp [Float], |x| x.exp());
define_unary!(/// Element-wise base-2 exponential.
    exp2 [Float], |x| x.exp2());
define_unary!(/// Element-wise square root.
    sqrt [Float], |x| x.sqrt());
define_unary!(/// Element-wise ceiling.
    ceil [Float], |x| x.ceil());
define_unary!(/// Element-wise floor.
    floor [Float], |x| x.floor());
define_unary!(/// Element-wise rounding.
    round [Float], |x| x.round());
define_unary!(/// Element-wise reciprocal: `1/x`.
    recip [Float], |x| x.recip());
define_unary!(/// Element-wise absolute value.
    abs [Signed], |x| x.abs());
define_unary!(/// Element-wise signum (−1, 0, or +1).
    sign [Signed], |x| x.signum());

define_binary!(/// Element-wise addition: `a + b`.
    add [ops::Add<Output = T>], |a, b| a + b);
define_binary!(/// Element-wise subtraction: `a - b`.
    subtract [ops::Sub<Output = T>], |a, b| a - b);
define_binary!(/// Element-wise multiplication: `a * b`.
    multiply [ops::Mul<Output = T>], |a, b| a * b);
define_binary!(/// Element-wise division: `a / b`.
    divide [ops::Div<Output = T>], |a, b| a / b);
define_binary!(/// Element-wise minimum (IEEE 754).
    min [Float], |a, b| a.min(b));
define_binary!(/// Element-wise maximum (IEEE 754).
    max [Float], |a, b| a.max(b));

/// Element-wise power: `x.powf(n)` (the one parameterized unary — its callable
/// captures the exponent, so it is not a plain function pointer).
pub fn pow<T: Scalar + Float, const N: usize>(n: T) -> Unary<T, N, impl Fn(T) -> T + Send + Sync> {
    Unary::new(move |x: T| x.powf(n))
}
