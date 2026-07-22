//! Named constructors — comparison (`Predicate` / `Compare`, i.e. `U = bool`).

use num_traits::Float;

use super::map::{Compare, CompareFn, Predicate, PredicateFn};
use crate::data::Scalar;

macro_rules! define_compare {
    ($(#[$meta:meta])* $name:ident, |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + PartialOrd, const N: usize>() -> CompareFn<T, N> {
            Compare::new(|$a, $b| $body)
        }
    };
}

define_compare!(/// Element-wise `a > b`; a `NaN` operand yields `false`.
    greater, |a, b| a > b);
define_compare!(/// Element-wise `a >= b`; a `NaN` operand yields `false`.
    greater_equal, |a, b| a >= b);
define_compare!(/// Element-wise `a < b`; a `NaN` operand yields `false`.
    less, |a, b| a < b);
define_compare!(/// Element-wise `a <= b`; a `NaN` operand yields `false`.
    less_equal, |a, b| a <= b);
define_compare!(/// Element-wise `a == b`; a `NaN` operand yields `false`.
    equal, |a, b| a == b);
define_compare!(/// Element-wise `a != b`; a `NaN` operand yields `true`.
    not_equal, |a, b| a != b);

macro_rules! define_predicate {
    ($(#[$meta:meta])* $name:ident, |$x:ident, $v:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + PartialOrd, const N: usize>(
            $v: T,
        ) -> Predicate<T, N, impl Fn(T) -> bool + Send + Sync> {
            Predicate::new(move |$x: T| $body)
        }
    };
}

define_predicate!(/// Element-wise `x > v`; `NaN` yields `false`.
    greater_than, |x, v| x > v);
define_predicate!(/// Element-wise `x >= v`; `NaN` yields `false`.
    at_least, |x, v| x >= v);
define_predicate!(/// Element-wise `x < v`; `NaN` yields `false`.
    less_than, |x, v| x < v);
define_predicate!(/// Element-wise `x <= v`; `NaN` yields `false`.
    at_most, |x, v| x <= v);
define_predicate!(/// Element-wise `x == v`; `NaN` yields `false`.
    equal_to, |x, v| x == v);
define_predicate!(/// Element-wise `x != v`; `NaN` yields `true`.
    not_equal_to, |x, v| x != v);

/// Element-wise `x.is_finite()` — test for missing data explicitly, rather than
/// relying on an ordering comparison to filter `NaN`s.
pub fn is_finite<T: Scalar + Float, const N: usize>() -> PredicateFn<T, N> {
    Predicate::new(|x: T| x.is_finite())
}

/// Element-wise `x.is_nan()`.
pub fn is_nan<T: Scalar + Float, const N: usize>() -> PredicateFn<T, N> {
    Predicate::new(|x: T| x.is_nan())
}
