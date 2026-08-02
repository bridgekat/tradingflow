use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise boolean and.
pub fn and<const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|&a, &b| a && b)
}

/// Elementwise boolean or.
pub fn or<const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|&a, &b| a || b)
}

/// Elementwise choice: `if cond { a } else { b }`.
pub fn choose<T: Scalar, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<bool, N>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::ternary_map(|&cond, a: &T, b: &T| if cond { a.clone() } else { b.clone() })
}

/// Elementwise indicator: `if cond { a } else { b }` for constant `a` and `b`.
pub fn indicator<T: Scalar, const N: usize>(
    a: T,
    b: T,
) -> impl Operator<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |&cond| if cond { a.clone() } else { b.clone() })
}
