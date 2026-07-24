use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise AND.
pub fn and<const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|&a, &b| a && b)
}

/// Elementwise OR.
pub fn or<const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::binary_map(|&a, &b| a || b)
}

/// Elementwise choice: select between two arrays under a boolean mask.
pub fn choose<T: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::ternary_map(|&cond, a: &T, b: &T| if cond { a.clone() } else { b.clone() })
}

/// Elementwise indicator: select between two constants under a boolean mask.
pub fn indicator<T: Scalar, const N: usize>(
    a: T,
    b: T,
) -> impl Segment<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |&cond| if cond { a.clone() } else { b.clone() })
}
