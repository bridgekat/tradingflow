use num_traits::AsPrimitive;

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise scalar type conversion: [`Into::into`].
pub fn into<T: Scalar + Into<U>, U: Scalar, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, N>, Context = Instant> {
    array::map(|x: &T| x.clone().into())
}

/// Elementwise scalar type conversion: [`AsPrimitive::as_`].
pub fn as_<T: Scalar + Copy + AsPrimitive<U>, U: Scalar + Copy, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, N>, Context = Instant> {
    array::map(|x: &T| x.as_())
}
