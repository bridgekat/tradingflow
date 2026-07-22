use num_traits::Float;

use super::WithLagged;
use super::lag::lag;
use crate::data::Scalar;
use crate::graph::cb::{Comp, Fork, Id};
use crate::operators::num::{BinaryFn, subtract};

/// `n`-tick change `x − x₋ₙ` (the momentum shape): `change(n) @ x`.
/// Self-recording; `NaN` until the lag is available. The one-tick special
/// case, without a private record, is [`diff`](crate::operators::num::diff).
pub fn change<T: Scalar + Float, const N: usize>(n: usize) -> WithLagged<T, N, BinaryFn<T, N>> {
    Comp(Fork(Id::default(), lag(n)), subtract())
}
