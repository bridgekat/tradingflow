use num_traits::Float;

use super::Windowed;
use super::mvar::mvar;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::num::{UnaryFn, sqrt};
use crate::operators::rolling::RollingVariance;

/// Rolling standard deviation of the last `n` ticks (variance → square root,
/// fused): `mstd(n) @ x`. Self-recording.
pub fn mstd<T: Scalar + Float, const N: usize>(
    n: usize,
) -> Comp<Windowed<T, N, RollingVariance<T, N>>, UnaryFn<T, N>> {
    Comp(mvar(n), sqrt())
}
