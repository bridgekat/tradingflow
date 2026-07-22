//! Graph fixtures shared by the per-operator trader tests.

use crate::data::{Array, Instant};
use crate::graph::typed::{Builder, NodeHandle, PortHandle};
use crate::operators::constant::*;
use crate::ports::ArrayPass;

pub(super) fn arr(v: &[f64]) -> Array<f64, 1> {
    Array::from_parts([v.len()], v.into())
}

/// Push a rank-1 array [`array_cell`] of `v`; return the source handle
/// (for `state_mut`) and its `ArrayPort` view handle (for wiring).
#[allow(clippy::type_complexity)]
pub(super) fn src(
    b: &mut Builder<Instant>,
    v: &[f64],
) -> (
    NodeHandle<ConstArray<f64, 1>>,
    PortHandle<ArrayPass<f64, 1>>,
) {
    b.source(const_array(arr(v)))
}
