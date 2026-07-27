//! Graph fixtures shared by the per-operator trader tests.

use crate::data::{Array, Instant};
use crate::graph::typed::{Builder, NodeHandle, PortHandle, Segment};
use crate::operators::{array, event};
use crate::ports::ArrayPass;

pub(super) fn arr(v: &[f64]) -> Array<f64, 1> {
    Array::from_parts([v.len()], v.into())
}

/// Push a pokeable rank-1 array cell of `v` re-badged as an event array
/// (each poke is one event batch; the edge resets to NaN between pokes);
/// return the source handle (for `state_mut`) and its `ArrayPort` view handle
/// (for wiring).
#[allow(clippy::type_complexity)]
pub(super) fn src(
    b: &mut Builder<Instant>,
    v: &[f64],
) -> (
    NodeHandle<impl Segment<State = Array<f64, 1>> + 'static>,
    PortHandle<ArrayPass<f64, 1>>,
) {
    let (h, raw) = b.source(array::constant(arr(v)));
    (h, b.segment(event::as_event(), raw))
}
