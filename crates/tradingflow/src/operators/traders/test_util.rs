//! Graph fixtures shared by the per-operator trader tests.

use crate::data::{Array, Instant};
use crate::graph::typed::{Builder, NodeHandle, Segment};
use crate::operators::{array, clock};
use crate::ports::{ArrayPortHandle, ClockPortHandle};

pub(super) fn arr(v: &[f64]) -> Array<f64, 1> {
    Array::from_parts([v.len()], v.into())
}

/// Push a pokeable rank-1 array cell of `v` paired with a derived clock
/// (each poke is one clock signal); return the source handle (for `state_mut`)
/// and the `(clock, values)` stream handles (for wiring). State-only
/// consumers wire just the values handle.
#[allow(clippy::type_complexity)]
pub(super) fn src(
    b: &mut Builder<Instant>,
    v: &[f64],
) -> (
    NodeHandle<impl Segment<State = Array<f64, 1>> + 'static>,
    (ClockPortHandle, ArrayPortHandle<f64, 1>),
) {
    let (h, raw) = b.source(array::constant(arr(v)));
    let clock = b.segment(clock::always(), raw);
    (h, (clock, raw))
}
