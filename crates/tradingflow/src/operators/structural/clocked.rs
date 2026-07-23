//! `Clocked` — clock-gated wrapper.

use crate::graph::typed::{Interface, Segment};
use crate::ports::UnitPort;

/// Prepends a leading `UnitPort` clock input; runs the inner operator's compute
/// path only when the clock notifies, else the inner passthrough. The auto-gate
/// routes to `compute` on any input notify, but `compute` runs the inner
/// `compute` only on a clock tick and otherwise falls through to the inner
/// `passthrough` — so a data-notify-without-clock behaves the same in either
/// branch.
#[derive(Debug, Clone)]
pub struct Clocked<O> {
    inner: O,
}

impl<O> Clocked<O> {
    pub fn new(inner: O) -> Self {
        Self { inner }
    }
}

impl<O: Segment> Segment for Clocked<O> {
    type Inputs = (UnitPort, O::Inputs);
    type Outputs = O::Outputs;
    // Forwarded, not pinned: `Clocked` is a gate, so it stays as
    // context-agnostic as whatever it wraps.
    type Context = O::Context;
    type State = O::State;

    fn init(self, (_, rest): ((bool, ()), <O::Inputs as Interface>::Values<'_>)) -> O::State {
        O::init(self.inner, rest)
    }

    fn output<'a, 'b: 'a>(
        (_, rest): ((bool, ()), <O::Inputs as Interface>::Values<'a>),
        state: &'b mut O::State,
    ) -> <O::Outputs as Interface>::Values<'a> {
        O::output(rest, state)
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), rest): ((bool, ()), <O::Inputs as Interface>::Values<'a>),
        state: &'b mut O::State,
        context: &O::Context,
    ) -> <O::Outputs as Interface>::Values<'a> {
        if clock_fired {
            O::compute(rest, state, context)
        } else {
            O::output(rest, state)
        }
    }
}

/// Prepend a leading clock port to `inner`, running its compute path only when
/// the clock notifies.
pub fn clocked<O>(inner: O) -> Clocked<O> {
    Clocked::new(inner)
}
