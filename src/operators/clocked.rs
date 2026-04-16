//! Clock-gated operator transformer.
//!
//! [`Clocked<O>`] wraps any [`Operator`] `O` — including those with
//! `!Sized` `Inputs` such as slice-input operators — and prepends a single
//! `Input<()>` clock input.  The inner operator runs only when the clock
//! produces; it reads the latest values from all its data inputs regardless
//! of whether they produced this cycle (time-series semantics).
//!
//! # Input layout
//!
//! `Clocked<O>::Inputs = (Input<()>, O::Inputs)`.  The clock occupies local
//! position 0; O's inputs occupy local positions 1‥.  Because `O::Inputs`
//! is the trailing field of the tuple, it may be `!Sized` (e.g.
//! `[Input<Array<T>>]` for a [`Stack`](crate::operators::Stack) operator).
//!
//! # Notify remapping
//!
//! The graph's flat position space covers positions 0‥(1 + |O|).  Position 0
//! is the clock; positions 1‥ belong to O's inputs.  [`Clocked::compute`]
//! uses [`Notify::skip_leading`] to build a zero-allocation inner `Notify`
//! whose positions are remapped to 0‥|O| before being forwarded to
//! `O::compute`.

use crate::data::{Input, InputTypes, Instant, Notify};
use crate::operator::Operator;

/// Wraps an operator so it only fires when a leading `Input<()>` clock
/// input produces.  All data inputs use time-series semantics (latest
/// value regardless of whether they produced this cycle).
///
/// Unlike putting the clock at the end, the leading position means
/// `O::Inputs` remains in trailing position and may be `?Sized`.
pub struct Clocked<O: Operator>(pub O);

impl<O: Operator> Clocked<O> {
    pub fn new(inner: O) -> Self {
        Self(inner)
    }
}

impl<O: Operator> Operator for Clocked<O> {
    type State = O::State;
    /// Clock `Input<()>` at position 0 followed by all of O's inputs.
    /// O::Inputs may be `?Sized` because it is the trailing field.
    type Inputs = (Input<()>, O::Inputs);
    type Output = O::Output;

    fn init(
        self,
        inputs: (&(), <O::Inputs as InputTypes>::Refs<'_>),
        timestamp: Instant,
    ) -> (O::State, O::Output) {
        self.0.init(inputs.1, timestamp)
    }

    fn compute(
        state: &mut O::State,
        inputs: (&(), <O::Inputs as InputTypes>::Refs<'_>),
        output: &mut O::Output,
        timestamp: Instant,
        notify: &Notify<'_>,
    ) -> bool {
        // Clock is at local position 0.  Only run the inner operator when
        // the clock ticks.  `produced()` yields local positions, so we
        // simply check for 0.
        if !notify.produced().any(|p| p == 0) {
            return false;
        }
        // Build a zero-allocation inner Notify that remaps positions 1.. to
        // 0..(num_inputs-1) for the wrapped operator.
        O::compute(state, inputs.1, output, timestamp, &notify.skip_leading(1))
    }
}
