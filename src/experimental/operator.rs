//! Operator trait for the wavefront execution model.
//!
//! Shadow of [`crate::Operator`] with two additions:
//! * `State: Clone` — state is cloned per parallel tick for stateless nodes.
//! * `is_stateful()` — drives vertical (temporal) dependency edges.

use crate::data::{InputTypes, Instant};

/// A synchronous computation node that reads typed inputs and writes a
/// typed output.  Compatible with the wavefront execution model.
pub trait Operator: 'static {
    /// Mutable runtime state.  Must be cloneable so stateless nodes can
    /// run multiple ticks concurrently on independent state copies.
    type State: Send + Clone + 'static;

    /// Input tree.  `Sized` only in the PoC.
    type Inputs: InputTypes + Sized;

    /// Output type.
    type Output: Send + Clone + 'static;

    /// Consume the spec and produce initial state and output.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output);

    /// Update the output from inputs and current state.
    ///
    /// Returns `true` if downstream propagation should occur.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool;

    /// Whether this operator mutates `State` in a way that depends on the
    /// previous tick.  Stateful nodes have a vertical dependency edge:
    /// `(self, t-1)` must complete before `(self, t)` can start.
    fn is_stateful() -> bool {
        false
    }
}
