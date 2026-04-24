//! Adapter that bridges [`crate::Operator`] impls to
//! [`experimental::Operator`](crate::experimental::Operator).

use crate::data::InputTypes;
use crate::experimental;
use crate::Instant;

/// Wraps a [`crate::Operator`] and implements
/// [`experimental::Operator`].  Conservatively treats the operator
/// as **stateful** (vertical dependency).  For the PoC, use the
/// operators in [`super::operators`] directly for stateless variants.
pub struct WavefrontAdapter<O: crate::Operator>(pub O);

impl<O> experimental::Operator for WavefrontAdapter<O>
where
    O: crate::Operator,
    O::State: Clone,
    O::Output: Clone,
    O::Inputs: Sized,
{
    type State = O::State;
    type Inputs = O::Inputs;
    type Output = O::Output;

    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output) {
        self.0.init(inputs, timestamp)
    }

    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        O::compute(state, inputs, output, timestamp, produced)
    }

    fn is_stateful() -> bool {
        true // conservative: treat all adapted operators as stateful
    }
}
