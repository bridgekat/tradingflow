use super::{Inputs, Output};

/// An operator that reads from one or more inputs and writes into an output.
///
/// Each input must be either [`Observable`](crate::Observable) or
/// [`Series`](crate::Series). The output must be an
/// [`Observable`](crate::Observable).
///
/// The [`Scenario`](crate::Scenario) will call [`Operator::init`] to create
/// the initial state, then call [`Operator::compute`] on the state at each
/// update.
pub trait Operator {
    /// Mutable runtime state, created by [`init`](Operator::init).
    type State: Send + 'static;

    /// The operator's input containers.
    ///
    /// Must be either an [`Observable`](crate::Observable), a
    /// [`Series`](crate::Series), or a slice or tuple thereof.
    type Inputs: Inputs + ?Sized;

    /// The operator's output container.
    ///
    /// Must be an [`Observable`](crate::Observable).
    type Output: Output;

    /// Infer the output element shape from input element shapes.
    fn shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]>;

    /// Provide the initial value for the output observable, used when the
    /// initial [`compute`](Operator::compute) does not produce a value.
    fn initial(&self, input_shapes: &[&[usize]]) -> Box<[<Self::Output as Output>::Scalar]>;

    /// Create the initial runtime state.
    fn init(self) -> Self::State;

    /// Write the output from the inputs and the current state,
    /// or return `false` if no output is produced.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as Inputs>::Refs<'_>,
        output: &mut Self::Output,
    ) -> bool;
}
