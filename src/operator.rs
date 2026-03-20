use super::{InputRefs, OutputRef};

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
    type State;

    /// Borrowed views of the operator's input containers.
    ///
    /// Must be either a reference to [`Observable`](crate::Observable), a
    /// reference to [`Series`](crate::Series), or a slice or tuple thereof.
    type Inputs<'a>: InputRefs<'a>;

    /// Mutable view of the operator's output container.
    ///
    /// Must be a mutable reference to [`Observable`](crate::Observable).
    type Output<'a>: OutputRef<'a>;

    /// Infer the output element shape from input element shapes.
    fn shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]>;

    /// Create the initial runtime state.
    fn init(self) -> Self::State;

    /// Write the output from the inputs and the current state,
    /// or return `false` if no output is produced.
    fn compute(state: &mut Self::State, inputs: Self::Inputs<'_>, output: Self::Output<'_>)
    -> bool;
}
