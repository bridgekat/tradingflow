use super::Interface;

/// A composable graph segment: can be a node or a subgraph.
pub trait Segment {
    /// Input tree (e.g. `(Port<f64>, Ports<f64>)`).
    type Inputs: Interface;
    /// Output tree (e.g. `(Port<f64>, Ports<f64>)`).
    type Outputs: Interface;
    /// Expected graph context.
    type Context: Sync;
    /// Mutable node state, must be completely owned.
    type State: Send + 'static;

    /// Typed state initialization function.
    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State;

    /// Typed output initialization function.
    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a>;

    /// Typed compute function.
    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a>;
}

/// A [`Segment`] that handles notification flags.
pub trait Operator {
    /// Input tree (e.g. `(Port<f64>, Ports<f64>)`).
    type Inputs: Interface;
    /// Output tree (e.g. `(Port<f64>, Ports<f64>)`).
    type Outputs: Interface;
    /// Expected graph-level context. See [`Segment::Context`].
    type Context: Sync;
    /// Mutable runtime state. See [`Segment::State`] for the `Send` bound.
    type State: Send + 'static;

    /// Typed state initialization function.
    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State;

    /// Typed output passthrough function.
    fn passthrough<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a>;

    /// Typed compute function.
    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a>;
}

impl<T: Operator> Segment for T {
    type Inputs = <Self as Operator>::Inputs;
    type Outputs = <Self as Operator>::Outputs;
    type Context = <Self as Operator>::Context;
    type State = <Self as Operator>::State;

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State {
        <Self as Operator>::init(self, inputs)
    }

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        <Self as Operator>::passthrough(inputs, state)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        if <Self::Inputs as Interface>::any_notify(&inputs) {
            <Self as Operator>::compute(inputs, state, context)
        } else {
            <Self as Operator>::passthrough(inputs, state)
        }
    }
}
