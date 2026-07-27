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

    /// Typed reset function.
    fn reset<'a, 'b: 'a>(
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
