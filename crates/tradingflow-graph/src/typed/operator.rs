use super::{Interface, Segment};

/// A [`Segment`] that automatically handles notification flags.
pub trait Operator {
    /// Input tree (e.g. `(RefPort<f64>, RefPorts<f64>)`).
    type Inputs: Interface;
    /// Output tree (e.g. `(RefPort<f64>, RefPorts<f64>)`).
    type Outputs: Interface;
    /// Expected graph-level context. See [`Segment::Context`].
    type Context: Send + Sync + 'static;
    /// Mutable runtime state. See [`Segment::State`] for the `Send` bound.
    type State: Send + 'static;

    /// Typed state initialization function.
    fn init(self) -> Self::State;

    /// Typed compute function.
    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a>;

    /// Re-derive outputs when no input notified, WITHOUT mutating state (note
    /// `&State`, not `&mut`): it therefore cannot reallocate output storage,
    /// which is what keeps emitted output pointers valid for out-of-cone
    /// consumers across generations.
    ///
    /// An input-less operator (a source) can never observe an input notify, so
    /// the gate *always* routes it here; such an operator MUST set every output
    /// `notify = true` so an externally-poked value propagates downstream.
    fn passthrough<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a>;
}

impl<T: Operator> Segment for T {
    type Inputs = <Self as Operator>::Inputs;
    type Outputs = <Self as Operator>::Outputs;
    type Context = <Self as Operator>::Context;
    type State = <Self as Operator>::State;

    fn init(self) -> Self::State {
        <Self as Operator>::init(self)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        if is_first_run || <Self::Inputs as Interface>::any_notify(&inputs) {
            <Self as Operator>::compute(inputs, context, state, is_first_run)
        } else {
            <Self as Operator>::passthrough(inputs, context, state)
        }
    }
}
