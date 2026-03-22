use super::types::InputKinds;

/// An operator that reads from typed inputs and writes into a typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Operator::init) consumes the operator spec together with
///    initial input references and a timestamp, producing the runtime
///    [`State`](Operator::State) and the initial
///    [`Output`](Operator::Output) value.
/// 2. [`compute`](Operator::compute) is called on each flush tick with
///    mutable state, immutable input references, a mutable output
///    reference, and the current timestamp.
pub trait Operator: Send + 'static {
    /// Mutable runtime state, created by [`init`](Operator::init).
    type State: Send + 'static;

    /// The operator's input types (e.g. `(ArrayD<f64>, ArrayD<f64>)`).
    type Inputs: InputKinds + ?Sized;

    /// The operator's output type.
    type Output: Send + 'static;

    /// Consume the operator spec and produce initial state and output.
    ///
    /// `inputs` are the current values of the input nodes at registration
    /// time.  `timestamp` is `i64::MIN` (reserved for future use).
    fn init(
        self,
        inputs: <Self::Inputs as InputKinds>::Refs<'_>,
        timestamp: i64,
    ) -> (Self::State, Self::Output);

    /// Compute the output from inputs and current state.
    ///
    /// Returns `true` if a value was produced.  If `false`, the output
    /// is considered unchanged and downstream propagation is skipped.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputKinds>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: i64,
    ) -> bool;
}
