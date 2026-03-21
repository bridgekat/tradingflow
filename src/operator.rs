use super::store::ElementViewMut;
use super::types::{InputKinds, Scalar};

/// An operator that reads from [`Store`](crate::store::Store) inputs and
/// writes into an [`ElementViewMut`] output.
///
/// # Lifecycle
///
/// 1. At construction, the operator provides [`default`](Operator::default)
///    (shape + default value) for the output store, and
///    [`window_sizes`](Operator::window_sizes) declaring which input stores
///    must retain history.
/// 2. [`init`](Operator::init) consumes the operator to create the runtime
///    [`State`](Operator::State).
/// 3. [`compute`](Operator::compute) is called on each flush tick with
///    references to input stores and a mutable view of the output element.
pub trait Operator: Send + 'static {
    /// Mutable runtime state, created by [`init`](Operator::init).
    type State: Send + 'static;

    /// The operator's input stores (e.g. `(Store<f64>, Store<f64>)`).
    ///
    /// Must be a [`Store<T>`](crate::store::Store), a slice `[Store<T>]`,
    /// or a tuple of stores.
    type Inputs: InputKinds + ?Sized;

    /// The scalar type of the output store.
    type Scalar: Scalar;

    /// Declare which inputs require windowed (history-keeping) stores.
    ///
    /// The returned value has one minimum window size per input position.
    /// `1` indicates that only the current element is needed.
    /// `0` indicates that all history must be preserved.
    fn window_sizes(&self, input_shapes: &[&[usize]]) -> <Self::Inputs as InputKinds>::WindowSizes;

    /// Compute the output element shape and default value from input shapes.
    ///
    /// Returns `(output_shape, default_values)`.
    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[Self::Scalar]>);

    /// Create the initial runtime state.
    fn init(self) -> Self::State;

    /// Compute the output from the inputs and the current state.
    ///
    /// Returns `true` if a value was produced.  If `false`, the output
    /// store is left unchanged: the tentative element is rolled back for
    /// all window sizes (including `window = 1`, which double-buffers).
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputKinds>::Refs<'_>,
        output: ElementViewMut<'_, Self::Scalar>,
    ) -> bool;
}
