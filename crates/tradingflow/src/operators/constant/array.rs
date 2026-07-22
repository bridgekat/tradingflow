use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Homes an [`Array`] and lends a view of it on every tick.
pub struct ConstArray<T: Scalar, const N: usize> {
    value: Array<T, N>,
}

impl<T: Scalar, const N: usize> ConstArray<T, N> {
    pub fn new(value: Array<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for ConstArray<T, N> {
    type Inputs = ();
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut Array<T, N>) -> (bool, ArrayView<'a, T, N>) {
        (true, state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        (true, state.view())
    }
}

/// A constant [`Array`] cell.
pub fn const_array<T: Scalar, const N: usize>(value: Array<T, N>) -> ConstArray<T, N> {
    ConstArray::new(value)
}
