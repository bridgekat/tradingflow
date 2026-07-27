use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Operator signature for [`constant`] etc.
pub struct Constant<T: Scalar, const N: usize> {
    value: Array<T, N>,
}

impl<T: Scalar, const N: usize> Constant<T, N> {
    pub fn new(value: Array<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for Constant<T, N> {
    type Inputs = ();
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn reset<'a, 'b: 'a>(_: (), state: &'b mut Array<T, N>) -> ArrayView<'a, T, N> {
        state.view()
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut Array<T, N>, _: &Instant) -> ArrayView<'a, T, N> {
        state.view()
    }
}

/// A constant array cell.
pub fn constant<T: Scalar, const N: usize>(
    value: Array<T, N>,
) -> impl Segment<Inputs = (), Outputs = ArrayPort<T, N>, Context = Instant, State = Array<T, N>> {
    Constant::new(value)
}

/// A constant rank-0 array holding one scalar: [`Array::scalar`].
pub fn scalar<T: Scalar>(
    value: T,
) -> impl Segment<Inputs = (), Outputs = ArrayPort<T, 0>, Context = Instant, State = Array<T, 0>> {
    Constant::new(Array::scalar(value))
}

/// An array filled with `value`: [`Array::full`].
pub fn full<T: Scalar, const N: usize>(
    extents: [usize; N],
    value: T,
) -> impl Segment<Inputs = (), Outputs = ArrayPort<T, N>, Context = Instant, State = Array<T, N>> {
    Constant::new(Array::full(extents, value))
}

/// An array filled with `T::default()`: [`Array::zeros`].
pub fn zeros<T: Scalar, const N: usize>(
    extents: [usize; N],
) -> impl Segment<Inputs = (), Outputs = ArrayPort<T, N>, Context = Instant, State = Array<T, N>> {
    Constant::new(Array::zeros(extents))
}

/// An array from a row-major contiguous buffer: [`Array::from_parts`].
pub fn from_parts<T: Scalar, const N: usize>(
    extents: [usize; N],
    data: Box<[T]>,
) -> impl Segment<Inputs = (), Outputs = ArrayPort<T, N>, Context = Instant, State = Array<T, N>> {
    Constant::new(Array::from_parts(extents, data))
}
