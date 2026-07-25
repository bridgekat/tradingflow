use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`filter`].
pub struct Filter<T: Scalar, const N: usize, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    predicate: F,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize, F> Filter<T, N, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, F> Operator for Filter<T, N, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = (F, Array<T, N>);

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        (self.predicate, x.to_array())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        (_, out): &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        (f, out): &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if f(a) {
            out.assign(a);
            (true, out.view())
        } else {
            (false, out.view())
        }
    }
}

/// Passes an input array through only when `predicate` holds.
pub fn filter<T: Scalar, const N: usize>(
    predicate: impl FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Filter::new(predicate)
}
