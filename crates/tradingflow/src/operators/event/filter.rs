use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Operator signature for [`filter`].
pub struct Filter<T: Scalar + Float, const N: usize, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    predicate: F,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize, F> Filter<T, N, F>
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

impl<T: Scalar + Float, const N: usize, F> Segment for Filter<T, N, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = (F, T);

    fn init(self, _: ArrayView<'_, T, N>) -> Self::State {
        (self.predicate, T::nan())
    }

    fn reset<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        (_, nan): &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        ArrayView::full(a.extents(), nan)
    }

    fn compute<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        (f, nan): &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        if f(a) {
            a
        } else {
            ArrayView::full(a.extents(), nan)
        }
    }
}

/// Passes an event array through only when `predicate` holds.
pub fn filter<T: Scalar + Float, const N: usize>(
    predicate: impl FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Filter::new(predicate)
}
