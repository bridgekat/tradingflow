use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Operator signature for [`as_event`].
pub struct AsEvent<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> AsEvent<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Segment for AsEvent<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ArrayView<'_, T, N>) -> T {
        T::nan()
    }

    fn reset<'a, 'b: 'a>(a: ArrayView<'a, T, N>, nan: &'b mut T) -> ArrayView<'a, T, N> {
        ArrayView::full(a.extents(), nan)
    }

    fn compute<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        _: &'b mut T,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        a
    }
}

/// Passes a state array as an event array.
pub fn as_event<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    AsEvent::new()
}
