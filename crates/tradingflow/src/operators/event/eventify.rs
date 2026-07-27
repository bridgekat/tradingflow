use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Scalar};
use crate::graph::{Interface, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`eventify`].
pub struct Eventify<T: Scalar + Float, const N: usize, S: Segment<Outputs = ArrayPort<T, N>>> {
    segment: S,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize, S: Segment<Outputs = ArrayPort<T, N>>> Eventify<T, N, S> {
    pub fn new(segment: S) -> Self {
        Self {
            segment,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize, S: Segment<Outputs = ArrayPort<T, N>>> Segment
    for Eventify<T, N, S>
{
    type Inputs = S::Inputs;
    type Outputs = ArrayPort<T, N>;
    type Context = S::Context;
    type State = (S::State, T);

    fn init(self, inputs: <S::Inputs as Interface>::Values<'_>) -> Self::State {
        (S::init(self.segment, inputs), T::nan())
    }

    fn reset<'a, 'b: 'a>(
        inputs: <S::Inputs as Interface>::Values<'a>,
        (state, nan): &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        let a = S::reset(inputs, state);
        ArrayView::full(a.extents(), nan)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <S::Inputs as Interface>::Values<'a>,
        (state, _): &'b mut Self::State,
        context: &Self::Context,
    ) -> ArrayView<'a, T, N> {
        S::compute(inputs, state, context)
    }
}

/// Turns a state-producing segment into an event-producing segment, which
/// resets its outputs to NaN each generation. Same as composing `segment`
/// with [`as_event`](super::as_event).
pub fn eventify<T: Scalar + Float, const N: usize, S: Segment<Outputs = ArrayPort<T, N>>>(
    segment: S,
) -> impl Segment<Inputs = S::Inputs, Outputs = ArrayPort<T, N>, Context = S::Context> {
    Eventify::new(segment)
}
