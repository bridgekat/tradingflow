use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort};

/// Operator signature for [`resample`].
pub struct Resample<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Resample<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Segment for Resample<T, N> {
    type Inputs = (ClockPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = T;

    fn init(self, _: (bool, ArrayView<'_, T, N>)) -> Self::State {
        T::nan()
    }

    fn reset<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        nan: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        ArrayView::full(a.extents(), nan)
    }

    fn compute<'a, 'b: 'a>(
        (clock, a): (bool, ArrayView<'a, T, N>),
        nan: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        if clock {
            a
        } else {
            ArrayView::full(a.extents(), nan)
        }
    }
}

/// Passes a state array through as an event array only when an additional
/// clock signal notifies.
pub fn sample<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    Resample::new()
}
