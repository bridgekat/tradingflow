use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort, is_eventful};

/// Operator signature for [`as_clock`].
pub struct AsClock<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> AsClock<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Segment for AsClock<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ClockPort;
    type Context = Instant;
    type State = ();

    fn init(self, _: ArrayView<'_, T, N>) {}

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, T, N>, _: &'b mut ()) -> bool {
        false
    }

    fn compute<'a, 'b: 'a>(a: ArrayView<'a, T, N>, _: &'b mut (), _: &Instant) -> bool {
        is_eventful(a)
    }
}

/// Turns an event array into a clock signal, which emits `true` whenever the
/// array carries any event.
pub fn as_clock<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ClockPort, Context = Instant> {
    AsClock::new()
}
