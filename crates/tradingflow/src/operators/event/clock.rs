use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, UnitPort};

/// Operator signature for [`clock`].
pub struct Clock<T: Scalar, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize> Clock<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Segment for Clock<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = UnitPort;
    type Context = Instant;
    type State = ();

    fn init(self, _: (bool, ArrayView<'_, T, N>)) {}

    fn output<'a, 'b: 'a>((notify, _): (bool, ArrayView<'a, T, N>), _: &'b mut ()) -> (bool, ()) {
        (notify, ())
    }

    fn compute<'a, 'b: 'a>(
        (notify, _): (bool, ArrayView<'a, T, N>),
        _: &'b mut (),
        _: &Instant,
    ) -> (bool, ()) {
        (notify, ())
    }
}

/// Turns an input array into a clock signal, which notifies whenever the
/// input array is notified.
pub fn clock<T: Scalar, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = UnitPort, Context = Instant> {
    Clock::new()
}
