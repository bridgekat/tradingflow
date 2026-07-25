use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, UnitPort};

/// Operator signature for [`resample`].
pub struct Resample<T: Scalar, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize> Resample<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Segment for Resample<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, (_, a)): ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        a.to_array()
    }

    fn output<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((notified, _), (_, a)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if notified {
            out.assign(a);
            (true, out.view())
        } else {
            (false, out.view())
        }
    }
}

/// Passes an input array through only when an additional first input is
/// notified. The array's notification state itself is ignored.
pub fn resample<T: Scalar, const N: usize>()
-> impl Segment<Inputs = (UnitPort, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    Resample::new()
}
