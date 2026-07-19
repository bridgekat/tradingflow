use crate::data::{Array, Instant, Scalar, Series};
use crate::graph::{Pass, Port, Segment};
use crate::ports::{ArrayPass, SeriesPass};

pub struct Constant<V: Pass> {
    value: V::Owned,
}

impl<V: Pass> Constant<V> {
    pub fn new(value: V::Owned) -> Self {
        Self { value }
    }
}

impl<V: Pass> Segment for Constant<V> {
    type Inputs = ();
    type Outputs = Port<V>;
    type Context = Instant;
    type State = V::Owned;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut V::Owned) -> (bool, V::View<'a>) {
        (false, V::view(state))
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut V::Owned, _: &Instant) -> (bool, V::View<'a>) {
        (true, V::view(state))
    }
}

/// A constant [`Array`] cell.
pub fn const_array<T: Scalar, const N: usize>(value: Array<T, N>) -> Constant<ArrayPass<T, N>> {
    Constant::new(value)
}

/// A constant [`Series`] cell.
pub fn const_series<T: Scalar, const N: usize>(value: Series<T, N>) -> Constant<SeriesPass<T, N>> {
    Constant::new(value)
}
