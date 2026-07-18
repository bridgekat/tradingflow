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

    fn init(self) -> Self::State {
        self.value
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        _: &Instant,
        state: &'b mut V::Owned,
        is_first_run: bool,
    ) -> (bool, V::View<'a>) {
        (!is_first_run, V::view(state))
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
