use crate::data::{Instant, Scalar, Series, SeriesView};
use crate::graph::Segment;
use crate::ports::SeriesPort;

/// Homes a [`Series`] and lends a view of it on every tick.
pub struct ConstSeries<T: Scalar, const N: usize> {
    value: Series<T, N>,
}

impl<T: Scalar, const N: usize> ConstSeries<T, N> {
    pub fn new(value: Series<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for ConstSeries<T, N> {
    type Inputs = ();
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = Series<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut Series<T, N>) -> (bool, SeriesView<'a, T, N>) {
        (true, state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Series<T, N>,
        _: &Instant,
    ) -> (bool, SeriesView<'a, T, N>) {
        (true, state.view())
    }
}

/// A constant [`Series`] cell.
pub fn const_series<T: Scalar, const N: usize>(value: Series<T, N>) -> ConstSeries<T, N> {
    ConstSeries::new(value)
}
