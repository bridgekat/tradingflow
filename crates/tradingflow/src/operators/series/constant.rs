use crate::data::{Instant, Scalar, Series, SeriesView};
use crate::graph::Segment;
use crate::ports::SeriesPort;

/// Operator signature for [`constant`] etc.
pub struct Constant<T: Scalar, const N: usize> {
    value: Series<T, N>,
}

impl<T: Scalar, const N: usize> Constant<T, N> {
    pub fn new(value: Series<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for Constant<T, N> {
    type Inputs = ();
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = Series<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn reset<'a, 'b: 'a>(_: (), state: &'b mut Series<T, N>) -> SeriesView<'a, T, N> {
        state.view()
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Series<T, N>,
        _: &Instant,
    ) -> SeriesView<'a, T, N> {
        state.view()
    }
}

/// A constant series cell.
pub fn constant<T: Scalar, const N: usize>(
    value: impl Into<Series<T, N>>,
) -> impl Segment<Inputs = (), Outputs = SeriesPort<T, N>, Context = Instant, State = Series<T, N>>
{
    Constant::new(value.into())
}
