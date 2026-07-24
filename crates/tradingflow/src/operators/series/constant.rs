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

/// A constant series cell.
pub fn constant<T: Scalar, const N: usize>(
    value: Series<T, N>,
) -> impl Segment<Inputs = (), Outputs = SeriesPort<T, N>, Context = Instant, State = Series<T, N>>
{
    Constant::new(value)
}

/// An empty series: [`Series::new`].
pub fn empty<T: Scalar, const N: usize>(
    extents: [usize; N],
) -> impl Segment<Inputs = (), Outputs = SeriesPort<T, N>, Context = Instant, State = Series<T, N>>
{
    Constant::new(Series::new(extents))
}

/// A series from element extents, instants, and a row-major contiguous buffer
/// of packed elements: [`Series::from_parts`].
pub fn from_parts<T: Scalar, const N: usize>(
    extents: [usize; N],
    instants: Vec<Instant>,
    data: Vec<T>,
    base: usize,
) -> impl Segment<Inputs = (), Outputs = SeriesPort<T, N>, Context = Instant, State = Series<T, N>>
{
    Constant::new(Series::from_parts(extents, instants, data, base))
}
