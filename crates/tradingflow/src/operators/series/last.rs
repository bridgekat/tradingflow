use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar, SeriesView};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SeriesPort};

/// Operator signature for [`last_or`] and [`last`].
pub struct Last<T: Scalar, const N: usize> {
    fill: T,
}

impl<T: Scalar, const N: usize> Last<T, N> {
    pub fn new(fill: T) -> Self {
        Self { fill }
    }
}

impl<T: Scalar, const N: usize> Segment for Last<T, N> {
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = (T, Array<T, N>);

    fn init(self, series: SeriesView<'_, T, N>) -> Self::State {
        let out = if series.is_empty() {
            Array::full(series.extents(), self.fill.clone())
        } else {
            let (_, a) = series.at(series.range().end - 1);
            a.to_array()
        };
        (self.fill, out)
    }

    fn reset<'a, 'b: 'a>(
        _: SeriesView<'a, T, N>,
        (_, out): &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        out.view()
    }

    fn compute<'a, 'b: 'a>(
        series: SeriesView<'a, T, N>,
        (fill, out): &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        if series.is_empty() {
            out.data_mut().fill(fill.clone());
        } else {
            let (_, a) = series.at(series.range().end - 1);
            out.assign(a);
        }
        out.view()
    }
}

/// The most recent element of a series view as an array view. Filled with
/// `fill` if the series is empty.
pub fn last_or<T: Scalar, const N: usize>(
    fill: T,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Last::new(fill)
}

/// The most recent element of a series view as an array view. Filled with
/// NaN if the series is empty.
pub fn last<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Last::new(T::nan())
}
