use futures::stream::Stream;

use crate::data::{Array, ArrayView, Instant, Scalar, Series};
use crate::graph::{Event, Source};
use crate::ports::{ArrayPort, SignalPort};

/// Source signature for [`array_series`].
pub struct ArraySeries<T: Scalar, const N: usize> {
    series: Series<T, N>,
    empty: Array<T, N>,
}

impl<T: Scalar, const N: usize> ArraySeries<T, N> {
    pub fn new(series: Series<T, N>, empty: Array<T, N>) -> Self {
        Self { series, empty }
    }
}

impl<T: Scalar, const N: usize> Source for ArraySeries<T, N> {
    type Instant = Instant;
    type Payload = Array<T, N>;
    type Outputs = (SignalPort<0>, ArrayPort<T, N>);
    type State = Array<T, N>;

    fn size_hint(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn init(
        self,
    ) -> (
        Array<T, N>,
        impl Stream<Item = Event<Array<T, N>, Instant>> + 'static,
    ) {
        let it = self
            .series
            .into_iter()
            .map(|(ts, payload)| Event::at(payload, ts));
        (self.empty, futures::stream::iter(it))
    }

    fn reset(state: &mut Array<T, N>) -> (ArrayView<'_, bool, 0>, ArrayView<'_, T, N>) {
        (ArrayView::scalar(&false), state.view())
    }

    fn output(state: &mut Array<T, N>) -> (ArrayView<'_, bool, 0>, ArrayView<'_, T, N>) {
        (ArrayView::scalar(&true), state.view())
    }

    fn write(payload: Array<T, N>, _: Instant, state: &mut Array<T, N>) -> usize {
        *state = payload;
        1
    }
}

/// Creates an array source from a time series.
///
/// The initial state is a placeholder `empty` array. Later events come from
/// `series`, each accompanied by a signal pulse.
pub fn array_series<T: Scalar, const N: usize>(
    series: Series<T, N>,
    empty: Array<T, N>,
) -> impl Source<Instant = Instant, Outputs = (SignalPort<0>, ArrayPort<T, N>)> {
    ArraySeries::new(series, empty)
}
