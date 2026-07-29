use futures::stream::Stream;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::{Event, Source};
use crate::ports::{ArrayPort, SignalPort};

/// Source signature for [`array_iter`].
pub struct ArrayIter<T: Scalar, const N: usize, I>
where
    I: Iterator<Item = (Instant, Array<T, N>)> + 'static,
{
    iter: I,
    empty: Array<T, N>,
}

impl<T: Scalar, const N: usize, I> ArrayIter<T, N, I>
where
    I: Iterator<Item = (Instant, Array<T, N>)> + 'static,
{
    pub fn new(iter: I, empty: Array<T, N>) -> Self {
        Self { iter, empty }
    }
}

impl<T: Scalar, const N: usize, I> Source for ArrayIter<T, N, I>
where
    I: Iterator<Item = (Instant, Array<T, N>)> + 'static,
{
    type Instant = Instant;
    type Payload = Array<T, N>;
    type Outputs = (SignalPort<0>, ArrayPort<T, N>);
    type State = Array<T, N>;

    fn size_hint(&self) -> Option<usize> {
        self.iter.size_hint().1
    }

    fn init(
        self,
    ) -> (
        Array<T, N>,
        impl Stream<Item = Event<Array<T, N>, Instant>> + 'static,
    ) {
        let it = self.iter.map(|(ts, payload)| Event::at(payload, ts));
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

/// Creates an array source from an iterator of timestamped arrays.
///
/// The initial state is a placeholder `empty` array. Later events come from
/// `iter`, each accompanied by a signal pulse.
pub fn array_iter<T: Scalar, const N: usize>(
    iter: impl Iterator<Item = (Instant, Array<T, N>)> + 'static,
    empty: Array<T, N>,
) -> impl Source<Instant = Instant, Outputs = (SignalPort<0>, ArrayPort<T, N>)> {
    ArrayIter::new(iter, empty)
}
