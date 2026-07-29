use futures::stream::Stream;

use crate::data::{ArrayView, Instant};
use crate::graph::{Event, Source};
use crate::ports::SignalPort;

/// Source signature for [`signal_iter`].
pub struct SignalIter<I>
where
    I: Iterator<Item = Instant> + 'static,
{
    iter: I,
}

impl<I> SignalIter<I>
where
    I: Iterator<Item = Instant> + 'static,
{
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> Source for SignalIter<I>
where
    I: Iterator<Item = Instant> + 'static,
{
    type Instant = Instant;
    type Payload = ();
    type Outputs = SignalPort<0>;
    type State = ();

    fn size_hint(&self) -> Option<usize> {
        self.iter.size_hint().1
    }

    fn init(self) -> ((), impl Stream<Item = Event<(), Instant>> + 'static) {
        let it = self.iter.map(|ts| Event::at((), ts));
        ((), futures::stream::iter(it))
    }

    fn reset(_: &mut ()) -> ArrayView<'_, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn output(_: &mut ()) -> ArrayView<'_, bool, 0> {
        ArrayView::scalar(&true)
    }

    fn write(_: (), _: Self::Instant, _: &mut ()) -> usize {
        1
    }
}

/// Creates a signal source from an iterator of timestamps.
pub fn signal_iter(
    iter: impl Iterator<Item = Instant> + 'static,
) -> impl Source<Instant = Instant, Outputs = SignalPort<0>> {
    SignalIter::new(iter)
}
