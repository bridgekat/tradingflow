use futures::stream::Stream;

use super::Event;
use crate::typed::Pass;

pub trait Source: Send + 'static
where
    for<'a> <Self::Pass as Pass>::View<'a>: Copy + Send + Sync,
{
    /// Event timestamp type.
    type Instant: Send + 'static;
    /// Event payload type.
    type Payload: Send + 'static;
    /// Stored value and passing policy.
    type Pass: Pass;

    /// The value the source cell holds before any event arrives.
    fn initial(&self) -> <Self::Pass as Pass>::Owned;

    /// Build the event stream and the writer.
    #[allow(clippy::type_complexity)]
    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Self::Instant, Self::Payload>> + Send + 'static,
        impl FnMut(Self::Instant, Self::Payload, &mut <Self::Pass as Pass>::Owned) -> usize
        + Send
        + 'static,
    );

    /// Estimated total number of events this source will emit over its
    /// lifetime, or `None` for unbounded/unknown.
    fn total_num_events(&self) -> Option<usize> {
        None
    }
}
