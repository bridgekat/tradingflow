use futures::stream::Stream;

use super::Event;
use crate::typed::Pass;

/// A stream of events writing into a source node.
pub trait Source
where
    for<'a> <Self::Pass as Pass>::View<'a>: Copy + Send + Sync,
{
    /// Event timestamp type.
    type Instant: Send + 'static;
    /// Event payload type.
    type Payload: Send + 'static;
    /// Stored value and passing policy.
    type Pass: Pass;

    /// Estimated total number of events this source will emit over its
    /// lifetime, or `None` for unbounded/unknown.
    fn size_hint(&self) -> Option<usize> {
        None
    }

    /// Typed initialization function.
    /// Returns the initial placeholder state, the event stream and the writer.
    #[allow(clippy::type_complexity)]
    fn init(
        self,
    ) -> (
        <Self::Pass as Pass>::Owned,
        impl Stream<Item = Event<Self::Instant, Self::Payload>> + Send + 'static,
        impl FnMut(Self::Instant, Self::Payload, &mut <Self::Pass as Pass>::Owned) -> usize
        + Send
        + 'static,
    );
}
