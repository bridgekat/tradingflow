use futures::stream::Stream;

use super::Event;
use crate::typed::Interface;

/// A stream of events writing into a source node.
pub trait Source {
    /// Event payload type.
    type Payload: Send + 'static;
    /// Event timestamp type.
    type Instant: Sync + 'static;
    /// Output tree (e.g. `(Port<f64>, Ports<f64>)`).
    type Outputs: Interface;
    /// Mutable node state, must be completely owned.
    type State: Send + 'static;

    /// Estimated total number of events this source will emit over its
    /// lifetime, or `None` for unbounded/unknown.
    fn size_hint(&self) -> Option<usize> {
        None
    }

    /// Typed state initialization function.
    fn init(
        self,
    ) -> (
        Self::State,
        impl Stream<Item = Event<Self::Payload, Self::Instant>> + 'static,
    );

    /// Typed reset function.
    fn reset(state: &mut Self::State) -> <Self::Outputs as Interface>::Values<'_>;

    /// Typed state update function.
    fn write(payload: Self::Payload, instant: Self::Instant, state: &mut Self::State) -> usize;

    /// Typed output function.
    fn output(state: &mut Self::State) -> <Self::Outputs as Interface>::Values<'_>;
}
