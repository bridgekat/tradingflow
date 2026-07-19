//! The scenario builder and the running session.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Clock, Queue, Source, StreamFeed};
use crate::typed::{HandlesInterface, Interface, InterfaceHandles, Pass, PortHandle, Segment};

/// The on-graph node for a source stream.
struct Store<T: Source> {
    state: T::State,
}

impl<T: Source> Store<T> {
    pub fn new(state: T::State) -> Self {
        Self { state }
    }
}

impl<T: Source> Segment for Store<T> {
    type Inputs = ();
    type Outputs = T::Outputs;
    type Context = T::Instant;
    type State = T::State;

    fn init(self, _: ()) -> Self::State {
        self.state
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T::State) -> <T::Outputs as Interface>::Values<'a> {
        T::output(state)
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut T::State,
        _: &T::Instant,
    ) -> <T::Outputs as Interface>::Values<'a> {
        T::output(state)
    }
}

/// Builder for the top-layer [`Graph`].
///
/// This is the default level of the public graph API.
pub struct Builder<I: Clone + Ord + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Builder<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    size_hint: Option<usize>,
}

impl<I: Clone + Ord + Sync + 'static, C: Clock<I>> Builder<I, C> {
    pub fn new(clock: C) -> Self {
        Self {
            inner: crate::typed::Builder::new(),
            queue: Queue::new(clock),
            num_events: Arc::new(AtomicUsize::new(0)),
            size_hint: Some(0),
        }
    }

    /// Adds a source stream to the graph.
    pub fn source<T>(&mut self, source: T) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        T: Source<Instant = I> + 'static,
        T::Outputs: InterfaceHandles,
    {
        self.size_hint = match (self.size_hint, source.size_hint()) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let (default, stream) = source.init();
        let (cell, handle) = self.inner.source(Store::<T>::new(default));
        let num_events = self.num_events.clone();
        self.queue.add_feed(StreamFeed::new(
            stream,
            move |payload, ts, graph: &mut crate::typed::Graph<I>| {
                let n = T::write(payload, ts, graph.state_mut(cell));
                num_events.fetch_add(n, Ordering::Relaxed);
            },
        ));
        handle
    }

    /// Adds a segment node to the graph.
    pub fn segment<T, H>(
        &mut self,
        segment: T,
        input_handles: H,
    ) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        T: Segment<Inputs = H::Interface, Context = I>,
        T::Outputs: InterfaceHandles,
    {
        self.inner.segment(segment, input_handles)
    }

    /// Finalize into a runnable [`Graph`].
    pub fn build(self) -> Graph<I, C> {
        let Builder {
            inner: graph,
            queue,
            num_events,
            size_hint,
        } = self;

        Graph {
            inner: graph.build(),
            instant: None,
            queue,
            num_events,
            size_hint,
        }
    }
}

/// The top-level wrapper over the typed layer [`Graph`](crate::typed::Graph)
/// and the event [`Queue`].
///
/// This is the default level of the public graph API.
pub struct Graph<I: Clone + Ord + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Graph<I>,
    instant: Option<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    size_hint: Option<usize>,
}

impl<I: Clone + Ord + Sync + 'static, C: Clock<I>> Graph<I, C> {
    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_> {
        self.inner.view(handle)
    }

    pub fn num_events(&self) -> usize {
        self.num_events.load(Ordering::Relaxed)
    }

    pub fn size_hint(&self) -> Option<usize> {
        self.size_hint
    }

    pub async fn step(&mut self) -> Option<I> {
        let t = self.queue.step(&mut self.inner).await;
        self.instant = t;
        self.instant.clone()
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool) {
        self.inner.stabilize(pool, self.instant.as_ref().unwrap());
    }

    pub async fn run(&mut self, pool: &mut crate::pool::Pool, mut on_stable: impl FnMut(&Self, I)) {
        while let Some(t) = self.step().await {
            self.stabilize(pool);
            on_stable(self, t);
        }
    }
}
