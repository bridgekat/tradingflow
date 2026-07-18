//! The scenario builder and the running session.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Clock, LazyFeed, Queue, Source, StreamFeed};
use crate::typed::{HandlesInterface, InterfaceHandles, Pass, PortHandle, Segment};

pub struct Builder<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Builder<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    total_num_events: Option<usize>,
}

impl<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> Builder<I, C> {
    pub fn new(clock: C) -> Self {
        Self {
            inner: crate::typed::Builder::new(clock.min()),
            queue: Queue::new(clock),
            num_events: Arc::new(AtomicUsize::new(0)),
            total_num_events: Some(0),
        }
    }

    /// Adds a source stream to the graph.
    pub fn source<T>(&mut self, source: T) -> PortHandle<T::Pass>
    where
        T: Source<Instant = I>,
    {
        self.total_num_events = match (self.total_num_events, source.total_num_events()) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let (cell, handle) = self
            .inner
            .source(crate::typed::Source::new(source.initial()));
        let num_events = self.num_events.clone();
        self.queue.add_feed(LazyFeed::new(move || {
            let (stream, mut write) = source.init();
            Box::new(StreamFeed::new(
                stream,
                move |ts, event, graph: &mut crate::typed::Graph<I>| {
                    let n = write(ts, event, graph.state_mut(cell));
                    num_events.fetch_add(n, Ordering::Relaxed);
                },
            ))
        }));
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
            total_num_events,
        } = self;

        Graph {
            inner: graph.build(),
            queue,
            num_events,
            total_num_events,
        }
    }
}

pub struct Graph<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Graph<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    total_num_events: Option<usize>,
}

impl<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> Graph<I, C> {
    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_>
    where
        for<'a> V::View<'a>: Copy,
    {
        self.inner.view(handle)
    }

    pub fn instant(&self) -> I {
        self.inner.context().clone()
    }

    pub fn num_events(&self) -> usize {
        self.num_events.load(Ordering::Relaxed)
    }

    pub fn total_num_events(&self) -> Option<usize> {
        self.total_num_events
    }

    pub async fn step(&mut self) -> Option<I> {
        let t = self.queue.step(&mut self.inner).await?;
        *self.inner.context_mut() = t.clone();
        Some(t)
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool) {
        self.inner.stabilize(pool);
    }

    pub async fn run(&mut self, pool: &mut crate::pool::Pool, mut on_stable: impl FnMut(&Self, I)) {
        while let Some(t) = self.step().await {
            self.stabilize(pool);
            on_stable(self, t);
        }
    }
}
