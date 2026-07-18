//! The scenario builder and the running session.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Clock, Queue, Source, StreamFeed};
use crate::typed::{HandlesInterface, InterfaceHandles, Pass, Port, PortHandle, Segment};

struct Store<I: Clone + Ord + Send + Sync + 'static, V: Pass> {
    value: V::Owned,
    _phantom: PhantomData<fn() -> I>,
}

impl<I: Clone + Ord + Send + Sync + 'static, V: Pass> Store<I, V> {
    pub fn new(value: V::Owned) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<I: Clone + Ord + Send + Sync + 'static, V: Pass> Segment for Store<I, V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Inputs = ();
    type Outputs = Port<V>;
    type Context = I;
    type State = V::Owned;

    fn init(self) -> Self::State {
        self.value
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        _: &I,
        state: &'b mut V::Owned,
        is_first_run: bool,
    ) -> (bool, V::View<'a>) {
        let owned: &'a V::Owned = state;
        (!is_first_run, V::borrow(owned))
    }
}

pub struct Builder<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Builder<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    size_hint: Option<usize>,
}

impl<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> Builder<I, C> {
    pub fn new(clock: C) -> Self {
        Self {
            inner: crate::typed::Builder::new(clock.min()),
            queue: Queue::new(clock),
            num_events: Arc::new(AtomicUsize::new(0)),
            size_hint: Some(0),
        }
    }

    /// Adds a source stream to the graph.
    pub fn source<T>(&mut self, source: T) -> PortHandle<T::Pass>
    where
        T: Source<Instant = I>,
    {
        self.size_hint = match (self.size_hint, source.size_hint()) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let (default, stream, mut write) = source.init();
        let (cell, handle) = self.inner.source(Store::new(default));
        let num_events = self.num_events.clone();
        self.queue.add_feed(StreamFeed::new(
            stream,
            move |ts, event, graph: &mut crate::typed::Graph<I>| {
                let n = write(ts, event, graph.state_mut(cell));
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
            queue,
            num_events,
            size_hint,
        }
    }
}

pub struct Graph<I: Clone + Ord + Send + Sync + 'static, C: Clock<I>> {
    inner: crate::typed::Graph<I>,
    queue: Queue<I, C, crate::typed::Graph<I>>,
    num_events: Arc<AtomicUsize>,
    size_hint: Option<usize>,
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

    pub fn size_hint(&self) -> Option<usize> {
        self.size_hint
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
