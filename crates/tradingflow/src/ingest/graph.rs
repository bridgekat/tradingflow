//! The scenario builder and the running session.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::graph::{Builder, Graph, Handle, Pool, RefPort, RefSource};

use super::clock::{Clock, WallClock};
use super::feed::{Feed, LazyFeed, StreamFeed};
use super::queue::Queue;
use super::source::EventSource;
use crate::Instant;

/// The strategy graph builder: a [`Builder`] over the TAI [`Instant`]
/// context, coupled with an ingest [`Queue`] so a source cell and the feed
/// that writes it register in one call ([`add_source`](Self::add_source)); it
/// also owns the event counter the finished [`Session`] inherits.
///
/// Construct it as `Scenario::new(WallClock)`; the event time before the first
/// batch is [`Instant::MIN`] (a floor; see the [module docs](super)). The clock
/// is generic only for tests; the default is the real [`WallClock`].
///
/// The builder derefs to the inner [`Builder`]: every segment
/// registers through the inherited [`push`](Builder::push) applied to
/// an [`operators`](crate::operators) constructor, and externally-poked
/// constant cells through [`push_source`](Builder::push_source)
/// (mutate via [`state_mut`](Graph::state_mut), then
/// [`stabilize`](Session::stabilize) — stamping the generation first with
/// [`context_mut`](Graph::context_mut) if it must carry an event
/// time). `build()` produces a [`Session`].
pub struct Scenario<C: Clock = WallClock> {
    graph: Builder<Instant>,
    queue: Queue<Graph<Instant>, C>,
    num_events: Arc<AtomicUsize>,
    total_num_events: Option<usize>,
}

impl<C: Clock> Scenario<C> {
    /// An empty builder over an explicit wall clock.
    ///
    /// The event-time context starts at [`Instant::MIN`] — the floor at or
    /// below every event the run can produce, so the context is non-decreasing
    /// across the run. It is only ever observed by an operator that reads time
    /// on the build call — which the `init` flag already identifies.
    pub fn new(clock: C) -> Self {
        Self {
            graph: Builder::new(Instant::MIN),
            queue: Queue::new(clock),
            num_events: Arc::new(AtomicUsize::new(0)),
            total_num_events: Some(0),
        }
    }

    /// Register an [`EventSource`]. Its cell is a `RefSource` node holding
    /// [`EventSource::initial`]; the source's stream is materialized lazily at
    /// the first [`Session::step`], and its events are merged in timestamp
    /// order and applied to the cell via [`EventSource::write`].
    pub fn add_source<S: EventSource>(&mut self, source: S) -> Handle<RefPort<S::Output>> {
        let handle = self.graph.push_source(RefSource::new(source.initial()));
        // Accumulate the progress estimate before the source is moved into the
        // feed; a single un-estimable source makes the whole total unknown.
        let est = source.total_num_events();
        self.total_num_events = match (self.total_num_events, est) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let counter = Arc::clone(&self.num_events);
        // The feed is lazy: `init` (which may spawn producer tasks) runs on
        // the driving task at the first step, not here.
        self.queue.add_feed(LazyFeed::new(move || {
            let (stream, mut state) = source.init();
            let feed: Box<dyn Feed<Graph<Instant>>> = Box::new(StreamFeed::new(
                stream,
                move |graph: &mut Graph<Instant>, ts, event| {
                    let n = S::write(&mut state, event, graph.state_mut(handle), ts);
                    counter.fetch_add(n, Ordering::Relaxed);
                },
            ));
            feed
        }));
        *handle
    }
}

impl<C: Clock> Deref for Scenario<C> {
    type Target = Builder<Instant>;

    fn deref(&self) -> &Builder<Instant> {
        &self.graph
    }
}

impl<C: Clock> DerefMut for Scenario<C> {
    fn deref_mut(&mut self) -> &mut Builder<Instant> {
        &mut self.graph
    }
}

/// A live execution: a [`Graph`] plus its worker [`Pool`] and a
/// [`Queue`] of event feeds. Built by [`Scenario::build`].
///
/// Per event the queue writes into a graph source node, and per completed
/// batch one `stabilize` runs — so the graph stabilizes at most once per
/// timestamp. See [`Queue`] for the merge semantics. The typed graph's
/// context is the event time [`Instant`] — set to the batch timestamp after
/// the batch's writes, before its stabilize, so time-stamping operators
/// (`type Context = Instant`) observe event time. The session also owns the
/// cumulative event counter fed by [`EventSource`] writes
/// ([`num_events`](Self::num_events)).
///
/// Derefs to the inner [`Graph`], so slot reads
/// ([`view`](Graph::view) / [`ref_view`](Graph::ref_view)),
/// source-cell pokes ([`state_mut`](Graph::state_mut)) and the event
/// time ([`context`](Graph::context)) are inherited. (The inherent
/// [`stabilize`](Self::stabilize) uses the owned pool and shadows the inner
/// pool-taking one.)
///
/// [`run`](Self::run) is the simple loop: apply a batch, stabilize, repeat.
/// Callers that need per-batch work between a batch's writes and its
/// stabilize — poking other source nodes, custom scheduling — drive it
/// manually: [`step`](Self::step) applies one batch's writes and advances the
/// time context *without* stabilizing, then the caller finishes the
/// generation with [`stabilize`](Self::stabilize).
pub struct Session<C: Clock = WallClock> {
    graph: Graph<Instant>,
    pool: Pool,
    queue: Queue<Graph<Instant>, C>,
    num_events: Arc<AtomicUsize>,
    total_num_events: Option<usize>,
}

impl<C: Clock> Scenario<C> {
    /// Finalize into a runnable [`Session`] with a single-threaded pool.
    pub fn build(self) -> Session<C> {
        self.build_with_threads(0)
    }

    /// Like [`build`](Self::build) but with `n` extra worker threads in the
    /// pool.
    pub fn build_with_threads(self, n: usize) -> Session<C> {
        let Scenario {
            graph,
            queue,
            num_events,
            total_num_events,
        } = self;

        Session {
            graph: graph.build(),
            pool: Pool::new(n),
            queue,
            num_events,
            total_num_events,
        }
    }
}

impl<C: Clock> Session<C> {
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut Pool {
        &mut self.pool
    }

    pub fn queue(&self) -> &Queue<Graph<Instant>, C> {
        &self.queue
    }

    pub fn queue_mut(&mut self) -> &mut Queue<Graph<Instant>, C> {
        &mut self.queue
    }

    /// Cumulative logical event count across all [`EventSource`] feeds.
    pub fn num_events(&self) -> usize {
        self.num_events.load(Ordering::Relaxed)
    }

    /// Sum of every source's [`EventSource::total_num_events`], or `None` if
    /// any source couldn't estimate (or feeds were registered raw). Pair it
    /// with [`num_events`](Self::num_events) to drive a progress bar.
    /// Advisory.
    pub fn total_num_events(&self) -> Option<usize> {
        self.total_num_events
    }

    /// Advance by one batch: apply its event writes and set the event-time
    /// context to the batch timestamp, returning it — or `None` once every
    /// feed is exhausted. The graph is left **un-stabilized**: do any
    /// per-batch mutation, then call [`stabilize`](Self::stabilize) before
    /// reading outputs.
    pub async fn step(&mut self) -> Option<Instant> {
        let t = self.queue.step(&mut self.graph).await?;
        *self.graph.context_mut() = t;
        Some(t)
    }

    /// Recompute the dirty cone after a batch's writes (and any manual pokes),
    /// on the owned pool.
    pub fn stabilize(&mut self) {
        self.graph.stabilize(&mut self.pool);
    }

    /// Replay every feed to exhaustion, invoking `on_stable(&session, batch_ts)`
    /// once after each batch's stabilize (read outputs via the deref'd
    /// [`view`](Graph::view) / [`ref_view`](Graph::ref_view),
    /// progress via [`num_events`](Self::num_events)).
    ///
    /// A live (never-exhausting) feed loops forever. For a shutdown path,
    /// cancel the future between batches (e.g. race it against a shutdown
    /// signal with `select!` — the queue's state lives in `self`, so a
    /// cancelled `run` leaves a coherent, resumable session), or drive
    /// [`step`](Self::step) yourself.
    pub async fn run(&mut self, mut on_stable: impl FnMut(&Self, Instant)) {
        while let Some(t) = self.step().await {
            self.stabilize();
            on_stable(self, t);
        }
    }
}

impl<C: Clock> Deref for Session<C> {
    type Target = Graph<Instant>;

    fn deref(&self) -> &Graph<Instant> {
        &self.graph
    }
}

impl<C: Clock> DerefMut for Session<C> {
    fn deref_mut(&mut self) -> &mut Graph<Instant> {
        &mut self.graph
    }
}
