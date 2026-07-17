//! Stamped events and the feeds that carry them into the merge.

use std::pin::Pin;
use std::task::{Context, Poll, ready};

use futures::stream::Stream;

use crate::data::Instant;

/// An explicit timestamp, or an implicit wall-clock time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stamp {
    Instant(Instant),
    Now,
}

/// An event from a stream.
///
/// Each event carries a frontier timestamp (non-decreasing in [`Stamp`]
/// ordering), and optionally a payload at that timestamp.
pub struct Event<D: Send + 'static> {
    pub stamp: Stamp,
    pub payload: Option<D>,
}

impl<D: Send + 'static> Event<D> {
    /// An explicitly-stamped event: `payload` at timestamp `t`.
    pub fn at(t: Instant, payload: D) -> Self {
        Self {
            stamp: Stamp::Instant(t),
            payload: Some(payload),
        }
    }

    /// An implicitly-stamped event: `payload` is stamped with the wall clock
    /// at receipt.
    pub fn now(payload: D) -> Self {
        Self {
            stamp: Stamp::Now,
            payload: Some(payload),
        }
    }

    /// A payload-less frontier advance: a promise that no event will arrive
    /// stamped strictly below `t`.
    pub fn frontier(t: Instant) -> Self {
        Self {
            stamp: Stamp::Instant(t),
            payload: None,
        }
    }
}

/// A type-erased feed writing into a sink of type `S`.
pub trait Feed<S>: Send {
    /// Polls for a new event, overwriting it into an internal buffer when
    /// ready, or leave the buffer empty if the feed is exhausted.
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()>;

    /// Returns the event stamp in the internal buffer, or [`None`] if the
    /// buffer is empty.
    fn stamp(&self) -> Option<Stamp>;

    /// Writes to `sink` the event payload in the internal buffer, if any.
    /// This is allowed to invalidate the internal buffer.
    fn write(&mut self, sink: &mut S, instant: Instant) -> bool;
}

/// A feed constructed from a [`Stream`] of events. The sink type is pinned by
/// the write closure's [`Feed`] implementation, not the struct.
pub struct StreamFeed<A, D>
where
    D: Send + 'static,
{
    stream: Pin<Box<dyn Stream<Item = Event<D>> + Send>>,
    buffer: Option<Event<D>>,
    write: A,
}

impl<A, D> StreamFeed<A, D>
where
    D: Send + 'static,
{
    pub fn new<St>(stream: St, write: A) -> Self
    where
        St: Stream<Item = Event<D>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
            buffer: None,
            write,
        }
    }
}

impl<A, D, S> Feed<S> for StreamFeed<A, D>
where
    D: Send + 'static,
    A: FnMut(&mut S, Instant, D) + Send + 'static,
{
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.buffer = ready!(self.stream.as_mut().poll_next(cx));
        Poll::Ready(())
    }

    fn stamp(&self) -> Option<Stamp> {
        self.buffer.as_ref().map(|e| e.stamp)
    }

    fn write(&mut self, sink: &mut S, instant: Instant) -> bool {
        self.buffer
            .take()
            .and_then(|e| e.payload)
            .map(|p| (self.write)(sink, instant, p))
            .is_some()
    }
}

/// A feed materialized on its first poll — which happens on the driving task,
/// once the merge is actually running. Use this when constructing the inner
/// feed has effects that must not happen at registration time, e.g. spawning
/// producer tasks on an async runtime or opening connections.
pub struct LazyFeed<S: 'static> {
    init: Option<Box<dyn FnOnce() -> BoxFeed<S> + Send>>,
    feed: Option<BoxFeed<S>>,
}

type BoxFeed<S> = Box<dyn Feed<S>>;

impl<S: 'static> LazyFeed<S> {
    pub fn new(init: impl FnOnce() -> Box<dyn Feed<S>> + Send + 'static) -> Self {
        Self {
            init: Some(Box::new(init)),
            feed: None,
        }
    }
}

impl<S: 'static> Feed<S> for LazyFeed<S> {
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        let feed = self.feed.get_or_insert_with(|| self.init.take().unwrap()());
        feed.poll_next(cx)
    }

    // The buffer starts empty and is only ever filled by `poll_next`, so
    // delegating to a not-yet-materialized feed cannot be observed.

    fn stamp(&self) -> Option<Stamp> {
        self.feed.as_ref().and_then(|f| f.stamp())
    }

    fn write(&mut self, sink: &mut S, instant: Instant) -> bool {
        self.feed.as_mut().is_some_and(|f| f.write(sink, instant))
    }
}
