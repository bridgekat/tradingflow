//! Stamped events and the feeds that carry them into the merge.

use std::pin::Pin;
use std::task::{Context, Poll, ready};

use futures::stream::Stream;

use super::{Event, Stamp};

/// A type-erased feed writing into a sink of type `S`.
pub trait Feed<I, S> {
    /// Polls for a new event, overwriting it into an internal buffer when
    /// ready, or leave the buffer empty if the feed is exhausted.
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()>;

    /// Returns the event stamp in the internal buffer, or [`None`] if the
    /// buffer is empty.
    fn stamp(&self) -> Option<Stamp<I>>;

    /// Writes to `sink` the event payload in the internal buffer, if any.
    /// This is allowed to invalidate the internal buffer.
    fn write(&mut self, instant: I, sink: &mut S) -> bool;
}

/// A feed constructed from a [`Stream`] of events. The sink type is pinned by
/// the write closure's [`Feed`] implementation, not the struct.
pub struct StreamFeed<I, T, F> {
    stream: Pin<Box<dyn Stream<Item = Event<I, T>>>>,
    buffer: Option<Event<I, T>>,
    write: F,
}

impl<I, T, F> StreamFeed<I, T, F> {
    pub fn new<S>(stream: S, write: F) -> Self
    where
        S: Stream<Item = Event<I, T>> + 'static,
    {
        Self {
            stream: Box::pin(stream),
            buffer: None,
            write,
        }
    }
}

impl<I, T, F, S> Feed<I, S> for StreamFeed<I, T, F>
where
    I: Clone,
    F: FnMut(I, T, &mut S) + 'static,
{
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.buffer = ready!(self.stream.as_mut().poll_next(cx));
        Poll::Ready(())
    }

    fn stamp(&self) -> Option<Stamp<I>> {
        self.buffer.as_ref().map(|e| e.stamp.clone())
    }

    fn write(&mut self, instant: I, sink: &mut S) -> bool {
        self.buffer
            .take()
            .and_then(|e| e.payload)
            .map(|p| (self.write)(instant, p, sink))
            .is_some()
    }
}

/// A feed materialized on its first poll — which happens on the driving task,
/// once the merge is actually running. Use this when constructing the inner
/// feed has effects that must not happen at registration time, e.g. spawning
/// producer tasks on an async runtime or opening connections.
pub struct LazyFeed<I, S: 'static> {
    init: Option<Box<dyn FnOnce() -> BoxFeed<I, S>>>,
    feed: Option<BoxFeed<I, S>>,
}

type BoxFeed<I, S> = Box<dyn Feed<I, S>>;

impl<I, S: 'static> LazyFeed<I, S> {
    pub fn new(init: impl FnOnce() -> Box<dyn Feed<I, S>> + 'static) -> Self {
        Self {
            init: Some(Box::new(init)),
            feed: None,
        }
    }
}

impl<I, S: 'static> Feed<I, S> for LazyFeed<I, S> {
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        let feed = self.feed.get_or_insert_with(|| self.init.take().unwrap()());
        feed.poll_next(cx)
    }

    fn stamp(&self) -> Option<Stamp<I>> {
        self.feed.as_ref().and_then(|f| f.stamp())
    }

    fn write(&mut self, instant: I, sink: &mut S) -> bool {
        self.feed.as_mut().is_some_and(|f| f.write(instant, sink))
    }
}
