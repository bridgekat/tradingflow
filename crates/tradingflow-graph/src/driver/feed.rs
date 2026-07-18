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
