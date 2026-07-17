//! The timestamp-ordered merge over event feeds.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::{Pin, pin};
use std::task::{Context, Poll, ready};

use futures::StreamExt;
use futures::future::Either;
use futures::stream::FuturesUnordered;

use super::clock::{Clock, WallClock};
use super::feed::{Feed, Stamp};
use crate::data::Instant;

/// The currently known frontier of a feed.
///
/// The derived `Ord` is the heap order: `None` (not yet heard from) is the
/// bottom and blocks everything, explicit stamps compare as instants,
/// `Stamp(Now)` sits above them (an implicit feed never blocks an explicit
/// batch, the wall clock covers it), and `Done` is the top and never blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Frontier {
    None,
    Stamp(Stamp),
    Done,
}

/// A feed that is being polled for the next event.
struct Active<S: 'static> {
    id: usize,
    feed: Option<Box<dyn Feed<S>>>,
}

impl<S: 'static> Active<S> {
    pub fn new(id: usize, feed: Box<dyn Feed<S>>) -> Self {
        Self {
            id,
            feed: Some(feed),
        }
    }
}

impl<S: 'static> Future for Active<S> {
    type Output = (usize, Box<dyn Feed<S>>);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let feed = this.feed.as_mut().unwrap();
        ready!(feed.poll_next(cx));
        Poll::Ready((this.id, this.feed.take().unwrap()))
    }
}

/// A timestamp-ordered merge stream of event feeds.
///
/// Each feed is a stream of events; each event executes one write against a
/// caller-supplied sink `S`, and each completed batch yields its timestamp
/// back to the caller (who runs the per-batch action — e.g. a graph
/// stabilize). The queue never owns the sink: [`step`](Self::step) borrows it
/// per call.
///
/// The queue merges all feeds' events in a global timestamp order and batches
/// together all events at the same timestamp; batches always have strictly
/// increasing timestamps.
///
/// With `k` feeds, each event is roughly processed in `O(log k)` time. However,
/// feeds emitting events with explicit timestamps must advance their own
/// frontiers periodically, in order for the queue to make progress. Events
/// with explicit timestamps are generally used for replaying "historical" data
/// rather than "real-time" events, in which case the next event is always
/// available, so the frontier is always advancing.
pub struct Queue<S: 'static, C: Clock = WallClock> {
    /// The wall clock source.
    clock: C,
    /// Feeds currently being polled for the next event.
    active: FuturesUnordered<Active<S>>,
    /// Parked feeds with buffered events waiting to be read.
    /// Complementary to `active`, except for exhausted feeds.
    parked: Vec<Option<Box<dyn Feed<S>>>>,
    /// Exact min-heap over parked feeds, keyed by buffered stamp.
    parked_heap: BinaryHeap<Reverse<(Instant, usize)>>,
    /// Per-feed frontiers. Each feed guarantees not to emit events strictly
    /// before its frontier.
    frontier: Vec<Frontier>,
    /// Lazy min-heap over frontiers, pushed on changes only.
    /// Entries can become stale as `frontier` advances.
    frontier_heap: BinaryHeap<Reverse<(Frontier, usize)>>,
    /// The timestamp of the current batch, if any.
    instant: Option<Instant>,
}

impl<S: 'static, C: Clock> Queue<S, C> {
    /// Create an empty queue over a (required) wall clock.
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            active: FuturesUnordered::new(),
            parked: Vec::new(),
            parked_heap: BinaryHeap::new(),
            frontier: Vec::new(),
            frontier_heap: BinaryHeap::new(),
            instant: None,
        }
    }

    /// Register a feed.
    pub fn add_feed(&mut self, feed: impl Feed<S> + 'static) {
        let id = self.parked.len();
        self.active.push(Active::new(id, Box::new(feed)));
        self.parked.push(None);
        self.frontier.push(Frontier::None);
        self.frontier_heap.push(Reverse((Frontier::None, id)));
    }

    /// Advance a feed's frontier, pushing a heap entry only on change.
    fn advance_frontier(&mut self, id: usize, frontier: Frontier) {
        if self.frontier[id] != frontier {
            self.frontier[id] = frontier;
            self.frontier_heap.push(Reverse((frontier, id)));
        }
    }

    /// Peeks the minimum frontier across feeds, dropping stale entries.
    /// `Frontier::Done` when empty (the minimum over no feeds is the top).
    fn peek_frontier(&mut self) -> Frontier {
        while let Some(Reverse((f, id))) = self.frontier_heap.peek() {
            if self.frontier[*id] == *f {
                return *f;
            }
            self.frontier_heap.pop();
        }
        Frontier::Done
    }

    /// Await the one thing that can unblock: an active feed received its next
    /// event, or the wall clock reaching `wall` (optional).
    async fn await_progress(&mut self, wall: Option<Instant>) {
        let res = match wall {
            Some(wall) if self.active.is_empty() => {
                self.clock.wait_until(wall).await;
                None
            }
            Some(wall) => {
                let next = self.active.next();
                let clock = self.clock.wait_until(wall);
                match futures::future::select(next, pin!(clock)).await {
                    Either::Left((res, _)) => res,
                    Either::Right(((), _)) => None,
                }
            }
            None => self.active.next().await,
        };
        if let Some((id, feed)) = res {
            if let Some(stamp) = feed.stamp() {
                let t = match stamp {
                    Stamp::Instant(t) => t,
                    Stamp::Now => self.clock.now(),
                };
                self.parked[id] = Some(feed);
                self.parked_heap.push(Reverse((t, id)));
                self.advance_frontier(id, Frontier::Stamp(stamp));
            } else {
                self.advance_frontier(id, Frontier::Done);
            };
        }
    }

    /// Advances to the next completed batch, returning its timestamp or `None`
    /// if every feed is exhausted.
    pub async fn step(&mut self, sink: &mut S) -> Option<Instant> {
        loop {
            let now = self.clock.now();
            // The minimum frontier over all feeds. `None` means the global
            // frontier is unknown (at least one feed is unheard-from), while
            // `Done` means every feed is exhausted.
            let frontier = self.peek_frontier();
            if frontier == Frontier::Done && self.instant.is_none() {
                return None;
            }
            // The concrete frontier: the minimum over every feed's abstract
            // frontier and the wall clock, resolved to an explicit instant.
            // Nothing can arrive stamped below it.
            let floor = match frontier {
                Frontier::None => None,
                Frontier::Stamp(Stamp::Instant(e)) => Some(e.min(now)),
                Frontier::Stamp(Stamp::Now) | Frontier::Done => Some(now),
            };
            // Re-activate parked feeds in timestamp order, up to the floor.
            // Coalesce all events at the current batch timestamp.
            while let Some(floor) = floor
                && let Some(Reverse((t, id))) = self.parked_heap.peek()
                && *t <= floor
                && self.instant.is_none_or(|g| g == *t)
            {
                let (t, id) = (*t, *id);
                self.parked_heap.pop();
                let mut feed = self.parked[id].take().unwrap();
                if feed.write(sink, t) {
                    self.instant = Some(t);
                }
                self.active.push(Active::new(id, feed));
            }
            // The batch closes strictly below the floor: frontiers are
            // inclusive and a live arrival may still be stamped `now`, so only
            // then can nothing more arrive at its timestamp.
            if self.instant.is_some() && self.instant < floor {
                return self.instant.take();
            }
            // Otherwise name what the wall clock must strictly pass, if it is
            // a blocker.
            let wall = match (self.instant, self.parked_heap.peek()) {
                (Some(g), _) if frontier > Frontier::Stamp(Stamp::Instant(g)) => Some(g),
                (None, Some(Reverse((t, _))))
                    if frontier >= Frontier::Stamp(Stamp::Instant(*t)) && now < *t =>
                {
                    Some(*t)
                }
                _ => None,
            };
            self.await_progress(wall).await;
        }
    }
}
