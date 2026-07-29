//! The timestamp-ordered merge over event feeds.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::{Pin, pin};
use std::task::{Context, Poll, ready};

use futures::StreamExt;
use futures::future::Either;
use futures::stream::FuturesUnordered;

use super::{Feed, Stamp, Time};

/// The currently known frontier of a feed.
///
/// The derived `Ord` is the heap order: `None` (not yet heard from) is the
/// bottom and blocks everything, explicit stamps compare as instants,
/// `Stamp(Now)` sits above them (an implicit feed never blocks an explicit
/// batch, the wall clock covers it), and `Done` is the top and never blocks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Frontier<I: Clone + Ord> {
    None,
    Stamp(Stamp<I>),
    Done,
}

/// A feed that is being polled for the next event.
struct Active<I, S> {
    id: usize,
    feed: Option<Box<dyn Feed<I, S>>>,
}

impl<I, S> Active<I, S> {
    pub fn new(id: usize, feed: Box<dyn Feed<I, S>>) -> Self {
        Self {
            id,
            feed: Some(feed),
        }
    }
}

impl<I, S> Future for Active<I, S> {
    type Output = (usize, Box<dyn Feed<I, S>>);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let feed = this.feed.as_mut().unwrap();
        ready!(feed.poll_next(cx));
        Poll::Ready((this.id, this.feed.take().unwrap()))
    }
}

/// A timestamp-ordered merge queue of event feeds.
///
/// Each feed is a stream of events. On ingestion of each event, one write
/// operation is executed against a caller-supplied sink `S`.
///
/// The queue merges all feeds' events in a global timestamp order and batches
/// together all events at the same timestamp. Batches always have strictly
/// increasing timestamps.
///
/// With `k` feeds, each event is roughly processed in `O(log k)` time. However,
/// feeds emitting events with explicit timestamps must advance their own
/// frontiers periodically, in order for the queue to make progress. Events
/// with explicit timestamps are generally used for replaying "historical" data
/// rather than "real-time" events, in which case the next event is always
/// available, so the frontier is always advancing.
pub struct Queue<I: Clone + Ord, T: Time<I>, S> {
    /// The wall clock source.
    time: T,
    /// Feeds currently being polled for the next event.
    active: FuturesUnordered<Active<I, S>>,
    /// Parked feeds with buffered events waiting to be read.
    /// Complementary to `active`, except for exhausted feeds.
    parked: Vec<Option<Box<dyn Feed<I, S>>>>,
    /// Exact min-heap over parked feeds, keyed by buffered stamp.
    parked_heap: BinaryHeap<Reverse<(I, usize)>>,
    /// Per-feed frontiers. Each feed guarantees not to emit events strictly
    /// before its frontier.
    frontier: Vec<Frontier<I>>,
    /// Lazy min-heap over frontiers, pushed on changes only.
    /// Entries can become stale as `frontier` advances.
    frontier_heap: BinaryHeap<Reverse<(Frontier<I>, usize)>>,
    /// The timestamp of the current batch, if any.
    instant: Option<I>,
}

impl<I: Clone + Ord, S, T: Time<I>> Queue<I, T, S> {
    /// Creates an empty queue over a wall clock.
    pub fn new(time: T) -> Self {
        Self {
            time,
            active: FuturesUnordered::new(),
            parked: Vec::new(),
            parked_heap: BinaryHeap::new(),
            frontier: Vec::new(),
            frontier_heap: BinaryHeap::new(),
            instant: None,
        }
    }

    /// Registers a feed.
    pub fn add_feed(&mut self, feed: impl Feed<I, S> + 'static) {
        let id = self.parked.len();
        self.active.push(Active::new(id, Box::new(feed)));
        self.parked.push(None);
        self.frontier.push(Frontier::None);
        self.frontier_heap.push(Reverse((Frontier::None, id)));
    }

    /// Advances a feed's frontier, pushing a heap entry only on change.
    fn advance_frontier(&mut self, id: usize, frontier: Frontier<I>) {
        if self.frontier[id] != frontier {
            self.frontier[id] = frontier.clone();
            self.frontier_heap.push(Reverse((frontier, id)));
        }
    }

    /// Peeks the minimum frontier across feeds, dropping stale entries.
    /// Returns `Frontier::Done` when empty.
    fn peek_frontier(&mut self) -> Frontier<I> {
        while let Some(Reverse((f, id))) = self.frontier_heap.peek() {
            if self.frontier[*id] == *f {
                return f.clone();
            }
            self.frontier_heap.pop();
        }
        Frontier::Done
    }

    /// Waits for one thing that can unblock: an active feed received its next
    /// event, or the wall clock reaching `wall` (optional).
    async fn await_progress(&mut self, wall: Option<I>) {
        let res = match wall {
            Some(wall) if self.active.is_empty() => {
                self.time.wait_until(wall).await;
                None
            }
            Some(wall) => {
                let next = self.active.next();
                let time = self.time.wait_until(wall);
                match futures::future::select(next, pin!(time)).await {
                    Either::Left((res, _)) => res,
                    Either::Right(((), _)) => None,
                }
            }
            None => self.active.next().await,
        };
        if let Some((id, feed)) = res {
            if let Some(stamp) = feed.stamp() {
                let t = match stamp.clone() {
                    Stamp::Instant(t) => t,
                    Stamp::Now => self.time.now(),
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
    pub async fn step(&mut self, sink: &mut S) -> Option<I> {
        loop {
            let now = self.time.now();
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
            let floor = match frontier.clone() {
                Frontier::None => None,
                Frontier::Stamp(Stamp::Instant(e)) => Some(e.min(now.clone())),
                Frontier::Stamp(Stamp::Now) | Frontier::Done => Some(now.clone()),
            };
            // Re-activate parked feeds in timestamp order, up to the floor.
            // Coalesce all events at the current batch timestamp.
            while let Some(ref floor) = floor
                && let Some(Reverse((t, id))) = self.parked_heap.peek().cloned()
                && t <= *floor
                && self.instant.clone().is_none_or(|g| g == t)
            {
                self.parked_heap.pop();
                let mut feed = self.parked[id].take().unwrap();
                if feed.write(t.clone(), sink) {
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
            let wall = match (&self.instant, self.parked_heap.peek()) {
                (Some(g), _) if frontier > Frontier::Stamp(Stamp::Instant(g.clone())) => Some(g),
                (None, Some(Reverse((t, _))))
                    if frontier >= Frontier::Stamp(Stamp::Instant(t.clone())) && now < *t =>
                {
                    Some(t)
                }
                _ => None,
            };
            self.await_progress(wall.cloned()).await;
        }
    }
}
