//! Iterator-based source — feeds events from an arbitrary iterator.

use tokio::sync::mpsc;

use crate::source::Source;
use crate::Instant;

/// A source driven by an iterator of `(timestamp, event)` pairs.
///
/// More flexible than [`ArraySource`](super::ArraySource) — supports lazy
/// or computed timestamp sequences, and arbitrary output types.
///
/// The iterator is drained by a spawned tokio task with bounded
/// back-pressure.
///
/// Requires a tokio runtime to be active when added to a scenario.
pub struct IterSource<T: Send + 'static> {
    iter: Box<dyn Iterator<Item = (Instant, T)> + Send>,
    default: T,
    estimated_event_count: Option<usize>,
}

impl<T: Clone + Send + 'static> IterSource<T> {
    /// Create from an iterator and a default output value.
    ///
    /// Each iterator item is `(timestamp, value)`.
    pub fn new(iter: impl Iterator<Item = (Instant, T)> + Send + 'static, default: T) -> Self {
        Self {
            iter: Box::new(iter),
            default,
            estimated_event_count: None,
        }
    }

    /// Advertise an estimated total event count.
    ///
    /// Call this when the iterator length is known at construction time
    /// (e.g. clock sources backed by a `Vec`).  Used only by
    /// [`Scenario::run`](crate::Scenario::run)'s progress callback.
    pub fn with_estimated_count(mut self, count: usize) -> Self {
        self.estimated_event_count = Some(count);
        self
    }
}

impl<T: Clone + Send + 'static> Source for IterSource<T> {
    type Event = T;
    type Output = T;

    fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    fn init(self, _timestamp: Instant) -> (mpsc::Receiver<(Instant, T)>, mpsc::Receiver<(Instant, T)>, T) {
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            for (ts, value) in self.iter {
                if hist_tx.send((ts, value)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, self.default)
    }

    fn write(payload: T, output: &mut T, _timestamp: Instant) -> bool {
        *output = payload;
        true
    }
}
