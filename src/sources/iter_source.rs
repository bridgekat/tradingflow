//! Iterator-based source — feeds events from an arbitrary iterator.

use tokio::sync::mpsc;

use crate::source::Source;

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
    iter: Box<dyn Iterator<Item = (i64, T)> + Send>,
    default: T,
    known_time_range: (Option<i64>, Option<i64>),
}

impl<T: Clone + Send + 'static> IterSource<T> {
    /// Create from an iterator and a default output value.
    ///
    /// Each iterator item is `(timestamp, value)`.
    pub fn new(iter: impl Iterator<Item = (i64, T)> + Send + 'static, default: T) -> Self {
        Self {
            iter: Box::new(iter),
            default,
            known_time_range: (None, None),
        }
    }

    /// Set the known time range for this source.
    ///
    /// Call this when the full set of timestamps is available at
    /// construction time (e.g. clock sources).
    pub fn with_time_range(mut self, first: i64, last: i64) -> Self {
        self.known_time_range = (Some(first), Some(last));
        self
    }
}

impl<T: Clone + Send + 'static> Source for IterSource<T> {
    type Event = T;
    type Output = T;

    fn time_range(&self) -> (Option<i64>, Option<i64>) {
        self.known_time_range
    }

    fn init(self, _timestamp: i64) -> (mpsc::Receiver<(i64, T)>, mpsc::Receiver<(i64, T)>, T) {
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

    fn write(payload: T, output: &mut T, _timestamp: i64) -> bool {
        *output = payload;
        true
    }
}
