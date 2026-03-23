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
pub struct IterSource<O: Send + 'static> {
    iter: Box<dyn Iterator<Item = (i64, O)> + Send>,
    default: O,
}

impl<O: Clone + Send + 'static> IterSource<O> {
    /// Create from an iterator and a default output value.
    ///
    /// Each iterator item is `(timestamp, value)`.
    pub fn new(iter: impl Iterator<Item = (i64, O)> + Send + 'static, default: O) -> Self {
        Self {
            iter: Box::new(iter),
            default,
        }
    }
}

impl<O: Clone + Send + 'static> Source for IterSource<O> {
    type Event = O;
    type Output = O;

    fn init(self, _timestamp: i64) -> (mpsc::Receiver<(i64, O)>, mpsc::Receiver<(i64, O)>, O) {
        let output = self.default;

        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            for (ts, value) in self.iter {
                if hist_tx.send((ts, value)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, output)
    }

    fn write(payload: O, output: &mut O, _timestamp: i64) -> bool {
        *output = payload;
        true
    }
}
