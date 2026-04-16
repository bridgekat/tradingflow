//! Historical-only source backed by pre-loaded arrays.

use tokio::sync::mpsc;

use crate::Instant;
use crate::{Array, Scalar, Series, Source};

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries an `Array<T>` value.  The historical channel is filled
/// by a spawned tokio task with bounded back-pressure; the live channel is
/// empty.
///
/// Requires a tokio runtime to be active when added to a scenario.
pub struct ArraySource<T: Scalar> {
    series: Series<T>,
    default: Array<T>,
}

impl<T: Scalar> ArraySource<T> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(series: Series<T>, default: Array<T>) -> Self {
        Self { series, default }
    }
}

impl<T: Scalar> Source for ArraySource<T> {
    type Event = Array<T>;
    type Output = Array<T>;

    fn estimated_event_count(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn init(
        self,
        _timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, Array<T>)>,
        mpsc::Receiver<(Instant, Array<T>)>,
        Array<T>,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            for (i, &ts) in self.series.timestamps().iter().enumerate() {
                let stride = self.series.stride();
                let start = i * stride;
                let slice = &self.series.values()[start..start + stride];
                let arr = Array::from_vec(self.series.shape(), slice.to_vec());
                if hist_tx.send((ts, arr)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, self.default)
    }

    fn write(payload: Array<T>, output: &mut Array<T>, _timestamp: Instant) -> bool {
        output.assign(payload.as_slice());
        true
    }
}
