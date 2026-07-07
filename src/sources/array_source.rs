//! Historical-only source backed by pre-loaded arrays.

use tokio::sync::mpsc;

use crate::Instant;
use crate::{Array, Scalar, Series, Source};

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries an `Array<T>` value.  The event channel is filled by a
/// spawned tokio task with bounded back-pressure.
///
/// Requires a tokio runtime to be active when added to a scenario.
#[derive(Clone)]
pub struct ArraySource<T: Scalar, const N: usize> {
    series: Series<T>,
    default: Array<T, N>,
}

impl<T: Scalar, const N: usize> ArraySource<T, N> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(series: Series<T>, default: Array<T, N>) -> Self {
        Self { series, default }
    }
}

impl<T: Scalar, const N: usize> Source for ArraySource<T, N> {
    type Event = Array<T, N>;
    type Output = Array<T, N>;
    type State = ();

    fn estimated_event_count(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn initial(&self) -> Array<T, N> {
        self.default.clone()
    }

    fn init(&self) -> (mpsc::Receiver<(Instant, Array<T, N>)>, ()) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        // Clone the series for the spawned driver; the spec stays
        // borrowable so the same source can drive multiple sessions.
        let series = self.series.clone();
        tokio::spawn(async move {
            for (i, &ts) in series.timestamps().iter().enumerate() {
                let stride = series.stride();
                let start = i * stride;
                let slice = &series.values()[start..start + stride];
                let extents = <[usize; N]>::try_from(series.shape())
                    .expect("ArraySource: series element rank != N");
                let arr = Array::from_vec(extents, slice.to_vec());
                if hist_tx.send((ts, arr)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, ())
    }

    fn write(_state: &mut (), payload: Array<T, N>, output: &mut Array<T, N>, _timestamp: Instant) -> usize {
        output.assign(payload.as_slice());
        1
    }
}
