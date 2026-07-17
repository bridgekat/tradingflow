//! Historical-only source backed by pre-loaded arrays.

use futures::stream::Stream;
use tokio::sync::mpsc;

use super::receiver_stream;
use crate::data::Instant;
use crate::data::{Array, Scalar, Series};
use crate::ingest::{Event, EventSource};
use crate::operators::ArrayValue;

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries an `Array<T>` value.  The event channel is filled by a
/// spawned tokio task with bounded back-pressure.
///
/// Requires a tokio runtime to be active when added to a scenario.
#[derive(Clone)]
pub struct ArraySource<T: Scalar, const N: usize> {
    series: Series<T, N>,
    default: Array<T, N>,
}

impl<T: Scalar, const N: usize> ArraySource<T, N> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(series: Series<T, N>, default: Array<T, N>) -> Self {
        Self { series, default }
    }
}

impl<T: Scalar, const N: usize> EventSource for ArraySource<T, N> {
    type Event = Array<T, N>;
    type Value = ArrayValue<T, N>;

    fn total_num_events(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn initial(&self) -> Array<T, N> {
        self.default.clone()
    }

    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Array<T, N>>> + Send + 'static,
        impl FnMut(Array<T, N>, &mut Array<T, N>, Instant) -> usize + Send + 'static,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        // Clone the series for the spawned driver; the spec stays
        // borrowable so the same source can drive multiple sessions.
        let series = self.series.clone();
        tokio::spawn(async move {
            let view = series.view();
            for i in 0..view.len() {
                let arr = view.elem(i).to_array();
                if hist_tx.send((view.timestamp(i), arr)).await.is_err() {
                    break;
                }
            }
        });

        (
            receiver_stream(hist_rx),
            |payload, output: &mut Array<T, N>, _| {
                output.assign(payload.view());
                1
            },
        )
    }
}
