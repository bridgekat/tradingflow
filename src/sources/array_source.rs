//! Historical-only source backed by pre-loaded arrays.

use std::future::Future;
use std::pin::Pin;

use tokio::sync::mpsc;

use crate::observable::Observable;
use crate::source::Source;

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries a single `T` value.  The historical channel is filled
/// by a spawned task with bounded back-pressure; the live channel is empty.
pub struct ArraySource<T: Copy + Send + 'static> {
    timestamps: Vec<i64>,
    values: Vec<T>,
    stride: usize,
}

impl<T: Copy + Send + 'static> ArraySource<T> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(timestamps: Vec<i64>, values: Vec<T>, stride: usize) -> Self {
        debug_assert_eq!(values.len(), timestamps.len() * stride);
        Self {
            timestamps,
            values,
            stride,
        }
    }
}

impl<T: Copy + Send + 'static> Source for ArraySource<T> {
    type Event = Vec<T>;
    type Output<'a>
        = &'a mut Observable<T>
    where
        Self: 'a;

    fn shape(&self) -> Box<[usize]> {
        if self.stride == 1 {
            Box::new([])
        } else {
            Box::new([self.stride])
        }
    }

    fn subscribe(
        self: Box<Self>,
    ) -> Pin<
        Box<
            dyn Future<Output = (mpsc::Receiver<(i64, Vec<T>)>, mpsc::Receiver<(i64, Vec<T>)>)>
                + Send,
        >,
    > {
        Box::pin(async move {
            let (hist_tx, hist_rx) = mpsc::channel(64);
            let (_, live_rx) = mpsc::channel(1);

            let stride = self.stride;
            tokio::spawn(async move {
                for (i, &ts) in self.timestamps.iter().enumerate() {
                    let start = i * stride;
                    let payload = self.values[start..start + stride].to_vec();
                    if hist_tx.send((ts, payload)).await.is_err() {
                        break;
                    }
                }
            });

            (hist_rx, live_rx)
        })
    }

    fn write(payload: Vec<T>, output: &mut Observable<T>) -> bool {
        output.write(&payload);
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn array_source_scalar() {
        let src = Box::new(ArraySource::<f64>::new(
            vec![10, 20, 30],
            vec![1.0, 2.0, 3.0],
            1,
        ));
        let (mut hist_rx, _live_rx) = src.subscribe().await;

        let (ts, val) = hist_rx.recv().await.unwrap();
        assert_eq!(ts, 10);
        assert_eq!(val, vec![1.0]);

        let (ts, val) = hist_rx.recv().await.unwrap();
        assert_eq!(ts, 20);
        assert_eq!(val, vec![2.0]);

        let (ts, val) = hist_rx.recv().await.unwrap();
        assert_eq!(ts, 30);
        assert_eq!(val, vec![3.0]);

        assert!(hist_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn array_source_strided() {
        let src = Box::new(ArraySource::<f64>::new(
            vec![1, 2],
            vec![10.0, 20.0, 30.0, 40.0],
            2,
        ));
        let (mut hist_rx, _live_rx) = src.subscribe().await;

        let (ts, val) = hist_rx.recv().await.unwrap();
        assert_eq!(ts, 1);
        assert_eq!(val, vec![10.0, 20.0]);

        let (ts, val) = hist_rx.recv().await.unwrap();
        assert_eq!(ts, 2);
        assert_eq!(val, vec![30.0, 40.0]);

        assert!(hist_rx.recv().await.is_none());
    }
}
