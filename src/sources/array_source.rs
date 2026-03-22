//! Historical-only source backed by pre-loaded arrays.

use tokio::sync::mpsc;

use crate::array::Array;
use crate::source::Source;
use crate::types::Scalar;

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries an `Array<T>` value.  The historical channel is filled
/// by a spawned task with bounded back-pressure; the live channel is empty.
pub struct ArraySource<T: Scalar> {
    timestamps: Vec<i64>,
    values: Vec<T>,
    stride: usize,
}

impl<T: Scalar> ArraySource<T> {
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

impl<T: Scalar> Source for ArraySource<T> {
    type Event = Array<T>;
    type Output = Array<T>;

    fn init(
        self,
        _timestamp: i64,
    ) -> (
        mpsc::Receiver<(i64, Array<T>)>,
        mpsc::Receiver<(i64, Array<T>)>,
        Array<T>,
    ) {
        let shape: Vec<usize> = if self.stride == 1 {
            vec![]
        } else {
            vec![self.stride]
        };
        let output = Array::zeros(&shape);

        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        let stride = self.stride;
        std::thread::spawn(move || {
            for (i, &ts) in self.timestamps.iter().enumerate() {
                let start = i * stride;
                let slice = &self.values[start..start + stride];
                let arr = if stride == 1 {
                    Array::scalar(slice[0].clone())
                } else {
                    Array::from_vec(&[stride], slice.to_vec())
                };
                if hist_tx.blocking_send((ts, arr)).is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, output)
    }

    fn write(payload: Array<T>, output: &mut Array<T>, _timestamp: i64) -> bool {
        output.assign(&payload);
        true
    }
}
