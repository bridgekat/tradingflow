//! Batch source trait for the wavefront execution model.
//!
//! Unlike [`crate::Source`], all events are known upfront — no async
//! channels.  This matches the PoC scope (historical replay only).

use crate::data::Instant;

/// A data source whose events are fully known at construction time.
pub trait Source: 'static {
    /// The node value type.
    type Output: Send + Clone + 'static;

    /// The per-event payload type.
    type Value: Send + 'static;

    /// All `(timestamp, value)` pairs in chronological order.
    fn events(&self) -> Vec<(Instant, Self::Value)>;

    /// Create the initial output value for this source's node.
    fn init_output(&self) -> Self::Output;

    /// Write one event value into the output.
    fn write(value: &Self::Value, output: &mut Self::Output);
}

// ---------------------------------------------------------------------------
// ArraySource
// ---------------------------------------------------------------------------

use crate::Array;
use crate::Scalar;

/// A source backed by pre-built arrays of timestamps and values.
pub struct ArraySource<T: Scalar> {
    events: Vec<(Instant, Vec<T>)>,
    #[allow(dead_code)]
    shape: Box<[usize]>,
    default: Array<T>,
}

impl<T: Scalar> ArraySource<T> {
    /// `timestamps` and `values` must have the same length.
    /// `values[i]` has `shape.len()` elements.
    pub fn new(
        timestamps: Vec<Instant>,
        values: &[T],
        shape: &[usize],
        default: Array<T>,
    ) -> Self {
        let stride: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            timestamps.len() * stride,
            "ArraySource: values length {} != timestamps {} * stride {}",
            values.len(),
            timestamps.len(),
            stride,
        );
        let mut events: Vec<(Instant, Vec<T>)> = Vec::with_capacity(timestamps.len());
        for (i, &ts) in timestamps.iter().enumerate() {
            let start = i * stride;
            events.push((ts, values[start..start + stride].to_vec()));
        }
        Self {
            events,
            shape: shape.into(),
            default,
        }
    }
}

impl<T: Scalar> Source for ArraySource<T> {
    type Output = Array<T>;
    type Value = Vec<T>;

    fn events(&self) -> Vec<(Instant, Self::Value)> {
        self.events.clone()
    }

    fn init_output(&self) -> Array<T> {
        self.default.clone()
    }

    fn write(value: &Vec<T>, output: &mut Array<T>) {
        output.assign(value);
    }
}
