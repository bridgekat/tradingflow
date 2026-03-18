//! Source traits and built-in implementations for data ingestion.
//!
//! A [`Source`] produces historical and/or live data streams via
//! [`subscribe`](Source::subscribe), mirroring the Python `Source.subscribe()`
//! contract.  Both iterators are async-native: Rust in-memory sources
//! complete instantly (zero-cost `async`), while I/O or Python-bridged
//! sources genuinely await.
//!
//! # Built-in sources
//!
//! * [`ArraySource`] — historical-only source backed by pre-loaded arrays.

use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// Iterator traits
// ---------------------------------------------------------------------------

/// Historical iterator: yields `(timestamp_ns, value)` in strictly increasing
/// timestamp order.
///
/// `next_into` writes the value into the caller-provided buffer and returns
/// the timestamp, or `None` when exhausted.
pub trait HistoricalIter {
    fn next_into<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = Option<i64>> + 'a>>;
}

/// Live iterator: yields values (the [`Scenario`] assigns wall-clock
/// timestamps).
///
/// `next_into` writes the value into the caller-provided buffer and returns
/// `true`, or `false` when exhausted.  It may block/await until data is
/// available.
pub trait LiveIter {
    fn next_into<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = bool> + 'a>>;
}

/// A data source providing historical and/or live data streams.
///
/// `subscribe` consumes the source and returns two iterators covering
/// non-overlapping time segments.  Use [`EmptyHistoricalIter`] /
/// [`EmptyLiveIter`] for segments with no data.
pub trait Source {
    fn subscribe(
        self: Box<Self>,
    ) -> Pin<Box<dyn Future<Output = (Box<dyn HistoricalIter>, Box<dyn LiveIter>)>>>;
}

// ---------------------------------------------------------------------------
// Empty iterators
// ---------------------------------------------------------------------------

/// A historical iterator that yields nothing (for live-only sources).
pub struct EmptyHistoricalIter;

impl HistoricalIter for EmptyHistoricalIter {
    fn next_into<'a>(
        &'a mut self,
        _buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = Option<i64>> + 'a>> {
        Box::pin(async { None })
    }
}

/// A live iterator that yields nothing (for historical-only sources).
pub struct EmptyLiveIter;

impl LiveIter for EmptyLiveIter {
    fn next_into<'a>(
        &'a mut self,
        _buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = bool> + 'a>> {
        Box::pin(async { false })
    }
}

// ---------------------------------------------------------------------------
// ArraySource<T>
// ---------------------------------------------------------------------------

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// `next_into()` completes instantly (no I/O).
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
        Self { timestamps, values, stride }
    }
}

impl<T: Copy + Send + 'static> Source for ArraySource<T> {
    fn subscribe(
        self: Box<Self>,
    ) -> Pin<Box<dyn Future<Output = (Box<dyn HistoricalIter>, Box<dyn LiveIter>)>>> {
        Box::pin(async move {
            let hist: Box<dyn HistoricalIter> = Box::new(ArrayHistoricalIter {
                timestamps: self.timestamps,
                values: unsafe {
                    // Reinterpret Vec<T> as Vec<u8> for the generic buffer interface.
                    // SAFETY: T: Copy, layout is compatible, and we track byte_stride.
                    let mut v = std::mem::ManuallyDrop::new(self.values);
                    let ptr = v.as_mut_ptr() as *mut u8;
                    let byte_len = v.len() * std::mem::size_of::<T>();
                    let byte_cap = v.capacity() * std::mem::size_of::<T>();
                    Vec::from_raw_parts(ptr, byte_len, byte_cap)
                },
                byte_stride: self.stride * std::mem::size_of::<T>(),
                pos: 0,
            });
            let live: Box<dyn LiveIter> = Box::new(EmptyLiveIter);
            (hist, live)
        })
    }
}

/// Historical iterator for [`ArraySource`].
struct ArrayHistoricalIter {
    timestamps: Vec<i64>,
    values: Vec<u8>, // type-erased bytes
    byte_stride: usize,
    pos: usize,
}

impl HistoricalIter for ArrayHistoricalIter {
    fn next_into<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = Option<i64>> + 'a>> {
        Box::pin(async move {
            if self.pos >= self.timestamps.len() {
                return None;
            }
            let byte_start = self.pos * self.byte_stride;
            buf[..self.byte_stride]
                .copy_from_slice(&self.values[byte_start..byte_start + self.byte_stride]);
            let ts = self.timestamps[self.pos];
            self.pos += 1;
            Some(ts)
        })
    }
}

// Ensure ArrayHistoricalIter's Vec<u8> is properly dropped (it was transmuted
// from Vec<T> — since T: Copy there are no destructors to run, and the
// allocator only cares about (ptr, layout) which is compatible because
// size_of::<u8>() == 1 and we preserved byte_len/byte_cap).

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
        let (mut hist, _live) = src.subscribe().await;

        let mut buf = vec![0u8; std::mem::size_of::<f64>()];

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, Some(10));
        assert_eq!(f64::from_ne_bytes(buf[..8].try_into().unwrap()), 1.0);

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, Some(20));
        assert_eq!(f64::from_ne_bytes(buf[..8].try_into().unwrap()), 2.0);

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, Some(30));
        assert_eq!(f64::from_ne_bytes(buf[..8].try_into().unwrap()), 3.0);

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, None);
    }

    #[tokio::test]
    async fn array_source_strided() {
        let src = Box::new(ArraySource::<f64>::new(
            vec![1, 2],
            vec![10.0, 20.0, 30.0, 40.0],
            2,
        ));
        let (mut hist, _live) = src.subscribe().await;

        let byte_stride = 2 * std::mem::size_of::<f64>();
        let mut buf = vec![0u8; byte_stride];

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, Some(1));
        let v0 = f64::from_ne_bytes(buf[..8].try_into().unwrap());
        let v1 = f64::from_ne_bytes(buf[8..16].try_into().unwrap());
        assert_eq!((v0, v1), (10.0, 20.0));

        let ts = hist.next_into(&mut buf).await;
        assert_eq!(ts, Some(2));
        let v0 = f64::from_ne_bytes(buf[..8].try_into().unwrap());
        let v1 = f64::from_ne_bytes(buf[8..16].try_into().unwrap());
        assert_eq!((v0, v1), (30.0, 40.0));

        assert_eq!(hist.next_into(&mut buf).await, None);
    }

    #[tokio::test]
    async fn empty_live_iter() {
        let mut live = EmptyLiveIter;
        let mut buf = [0u8; 8];
        assert!(!live.next_into(&mut buf).await);
    }
}
