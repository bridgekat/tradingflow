//! Source registration for the Python bridge.
//!
//! Provides:
//! * [`ChannelSource`] — a [`Source`](crate::source::Source) impl driven
//!   by Python via mpsc channels.
//! * [`HistoricalEventSender`] / [`LiveEventSender`] — pyclasses for Python
//!   to push events into the channel source.
//! * [`dispatch_native_source`] — register a Rust source by kind string.
//! * [`register_channel_source`] — register a channel-based source.

use std::marker::PhantomData;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use tokio::sync::mpsc;

use crate::sources::ArraySource;
use crate::{Array, Scalar, Scenario, Source};

use super::dispatch::{dispatch_dtype, dtype_element_bytes};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

/// Reinterpret a byte buffer as a `Vec<T>`.
pub fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    let n = bytes.len() / elem_size;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * elem_size;
        let val = unsafe { std::ptr::read_unaligned(bytes[offset..].as_ptr() as *const T) };
        result.push(val);
    }
    result
}

/// Extract raw bytes from a numpy array.
pub fn extract_value_bytes(
    py: Python<'_>,
    value: &PyObject,
    element_size: usize,
) -> PyResult<Vec<u8>> {
    let np = py.import("numpy")?;
    let contiguous = np.call_method1("ascontiguousarray", (value.bind(py),))?;
    let interface = contiguous.getattr("__array_interface__")?;
    let data_tuple = interface.get_item("data")?;
    let ptr_int: usize = data_tuple.get_item(0)?.extract()?;
    let mut bytes = vec![0u8; element_size];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr_int as *const u8, bytes.as_mut_ptr(), element_size);
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// ChannelSource — for concurrent async Python sources
// ---------------------------------------------------------------------------

/// Source that receives events from a Python background thread via channels.
pub struct ChannelSource<T: Scalar> {
    py_hist_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    py_live_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    shape: Box<[usize]>,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Source for ChannelSource<T> {
    type Event = Vec<u8>;
    type Output = Array<T>;

    fn init(
        self,
        _timestamp: i64,
    ) -> (
        mpsc::Receiver<(i64, Vec<u8>)>,
        mpsc::Receiver<(i64, Vec<u8>)>,
        Array<T>,
    ) {
        let output = Array::zeros(&self.shape);
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (live_tx, live_rx) = mpsc::channel(64);

        let mut py_hist = self.py_hist_rx;
        tokio::spawn(async move {
            while let Some(item) = py_hist.recv().await {
                if hist_tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        let mut py_live = self.py_live_rx;
        tokio::spawn(async move {
            while let Some(item) = py_live.recv().await {
                if live_tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, output)
    }

    fn write(payload: Vec<u8>, output: &mut Array<T>, _timestamp: i64) -> bool {
        let values = unsafe {
            std::slice::from_raw_parts(
                payload.as_ptr() as *const T,
                payload.len() / std::mem::size_of::<T>(),
            )
        };
        output.as_slice_mut().clone_from_slice(values);
        true
    }
}

// ---------------------------------------------------------------------------
// Python-callable event senders
// ---------------------------------------------------------------------------

/// Sends historical `(timestamp, value)` events from Python to Rust.
#[pyclass]
pub struct HistoricalEventSender {
    tx: Option<mpsc::UnboundedSender<(i64, Vec<u8>)>>,
    element_size: usize,
}

unsafe impl Send for HistoricalEventSender {}
unsafe impl Sync for HistoricalEventSender {}

#[pymethods]
impl HistoricalEventSender {
    fn send(&self, py: Python<'_>, ts_ns: i64, value: PyObject) -> PyResult<()> {
        let bytes = extract_value_bytes(py, &value, self.element_size)?;
        if let Some(ref tx) = self.tx {
            tx.send((ts_ns, bytes))
                .map_err(|_| PyRuntimeError::new_err("historical channel closed"))?;
        }
        Ok(())
    }

    fn close(&mut self) {
        self.tx.take();
    }
}

/// Sends live value events (with timestamp) from Python to Rust.
#[pyclass]
pub struct LiveEventSender {
    tx: Option<mpsc::UnboundedSender<(i64, Vec<u8>)>>,
    element_size: usize,
}

unsafe impl Send for LiveEventSender {}
unsafe impl Sync for LiveEventSender {}

#[pymethods]
impl LiveEventSender {
    fn send(&self, py: Python<'_>, ts_ns: i64, value: PyObject) -> PyResult<()> {
        let bytes = extract_value_bytes(py, &value, self.element_size)?;
        if let Some(ref tx) = self.tx {
            tx.send((ts_ns, bytes))
                .map_err(|_| PyRuntimeError::new_err("live channel closed"))?;
        }
        Ok(())
    }

    fn close(&mut self) {
        self.tx.take();
    }
}

// ---------------------------------------------------------------------------
// Native source dispatch
// ---------------------------------------------------------------------------

/// Register a Rust-native source by `(kind, dtype)` and return the output
/// node index.
pub fn dispatch_native_source(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    params: &Bound<'_, PyDict>,
) -> PyResult<usize> {
    match kind {
        "array" => {
            let timestamps: Vec<i64> = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'timestamps'"))?
                .extract()?;
            let values_bytes: Vec<u8> = params
                .get_item("values_bytes")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'values_bytes'"))?
                .extract()?;
            let stride: usize = params
                .get_item("stride")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'stride'"))?
                .extract()?;
            register_array_source(sc, dtype, timestamps, values_bytes, stride)
        }
        "csv" => {
            let path: String = params
                .get_item("path")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'path'"))?
                .extract()?;
            let time_column: String = params
                .get_item("time_column")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'time_column'"))?
                .extract()?;
            let value_columns: Vec<String> = params
                .get_item("value_columns")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'value_columns'"))?
                .extract()?;
            use crate::sources::CsvSource;
            let source = CsvSource::new(path, time_column, value_columns);
            Ok(sc.add_source_untyped(source))
        }
        "clock" => {
            let timestamps: Vec<i64> = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("clock source requires 'timestamps'"))?
                .extract()?;
            use crate::sources::clock;
            Ok(sc.add_source_untyped(clock(timestamps)))
        }
        "daily_clock" => {
            let start_ns: i64 = params
                .get_item("start_ns")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'start_ns'"))?
                .extract()?;
            let end_ns: i64 = params
                .get_item("end_ns")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'end_ns'"))?
                .extract()?;
            let tz: String = params
                .get_item("tz")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'tz'"))?
                .extract()?;
            use crate::sources::daily_clock;
            Ok(sc.add_source_untyped(daily_clock(start_ns, end_ns, &tz)))
        }
        "monthly_clock" => {
            let start_ns: i64 = params
                .get_item("start_ns")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'start_ns'"))?
                .extract()?;
            let end_ns: i64 = params
                .get_item("end_ns")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'end_ns'"))?
                .extract()?;
            let tz: String = params
                .get_item("tz")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'tz'"))?
                .extract()?;
            use crate::sources::monthly_clock;
            Ok(sc.add_source_untyped(monthly_clock(start_ns, end_ns, &tz)))
        }
        other => Err(PyTypeError::new_err(format!(
            "unknown native source kind: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration helpers
// ---------------------------------------------------------------------------

/// Create a node and register an `ArraySource` in one step.
pub fn register_array_source(
    sc: &mut Scenario,
    dtype: &str,
    timestamps: Vec<i64>,
    values_bytes: Vec<u8>,
    stride: usize,
) -> PyResult<usize> {
    macro_rules! register {
        ($T:ty) => {{
            let values = bytes_to_vec::<$T>(&values_bytes);
            let source = ArraySource::new(timestamps, values, stride);
            sc.add_source_untyped(source)
        }};
    }
    Ok(dispatch_dtype!(dtype, register))
}

/// Create a channel source and return (node_index, hist_sender, live_sender).
pub fn register_channel_source(
    sc: &mut Scenario,
    shape: &[usize],
    dtype: &str,
) -> PyResult<(usize, HistoricalEventSender, LiveEventSender)> {
    let stride: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let elem_bytes = dtype_element_bytes(dtype)?;
    let element_size = stride * elem_bytes;

    let (hist_tx, hist_rx) = mpsc::unbounded_channel();
    let (live_tx, live_rx) = mpsc::unbounded_channel();

    let shape_box: Box<[usize]> = shape.into();

    macro_rules! register_channel {
        ($T:ty) => {{
            let source = ChannelSource::<$T> {
                py_hist_rx: hist_rx,
                py_live_rx: live_rx,
                shape: shape_box,
                _phantom: PhantomData,
            };
            sc.add_source_untyped(source)
        }};
    }

    let node_index = dispatch_dtype!(dtype, register_channel);

    let hist_sender = HistoricalEventSender {
        tx: Some(hist_tx),
        element_size,
    };
    let live_sender = LiveEventSender {
        tx: Some(live_tx),
        element_size,
    };

    Ok((node_index, hist_sender, live_sender))
}
