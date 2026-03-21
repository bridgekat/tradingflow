//! Source registration for the Python bridge.
//!
//! Provides:
//! * [`ChannelSource`] — a [`Source`] impl driven by Python via mpsc channels.
//! * [`HistoricalEventSender`] / [`LiveEventSender`] — pyclasses for Python
//!   to push events into the channel source.
//! * Array source registration helpers.

use std::marker::PhantomData;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::mpsc;

use crate::scenario::{Handle, Scenario};
use crate::source::Source;
use crate::sources::ArraySource;
use crate::store::ElementViewMut;
use crate::types::Scalar;

use super::dispatch::{dtype_element_bytes, normalise_dtype};

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
///
/// Uses unbounded receivers from Python, then forwards to bounded channels
/// for the Scenario's POCQ runtime.
pub struct ChannelSource<T: Copy> {
    py_hist_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    py_live_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    shape: Box<[usize]>,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Source for ChannelSource<T> {
    type Event = Vec<u8>;
    type Scalar = T;

    fn default(&self) -> (Box<[usize]>, Box<[T]>) {
        let shape = self.shape.clone();
        let stride = shape.iter().product::<usize>();
        (shape, vec![T::default(); stride].into())
    }

    fn subscribe(
        self,
    ) -> (
        mpsc::Receiver<(i64, Vec<u8>)>,
        mpsc::Receiver<(i64, Vec<u8>)>,
    ) {
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

        (hist_rx, live_rx)
    }

    fn write(payload: Vec<u8>, output: ElementViewMut<'_, T>) -> bool {
        let values = unsafe {
            std::slice::from_raw_parts(
                payload.as_ptr() as *const T,
                payload.len() / std::mem::size_of::<T>(),
            )
        };
        output.values.copy_from_slice(values);
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
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Create a node and register an `ArraySource` in one step.
///
/// Returns the node index.
pub fn register_array_source(
    sc: &mut Scenario,
    dtype: &str,
    timestamps: Vec<i64>,
    values_bytes: Vec<u8>,
    stride: usize,
    windowed: bool,
) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    macro_rules! register {
        ($T:ty) => {{
            let values = bytes_to_vec::<$T>(&values_bytes);
            let source = ArraySource::new(timestamps, values, stride);
            let handle: Handle<$T> = sc.add_source(source, windowed);
            handle.index()
        }};
    }
    let idx = match dtype {
        "float64" => register!(f64),
        "float32" => register!(f32),
        "int64" => register!(i64),
        "int32" => register!(i32),
        "uint64" => register!(u64),
        "uint32" => register!(u32),
        "bool" => register!(u8),
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "unsupported dtype: {other}"
            )));
        }
    };
    Ok(idx)
}

/// Create a channel source and return (node_index, hist_sender, live_sender).
pub fn register_channel_source(
    sc: &mut Scenario,
    shape: &[usize],
    dtype: &str,
    windowed: bool,
) -> PyResult<(usize, HistoricalEventSender, LiveEventSender)> {
    let dtype_norm = normalise_dtype(dtype);
    let stride: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let elem_bytes = dtype_element_bytes(dtype_norm)?;
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
            let handle: Handle<$T> = sc.add_source(source, windowed);
            handle.index()
        }};
    }

    let node_index = match dtype_norm {
        "float64" => register_channel!(f64),
        "float32" => register_channel!(f32),
        "int64" => register_channel!(i64),
        "int32" => register_channel!(i32),
        "uint64" => register_channel!(u64),
        "uint32" => register_channel!(u32),
        "bool" => register_channel!(u8),
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "unsupported dtype: {other}"
            )));
        }
    };

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
