//! Source registration for the Python bridge.
//!
//! Provides:
//! * [`HistoricalEventSender`] / [`LiveEventSender`] — pyclasses for Python
//!   to push events into a channel-based source.
//! * [`dispatch_native_source`] — register a Rust source by kind string.
//! * [`register_channel_source`] — register a channel-based source backed
//!   by an [`ErasedSource`](crate::source::ErasedSource).

use std::any::TypeId;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use tokio::sync::mpsc;

use crate::source::{ErasedReceiver, ErasedSource};
use crate::sources::ArraySource;
use crate::{Array, Scalar, Scenario};

use super::dispatch::{dispatch_dtype, dtype_element_bytes};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

/// Reinterpret a byte buffer as a `Vec<T>`.
///
/// # Safety
///
/// `T` must be a numeric type where every bit pattern of size
/// `size_of::<T>()` is a valid value.
pub unsafe fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    let n = bytes.len() / elem_size;
    let mut result = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            result.as_mut_ptr() as *mut u8,
            n * elem_size,
        );
        result.set_len(n);
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
            // SAFETY: dispatch_dtype only dispatches to numeric types where
            // all bit patterns are valid.
            let values = unsafe { bytes_to_vec::<$T>(&values_bytes) };
            let source = ArraySource::new(timestamps, values, stride);
            sc.add_source_untyped(source)
        }};
    }
    Ok(dispatch_dtype!(dtype, register, numeric))
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

    let (hist_tx, py_hist_rx) = mpsc::unbounded_channel();
    let (live_tx, py_live_rx) = mpsc::unbounded_channel();

    let shape_box: Box<[usize]> = shape.into();

    macro_rules! register_channel {
        ($T:ty) => {{
            let erased = make_channel_source::<$T>(py_hist_rx, py_live_rx, shape_box);
            sc.add_erased_source(erased)
        }};
    }

    let node_index = dispatch_dtype!(dtype, register_channel, numeric);

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

/// Construct an [`ErasedSource`] for a Python channel source.
///
/// Spawns async tasks to forward events from unbounded Python channels
/// to bounded internal channels.
fn make_channel_source<T: Scalar + Copy>(
    py_hist_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    py_live_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    shape: Box<[usize]>,
) -> ErasedSource {
    let event_type_id = TypeId::of::<Vec<u8>>();
    let output_type_id = TypeId::of::<Array<T>>();

    let init_fn: Box<dyn FnOnce(i64) -> (ErasedReceiver, ErasedReceiver, *mut u8)> =
        Box::new(move |_timestamp: i64| {
            let output = Array::<T>::zeros(&shape);

            let (hist_tx, hist_rx) = mpsc::channel(64);
            let (live_tx, live_rx) = mpsc::channel(64);

            // Forward unbounded → bounded.
            let mut py_hist = py_hist_rx;
            tokio::spawn(async move {
                while let Some(item) = py_hist.recv().await {
                    if hist_tx.send(item).await.is_err() {
                        break;
                    }
                }
            });
            let mut py_live = py_live_rx;
            tokio::spawn(async move {
                while let Some(item) = py_live.recv().await {
                    if live_tx.send(item).await.is_err() {
                        break;
                    }
                }
            });

            let hist = ErasedReceiver::from_receiver(hist_rx);
            let live = ErasedReceiver::from_receiver(live_rx);
            let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
            (hist, live, output_ptr)
        });

    // SAFETY: init_fn returns valid receivers and a valid Array<T> pointer;
    // write_fn correctly extracts Vec<u8> from ReceiverState and copies into Array<T>;
    // output_drop_fn correctly drops Array<T>.
    unsafe {
        ErasedSource::new(
            event_type_id,
            output_type_id,
            init_fn,
            channel_write_fn::<T>,
            erased_drop_fn::<Array<T>>,
        )
    }
}

/// Write function for channel sources: extract pending `Vec<u8>` from the
/// receiver state and copy bytes into `Array<T>`.
///
/// # Safety
///
/// * `receiver_state` must point to a valid `ReceiverState<Vec<u8>>`.
/// * `output_ptr` must point to a valid `Array<T>`.
/// * `T` must be a numeric type where all bit patterns are valid (ensured
///   by `dispatch_dtype!(..., numeric)`).
unsafe fn channel_write_fn<T: Scalar + Copy>(receiver_state: *mut u8, output_ptr: *mut u8) -> bool {
    // ReceiverState is private to source.rs, but its layout is:
    // { rx: Receiver<(i64, Vec<u8>)>, pending: Option<(i64, Vec<u8>)> }
    // We access it through the same struct definition re-created here.
    // Instead, we use the crate-internal access pattern: the pending event
    // is stored as Option<(i64, E)> at a known offset.
    //
    // Actually, we can't access ReceiverState directly since it's private.
    // The write_fn is monomorphized and stored when the ErasedSource is
    // created — at that point, the concrete type is known.  The function
    // accesses the state through the same type it was created with.
    use crate::source::ReceiverState;
    let rs = unsafe { &mut *(receiver_state as *mut ReceiverState<Vec<u8>>) };
    if let Some((_ts, payload)) = rs.pending.take() {
        let output = unsafe { &mut *(output_ptr as *mut Array<T>) };
        let byte_len = output.as_slice().len() * std::mem::size_of::<T>();
        assert_eq!(
            payload.len(),
            byte_len,
            "channel payload size mismatch: expected {byte_len}, got {}",
            payload.len(),
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                output.as_slice_mut().as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        true
    } else {
        false
    }
}

/// Type-erased box drop function.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
