//! Python source machinery for the bridge.
//!
//! [`register_py_source`] constructs an [`ErasedSource`] for a
//! Python-implemented source.  A tokio driver task iterates the
//! source's async iterators by scheduling each `__anext__()` coroutine
//! on the main-thread asyncio event loop via `run_coroutine_threadsafe`
//! and awaiting completion through a [`DoneCallback`].
//!
//! Channel events carry `(i64, PyObject)` — the raw Python value is
//! converted to bytes and copied into the output `Array<T>` during the
//! write step (which runs under GIL on the POCQ thread).

use std::any::TypeId;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::mpsc;

use crate::source::{ErasedSource, PeekableReceiver, PollFn, WriteFn};
use crate::{Array, Scalar, Scenario};

use super::dispatch::dispatch_dtype;
use super::{ErrorSlot, set_error};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Stored state for a Python source.
pub struct PySourceSpec {
    pub py_source: PyObject,
    pub shape: Box<[usize]>,
    pub dtype: String,
}

/// A Python-callable callback that signals a tokio oneshot channel when
/// a `concurrent.futures.Future` completes.
#[pyclass]
pub struct DoneCallback {
    tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

#[pymethods]
impl DoneCallback {
    fn __call__(&self, _future: &Bound<'_, PyAny>) {
        if let Some(tx) = self.tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register a Python source and return the node index.
///
/// Constructs an [`ErasedSource`] whose `init_fn` creates bounded channels
/// and spawns a tokio driver task.  The driver iterates the source's async
/// iterators by scheduling coroutines on `event_loop`.
pub fn register_py_source(
    sc: &mut Scenario,
    py: Python<'_>,
    spec: PySourceSpec,
    event_loop: PyObject,
    error_slot: ErrorSlot,
) -> PyResult<usize> {
    let dtype = spec.dtype.clone();
    macro_rules! register {
        ($T:ty) => {{
            let erased = make_erased_source::<$T>(spec, event_loop, error_slot);
            sc.add_erased_source(erased)
        }};
    }
    let node_index = dispatch_dtype!(&dtype, register, numeric);
    let _ = py;
    Ok(node_index)
}

// ---------------------------------------------------------------------------
// ErasedSource construction
// ---------------------------------------------------------------------------

/// Construct an [`ErasedSource`] for a Python source.
///
/// Channel events are `(i64, PyObject)` — the Python value object is kept
/// alive until the write step, where it is converted to a C-contiguous
/// numpy array and its bytes are copied into the output `Array<T>`.
fn make_erased_source<T: Scalar + Copy>(
    spec: PySourceSpec,
    event_loop: PyObject,
    error_slot: ErrorSlot,
) -> ErasedSource {
    let event_type_id = TypeId::of::<PyObject>();
    let output_type_id = TypeId::of::<Array<T>>();

    let poll_fn: PollFn = |state, cx| unsafe {
        (&mut *(state as *mut PeekableReceiver<(i64, PyObject)>))
            .poll_pending(cx)
            .map(|opt| opt.map(|item| item.0))
    };
    let rx_drop_fn: unsafe fn(*mut u8) =
        |ptr| unsafe { drop(Box::from_raw(ptr as *mut PeekableReceiver<(i64, PyObject)>)) };

    let init_fn: Box<dyn FnOnce(i64) -> (*mut u8, *mut u8, *mut u8)> =
        Box::new(move |_timestamp: i64| {
            let output = Array::<T>::zeros(&spec.shape);

            let (hist_tx, hist_rx) = mpsc::channel(64);
            let (live_tx, live_rx) = mpsc::channel(64);

            tokio::spawn(drive_source(
                spec.py_source,
                event_loop,
                hist_tx,
                live_tx,
                error_slot,
            ));

            let hist_ptr = Box::into_raw(Box::new(PeekableReceiver::new(hist_rx))) as *mut u8;
            let live_ptr = Box::into_raw(Box::new(PeekableReceiver::new(live_rx))) as *mut u8;
            let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
            (hist_ptr, live_ptr, output_ptr)
        });

    // SAFETY: init_fn returns valid PeekableReceiver<(i64, PyObject)> pointers
    // and a valid Array<T> output pointer; all function pointers match.
    unsafe {
        ErasedSource::new(
            event_type_id,
            output_type_id,
            init_fn,
            poll_fn,
            write_fn::<T> as WriteFn,
            rx_drop_fn,
            drop_fn::<Array<T>>,
        )
    }
}

/// Write function: convert a pending `PyObject` value to a C-contiguous
/// numpy array and copy its bytes into the output `Array<T>`.
///
/// Acquires the GIL to perform the numpy conversion.
///
/// # Safety
///
/// * `rx_ptr` must point to a valid `PeekableReceiver<(i64, PyObject)>`.
/// * `out_ptr` must point to a valid `Array<T>`.
unsafe fn write_fn<T: Scalar + Copy>(rx_ptr: *mut u8, out_ptr: *mut u8, _ts: i64) -> bool {
    use super::dispatch::ContiguousArrayInfo;

    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(i64, PyObject)>) };
    if let Some((_ts, py_value)) = rx.take_pending() {
        let output = unsafe { &mut *(out_ptr as *mut Array<T>) };
        let expected = output.as_slice().len();

        Python::attach(|py| {
            let src = ContiguousArrayInfo::try_from(py_value.bind(py))
                .expect("failed to read __array_interface__");
            assert_eq!(
                src.len(),
                expected,
                "numpy array element count mismatch: expected {expected}, got {}",
                src.len(),
            );
            unsafe { src.clone_to_slice(output.as_slice_mut()) };
        });

        true
    } else {
        false
    }
}

unsafe fn drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

// ---------------------------------------------------------------------------
// Async driver
// ---------------------------------------------------------------------------

/// Top-level driver: iterates both async iterators, reports errors.
async fn drive_source(
    py_source: PyObject,
    event_loop: PyObject,
    hist_tx: mpsc::Sender<(i64, PyObject)>,
    live_tx: mpsc::Sender<(i64, PyObject)>,
    error_slot: ErrorSlot,
) {
    let result = async {
        let (hist_iter, live_iter) = Python::attach(|py| -> PyResult<(PyObject, PyObject)> {
            let result = py_source.call_method0(py, "init")?;
            let tuple = result.bind(py);
            Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
        })
        .map_err(|e| format!("source.init() failed: {e}"))?;

        drive_async_iter(&hist_iter, &event_loop, &hist_tx).await?;
        drop(hist_tx);

        drive_async_iter(&live_iter, &event_loop, &live_tx).await?;
        drop(live_tx);

        Ok::<(), String>(())
    }
    .await;

    if let Err(msg) = result {
        set_error(&error_slot, msg);
    }
}

/// Drive one Python async iterator, sending `(timestamp_ns, PyObject)` events.
///
/// Each `__anext__()` coroutine is scheduled on the asyncio event loop via
/// `run_coroutine_threadsafe`.  A [`DoneCallback`] signals a tokio oneshot
/// channel when the coroutine completes, so the driver suspends without
/// holding the GIL.
async fn drive_async_iter(
    py_iter: &PyObject,
    event_loop: &PyObject,
    tx: &mpsc::Sender<(i64, PyObject)>,
) -> Result<(), String> {
    loop {
        // Acquire GIL: call __anext__(), schedule on asyncio, register callback.
        let (cf_future, rx) = Python::attach(|py| -> PyResult<_> {
            let coro = py_iter.call_method0(py, "__anext__")?;
            let asyncio = py.import("asyncio")?;
            let cf_future = asyncio.call_method1(
                "run_coroutine_threadsafe",
                (coro.bind(py), event_loop.bind(py)),
            )?;
            let (tx_done, rx_done) = tokio::sync::oneshot::channel();
            let callback = Py::new(
                py,
                DoneCallback {
                    tx: Mutex::new(Some(tx_done)),
                },
            )?;
            cf_future.call_method1("add_done_callback", (callback,))?;
            Ok((cf_future.unbind(), rx_done))
        })
        .map_err(|e| format!("__anext__ scheduling failed: {e}"))?;

        // Suspend until the coroutine completes (no GIL held).
        let _ = rx.await;

        // Acquire GIL: extract (timestamp, value) or detect StopAsyncIteration.
        let event = Python::attach(|py| -> PyResult<Option<(i64, PyObject)>> {
            match cf_future.call_method0(py, "result") {
                Ok(value) => {
                    let tuple = value.bind(py);
                    let ts_ns = extract_timestamp_ns(py, &tuple.get_item(0)?)?;
                    let val = tuple.get_item(1)?.unbind();
                    Ok(Some((ts_ns, val)))
                }
                Err(e) => {
                    if e.get_type(py).name()?.to_string() == "StopAsyncIteration" {
                        Ok(None)
                    } else {
                        Err(e)
                    }
                }
            }
        })
        .map_err(|e| format!("event extraction failed: {e}"))?;

        match event {
            Some((ts_ns, val)) => {
                if tx.send((ts_ns, val)).await.is_err() {
                    break;
                }
            }
            None => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a numpy datetime64 as nanoseconds (i64).
fn extract_timestamp_ns(py: Python<'_>, ts_obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let np = py.import("numpy")?;
    let dt = np.call_method1("datetime64", (ts_obj, "ns"))?;
    let view = dt.call_method1("view", ("int64",))?;
    view.extract()
}
