//! Python source machinery for the bridge.
//!
//! [`make_py_source`] constructs an [`ErasedSource`] for a
//! Python-implemented source.  A tokio driver task iterates the
//! source's async iterators by scheduling each `__anext__()` coroutine
//! on the main-thread asyncio event loop via `run_coroutine_threadsafe`
//! and awaiting completion through a [`DoneCallback`].
//!
//! The write function ([`py_write_fn`]) is **not** generic — it delegates
//! to the Python output view's `write` method, mirroring the operator
//! bridge pattern.  Dtype dispatch is only needed for output allocation
//! and view creation.

use std::any::TypeId;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::mpsc;

use crate::Instant;
use crate::{Array, ErasedSource, PeekableReceiver, Series};

use super::dispatch::dispatch_dtype;
use super::views::{ViewKind, create_view};
use super::{ErrorSlot, set_error, set_error_msg};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Per-channel receiver state for the non-generic [`py_write_fn`].
///
/// Holds the channel receiver, output view, and error slot — mirroring
/// [`PyOperatorState`](super::operator) which stores views alongside
/// the compute callback.
struct PySourceState {
    rx: PeekableReceiver<(Instant, PyObject)>,
    py_output: PyObject,
    error_slot: ErrorSlot,
}

unsafe impl Send for PySourceState {}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

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
// Construction
// ---------------------------------------------------------------------------

/// Construct an [`ErasedSource`] for a Python-implemented source.
///
/// Allocates the output value, creates its Python view, and packages
/// everything into a type-erased source.  The caller provides the
/// resolved `output_type_id` and `out_view_kind`, mirroring how
/// [`make_py_operator`](super::operator::make_py_operator) receives
/// these from the scenario.  Dtype dispatch is only needed for the
/// output allocation — the write function is non-generic.
pub fn make_py_source(
    py: Python<'_>,
    output_type_id: TypeId,
    out_dtype: &str,
    out_view_kind: ViewKind,
    output_shape: &[usize],
    py_source: PyObject,
    event_loop: PyObject,
    error_slot: ErrorSlot,
) -> PyResult<ErasedSource> {
    // Allocate output and create its Python view.
    let (output_ptr, output_drop_fn): (*mut u8, unsafe fn(*mut u8)) =
        if out_view_kind == ViewKind::Unit {
            (Box::into_raw(Box::new(())) as *mut u8, drop_fn::<()>)
        } else {
            macro_rules! alloc_output {
                ($T:ty) => {
                    match out_view_kind {
                        ViewKind::Array => (
                            Box::into_raw(Box::new(Array::<$T>::zeros(output_shape))) as *mut u8,
                            drop_fn::<Array<$T>> as unsafe fn(*mut u8),
                        ),
                        ViewKind::Series => (
                            Box::into_raw(Box::new(Series::<$T>::new(output_shape))) as *mut u8,
                            drop_fn::<Series<$T>> as unsafe fn(*mut u8),
                        ),
                        ViewKind::Unit => unreachable!(),
                    }
                };
            }
            dispatch_dtype!(out_dtype, alloc_output)
        };

    let py_output = create_view(py, output_ptr, output_shape, out_dtype, out_view_kind)?;

    // Clone the view for each receiver (hist + live share the same output).
    let view_for_hist = py_output.clone_ref(py);
    let view_for_live = py_output;
    let error_for_driver = error_slot.clone();

    let init_fn: Box<dyn FnOnce(Instant) -> (*mut u8, *mut u8, *mut u8)> =
        Box::new(move |timestamp: Instant| {
            let (hist_tx, hist_rx) = mpsc::channel(64);
            let (live_tx, live_rx) = mpsc::channel(64);

            tokio::spawn(drive_source(
                py_source,
                event_loop,
                hist_tx,
                live_tx,
                error_for_driver,
                timestamp,
            ));

            let hist_state = PySourceState {
                rx: PeekableReceiver::new(hist_rx),
                py_output: view_for_hist,
                error_slot: error_slot.clone(),
            };
            let live_state = PySourceState {
                rx: PeekableReceiver::new(live_rx),
                py_output: view_for_live,
                error_slot,
            };

            let hist_ptr = Box::into_raw(Box::new(hist_state)) as *mut u8;
            let live_ptr = Box::into_raw(Box::new(live_state)) as *mut u8;
            (hist_ptr, live_ptr, output_ptr)
        });

    // SAFETY: init_fn returns valid PySourceState pointers and a valid
    // output pointer.  poll/write/drop fn ptrs match the types.
    Ok(unsafe {
        ErasedSource::new(
            TypeId::of::<PyObject>(),
            output_type_id,
            None, // Python sources don't provide an estimate by default
            init_fn,
            py_poll_fn,
            py_write_fn,
            drop_fn::<PySourceState>,
            output_drop_fn,
        )
    })
}

// ---------------------------------------------------------------------------
// Non-generic function pointers
// ---------------------------------------------------------------------------

/// Poll function for Python source channels.
///
/// Not generic — delegates to the inner [`PeekableReceiver`].
///
/// # Safety
///
/// `state` must point to a valid `PySourceState`.
unsafe fn py_poll_fn(
    state: *mut u8,
    cx: &mut std::task::Context<'_>,
) -> std::task::Poll<Option<Instant>> {
    let state = unsafe { &mut *(state as *mut PySourceState) };
    state.rx.poll_pending(cx).map(|opt| opt.map(|item| item.0))
}

/// Write function for Python source channels.
///
/// Delegates to the output view's `write` method, mirroring the operator
/// bridge's [`py_compute_fn`](super::operator).  Not generic — works
/// entirely through the Python view in [`PySourceState`].
///
/// # Safety
///
/// `state_ptr` must point to a valid `PySourceState`.
unsafe fn py_write_fn(state_ptr: *mut u8, _output_ptr: *mut u8, _ts: Instant) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PySourceState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    if let Some((_ts, py_value)) = state.rx.take_pending() {
        // Unit outputs carry no data — just consume the event and signal
        // downstream propagation without calling .write().
        let is_none = Python::attach(|py| state.py_output.is_none(py));
        if is_none {
            return true;
        }
        let result = Python::attach(|py| -> PyResult<()> {
            state
                .py_output
                .call_method1(py, "write", (py_value.bind(py),))
                .map(|_| ())
        });
        match result {
            Ok(()) => true,
            Err(e) => {
                set_error(&state.error_slot, e);
                false
            }
        }
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Type-erased box drop function, monomorphised per value type.
unsafe fn drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

/// Extract a numpy datetime64 as nanoseconds (i64).
fn extract_timestamp_ns(py: Python<'_>, ts_obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let np = py.import("numpy")?;
    let dt = np.call_method1("datetime64", (ts_obj, "ns"))?;
    let view = dt.call_method1("view", ("int64",))?;
    view.extract()
}

// ---------------------------------------------------------------------------
// Async driver
// ---------------------------------------------------------------------------

/// Top-level driver: iterates both async iterators, reports errors.
async fn drive_source(
    py_source: PyObject,
    event_loop: PyObject,
    hist_tx: mpsc::Sender<(Instant, PyObject)>,
    live_tx: mpsc::Sender<(Instant, PyObject)>,
    error_slot: ErrorSlot,
    timestamp: Instant,
) {
    let result = async {
        let (hist_iter, live_iter) = Python::attach(|py| -> PyResult<(PyObject, PyObject)> {
            // Wire format is TAI ns (matches numpy naive `datetime64[ns]`).
            let result = py_source.call_method1(py, "init", (timestamp.as_nanos(),))?;
            let tuple = result.bind(py);
            Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
        })
        .map_err(|e| format!("Source.init() failed: {e}"))?;

        drive_async_iter(&hist_iter, &event_loop, &hist_tx).await?;
        drop(hist_tx);

        drive_async_iter(&live_iter, &event_loop, &live_tx).await?;
        drop(live_tx);

        Ok::<(), String>(())
    }
    .await;

    if let Err(msg) = result {
        set_error_msg(&error_slot, msg);
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
    tx: &mpsc::Sender<(Instant, PyObject)>,
) -> Result<(), String> {
    loop {
        // Acquire GIL: call __anext__(), schedule on asyncio, register callback.
        let (coro_future, rx_done) = Python::attach(|py| -> PyResult<_> {
            let coro = py_iter.call_method0(py, "__anext__")?;
            let asyncio = py.import("asyncio")?;
            let coro_future = asyncio.call_method1(
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
            coro_future.call_method1("add_done_callback", (callback,))?;
            Ok((coro_future.unbind(), rx_done))
        })
        .map_err(|e| format!("__anext__() scheduling failed: {e}"))?;

        // Suspend until the coroutine completes (no GIL held).
        let _ = rx_done.await;

        // Acquire GIL: extract (timestamp, value) or detect StopAsyncIteration.
        let event = Python::attach(|py| -> PyResult<Option<(Instant, PyObject)>> {
            match coro_future.call_method0(py, "result") {
                Ok(value) => {
                    let tuple = value.bind(py);
                    let ts_ns = extract_timestamp_ns(py, &tuple.get_item(0)?)?;
                    let val = tuple.get_item(1)?.unbind();
                    // Wire format is TAI ns (matches numpy naive `datetime64[ns]`).
                    Ok(Some((Instant::from_nanos(ts_ns), val)))
                }
                Err(e) => {
                    if e.get_type(py).name()? == "StopAsyncIteration" {
                        Ok(None)
                    } else {
                        Err(e)
                    }
                }
            }
        })
        .map_err(|e| format!("event extraction failed: {e}"))?;

        match event {
            Some((ts, val)) => {
                if tx.send((ts, val)).await.is_err() {
                    break;
                }
            }
            None => break,
        }
    }
    Ok(())
}
