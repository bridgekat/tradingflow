//! Python-visible scenario wrapper.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`](crate::scenario::Scenario)
//! and provides registration entry points for sources and operators from
//! both Rust and Python.
//!
//! # Lifecycle
//!
//! - Registration (`add_*`) appends descriptors to the underlying
//!   [`Scenario`] definition.  No node buffers are allocated and no
//!   Python views are constructed at this point.
//! - On [`run`](NativeScenario::run), a fresh [`Session`](crate::scenario::Session)
//!   is built by replaying every descriptor's `init`; the bridge then
//!   rebuilds its cached Python views to point at the new session's
//!   buffers and drives the event loop.  After `run()` returns, view
//!   accessors expose those buffers.  Calling `run()` again replaces
//!   the session and invalidates any view obtained from the previous run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::Scenario;
use crate::scenario::{Session, ShutdownFlag};

use super::dispatch::resolve_type_id;
use super::operator::PyInputInfo;
use super::views::{NativeNodeKind, create_view};
use super::{ErrorSlot, operator, operators, source, sources};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// DoneGuard — panic-safe asyncio Event signalling
// ---------------------------------------------------------------------------

/// Signals an `asyncio.Event` via `call_soon_threadsafe(event.set)` on drop.
///
/// Created on the event loop background thread; fires on both normal return and
/// panic (stack unwinding), ensuring the main thread's
/// `run_until_complete(event.wait())` always unblocks.
struct DoneGuard(Option<PyObject>, PyObject);

impl Drop for DoneGuard {
    fn drop(&mut self) {
        if let Some(set_fn) = self.0.take() {
            // Best-effort: if GIL acquisition or the call fails (e.g.
            // interpreter shutting down), we silently ignore — the main
            // thread will notice the join failure.
            let _ = Python::attach(|py| -> PyResult<()> {
                let loop_ = self.1.bind(py);
                loop_.call_method1("call_soon_threadsafe", (&set_fn.bind(py),))?;
                Ok(())
            });
        }
    }
}

// SAFETY: PyObject is Send; the guard is created and dropped on the
// background thread.
unsafe impl Send for DoneGuard {}

// ---------------------------------------------------------------------------
// Per-node Python metadata
// ---------------------------------------------------------------------------

/// Per-node Python-side metadata captured at registration time.  Used to
/// rebuild [`NativeArrayView`](super::views::NativeArrayView) /
/// [`NativeSeriesView`](super::views::NativeSeriesView) wrappers against
/// each freshly built [`Session`]'s buffers.
#[derive(Clone)]
struct NodeInfo {
    dtype: String,
    kind: NativeNodeKind,
    shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
///
/// Owns a tokio runtime used to drive the event loop and source
/// driver tasks.  Registration methods enter the runtime so that
/// sources can use [`tokio::spawn`] during `init`.
#[pyclass]
pub struct NativeScenario {
    /// Pure definition: descriptors only, no per-run state.
    scenario: Scenario,
    /// Most-recently-built session.  `None` until the first `run()`.
    /// Replaced by every subsequent `run()`; views obtained from a
    /// previous session are invalidated when this is overwritten.
    session: Option<Session>,
    error_slot: ErrorSlot,
    /// Per-node metadata in registration order.
    node_info: Vec<NodeInfo>,
    /// Cached Python view objects, indexed by node index.  Built when
    /// `session` is replaced; `None` for unit nodes.
    cached_views: Vec<Option<PyObject>>,
    /// Tokio runtime — kept alive for the scenario's lifetime.
    runtime: tokio::runtime::Runtime,
    /// Asyncio event loop for Python source coroutines, created on first
    /// `add_py_source`.  Runs on the **main thread** during `run()` for
    /// proper signal handling; the tokio event loop runs on a background
    /// thread.
    event_loop: Option<PyObject>,
    /// Cooperative shutdown flag — set on drop to stop the event loop.
    shutdown: ShutdownFlag,
}

unsafe impl Send for NativeScenario {}
unsafe impl Sync for NativeScenario {}

impl Drop for NativeScenario {
    fn drop(&mut self) {
        // Signal the event loop background thread to exit so it doesn't keep
        // the process alive after the Python object is garbage-collected.
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl NativeScenario {
    /// Append per-node metadata in registration order.
    fn push_info(&mut self, dtype: &str, kind: NativeNodeKind, shape: &[usize]) {
        self.node_info.push(NodeInfo {
            dtype: dtype.to_string(),
            kind,
            shape: shape.to_vec(),
        });
        self.cached_views.push(None);
    }

    /// Rebuild every cached view to point at the buffers of the current
    /// session.  Called after each `run()` populates a new session.
    fn rebuild_cached_views(&mut self, py: Python<'_>) -> PyResult<()> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no active session"))?;
        for (idx, info) in self.node_info.iter().enumerate() {
            let view = if info.kind == NativeNodeKind::Unit {
                py.None()
            } else {
                let ptr = session.value_ptr(idx);
                create_view(py, ptr, &info.shape, &info.dtype, info.kind)?
            };
            self.cached_views[idx] = Some(view);
        }
        Ok(())
    }

    /// Ensure the asyncio event loop exists.
    ///
    /// Created lazily on the first `add_py_source` call.  The loop is
    /// **not** started here — it runs on the main thread during `run()`.
    fn ensure_event_loop(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(ref loop_) = self.event_loop {
            return Ok(loop_.clone_ref(py));
        }

        let asyncio = py.import("asyncio")?;
        let event_loop: PyObject = asyncio.call_method0("new_event_loop")?.unbind();
        self.event_loop = Some(event_loop.clone_ref(py));
        Ok(event_loop)
    }
}

#[pymethods]
impl NativeScenario {
    #[new]
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        Self {
            scenario: Scenario::new(),
            session: None,
            error_slot: Arc::new(Mutex::new(None)),
            node_info: Vec::new(),
            cached_views: Vec::new(),
            runtime,
            event_loop: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get a cached view for a node.  Available only after the first
    /// successful `run()`.
    fn view(&self, py: Python<'_>, node_index: usize) -> PyResult<PyObject> {
        match self.cached_views.get(node_index) {
            Some(Some(view)) => Ok(view.clone_ref(py)),
            Some(None) => Err(PyRuntimeError::new_err(format!(
                "node {node_index} has no view (run() the scenario first)"
            ))),
            None => Err(PyRuntimeError::new_err(format!(
                "node {node_index} out of range"
            ))),
        }
    }

    /// Register a Rust-native source by `(source_kind, dtype)` + params.
    ///
    /// The output [`NativeNodeKind`] is determined by the Rust source type — the
    /// Python caller does not need to specify it.
    #[pyo3(signature = (source_kind, dtype, shape, params))]
    fn add_native_source(
        &mut self,
        source_kind: &str,
        dtype: &str,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        let (idx, view_kind) =
            sources::dispatch_native_source(&mut self.scenario, source_kind, dtype, params)?;
        self.push_info(dtype, view_kind, &shape);
        Ok(idx)
    }

    /// Register a Rust-native operator by kind + dtype + params.
    ///
    /// `dtype` is `None` for operators whose output is [`NativeNodeKind::Unit`]
    /// (no value, only a trigger).  For Array/Series outputs it is required
    /// and dispatched on; arms that receive `None` with such an output will
    /// surface a dispatch-time error.
    #[pyo3(signature = (kind, dtype, input_indices, shape, params))]
    fn add_native_operator(
        &mut self,
        kind: &str,
        dtype: Option<String>,
        input_indices: Vec<usize>,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        // For Array/Series outputs each dispatch arm consults dtype via
        // `dispatch_dtype!`, which itself errors on an unknown dtype string;
        // an absent dtype surfaces through the same path as an empty string.
        // For Unit outputs dtype is never consulted — the arm returns early
        // before dispatching on it.
        let dtype_str = dtype.as_deref().unwrap_or("");
        let (idx, view_kind) = operators::dispatch_native_operator(
            &mut self.scenario,
            kind,
            dtype_str,
            &input_indices,
            params,
        )?;
        self.push_info(dtype_str, view_kind, &shape);
        Ok(idx)
    }

    /// Register a Python source.
    ///
    /// Appends a descriptor to the scenario.  The actual `init()` and
    /// tokio driver task are deferred to session-build time inside
    /// [`run`](Self::run).
    #[pyo3(signature = (py_source, output_kind, dtype, output_shape))]
    fn add_py_source(
        &mut self,
        py: Python<'_>,
        py_source: PyObject,
        output_kind: NativeNodeKind,
        dtype: &str,
        output_shape: Vec<usize>,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        let output_type_id = resolve_type_id(output_kind, dtype)?;

        // Ensure asyncio event loop exists.
        let event_loop = self.ensure_event_loop(py)?;

        let erased = source::make_py_source(
            output_type_id,
            dtype,
            output_kind,
            &output_shape,
            py_source,
            event_loop,
            self.error_slot.clone(),
        )?;
        let idx = self.scenario.add_erased_source(erased);
        self.push_info(dtype, output_kind, &output_shape);
        Ok(idx)
    }

    /// Register a Python operator via
    /// [`ErasedOperator`](crate::operator::ErasedOperator).
    ///
    /// `input_types` is a list of `(NativeNodeKind, dtype)` pairs.
    /// `output_type` is a `(NativeNodeKind, dtype)` pair for the output node.
    /// `dtype` strings are canonical numpy dtype names (e.g. `"float64"`).
    ///
    /// Input type validation is performed by
    /// [`Scenario::add_erased_operator`].  Each input's
    /// `(kind, dtype, shape)` is captured into the operator's
    /// [`InitFn`](crate::operator::InitFn) so the operator can rebuild
    /// Python input views on every session start.
    #[pyo3(signature = (input_indices, input_types, output_type, output_shape, py_operator))]
    fn add_py_operator(
        &mut self,
        input_indices: Vec<usize>,
        input_types: Vec<(NativeNodeKind, String)>,
        output_type: (NativeNodeKind, String),
        output_shape: Vec<usize>,
        py_operator: PyObject,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        let (out_view_kind, out_dtype) = output_type;

        // 1. Resolve TypeIds from Python-declared types.
        let input_type_ids = input_types
            .iter()
            .map(|(kind, dtype)| resolve_type_id(*kind, dtype))
            .collect::<PyResult<Box<[_]>>>()?;
        let output_type_id = resolve_type_id(out_view_kind, &out_dtype)?;

        // 2. Build per-input metadata for the operator's init closure.
        //    The closure rebuilds views per session from this metadata
        //    plus the input pointers handed in at session-build time.
        let input_metadata: Vec<PyInputInfo> = input_indices
            .iter()
            .map(|&idx| {
                let info = &self.node_info[idx];
                PyInputInfo {
                    kind: info.kind,
                    dtype: info.dtype.clone(),
                    shape: info.shape.clone(),
                }
            })
            .collect();

        // 3. Construct the erased operator (calls operator.init at
        //    session-build time, not now).
        let erased = operator::make_py_operator(
            input_type_ids,
            input_metadata,
            output_type_id,
            &out_dtype,
            out_view_kind,
            &output_shape,
            py_operator,
            self.error_slot.clone(),
        )?;

        // 4. Register via the unified path (validates input TypeIds against
        //    upstream descriptors).
        let output_idx = self.scenario.add_erased_operator(erased, &input_indices);

        // 5. Cache output node metadata.
        self.push_info(&out_dtype, out_view_kind, &output_shape);

        Ok(output_idx)
    }

    /// Return the aggregate estimated event count across all sources.
    ///
    /// Returns `Some(total)` only when every registered source provides an
    /// estimate; otherwise `None`.
    fn estimated_event_count(&self) -> Option<usize> {
        self.scenario.estimated_event_count()
    }

    /// Run the event loop.
    ///
    /// Builds a fresh [`Session`] from the registered descriptors and
    /// drives it to completion.  After return, [`view`](Self::view) can
    /// be called to inspect the populated buffers.  Calling `run()` again
    /// replaces the session and invalidates any view obtained from the
    /// previous run.
    ///
    /// If Python sources have been registered, the asyncio event loop runs
    /// on the **main thread** (for proper signal handling) while the tokio
    /// event loop runs on a background thread.  Driver tasks on the tokio
    /// runtime iterate Python source async iterators by scheduling
    /// coroutines on the main-thread asyncio loop via
    /// `run_coroutine_threadsafe`.
    #[pyo3(signature = (on_flush=None))]
    fn run(&mut self, py: Python<'_>, on_flush: Option<PyObject>) -> PyResult<()> {
        // Drop any prior session before building a fresh one.  Releases
        // the previous session's buffers (and invalidates any user-held
        // views from the prior run).
        self.session = None;
        // Reset error slot so a previous failed run does not poison the
        // next one.
        *self.error_slot.lock().unwrap() = None;

        // Build a fresh session under the runtime guard so any
        // tokio::spawn calls inside source `init` find an active runtime.
        let mut session = {
            let _guard = self.runtime.enter();
            self.scenario.build_session()
        };

        // Two-thread model:
        //   Background thread: tokio block_on(event loop + driver tasks)
        //   Main thread:       asyncio run_until_complete(done_event.wait())
        //                      (if Python sources exist) or just wait (if not)
        //
        // The asyncio loop runs on the main thread for proper signal
        // handling.  A Done guard on the background thread sets the
        // asyncio.Event on both normal exit and panic (unwinding), so the
        // main thread always unblocks.

        // Temporarily take the runtime so we can move it into the
        // background thread (std::thread::spawn requires 'static).
        let rt = std::mem::replace(
            &mut self.runtime,
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );

        let event_loop = self.event_loop.take();

        // Create an asyncio.Event + its set callback for the Done guard.
        let done_signal = event_loop
            .as_ref()
            .map(|el| -> PyResult<_> {
                let asyncio = py.import("asyncio")?;
                let loop_ = el.bind(py);
                asyncio.call_method1("set_event_loop", (loop_,))?;
                let event: PyObject = asyncio.call_method0("Event")?.unbind();
                // Pre-bind event.set so the guard only needs call_soon_threadsafe.
                let set_fn: PyObject = event.getattr(py, "set")?;
                let el_ref = el.clone_ref(py);
                Ok((event, set_fn, el_ref))
            })
            .transpose()?;

        let done_guard_parts = done_signal
            .as_ref()
            .map(|(_, set_fn, el_ref)| (set_fn.clone_ref(py), el_ref.clone_ref(py)));

        // Spawn the event loop + drivers on a background thread.
        let shutdown = self.shutdown.clone();
        let error_slot_for_bg = self.error_slot.clone();
        let bg_handle = {
            let _guard = rt.enter();
            std::thread::spawn(move || {
                // Drop guard: signals the asyncio Event on both normal
                // return and panic (stack unwinding).
                let _done = done_guard_parts.map(|(set_fn, el)| DoneGuard(Some(set_fn), el));

                let shutdown_ref = shutdown.clone();
                let error_slot_ref = error_slot_for_bg.clone();

                let on_flush_fn =
                    move |ts: crate::data::Instant,
                          events_so_far: usize,
                          total_estimate: Option<usize>| {
                        // Check if a Python operator/source already failed.
                        if error_slot_ref.lock().unwrap().is_some() {
                            shutdown_ref.store(true, Ordering::Relaxed);
                            return;
                        }
                        if let Some(ref cb) = on_flush {
                            Python::attach(|py| {
                                // Wire format is TAI ns (matches numpy naive
                                // `datetime64[ns]` arithmetic).
                                let args = (ts.as_nanos(), events_so_far, total_estimate);
                                if let Err(e) = cb.call1(py, args) {
                                    super::set_error(&error_slot_ref, e);
                                    shutdown_ref.store(true, Ordering::Relaxed);
                                }
                            });
                        }
                    };

                rt.block_on(session.run_with_shutdown(on_flush_fn, shutdown));
                (session, rt)
            })
        };

        // Main thread: run the asyncio event loop until the Done guard
        // fires, or just release GIL and wait if no event loop.
        if let Some((event, _, _)) = done_signal {
            let loop_ = event_loop.as_ref().unwrap().bind(py);
            let wait_coro = event.call_method0(py, "wait")?;
            loop_.call_method1("run_until_complete", (wait_coro.bind(py),))?;
            loop_.call_method0("close")?;
            // The loop has been closed, but `add_py_source` will create a
            // fresh one on the next registration if needed.
            self.event_loop = None;
        }

        // Join the background thread, periodically reacquiring the GIL
        // so Python can process signals (e.g. KeyboardInterrupt from
        // Ctrl+C).  If interrupted, set the shutdown flag so the event
        // loop exits cooperatively.
        let join_result = loop {
            let is_finished = bg_handle.is_finished();
            if is_finished {
                break py.detach(|| bg_handle.join());
            }
            // Release GIL briefly, then check for Python signals.
            py.detach(|| std::thread::sleep(std::time::Duration::from_millis(100)));
            if let Err(e) = py.check_signals() {
                self.shutdown.store(true, Ordering::Relaxed);
                // Wait for the background thread to finish after shutdown.
                let _ = py.detach(|| bg_handle.join());
                return Err(e);
            }
        };
        match join_result {
            Ok((session, rt)) => {
                self.session = Some(session);
                self.runtime = rt;
            }
            Err(_) => {
                return Err(PyRuntimeError::new_err(
                    "event loop background thread panicked",
                ));
            }
        }

        // Rebuild cached views to point at the new session's buffers.
        self.rebuild_cached_views(py)?;

        if let Some(err) = self.error_slot.lock().unwrap().take() {
            Err(err)
        } else {
            Ok(())
        }
    }
}
