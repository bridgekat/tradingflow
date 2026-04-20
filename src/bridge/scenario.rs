//! Python-visible scenario wrapper.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`](crate::scenario::Scenario)
//! and provides registration entry points for sources and operators from
//! both Rust and Python.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::Scenario;
use crate::scenario::ShutdownFlag;

use super::dispatch::resolve_type_id;
use super::views::{NativeNodeKind, create_view};
use super::{ErrorSlot, operator, operators, source, sources};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// DoneGuard — panic-safe asyncio Event signalling
// ---------------------------------------------------------------------------

/// Signals an `asyncio.Event` via `call_soon_threadsafe(event.set)` on drop.
///
/// Created on the POCQ background thread; fires on both normal return and
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
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
///
/// Owns a tokio runtime used to drive the POCQ event loop and source
/// driver tasks.  Registration methods enter the runtime so that
/// sources can use [`tokio::spawn`] during `init`.
#[pyclass]
pub struct NativeScenario {
    scenario: Option<Scenario>,
    error_slot: ErrorSlot,
    /// Per-node metadata: (dtype_str, view_kind).
    node_info: Vec<(String, NativeNodeKind)>,
    /// Cached Python view objects, indexed by node index.
    cached_views: Vec<Option<PyObject>>,
    /// Tokio runtime — kept alive for the scenario's lifetime.
    runtime: tokio::runtime::Runtime,
    /// Asyncio event loop for Python source coroutines, created on first
    /// `add_py_source`.  Runs on the **main thread** during `run()` for
    /// proper signal handling; the tokio POCQ loop runs on a background
    /// thread.
    event_loop: Option<PyObject>,
    /// Cooperative shutdown flag — set on drop to stop the POCQ loop.
    shutdown: ShutdownFlag,
}

unsafe impl Send for NativeScenario {}
unsafe impl Sync for NativeScenario {}

impl Drop for NativeScenario {
    fn drop(&mut self) {
        // Signal the POCQ background thread to exit so it doesn't keep
        // the process alive after the Python object is garbage-collected.
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl NativeScenario {
    /// Record a node's metadata and eagerly create + cache its Python view.
    fn push_node(
        &mut self,
        py: Python<'_>,
        node_index: usize,
        dtype: &str,
        kind: NativeNodeKind,
        shape: &[usize],
    ) -> PyResult<()> {
        // Ensure vectors are sized to accommodate node_index.
        while self.node_info.len() <= node_index {
            self.node_info.push((String::new(), NativeNodeKind::Unit)); // placeholder; overwritten below
            self.cached_views.push(None);
        }
        self.node_info[node_index] = (dtype.to_string(), kind);

        // Unit nodes carry no data; store Python None as the view.
        let view = if kind == NativeNodeKind::Unit {
            py.None()
        } else {
            let sc = self.scenario.as_ref().unwrap();
            let ptr = sc.value_ptr(node_index);
            create_view(py, ptr, shape, dtype, kind)?
        };
        self.cached_views[node_index] = Some(view);
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
            scenario: Some(Scenario::new()),
            error_slot: Arc::new(Mutex::new(None)),
            node_info: Vec::new(),
            cached_views: Vec::new(),
            runtime,
            event_loop: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get a cached view for a node.
    fn view(&self, py: Python<'_>, node_index: usize) -> PyResult<PyObject> {
        match self.cached_views.get(node_index) {
            Some(Some(view)) => Ok(view.clone_ref(py)),
            _ => Err(PyRuntimeError::new_err(format!(
                "node {node_index} has no Python-representable view"
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
        py: Python<'_>,
        source_kind: &str,
        dtype: &str,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        let sc = self.scenario.as_mut().unwrap();
        let (idx, view_kind) = sources::dispatch_native_source(sc, source_kind, dtype, params)?;
        self.push_node(py, idx, dtype, view_kind, &shape)?;
        Ok(idx)
    }

    /// Register a Rust-native operator by kind + dtype + params.
    #[pyo3(signature = (kind, dtype, input_indices, shape, params))]
    fn add_native_operator(
        &mut self,
        py: Python<'_>,
        kind: &str,
        dtype: &str,
        input_indices: Vec<usize>,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let _guard = self.runtime.enter();
        let sc = self.scenario.as_mut().unwrap();
        let (idx, view_kind) =
            operators::dispatch_native_operator(sc, kind, dtype, &input_indices, params)?;
        self.push_node(py, idx, dtype, view_kind, &shape)?;
        Ok(idx)
    }

    /// Register a Python source.
    ///
    /// Immediately creates the DAG node with channels.  A tokio driver task
    /// is spawned that will call `source.init()` and iterate the returned
    /// async iterators when the tokio runtime runs.
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
            py,
            output_type_id,
            dtype,
            output_kind,
            &output_shape,
            py_source,
            event_loop,
            self.error_slot.clone(),
        )?;
        let sc = self.scenario.as_mut().unwrap();
        let idx = sc.add_erased_source(erased);
        self.push_node(py, idx, dtype, output_kind, &output_shape)?;
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
    /// [`Scenario::add_erased_operator`].
    #[pyo3(signature = (input_indices, input_types, output_type, output_shape, py_operator))]
    fn add_py_operator(
        &mut self,
        py: Python<'_>,
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

        // 2. Build input views (input nodes already exist).
        //    Wrap NativeArrayView / NativeSeriesView in their Python-side
        //    wrappers (ArrayView / SeriesView) so Python operators receive
        //    the high-level types.
        let views_mod = py.import("tradingflow.data.views")?;
        let array_view_cls = views_mod.getattr("ArrayView")?;
        let series_view_cls = views_mod.getattr("SeriesView")?;

        let input_views: Vec<PyObject> = input_indices
            .iter()
            .map(|&idx| {
                let native = self.cached_views[idx]
                    .as_ref()
                    .map(|v| v.clone_ref(py))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!("node {idx} has no cached view"))
                    })?;
                let (_, kind) = &self.node_info[idx];
                let wrapped: PyObject = match kind {
                    NativeNodeKind::Array => array_view_cls.call1((native.bind(py),))?.unbind(),
                    NativeNodeKind::Series => series_view_cls.call1((native.bind(py),))?.unbind(),
                    // Unit inputs carry no data — pass None to the Python operator.
                    NativeNodeKind::Unit => py.None(),
                };
                Ok(wrapped)
            })
            .collect::<PyResult<_>>()?;
        let py_inputs: PyObject = pyo3::types::PyTuple::new(py, &input_views)?
            .into_any()
            .unbind();

        // 3. Construct the erased operator (calls operator.init internally).
        let erased = operator::make_py_operator(
            py,
            input_type_ids,
            output_type_id,
            &out_dtype,
            out_view_kind,
            &output_shape,
            py_inputs,
            py_operator,
            crate::data::Instant::MIN,
            self.error_slot.clone(),
        )?;

        // 4. Register via the unified path (validates input TypeIds).
        let sc = self.scenario.as_mut().unwrap();
        let output_idx = sc.add_erased_operator(erased, &input_indices);

        // 5. Cache output node metadata and view.
        self.push_node(py, output_idx, &out_dtype, out_view_kind, &output_shape)?;

        Ok(output_idx)
    }

    /// Return the aggregate estimated event count across all sources.
    ///
    /// Returns `Some(total)` only when every registered source provides an
    /// estimate; otherwise `None`.
    fn estimated_event_count(&self) -> Option<usize> {
        self.scenario
            .as_ref()
            .and_then(|sc| sc.estimated_event_count())
    }

    /// Run the POCQ event loop.
    ///
    /// If Python sources have been registered, the asyncio event loop runs
    /// on the **main thread** (for proper signal handling) while the tokio
    /// POCQ loop runs on a background thread.  Driver tasks on the tokio
    /// runtime iterate Python source async iterators by scheduling
    /// coroutines on the main-thread asyncio loop via
    /// `run_coroutine_threadsafe`.
    #[pyo3(signature = (on_flush=None))]
    fn run(&mut self, py: Python<'_>, on_flush: Option<PyObject>) -> PyResult<()> {
        let mut scenario = self.scenario.take().ok_or_else(|| {
            PyRuntimeError::new_err("scenario already consumed by a previous run()")
        })?;

        // Two-thread model:
        //   Background thread: tokio block_on(POCQ + driver tasks)
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

        // Spawn the POCQ + drivers on a background thread.
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

                rt.block_on(scenario.run_with_shutdown(on_flush_fn, shutdown));
                (scenario, rt)
            })
        };

        // Main thread: run the asyncio event loop until the Done guard
        // fires, or just release GIL and wait if no event loop.
        if let Some((event, _, _)) = done_signal {
            let loop_ = event_loop.as_ref().unwrap().bind(py);
            let wait_coro = event.call_method0(py, "wait")?;
            loop_.call_method1("run_until_complete", (wait_coro.bind(py),))?;
            loop_.call_method0("close")?;
        }

        // Join the background thread, periodically reacquiring the GIL
        // so Python can process signals (e.g. KeyboardInterrupt from
        // Ctrl+C).  If interrupted, set the shutdown flag so the POCQ
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
            Ok((scenario, rt)) => {
                self.scenario = Some(scenario);
                self.runtime = rt;
            }
            Err(_) => {
                return Err(PyRuntimeError::new_err("POCQ background thread panicked"));
            }
        }

        if let Some(err) = self.error_slot.lock().unwrap().take() {
            Err(err)
        } else {
            Ok(())
        }
    }
}
