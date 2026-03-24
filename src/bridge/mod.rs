//! PyO3 bridge — exposes the Rust runtime to Python.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`](crate::scenario::Scenario),
//! providing four registration entry points:
//!
//! * [`add_native_operator`](NativeScenario::add_native_operator) — register a
//!   Rust-implemented operator by kind string + dtype + params.
//! * [`add_py_operator`](NativeScenario::add_py_operator) — register a
//!   Python-implemented operator whose `compute()` is called via GIL during
//!   flush.
//! * [`add_source_raw`](NativeScenario::add_source_raw) — register a
//!   channel-based source driven by Python async iterators.
//! * [`add_native_source`](NativeScenario::add_native_source) — register a
//!   Rust-implemented source by kind string.
//!
//! Views ([`ArrayView`], [`SeriesView`]) are cached per node and passed to
//! Python operator callbacks.  All view methods copy data across the boundary —
//! no reference to graph memory ever reaches Python.

mod dispatch;
mod operators;
mod sources;
mod views;

use std::any::TypeId;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

type PyObject = Py<PyAny>;

use crate::{Array, Scalar, Scenario, Series};

use dispatch::{dispatch_dtype, normalise_dtype};
pub use sources::{HistoricalEventSender, LiveEventSender};
pub use views::{_ArrayView, _SeriesView};

// ---------------------------------------------------------------------------
// Error slot
// ---------------------------------------------------------------------------

type ErrorSlot = Arc<Mutex<Option<String>>>;

fn set_error(slot: &ErrorSlot, msg: String) {
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(msg);
    }
}

// ---------------------------------------------------------------------------
// View creation helpers (monomorphized)
// ---------------------------------------------------------------------------

/// What kind of value a node holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ViewKind {
    Array,
    Series,
}

/// Create a cached Python view for a node, given its kind and dtype.
///
/// Returns `None` if the dtype is unsupported.
fn create_view(
    py: Python<'_>,
    ptr: *mut u8,
    shape: &[usize],
    dtype: &str,
    kind: ViewKind,
) -> PyResult<PyObject> {
    macro_rules! make_view {
        ($T:ty) => {
            match kind {
                ViewKind::Array => {
                    let v = views::make_array_view::<$T>(ptr, shape, dtype);
                    Ok(Py::new(py, v)?.into_any())
                }
                ViewKind::Series => {
                    let v = views::make_series_view::<$T>(ptr, shape, dtype);
                    Ok(Py::new(py, v)?.into_any())
                }
            }
        };
    }
    dispatch_dtype!(dtype, make_view)
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Create a node holding `Array<T>` with given shape/dtype.
fn create_array_node(sc: &mut Scenario, shape: &[usize], dtype: &str) -> PyResult<usize> {
    macro_rules! create {
        ($T:ty) => {{
            let arr = Array::<$T>::zeros(shape);
            Ok(sc.create_node(arr).index())
        }};
    }
    dispatch_dtype!(dtype, create)
}

/// Resolve a Python-side `(kind, dtype)` pair to a Rust `TypeId`.
fn resolve_type_id(kind: &str, dtype: &str) -> PyResult<TypeId> {
    let dtype = normalise_dtype(dtype);
    macro_rules! resolve {
        ($T:ty) => {
            match kind {
                "array" => Ok(TypeId::of::<Array<$T>>()),
                "series" => Ok(TypeId::of::<Series<$T>>()),
                other => Err(PyTypeError::new_err(format!("unknown node kind: {other}"))),
            }
        };
    }
    dispatch_dtype!(dtype, resolve)
}

// ---------------------------------------------------------------------------
// PyOperatorState — per-operator Python callback
// ---------------------------------------------------------------------------

struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_output: PyObject,
    py_state: PyObject,
    error_slot: ErrorSlot,
}

unsafe impl Send for PyOperatorState {}

// ---------------------------------------------------------------------------
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
///
/// Owns a tokio runtime that is entered on construction, so sources
/// can use [`tokio::spawn`] during `add_source`.
#[pyclass]
pub struct NativeScenario {
    scenario: Option<Scenario>,
    error_slot: ErrorSlot,
    /// Per-node metadata: (dtype_str, view_kind).
    node_info: Vec<(String, ViewKind)>,
    /// Cached Python view objects, indexed by node index.
    cached_views: Vec<Option<PyObject>>,
    /// Tokio runtime — kept alive for the scenario's lifetime.
    _rt: tokio::runtime::Runtime,
}

unsafe impl Send for NativeScenario {}
unsafe impl Sync for NativeScenario {}

impl NativeScenario {
    /// Record a node's metadata and eagerly create + cache its Python view.
    fn push_node(
        &mut self,
        py: Python<'_>,
        node_index: usize,
        dtype: &str,
        kind: ViewKind,
        shape: &[usize],
    ) -> PyResult<()> {
        // Ensure vectors are sized to accommodate node_index.
        while self.node_info.len() <= node_index {
            self.node_info.push((String::new(), ViewKind::Array));
            self.cached_views.push(None);
        }
        self.node_info[node_index] = (dtype.to_string(), kind);

        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.node_value_ptr(node_index);
        let view = create_view(py, ptr, shape, dtype, kind)?;
        self.cached_views[node_index] = Some(view);
        Ok(())
    }
}

#[pymethods]
impl NativeScenario {
    #[new]
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        Self {
            scenario: Some(Scenario::new()),
            error_slot: Arc::new(Mutex::new(None)),
            node_info: Vec::new(),
            cached_views: Vec::new(),
            _rt: rt,
        }
    }

    // -- Goal 1: Native (Rust) operator/source registration ------------------

    /// Register a Rust-native operator by kind + dtype + params.
    ///
    /// If `clock_index` is provided, the operator is triggered by the
    /// clock node instead of its inputs.
    #[pyo3(signature = (kind, dtype, input_indices, shape, params, clock_index=None))]
    fn add_native_operator(
        &mut self,
        py: Python<'_>,
        kind: &str,
        dtype: &str,
        input_indices: Vec<usize>,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
        clock_index: Option<usize>,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(dtype).to_string();
        let (idx, view_kind) = operators::dispatch_native_operator(
            sc,
            kind,
            &dtype_norm,
            &input_indices,
            clock_index,
            params,
        )?;
        self.push_node(py, idx, &dtype_norm, view_kind, &shape)?;
        Ok(idx)
    }

    /// Register a Rust-native source by kind + dtype + params.
    ///
    /// Returns the output node index.
    fn add_native_source(
        &mut self,
        py: Python<'_>,
        kind: &str,
        dtype: &str,
        shape: Vec<usize>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let _guard = self._rt.enter();
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(dtype).to_string();
        let idx = sources::dispatch_native_source(sc, kind, &dtype_norm, params)?;
        self.push_node(py, idx, &dtype_norm, ViewKind::Array, &shape)?;
        Ok(idx)
    }

    // -- Goal 2: Python operator/source registration -------------------------

    /// Register a Python operator.
    ///
    /// `input_types` is a list of `(kind, dtype)` pairs (e.g.
    /// `[("array", "float64"), ("series", "int32")]`).
    /// `output_type` is a `(kind, dtype)` pair for the output node.
    #[pyo3(signature = (input_indices, input_types, output_type, output_shape, py_operator, py_state, clock_index=None))]
    fn add_py_operator(
        &mut self,
        py: Python<'_>,
        input_indices: Vec<usize>,
        input_types: Vec<(String, String)>,
        output_type: (String, String),
        output_shape: Vec<usize>,
        py_operator: PyObject,
        py_state: PyObject,
        clock_index: Option<usize>,
    ) -> PyResult<usize> {
        let (out_kind_str, out_dtype_str) = &output_type;
        let out_dtype = normalise_dtype(out_dtype_str).to_string();

        let out_view_kind = match out_kind_str.as_str() {
            "array" => ViewKind::Array,
            "series" => ViewKind::Series,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported output kind: {other}"
                )));
            }
        };

        // 1. Validate input TypeIds, create output node, wire edges.
        let output_idx = {
            let sc = self.scenario.as_mut().unwrap();

            for (i, (&idx, (kind, dtype))) in
                input_indices.iter().zip(input_types.iter()).enumerate()
            {
                let expected = resolve_type_id(kind, dtype)?;
                let actual = sc.node_type_id(idx);
                if expected != actual {
                    return Err(PyTypeError::new_err(format!(
                        "input {i} (node {idx}): type mismatch — \
                         expected ({kind}, {dtype}), got a different type"
                    )));
                }
            }

            let output_idx = match out_view_kind {
                ViewKind::Array => create_array_node(sc, &output_shape, &out_dtype)?,
                ViewKind::Series => {
                    return Err(PyTypeError::new_err(
                        "Python operators cannot directly produce Series; \
                         output Array and chain a Record operator instead",
                    ));
                }
            };

            match clock_index {
                None => {
                    for &input_idx in &input_indices {
                        sc.add_trigger_edge(input_idx, output_idx);
                    }
                }
                Some(clock_idx) => {
                    sc.add_trigger_edge(clock_idx, output_idx);
                }
            }

            output_idx
        }; // sc borrow dropped

        // 2. Create views.
        self.push_node(py, output_idx, &out_dtype, out_view_kind, &output_shape)?;

        let input_views: Vec<PyObject> = input_indices
            .iter()
            .map(|&idx| {
                self.cached_views[idx]
                    .as_ref()
                    .map(|v| v.clone_ref(py))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!("node {idx} has no cached view"))
                    })
            })
            .collect::<PyResult<_>>()?;
        let py_inputs: PyObject = pyo3::types::PyTuple::new(py, &input_views)?
            .into_any()
            .unbind();
        let py_output = self.cached_views[output_idx]
            .as_ref()
            .map(|v| v.clone_ref(py))
            .unwrap();

        // 3. Build state and attach closure.
        let op_state = PyOperatorState {
            py_operator,
            py_inputs,
            py_output,
            py_state,
            error_slot: self.error_slot.clone(),
        };
        let sc = self.scenario.as_mut().unwrap();
        register_py_closure(sc, output_idx, op_state, &out_dtype)?;

        Ok(output_idx)
    }

    /// Register a channel-based source (for async Python sources).
    ///
    /// Returns `(node_index, hist_sender, live_sender)`.
    fn add_source_raw(
        &mut self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: String,
    ) -> PyResult<(usize, HistoricalEventSender, LiveEventSender)> {
        let _guard = self._rt.enter();
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();
        let (node_index, hist_sender, live_sender) =
            sources::register_channel_source(sc, &shape, &dtype_norm)?;
        self.push_node(py, node_index, &dtype_norm, ViewKind::Array, &shape)?;
        Ok((node_index, hist_sender, live_sender))
    }

    // -- View access ---------------------------------------------------------

    /// Get a cached view for a node.
    fn get_view(&self, py: Python<'_>, node_index: usize) -> PyResult<PyObject> {
        match self.cached_views.get(node_index) {
            Some(Some(view)) => Ok(view.clone_ref(py)),
            _ => Err(PyRuntimeError::new_err(format!(
                "node {node_index} has no Python-representable view"
            ))),
        }
    }

    /// Convenience: get the number of recorded elements (Series nodes).
    fn series_len(&self, node_index: usize) -> usize {
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.node_value_ptr(node_index) as *const u8;
        // All Series<T> have the same len() layout.
        let series = unsafe { &*(ptr as *const Series<f64>) };
        series.len()
    }

    /// Convenience: get recorded timestamps as numpy int64 array.
    fn series_timestamps<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.node_value_ptr(node_index) as *const u8;
        let series = unsafe { &*(ptr as *const Series<f64>) };
        let arr = numpy::ndarray::Array1::from(series.timestamps().to_vec());
        Ok(numpy::PyArray1::from_owned_array(py, arr)
            .into_any()
            .unbind())
    }

    /// Convenience: get recorded values as numpy array.
    fn series_values<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let (dtype, _) = &self.node_info[node_index];
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.node_value_ptr(node_index) as *const u8;
        macro_rules! extract {
            ($T:ty) => {{
                let series = unsafe { &*(ptr as *const Series<$T>) };
                let nd = numpy::ndarray::Array1::from(series.values().to_vec());
                Ok(numpy::PyArray1::from_owned_array(py, nd)
                    .into_any()
                    .unbind())
            }};
        }
        dispatch_dtype!(dtype, extract)
    }

    // -- Execution -----------------------------------------------------------

    /// Run the POCQ event loop.
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut scenario = self.scenario.take().ok_or_else(|| {
            PyRuntimeError::new_err("scenario already consumed by a previous run()")
        })?;

        py.detach(|| {
            self._rt.block_on(scenario.run());
        });

        self.scenario = Some(scenario);

        if let Some(err) = self.error_slot.lock().unwrap().take() {
            Err(PyRuntimeError::new_err(err))
        } else {
            Ok(())
        }
    }

    /// Run with a Python driver on a background thread.
    fn run_with_driver(&mut self, py: Python<'_>, driver: PyObject) -> PyResult<()> {
        let mut scenario = self.scenario.take().ok_or_else(|| {
            PyRuntimeError::new_err("scenario already consumed by a previous run()")
        })?;

        let rt = &self._rt;
        let (scenario, bg_result) = py.detach(move || {
            let bg_handle = std::thread::spawn(move || -> Result<(), String> {
                Python::attach(|py| {
                    driver.call0(py).map_err(|e| e.to_string())?;
                    Ok(())
                })
            });

            rt.block_on(scenario.run());

            let bg_result = bg_handle
                .join()
                .map_err(|_| "background thread panicked".to_string())
                .and_then(|r| r);

            (scenario, bg_result)
        });

        self.scenario = Some(scenario);

        if let Some(err) = self.error_slot.lock().unwrap().take() {
            return Err(PyRuntimeError::new_err(err));
        }

        match bg_result {
            Ok(()) => Ok(()),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }
}

// ---------------------------------------------------------------------------
// Python operator closure (monomorphized per output dtype)
// ---------------------------------------------------------------------------

fn register_py_closure(
    sc: &mut Scenario,
    output_index: usize,
    op_state: PyOperatorState,
    dtype: &str,
) -> PyResult<()> {
    let input_ptrs: Box<[*const u8]> = Box::new([]);

    let state = Box::new(op_state);

    macro_rules! register {
        ($T:ty) => {
            sc.attach_raw_closure(output_index, input_ptrs, py_compute_fn::<$T>, state)
        };
    }
    dispatch_dtype!(dtype, register);

    Ok(())
}

/// Monomorphized compute function for Python operators.
///
/// Calls `operator.compute(timestamp, inputs, output, state)` via GIL.
/// The Python callback writes into the output view; returns `(bool, new_state)`.
///
/// # Safety
///
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn<T: Scalar>(
    _input_ptrs: &[*const u8],
    _output_ptr: *mut u8,
    state_ptr: *mut u8,
    timestamp: i64,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        let result = state.py_operator.call_method1(
            py,
            "compute",
            (
                timestamp,
                &state.py_inputs,
                &state.py_output,
                &state.py_state,
            ),
        )?;

        let tuple = result.bind(py);
        let produced: bool = tuple.get_item(0)?.extract()?;
        let new_state = tuple.get_item(1)?;
        state.py_state = new_state.unbind();

        Ok(produced)
    });

    match result {
        Ok(produced) => produced,
        Err(e) => {
            set_error(&state.error_slot, e.to_string());
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<NativeScenario>()?;
    m.add_class::<_ArrayView>()?;
    m.add_class::<_SeriesView>()?;
    m.add_class::<HistoricalEventSender>()?;
    m.add_class::<LiveEventSender>()?;
    Ok(())
}
