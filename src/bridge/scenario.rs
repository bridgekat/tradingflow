//! Python-visible scenario wrapper.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`](crate::scenario::Scenario)
//! and provides registration entry points for sources and operators from
//! both Rust and Python.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::{Scenario, Series};

use super::dispatch::{dispatch_dtype, normalize_dtype};
use super::sources::EventSender;
use super::views::{ViewKind, create_view};
use super::{ErrorSlot, operator, operators, sources};

type PyObject = Py<PyAny>;

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
        let ptr = sc.value_ptr(node_index);
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

    // -- Native (Rust) operator/source registration --------------------------

    /// Register a Rust-native source by kind + dtype + params.
    #[pyo3(signature = (kind, dtype, shape, params))]
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
        let dtype_norm = normalize_dtype(dtype).to_string();
        let idx = sources::dispatch_native_source(sc, kind, &dtype_norm, params)?;
        self.push_node(py, idx, &dtype_norm, ViewKind::Array, &shape)?;
        Ok(idx)
    }

    /// Register a Rust-native operator by kind + dtype + params.
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
        let _guard = self._rt.enter();
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalize_dtype(dtype).to_string();
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

    // -- Python operator/source registration ---------------------------------

    /// Register a channel-based source (for async Python sources).
    ///
    /// Returns `(node_index, hist_sender, live_sender)`.
    fn add_py_source(
        &mut self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: String,
    ) -> PyResult<(usize, EventSender, EventSender)> {
        let _guard = self._rt.enter();
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalize_dtype(&dtype).to_string();
        let (node_index, hist_sender, live_sender) =
            sources::register_channel_source(sc, &shape, &dtype_norm)?;
        self.push_node(py, node_index, &dtype_norm, ViewKind::Array, &shape)?;
        Ok((node_index, hist_sender, live_sender))
    }

    /// Register a Python operator via
    /// [`ErasedOperator`](crate::operator::ErasedOperator).
    ///
    /// `input_types` is a list of `(kind, dtype)` pairs (e.g.
    /// `[("array", "float64"), ("series", "int32")]`).
    /// `output_type` is a `(kind, dtype)` pair for the output node (e.g.
    /// `("array", "float64")` or `("series", "float64")`).
    ///
    /// Input type validation is performed by
    /// [`Scenario::add_erased_operator`].
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
        let _guard = self._rt.enter();
        let (out_kind_str, out_dtype_str) = &output_type;
        let out_dtype = normalize_dtype(out_dtype_str).to_string();

        let out_view_kind = match out_kind_str.as_str() {
            "array" => ViewKind::Array,
            "series" => ViewKind::Series,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported output kind: {other}"
                )));
            }
        };

        // 1. Resolve TypeIds from Python-declared types.
        let input_type_ids = input_types
            .iter()
            .map(|(kind, dtype)| operator::resolve_type_id(kind, dtype))
            .collect::<PyResult<Box<[_]>>>()?;
        let output_type_id = operator::resolve_type_id(out_kind_str, &out_dtype)?;

        // 2. Build input views (input nodes already exist).
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

        // 3. Construct the erased operator.
        let erased = operator::make_py_operator(
            py,
            input_type_ids,
            output_type_id,
            py_inputs,
            py_operator,
            py_state,
            self.error_slot.clone(),
            &out_dtype,
            out_view_kind,
            &output_shape,
        )?;

        // 4. Register via the unified path (validates input TypeIds).
        let sc = self.scenario.as_mut().unwrap();
        let output_idx = sc.add_erased_operator(erased, &input_indices, clock_index);

        // 5. Cache output node metadata and view.
        self.push_node(py, output_idx, &out_dtype, out_view_kind, &output_shape)?;

        Ok(output_idx)
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
    fn series_len(&self, node_index: usize) -> PyResult<usize> {
        let (dtype, _) = &self.node_info[node_index];
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.value_ptr(node_index) as *const u8;
        macro_rules! get_len {
            ($T:ty) => {{
                let series = unsafe { &*(ptr as *const Series<$T>) };
                series.len()
            }};
        }
        Ok(dispatch_dtype!(dtype, get_len))
    }

    /// Convenience: get recorded timestamps as numpy int64 array.
    fn series_timestamps<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let (dtype, _) = &self.node_info[node_index];
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.value_ptr(node_index) as *const u8;
        macro_rules! get_ts {
            ($T:ty) => {{
                let series = unsafe { &*(ptr as *const Series<$T>) };
                let arr = numpy::ndarray::Array1::from(series.timestamps().to_vec());
                Ok(numpy::PyArray1::from_owned_array(py, arr)
                    .into_any()
                    .unbind())
            }};
        }
        dispatch_dtype!(dtype, get_ts)
    }

    /// Convenience: get recorded values as numpy array.
    fn series_values<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let (dtype, _) = &self.node_info[node_index];
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.value_ptr(node_index) as *const u8;
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
