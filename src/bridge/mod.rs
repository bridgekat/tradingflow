//! PyO3 bridge — exposes the new generalized Rust runtime to Python.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`](crate::scenario::Scenario),
//! allowing Python's `Scenario.run()` to delegate the POCQ event loop and
//! DAG propagation to Rust.
//!
//! # Python operator restrictions
//!
//! Python operators registered via [`NativeScenario::add_py_operator`] have
//! their `compute(inputs, state)` callback invoked during Rust's synchronous
//! `flush()`.  The following restrictions apply:
//!
//! 1. **No re-entrant scenario access.**  The `compute` callback MUST NOT
//!    call back into the `NativeScenario` instance (e.g. `store_view()`,
//!    `run()`, `record()`).  The scenario is mutably borrowed during
//!    `run()`; re-entering causes `RuntimeError: Already mutably borrowed`.
//!
//! 2. **Use pre-captured views.**  Input data should be read via `StoreView`
//!    objects captured before `run()` and passed as `py_inputs`.  These hold
//!    raw pointers to node values that remain valid throughout the scenario's
//!    lifetime.
//!
//! 3. **No long-running compute.**  The callback runs under the GIL during
//!    the synchronous DAG flush.  Long-running Python code blocks the entire
//!    DAG propagation.

mod dispatch;
mod operators;
mod sources;
mod views;

use std::sync::{Arc, Mutex};

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

type PyObject = Py<PyAny>;

use crate::array::Array;
use crate::scenario::Scenario;
use crate::series::Series;

use dispatch::{dtype_element_bytes, normalise_dtype};
pub use operators::NativeOpHandle;
pub use sources::{HistoricalEventSender, LiveEventSender};
pub use views::{NodeKind, StoreView};

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
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Create a node holding `Array<T>` with given shape/dtype.
fn create_array_node_dispatch(
    sc: &mut Scenario,
    shape: &[usize],
    dtype: &str,
    default_bytes: &[u8],
) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    macro_rules! create {
        ($T:ty) => {{
            let values = sources::bytes_to_vec::<$T>(default_bytes);
            let arr = Array::from_vec(shape, values);
            sc.create_node(arr).index()
        }};
    }
    match dtype {
        "float64" => Ok(create!(f64)),
        "float32" => Ok(create!(f32)),
        "int64" => Ok(create!(i64)),
        "int32" => Ok(create!(i32)),
        "uint64" => Ok(create!(u64)),
        "uint32" => Ok(create!(u32)),
        "bool" => Ok(create!(u8)),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

/// Add a record operator for a given Array node.
fn record_dispatch(sc: &mut Scenario, node_index: usize, dtype: &str) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    macro_rules! mat {
        ($T:ty) => {{
            use crate::operators::Record;
            use crate::scenario::handle::Handle;
            let h = Handle::<Array<$T>>::new(node_index);
            sc.add_operator(Record::<$T>::new(), (h,)).index()
        }};
    }
    match dtype {
        "float64" => Ok(mat!(f64)),
        "float32" => Ok(mat!(f32)),
        "int64" => Ok(mat!(i64)),
        "int32" => Ok(mat!(i32)),
        "uint64" => Ok(mat!(u64)),
        "uint32" => Ok(mat!(u32)),
        "bool" => Ok(mat!(u8)),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// PyOperatorState — per-operator Python callback
// ---------------------------------------------------------------------------

struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_state: PyObject,
    element_size: usize,
    #[allow(dead_code)]
    dtype_str: String,
    error_slot: ErrorSlot,
}

unsafe impl Send for PyOperatorState {}

// ---------------------------------------------------------------------------
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
#[pyclass]
pub struct NativeScenario {
    scenario: Option<Scenario>,
    error_slot: ErrorSlot,
    /// Per-node: (dtype_str, kind).
    node_info: Vec<(String, NodeKind)>,
}

unsafe impl Send for NativeScenario {}
unsafe impl Sync for NativeScenario {}

#[pymethods]
impl NativeScenario {
    #[new]
    fn new() -> Self {
        Self {
            scenario: Some(Scenario::new()),
            error_slot: Arc::new(Mutex::new(None)),
            node_info: Vec::new(),
        }
    }

    /// Register a source with pre-drained array data.
    fn add_source(
        &mut self,
        _py: Python<'_>,
        _shape: Vec<usize>,
        dtype: String,
        timestamps: PyReadonlyArrayDyn<'_, i64>,
        values_bytes: &[u8],
        stride: usize,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();
        let ts_vec = timestamps.as_slice()?.to_vec();
        let node_index =
            sources::register_array_source(sc, &dtype_norm, ts_vec, values_bytes.to_vec(), stride)?;
        self.node_info.push((dtype_norm, NodeKind::Array));
        Ok(node_index)
    }

    /// Register a Python operator.
    fn add_py_operator(
        &mut self,
        _py: Python<'_>,
        input_indices: Vec<usize>,
        shape: Vec<usize>,
        dtype: String,
        default_value: Vec<u8>,
        py_operator: PyObject,
        py_inputs: PyObject,
        py_state: PyObject,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();

        // Create a properly typed Array<T> node.
        let node_index = create_array_node_dispatch(sc, &shape, &dtype_norm, &default_value)?;

        let stride: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let elem_bytes = dtype_element_bytes(&dtype_norm)?;
        let element_size = stride * elem_bytes;

        let op_state = PyOperatorState {
            py_operator,
            py_inputs,
            py_state,
            element_size,
            dtype_str: dtype_norm.clone(),
            error_slot: self.error_slot.clone(),
        };

        // Dispatch to monomorphized py_compute_fn<T> based on dtype.
        register_py_operator(sc, &input_indices, node_index, op_state, &dtype_norm)?;

        self.node_info.push((dtype_norm, NodeKind::Array));
        Ok(node_index)
    }

    /// Register a Rust-native operator from an opaque handle.
    fn register_handle_operator(
        &mut self,
        handle: &mut NativeOpHandle,
        input_indices: Vec<usize>,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let idx = handle.take_and_register(sc, &input_indices)?;
        self.node_info
            .push((handle.dtype_str.clone(), NodeKind::Array));
        Ok(idx)
    }

    /// Record a node into a Series (adds a record operator).
    ///
    /// Returns the index of the new Series node.
    fn record(&mut self, node_index: usize) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let (dtype, _) = &self.node_info[node_index];
        let dtype = dtype.clone();
        let series_idx = record_dispatch(sc, node_index, &dtype)?;
        self.node_info.push((dtype, NodeKind::Series));
        Ok(series_idx)
    }

    /// Get a StoreView for a node.
    fn store_view(&self, node_index: usize) -> PyResult<StoreView> {
        let sc = self.scenario.as_ref().unwrap();
        let (dtype, kind) = &self.node_info[node_index];
        let value_ptr = sc.node_value_ptr(node_index) as *const u8;

        // Determine shape from the stored value.
        let shape = match kind {
            NodeKind::Array => {
                // All Array<T> have the same layout; use f64 to read shape.
                let arr = unsafe { &*(value_ptr as *const Array<f64>) };
                arr.shape().to_vec()
            }
            NodeKind::Series => {
                let series = unsafe { &*(value_ptr as *const Series<f64>) };
                series.shape().to_vec()
            }
        };

        Ok(StoreView::new(value_ptr, *kind, shape, dtype.clone()))
    }

    /// Register a channel-based source (for async Python sources).
    fn add_channel_source(
        &mut self,
        _py: Python<'_>,
        shape: Vec<usize>,
        dtype: String,
    ) -> PyResult<(usize, HistoricalEventSender, LiveEventSender)> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();
        let (node_index, hist_sender, live_sender) =
            sources::register_channel_source(sc, &shape, &dtype_norm)?;
        self.node_info.push((dtype_norm, NodeKind::Array));
        Ok((node_index, hist_sender, live_sender))
    }

    /// Run the POCQ event loop.
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut scenario = self.scenario.take().ok_or_else(|| {
            PyRuntimeError::new_err("scenario already consumed by a previous run()")
        })?;

        py.detach(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(scenario.run());
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

        let (scenario, bg_result) = py.detach(move || {
            let bg_handle = std::thread::spawn(move || -> Result<(), String> {
                Python::attach(|py| {
                    driver.call0(py).map_err(|e| e.to_string())?;
                    Ok(())
                })
            });

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
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

    /// Convenience: get the number of recorded elements.
    fn series_len(&self, node_index: usize) -> PyResult<usize> {
        let sv = self.store_view(node_index)?;
        Ok(sv.__len__())
    }

    /// Convenience: get recorded timestamps.
    fn series_timestamps<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.store_view(node_index)?;
        sv.index(py)
    }

    /// Convenience: get recorded values.
    fn series_values<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.store_view(node_index)?;
        sv.values(py)
    }
}

// ---------------------------------------------------------------------------
// Python operator registration (monomorphized per dtype)
// ---------------------------------------------------------------------------

use crate::types::Scalar;

fn register_py_operator(
    sc: &mut Scenario,
    input_indices: &[usize],
    output_index: usize,
    op_state: PyOperatorState,
    dtype: &str,
) -> PyResult<()> {
    let input_ptrs: Box<[*const u8]> = input_indices
        .iter()
        .map(|&idx| sc.node_value_ptr(idx) as *const u8)
        .collect();

    for &input_idx in input_indices {
        sc.add_edge(input_idx, output_index);
    }

    let state = Box::new(op_state);

    macro_rules! register {
        ($T:ty) => {
            sc.attach_raw_closure(output_index, input_ptrs, py_compute_fn::<$T>, state)
        };
    }

    match normalise_dtype(dtype) {
        "float64" => register!(f64),
        "float32" => register!(f32),
        "int64" => register!(i64),
        "int32" => register!(i32),
        "uint64" => register!(u64),
        "uint32" => register!(u32),
        "bool" => register!(u8),
        other => {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype for Python operator: {other}"
            )));
        }
    }

    Ok(())
}

/// Monomorphized compute function for Python operators.
///
/// `T` matches the `Array<T>` stored in the output node.  The Python callback
/// produces a numpy array; we copy its raw data into `Array<T>::as_slice_mut()`.
///
/// # Safety
///
/// * `output_ptr` must point to a valid `Array<T>`.
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn<T: Scalar>(
    _input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    state_ptr: *mut u8,
    _timestamp: i64,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        let result =
            state
                .py_operator
                .call_method1(py, "compute", (&state.py_inputs, &state.py_state))?;

        let tuple = result.bind(py);
        let raw_value = tuple.get_item(0)?;
        let new_state = tuple.get_item(1)?;
        state.py_state = new_state.unbind();

        if raw_value.is_none() {
            return Ok(false);
        }

        // Convert value to contiguous numpy array.
        let np = py.import("numpy")?;
        let contiguous = np.call_method1("ascontiguousarray", (&raw_value,))?;
        let interface = contiguous.getattr("__array_interface__")?;
        let data_tuple = interface.get_item("data")?;
        let ptr_int: usize = data_tuple.get_item(0)?.extract()?;

        // Copy typed data into Array<T>.
        let arr = unsafe { &mut *(output_ptr as *mut Array<T>) };
        let dst = arr.as_slice_mut();
        unsafe {
            std::ptr::copy_nonoverlapping(ptr_int as *const T, dst.as_mut_ptr(), dst.len());
        }

        Ok(true)
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
    m.add_class::<StoreView>()?;
    m.add_class::<HistoricalEventSender>()?;
    m.add_class::<LiveEventSender>()?;
    m.add_class::<NativeOpHandle>()?;
    m.add_function(pyo3::wrap_pyfunction!(operators::add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::subtract, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::multiply, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::divide, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::negate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::select, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::concat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::stack, m)?)?;
    Ok(())
}
