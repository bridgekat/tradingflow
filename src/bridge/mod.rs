//! PyO3 bridge — exposes the Rust `Scenario` runtime to Python.
//!
//! [`NativeScenario`] wraps the Rust [`Scenario`], allowing Python's
//! `Scenario.run()` to delegate the POCQ event loop and DAG propagation
//! to Rust.
//!
//! **Sources** are driven concurrently on a background Python asyncio
//! thread and fed to the Rust POCQ via channels.
//!
//! **Rust operators** are registered via [`NativeOpHandle`] — an opaque
//! handle created by factory pyfunctions (`add`, `negate`, etc.) that
//! captures a fully-monomorphized operator closure.
//!
//! **Python operators** participate via GIL callbacks during flush.

mod benches;
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

use crate::scenario::{Handle, Scenario};
use crate::store::Store;

use dispatch::{dtype_element_bytes, normalise_dtype};
pub use operators::NativeOpHandle;
pub use sources::{HistoricalEventSender, LiveEventSender};
pub use views::StoreView;

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

/// Create a node with given shape/dtype and zero-filled default value.
fn create_node_dispatch(
    sc: &mut Scenario,
    shape: &[usize],
    dtype: &str,
    default_bytes: &[u8],
) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => Ok(sc
            .create_node::<f64>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "float32" => Ok(sc
            .create_node::<f32>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "int64" => Ok(sc
            .create_node::<i64>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "int32" => Ok(sc
            .create_node::<i32>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "uint64" => Ok(sc
            .create_node::<u64>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "uint32" => Ok(sc
            .create_node::<u32>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        "bool" => Ok(sc
            .create_node::<u8>(shape, &sources::bytes_to_vec(default_bytes))
            .index()),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

/// Ensure a node's store keeps full history (window = 0).
fn ensure_windowed_dispatch(sc: &mut Scenario, node_index: usize, dtype: &str) -> PyResult<()> {
    let dtype = normalise_dtype(dtype);
    macro_rules! ensure {
        ($T:ty) => {{
            let h = Handle::<$T>::new(node_index);
            sc.store_mut(h).ensure_min_window(0);
        }};
    }
    match dtype {
        "float64" => ensure!(f64),
        "float32" => ensure!(f32),
        "int64" => ensure!(i64),
        "int32" => ensure!(i32),
        "uint64" => ensure!(u64),
        "uint32" => ensure!(u32),
        "bool" => ensure!(u8),
        other => return Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyOperatorState — per-operator Python callback
// ---------------------------------------------------------------------------

/// Holds the Python objects needed to call a Python operator's `compute()`
/// from Rust's `flush()`.
struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_state: PyObject,
    element_size: usize,
    #[allow(dead_code)]
    dtype_str: String,
    error_slot: ErrorSlot,
}

// SAFETY: PyOperatorState fields are only accessed while holding the GIL.
unsafe impl Send for PyOperatorState {}

// ---------------------------------------------------------------------------
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
#[pyclass]
pub struct NativeScenario {
    scenario: Option<Scenario>,
    error_slot: ErrorSlot,
    node_dtypes: Vec<String>,
}

// SAFETY: NativeScenario is only accessed from a single Python thread at a time.
unsafe impl Send for NativeScenario {}
unsafe impl Sync for NativeScenario {}

#[pymethods]
impl NativeScenario {
    #[new]
    fn new() -> Self {
        Self {
            scenario: Some(Scenario::new()),
            error_slot: Arc::new(Mutex::new(None)),
            node_dtypes: Vec::new(),
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
        let node_index = sources::register_array_source(
            sc,
            &dtype_norm,
            ts_vec,
            values_bytes.to_vec(),
            stride.max(1),
        )?;
        self.node_dtypes.push(dtype_norm);
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
        let node_index = create_node_dispatch(sc, &shape, &dtype_norm, &default_value)?;

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

        // Wire edges from inputs to the new node.
        // We need to add edges and a closure manually since Python operators
        // are not typed through our Operator trait.
        register_py_operator(sc, &input_indices, node_index, op_state);

        self.node_dtypes.push(dtype_norm);
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
        self.node_dtypes.push(handle.dtype_str.clone());
        Ok(idx)
    }

    /// Ensure a node keeps full history (materialize equivalent).
    fn materialize(&mut self, node_index: usize) -> PyResult<()> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype = &self.node_dtypes[node_index];
        ensure_windowed_dispatch(sc, node_index, dtype)
    }

    /// Get a StoreView for a node.
    fn store_view(&self, node_index: usize) -> PyResult<StoreView> {
        let sc = self.scenario.as_ref().unwrap();
        let stride = sc.node_stride(node_index);
        let shape = sc.node_shape(node_index);
        let dtype = &self.node_dtypes[node_index];
        let store_ptr = sc.node_store_ptr(node_index) as *const u8;
        Ok(StoreView::new(
            store_ptr,
            if stride == 1 { vec![] } else { shape.to_vec() },
            stride,
            dtype.clone(),
        ))
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
        self.node_dtypes.push(dtype_norm);
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

    /// Convenience: get store history length.
    fn series_len(&self, node_index: usize) -> PyResult<usize> {
        let sv = self.store_view(node_index)?;
        Ok(sv.__len__())
    }

    /// Convenience: get store timestamps.
    fn series_timestamps<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.store_view(node_index)?;
        sv.index(py)
    }

    /// Convenience: get store values.
    fn series_values<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.store_view(node_index)?;
        sv.values(py)
    }
}

// ---------------------------------------------------------------------------
// Python operator registration (raw closure approach)
// ---------------------------------------------------------------------------

/// Register a Python operator as a raw closure on the scenario graph.
///
/// This creates edges from each input node to the output node and attaches
/// a compute closure that calls back into Python via the GIL.
fn register_py_operator(
    sc: &mut Scenario,
    input_indices: &[usize],
    output_index: usize,
    op_state: PyOperatorState,
) {
    // Collect input store pointers.
    let input_ptrs: Box<[*const u8]> = input_indices
        .iter()
        .map(|&idx| sc.node_store_ptr(idx) as *const u8)
        .collect();

    // Wire edges.
    for &input_idx in input_indices {
        sc.add_edge(input_idx, output_index);
    }

    // Attach closure.
    let state = Box::new(op_state);
    sc.attach_raw_closure(output_index, input_ptrs, py_compute_fn, state);
}

/// Type-erased compute function for Python operators.
///
/// # Safety
///
/// * `input_ptrs` — unused (Python inputs use their own view objects).
/// * `output_ptr` — `*mut u8` pointing to a `Store<u8>` (raw bytes).
/// * `state_ptr` — `*mut u8` pointing to a `PyOperatorState`.
/// * `timestamp` — flush timestamp.
unsafe fn py_compute_fn(
    _input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    state_ptr: *mut u8,
    timestamp: i64,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        // Call operator.compute(inputs, state).
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

        // Convert value to contiguous array and write to Store's value buffer.
        let np = py.import("numpy")?;
        let contiguous = np.call_method1("ascontiguousarray", (&raw_value,))?;
        let interface = contiguous.getattr("__array_interface__")?;
        let data_tuple = interface.get_item("data")?;
        let ptr_int: usize = data_tuple.get_item(0)?.extract()?;

        // Write raw bytes into the output store's current buffer.
        // We treat the store as Store<u8> since all Store<T> have the same
        // physical memory layout.
        let store = unsafe { &mut *(output_ptr as *mut Store<u8>) };
        store.push_default(timestamp);
        unsafe {
            let view = store.current_view_mut();
            std::ptr::copy_nonoverlapping(
                ptr_int as *const u8,
                view.values.as_mut_ptr(),
                state.element_size,
            );
        }
        store.commit();
        let produced = true;

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
    m.add_class::<StoreView>()?;
    m.add_class::<HistoricalEventSender>()?;
    m.add_class::<LiveEventSender>()?;
    m.add_class::<NativeOpHandle>()?;
    m.add_class::<benches::BenchResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(operators::add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::subtract, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::multiply, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::divide, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::negate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::select, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::concat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(operators::stack, m)?)?;
    // Benchmark functions (mirrors benches/bench_add.rs)
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_baseline_add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        benches::bench_baseline_add_series,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_store_add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_store_add_series, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_store_compute, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        benches::bench_store_compute_series,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_scenario_operator, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        benches::bench_scenario_operator_series,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_scenario_chain, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(benches::bench_scenario_sparse, m)?)?;
    Ok(())
}
