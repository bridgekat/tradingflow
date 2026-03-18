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
//! Each Python operator's inputs are presented as [`ObservableView`] /
//! [`SeriesView`] objects that read directly from Rust memory.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use numpy::ndarray::{Array1, ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::mpsc;

type PyObject = Py<PyAny>;

use crate::input::Obs;
use crate::observable::{Observable, ObservableHandle};
use crate::operators;
use crate::scenario::Scenario;
use crate::series::Series;
use crate::source::{self, ArraySource, HistoricalIter, LiveIter, Source};

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
// Dtype dispatch helpers
// ---------------------------------------------------------------------------

fn dtype_element_bytes(dtype: &str) -> PyResult<usize> {
    match dtype {
        "float64" | "<f8" => Ok(8),
        "float32" | "<f4" => Ok(4),
        "int64" | "<i8" => Ok(8),
        "int32" | "<i4" => Ok(4),
        "uint64" | "<u8" => Ok(8),
        "uint32" | "<u4" => Ok(4),
        "bool" | "|b1" => Ok(1),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

/// Normalise a numpy dtype string to a canonical form.
fn normalise_dtype(dtype: &str) -> &str {
    match dtype {
        "float64" | "<f8" => "float64",
        "float32" | "<f4" => "float32",
        "int64" | "<i8" => "int64",
        "int32" | "<i4" => "int32",
        "uint64" | "<u8" => "uint64",
        "uint32" | "<u4" => "uint32",
        "bool" | "|b1" => "bool",
        other => other,
    }
}

fn add_source_dispatch(
    py: Python<'_>,
    sc: &mut Scenario,
    shape: &[usize],
    dtype: &str,
    initial: &PyObject,
) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => {
            let arr: PyReadonlyArrayDyn<f64> = initial.extract(py)?;
            Ok(sc.add_source_typed::<f64>(shape, arr.as_slice()?))
        }
        "float32" => {
            let arr: PyReadonlyArrayDyn<f32> = initial.extract(py)?;
            Ok(sc.add_source_typed::<f32>(shape, arr.as_slice()?))
        }
        "int64" => {
            let arr: PyReadonlyArrayDyn<i64> = initial.extract(py)?;
            Ok(sc.add_source_typed::<i64>(shape, arr.as_slice()?))
        }
        "int32" => {
            let arr: PyReadonlyArrayDyn<i32> = initial.extract(py)?;
            Ok(sc.add_source_typed::<i32>(shape, arr.as_slice()?))
        }
        "uint64" => {
            let arr: PyReadonlyArrayDyn<u64> = initial.extract(py)?;
            Ok(sc.add_source_typed::<u64>(shape, arr.as_slice()?))
        }
        "uint32" => {
            let arr: PyReadonlyArrayDyn<u32> = initial.extract(py)?;
            Ok(sc.add_source_typed::<u32>(shape, arr.as_slice()?))
        }
        "bool" => {
            let arr: PyReadonlyArrayDyn<u8> = initial.extract(py)?;
            Ok(sc.add_source_typed::<u8>(shape, arr.as_slice()?))
        }
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

fn register_array_source_dispatch(
    sc: &mut Scenario,
    node_index: usize,
    dtype: &str,
    timestamps: Vec<i64>,
    values_bytes: Vec<u8>,
    stride: usize,
) -> PyResult<()> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => {
            let values = bytes_to_vec::<f64>(&values_bytes);
            sc.register_source_typed::<f64>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "float32" => {
            let values = bytes_to_vec::<f32>(&values_bytes);
            sc.register_source_typed::<f32>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "int64" => {
            let values = bytes_to_vec::<i64>(&values_bytes);
            sc.register_source_typed::<i64>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "int32" => {
            let values = bytes_to_vec::<i32>(&values_bytes);
            sc.register_source_typed::<i32>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "uint64" => {
            let values = bytes_to_vec::<u64>(&values_bytes);
            sc.register_source_typed::<u64>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "uint32" => {
            let values = bytes_to_vec::<u32>(&values_bytes);
            sc.register_source_typed::<u32>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        "bool" => {
            let values = bytes_to_vec::<u8>(&values_bytes);
            sc.register_source_typed::<u8>(node_index, Box::new(ArraySource::new(timestamps, values, stride)));
        }
        other => return Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
    Ok(())
}

fn create_node_dispatch(sc: &mut Scenario, shape: &[usize], dtype: &str) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => Ok(sc.create_node_typed::<f64>(shape)),
        "float32" => Ok(sc.create_node_typed::<f32>(shape)),
        "int64" => Ok(sc.create_node_typed::<i64>(shape)),
        "int32" => Ok(sc.create_node_typed::<i32>(shape)),
        "uint64" => Ok(sc.create_node_typed::<u64>(shape)),
        "uint32" => Ok(sc.create_node_typed::<u32>(shape)),
        "bool" => Ok(sc.create_node_typed::<u8>(shape)),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// NativeOpHandle — opaque operator handle for Python
// ---------------------------------------------------------------------------

/// Registration closure: given a Scenario, input indices, and output shape,
/// registers the operator and returns the output node index.
type RegisterFn = Box<dyn FnOnce(&mut Scenario, &[usize], &[usize]) -> usize + Send + Sync>;

/// Opaque handle holding a pre-constructed, type-erased Rust operator.
///
/// Created by Python-callable factory functions (`add`, `negate`, etc.) and
/// consumed by [`NativeScenario::register_handle_operator`].  The handle is
/// single-use: the inner closure is consumed on registration.
#[pyclass]
pub struct NativeOpHandle {
    register_fn: Option<RegisterFn>,
    dtype_str: String,
}

// SAFETY: The captured operator is Send+Sync (plain data structs with PhantomData).
// NativeOpHandle is only mutated via &mut self in register_handle_operator.
unsafe impl Send for NativeOpHandle {}
unsafe impl Sync for NativeOpHandle {}

impl NativeOpHandle {
    /// Binary operator with two observable inputs of the same type.
    fn binary_obs<T, Op>(op: Op, dtype_str: String) -> Self
    where
        T: Copy + Send + Sync + 'static,
        Op: crate::operator::Operator<Output = T> + Send + Sync + 'static,
        for<'a> Op: crate::operator::Operator<Inputs<'a> = (&'a Observable<T>, &'a Observable<T>)>,
    {
        Self {
            register_fn: Some(Box::new(move |sc, inputs, shape| {
                let h0 = ObservableHandle::<T>::new(inputs[0]);
                let h1 = ObservableHandle::<T>::new(inputs[1]);
                sc.add_operator::<(Obs<T>, Obs<T>), _>((h0, h1), shape, op)
                    .index
            })),
            dtype_str,
        }
    }

    /// Unary operator with one observable input.
    fn unary_obs<T, Op>(op: Op, dtype_str: String) -> Self
    where
        T: Copy + Send + Sync + 'static,
        Op: crate::operator::Operator<Output = T> + Send + Sync + 'static,
        for<'a> Op: crate::operator::Operator<Inputs<'a> = (&'a Observable<T>,)>,
    {
        Self {
            register_fn: Some(Box::new(move |sc, inputs, shape| {
                let h0 = ObservableHandle::<T>::new(inputs[0]);
                sc.add_operator::<(Obs<T>,), _>((h0,), shape, op).index
            })),
            dtype_str,
        }
    }

    /// Variable-arity operator with homogeneous observable inputs.
    fn slice_obs<T, Op>(op: Op, dtype_str: String) -> Self
    where
        T: Copy + Send + Sync + 'static,
        Op: crate::operator::Operator<Output = T> + Send + Sync + 'static,
        for<'a> Op: crate::operator::Operator<Inputs<'a> = &'a [&'a Observable<T>]>,
    {
        Self {
            register_fn: Some(Box::new(move |sc, inputs, shape| {
                let handles: Vec<ObservableHandle<T>> =
                    inputs.iter().map(|&i| ObservableHandle::new(i)).collect();
                sc.add_slice_operator::<Obs<T>, _>(&handles, shape, op)
                    .index
            })),
            dtype_str,
        }
    }
}

// ---------------------------------------------------------------------------
// Macro-generated operator factory pyfunctions
// ---------------------------------------------------------------------------

/// Generate a `#[pyfunction]` that creates a binary-obs `NativeOpHandle`.
///
/// The pyfunction name must match the Rust operator factory in
/// `crate::operators` (e.g. `add` → `operators::add::<f64>()`).
macro_rules! def_binary_op {
    ($py_name:ident) => {
        #[pyfunction]
        fn $py_name(dtype: &str) -> PyResult<NativeOpHandle> {
            let d = normalise_dtype(dtype).to_string();
            match d.as_str() {
                "float64" => Ok(NativeOpHandle::binary_obs(operators::$py_name::<f64>(), d)),
                "float32" => Ok(NativeOpHandle::binary_obs(operators::$py_name::<f32>(), d)),
                other => Err(PyTypeError::new_err(format!(
                    "Rust operator '{}' does not support dtype '{other}'",
                    stringify!($py_name),
                ))),
            }
        }
    };
}

/// Generate a `#[pyfunction]` that creates a unary-obs `NativeOpHandle`.
macro_rules! def_unary_op {
    ($py_name:ident) => {
        #[pyfunction]
        fn $py_name(dtype: &str) -> PyResult<NativeOpHandle> {
            let d = normalise_dtype(dtype).to_string();
            match d.as_str() {
                "float64" => Ok(NativeOpHandle::unary_obs(operators::$py_name::<f64>(), d)),
                "float32" => Ok(NativeOpHandle::unary_obs(operators::$py_name::<f32>(), d)),
                other => Err(PyTypeError::new_err(format!(
                    "Rust operator '{}' does not support dtype '{other}'",
                    stringify!($py_name),
                ))),
            }
        }
    };
}

def_binary_op!(add);
def_binary_op!(subtract);
def_binary_op!(multiply);
def_binary_op!(divide);
def_unary_op!(negate);

// -- Parameterised operators (hand-written) ----------------------------------

#[pyfunction]
fn select(dtype: &str, indices: Vec<usize>) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => Ok(NativeOpHandle::unary_obs(operators::Select::<f64>::flat(indices), d)),
        "float32" => Ok(NativeOpHandle::unary_obs(operators::Select::<f32>::flat(indices), d)),
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'select' does not support dtype '{other}'"
        ))),
    }
}

#[pyfunction]
fn concat(dtype: &str, input_shape: Vec<usize>, axis: usize) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => Ok(NativeOpHandle::slice_obs(
            operators::Concat::<f64>::new(&input_shape, axis), d,
        )),
        "float32" => Ok(NativeOpHandle::slice_obs(
            operators::Concat::<f32>::new(&input_shape, axis), d,
        )),
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'concat' does not support dtype '{other}'"
        ))),
    }
}

#[pyfunction]
fn stack(dtype: &str, input_shape: Vec<usize>, axis: usize) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => Ok(NativeOpHandle::slice_obs(
            operators::Stack::<f64>::new(&input_shape, axis), d,
        )),
        "float32" => Ok(NativeOpHandle::slice_obs(
            operators::Stack::<f32>::new(&input_shape, axis), d,
        )),
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'stack' does not support dtype '{other}'"
        ))),
    }
}

fn materialize_dispatch(sc: &mut Scenario, node_index: usize, dtype: &str) -> PyResult<()> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => sc.materialize_by_index::<f64>(node_index),
        "float32" => sc.materialize_by_index::<f32>(node_index),
        "int64" => sc.materialize_by_index::<i64>(node_index),
        "int32" => sc.materialize_by_index::<i32>(node_index),
        "uint64" => sc.materialize_by_index::<u64>(node_index),
        "uint32" => sc.materialize_by_index::<u32>(node_index),
        "bool" => sc.materialize_by_index::<u8>(node_index),
        other => return Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
    Ok(())
}

/// Reinterpret a byte buffer as a Vec<T>.
fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
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

// ---------------------------------------------------------------------------
// Channel-based source (for concurrent async Python sources)
// ---------------------------------------------------------------------------

/// Source that receives events from a Python background thread via channels.
struct ChannelSource {
    hist_rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    live_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    element_size: usize,
}

impl Source for ChannelSource {
    fn subscribe(
        self: Box<Self>,
    ) -> Pin<Box<dyn Future<Output = (Box<dyn HistoricalIter>, Box<dyn LiveIter>)>>> {
        Box::pin(async move {
            let hist: Box<dyn HistoricalIter> = Box::new(ChannelHistoricalIter {
                rx: self.hist_rx,
                element_size: self.element_size,
            });
            let live: Box<dyn LiveIter> = Box::new(ChannelLiveIter {
                rx: self.live_rx,
                element_size: self.element_size,
            });
            (hist, live)
        })
    }
}

struct ChannelHistoricalIter {
    rx: mpsc::UnboundedReceiver<(i64, Vec<u8>)>,
    element_size: usize,
}

impl HistoricalIter for ChannelHistoricalIter {
    fn next_into<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = Option<i64>> + 'a>> {
        Box::pin(async move {
            match self.rx.recv().await {
                Some((ts, bytes)) => {
                    let n = self.element_size.min(bytes.len()).min(buf.len());
                    buf[..n].copy_from_slice(&bytes[..n]);
                    Some(ts)
                }
                None => None,
            }
        })
    }
}

struct ChannelLiveIter {
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    element_size: usize,
}

impl LiveIter for ChannelLiveIter {
    fn next_into<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = bool> + 'a>> {
        Box::pin(async move {
            match self.rx.recv().await {
                Some(bytes) => {
                    let n = self.element_size.min(bytes.len()).min(buf.len());
                    buf[..n].copy_from_slice(&bytes[..n]);
                    true
                }
                None => false,
            }
        })
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
    /// Send a historical event.
    fn send(&self, py: Python<'_>, ts_ns: i64, value: PyObject) -> PyResult<()> {
        let bytes = extract_value_bytes(py, &value, self.element_size)?;
        if let Some(ref tx) = self.tx {
            tx.send((ts_ns, bytes))
                .map_err(|_| PyRuntimeError::new_err("historical channel closed"))?;
        }
        Ok(())
    }

    /// Close the channel (signals exhaustion).
    fn close(&mut self) {
        self.tx.take();
    }
}

/// Sends live value events (no timestamp) from Python to Rust.
#[pyclass]
pub struct LiveEventSender {
    tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    element_size: usize,
}

unsafe impl Send for LiveEventSender {}
unsafe impl Sync for LiveEventSender {}

#[pymethods]
impl LiveEventSender {
    /// Send a live event.
    fn send(&self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        let bytes = extract_value_bytes(py, &value, self.element_size)?;
        if let Some(ref tx) = self.tx {
            tx.send(bytes)
                .map_err(|_| PyRuntimeError::new_err("live channel closed"))?;
        }
        Ok(())
    }

    /// Close the channel (signals exhaustion).
    fn close(&mut self) {
        self.tx.take();
    }
}

/// Extract raw bytes from a numpy array.
fn extract_value_bytes(py: Python<'_>, value: &PyObject, element_size: usize) -> PyResult<Vec<u8>> {
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
// ObservableView
// ---------------------------------------------------------------------------

/// Python-visible view of a Rust `Observable`.
///
/// Duck-type compatible with Python `Observable`: provides `.last`, `.shape`,
/// `.dtype` properties that Python operators read during `compute()`.
#[pyclass]
pub struct ObservableView {
    obs_ptr: *const u8,
    shape: Vec<usize>,
    stride: usize,
    dtype_str: String,
}

// SAFETY: ObservableView holds a raw pointer into Scenario-owned memory.
// It is only dereferenced while holding the GIL, on threads that hold the
// GIL.  The pointed-to Observable is stable (never reallocated).
unsafe impl Send for ObservableView {}
unsafe impl Sync for ObservableView {}

#[pymethods]
impl ObservableView {
    /// The current value as a numpy array (copy).
    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        observable_to_numpy(py, self.obs_ptr, &self.shape, &self.dtype_str)
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(pyo3::types::PyTuple::new(py, &self.shape)?.into_any().unbind())
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (&self.dtype_str,))?.unbind())
    }
}

/// Read an Observable's value and return as numpy array.
fn observable_to_numpy(
    py: Python<'_>,
    obs_ptr: *const u8,
    shape: &[usize],
    dtype_str: &str,
) -> PyResult<PyObject> {
    let ix = if shape.is_empty() { &[][..] } else { shape };
    let dtype = normalise_dtype(dtype_str);
    match dtype {
        "float64" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<f64>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "float32" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<f32>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "int64" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<i64>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "int32" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<i32>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "uint64" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<u64>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "uint32" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<u32>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        "bool" => {
            let obs = unsafe { &*(obs_ptr as *const Observable<u8>) };
            let arr = ArrayD::from_shape_vec(IxDyn(ix), obs.last().to_vec()).unwrap();
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }
        _ => Err(PyTypeError::new_err(format!("unsupported dtype: {dtype_str}"))),
    }
}

// ---------------------------------------------------------------------------
// SeriesView
// ---------------------------------------------------------------------------

/// Python-visible view of a Rust `Series`.
///
/// Duck-type compatible with Python `Series`: provides `.index`, `.values`,
/// `.last`, `__len__`, `.shape`, `.dtype`.
#[pyclass]
pub struct SeriesView {
    series_ptr: *const u8,
    shape: Vec<usize>,
    dtype_str: String,
}

unsafe impl Send for SeriesView {}
unsafe impl Sync for SeriesView {}

#[pymethods]
impl SeriesView {
    /// Timestamps as numpy int64 array (nanoseconds).
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyObject {
        // Timestamps are always i64, regardless of value dtype.
        let series = unsafe { &*(self.series_ptr as *const Series<f64>) };
        let arr = Array1::from_vec(series.timestamps_to_vec());
        PyArray1::from_owned_array(py, arr).into_any().unbind()
    }

    /// Values as numpy array (copy).
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        series_values_to_numpy(py, self.series_ptr, &self.shape, &self.dtype_str)
    }

    /// Last value as numpy array (copy).
    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        series_last_to_numpy(py, self.series_ptr, &self.shape, &self.dtype_str)
    }

    fn __len__(&self) -> usize {
        let series = unsafe { &*(self.series_ptr as *const Series<f64>) };
        series.len()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(pyo3::types::PyTuple::new(py, &self.shape)?.into_any().unbind())
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (&self.dtype_str,))?.unbind())
    }
}

macro_rules! series_values_match {
    ($py:expr, $ptr:expr, $shape:expr, $T:ty) => {{
        let s = unsafe { &*($ptr as *const Series<$T>) };
        let vals = s.values_to_vec();
        let n = s.len();
        let mut full_shape = vec![n];
        full_shape.extend_from_slice($shape);
        let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), vals).unwrap();
        Ok(PyArrayDyn::from_owned_array($py, arr).into_any().unbind())
    }};
}

fn series_values_to_numpy(
    py: Python<'_>,
    ptr: *const u8,
    shape: &[usize],
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    match dtype {
        "float64" => series_values_match!(py, ptr, shape, f64),
        "float32" => series_values_match!(py, ptr, shape, f32),
        "int64" => series_values_match!(py, ptr, shape, i64),
        "int32" => series_values_match!(py, ptr, shape, i32),
        "uint64" => series_values_match!(py, ptr, shape, u64),
        "uint32" => series_values_match!(py, ptr, shape, u32),
        "bool" => series_values_match!(py, ptr, shape, u8),
        _ => Err(PyTypeError::new_err(format!("unsupported dtype: {dtype_str}"))),
    }
}

macro_rules! series_last_match {
    ($py:expr, $ptr:expr, $shape:expr, $T:ty) => {{
        let s = unsafe { &*($ptr as *const Series<$T>) };
        let data = s.last().to_vec();
        let ix = if $shape.is_empty() { &[][..] } else { $shape };
        let arr = ArrayD::from_shape_vec(IxDyn(ix), data).unwrap();
        Ok(PyArrayDyn::from_owned_array($py, arr).into_any().unbind())
    }};
}

fn series_last_to_numpy(
    py: Python<'_>,
    ptr: *const u8,
    shape: &[usize],
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    match dtype {
        "float64" => series_last_match!(py, ptr, shape, f64),
        "float32" => series_last_match!(py, ptr, shape, f32),
        "int64" => series_last_match!(py, ptr, shape, i64),
        "int32" => series_last_match!(py, ptr, shape, i32),
        "uint64" => series_last_match!(py, ptr, shape, u64),
        "uint32" => series_last_match!(py, ptr, shape, u32),
        "bool" => series_last_match!(py, ptr, shape, u8),
        _ => Err(PyTypeError::new_err(format!("unsupported dtype: {dtype_str}"))),
    }
}

// ---------------------------------------------------------------------------
// PyOperatorState — per-operator Python callback
// ---------------------------------------------------------------------------

/// Holds the Python objects needed to call a Python operator's `compute()`
/// from Rust's `flush()`.
struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject, // tuple of ObservableView / SeriesView
    py_state: PyObject,
    element_size: usize,
    dtype_str: String,
    error_slot: ErrorSlot,
}

/// Compute function for Python operators, matching the `OperatorSlot`
/// `compute_fn` signature.
///
/// # Safety
///
/// * `output_obs_ptr` must point to a valid `Observable<T>` whose byte
///   size matches `PyOperatorState.element_size`.
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn(
    timestamp: i64,
    _input_ptrs: *const *mut u8,
    _n_inputs: usize,
    output_obs_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        // Convert i64 timestamp to np.datetime64.
        let np = py.import("numpy")?;
        let ts = np.call_method1("datetime64", (timestamp, "ns"))?;

        // Call operator.compute(timestamp, inputs, state).
        let result = state
            .py_operator
            .call_method1(py, "compute", (&ts, &state.py_inputs, &state.py_state))?;

        let tuple = result.bind(py);
        let raw_value = tuple.get_item(0)?;
        let new_state = tuple.get_item(1)?;
        state.py_state = new_state.unbind();

        if raw_value.is_none() {
            return Ok(false);
        }

        // Convert value to contiguous array and write to Observable's value buffer.
        let contiguous = np.call_method1("ascontiguousarray", (&raw_value,))?;
        let interface = contiguous.getattr("__array_interface__")?;
        let data_tuple = interface.get_item("data")?;
        let ptr_int: usize = data_tuple.get_item(0)?.extract()?;

        // output_obs_ptr points to the Observable struct.  Get the
        // internal value buffer pointer (type-agnostic — all Observable<T>
        // have the same struct layout; only the pointed-to buffer type differs).
        let obs = unsafe { &mut *(output_obs_ptr as *mut Observable<u8>) };
        let dest = obs.vals_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr_int as *const u8,
                dest,
                state.element_size,
            );
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

/// Drop function for `PyOperatorState` — acquires GIL for PyObject cleanup.
unsafe fn drop_py_op(state: *mut u8, input_ptrs: *const *mut u8, n_inputs: usize) {
    Python::attach(|_py| {
        unsafe { drop(Box::from_raw(state as *mut PyOperatorState)) };
    });
    unsafe {
        drop(Vec::from_raw_parts(
            input_ptrs as *mut *mut u8,
            n_inputs,
            n_inputs,
        ));
    }
}

// ---------------------------------------------------------------------------
// NativeScenario
// ---------------------------------------------------------------------------

/// Python-visible wrapper around the Rust `Scenario` runtime.
///
/// Sources are pre-drained on the Python side and passed as arrays.
/// Operators participate via GIL callbacks, reading inputs through
/// `ObservableView` / `SeriesView`.
#[pyclass]
pub struct NativeScenario {
    scenario: Option<Scenario>,
    error_slot: ErrorSlot,
    node_dtypes: Vec<String>,
}

// SAFETY: Scenario owns all its heap data.  NativeScenario is moved into
// py.detach() which requires Send.
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

    /// Register a source with pre-drained data.
    ///
    /// `initial`: flat numpy array of initial values.
    /// `timestamps`: int64 nanosecond timestamps.
    /// `values_bytes`: raw bytes of the value array (contiguous, native endian).
    /// `stride`: number of typed elements per event.
    fn add_source(
        &mut self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: String,
        initial: PyObject,
        timestamps: PyReadonlyArrayDyn<'_, i64>,
        values_bytes: &[u8],
        stride: usize,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let node_index = add_source_dispatch(py, sc, &shape, &dtype, &initial)?;
        let ts_vec = timestamps.as_slice()?.to_vec();
        register_array_source_dispatch(
            sc,
            node_index,
            &dtype,
            ts_vec,
            values_bytes.to_vec(),
            stride.max(1),
        )?;
        self.node_dtypes.push(normalise_dtype(&dtype).to_string());
        Ok(node_index)
    }

    /// Register a Python operator.
    fn add_py_operator(
        &mut self,
        _py: Python<'_>,
        input_indices: Vec<usize>,
        shape: Vec<usize>,
        dtype: String,
        py_operator: PyObject,
        py_inputs: PyObject,
        py_state: PyObject,
    ) -> PyResult<usize> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();
        let node_index = create_node_dispatch(sc, &shape, &dtype_norm)?;

        let stride: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let elem_bytes = dtype_element_bytes(&dtype_norm)?;
        let element_size = stride * elem_bytes;

        let op_state = Box::new(PyOperatorState {
            py_operator,
            py_inputs,
            py_state,
            element_size,
            dtype_str: dtype_norm.clone(),
            error_slot: self.error_slot.clone(),
        });
        let state_ptr = Box::into_raw(op_state) as *mut u8;

        sc.add_raw_operator(&input_indices, node_index, py_compute_fn, state_ptr, drop_py_op);

        self.node_dtypes.push(dtype_norm);
        Ok(node_index)
    }

    /// Register a Rust-native operator from an opaque handle.
    ///
    /// The handle is consumed (single-use).  Returns the output node index.
    fn register_handle_operator(
        &mut self,
        handle: &mut NativeOpHandle,
        input_indices: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> PyResult<usize> {
        let f = handle.register_fn.take().ok_or_else(|| {
            PyRuntimeError::new_err("NativeOpHandle has already been consumed")
        })?;
        let sc = self.scenario.as_mut().unwrap();
        let idx = f(sc, &input_indices, &output_shape);
        self.node_dtypes.push(handle.dtype_str.clone());
        Ok(idx)
    }

    /// Materialise a node.
    fn materialize(&mut self, node_index: usize) -> PyResult<()> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype = &self.node_dtypes[node_index];
        materialize_dispatch(sc, node_index, dtype)
    }

    /// Get an ObservableView for a node.
    fn observable_view(&self, node_index: usize) -> PyResult<ObservableView> {
        let sc = self.scenario.as_ref().unwrap();
        let dtype = &self.node_dtypes[node_index];
        let stride = sc.node_stride(node_index);
        Ok(ObservableView {
            obs_ptr: sc.node_obs_ptr(node_index) as *const u8,
            shape: if stride == 1 { vec![] } else { vec![stride] },
            stride,
            dtype_str: dtype.clone(),
        })
    }

    /// Get a SeriesView for a materialised node.
    fn series_view(&self, node_index: usize) -> PyResult<SeriesView> {
        let sc = self.scenario.as_ref().unwrap();
        let ptr = sc.node_series_ptr(node_index);
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err("node is not materialised"));
        }
        let dtype = &self.node_dtypes[node_index];
        let stride = sc.node_stride(node_index);
        Ok(SeriesView {
            series_ptr: ptr as *const u8,
            shape: if stride == 1 { vec![] } else { vec![stride] },
            dtype_str: dtype.clone(),
        })
    }

    /// Register a channel-based source (for async Python sources).
    ///
    /// Returns ``(node_index, hist_sender, live_sender)``.  The Python
    /// driver iterates the source and sends events through the senders.
    fn add_channel_source(
        &mut self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: String,
        initial: PyObject,
    ) -> PyResult<(usize, HistoricalEventSender, LiveEventSender)> {
        let sc = self.scenario.as_mut().unwrap();
        let dtype_norm = normalise_dtype(&dtype).to_string();
        let node_index = add_source_dispatch(py, sc, &shape, &dtype_norm, &initial)?;

        let stride: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let elem_bytes = dtype_element_bytes(&dtype_norm)?;
        let element_size = stride * elem_bytes;

        let (hist_tx, hist_rx) = mpsc::unbounded_channel();
        let (live_tx, live_rx) = mpsc::unbounded_channel();

        let channel_source = ChannelSource {
            hist_rx,
            live_rx,
            element_size,
        };
        register_channel_source_dispatch(sc, node_index, &dtype_norm, Box::new(channel_source))?;

        self.node_dtypes.push(dtype_norm);

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

    /// Run the POCQ event loop (single-thread, no Python source driver).
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

    /// Run with a Python driver on a background thread for concurrent
    /// source iteration.
    ///
    /// ``driver`` is a Python callable (takes no arguments) that drives
    /// all Python sources via asyncio on a background thread.  The main
    /// thread runs the Rust POCQ + DAG via tokio.
    fn run_with_driver(&mut self, py: Python<'_>, driver: PyObject) -> PyResult<()> {
        let mut scenario = self.scenario.take().ok_or_else(|| {
            PyRuntimeError::new_err("scenario already consumed by a previous run()")
        })?;

        // Run both the Python driver and the Rust POCQ concurrently.
        // py.detach releases the GIL so the background thread can acquire it.
        let (scenario, bg_result) = py.detach(move || {
            // Spawn background thread for Python source driving.
            let bg_handle = std::thread::spawn(move || -> Result<(), String> {
                Python::attach(|py| {
                    driver.call0(py).map_err(|e| e.to_string())?;
                    Ok(())
                })
            });

            // Run tokio POCQ on this thread (GIL released).
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(scenario.run());

            // Join background thread (GIL still released, so bg can finish).
            let bg_result = bg_handle
                .join()
                .map_err(|_| "background thread panicked".to_string())
                .and_then(|r| r);

            (scenario, bg_result)
        });

        self.scenario = Some(scenario);

        // Check operator error slot first.
        if let Some(err) = self.error_slot.lock().unwrap().take() {
            return Err(PyRuntimeError::new_err(err));
        }

        // Check background thread result.
        match bg_result {
            Ok(()) => Ok(()),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }

    /// Number of entries in a materialised series.
    fn series_len(&self, node_index: usize) -> PyResult<usize> {
        let sv = self.series_view(node_index)?;
        Ok(sv.__len__())
    }

    /// Timestamps of a materialised series as numpy int64 array.
    fn series_timestamps<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.series_view(node_index)?;
        Ok(sv.index(py))
    }

    /// Values of a materialised series as numpy array.
    fn series_values<'py>(&self, py: Python<'py>, node_index: usize) -> PyResult<PyObject> {
        let sv = self.series_view(node_index)?;
        sv.values(py)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

fn register_channel_source_dispatch(
    sc: &mut Scenario,
    node_index: usize,
    dtype: &str,
    source: Box<dyn Source>,
) -> PyResult<()> {
    let dtype = normalise_dtype(dtype);
    match dtype {
        "float64" => sc.register_source_typed::<f64>(node_index, source),
        "float32" => sc.register_source_typed::<f32>(node_index, source),
        "int64" => sc.register_source_typed::<i64>(node_index, source),
        "int32" => sc.register_source_typed::<i32>(node_index, source),
        "uint64" => sc.register_source_typed::<u64>(node_index, source),
        "uint32" => sc.register_source_typed::<u32>(node_index, source),
        "bool" => sc.register_source_typed::<u8>(node_index, source),
        other => return Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
    Ok(())
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<NativeScenario>()?;
    m.add_class::<ObservableView>()?;
    m.add_class::<SeriesView>()?;
    m.add_class::<HistoricalEventSender>()?;
    m.add_class::<LiveEventSender>()?;
    m.add_class::<NativeOpHandle>()?;
    // Operator factory pyfunctions (macro-generated + hand-written).
    m.add_function(pyo3::wrap_pyfunction!(add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(subtract, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(multiply, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(divide, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(negate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(select, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(concat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(stack, m)?)?;
    Ok(())
}
