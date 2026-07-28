//! [`PyClassOperator`] — hosts a Python operator object over inputs `I`.

use std::ffi::CString;
use std::marker::PhantomData;

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::data::{Array, ArrayView, Instant};
use crate::graph::{Interface, Segment};
use crate::ports::{ArrayPort, ArrayPorts, ClockPort};

use super::{PyArgs, PyParams};

/// Where a Python operator's definition comes from. Each defines a factory
/// `build(**kwargs) -> operator` (called with [`PyParams`]) or binds the
/// instance to `__op__` directly.
#[derive(Clone)]
enum Loader {
    /// Inline Python program (statements).
    Source(String),
    /// An importable module on the embedded interpreter's path.
    Module(String),
}

/// Resolve the operator instance from its [`Loader`] + [`PyParams`].
fn resolve_operator<'py>(
    py: Python<'py>,
    loader: &Loader,
    params: &PyParams,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = params.to_dict(py)?;
    let (build, op_obj) = match loader {
        Loader::Source(src) => {
            let g = PyDict::new(py);
            let np = py.import("numpy")?;
            g.set_item("np", &np)?;
            g.set_item("numpy", &np)?;
            let code = CString::new(src.as_str())
                .map_err(|_| PyValueError::new_err("python source has interior NUL"))?;
            py.run(code.as_c_str(), Some(&g), None)?;
            (g.get_item("build")?, g.get_item("__op__")?)
        }
        Loader::Module(name) => {
            let module = py.import(name.as_str())?;
            (module.getattr("build").ok(), module.getattr("__op__").ok())
        }
    };
    if let Some(build) = build {
        build.call((), Some(&kwargs))
    } else if let Some(op) = op_obj {
        Ok(op)
    } else {
        Err(PyValueError::new_err(
            "python operator must define `build(**kwargs)` or bind `__op__`",
        ))
    }
}

/// A class-based Python operator over input ports `I` (e.g. `ArrayPorts<f64,
/// 1>` or `(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)`)
/// producing a `(clock, values)` event-stream output: the clock fires on
/// computes where the Python side returned a batch (rather than `None`), and
/// the values port is a rank-`NO` view of the host's owned buffer. Load its
/// definition from a `.py` file, an importable module, or an inline source;
/// each defines `build(**kwargs)` (called with [`PyParams`]) or binds
/// `__op__`.
///
/// `NO` is the static output rank, defaulting to `1` because the strategy
/// operators all emit `(N,)` predictions / weights. The output's element shape
/// comes from `out_shape` (its product must equal the rank-`NO` extents'
/// product); the NumPy boundary negotiates by element count.
pub struct PyClassOperator<I: PyArgs = ArrayPorts<f64, 1>, const NO: usize = 1> {
    loader: Loader,
    params: PyParams,
    out_shape: Vec<usize>,
    _marker: PhantomData<fn() -> I>,
}

impl<I: PyArgs, const NO: usize> PyClassOperator<I, NO> {
    /// Load from a `.py` file on disk (read now).
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        params: PyParams,
        out_shape: Vec<usize>,
    ) -> Self {
        let path = path.as_ref();
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read python operator file {}: {e}", path.display()));
        Self::from_source(src, params, out_shape)
    }

    /// Load from an importable module (on the embedded interpreter's path).
    pub fn from_module(module: impl Into<String>, params: PyParams, out_shape: Vec<usize>) -> Self {
        Self::new(Loader::Module(module.into()), params, out_shape)
    }

    /// Load from an inline Python program (handy for tests / one-offs).
    pub fn from_source(source: impl Into<String>, params: PyParams, out_shape: Vec<usize>) -> Self {
        Self::new(Loader::Source(source.into()), params, out_shape)
    }

    fn new(loader: Loader, params: PyParams, out_shape: Vec<usize>) -> Self {
        Self {
            loader,
            params,
            out_shape,
            _marker: PhantomData,
        }
    }
}

/// State: the Python operator instance + its Python state object (created at
/// init), and the rank-`NO` output buffer. No clock: the event time arrives as
/// the graph context, per `compute` call.
pub struct PyClassState<const NO: usize> {
    operator: Py<PyAny>,
    py_state: Py<PyAny>,
    /// `numpy.ascontiguousarray`, resolved once — normalizes whatever
    /// `compute` returns (a list, a scalar, a non-`f64` or strided array)
    /// before the copy into `out`. A C-contiguous `f64` array passes
    /// through untouched.
    as_array: Py<PyAny>,
    out: Array<f64, NO>,
}

/// Copy a Python `compute` return value into the host's output buffer. The
/// NumPy boundary negotiates by element count (as the output extents do).
fn write_output<const NO: usize>(
    value: &Bound<'_, PyAny>,
    as_array: &Bound<'_, PyAny>,
    out: &mut Array<f64, NO>,
) -> PyResult<()> {
    let arr = as_array.call1((value, "float64"))?;
    let readonly = arr.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let src = readonly
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("compute: non-contiguous output: {e}")))?;
    let dst = out.data_mut();
    if src.len() != dst.len() {
        return Err(PyValueError::new_err(format!(
            "compute: expected {} output elements, got {}",
            dst.len(),
            src.len()
        )));
    }
    dst.copy_from_slice(src);
    Ok(())
}

impl<I: PyArgs + 'static, const NO: usize> Segment for PyClassOperator<I, NO> {
    type Inputs = I;
    type Outputs = (ClockPort, ArrayPort<f64, NO>);
    type Context = Instant;
    type State = PyClassState<NO>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> PyClassState<NO> {
        // Instantiate the Python operator and call its `init` with the
        // build-time inputs tuple (clock leaves read `False` — build-time
        // renders are quiescent); allocate the output buffer. No Python
        // `compute` runs here.
        let (operator, py_state, as_array) = Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
                let operator = resolve_operator(py, &self.loader, &self.params)?;
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                I::append_views(py, inputs, &mut views)?;
                let st = operator.call_method1("init", (PyTuple::new(py, views)?,))?;
                let as_array = py.import("numpy")?.getattr("ascontiguousarray")?;
                Ok((operator.unbind(), st.unbind(), as_array.unbind()))
            };
            run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator init failed (see traceback above)");
            })
        });
        PyClassState {
            operator,
            py_state,
            as_array,
            out: Array::zeros(out_extents::<NO>(&self.out_shape)),
        }
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyClassState<NO>,
        time: &Instant,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, f64, NO>) {
        // The batch's event time, straight from the graph context.
        let ts = time.as_offset().as_nanos();

        let PyClassState {
            operator,
            py_state,
            as_array,
            out,
        } = state;

        let result: Result<bool, ()> = Python::attach(|py| {
            let mut run = || -> PyResult<bool> {
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                I::append_views(py, inputs, &mut views)?;
                let op = operator.bind(py);
                let value =
                    op.call_method1("compute", (PyTuple::new(py, views)?, py_state.bind(py), ts))?;
                // `None` is the quiescent return: no event this generation.
                if value.is_none() {
                    return Ok(false);
                }
                write_output(&value, as_array.bind(py), out)?;
                Ok(true)
            };
            run().map_err(|e| e.print(py))
        });
        let notify = result
            .unwrap_or_else(|()| panic!("python operator compute failed (see traceback above)"));
        // A returned batch pulses the output clock; the value buffer is
        // retained either way (its state face holds the last produced batch).
        (
            ArrayView::scalar(if notify { &true } else { &false }),
            out.view(),
        )
    }

    fn reset<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut PyClassState<NO>,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, f64, NO>) {
        (ArrayView::scalar(&false), state.out.view())
    }
}

/// Resolve the rank-`NO` output extents from the user-supplied element
/// `out_shape`. A shape of the right rank is taken verbatim; otherwise (the
/// common `NO == 1` case, where strategy operators pass `vec![n]`) the shape is
/// flattened to its element-count product along axis 0.
fn out_extents<const NO: usize>(out_shape: &[usize]) -> [usize; NO] {
    if let Ok(ext) = <[usize; NO]>::try_from(out_shape) {
        return ext;
    }
    let total: usize = out_shape.iter().product();
    let mut ext = [1usize; NO];
    if NO > 0 {
        ext[0] = total;
    } else {
        assert_eq!(
            total, 1,
            "PyClassOperator: scalar output (NO=0) requires a single-element out_shape, got {out_shape:?}",
        );
    }
    ext
}

// ===========================================================================
// Constructors
// ===========================================================================

/// A class-based Python operator loaded from an importable **module** (on the
/// embedded interpreter's path). The input port tree `I` is inferred from the
/// wiring; `out_shape` is the output element shape (`[]` for a scalar).
pub fn py_class_operator<I: PyArgs, const NO: usize>(
    module: impl Into<String>,
    params: PyParams,
    out_shape: Vec<usize>,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_module(module, params, out_shape)
}

/// [`py_class_operator`] from an inline Python **program** binding `__op__`.
pub fn py_class_operator_source<I: PyArgs, const NO: usize>(
    source: impl Into<String>,
    params: PyParams,
    out_shape: Vec<usize>,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_source(source, params, out_shape)
}

/// [`py_class_operator`] from a plain `.py` **file** on disk (read now).
pub fn py_class_operator_file<I: PyArgs, const NO: usize>(
    path: impl AsRef<std::path::Path>,
    params: PyParams,
    out_shape: Vec<usize>,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_file(path, params, out_shape)
}
