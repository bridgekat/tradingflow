use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::ffi::CString;
use std::marker::PhantomData;

use super::PyInterface;
use crate::data::Instant;
use crate::graph::{Interface, Segment};

/// Where a Python segment's definition comes from.
#[derive(Clone)]
enum Loader {
    /// Inline Python program (statements).
    Source(String),
    /// An importable module on the embedded interpreter's path.
    Module(String),
}

/// Resolves the segment instance from its [`Loader`] and keyword arguments.
fn resolve<'py>(
    py: Python<'py>,
    loader: &Loader,
    params: Option<&Py<PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (build, instance) = match loader {
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
        build.call((), params.map(|p| p.bind(py)))
    } else if let Some(instance) = instance {
        Ok(instance)
    } else {
        Err(PyValueError::new_err(
            "python segment must define `build(**kwargs)` or bind `__op__`",
        ))
    }
}

/// A graph node hosting a Python segment over input interface `I` and output
/// interface `O` (both trees of array/signal ports; see [`PyInterface`]).
///
/// # The Python-side contract
///
/// The Python segment mirrors the [`Segment`] trait method-for-method:
///
/// ```python
/// class MySegment:
///     def init(self, inputs):                    # -> state
///         ...
///     def reset(self, inputs, state):            # -> outputs
///         ...
///     def compute(self, inputs, state, instant): # -> outputs
///         ...
/// ```
///
/// `inputs` is a single tuple with one NumPy array per input leaf, in tree
/// order, and `outputs` is one value per output leaf in the same shape — a
/// bare array for a single output port, a sequence for a tuple of ports (see
/// [`PyInterface`]). `state` is an arbitrary Python object carried across
/// calls, and `instant` is the graph's ambient event time in naive
/// nanoseconds.
///
/// Like [`Segment::reset`], the Python `reset` is called once right after
/// `init` (its first return sizes the node's output buffers) and again after
/// every generation, returning quiescent outputs — signals `False`, values
/// retained or default, per the segment's semantics.
///
/// # Loading and parameters
///
/// A segment definition is loaded from an inline source, an importable
/// module, or a `.py` file; each binds an instance to `__op__` or defines a
/// factory `build(**kwargs)`, called with the keyword arguments given at
/// construction as a plain Python dict (build it with PyO3, e.g.
/// [`IntoPyDict`](pyo3::types::IntoPyDict) — any Python values work).
///
/// # Failure
///
/// Python exceptions are printed and then panic the node, poisoning the graph
/// — the engine's no-recovery stance. Recoverable failures should be values
/// (e.g. an output signal that simply does not fire).
pub struct PySegment<I: PyInterface, O: PyInterface> {
    loader: Loader,
    params: Option<Py<PyDict>>,
    _marker: PhantomData<fn() -> (I, O)>,
}

impl<I: PyInterface, O: PyInterface> PySegment<I, O> {
    /// Loads from an inline Python program (handy for tests / one-offs).
    pub fn from_source(source: impl Into<String>, params: Option<Py<PyDict>>) -> Self {
        Self::new(Loader::Source(source.into()), params)
    }

    /// Loads from an importable module (on the embedded interpreter's path).
    pub fn from_module(module: impl Into<String>, params: Option<Py<PyDict>>) -> Self {
        Self::new(Loader::Module(module.into()), params)
    }

    /// Loads from a `.py` file on disk (read now).
    pub fn from_file(path: impl AsRef<std::path::Path>, params: Option<Py<PyDict>>) -> Self {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read python segment file {}: {e}", path.display()));
        Self::from_source(source, params)
    }

    fn new(loader: Loader, params: Option<Py<PyDict>>) -> Self {
        Self {
            loader,
            params,
            _marker: PhantomData,
        }
    }
}

/// State of a [`PySegment`] node: the Python segment instance, its Python
/// state object, and the Rust-owned buffers backing the node's outputs.
pub struct PyState<O: PyInterface> {
    instance: Py<PyAny>,
    state: Py<PyAny>,
    buffers: O::Buffers,
}

/// Builds the flat `inputs` tuple: one array per leaf, in tree order.
fn args_tuple<'py, I: PyInterface>(
    py: Python<'py>,
    values: I::Values<'_>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut args = Vec::new();
    I::to_python(py, values, &mut args)?;
    PyTuple::new(py, args)
}

/// Runs one Python `reset`/`compute` call and copies its return into the
/// node's output buffers.
fn call<'a, I: PyInterface, O: PyInterface>(
    inputs: I::Values<'a>,
    state: &'a mut PyState<O>,
    method: &str,
    instant: Option<i64>,
) -> O::Values<'a> {
    let PyState {
        instance,
        state,
        buffers,
    } = state;
    Python::attach(move |py| {
        let run = || -> PyResult<Bound<'_, PyAny>> {
            let args = args_tuple::<I>(py, inputs)?;
            let instance = instance.bind(py);
            let state = state.bind(py);
            match instant {
                Some(ns) => instance.call_method1(method, (args, state, ns)),
                None => instance.call_method1(method, (args, state)),
            }
        };
        let value = run().unwrap_or_else(|e| {
            e.print(py);
            panic!("python segment {method} failed (see traceback above)");
        });
        O::from_python(&value, buffers).unwrap_or_else(|e| {
            e.print(py);
            panic!("python segment {method} returned an incompatible value (see traceback above)");
        })
    })
}

impl<I, O> Segment for PySegment<I, O>
where
    I: PyInterface + 'static,
    O: PyInterface + 'static,
{
    type Inputs = I;
    type Outputs = O;
    type Context = Instant;
    type State = PyState<O>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> PyState<O> {
        Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let instance = resolve(py, &self.loader, self.params.as_ref())?;
                let state = instance.call_method1("init", (args_tuple::<I>(py, inputs)?,))?;
                Ok((instance.unbind(), state.unbind()))
            };
            let (instance, state) = run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python segment init failed (see traceback above)");
            });
            PyState {
                instance,
                state,
                buffers: O::Buffers::default(),
            }
        })
    }

    fn reset<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyState<O>,
    ) -> <O as Interface>::Values<'a> {
        call::<I, O>(inputs, state, "reset", None)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyState<O>,
        time: &Instant,
    ) -> <O as Interface>::Values<'a> {
        call::<I, O>(inputs, state, "compute", Some(time.as_offset().as_nanos()))
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// A Python segment loaded from an importable module (on the embedded
/// interpreter's path). The interfaces `I` and `O` are inferred from the
/// wiring.
pub fn py_segment_module<I: PyInterface, O: PyInterface>(
    module: impl Into<String>,
    params: Option<Py<PyDict>>,
) -> PySegment<I, O> {
    PySegment::from_module(module, params)
}

/// A Python segment loaded from an inline Python program binding `__op__`
/// or defining `build(**kwargs)`.
pub fn py_segment_source<I: PyInterface, O: PyInterface>(
    source: impl Into<String>,
    params: Option<Py<PyDict>>,
) -> PySegment<I, O> {
    PySegment::from_source(source, params)
}

/// A Python segment loaded from a plain `.py` file on disk (read now).
pub fn py_segment_file<I: PyInterface, O: PyInterface>(
    path: impl AsRef<std::path::Path>,
    params: Option<Py<PyDict>>,
) -> PySegment<I, O> {
    PySegment::from_file(path, params)
}
